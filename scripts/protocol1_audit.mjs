// Protocol 1: Paired Retraction Intervention Audit, deterministic arms.
//
// For each k=1 base conversation, truncate to the X-support prefix (turns 0..6),
// then attach exactly one condition turn:
//   H_0 : nothing (baseline)
//   H_N : topically matched mention of Y that is NOT a retraction
//   H_R : one of 5 retraction paraphrases drawn from the corpus generator
// Both directions are run (commit X then revise to Y, and the mirrored conversation
// with X/Y swapped), so 50 convs x 2 directions x 5 paraphrases = 500 H_R items.
//
// Target candidate c0 = the committed utterance ("Locked in: ... X"). Measured per
// condition: s(c0) cosine score, rank(c0), Top-5 inclusion, BAF eligibility (grounded
// extension survival), a_top eligibility (cutoff filter survival), detector firing on
// the condition turn. All stages are deterministic, so any nonzero H_R-vs-H_N change
// is an intervention response; the within-condition noise floor is exactly zero.
//
// Expected per the candidate-local class analysis:
//   cosine  : s(c0) exactly invariant; rank may move only via pool composition
//   BAF     : s(c0) invariant; eligibility flips iff the detector fires
//   a_top   : same, via the cutoff filter
// Detector coverage per paraphrase is reported separately (coverage, not semantics).

import { writeFileSync, readFileSync } from "fs";
import { initEmbedding, embed, embedBatch, bufferToFloat32Array } from "../dist/embedding.js";
import { detectRetraction, filterByGroundedExtension } from "../dist/mrp.js";

const CONVS_PATH = "./exp_tmp/synth_chain_k1_convs.json";
const OUT_PATH = process.env.OUT ?? "/tmp/protocol1_audit.json";

const PRIOR_CTX = 4, EVENT_FORCE = 1.0, COS_T = 0.30, EDGE_T = 0.15, COMMIT_LOW = 0.5, COMMIT_HIGH = 1.0;
const EVENT_RE = /(?:\bI\s+(?:just|recently|finally|honestly)\s+(?:had|got|started|moved|became|quit|lost|met|broke)|\bsince\s+(?:I|my|the|that|then|last|we)|\bnow\s+(?:that|I'm|I\s+am|I\s+have|I've))/i;
const COMMITMENT_RE = /(?:\bbook(?:ed|ing)?\b|\bconfirm(?:ed)?\b|\bdecide[ds]?\b|\b(?:will|would|shall)\s+(?:use|do|go|book|take)\b|\bgoing\s+to\b|\bplan(?:ning)?\s+to\b|\blocked?\s+in\b|\border(?:ed)?\b|\breserv(?:ed|ing)?\b)/i;

const PARAPHRASES = [
  y => `Wait, change of plan: ${y}.`,
  y => `No, let's do ${y} instead.`,
  y => `Actually, let's switch to ${y} instead.`,
  y => `Actually scratch that, ${y}.`,
  y => `Forget that, going with ${y}.`,
];
const NEUTRAL = y => `My colleague mentioned ${y} the other day.`;

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function wbReplace(text, from, to) { return text.replace(new RegExp(`\\b${escapeRe(from)}\\b`, "g"), to); }

function detectCarrier(text, ctx) {
  const tag = detectRetraction(text, ctx);
  if (tag.is_retraction) return tag;
  if (EVENT_RE.test(text)) return { is_retraction: true, speech_act_class: "EventUtterance", retraction_weight: EVENT_FORCE, target_keywords: [], destination_slot: null };
  return tag;
}

function tagTurns(turns) {
  return turns.map((t, i) => {
    const ctxStart = Math.max(0, i - PRIOR_CTX);
    const priorCtx = turns.slice(ctxStart, i).map(x => x.content).join(" ");
    const tag = detectCarrier(t.content, priorCtx);
    return { i, role: t.role, content: t.content, ...tag };
  });
}

function bafSurvivors(scoredCands, tagged, docVecs) {
  const cands = scoredCands.map(c => {
    const t = tagged[c.i];
    return { id: c.i, chunk_text: t.content, timestamp: `T${String(c.i).padStart(6, "0")}`, role: t.role, is_retraction: t.is_retraction, target_keywords: t.target_keywords, retraction_weight: t.retraction_weight };
  });
  const carriers = cands.filter(c => c.is_retraction && c.role === "user");
  const embById = new Map(scoredCands.map(c => [c.i, docVecs[c.i]]));
  const edges = [];
  for (const carrier of carriers) {
    const cEmb = embById.get(carrier.id); if (!cEmb) continue;
    for (const target of cands) {
      if (target.id === carrier.id || target.is_retraction) continue;
      if (target.timestamp >= carrier.timestamp) continue;
      const tEmb = embById.get(target.id);
      const cosSim = tEmb ? cosine(cEmb, tEmb) : 0;
      const matchScore = cosSim >= COS_T ? cosSim : 0;
      if (matchScore === 0) continue;
      const f = COMMITMENT_RE.test(target.chunk_text) ? COMMIT_HIGH : COMMIT_LOW;
      const w = carrier.retraction_weight * matchScore * f;
      if (w < EDGE_T) continue;
      const rescaled = 0.5 + (w - EDGE_T) / (1.0 - EDGE_T) * 0.5;
      edges.push({ attacker_id: carrier.id, target_id: target.id, weight: rescaled });
    }
  }
  const { survivors } = filterByGroundedExtension(cands, edges);
  return new Set(survivors.map(s => s.id));
}

function aTopSurvivors(scoredCands, tagged, docVecs) {
  const carrierIds = scoredCands.map(c => c.i).filter(i => tagged[i].is_retraction && tagged[i].role === "user").sort((a, b) => a - b);
  if (carrierIds.length === 0) return new Set(scoredCands.map(c => c.i));
  const last = carrierIds[carrierIds.length - 1];
  const lastVec = docVecs[last];
  const kept = [];
  for (const c of scoredCands) {
    if (c.i >= last) { kept.push(c.i); continue; }
    const v = docVecs[c.i];
    const superseded = !!(v && lastVec) && cosine(lastVec, v) >= COS_T;
    if (!superseded) kept.push(c.i);
  }
  return new Set(kept.length ? kept : scoredCands.map(c => c.i));
}

async function measure(turns, query, c0Idx) {
  const docTexts = turns.map(t => `${t.role}: ${t.content}`);
  const docBufs = await embedBatch(docTexts);
  const docVecs = docBufs.map(bufferToFloat32Array);
  const qVec = bufferToFloat32Array(await embed(query));
  const scored = docVecs.map((v, j) => ({ i: j, s: cosine(qVec, v) })).sort((a, b) => b.s - a.s);
  const tagged = tagTurns(turns);
  const s0 = scored.find(c => c.i === c0Idx).s;
  const rank0 = scored.findIndex(c => c.i === c0Idx) + 1;
  const top5 = scored.slice(0, 5).some(c => c.i === c0Idx) ? 1 : 0;
  const baf = bafSurvivors(scored, tagged, docVecs).has(c0Idx) ? 1 : 0;
  const atop = aTopSurvivors(scored, tagged, docVecs).has(c0Idx) ? 1 : 0;
  const lastTurnFired = tagged[turns.length - 1].is_retraction && tagged[turns.length - 1].role === "user" ? 1 : 0;
  return { s0, rank0, top5, baf, atop, lastTurnFired };
}

const items = JSON.parse(readFileSync(CONVS_PATH, "utf8"));
await initEmbedding();
console.log(`Embedder loaded; ${items.length} base conversations`);

const results = [];
let skipped = 0;
for (const item of items) {
  const Xs = item.Xs ?? [item.X1, item.X2].filter(Boolean);
  const [X, Y] = [Xs[0], Xs[Xs.length - 1]];
  const prefixOrig = item.turns.slice(0, 7);
  if (!prefixOrig[2] || !prefixOrig[2].content.includes(X)) { skipped++; continue; }
  for (const dir of ["fwd", "rev"]) {
    const [xa, xb] = dir === "fwd" ? [X, Y] : [Y, X];
    const prefix = dir === "fwd" ? prefixOrig : prefixOrig.map(t => ({ role: t.role, content: wbReplace(t.content, X, xb) }));
    const c0Idx = 2;
    const conds = { H0: prefix, HN: [...prefix, { role: "user", content: NEUTRAL(xb === X ? Y : xb) }] };
    conds.HN = [...prefix, { role: "user", content: NEUTRAL(dir === "fwd" ? Y : X) }];
    const base = {};
    for (const [cond, turns] of Object.entries(conds)) base[cond] = await measure(turns, item.query, c0Idx);
    for (let p = 0; p < PARAPHRASES.length; p++) {
      const yTo = dir === "fwd" ? Y : X;
      const hr = [...prefix, { role: "user", content: PARAPHRASES[p](yTo) }];
      const m = await measure(hr, item.query, c0Idx);
      results.push({ idx: item.idx, dir, paraphrase: p, query: item.query, H0: base.H0, HN: base.HN, HR: m });
    }
  }
}
console.log(`items measured: ${results.length}, skipped convs: ${skipped}`);

const agg = { n: results.length, skipped, hn_false_positive: 0, detector_fired: 0, by_paraphrase: {}, stages: {} };
const stageKeys = ["s0", "rank0", "top5", "baf", "atop"];
const delta = (a, b) => Math.abs(a - b);
const st = Object.fromEntries(stageKeys.map(k => [k, { rn_changed: 0, maxAbsScoreDelta: 0 }]));
for (const r of results) {
  if (r.HN.lastTurnFired) agg.hn_false_positive++;
  if (r.HR.lastTurnFired) agg.detector_fired++;
  const pk = String(r.paraphrase);
  agg.by_paraphrase[pk] = agg.by_paraphrase[pk] ?? { n: 0, fired: 0, baf_flip: 0, atop_flip: 0 };
  agg.by_paraphrase[pk].n++;
  if (r.HR.lastTurnFired) agg.by_paraphrase[pk].fired++;
  if (r.HR.baf !== r.HN.baf) agg.by_paraphrase[pk].baf_flip++;
  if (r.HR.atop !== r.HN.atop) agg.by_paraphrase[pk].atop_flip++;
  for (const k of stageKeys) {
    const d = delta(r.HR[k], r.HN[k]);
    if (k === "s0") st[k].maxAbsScoreDelta = Math.max(st[k].maxAbsScoreDelta, d);
    if (d > 0) st[k].rn_changed++;
  }
}
agg.stages = st;
agg.eligibility_flip_given_fired = results.filter(r => r.HR.lastTurnFired && r.HR.baf !== r.HN.baf).length;
agg.fired_total = agg.detector_fired;

writeFileSync(OUT_PATH, JSON.stringify({ config: { COS_T, EDGE_T, paraphrases: PARAPHRASES.map(f => f("Y")) }, agg, results }, null, 1));
console.log(JSON.stringify(agg, null, 2));
console.log(`written: ${OUT_PATH}`);
