// Retrieval-audit metric split (review fix #1).
//
// The published "superseded-item leakage@5" was operationalized as: some top-5 item
// MENTIONS a superseded value (deterministic systems: pre-carrier turn containing a
// prior value, substring; Mem0: memory text containing a prior value, substring).
// For Mem0 that conflates currently-valid supersession narratives ("planned Paris but
// later changed to Lisbon") with stale assertions. This script keeps the original
// column under its honest name (superseded-value mention@5) and adds a uniform
// item-level split:
//
//   unresolved stale mention@5 : some top-5 item word-boundary-mentions a superseded
//     value AND that same item neither mentions the current value nor contains a
//     supersession marker (change/switch/instead/no longer/update word families).
//
// Deterministic rankings are re-derived exactly as scripts/synth_chain_twochannel.mjs
// (functions copied verbatim, same constants, same embedder); the re-derivation is
// gated by reproducing the pinned aggregates in exp_tmp/pinned_twochannel_k{K}.json
// cell by cell before any new number is emitted. Mem0 items are read from the stored
// retrieved_top5 in exp_tmp/pinned_mem0_k{K}.json and gated against its stored
// retrieval_hit_pct / retrieval_distractor_pct at @5.
//
// Zero LLM calls. Run: node scripts/retrieval_audit_split.mjs

import { writeFileSync, readFileSync } from "fs";
import { initEmbedding, embed, embedBatch, bufferToFloat32Array } from "../dist/embedding.js";
import { detectRetraction, filterByGroundedExtension } from "../dist/mrp.js";

const BASE = "./exp_tmp";
const OUT = process.env.OUT ?? "/tmp/retrieval_audit_split.json";
const TOP_K = 5;
const CAND_M = 50;

const PRIOR_CTX = 4, EVENT_FORCE = 1.0, COS_T = 0.30, EDGE_T = 0.15, COMMIT_LOW = 0.5, COMMIT_HIGH = 1.0;
const EVENT_RE = /(?:\bI\s+(?:just|recently|finally|honestly)\s+(?:had|got|started|moved|became|quit|lost|met|broke)|\bsince\s+(?:I|my|the|that|then|last|we)|\bnow\s+(?:that|I'm|I\s+am|I\s+have|I've))/i;
const COMMITMENT_RE = /(?:\bbook(?:ed|ing)?\b|\bconfirm(?:ed)?\b|\bdecide[ds]?\b|\b(?:will|would|shall)\s+(?:use|do|go|book|take)\b|\bgoing\s+to\b|\bplan(?:ning)?\s+to\b|\blocked?\s+in\b|\border(?:ed)?\b|\breserv(?:ed|ing)?\b)/i;
const MARKER_RE = /\b(chang(?:e|ed|es|ing)|switch(?:ed|es|ing)?|instead|no\s+longer|updat(?:e|ed|es|ing))\b/i;

const CONFIGS = ["cosine", "baf", "a_top", "a_lex", "b_slot"];
const KS = [[1, 1], [3, 2], [8, 8]];

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
function wbContains(text, term) { return new RegExp(`\\b${escapeRe(term)}\\b`, "i").test(text); }

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

function bafFilter(scoredCands, tagged, docVecs) {
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
  const scoreById = new Map(scoredCands.map(c => [c.i, c.s]));
  survivors.sort((a, b) => (scoreById.get(b.id) ?? 0) - (scoreById.get(a.id) ?? 0));
  return survivors.map(s => s.id);
}

function twoChannelRetrievalTime(scoredCands, tagged, docVecs, mode) {
  const carrierIds = scoredCands
    .map(c => c.i)
    .filter(i => tagged[i].is_retraction && tagged[i].role === "user")
    .sort((a, b) => a - b);
  if (carrierIds.length === 0) return scoredCands.map(c => c.i);
  const last = carrierIds[carrierIds.length - 1];
  const lastTag = tagged[last];
  const lastVec = docVecs[last];
  const kept = [];
  for (const c of scoredCands) {
    if (c.i >= last) { kept.push(c); continue; }
    let superseded;
    if (mode === "lex") {
      const kws = lastTag.target_keywords ?? [];
      superseded = kws.length > 0 && kws.some(kw => wbContains(tagged[c.i].content, kw));
    } else {
      const v = docVecs[c.i];
      superseded = !!(v && lastVec) && cosine(lastVec, v) >= COS_T;
    }
    if (!superseded) kept.push(c);
  }
  if (kept.length === 0) return scoredCands.map(c => c.i);
  return kept.map(c => c.i);
}

function slotTrackIngest(turns, tagged) {
  let slot = null;
  const superseded = [];
  let ok = 0, failed = 0;
  for (let i = 0; i < turns.length; i++) {
    const t = tagged[i];
    if (t.role !== "user" || !t.is_retraction) continue;
    const dest = t.destination_slot;
    if (dest && dest.trim().length > 0) {
      if (slot) superseded.push(slot);
      slot = dest.trim();
      ok++;
    } else {
      failed++;
    }
  }
  return { slot, superseded, updates_ok: ok, updates_failed: failed };
}

function slotFilter(scoredCands, tagged, state) {
  if (!state.slot) return scoredCands.map(c => c.i);
  const kept = scoredCands.filter(c => {
    const txt = tagged[c.i].content;
    const hitsStale = state.superseded.some(v => wbContains(txt, v));
    if (!hitsStale) return true;
    return wbContains(txt, state.slot);
  });
  return (kept.length ? kept : scoredCands).map(c => c.i);
}

function wilson(k, n) {
  const z = 1.959963984540054, p = k / n;
  const d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d;
  const h = (z / d) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, c - h) * 100, Math.min(1, c + h) * 100];
}
const fmt = (k, n) => {
  const [lo, hi] = wilson(k, n);
  return `${(100 * k / n).toFixed(0)} [${lo.toFixed(1)},${hi.toFixed(1)}]`;
};

function unresolvedStale(texts, Xprior, Xfinal) {
  return texts.some(t =>
    Xprior.some(x => wbContains(t, x)) && !wbContains(t, Xfinal) && !MARKER_RE.test(t));
}
function mentionsPrior(texts, Xprior) {
  return texts.some(t => Xprior.some(x => wbContains(t, x)));
}

await initEmbedding();
console.log("Embedder loaded");

const report = { marker_re: String(MARKER_RE), per_m: {} };

for (const [K, m] of KS) {
  const items = JSON.parse(readFileSync(`${BASE}/synth_chain_k${K}_convs.json`, "utf8"));
  const pinned = JSON.parse(readFileSync(`${BASE}/pinned_twochannel_k${K}.json`, "utf8"));
  const mk = () => Object.fromEntries(CONFIGS.map(c => [c, { 1: 0, 3: 0, 5: 0 }]));
  const retr_agg = mk(), dist_agg = mk();
  const split = Object.fromEntries(CONFIGS.map(c => [c, { cover5: 0, mention5: 0, unresolved5: 0 }]));
  const perSample = [];

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const turns = item.turns;
    const docTexts = turns.map(t => `${t.role}: ${t.content}`);
    const Xs = item.Xs ?? [item.X1, item.X2, item.X3].filter(Boolean);
    const Xfinal = Xs[Xs.length - 1];
    const Xprior = Xs.slice(0, -1);

    const carrierRe = /(actually|change of plan|forget|scratch|let'?s? (switch|do))/i;
    const carrierIdxs = [];
    for (let j = 0; j < turns.length; j++) {
      if (turns[j].role === "user" && carrierRe.test(turns[j].content)) carrierIdxs.push(j);
    }
    const lastCarrier = carrierIdxs[carrierIdxs.length - 1] ?? -1;
    const goldIdxs = [], distIdxs = [];
    for (let j = 0; j < turns.length; j++) {
      const t = turns[j].content;
      if (j > lastCarrier && t.includes(Xfinal) && !Xprior.some(x => t.includes(x))) goldIdxs.push(j);
      if (j <= lastCarrier && Xprior.some(x => t.includes(x))) distIdxs.push(j);
    }
    const goldSet = new Set(goldIdxs);
    const distSet = new Set(distIdxs);

    const docBufs = await embedBatch(docTexts);
    const docVecs = docBufs.map(bufferToFloat32Array);
    const qBuf = await embed(item.query);
    const qVec = bufferToFloat32Array(qBuf);
    const M = Math.min(CAND_M, docTexts.length);
    const cosCands = docVecs.map((v, j) => ({ i: j, s: cosine(qVec, v) })).sort((a, b) => b.s - a.s).slice(0, M);

    const tagged = tagTurns(turns);
    const state = slotTrackIngest(turns, tagged);
    const ranked = {
      cosine: cosCands.map(c => c.i),
      baf: bafFilter(cosCands, tagged, docVecs),
      a_top: twoChannelRetrievalTime(cosCands, tagged, docVecs, "top"),
      a_lex: twoChannelRetrievalTime(cosCands, tagged, docVecs, "lex"),
      b_slot: slotFilter(cosCands, tagged, state),
    };

    const sampleFlags = { idx: i };
    for (const cfg of CONFIGS) {
      const topK = ranked[cfg].slice(0, TOP_K);
      const f = {};
      for (const k of [1, 3, 5]) {
        f[`hit${k}`] = topK.slice(0, k).some(t => goldSet.has(t)) ? 1 : 0;
        f[`dist${k}`] = topK.slice(0, k).some(t => distSet.has(t)) ? 1 : 0;
        retr_agg[cfg][k] += f[`hit${k}`];
        dist_agg[cfg][k] += f[`dist${k}`];
      }
      const texts = topK.map(idx => turns[idx].content);
      split[cfg].cover5 += f.hit5;
      split[cfg].mention5 += f.dist5;
      f.unresolved5 = unresolvedStale(texts, Xprior, Xfinal) ? 1 : 0;
      split[cfg].unresolved5 += f.unresolved5;
      sampleFlags[cfg] = f;
    }
    perSample.push(sampleFlags);
    if ((i + 1) % 10 === 0) console.log(`  m=${m} [${i + 1}/${items.length}]`);
  }

  for (const cfg of CONFIGS) {
    for (const k of [1, 3, 5]) {
      if (retr_agg[cfg][k] !== pinned.retrieval_hit[cfg][String(k)])
        throw new Error(`GATE FAIL m=${m} ${cfg} hit@${k}: ${retr_agg[cfg][k]} vs pinned ${pinned.retrieval_hit[cfg][String(k)]}`);
      if (dist_agg[cfg][k] !== pinned.retrieval_distractor[cfg][String(k)])
        throw new Error(`GATE FAIL m=${m} ${cfg} dist@${k}: ${dist_agg[cfg][k]} vs pinned ${pinned.retrieval_distractor[cfg][String(k)]}`);
    }
  }
  console.log(`m=${m}: deterministic re-derivation gate PASSED (30 aggregate cells match pinned)`);

  const mem0 = JSON.parse(readFileSync(`${BASE}/pinned_mem0_k${K}.json`, "utf8"));
  const msplit = { cover5: 0, mention5: 0, unresolved5: 0, n: 0 };
  const mem0PerSample = [];
  for (const s of mem0.per_sample) {
    if (!s.retrieved_top5) continue;
    msplit.n++;
    const Xs = s.Xs, Xfinal = Xs[Xs.length - 1], Xprior = Xs.slice(0, -1);
    const texts = s.retrieved_top5;
    const f = { idx: s.idx };
    f.cover5 = texts.some(t => t.toLowerCase().includes(Xfinal.toLowerCase())) ? 1 : 0;
    f.mention5 = texts.some(t => Xprior.some(x => t.toLowerCase().includes(x.toLowerCase()))) ? 1 : 0;
    f.unresolved5 = unresolvedStale(texts, Xprior, Xfinal) ? 1 : 0;
    msplit.cover5 += f.cover5; msplit.mention5 += f.mention5; msplit.unresolved5 += f.unresolved5;
    mem0PerSample.push(f);
  }
  const hitPct = Math.round(100 * msplit.cover5 / msplit.n * 100) / 100;
  const distPct = Math.round(100 * msplit.mention5 / msplit.n * 100) / 100;
  if (hitPct !== mem0.retrieval_hit_pct["5"] || distPct !== mem0.retrieval_distractor_pct["5"])
    throw new Error(`GATE FAIL m=${m} mem0 @5: ${hitPct}/${distPct} vs pinned ${mem0.retrieval_hit_pct["5"]}/${mem0.retrieval_distractor_pct["5"]}`);
  console.log(`m=${m}: Mem0 stored-top5 gate PASSED (hit/dist @5 match pinned)`);

  const n = items.length;
  report.per_m[m] = { n, deterministic: {}, mem0: msplit, per_sample: perSample, mem0_per_sample: mem0PerSample };
  console.log(`\n=== m=${m} (n=${n}) cover5 / mention5 / unresolved5 ===`);
  for (const cfg of CONFIGS) {
    const s = split[cfg];
    report.per_m[m].deterministic[cfg] = s;
    console.log(`${cfg.padEnd(8)} ${fmt(s.cover5, n).padEnd(16)} ${fmt(s.mention5, n).padEnd(16)} ${fmt(s.unresolved5, n)}`);
  }
  console.log(`${"Mem0".padEnd(8)} ${fmt(msplit.cover5, msplit.n).padEnd(16)} ${fmt(msplit.mention5, msplit.n).padEnd(16)} ${fmt(msplit.unresolved5, msplit.n)}`);
}

writeFileSync(OUT, JSON.stringify(report, null, 1));
console.log(`\nwritten: ${OUT}`);
