// Two-channel supersession baselines on synth chain-retraction (external baseline request).
//
// The request proposed: "retrieve relevant chunks, retrieve explicit negative/supersession
// chunks using lexical or classifier-based detection, then apply a latest-valid-state
// or contradiction-resolution rule." This script implements that, and splits it by
// WHERE the resolution is applied, which is the paper's actual claim:
//
//   cosine  : control, no supersession channel
//   baf     : the paper's witness (grounded extension over defeat edges)
//   a_top   : A/retrieval-time, topical scoping   - drop pre-carrier candidates whose
//             cosine to the latest carrier >= COS_T
//   a_lex   : A/retrieval-time, lexical scoping   - drop pre-carrier candidates matching
//             the latest carrier's target keywords
//   b_slot  : B/ingest-time, event-local slot overwrite - walk turns in order, update a
//             single slot per detected carrier, filter candidates against the final slot
//
// All five share ONE carrier detector, so any A-vs-B gap is attributable to placement
// rather than to detection quality. b_slot updates event by event with no cross-event
// context, which is exactly the reference class the p^k proposition declares; its
// per-item update success is logged so p_eff can be read off directly.
//
// Reuses exp_tmp/synth_chain_k${K}_convs.json so the numbers sit alongside the existing
// cosine/recency/BAF/Mem0/BM25/RRF/cross-encoder results on identical conversations.

import { writeFileSync, readFileSync } from "fs";
import { initEmbedding, embed, embedBatch, bufferToFloat32Array } from "../dist/embedding.js";
import { detectRetraction, filterByGroundedExtension } from "../dist/mrp.js";

const KEY_FILE = `${process.env.HOME}/.api_keys/anthropic_key.env`;
let ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
  const txt = readFileSync(KEY_FILE, "utf8");
  for (const line of txt.split("\n")) {
    const m = line.match(/^export\s+ANTHROPIC_API_KEY=["']?(.+?)["']?$/);
    if (m) { ANTHROPIC_API_KEY = m[1]; break; }
  }
}

const K = Number(process.env.K ?? 1);
const TOP_K = 5;
const CAND_M = Number(process.env.MCAP ?? 50);
const N_LIMIT = Number(process.env.NLIM ?? 0);
const RUN_E2E = (process.env.E2E ?? "1") !== "0";
const MODEL = "claude-haiku-4-5-20251001";
const CONVS_PATH = process.env.CONVS_IN ?? `./exp_tmp/synth_chain_k${K}_convs.json`;
const OUT_PATH = process.env.OUT ?? `/tmp/synth_chain_k${K}_twochannel.json`;

const PRIOR_CTX = 4, EVENT_FORCE = 1.0, COS_T = 0.30, EDGE_T = 0.15, COMMIT_LOW = 0.5, COMMIT_HIGH = 1.0;
const EVENT_RE = /(?:\bI\s+(?:just|recently|finally|honestly)\s+(?:had|got|started|moved|became|quit|lost|met|broke)|\bsince\s+(?:I|my|the|that|then|last|we)|\bnow\s+(?:that|I'm|I\s+am|I\s+have|I've))/i;
const COMMITMENT_RE = /(?:\bbook(?:ed|ing)?\b|\bconfirm(?:ed)?\b|\bdecide[ds]?\b|\b(?:will|would|shall)\s+(?:use|do|go|book|take)\b|\bgoing\s+to\b|\bplan(?:ning)?\s+to\b|\blocked?\s+in\b|\border(?:ed)?\b|\breserv(?:ed|ing)?\b)/i;

const CONFIGS = ["cosine", "baf", "a_top", "a_lex", "b_slot"];

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
function wbContains(text, term) { return new RegExp(`\\b${escapeRe(term)}\\b`, "i").test(text); }

function detectCarrier(text, ctx) {
  const tag = detectRetraction(text, ctx);
  if (tag.is_retraction) return tag;
  if (EVENT_RE.test(text)) return { is_retraction: true, speech_act_class: "EventUtterance", retraction_weight: EVENT_FORCE, target_keywords: [], destination_slot: null };
  return tag;
}

// Carrier tag for every turn, computed once per conversation and shared by all configs.
function tagTurns(turns) {
  return turns.map((t, i) => {
    const ctxStart = Math.max(0, i - PRIOR_CTX);
    const priorCtx = turns.slice(ctxStart, i).map(x => x.content).join(" ");
    const tag = detectCarrier(t.content, priorCtx);
    return { i, role: t.role, content: t.content, ...tag };
  });
}

// ─── The paper's witness ──────────────────────────────────────────────────
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

// ─── A: retrieval-time latest-valid-state ─────────────────────────────────
// One pass over the candidate pool at query time. The latest carrier defines the
// cutoff; anything before it that the carrier's scope covers is treated as stale.
// No state is carried between events, so nothing compounds across k.
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

// ─── B: ingest-time event-local slot overwrite ────────────────────────────
// Walks the transcript in order and maintains one slot value. Each carrier is an
// independent update attempt: a carrier whose destination cannot be extracted leaves
// the slot holding the previous value, and every later turn is then judged against a
// stale state. This is the compounding the p^k reference class describes.
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
  if (!state.slot) return scoredCands.map(c => c.i); // extraction never succeeded
  const kept = scoredCands.filter(c => {
    const txt = tagged[c.i].content;
    const hitsStale = state.superseded.some(v => wbContains(txt, v));
    if (!hitsStale) return true;
    return wbContains(txt, state.slot); // mentions current value too, keep
  });
  return (kept.length ? kept : scoredCands).map(c => c.i);
}

// ─── E2E ──────────────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function ask(retrieved, query) {
  const ctxBlock = retrieved.map((c, i) => `[${i + 1}] ${c}`).join("\n");
  const prompt = `Below is retrieved context from a conversation. Answer the question using ONLY this context. Respond in one short word/phrase, just the answer with no explanation.\n\nContext:\n${ctxBlock}\n\nQuestion: ${query}\n\nAnswer:`;
  for (let attempt = 0; attempt < 5; attempt++) {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      // temperature pinned; see the note in synth_chain_retract.mjs.
      body: JSON.stringify({ model: MODEL, max_tokens: 50, temperature: 0, messages: [{ role: "user", content: prompt }] }),
    });
    if (resp.ok) return (await resp.json()).content[0].text.trim();
    if (resp.status === 429) {
      const wait = 5000 * (attempt + 1);
      console.log(`    rate limit, sleeping ${wait}ms`);
      await sleep(wait);
      continue;
    }
    throw new Error(`API ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
  }
  throw new Error("API 429 exhausted retries");
}

// ─── Main ─────────────────────────────────────────────────────────────────
let items = JSON.parse(readFileSync(CONVS_PATH, "utf8"));
if (N_LIMIT > 0) items = items.slice(0, N_LIMIT);
console.log(`Loaded ${items.length} chain k=${K} conversations (E2E=${RUN_E2E})`);

await initEmbedding();
console.log("Embedder loaded");

const mk = () => Object.fromEntries(CONFIGS.map(c => [c, { 1: 0, 3: 0, 5: 0 }]));
const retr_agg = mk();
const dist_agg = mk();
const e2e_agg = Object.fromEntries(CONFIGS.map(c => [c, { y_correct: 0, x_distractor: 0, neither: 0 }]));
const slotStats = { updates_ok: 0, updates_failed: 0, slot_null: 0 };

// State correctness, measured without an LLM. This is where the placement question
// is actually decided, and it has far more power than the downstream answer metric
// at n=50: the resolution rule fires on only the fraction of conversations whose
// carriers the detector recovers.
//
//   a_state : does the retrieval-time rule adopt the true final carrier as its
//             cutoff? One event has to be recovered, whatever k is.
//   b_state : does the ingest-tracked slot equal the final committed value? Every
//             carrier in the chain has to be recovered and extracted, which is the
//             all-relations-correct event the p^k reference class describes.
const stateStats = {
  a_correct: 0, a_stale_cutoff: 0, a_no_carrier: 0,
  b_correct: 0, b_stale_value: 0, b_no_slot: 0,
  carriers_gold: 0, carriers_detected: 0,
};
const perSample = [];

for (let i = 0; i < items.length; i++) {
  const item = items[i];
  const turns = item.turns;
  const docTexts = turns.map(t => `${t.role}: ${t.content}`);
  const Xs = item.Xs ?? [item.X1, item.X2, item.X3].filter(Boolean);
  const Xfinal = Xs[Xs.length - 1];
  const Xprior = Xs.slice(0, -1);

  // Gold / distractor labelling, identical to the sibling baselines.
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
  slotStats.updates_ok += state.updates_ok;
  slotStats.updates_failed += state.updates_failed;
  if (!state.slot) slotStats.slot_null++;

  const ranked = {
    cosine: cosCands.map(c => c.i),
    baf: bafFilter(cosCands, tagged, docVecs),
    a_top: twoChannelRetrievalTime(cosCands, tagged, docVecs, "top"),
    a_lex: twoChannelRetrievalTime(cosCands, tagged, docVecs, "lex"),
    b_slot: slotFilter(cosCands, tagged, state),
  };

  for (const cfg of CONFIGS) {
    const topK = ranked[cfg].slice(0, TOP_K);
    for (const k of [1, 3, 5]) {
      if (topK.slice(0, k).some(t => goldSet.has(t))) retr_agg[cfg][k]++;
      if (topK.slice(0, k).some(t => distSet.has(t))) dist_agg[cfg][k]++;
    }
  }

  // ── State correctness (no LLM) ──────────────────────────────────────────
  // A adopts the latest carrier the detector recovered anywhere in the transcript,
  // which is the same rule twoChannelRetrievalTime applies over the candidate pool.
  const detectedCarrierIdxs = [];
  for (let j = 0; j < turns.length; j++) {
    if (turns[j].role === "user" && tagged[j].is_retraction) detectedCarrierIdxs.push(j);
  }
  const aCutoff = detectedCarrierIdxs.length ? detectedCarrierIdxs[detectedCarrierIdxs.length - 1] : -1;
  let aState;
  if (aCutoff < 0) { aState = "no_carrier"; stateStats.a_no_carrier++; }
  else if (aCutoff === lastCarrier) { aState = "correct"; stateStats.a_correct++; }
  else { aState = "stale_cutoff"; stateStats.a_stale_cutoff++; }

  let bState;
  if (!state.slot) { bState = "no_slot"; stateStats.b_no_slot++; }
  else if (wbContains(state.slot, Xfinal) || wbContains(Xfinal, state.slot)) { bState = "correct"; stateStats.b_correct++; }
  else { bState = "stale_value"; stateStats.b_stale_value++; }

  stateStats.carriers_gold += carrierIdxs.length;
  stateStats.carriers_detected += detectedCarrierIdxs.length;

  const sample = { idx: i, Xs, Xfinal, query: item.query, slot: state.slot, slot_ok: state.updates_ok, slot_failed: state.updates_failed, a_state: aState, b_state: bState, a_cutoff: aCutoff, gold_last_carrier: lastCarrier };
  if (RUN_E2E) {
    for (const cfg of CONFIGS) {
      const topK = ranked[cfg].slice(0, TOP_K);
      if (topK.length === 0) { sample[cfg] = { error: "empty ranking" }; continue; }
      const ctx = topK.map(idx => docTexts[idx]);
      try {
        const ans = await ask(ctx, item.query);
        const yMatch = wbContains(ans, Xfinal);
        const xMatch = Xprior.some(x => wbContains(ans, x));
        let cat;
        if (yMatch && !xMatch) cat = "y_correct";
        else if (!yMatch && xMatch) cat = "x_distractor";
        else cat = "neither";
        e2e_agg[cfg][cat]++;
        sample[cfg] = { answer: ans, category: cat };
      } catch (e) { sample[cfg] = { error: e.message }; }
    }
  }
  perSample.push(sample);
  if ((i + 1) % 5 === 0) {
    const s = CONFIGS.map(c => `${c}:${e2e_agg[c].y_correct}`).join(" ");
    console.log(`  [${i + 1}/${items.length}] ${s}`);
  }
}

const N = items.length;
const report = {
  config: { n: N, k_retractions: K, top_k: TOP_K, model: MODEL, cand_M: CAND_M, e2e: RUN_E2E },
  retrieval_hit: retr_agg,
  retrieval_distractor: dist_agg,
  e2e_llm: e2e_agg,
  slot_stats: slotStats,
  state_correctness: stateStats,
  per_sample: perSample,
};
writeFileSync(OUT_PATH, JSON.stringify(report, null, 2));

const pct = v => (v / N * 100).toFixed(1).padStart(5);
console.log(`\n=== Two-channel chain k=${K}, n=${N} ===`);
console.log(`Retrieval Hit@K`);
console.log(`K      ${CONFIGS.map(c => c.padEnd(7)).join("")}`);
for (const k of [1, 3, 5]) console.log(`@${k}   ${CONFIGS.map(c => pct(retr_agg[c][k]) + "  ").join("")}`);
console.log(`\nDistractor@K (lower better)`);
for (const k of [1, 3, 5]) console.log(`@${k}   ${CONFIGS.map(c => pct(dist_agg[c][k]) + "  ").join("")}`);
if (RUN_E2E) {
  console.log(`\nE2E LLM (${MODEL}, top-${TOP_K})`);
  console.log(`Method          Y_correct  X_distractor  neither`);
  for (const c of CONFIGS) {
    const a = e2e_agg[c];
    console.log(`${c.padEnd(15)} ${(a.y_correct / N * 100).toFixed(1)}%      ${(a.x_distractor / N * 100).toFixed(1)}%        ${(a.neither / N * 100).toFixed(1)}%`);
  }
}
const tot = slotStats.updates_ok + slotStats.updates_failed;
console.log(`\nb_slot ingest updates: ok=${slotStats.updates_ok} failed=${slotStats.updates_failed}` +
  (tot ? ` p_eff=${(slotStats.updates_ok / tot).toFixed(3)}` : "") + ` slot_never_set=${slotStats.slot_null}/${N}`);

const s = stateStats;
const recall = s.carriers_gold ? s.carriers_detected / s.carriers_gold : 0;
console.log(`\n=== State correctness (no LLM), k=${K}, n=${N} ===`);
console.log(`carrier recall              : ${s.carriers_detected}/${s.carriers_gold} = ${(recall * 100).toFixed(1)}%`);
console.log(`A retrieval-time cutoff     : correct ${(s.a_correct / N * 100).toFixed(1)}%   stale ${(s.a_stale_cutoff / N * 100).toFixed(1)}%   none ${(s.a_no_carrier / N * 100).toFixed(1)}%`);
console.log(`B ingest-time slot value    : correct ${(s.b_correct / N * 100).toFixed(1)}%   stale ${(s.b_stale_value / N * 100).toFixed(1)}%   none ${(s.b_no_slot / N * 100).toFixed(1)}%`);
console.log(`  predicted if independent  : A ~ p = ${(recall * 100).toFixed(1)}%   B ~ p^k = ${(Math.pow(recall, K) * 100).toFixed(1)}%`);
console.log(`\nReport: ${OUT_PATH}`);
