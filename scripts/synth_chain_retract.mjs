// Chain-retraction synthetic benchmark (k=3) — validates Theorem 2 prediction
// that retrieval-time wins at k >= k*.
//
// Conversation: setup X1 → carrier1 X2 → carrier2 X3 → final commit X3
// Gold = X3 (final). Distractors = {X1, X2}.
//
// Compares cosine, recency, BAF, (and outputs Mem0-ready transcript for paired run).
// All under same Claude haiku 4.5 E2E grading.

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

const N = Number(process.env.N ?? 50);
const K_RETRACTIONS = Number(process.env.K ?? 3);  // chain length
const TOP_K = 5;
const SEED = Number(process.env.SEED ?? 42);
const MODEL = "claude-haiku-4-5-20251001";
const OUT_PATH = process.env.OUT ?? `/tmp/synth_chain_k${K_RETRACTIONS}_report.json`;
const CONVS_PATH = process.env.CONVS_OUT ?? `/tmp/synth_chain_k${K_RETRACTIONS}_convs.json`;

// ─── Templates with 3 options per domain (X1, X2, X3) ─────────────────────

const DOMAINS = [
  { name:"travel", options:["Paris","Tokyo","London","Rome","Barcelona","Bangkok","Sydney","Lisbon","Vienna","Reykjavik"], setupOriginal:x=>`I want to plan a trip to ${x}.`, setupCommit:x=>`Locked in: we're going to ${x}.`, setupDetail:x=>`Looking at hotels in ${x}.`, afterCommit:y=>`${y} confirmed.`, query:"Where is the user actually traveling?" },
  { name:"framework", options:["React","Vue","Svelte","Solid","Angular","Preact","Lit","Qwik","Astro","Remix"], setupOriginal:x=>`I think we should use ${x}.`, setupCommit:x=>`Decided: ${x} is our stack.`, setupDetail:x=>`Started scaffolding ${x}.`, afterCommit:y=>`${y} confirmed.`, query:"Which frontend framework is the user using?" },
  { name:"restaurant", options:["Italian","Japanese","Mexican","Thai","French","Korean","Indian","Greek","Vietnamese","Spanish"], setupOriginal:x=>`I want ${x} food tonight.`, setupCommit:x=>`${x} place booked.`, setupDetail:x=>`Looking up ${x} options.`, afterCommit:y=>`${y} confirmed for dinner.`, query:"What cuisine did the user end up choosing?" },
  { name:"vacation", options:["beach house","cabin","downtown loft","treehouse","yurt","farmhouse","lake cottage","ski chalet","desert villa","houseboat"], setupOriginal:x=>`Considering booking a ${x}.`, setupCommit:x=>`${x} rental locked in.`, setupDetail:x=>`Reading reviews for ${x}.`, afterCommit:y=>`${y} confirmed.`, query:"Which vacation rental did the user finalize?" },
  { name:"gift", options:["watch","book","espresso machine","perfume","headphones","scarf","fountain pen","camera lens","wine subscription","kitchen knife"], setupOriginal:x=>`Thinking of getting them a ${x}.`, setupCommit:x=>`Ordered the ${x}.`, setupDetail:x=>`Wrapping the ${x}.`, afterCommit:y=>`${y} delivered.`, query:"What gift did the user actually purchase?" },
];

const CARRIERS = [
  y=>`Actually, let's switch to ${y} instead.`,
  y=>`Wait, change of plan: ${y}.`,
  y=>`Forget that, going with ${y}.`,
  y=>`No, let's do ${y} instead.`,
  y=>`Actually scratch that, ${y}.`,
];
const FILLERS = ["Any thoughts?","What's the timeline?","Tell me more.","What's the budget?"];

function mulberry32(s){let a=s>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
const rng = mulberry32(SEED);
const pick = a => a[Math.floor(rng()*a.length)];

function genConv(idx) {
  const dom = DOMAINS[idx % DOMAINS.length];
  const opts = [...dom.options];
  for (let i=opts.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[opts[i],opts[j]]=[opts[j],opts[i]];}
  // X[0..k] are the supersession chain; final X[k] is gold
  const Xs = opts.slice(0, K_RETRACTIONS + 1);
  const Xfinal = Xs[Xs.length - 1];
  const turns = [];
  const carrierIdxs = [];
  // Phase 0: setup X[0]
  turns.push({role:"user",content:dom.setupOriginal(Xs[0])});
  turns.push({role:"assistant",content:`${Xs[0]} sounds great.`});
  turns.push({role:"user",content:dom.setupCommit(Xs[0])});
  turns.push({role:"assistant",content:`Confirmed: ${Xs[0]}.`});
  turns.push({role:"user",content:dom.setupDetail(Xs[0])});
  turns.push({role:"user",content:pick(FILLERS)});
  turns.push({role:"assistant",content:`Sure.`});
  // Phases 1..k: each retraction switches X[i-1] → X[i]
  for (let phase = 1; phase < Xs.length; phase++) {
    const Xnew = Xs[phase];
    carrierIdxs.push(turns.length);
    turns.push({role:"user",content:pick(CARRIERS)(Xnew)});
    turns.push({role:"assistant",content:`Got it, switching to ${Xnew}.`});
    turns.push({role:"user",content:dom.setupCommit(Xnew)});
    turns.push({role:"assistant",content:`Confirmed: ${Xnew}.`});
    turns.push({role:"user",content:dom.setupDetail(Xnew)});
    if (phase < Xs.length - 1) {
      turns.push({role:"user",content:pick(FILLERS)});
      turns.push({role:"assistant",content:`Understood.`});
    } else {
      turns.push({role:"user",content:dom.afterCommit(Xnew)});
      turns.push({role:"assistant",content:`${Xnew} locked in.`});
    }
  }

  // Gold = chunks mentioning X_final after the last carrier
  // Distractors = chunks mentioning any X[0..k-1] in their respective phase
  const lastCarrier = carrierIdxs[carrierIdxs.length - 1];
  const goldIdxs=[], distIdxs=[];
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i].content;
    if (i > lastCarrier && t.includes(Xfinal) && !Xs.slice(0, -1).some(x => t.includes(x))) goldIdxs.push(i);
    // For each phase j in 0..k-1, dist chunks are between carrierIdxs[j-1]+1 and carrierIdxs[j]
    for (let j = 0; j < Xs.length - 1; j++) {
      const phaseStart = j === 0 ? 0 : carrierIdxs[j-1];
      const phaseEnd = carrierIdxs[j];
      if (i >= phaseStart && i < phaseEnd && t.includes(Xs[j])) {
        distIdxs.push(i);
        break;
      }
    }
  }
  return { idx, Xs, X1: Xs[0], X2: Xs[1] ?? "", X3: Xfinal, lastCarrier, carrierIdxs, query:dom.query, turns, goldIdxs, distIdxs };
}

// ─── Filter logic (P2) ──────────────────────────────────────────────────────

const PRIOR_CTX=4, EVENT_FORCE=1.0, COS_T=0.30, EDGE_T=0.15, COMMIT_LOW=0.5, COMMIT_HIGH=1.0;
const EVENT_RE=/(?:\bI\s+(?:just|recently|finally|honestly)\s+(?:had|got|started|moved|became|quit|lost|met|broke)|\bsince\s+(?:I|my|the|that|then|last|we)|\bnow\s+(?:that|I'm|I\s+am|I\s+have|I've))/i;
const COMMITMENT_RE=/(?:\bbook(?:ed|ing)?\b|\bconfirm(?:ed)?\b|\bdecide[ds]?\b|\b(?:will|would|shall)\s+(?:use|do|go|book|take)\b|\bgoing\s+to\b|\bplan(?:ning)?\s+to\b|\blocked?\s+in\b|\border(?:ed)?\b|\breserv(?:ed|ing)?\b)/i;

function cosine(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
function wbContains(text, term) { return new RegExp(`\\b${escapeRe(term)}\\b`, "i").test(text); }
function topMCos(qVec,docVecs,m){return docVecs.map((v,i)=>({i,s:cosine(qVec,v)})).sort((a,b)=>b.s-a.s).slice(0,m);}
function topKRec(qVec,docVecs,k,lambda=0.1){const N=docVecs.length;return docVecs.map((v,i)=>({i,s:cosine(qVec,v)+lambda*(i/Math.max(1,N-1))})).sort((a,b)=>b.s-a.s).slice(0,k);}

function detectCarrier(text, ctx) {
  const tag = detectRetraction(text, ctx);
  if (tag.is_retraction) return tag;
  if (EVENT_RE.test(text)) return { is_retraction:true, speech_act_class:"EventUtterance", retraction_weight:EVENT_FORCE, target_keywords:[], destination_slot:null };
  return tag;
}

function bafFilter(scoredCands, flatTurns, k, docVecs) {
  const cands = scoredCands.map(c => {
    const turn = flatTurns[c.i];
    const ctxStart = Math.max(0, c.i - PRIOR_CTX);
    const priorCtx = flatTurns.slice(ctxStart, c.i).map(t=>t.content).join(" ");
    const tag = detectCarrier(turn.content, priorCtx);
    return { id:c.i, chunk_text:turn.content, timestamp:`T${String(c.i).padStart(6,"0")}`, role:turn.role, is_retraction:tag.is_retraction, target_keywords:tag.target_keywords, retraction_weight:tag.retraction_weight };
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
      edges.push({ attacker_id:carrier.id, target_id:target.id, weight:rescaled });
    }
  }
  const { survivors } = filterByGroundedExtension(cands, edges);
  const scoreById = new Map(scoredCands.map(c => [c.i, c.s]));
  survivors.sort((a, b) => (scoreById.get(b.id) ?? 0) - (scoreById.get(a.id) ?? 0));
  return { topK: survivors.slice(0, k).map(s => s.id), nCarriers: cands.filter(c => c.is_retraction).length, nFiltered: cands.length - survivors.length };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function ask(retrieved, query) {
  const ctxBlock = retrieved.map((c, i) => `[${i+1}] ${c}`).join("\n");
  const prompt = `Below is retrieved context from a conversation. Answer the question using ONLY this context. Respond in one short word/phrase, just the answer with no explanation.\n\nContext:\n${ctxBlock}\n\nQuestion: ${query}\n\nAnswer:`;
  for (let attempt = 0; attempt < 5; attempt++) {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      // temperature pinned 2026-07-27. Left unset, the API default of 1.0 made the
      // answer metric irreproducible: on short chains the model intermittently
      // answers with both the superseded and the current value ("Both", "X and Y"),
      // which the grader scores as neither, and the rate of that behaviour moves
      // Y-correct by tens of points between runs of identical code.
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

// ─── Main ──────────────────────────────────────────────────────────────────

console.log(`Chain-retract synth k=${K_RETRACTIONS}, n=${N}`);
const items = Array.from({ length: N }, (_, i) => genConv(i));

// Save conversations for Mem0 to ingest later
const mem0Input = items.map(it => ({ idx: it.idx, Xs: it.Xs, X1: it.X1, X2: it.X2, X3: it.X3, query: it.query, turns: it.turns }));
writeFileSync(CONVS_PATH, JSON.stringify(mem0Input, null, 2));

if (process.env.SKIP_LLM === "1") {
  console.log(`Dumped ${items.length} convs to ${CONVS_PATH}; SKIP_LLM=1, exiting.`);
  process.exit(0);
}

await initEmbedding();
console.log(`Embedding loaded`);

const retr_agg = { cos: {1:0,3:0,5:0}, baf: {1:0,3:0,5:0}, rec: {1:0,3:0,5:0} };
const dist_agg = { cos: {1:0,3:0,5:0}, baf: {1:0,3:0,5:0}, rec: {1:0,3:0,5:0} };
const e2e_agg = { cosine: {y_correct:0, x_distractor:0, neither:0}, recency: {y_correct:0, x_distractor:0, neither:0}, baf: {y_correct:0, x_distractor:0, neither:0} };
const perSample = [];

for (let i = 0; i < items.length; i++) {
  const item = items[i];
  const docTexts = item.turns.map(t => `${t.role}: ${t.content}`);
  const docBufs = await embedBatch(docTexts);
  const docVecs = docBufs.map(bufferToFloat32Array);
  const qBuf = await embed(item.query);
  const qVec = bufferToFloat32Array(qBuf);
  const M = Math.min(Number(process.env.MCAP ?? 20), docTexts.length);

  const cosCands = topMCos(qVec, docVecs, M);
  const cosIdxs = cosCands.map(c=>c.i);
  const recIdxs = topKRec(qVec, docVecs, M).map(r=>r.i);
  const baf = bafFilter(cosCands, item.turns, M, docVecs);
  const bafIdxs = baf.topK;

  // Retrieval-level Hit/Distractor
  const goldSet = new Set(item.goldIdxs);
  const distSet = new Set(item.distIdxs);
  for (const k of [1,3,5]) {
    if (cosIdxs.slice(0,k).some(t=>goldSet.has(t))) retr_agg.cos[k]++;
    if (bafIdxs.slice(0,k).some(t=>goldSet.has(t))) retr_agg.baf[k]++;
    if (recIdxs.slice(0,k).some(t=>goldSet.has(t))) retr_agg.rec[k]++;
    if (cosIdxs.slice(0,k).some(t=>distSet.has(t))) dist_agg.cos[k]++;
    if (bafIdxs.slice(0,k).some(t=>distSet.has(t))) dist_agg.baf[k]++;
    if (recIdxs.slice(0,k).some(t=>distSet.has(t))) dist_agg.rec[k]++;
  }

  // E2E LLM (top-5)
  const sample = { idx: i, Xs: item.Xs, X1: item.X1, X2: item.X2, X3: item.X3, query: item.query };
  const Xfinal = item.Xs[item.Xs.length - 1];
  const Xprior = item.Xs.slice(0, -1);
  for (const [label, idxs] of [["cosine", cosIdxs.slice(0,TOP_K)], ["recency", recIdxs.slice(0,TOP_K)], ["baf", bafIdxs.slice(0,TOP_K)]]) {
    const ctx = idxs.map(idx => docTexts[idx]);
    try {
      const ans = await ask(ctx, item.query);
      const yMatch = wbContains(ans, Xfinal);
      const xMatch = Xprior.some(x => wbContains(ans, x));
      let cat;
      if (yMatch && !xMatch) cat = "y_correct";
      else if (!yMatch && xMatch) cat = "x_distractor";
      else cat = "neither";
      e2e_agg[label][cat]++;
      sample[label] = { answer: ans, category: cat };
    } catch (e) { sample[label] = { error: e.message }; }
  }
  perSample.push(sample);
  if ((i+1) % 5 === 0) console.log(`  [${i+1}/${N}] e2e cos:y=${e2e_agg.cosine.y_correct}/x=${e2e_agg.cosine.x_distractor}  baf:y=${e2e_agg.baf.y_correct}/x=${e2e_agg.baf.x_distractor}`);
}

const report = {
  config: { n: N, k_retractions: K_RETRACTIONS, top_k: TOP_K, model: MODEL },
  retrieval_hit: retr_agg,
  retrieval_distractor: dist_agg,
  e2e_llm: e2e_agg,
  per_sample: perSample,
};
writeFileSync(OUT_PATH, JSON.stringify(report, null, 2));

console.log(`\n=== Chain-Retraction k=${K_RETRACTIONS}, n=${N} ===`);
console.log(`Retrieval Hit@K (gold = X${K_RETRACTIONS} chunks):`);
console.log(`K     cosine  recency  baf`);
for (const k of [1,3,5]) console.log(`@${k}    ${retr_agg.cos[k]/N*100}    ${retr_agg.rec[k]/N*100}    ${retr_agg.baf[k]/N*100}`);
console.log(`\nRetrieval Distractor@K (any X[0..${K_RETRACTIONS-1}] chunks):`);
console.log(`K     cosine  recency  baf`);
for (const k of [1,3,5]) console.log(`@${k}    ${dist_agg.cos[k]/N*100}    ${dist_agg.rec[k]/N*100}    ${dist_agg.baf[k]/N*100}`);
console.log(`\nE2E LLM (Claude haiku-4-5, top-${TOP_K}):`);
console.log(`Method      Y_correct  X_distractor  neither`);
for (const m of ["cosine","recency","baf"]) {
  const a = e2e_agg[m];
  console.log(`${m.padEnd(10)}  ${(a.y_correct/N*100).toFixed(1)}%      ${(a.x_distractor/N*100).toFixed(1)}%        ${(a.neither/N*100).toFixed(1)}%`);
}
console.log(`\nReport: /tmp/synth_chain_k3_report.json`);
console.log(`Mem0 input: /tmp/synth_chain_k3_convs.json (run mem0_chain_baseline.py for paired comparison)`);
