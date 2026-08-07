// Delta-retract multi-scorer audit (Protocol 1 extension, deterministic).
//
// Adds the commitment control H_C to the deterministic suite and widens the scorer
// family beyond the pipeline embedder. For each of the 100 base-direction units
// (50 k=1 conversations x 2 directions) and each of the 5 retraction paraphrases:
//   H_0 : 7-turn X-support prefix, nothing attached
//   H_N : prefix + neutral mention of the incoming value ("My colleague mentioned Y ...")
//   H_C : prefix + bare new commitment ("Let's go with Y.")  <- same wording as the
//         Mem0 commitment-control arm (protocol1_mem0_control.py)
//   H_R : prefix + retraction paraphrase p (same 5 templates as protocol1_audit.mjs)
//
// Target candidate c0 = the committed utterance (turn index 2). Measured per scorer
// and condition: pointwise support s(c0, q), rank(c0), top-5 inclusion. Headline
// deltas: d_RC = s(H_R) - s(H_C) (marginal effect of explicit retraction marking,
// new commitment held fixed), d_CN = s(H_C) - s(H_N), d_RN = s(H_R) - s(H_N).
//
// Scorers:
//   cos_mxbai  : cosine over mxbai-embed-large-v1 (pipeline embedder, dist/embedding.js)
//   cos_minilm : cosine over Xenova/all-MiniLM-L6-v2 (mean pool, normalized)
//   cos_bge    : cosine over Xenova/bge-small-en-v1.5 (mean pool, normalized; the
//                model's optional query instruction prefix is not used, since the
//                measured property is history-invariance of s(c, q), which the
//                prefix cannot affect)
//   ce_msmarco : cross-encoder Xenova/ms-marco-MiniLM-L-6-v2 relevance logit on (q, c)
//   bm25       : Okapi BM25 (k1=1.5, b=0.75) over the per-history turn pool; the only
//                scorer whose corpus statistics (N, df, avgdl) read the full history,
//                so the only one where d_RC is not fixed at zero a priori
//
// The three embedding scorers and the cross-encoder read only the (candidate, query)
// pair, so their per-candidate support is cached by text and identical across
// conditions by construction; their rank and top-5 columns still vary with pool
// composition and are measured per history. BM25 is recomputed per history.
//
// Two-stage execution (onnxruntime-node crashes when the dist/embedding.js worker
// thread and main-thread transformers.js sessions coexist in one process, so the
// pipeline embedder runs in its own process):
//   STAGE=mxbai node scripts/delta_retract_multiscorer.mjs
//   STAGE=rest  node scripts/delta_retract_multiscorer.mjs
// Stage mxbai writes STAGE1_PATH (default /tmp/delta_retract_stage1.json) and enforces
// the verification gate; stage rest reads it, adds the other four scorers, and writes
// the final OUT (default /tmp/delta_retract_multiscorer.json).
//
// Verification gate (stage mxbai): the cos_mxbai s0 values for H_0/H_N/H_R must
// reproduce the archived exp_tmp/protocol1_audit.json per-item values to 1e-6,
// otherwise the stage refuses to write output (guards against suite drift).
//
// Unit construction is copied verbatim from protocol1_audit.mjs, including its
// "rev" branch: there wbReplace(content, X, xb) has xb = X, an identity map, so
// the rev variant keeps the X-committing prefix and its continuation turns name
// the already-committed X (a same-value marking control), whereas the fwd variant
// names the incoming Y. The header comment of the original script says the rev
// arm mirrors X and Y; the code does not, and the archived suite is the code's
// output, so this script reproduces the code, not the comment.

import { writeFileSync, readFileSync } from "fs";

const BASE = ".";
const CONVS_PATH = `${BASE}/exp_tmp/synth_chain_k1_convs.json`;
const REF_AUDIT_PATH = `${BASE}/exp_tmp/protocol1_audit.json`;
const STAGE1_PATH = process.env.STAGE1 ?? "/tmp/delta_retract_stage1.json";
const OUT_PATH = process.env.OUT ?? "/tmp/delta_retract_multiscorer.json";
const STAGE = process.env.STAGE ?? "rest";
const GATE_TOL = 1e-6;

const PARAPHRASES = [
  y => `Wait, change of plan: ${y}.`,
  y => `No, let's do ${y} instead.`,
  y => `Actually, let's switch to ${y} instead.`,
  y => `Actually scratch that, ${y}.`,
  y => `Forget that, going with ${y}.`,
];
const NEUTRAL = y => `My colleague mentioned ${y} the other day.`;
const COMMIT = y => `Let's go with ${y}.`;

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function wbReplace(text, from, to) { return text.replace(new RegExp(`\\b${escapeRe(from)}\\b`, "g"), to); }

function tokenizeBm25(s) { return s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean); }

function bm25AllScores(docTexts, query) {
  const K1 = 1.5, B = 0.75;
  const docs = docTexts.map(tokenizeBm25);
  const N = docs.length;
  const avgdl = docs.reduce((a, d) => a + d.length, 0) / N;
  const df = new Map();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
  const qTerms = tokenizeBm25(query);
  return docs.map(d => {
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    let s = 0;
    for (const t of qTerms) {
      const f = tf.get(t) ?? 0;
      if (f === 0) continue;
      const n = df.get(t) ?? 0;
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);
      s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * d.length / avgdl));
    }
    return s;
  });
}

function rankStats(scores, c0Idx) {
  const order = scores.map((s, i) => ({ i, s })).sort((a, b) => b.s - a.s);
  const rank0 = order.findIndex(c => c.i === c0Idx) + 1;
  return { s0: scores[c0Idx], rank0, top5: order.slice(0, 5).some(c => c.i === c0Idx) ? 1 : 0 };
}

function buildUnits(items) {
  const units = [];
  let skipped = 0;
  for (const item of items) {
    const Xs = item.Xs ?? [item.X1, item.X2].filter(Boolean);
    const [X, Y] = [Xs[0], Xs[Xs.length - 1]];
    const prefixOrig = item.turns.slice(0, 7);
    if (!prefixOrig[2] || !prefixOrig[2].content.includes(X)) { skipped++; continue; }
    for (const dir of ["fwd", "rev"]) {
      const xb = dir === "fwd" ? Y : X;
      const prefix = dir === "fwd" ? prefixOrig : prefixOrig.map(t => ({ role: t.role, content: wbReplace(t.content, X, xb) }));
      const yTo = dir === "fwd" ? Y : X;
      const conds = {
        H0: prefix,
        HN: [...prefix, { role: "user", content: NEUTRAL(yTo) }],
        HC: [...prefix, { role: "user", content: COMMIT(yTo) }],
      };
      for (let p = 0; p < PARAPHRASES.length; p++) {
        conds[`HR${p}`] = [...prefix, { role: "user", content: PARAPHRASES[p](yTo) }];
      }
      units.push({ idx: item.idx, dir, query: item.query, c0Idx: 2, conds });
    }
  }
  return { units, skipped };
}

function docTextsOf(turns) { return turns.map(t => `${t.role}: ${t.content}`); }

const items = JSON.parse(readFileSync(CONVS_PATH, "utf8"));
const { units, skipped } = buildUnits(items);
console.log(`units: ${units.length}, skipped convs: ${skipped}, stage: ${STAGE}`);

if (STAGE === "mxbai") {
  const { initEmbedding, embedBatch, bufferToFloat32Array } = await import("../dist/embedding.js");
  const refAudit = JSON.parse(readFileSync(REF_AUDIT_PATH, "utf8"));
  const refByKey = new Map(refAudit.results.map(r => [`${r.idx}|${r.dir}|${r.paraphrase}`, r]));
  await initEmbedding();
  console.log("mxbai loaded");

  const cache = new Map();
  async function fill(texts) {
    const missing = [...new Set(texts)].filter(t => !cache.has(t));
    if (missing.length === 0) return;
    const bufs = await embedBatch(missing);
    missing.forEach((t, i) => cache.set(t, bufferToFloat32Array(bufs[i])));
  }

  const stage1 = {};
  const gateFails = [];
  for (const u of units) {
    const rec = {};
    for (const [cond, turns] of Object.entries(u.conds)) {
      const docs = docTextsOf(turns);
      await fill([...docs, u.query]);
      const qVec = cache.get(u.query);
      rec[cond] = rankStats(docs.map(t => cosine(qVec, cache.get(t))), u.c0Idx);
    }
    for (let p = 0; p < PARAPHRASES.length; p++) {
      const ref = refByKey.get(`${u.idx}|${u.dir}|${p}`);
      if (!ref) { gateFails.push(`missing archived item ${u.idx}|${u.dir}|${p}`); continue; }
      for (const [mineCond, refCond] of [["H0", "H0"], ["HN", "HN"], [`HR${p}`, "HR"]]) {
        const d = Math.abs(rec[mineCond].s0 - ref[refCond].s0);
        if (d > GATE_TOL) gateFails.push(`${u.idx}|${u.dir}|${p} ${refCond} mxbai s0 drift ${d}`);
      }
    }
    stage1[`${u.idx}|${u.dir}`] = rec;
    console.log(`unit ${u.idx}|${u.dir} done`);
  }
  if (gateFails.length > 0) {
    console.error(`VERIFICATION GATE FAILED (${gateFails.length}):`);
    for (const f of gateFails.slice(0, 12)) console.error(" ", f);
    process.exit(1);
  }
  console.log(`gate passed: cos_mxbai reproduces archived protocol1_audit.json to ${GATE_TOL}`);
  writeFileSync(STAGE1_PATH, JSON.stringify({ gate: "passed", gate_tol: GATE_TOL, stage1 }, null, 1));
  console.log(`written: ${STAGE1_PATH}`);
  process.exit(0);
}

const stage1File = JSON.parse(readFileSync(STAGE1_PATH, "utf8"));
if (stage1File.gate !== "passed") { console.error("stage1 gate not passed"); process.exit(1); }
const stage1 = stage1File.stage1;

const { pipeline, AutoTokenizer, AutoModelForSequenceClassification } = await import("@huggingface/transformers");
console.log("loading minilm, bge, cross-encoder");
const minilmPipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const bgePipe = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5");
const ceTok = await AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
const ceModel = await AutoModelForSequenceClassification.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
console.log("all stage-2 scorers loaded");

const embCache = { cos_minilm: new Map(), cos_bge: new Map() };
const ceCache = new Map();

async function fillEmbCache(scorer, texts) {
  const cache = embCache[scorer];
  const missing = [...new Set(texts)].filter(t => !cache.has(t));
  if (missing.length === 0) return;
  const pipe = scorer === "cos_minilm" ? minilmPipe : bgePipe;
  const BATCH = 32;
  for (let i = 0; i < missing.length; i += BATCH) {
    const chunk = missing.slice(i, i + BATCH);
    const out = await pipe(chunk, { pooling: "mean", normalize: true });
    const rows = out.tolist();
    chunk.forEach((t, j) => cache.set(t, Float32Array.from(rows[j])));
  }
}

async function fillCeCache(pairs) {
  const seen = new Set();
  const uniq = [];
  for (const p of pairs) {
    const k = `${p[0]} ${p[1]}`;
    if (!ceCache.has(k) && !seen.has(k)) { seen.add(k); uniq.push(p); }
  }
  const BATCH = 16;
  for (let i = 0; i < uniq.length; i += BATCH) {
    const chunk = uniq.slice(i, i + BATCH);
    const inputs = ceTok(chunk.map(p => p[0]), { text_pair: chunk.map(p => p[1]), padding: true, truncation: true });
    const { logits } = await ceModel(inputs);
    const data = logits.data;
    chunk.forEach((p, j) => ceCache.set(`${p[0]} ${p[1]}`, Number(data[j])));
  }
}

const SCORERS = ["cos_mxbai", "cos_minilm", "cos_bge", "ce_msmarco", "bm25"];
const results = [];

for (const u of units) {
  const s1 = stage1[`${u.idx}|${u.dir}`];
  if (!s1) { console.error(`stage1 missing unit ${u.idx}|${u.dir}`); process.exit(1); }
  const perCond = {};
  for (const [cond, turns] of Object.entries(u.conds)) {
    const docs = docTextsOf(turns);
    const m = { cos_mxbai: s1[cond] };
    for (const scorer of ["cos_minilm", "cos_bge"]) {
      await fillEmbCache(scorer, [...docs, u.query]);
      const cache = embCache[scorer];
      const qVec = cache.get(u.query);
      m[scorer] = rankStats(docs.map(t => cosine(qVec, cache.get(t))), u.c0Idx);
    }
    await fillCeCache(docs.map(d => [u.query, d]));
    m.ce_msmarco = rankStats(docs.map(d => ceCache.get(`${u.query} ${d}`)), u.c0Idx);
    m.bm25 = rankStats(bm25AllScores(docs, u.query), u.c0Idx);
    perCond[cond] = m;
  }
  for (let p = 0; p < PARAPHRASES.length; p++) {
    const row = { idx: u.idx, dir: u.dir, paraphrase: p };
    for (const sc of SCORERS) {
      row[sc] = { H0: perCond.H0[sc], HN: perCond.HN[sc], HC: perCond.HC[sc], HR: perCond[`HR${p}`][sc] };
    }
    results.push(row);
  }
  console.log(`unit ${u.idx}|${u.dir} done (${results.length} rows)`);
}

const agg = {};
for (const sc of SCORERS) {
  const a = {
    max_abs_dRC_s0: 0, max_abs_dCN_s0: 0, max_abs_dRN_s0: 0,
    mean_dRC_s0: 0, n_dRC_nonzero: 0,
    rank_changed_RC: 0, rank_changed_RN: 0, rank_changed_CN: 0,
    top5_changed_RC: 0, top5_changed_RN: 0,
  };
  for (const r of results) {
    const dRC = r[sc].HR.s0 - r[sc].HC.s0;
    const dCN = r[sc].HC.s0 - r[sc].HN.s0;
    const dRN = r[sc].HR.s0 - r[sc].HN.s0;
    a.max_abs_dRC_s0 = Math.max(a.max_abs_dRC_s0, Math.abs(dRC));
    a.max_abs_dCN_s0 = Math.max(a.max_abs_dCN_s0, Math.abs(dCN));
    a.max_abs_dRN_s0 = Math.max(a.max_abs_dRN_s0, Math.abs(dRN));
    a.mean_dRC_s0 += dRC / results.length;
    if (Math.abs(dRC) > 0) a.n_dRC_nonzero++;
    if (r[sc].HR.rank0 !== r[sc].HC.rank0) a.rank_changed_RC++;
    if (r[sc].HR.rank0 !== r[sc].HN.rank0) a.rank_changed_RN++;
    if (r[sc].HC.rank0 !== r[sc].HN.rank0) a.rank_changed_CN++;
    if (r[sc].HR.top5 !== r[sc].HC.top5) a.top5_changed_RC++;
    if (r[sc].HR.top5 !== r[sc].HN.top5) a.top5_changed_RN++;
  }
  agg[sc] = a;
}

const out = {
  config: {
    scorers: SCORERS, n_rows: results.length, gate_tol: GATE_TOL,
    bm25: { k1: 1.5, b: 0.75 },
    models: {
      cos_mxbai: "mxbai-embed-large-v1 (pipeline embedder, stage 1)",
      cos_minilm: "Xenova/all-MiniLM-L6-v2",
      cos_bge: "Xenova/bge-small-en-v1.5",
      ce_msmarco: "Xenova/ms-marco-MiniLM-L-6-v2",
    },
    conditions: { HN: NEUTRAL("Y"), HC: COMMIT("Y"), HR: PARAPHRASES.map(f => f("Y")) },
  },
  agg, results,
};
writeFileSync(OUT_PATH, JSON.stringify(out, null, 1));
console.log(JSON.stringify(agg, null, 2));
console.log(`written: ${OUT_PATH}`);
