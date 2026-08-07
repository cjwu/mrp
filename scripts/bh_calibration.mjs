// B/H calibration for Corollary 1 (post-index MRP under bounded base-score advantage).
//
// For each chain conversation at final query time, take the superseded commitment c-
// (second-to-last value's "Locked in" utterance) and the replacement commitment c+
// (final value's). Measure, with the pipeline's own embedder:
//   b(c,q)  : cosine relevance of the commitment utterance to the query
//   h+(v)   : positive-history count = turns mentioning the value, excluding gold carriers
// Report per depth m in {1,2,8}: distributions of db = b(c+)-b(c-) and dh = h(c-)-h(c+),
// B_hat = P95(db), H_hat = P5(dh), the fraction with dh > 0 (dominance premise), the
// pure-relevance inversion rate (b(c-) > b(c+)), and the lambda curve: for an additive
// scorer s = b + lambda*h+, the fraction of pairs with lambda*dh > db, i.e. predicted
// strict MRP occurrence rate as a function of lambda.

import { writeFileSync, readFileSync } from "fs";
import { initEmbedding, embed, embedBatch, bufferToFloat32Array } from "../dist/embedding.js";

const BASE = "./exp_tmp";
const OUT_PATH = process.env.OUT ?? "/tmp/bh_calibration.json";
const FILES = [["k1", 1], ["k3", 2], ["k8", 8]];
const CARRIER_RE = /(actually|change of plan|forget|scratch|let'?s? (switch|do))/i;
const LAMBDAS = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function wbContains(text, term) { return new RegExp(`\\b${escapeRe(term)}\\b`, "i").test(text); }
function pct(arr, p) { const a = [...arr].sort((x, y) => x - y); const i = Math.min(a.length - 1, Math.max(0, Math.ceil(p / 100 * a.length) - 1)); return a[i]; }
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

await initEmbedding();
console.log("Embedder loaded");

const out = { per_m: {}, per_conv: [] };
for (const [tag, m] of FILES) {
  const items = JSON.parse(readFileSync(`${BASE}/synth_chain_${tag}_convs.json`, "utf8"));
  const rows = [];
  for (const item of items) {
    const turns = item.turns;
    const Xs = item.Xs ?? [item.X1, item.X2, item.X3].filter(Boolean);
    const vMinus = Xs[Xs.length - 2], vPlus = Xs[Xs.length - 1];
    const isCarrier = turns.map(t => t.role === "user" && CARRIER_RE.test(t.content));
    const commitIdx = v => {
      let first = -1;
      for (let j = 0; j < turns.length; j++) {
        if (isCarrier[j] || turns[j].role !== "user" || !wbContains(turns[j].content, v)) continue;
        if (/locked in/i.test(turns[j].content)) return j;
        if (first < 0) first = j;
      }
      return first;
    };
    const iMinus = commitIdx(vMinus), iPlus = commitIdx(vPlus);
    if (iMinus < 0 || iPlus < 0) { rows.push(null); continue; }
    const hOf = v => turns.reduce((n, t, j) => n + (!isCarrier[j] && wbContains(t.content, v) ? 1 : 0), 0);
    const texts = [`${turns[iMinus].role}: ${turns[iMinus].content}`, `${turns[iPlus].role}: ${turns[iPlus].content}`];
    const vecs = (await embedBatch(texts)).map(bufferToFloat32Array);
    const qVec = bufferToFloat32Array(await embed(item.query));
    const bMinus = cosine(qVec, vecs[0]), bPlus = cosine(qVec, vecs[1]);
    const row = { m, idx: item.idx, vMinus, vPlus, bMinus, bPlus, hMinus: hOf(vMinus), hPlus: hOf(vPlus) };
    row.db = bPlus - bMinus;
    row.dh = row.hMinus - row.hPlus;
    rows.push(row);
    out.per_conv.push(row);
  }
  const ok = rows.filter(Boolean);
  const dbs = ok.map(r => r.db), dhs = ok.map(r => r.dh);
  const lambdaCurve = Object.fromEntries(LAMBDAS.map(l => [l, ok.filter(r => l * r.dh > r.db).length / ok.length]));
  out.per_m[m] = {
    n: ok.length, skipped: rows.length - ok.length,
    db_mean: mean(dbs), db_p5: pct(dbs, 5), db_p50: pct(dbs, 50), db_p95: pct(dbs, 95),
    dh_mean: mean(dhs), dh_p5: pct(dhs, 5), dh_p50: pct(dhs, 50), dh_p95: pct(dhs, 95),
    B_hat_p95: pct(dbs, 95),
    H_hat_p5: pct(dhs, 5),
    frac_dh_pos: ok.filter(r => r.dh > 0).length / ok.length,
    pure_relevance_inversion: ok.filter(r => r.bMinus > r.bPlus).length / ok.length,
    lambda_curve: lambdaCurve,
  };
  console.log(`m=${m}:`, JSON.stringify(out.per_m[m], null, 1));
}
writeFileSync(OUT_PATH, JSON.stringify(out, null, 1));
console.log(`written: ${OUT_PATH}`);
