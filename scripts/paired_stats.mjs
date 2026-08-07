// Paired statistics for the chain grid and the Mem0 marker (review fixes #3 #4).
//
// All method comparisons run on the SAME 50 conversations per depth, so
// independent-sample interval methods are the wrong tool. This script computes,
// from the pinned per-sample records:
//   - discordant pair counts (b = first method correct & second wrong, c = reverse)
//   - exact two-sided McNemar p (binomial on the discordant pairs)
//   - paired risk difference with a seeded percentile bootstrap CI over
//     conversations (B = 100,000, mulberry32 seed 20260731)
// for the end-to-end comparisons cited in the paper and for the deterministic
// retrieval-side comparisons (Hit@1 / Hit@3 at m = 8).
//
// Also emits cluster-safe Mem0 marker statistics: the sampling unit is the item
// (30 items x 3 replicates), so the per-item all-replicates marker rate is the
// honest count, with the held-out test split reported separately.
//
// Inputs: exp_tmp/pinned_twochannel_k{1,3,8}.json, exp_tmp/pinned_mem0_k{1,3,8}.json,
//         exp_tmp/retrieval_audit_split_2026-07-31.json, exp_tmp/protocol1_mem0_ext.json
// Run: node scripts/paired_stats.mjs

import { writeFileSync, readFileSync } from "fs";

const BASE = "./exp_tmp";
const OUT = process.env.OUT ?? "/tmp/paired_stats.json";
const B = 100000;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function lnChoose(n, k) {
  let s = 0;
  for (let i = 0; i < k; i++) s += Math.log(n - i) - Math.log(i + 1);
  return s;
}
function binomTail2sided(b, c) {
  const n = b + c;
  if (n === 0) return 1;
  const k = Math.max(b, c);
  let tail = 0;
  for (let i = k; i <= n; i++) tail += Math.exp(lnChoose(n, i) - n * Math.LN2);
  return Math.min(1, 2 * tail);
}

function pairedStats(x, y, seed) {
  const n = x.length;
  let b = 0, c = 0;
  for (let i = 0; i < n; i++) {
    if (x[i] === 1 && y[i] === 0) b++;
    if (x[i] === 0 && y[i] === 1) c++;
  }
  const diff = (b - c) / n;
  const rng = mulberry32(seed);
  const diffs = new Float64Array(B);
  for (let r = 0; r < B; r++) {
    let d = 0;
    for (let i = 0; i < n; i++) {
      const j = Math.floor(rng() * n);
      d += x[j] - y[j];
    }
    diffs[r] = d / n;
  }
  diffs.sort();
  const lo = diffs[Math.floor(0.025 * B)];
  const hi = diffs[Math.ceil(0.975 * B) - 1];
  return { n, b, c, diff_pp: 100 * diff, ci_lo_pp: 100 * lo, ci_hi_pp: 100 * hi, mcnemar_p: binomTail2sided(b, c) };
}

function wilson(k, n) {
  const z = 1.959963984540054, p = k / n;
  const d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d;
  const h = (z / d) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, c - h) * 100, Math.min(1, c + h) * 100];
}

const KS = [[1, 1], [3, 2], [8, 8]];
const report = { e2e: {}, retrieval: {}, mem0_marker: {} };

let seed = 20260731;
for (const [K, m] of KS) {
  const det = JSON.parse(readFileSync(`${BASE}/pinned_twochannel_k${K}.json`, "utf8"));
  const mem0 = JSON.parse(readFileSync(`${BASE}/pinned_mem0_k${K}.json`, "utf8"));
  const corr = {};
  for (const cfg of ["cosine", "baf", "a_top", "a_lex", "b_slot"]) {
    corr[cfg] = det.per_sample.map(s => (s[cfg] && s[cfg].category === "y_correct") ? 1 : 0);
  }
  corr.mem0 = mem0.per_sample.map(s => s.e2e_category === "y_correct" ? 1 : 0);
  if (corr.mem0.length !== corr.cosine.length) throw new Error(`n mismatch at m=${m}`);

  report.e2e[m] = {};
  for (const [a, bb] of [["a_top", "cosine"], ["baf", "cosine"], ["mem0", "cosine"], ["mem0", "a_top"]]) {
    const st = pairedStats(corr[a], corr[bb], seed++);
    report.e2e[m][`${a}_minus_${bb}`] = st;
    console.log(`E2E m=${m} ${a}-${bb}: diff ${st.diff_pp.toFixed(1)}pp [${st.ci_lo_pp.toFixed(1)},${st.ci_hi_pp.toFixed(1)}], b=${st.b} c=${st.c}, McNemar p=${st.mcnemar_p.toPrecision(3)}`);
  }
}

const split = JSON.parse(readFileSync(`${BASE}/retrieval_audit_split_2026-07-31.json`, "utf8"));
for (const m of [8]) {
  const ps = split.per_m[m].per_sample;
  report.retrieval[m] = {};
  for (const metric of ["hit1", "hit3"]) {
    for (const [a, bb] of [["baf", "cosine"], ["a_top", "cosine"]]) {
      const x = ps.map(s => s[a][metric]);
      const y = ps.map(s => s[bb][metric]);
      const st = pairedStats(x, y, seed++);
      report.retrieval[m][`${metric}_${a}_minus_${bb}`] = st;
      console.log(`RETR m=${m} ${metric} ${a}-${bb}: diff ${st.diff_pp.toFixed(1)}pp [${st.ci_lo_pp.toFixed(1)},${st.ci_hi_pp.toFixed(1)}], b=${st.b} c=${st.c}, McNemar p=${st.mcnemar_p.toPrecision(3)}`);
    }
  }
}

const ext = JSON.parse(readFileSync(`${BASE}/protocol1_mem0_ext.json`, "utf8"));
const items = ext.per_item;
const allHR = items.filter(it => it.sup_hr === 1.0).length;
const anyHN = items.filter(it => it.sup_hn > 0).length;
const test = items.slice(10);
const testHR = test.filter(it => it.sup_hr === 1.0).length;
const w = (k, n) => { const [lo, hi] = wilson(k, n); return `${k}/${n} [${lo.toFixed(1)},${hi.toFixed(1)}]`; };
report.mem0_marker = {
  items_all_replicates_HR: [allHR, items.length, wilson(allHR, items.length)],
  items_any_HN: [anyHN, items.length, wilson(anyHN, items.length)],
  test_items_all_replicates_HR: [testHR, test.length, wilson(testHR, test.length)],
};
console.log(`MARKER item-level: H_R all-replicates ${w(allHR, items.length)}; H_N any ${w(anyHN, items.length)}; test-split H_R ${w(testHR, test.length)}`);

writeFileSync(OUT, JSON.stringify(report, null, 1));
console.log(`written: ${OUT}`);
