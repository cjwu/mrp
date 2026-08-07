// Interventional ablation for the m=8 stale cell: does an assertive stale
// narrative causally interfere with the answering stage when the current value
// is present in context?
//
// The descriptive decomposition (scripts/mem0_stale_decomp.mjs) bounded the
// pinned m=8 stale answers into 15 context-absent / 8 in-context cases and
// observed that 7 of the 8 in-context losses face a consolidated memory
// asserting a superseded value as final. This script turns that observation
// into an intervention: constructed five-item contexts are fed to the same
// pinned answering model, varying only the wording of the stale item.
//
// Materials: the 50 pinned m=8 conversations (exp_tmp/synth_chain_k8_convs.json,
// fields Xs and query). CUR = last chain value (gold), STALE = second-to-last
// (one revision behind, the lag observed in the archived Mem0 narratives).
//
// Arms (paired over the same 50 conversations; context length fixed at 5):
//   a  current_only : the current-value memory plus four neutral pads
//   b  plain_stale  : current-value memory + "User plans to go with {STALE}."
//                     (bare commitment wording, no finality-family word)
//   c  final_stale  : current-value memory + "User initially considered several
//                     options, ultimately locking in {STALE} as the final
//                     choice." (mirrors the archived Mem0 narrative, idx 0)
//   d  final_only   : the finality narrative without the current value
// Primary contrast: stale-answer rate (c) minus (b) = causal interference of
// assertive wording given the current value present. Secondary: (b) minus (a),
// and (d) as a dominance check. Arm (a) is the calibration ceiling.
//
// Answering call, prompt template, model pin (claude-haiku-4-5-20251001,
// temperature 0, max_tokens 50), and grading (case-insensitive word-boundary
// match against current and all superseded values; both/neither -> neither)
// are copied verbatim from scripts/synth_chain_twochannel.mjs. Paired stats
// (percentile bootstrap B=100,000, mulberry32 seed 20260801; exact two-sided
// McNemar) copied from scripts/paired_stats.mjs.
//
// Integrity checks before any API call: pad texts must not word-boundary-match
// any chain value of any conversation, and the two stale templates must not
// match the current value.
//
// Run:    node scripts/stale_narrative_intervention.mjs
// Writes: /tmp/stale_narrative_intervention.json

import { readFileSync, writeFileSync } from "fs";

const KEY_FILE = `${process.env.HOME}/.api_keys/anthropic_key.env`;
let ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
  const txt = readFileSync(KEY_FILE, "utf8");
  for (const line of txt.split("\n")) {
    const m = line.match(/^export\s+ANTHROPIC_API_KEY=["']?(.+?)["']?$/);
    if (m) { ANTHROPIC_API_KEY = m[1]; break; }
  }
}

const MODEL = "claude-haiku-4-5-20251001";
const CONVS_PATH = process.env.CONVS_IN ?? "./exp_tmp/synth_chain_k8_convs.json";
const OUT_PATH = process.env.OUT ?? "/tmp/stale_narrative_intervention.json";
const B = 100000;
const SEED = 20260801;

const PADS = [
  "User prefers morning meetings over afternoon ones.",
  "User subscribes to a weekly science newsletter.",
  "User enjoys cooking at home on weekends.",
  "User keeps a running list of podcast recommendations.",
];

const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const wbContains = (text, term) => new RegExp(`\\b${escapeRe(term)}\\b`, "i").test(text);
const sleep = ms => new Promise(r => setTimeout(r, ms));

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
  return { n, b, c, diff_pp: 100 * diff, ci_lo_pp: 100 * diffs[Math.floor(0.025 * B)], ci_hi_pp: 100 * diffs[Math.ceil(0.975 * B) - 1], mcnemar_p: binomTail2sided(b, c) };
}

async function ask(retrieved, query) {
  const ctxBlock = retrieved.map((c, i) => `[${i + 1}] ${c}`).join("\n");
  const prompt = `Below is retrieved context from a conversation. Answer the question using ONLY this context. Respond in one short word/phrase, just the answer with no explanation.\n\nContext:\n${ctxBlock}\n\nQuestion: ${query}\n\nAnswer:`;
  for (let attempt = 0; attempt < 5; attempt++) {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({ model: MODEL, max_tokens: 50, temperature: 0, messages: [{ role: "user", content: prompt }] }),
    });
    if (resp.ok) return (await resp.json()).content[0].text.trim();
    if (resp.status === 429) { await sleep(5000 * (attempt + 1)); continue; }
    throw new Error(`API ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
  }
  throw new Error("API 429 exhausted retries");
}

const items = JSON.parse(readFileSync(CONVS_PATH, "utf8"));
const ARMS = ["a_current_only", "b_plain_stale", "c_final_stale", "d_final_only"];

function contexts(cur, stale) {
  const curMem = `User has decided on ${cur}.`;
  const plainStale = `User plans to go with ${stale}.`;
  const finalStale = `User initially considered several options, ultimately locking in ${stale} as the final choice.`;
  return {
    a_current_only: [curMem, PADS[0], PADS[1], PADS[2], PADS[3]],
    b_plain_stale: [curMem, plainStale, PADS[0], PADS[1], PADS[2]],
    c_final_stale: [curMem, finalStale, PADS[0], PADS[1], PADS[2]],
    d_final_only: [finalStale, PADS[0], PADS[1], PADS[2], PADS[3]],
  };
}

for (const it of items) {
  for (const v of it.Xs) {
    if (PADS.some(p => wbContains(p, v))) { console.error(`INTEGRITY FAILED: pad collides with chain value "${v}"`); process.exit(1); }
  }
  const cur = it.Xs[it.Xs.length - 1], stale = it.Xs[it.Xs.length - 2];
  if (wbContains(`User plans to go with ${stale}.`, cur) || wbContains(`ultimately locking in ${stale} as the final choice`, cur)) {
    console.error(`INTEGRITY FAILED: stale template matches current value, idx ${it.idx}`); process.exit(1);
  }
}
console.log(`integrity checks passed (${items.length} conversations, ${ARMS.length} arms, ${items.length * ARMS.length} calls)`);

const perSample = [];
const agg = Object.fromEntries(ARMS.map(a => [a, { y_correct: 0, x_distractor: 0, neither: 0 }]));
for (let i = 0; i < items.length; i++) {
  const it = items[i];
  const cur = it.Xs[it.Xs.length - 1], stale = it.Xs[it.Xs.length - 2];
  const prior = it.Xs.slice(0, -1);
  const ctxs = contexts(cur, stale);
  const sample = { idx: it.idx, current: cur, stale_used: stale, query: it.query };
  for (const arm of ARMS) {
    const ans = await ask(ctxs[arm], it.query);
    const yMatch = wbContains(ans, cur);
    const xMatch = prior.some(x => wbContains(ans, x));
    const cat = yMatch && !xMatch ? "y_correct" : !yMatch && xMatch ? "x_distractor" : "neither";
    agg[arm][cat]++;
    sample[arm] = { answer: ans, category: cat };
  }
  perSample.push(sample);
  if ((i + 1) % 10 === 0) console.log(`  [${i + 1}/${items.length}] ` + ARMS.map(a => `${a.split("_")[0]}:y=${agg[a].y_correct},x=${agg[a].x_distractor}`).join("  "));
}

const staleVec = arm => perSample.map(s => (s[arm].category === "x_distractor" ? 1 : 0));
const stats = {
  c_vs_b_stale: pairedStats(staleVec("c_final_stale"), staleVec("b_plain_stale"), SEED),
  b_vs_a_stale: pairedStats(staleVec("b_plain_stale"), staleVec("a_current_only"), SEED + 1),
  d_stale_rate: agg.d_final_only.x_distractor / items.length,
  a_correct_rate: agg.a_current_only.y_correct / items.length,
};

const report = { config: { n: items.length, model: MODEL, temperature: 0, seed: SEED, convs: CONVS_PATH, pads: PADS, arms: ARMS }, agg, stats, per_sample: perSample };
writeFileSync(OUT_PATH, JSON.stringify(report, null, 2));
console.log("aggregates:", JSON.stringify(agg));
console.log("stats:", JSON.stringify(stats));
console.log(`written: ${OUT_PATH}`);
