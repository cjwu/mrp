// Stage-bounding decomposition of Mem0's stale (x_distractor) answers against
// the archived retrieval contexts of the pinned chain runs.
//
// Reads:  ./exp_tmp/pinned_mem0_k{1,3,8}.json  (per_sample: Xs chain, retrieved_top5
//         memory texts, e2e_answer, e2e_category from the pinned runs)
// Writes: /tmp/mem0_stale_decomp.json
//
// Verification gate: before emitting anything, the script recomputes from
// per_sample (a) current-value hit@5 and (b) the three end-to-end category
// percentages, and requires exact equality with the aggregates archived in the
// same files (retrieval_hit_pct["5"], e2e_llm). The value matcher is the same
// case-insensitive word-boundary rule used for end-to-end grading.
//
// Decomposition per stale-answer conversation:
//   context-absent : current (gold) value does not appear in any retrieved
//                    top-5 text -> the failure precedes context assembly
//                    (store or retrieval; the top-5 archive cannot separate
//                    extraction from retrieval within this cell)
//   in-context     : current value present in top-5 -> the stale value won
//                    downstream of context assembly
// Additionally flagged, with a fixed finality word family (final, ultimately,
// locking in, locked in, settled on, decided on, final choice, final decision):
//   wrong-final narrative : some top-5 text matches the finality family, names
//                           a superseded chain value, and does not name the
//                           current value
//   answer-in-finality    : the served stale answer itself is named inside a
//                           finality-matching top-5 text
// The split is descriptive: it bounds the failure stage from archived contexts
// and is not an intervention.

import fs from "node:fs";

const FILES = { 1: "./exp_tmp/pinned_mem0_k1.json", 2: "./exp_tmp/pinned_mem0_k3.json", 8: "./exp_tmp/pinned_mem0_k8.json" };
const FINALITY_RE = /(final|ultimately|locking in|locked in|settled on|decided on|final choice|final decision)/i;

const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const wb = (hay, needle) => new RegExp(`\\b${esc(needle)}\\b`, "i").test(hay);

const out = {};
for (const [m, file] of Object.entries(FILES)) {
  const d = JSON.parse(fs.readFileSync(file, "utf8"));
  const n = d.per_sample.length;

  let hit5 = 0;
  const cat = { y_correct: 0, x_distractor: 0, neither: 0 };
  for (const s of d.per_sample) {
    const cur = s.Xs[s.Xs.length - 1];
    if (wb(s.retrieved_top5.join(" \n "), cur)) hit5++;
    cat[s.e2e_category]++;
  }
  const hit5Pct = (100 * hit5) / n;
  if (hit5Pct !== d.retrieval_hit_pct["5"]) {
    console.error(`GATE FAILED m=${m}: recomputed hit@5 ${hit5Pct} != archived ${d.retrieval_hit_pct["5"]}`);
    process.exit(1);
  }
  for (const c of Object.keys(cat)) {
    const pct = (100 * cat[c]) / n;
    if (pct !== d.e2e_llm[c]) {
      console.error(`GATE FAILED m=${m}: recomputed ${c} ${pct}% != archived ${d.e2e_llm[c]}%`);
      process.exit(1);
    }
  }

  const items = [];
  const agg = { n, stale: 0, context_absent: 0, in_context: 0, context_absent_wrong_final: 0, in_context_wrong_final: 0, answer_in_finality: 0 };
  for (const s of d.per_sample) {
    if (s.e2e_category !== "x_distractor") continue;
    const cur = s.Xs[s.Xs.length - 1];
    const curIn = wb(s.retrieved_top5.join(" \n "), cur);
    const wrongFinal = s.retrieved_top5.some(t => FINALITY_RE.test(t) && !wb(t, cur) && s.Xs.slice(0, -1).some(x => wb(t, x)));
    const ansInFinal = s.retrieved_top5.some(t => FINALITY_RE.test(t) && wb(t, s.e2e_answer));
    agg.stale++;
    if (curIn) { agg.in_context++; if (wrongFinal) agg.in_context_wrong_final++; }
    else { agg.context_absent++; if (wrongFinal) agg.context_absent_wrong_final++; }
    if (ansInFinal) agg.answer_in_finality++;
    items.push({ idx: s.idx, current: cur, answer: s.e2e_answer, current_in_top5: curIn, wrong_final_narrative: wrongFinal, answer_in_finality: ansInFinal });
  }
  out[`m${m}`] = { gate: { hit5_pct: hit5Pct, e2e: { ...d.e2e_llm } }, agg, items };
  console.log(`m=${m}: gate passed (hit@5 ${hit5Pct}%, e2e ${cat.y_correct}/${cat.x_distractor}/${cat.neither}); stale ${agg.stale}: context-absent ${agg.context_absent} (wrong-final ${agg.context_absent_wrong_final}), in-context ${agg.in_context} (wrong-final ${agg.in_context_wrong_final}), answer-in-finality ${agg.answer_in_finality}`);
}

fs.writeFileSync("/tmp/mem0_stale_decomp.json", JSON.stringify(out, null, 2));
console.log("written: /tmp/mem0_stale_decomp.json");
