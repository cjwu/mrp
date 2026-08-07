# Stale-narrative generation-side intervention (2026-08-01)

Interventional follow-up to the descriptive stage-bounding decomposition
(MEM0_STALE_DECOMP_2026-08-01.md). The descriptive round observed that 7 of the
8 in-context stale losses at m=8 face a consolidated memory asserting a
superseded value as final, which suggested assertive wording as the driver.
This intervention tests that suggestion causally and refutes the wording part.

## Script

`scripts/stale_narrative_intervention.mjs`. Materials: the 50 pinned m=8
conversations (`exp_tmp/synth_chain_k8_convs.json`, fields Xs and query).
CUR = last chain value (gold), STALE = second-to-last (the one-revision lag
observed in the archived Mem0 narratives). Contexts are five fixed-template
items fed to the pinned answering model (claude-haiku-4-5-20251001,
temperature 0, max_tokens 50), with the prompt template, grading rule
(case-insensitive word-boundary, both/neither -> neither), bootstrap
(B=100,000, mulberry32 seed 20260801), and exact McNemar all copied verbatim
from the pinned pipeline scripts. Integrity checks before any call: pads never
word-boundary-match any chain value; stale templates never match CUR.

Arms (paired over the same 50 conversations, context length fixed at 5):
- a current_only: "User has decided on {CUR}." + 4 neutral pads
- b plain_stale:  a + "User plans to go with {STALE}." (no finality-family word)
- c final_stale:  a + "User initially considered several options, ultimately
  locking in {STALE} as the final choice." (mirrors archived Mem0 idx-0 text)
- d final_only:   the finality narrative without the current value

Output archived: `exp_tmp/stale_narrative_intervention_2026-08-01.json`
(config incl. pad texts, per-arm aggregates, paired stats, all 200 answers).

## Results (n=50 per arm, 200 calls, exit 0)

| arm | y_correct | x_distractor | neither |
|---|---|---|---|
| a current_only | 44 (88%) | 0 | 6 |
| b plain_stale | 29 | 14 (28%) | 7 |
| c final_stale | 31 | 17 (34%) | 2 |
| d final_only | 0 | 50 (100%) | 0 |

Paired contrasts (stale indicator):
- b vs a: +28 pp [+16, +40], discordant 14/0, McNemar p = 1.2e-4
- c vs b: +6 pp [-6, +18], discordant 7/4, McNemar p = 0.55

## Readings

1. **The interference is caused by the mere presence of an unresolved
   superseded commitment in context**: a bare "plans to go with {STALE}"
   sentence, with the current value present and no finality vocabulary, flips
   28% of answers to the superseded value.
2. **Assertive finality wording adds no separable margin at n=50** (+6 pp,
   CI spans zero). The descriptive 7/8 co-occurrence was suggestive but the
   wording is not established as the driver; the earlier paper sentence
   "writes new stale records whose assertive wording the answering model then
   follows" was causally over-specific and is corrected in the same write-back.
3. **d = 50/50 stale**: the answering stage has no parametric defense; it
   reports whatever the context asserts. Consistent with the paper's position
   that validity must be represented and consumed before generation.
4. Calibration ceiling a = 88%: the six misses are all "Not mentioned"
   responses where the query asks what the user "actually purchased" / "ended
   up choosing" and the template only states a decision; a template-query
   pragmatic gap, reported as-is.

## Honest limits

- Constructed five-item contexts with fixed templates and fixed item order;
  one stale value (the second-to-last) per conversation; no paraphrase
  variation. Single run at temperature 0.
- The 88% ceiling means arm differences ride on a slightly imperfect base.
- This intervenes on the generation stage only; the context-absent cell
  (15/23 at m=8) remains bounded, with extraction vs retrieval unseparated.

Paper write-back: §5 intervention sentences replacing the causally
over-specific wording sentence, §9 limitation updated (intervention now run,
template-context caveat), intro bullet 3 clause. Zip: script + archives +
README map row.
