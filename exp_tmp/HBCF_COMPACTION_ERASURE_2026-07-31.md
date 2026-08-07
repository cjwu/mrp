# HB-CF compaction erasure check, 2026-07-31

Backing record for the compaction instance added to palm/mrp.tex section 3.6
(approximate observability). Also documents why the old fig_conv5.pdf was retired.

## Why the old figure could not be reused

- `paper/palm/fig_conv5.py` hardcodes per-conversation "preference-topic mention"
  counts (raw 920/426/214/214/152/93/83/68/66/25, compact 1/0/0/0/1/0/0/0/0/0).
- The script that produced those counts lived in /tmp and is gone; no definition
  survives. The hardcoded totals (2261 raw, 2 compact) also disagree with the old
  paper text (1989 -> 1), so the figure and its caption were already mutually
  inconsistent.
- Recounting under the two obvious mechanical definitions does not reproduce them:
  word-boundary mentions of the preference values give totals raw 341 / compact 33
  (occurrences) and 281 / 26 (turns), with per-idx values nowhere near the figure's.
  Value labels are abstract (care_ethics, 0.95, firm_directness) and collide with
  generic words (minimal, high, 2), so no defensible mention-count figure exists.
- Decision: retire the figure; replace with the summary-channel check below, which
  is fully verifiable.

## The verified replacement (what the paper now claims)

Data: `exp_tmp/hb_chain_format_cf_n10.json` (raw) and
`exp_tmp/hb_compacted_cf_r400_n10_convs.json` (compacted, produced by
`scripts/simulate_compact.mjs`: Claude-Code-style compact, model
claude-haiku-4-5-20251001, keeps most recent 400 turns, older span replaced by an
LLM-written summary; hierarchical two-level at these lengths; the compaction prompt
explicitly instructs preserving "changes / retractions / replacements the user has
made along the way").

- 10 HorizonBench counterfactual conversations, 3,263 to 4,336 turns each.
- Summary lengths: 767 to 1,420 characters (`_compact_summary_chars`).
- The paired counterfactual queries ask for the ORIGINAL (superseded) value of one
  preference dimension (X1 in Xs, e.g. idx 38: favored_framework,
  care_ethics -> deontology).
- Check 1 (mechanical, conservative): word-boundary match of X1 and Xfinal
  (underscores as spaces) against each summary text. 0/10 summaries contain either
  value. Collisions would only overcount mentions, so 0/10 is a safe claim.
- Check 2 (mechanical): dimension keyword in summary
  (humor/listening/detail/evidence/framework|ethic/replication/tone/rigor):
  2/10 hit (idx 38 "framework", idx 2 "rigor").
- Check 3 (manual): all ten summaries read in full. The two keyword hits are
  unrelated usages (idx 38: "peer-to-peer intervention frameworks" in the veteran
  crisis project; idx 2: "analytical rigor" of the assistant's market analysis).
  No paraphrase-level survival of the preference values found in any summary.
- Reading: content older than the kept 400-turn tail survives only through the
  summary; for these queries the compacted observation carries no evidence of the
  queried value pair, an instance of the Delta_A -> 0 endpoint of the observability
  remark, produced by a routine pipeline stage before any detection layer runs.

Raw per-conversation counts (value-mention recount, both definitions):
`exp_tmp/hbcf_mention_counts_2026-07-31.json`.

Caution for reuse: the April-era E2E numbers on these compacted conversations
(`exp_tmp/baf_hb_cf_compact_r400_n10.json`, all systems 0/10) predate decoder
pinning and are NOT cited in the paper; only the deterministic text-level checks
above are.
