# Mem0 stale-answer stage-bounding decomposition (2026-08-01)

Motivation: the paper conceded that the m=8 stale-serving cell (46%) was "not
isolated across pipeline stages" (extraction / representation / retrieval /
generation). The pinned run archives contain, per conversation, the retrieved
top-5 memory texts, the served answer, and the grading category, which is
enough to bound the failure stage descriptively with zero new runs.

## Script

`scripts/mem0_stale_decomp.mjs` (Node, no API, no embeddings). Reads
`exp_tmp/pinned_mem0_k{1,3,8}.json`. Verification gate: recomputes hit@5
(current value in top-5, case-insensitive word-boundary match, same rule as
end-to-end grading) and the three e2e category percentages from `per_sample`,
requires exact equality with the aggregates archived in the same files
(`retrieval_hit_pct["5"]`, `e2e_llm`), refuses to emit otherwise. All three
gates passed (hit@5 100/100/66%; e2e 50-0-0, 40-7-3, 18-23-9 counts).

Output archived: `exp_tmp/mem0_stale_decomp_2026-08-01.json` (aggregates +
per-item records: idx, current value, served answer, current_in_top5,
wrong_final_narrative, answer_in_finality).

## Definitions

- **context-absent**: current (gold, last-in-chain) value appears in no
  retrieved top-5 text. Failure precedes context assembly (store or retrieval;
  the top-5 archive cannot separate extraction from retrieval inside this cell
  because the full store was not archived).
- **in-context**: current value present in top-5; the stale value won
  downstream of context assembly.
- **wrong-final narrative**: some top-5 text matches the fixed finality word
  family `(final|ultimately|locking in|locked in|settled on|decided on|final
  choice|final decision)`, names a superseded chain value, and does not name
  the current value.
- **answer-in-finality**: the served stale answer is itself named inside a
  finality-matching top-5 text.

## Results

| m | stale answers | context-absent | in-context | wrong-final in ctx-absent | wrong-final in in-context | answer-in-finality |
|---|---|---|---|---|---|---|
| 1 | 0/50 | 0 | 0 | 0 | 0 | 0 |
| 2 | 7/50 | 0 | 7 | 0 | 3 | 3 |
| 8 | 23/50 | 15 | 8 | 8 | 7 | 15 |

Readings (descriptive, no intervention):

1. At m=8 the 46% stale cell splits 15 context-absent / 8 in-context: 65% of
   stale answers could not have been answered correctly from the retrieved
   context, 35% lost with the current value present.
2. 15/23 served stale answers are named inside a consolidated Mem0 memory that
   asserts a superseded value as final (e.g. sample idx 0: "ultimately locking
   in Reykjavik as the final choice" while gold current = Barcelona, one
   revision later). 7 of the 8 in-context losses face such a narrative.
   Ingest-time rewriting lags the chain and writes new stale records with
   assertive wording that the answering model follows.
3. At m=2 all 7 stale answers are in-context losses: shallow-depth failures are
   resolution failures, depth shifts the mass upstream.

## Honest limits

- Descriptive cross-tabulation over archived contexts; bounds the stage, does
  not intervene. A controlled-context generation test (feeding constructed
  contexts to the pinned answering model) would be the interventional follow-up
  and needs API calls; not run.
- The finality word family is fixed and ships with the release; borderline
  narrative wordings can fall on either side of it.
- Extraction vs retrieval cannot be separated inside the context-absent cell
  (top-5 only was archived, not the full store).

Paper write-back: §5 decomposition sentences after the Table `tab:decomp`
discussion, tab:context Mem0 diagnostics cell, §9 limitation sentence upgraded,
intro bullet 3 clause. Zip: script + archived JSON + README map/gates rows.
