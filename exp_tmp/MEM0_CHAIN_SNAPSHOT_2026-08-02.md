# Per-revision snapshot instrumented arm, m=8 (2026-08-02)

Response to the critique that "local updates do not compose" was asserted
without verifying each local update. This arm measures it: only conversations
in which every local update is written and marked, yet the final answer is
still wrong, count as composition failures.

## Script

`scripts/mem0_chain_snapshot.py` (pyenv 3.11.9, mem0ai 2.0.1, extraction
gpt-4o-mini-2024-07-18@t0, per-run qdrant, answering
claude-haiku-4-5-20251001@t0, pinned prompt + word-boundary grading).
Materials: the pinned `exp_tmp/synth_chain_k8_convs.json` (50 conversations).
Segmentation: boundary = first user turn word-boundary-matching each chain
value (verified strictly increasing, 50/50). One `m.add` per segment (9 calls
per conversation); store dumped via `get_all` after each segment.

Layers per step r (transition X_{r-1} -> X_r):
- L2 incoming_present: X_r appears anywhere in the store
- L3 step_marker: one memory names both X_{r-1} and X_r, or names X_r beside a
  change-marker word (change/switch/instead/no longer/update); mirrors the
  protocol-1 supersession-marker definition
- L4 final current@5; L5 final answer category (pinned prompt)

Output archived: `exp_tmp/mem0_chain_snapshot_k8_2026-08-02.json` (per-step
matrices + full store dumps per snapshot + retrieved top-5 + answers).

## Results (n=50, single run, exit 0)

- **L2: 400/400 (100% at every depth 1-8).** Update writing never failed.
- All-steps written AND marked (L2+L3 for all 8 steps): **37/50** conversations.
- Outcomes of those 37: **2 y_correct / 24 x_distractor / 11 neither.**
  Conditional on perfect local updates, correctness = 2/37.
- Overall e2e: 3 / 32 / 15; current@5 = 21/50 (42%).
- 7 of the 24 pure-composition stale answers had the current value in top-5;
  17 did not (stored but not retrieved: the compositional loss includes the
  retrieval cut of an unordered record set).
- Neither answers are conflict-visible: "Cannot determine from context",
  "All of them." (model sees multiple locally final records and abstains).
- wrong-final assertions: constant 10 conversations at every depth.
- Final store sizes 9 to 20 memories.

## Mechanism (visible in dumps)

Store records are pairwise and each asserts its own locally final value:
"User changed their travel plans from Bangkok to London and has confirmed the
trip to London." Every edge of the chain is present; no record carries global
order or a chain-level current state. Top-5 returns an unordered subset;
generation picks one asserted final or declines.

## Batching contrast (deployment-relevant observation)

Pinned run ingested 50-turn batches (2 add calls): extraction produced
consolidated narratives ("ultimately locking in Reykjavik as the final
choice"), e2e 36% correct / 46% stale. Segment-wise ingestion (9 add calls):
pairwise change records, e2e 6% correct / 64% stale / 30% neither. Extraction
output structure depends on ingestion granularity; both structures fail at
m=8. The arms are therefore reported side by side, never merged.

## Honest limits

- Single run; no replicates. Ingestion granularity differs from the pinned
  run by design, so rates are not comparable cell-to-cell with the pinned 46%.
- L3 marker is string-level (fixed word families); store dumps ship for
  re-analysis.
- The 11 neither outcomes are abstentions under conflict, a different failure
  surface than stale serving; both counted separately.

Paper write-back: abstract sentence upgrade (composition now measured), §5
\paragraph{Per-revision snapshots.}, Limitations caveat, conclusion clause,
intro bullet 3 clause. Zip: script + JSON + this note + README row.
