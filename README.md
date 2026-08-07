# MRP Audit Release (anonymized)

Code, data, and archived raw outputs for the paper *Retrieval-Augmented Memory Is
Not Enough: The Monotonic Retrieval Paradox in Conversational Assistants*
(anonymized supplementary).
Every number in the paper traces to a script here and an archived JSON under
`exp_tmp/`.

## Layout

- `scripts/` evaluation and audit scripts (Node ESM + Python)
- `dist/` shared pipeline modules: `embedding.js` / `embedding_worker.js`
  (mxbai-embed-large-v1 via transformers.js, worker-threaded) and `mrp.js`
  (carrier detection and grounded-extension filter)
- `exp_tmp/` the synthetic benchmark, all archived raw outputs, and result notes

Run everything from the repository root. Scripts reference `./exp_tmp/...` for
data and `../dist/...` for module imports.

## Requirements

- Node.js 22+ with `@huggingface/transformers` installed (`npm i @huggingface/transformers`)
- Python 3.11 with `mem0ai==2.0.1` and `qdrant-client` for the Mem0 arms
- API keys only where a script says so: Anthropic (answering model
  `claude-haiku-4-5-20251001`, temperature 0) and OpenAI (Mem0 extraction snapshot
  `gpt-4o-mini-2024-07-18`, temperature 0). Deterministic scripts need no keys.

## Paper-to-artifact map

| Paper item | Script | Archived output |
|---|---|---|
| Synthetic chains, m in {1,2,8} | `scripts/synth_chain_retract.mjs` (generator) | `exp_tmp/synth_chain_k{1,3,8}_convs.json` |
| End-to-end grid + retrieval hit/mention (Tables 3, 4, appendix full table) | `scripts/synth_chain_twochannel.mjs`, `scripts/mem0_chain_baseline.py` | `exp_tmp/pinned_twochannel_k*.json`, `exp_tmp/pinned_mem0_k*.json`, `exp_tmp/PINNED_RERUN_2026-07-27.md` |
| Mention / unresolved split (Table 4 discussion) | `scripts/retrieval_audit_split.mjs` (re-derivation gated against pinned aggregates) | `exp_tmp/retrieval_audit_split_2026-07-31.json` |
| Paired bootstrap + McNemar comparisons | `scripts/paired_stats.mjs` | `exp_tmp/paired_stats_2026-07-31.json` |
| Protocol 1 deterministic arms (Section 4, appendix Table 8) | `scripts/protocol1_audit.mjs` | `exp_tmp/protocol1_audit.json` |
| Scorer generality + deterministic H_C (Section 4) | `scripts/delta_retract_multiscorer.mjs` (two-stage: `STAGE=mxbai` then `STAGE=rest`; stage 1 gated) | `exp_tmp/delta_retract_multiscorer_2026-07-31.json`, `exp_tmp/delta_retract_stage1_2026-07-31.json` |
| Additive characterization calibration (appendix Table 9) | `scripts/bh_calibration.mjs` | `exp_tmp/bh_calibration.json` |
| m=8 stale-answer stage split (Section 5) | `scripts/mem0_stale_decomp.mjs` (gated, no API) | `exp_tmp/mem0_stale_decomp_2026-08-01.json`, `exp_tmp/MEM0_STALE_DECOMP_2026-08-01.md` |
| Generation-side stale-interference intervention (Section 5) | `scripts/stale_narrative_intervention.mjs` (Anthropic key; integrity-checked templates) | `exp_tmp/stale_narrative_intervention_2026-08-01.json`, `exp_tmp/STALE_NARRATIVE_INTERVENTION_2026-08-01.md` |
| Per-revision snapshot arm, composition failure (Section 5) | `scripts/mem0_chain_snapshot.py` (OpenAI + Anthropic keys; segment-wise ingestion, store dumped after every revision) | `exp_tmp/mem0_chain_snapshot_k8_2026-08-02.json`, `exp_tmp/MEM0_CHAIN_SNAPSHOT_2026-08-02.md` |
| Mem0 stochastic arm (Table 1) | `scripts/protocol1_mem0_arm.py` | `exp_tmp/protocol1_mem0.json` |
| Five-dimensional marker analysis (reproducible, gated) | `scripts/protocol1_mem0_ext.py` | `exp_tmp/protocol1_mem0_ext.json` |
| Commitment control H_C | `scripts/protocol1_mem0_control.py` | `exp_tmp/protocol1_mem0_control_2026-07-31.json` |
| Closed-family exactness witness (Section 6) | `scripts/closedworld_exact.py` | `exp_tmp/closedworld_exact.json` |
| Compaction omission case study (Section 3.6) | `scripts/simulate_compact.mjs` | `exp_tmp/hb_chain_format_cf_n10.json`, `exp_tmp/hb_compacted_cf_r400_n10_convs.json`, `exp_tmp/hbcf_mention_counts_2026-07-31.json`, `exp_tmp/HBCF_COMPACTION_ERASURE_2026-07-31.md` |
| Benchmark feature matrix (Table 7) | audit notes | `exp_tmp/BENCHMARK_FEATURE_MATRIX_2026-07-31.md` |
| Protocol and calibration record | notes | `exp_tmp/PROTOCOL1_BH_2026-07-31.md` |

## Verification gates

Four scripts re-derive published numbers and refuse to emit anything unless the
re-derivation matches the archived originals cell by cell:
`scripts/retrieval_audit_split.mjs` (90 aggregate cells against the pinned
deterministic runs, plus the stored Mem0 top-5),
`scripts/protocol1_mem0_ext.py` (tau, RIS, permutation p, and all 30 per-item
values against the archived extended analysis),
`scripts/mem0_stale_decomp.mjs` (hit@5 and all end-to-end category percentages
against the aggregates archived inside the pinned Mem0 run files), and stage
`mxbai` of `scripts/delta_retract_multiscorer.mjs` (per-item cosine scores
against `exp_tmp/protocol1_audit.json` to 1e-6; the extra embedders, the
cross-encoder, and BM25 run in stage `rest`, which requires a passed stage-1
file).

## Notes

- File names keep the historical `k` labels (`k1/k3/k8`); the paper's relation
  depth is m = 1, 2, 8 (the `k3` corpus contains two revisions).
- Served-model outputs changed over time even with pinned decoder settings; the
  archived JSONs are the record the paper cites.
- The ten counterfactual conversations under `exp_tmp/hb_*` derive from the
  HorizonBench corpus (CC-BY-4.0).
- Paths and run directories were anonymized (`REDACTED_RUN_DIR` marks a local
  Mem0 store location).
