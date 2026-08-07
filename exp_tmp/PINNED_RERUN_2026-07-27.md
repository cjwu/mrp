# Pinned re-run, 2026-07-27

Triggered by an external baseline request for a two-channel / state-tracking baseline.
Running it surfaced a reproducibility problem in the previously reported E2E numbers, so the whole
chain grid was re-run under pinned decoder settings.

## Configuration

- Answering model: `claude-haiku-4-5-20251001`, **`temperature: 0`** (was unset, defaulting to 1.0)
- Mem0 extraction: **`gpt-4o-mini-2024-07-18`** (was the moving alias `gpt-4o-mini`)
- n = 50 per cell, top-5 context, MCAP = 50
- Data: `exp_tmp/synth_chain_k{1,3,8}_convs.json` (unchanged since April)
- Raw output: `exp_tmp/pinned_twochannel_k{1,3,8}.json`, `exp_tmp/pinned_mem0_k{1,3,8}.json`
- Script changes: `synth_chain_retract.mjs`, `synth_chain_twochannel.mjs`, `mem0_chain_baseline.py`

## Methods

| id | description |
|---|---|
| `cosine` | control, no supersession channel |
| `baf` | the paper's witness (grounded extension over defeat edges) |
| `a_top` | external's two-channel rule, retrieval time, topical scoping |
| `a_lex` | same, lexical (keyword) scoping |
| `b_slot` | ingest-time event-local slot overwrite, filtered against the tracked value |
| `Mem0` | LLM-extraction ingest memory |

`a_top` and `baf` share one carrier detector and one topical threshold, so their difference
isolates the aggregation rule.

## E2E Y-correct %, Wilson 95% CI

| method | k=1 | k=3 | k=8 |
|---|---|---|---|
| cosine | 48.0 [34.8,61.5] | 28.0 [17.5,41.7] | 10.0 [4.3,21.4] |
| baf | 54.0 [40.4,67.0] | 44.0 [31.2,57.7] | 24.0 [14.3,37.4] |
| **a_top** | 54.0 [40.4,67.0] | 46.0 [33.0,59.6] | **32.0 [20.8,45.8]** |
| a_lex | 54.0 [40.4,67.0] | 36.0 [24.1,49.9] | 20.0 [11.2,33.0] |
| b_slot | 48.0 [34.8,61.5] | 30.0 [19.1,43.8] | 18.0 [9.8,30.8] |
| Mem0 | 100.0 [92.9,100.0] | 80.0 [67.0,88.8] | 36.0 [24.1,49.9] |

## What separates (Newcombe 95% CI excludes zero)

E2E:
- `a_top` - `cosine`, k=8: **+22.0 [+6.0,+36.9]**
- `Mem0` - `a_top`, k=1: **+46.0 [+31.1,+59.6]**; k=3: **+34.0 [+15.2,+49.7]**; k=8: +4.0 [-14.2,+21.8] **crosses 0**

Deterministic retrieval:
- Hit@1 k=8, `baf` and `a_top` vs `cosine`: **+22.0 [+4.2,+38.1]**
- Hit@3 k=8, `baf` and `a_top` vs `cosine`: **+30.0 [+10.4,+46.6]**
- Distractor@5 k=1, both vs `cosine`: **-14.0 [-26.6,-2.6]**
- Distractor@5 k=3, `a_top` vs `cosine`: **-18.0 [-30.8,-7.1]** (`baf` -6.0, crosses 0)
- Distractor@5 k=8, `a_top` vs `cosine`: **-14.0 [-26.2,-4.0]**; `a_top` vs `baf`: **-14.0 [-26.2,-4.0]** (`baf` vs `cosine` = 0.0)

Does not separate at n=50: every `baf` - `cosine` E2E comparison (+6.0 / +16.0 / +14.0), and
every `a_top` - `baf` E2E comparison.

## The crossover

Mem0 dominates the zero-LLM rule by 46 pp at k=1 and 34 pp at k=3, and the two become
indistinguishable at k=8. Degradation from k=1 to k=8: Mem0 -64 pp, cosine -38 pp,
baf -30 pp, a_top -22 pp. The ingest-time advantage erodes with chain length, which is the
qualitative content of the resolution-stability argument, obtained here without the p^k
machinery and with intervals attached.

## Carrier detection ceiling

`detectRetraction` recall against the labelled carriers: 16.0% (k=1), 21.0% (k=3),
15.8% (k=8), precision 1.0. This matches the range already reported in the paper
(1.0 / 0.16-0.21) and caps every carrier-based method, ours included. Extraction never
failed once a carrier fired (`p_eff` = 1.000), so the loss is entirely detection.

## Placement contrast did not separate, for a principled reason

`a_top` and `b_slot` state correctness is identical at every k (16/16, 18/18, 12/12). A
single-slot latest-value overwrite is idempotent in the tail: missing carriers 1..k-1 costs
nothing as long as carrier k is caught. So this design escapes p^k compounding, which is
consistent with the reference class the proposition declares (all-relations-correct), and it
supports the external placement question. The observed a_top vs b_slot retrieval gap comes from scoping
(embedding vs keyword), not from placement.

## Reproduction verdict against the previously reported numbers

| cell | paper | pinned re-run | |
|---|---|---|---|
| cosine k=3 | 30 | 28.0 | reproduces |
| baf k=3 | 48 | 44.0 | reproduces |
| cosine k=8 | 10 | 10.0 | reproduces |
| baf k=8 | 22 | 24.0 | reproduces |
| Mem0 k=1 | 100 | 100.0 | reproduces |
| Mem0 k=3 | 86 | 80.0 | reproduces (close) |
| **cosine k=1** | **74** | **48.0** | **does not** |
| **baf k=1** | **80** | **54.0** | **does not** |
| **Mem0 k=8** | **58** | **36.0** | **does not** |

All deterministic retrieval metrics reproduce exactly.

Causes identified:
1. **k=1 cosine/baf/recency.** The answering model now intermittently returns both the
   superseded and the current value ("Both", "X and Y"); the grader scores that as neither.
   Every `neither` case at k=1 is of this form (verified per-case against the raw JSON on
   2026-07-30: cosine 15/50, baf 12/50, a_top 12/50, a_lex 12/50, b_slot 15/50; an earlier
   draft of this note miscounted 14). Pinning temperature to 0 does not restore
   the April values (48/54 pinned vs 50/52 unpinned), so this is a behaviour shift in the
   served model, not a sampling artifact.
2. **Mem0 k=8.** The `gpt-4o-mini` alias moved between April and July. Pinning the snapshot
   recovers part of it (22.0 unpinned to 36.0 pinned) but not all of the published 58.

## Consequences to carry into any revision

- Report intervals on every proportion; several earlier claims do not survive at n=50.
- The witness's end-to-end advantage over cosine is directional at all three k (+6/+16/+14) and
  significant at none.
- The witness's retrieval advantage at k=8 is real and significant (Hit@1 +22, Hit@3 +30).
- The simpler two-channel rule matches the witness on Hit@K and beats it on distractor
  suppression at k=3 and k=8; at k=8 the witness suppresses no distractors at all.
- Pin decoder settings and model snapshots in every eval script (done).
