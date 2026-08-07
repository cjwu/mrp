# Protocol 1 audit (deterministic arms) + B/H calibration, 2026-07-31

Scripts: `scripts/protocol1_audit.mjs`, `scripts/bh_calibration.mjs` (run with
`~/tools/node-v22-arm64/bin/node`; the x64 node at /usr/local/bin fails on sharp/arm64).
Raw output: `exp_tmp/protocol1_audit.json`, `exp_tmp/bh_calibration.json`.
Everything below is deterministic: pipeline embedder, detectRetraction, BAF grounded
extension, a_top cutoff. Zero LLM calls.

## Protocol 1 design

Base = 50 k1 conversations truncated to the X-support prefix (turns 0..6, target
candidate c0 = the "Locked in" commitment at turn 2). Conditions attach one turn:
H_N = "My colleague mentioned {Y} the other day." (topical, non-retracting);
H_R = one of 5 corpus retraction paraphrases. Both directions (X->Y, Y->X).
n = 50 x 2 x 5 = 500 paired items. Query and c0 record held fixed.

## Results

| stage | H_R vs H_N changed | note |
|---|---|---|
| s(c0) score | 0 beyond numeric floor | max abs delta = 1.4e-7 (embed batch numerics) |
| rank(c0) | 45/500 | displacement -1: 12, +1: 33; score unchanged |
| Top-5 inclusion | 0/500 | |
| BAF eligibility | 100/500 | flips iff detector fired (100/100, 0 without fire) |
| a_top eligibility | 100/500 | identical conditional structure |

- H_N false positives: 0/100 (precision on the neutral arm holds).
- Detector coverage 100/500 = 20.0%, inside the pinned 16-21% recall band, and it is
  **family-structured**: "Actually, let's switch to {Y} instead." fired 100/100; the
  other four families ("Wait, change of plan:", "No, let's do {Y} instead.",
  "Actually scratch that,", "Forget that, going with") fired 0/100 each.

## Readings

1. Proposition 1 verified behaviorally: pointwise score invariant to 1.4e-7 across all
   500 pairs on a real embedder.
2. Score/rank separation is real: 45 rank moves with zero score change (pool
   composition effect from the added turn).
3. BAF and a_top retraction response is exactly the detection channel: eligibility
   flip iff carrier fired, both directions. RIS_eligibility = detector coverage.
4. The 16-21% recall is not uniform noise: coverage concentrates on one paraphrase
   family. Channel-coverage failure, not aggregation failure.

## B/H calibration (Corollary 1)

Per conversation: c- = superseded "Locked in" commitment, c+ = final one.
b(c,q) = pipeline cosine; h+(v) = turns mentioning v excluding gold carriers.
db = b(c+)-b(c-); dh = h(c-)-h(c+).

| m | db mean [p5, p95] | dh (degenerate) | frac dh>0 | pure-relevance inversion (lambda=0) |
|---|---|---|---|---|
| 1 | -0.000 [-0.089, +0.134] | -1 (all 50) | 0% | 52% |
| 2 | +0.005 [-0.068, +0.072] | -2 (all 50) | 0% | 42% |
| 8 | +0.003 [-0.063, +0.074] | -2 (all 50) | 0% | 52% |

Lambda curve (fraction with lambda*dh > db, i.e. predicted strict inversion for an
additive scorer s = b + lambda*h+): starts at the lambda=0 inversion rate and falls to
~0 by lambda=0.05-0.1 at every depth, because dh < 0 here.

## Verdict per the pre-registered rule

H_hat_p5 < 0 at every depth: no independent bound H >= 0 exists on this corpus, so
**Corollary 1 is demoted to an additive characterization** for the PALM version
(decision rule stated in `paper/mrp_theorem_revision_notes_v2.md` §5).

What the calibration establishes instead:

- The synthetic chains are **relevance-parity corpora**: db is centred on zero with
  spread ~0.07, and the superseded commitment outranks its replacement on pure cosine
  in 42-52% of conversations. The observed superseded-item leakage (98% at m=1) rides
  on this parity plus carrier invisibility; it does not need the h-dominance channel.
- The template gives the final value 1-2 extra confirmation turns (dh = -1/-2
  degenerate), so positive-history dominance cannot be measured on this corpus by
  construction. Measuring it requires natural conversations where old commitments are
  reinforced over time. This is a third concrete consequence of the missing
  retraction-labelled corpus, alongside untrained retrievers and unevaluable MRP-Soft.

## Mem0 stochastic arm (run 2026-07-31)

Script: `scripts/protocol1_mem0_arm.py` (pyenv 3.11.9, mem0 2.0.1, extraction snapshot
`gpt-4o-mini-2024-07-18`, temperature 0). 30 base histories (15 fwd / 15 rev,
paraphrase rotating so each of the 5 families gets 6 items) x {H_N, H_R} x R=3
replicates = 180 ingests, each into a fresh user_id. Items 1-10 = held-out
calibration suite (sets tau = P95 of within-condition pair distances), items 11-30 =
test suite. Raw: `exp_tmp/protocol1_mem0.json`, `exp_tmp/protocol1_mem0_ext.json`.
One incident: the first pass crashed at get_all (mem0 2.0.1 wants `filters=`), all
180 ingests survived in the run dir and were re-read without re-ingesting.

Observation vector per run, computed on the stored-memory state and search top-1:

- 4-dim entity presence: (C_in_store, I_in_store, C_top1, I_top1), C = committed value
- 5-dim adds a supersession marker: one stored memory naming both values, or a
  lexical change marker (changed / instead / switched / no longer / updated) next to
  either value. Added after inspecting calibration outputs (a stored text reading
  "changed their decision to use Remix instead of Angular" scored as no-response
  under the 4-dim vector); tau was recalibrated on the calibration suite and results
  reported on the untouched test suite.

| observable | tau | RIS_test | RIS_null | mean d_RN | mean d_null | perm p (one-sided) |
|---|---|---|---|---|---|---|
| 4-dim entity presence | 1 | 0.05 | 0.00 | 0.63 | 0.17 | 6.0e-4 |
| 5-dim + supersession marker | 1 | **0.70** | 0.00 | 1.63 | 0.17 | 5.0e-5 |

Supersession-recorded rate in the stored state, all 30 items:

| paraphrase family | H_R | H_N |
|---|---|---|
| all five, individually | 1.00 | 0.00 |

## Cross-system coverage comparison (the audit's head-line table)

| system | retraction channel | family coverage | conditional structure |
|---|---|---|---|
| cosine | none | 0/5 | score invariant (1.4e-7), rank moves 45/500 |
| BAF / a_top (regex detector) | query-time defeat / cutoff | 1/5 (20% of H_R items) | eligibility flip iff detector fired, 100/100 vs 0 |
| Mem0 (LLM extractor) | ingest-time narrative rewrite | 5/5 (100% of H_R runs) | supersession recorded in stored state; 0% false positives on H_N |

Readings:

1. Mem0's ingest extractor is fully retraction-sensitive **per event** on these short
   prefixes, in narrative form ("planning a trip to Paris, which was later changed to
   Lisbon"). No contradiction with its 36% end-to-end at m=8: the audit measures
   single-event sensitivity, the chain grid measures accumulation over 8 events plus
   retrieval and generation. Consistent with placing holistic extractors outside both
   the retraction-blind class and the independent p^m reference model.
2. The 4-dim vs 5-dim gap is a measurement lesson the protocol should keep: entity
   presence alone understates ingest-time sensitivity (RIS 0.05 vs 0.70); the stored
   state must be read at the level of recorded relations, matching v2 section 12's
   m_A guidance.
3. Within-condition replicate distance is nonzero (mean 0.17) at temperature 0, so
   the replicate arms and the noise-floor comparison are load-bearing, not ceremony.

## Closed-world exactness demonstration (Proposition 3 witness, run 2026-07-31)

Script: `scripts/closedworld_exact.py`. Raw: `exp_tmp/closedworld_exact.json`.
Zero embeddings, zero LLM calls, runs in seconds.

Claim: when the carrier family is closed (all five synthetic templates known), the
fully deterministic pipeline closed-family detection -> template-parse extraction ->
single-slot overwrite -> deterministic readout achieves 100% current-state
correctness at every depth.

| m | correct | carrier-count mismatches | parse failures |
|---|---|---|---|
| 1 | 50/50 = 100.0% | 0 | 0 |
| 2 | 50/50 = 100.0% | 0 | 0 |
| 8 | 50/50 = 100.0% | 0 | 0 |

Checks enforced, not assumed: detected-carrier count equals gold m in every
conversation; every carrier turn parsed by exactly one template parser; readout
compared to Xfinal exactly.

Reading. This is the conditional-achievability half of the argument: with detection
solved (here by fiat, the family is closed), the rest of the pipeline is a state
machine and delivers exactness at every chain depth, including m=8 where cosine
scores 10%, the two-channel rule 32%, and Mem0 36%. The gap between this 100% and
those numbers is carried entirely by the open-language detection layer plus
free-form generation. The impossibility half (utterance-level identifiability:
same surface text, different intent, opposite validity; MRP-Soft with no carrier;
silent change) is what blocks extending this guarantee to open language, and
deciding which side of that line an utterance falls on is a labelled-corpus
question. Label this an oracle-family demonstration in the paper; it is not a
competitive method.

## Review-fix round (2026-07-31, after external Weak Reject review)

1. **Retrieval metric split** (`scripts/retrieval_audit_split.mjs`, raw
   `exp_tmp/retrieval_audit_split_2026-07-31.json`). The published "superseded-item
   leakage@5" was mention-based (deterministic: pre-carrier turn containing a prior
   value; Mem0: memory text containing a prior value). Renamed to superseded-value
   mention@5 and split by within-item resolution (item also names current value or a
   change-marker family: change/switch/instead/no longer/update). Verification gates:
   deterministic re-derivation reproduces all 90 pinned aggregate cells; Mem0
   stored-top5 reproduces hit/dist @5. Result: raw-turn systems mention ~= unresolved
   (widest gap baf m=8: 100 vs 84); Mem0 unresolved = 10 / 42 / 88 % at m=1/2/8
   while mention stays 96-100%. The unresolved gradient rises with the E2E stale rate
   (0/14/46%).
2. **Paired statistics** (`scripts/paired_stats.mjs`, raw
   `exp_tmp/paired_stats_2026-07-31.json`): seeded percentile bootstrap over
   conversations (B=1e5) + exact two-sided McNemar, replacing independent-sample
   Newcombe. Key: a_top-cosine m=8 +22.0 [+8.0,+36.0] b=13 c=2 p=0.0074; m=2 +18.0
   [+6.0,+30.0] p=0.012; baf-cosine m=2 +16.0 [+4.0,+28.0] p=0.022 (newly separates
   under paired analysis; m=1/m=8 do not); mem0-a_top m=8 +4.0 [-14.0,+22.0] p=0.83;
   retrieval hit1/hit3 m=8 +22.0 [+12.0,+34.0] p=0.00098 / +30.0 [+18.0,+42.0]
   p=6.1e-5.
3. **Cluster-safe marker stats**: item is the sampling unit. H_R 30/30 items
   all-replicates [88.6,100], H_N 0/30 [0,11.4], test split 20/20 [83.9,100].
4. **Commitment control arm H_C** (`scripts/protocol1_mem0_control.py`, raw
   `exp_tmp/protocol1_mem0_control_2026-07-31.json`, run dir
   `~/.mem0_runs/protocol1_ctrl_34947_1785480109`): prefix + "Let's go with {Y}."
   (bare new commitment, no retraction marking), 30 items x R=3, same pinned config.
   Marker: 58/90 runs, 19/30 items all-replicates [45.5,78.1], 20/30 any, one item
   1/3. Reading: the extractor records supersession from commitment semantics alone
   in ~2/3 of items; explicit wording accounts for the remainder; H_R vs H_N measures
   the combined effect.
5. **Reproducible 5-dim analysis** (`scripts/protocol1_mem0_ext.py`): recomputes the
   extended analysis from archived protocol1_mem0.json; verified to reproduce
   protocol1_mem0_ext.json exactly (tau=1, RIS 0.70/0.00, perm p, all 30 per-item
   values).

## Status

- Deterministic arms: done.
- Mem0 stochastic arm: done; commitment control H_C done.
- B/H calibration: done; Corollary 1 demoted per the pre-registered rule.
- Closed-world exactness (Prop 3 witness): done, 100% at every depth.
- Review-fix round: metric split, paired stats, cluster stats, H_C, 5-dim repro all done.

## Correction (2026-07-31, delta-retract round)

Line "H_R = one of 5 corpus retraction paraphrases. Both directions (X->Y, Y->X)." above
is inaccurate as a description of the archived artifact. In protocol1_audit.mjs the rev
branch computes wbReplace(content, X, xb) with xb = X, an identity map: the rev variant
keeps the X-committing prefix and its continuation turns name the already-committed X
(a same-value marking control), while only the fwd variant names the incoming Y. The
header comment of that script claims X/Y mirroring; the code does not (its xa variable
is never used), and the archived protocol1_audit.json is the code's output. Paper wording
updated to "100 base units ... two variants over the same X-committing prefix"
(mrp.tex Sections 4/9 and Appendix C). Score-invariance and eligibility-flip claims are
unaffected (both hold per comparison regardless of the named value); a true mirrored
re-run would change pinned rank-move counts and is deliberately not done before the
PALM deadline. delta_retract_multiscorer.mjs reproduces the archived construction
verbatim, quirk included, so its gate and its H_C arm stay comparable to the archive.

## Delta-retract multi-scorer results (2026-07-31)

Script: scripts/delta_retract_multiscorer.mjs (two-stage; stage mxbai gate PASSED,
reproducing archived protocol1_audit.json s0 per item to 1e-6). Archived:
exp_tmp/delta_retract_multiscorer_2026-07-31.json, exp_tmp/delta_retract_stage1_2026-07-31.json.

Conditions per unit: H_0 / H_N (neutral) / H_C ("Let's go with {Y}.") / H_R (5 paraphrases),
500 rows. Scorers: cos_mxbai, cos_minilm (all-MiniLM-L6-v2), cos_bge (bge-small-en-v1.5),
ce_msmarco (ms-marco-MiniLM-L-6-v2 logit), bm25 (Okapi k1=1.5 b=0.75, per-history pool).

- Four pointwise scorers: s0 bit-identical across all four conditions in all 500 rows
  (verified row by row; identical-by-construction of (c, q) inputs). Rank moves via pool
  composition only: RN rank changes mxbai 45 (matches archived 45/500), minilm 72, bge 5,
  ce 84; RC rank changes 82/82/132/109. top5 changes: 0 everywhere.
- BM25 (only a-priori-open cell): d_RC = s(HR)-s(HC): min -0.0319, max +0.0611,
  mean +0.0068 (neg 100 / zero 200 / pos 200 of 500). d_RN: min -0.0930, max +0.3583,
  mean +0.0382. d_CN: min -0.0611, max +0.3146. s0(HN) range 0.409 to 2.231.
  top5 changes 0/500. Largest support DECREASE from adding retraction marking vs bare
  commitment: 0.032; vs neutral: 0.093. Mean shift positive (longer added turn raises
  avgdl, mechanically raising c0's length-normalized score).
- Paper: mrp.tex Section 4 new paragraph "Scorer generality" + Appendix C sentence.
  Equality case of Definition 1 exact for the four pointwise scorers; epsilon <= 0.093
  for BM25 on this suite.
