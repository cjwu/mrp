# Benchmark feature matrix audit, 2026-07-31

Backing record for palm/mrp.tex Table "Benchmark feature matrix" (section 7). Six
benchmarks x 7 fields, audited by six parallel Sonnet agents against released data
schemas (local copies where available) and primary documentation. Column semantics:
a cell is "yes" only if the field exists as an explicit, per-item machine-readable
label in the released data.

Local sources: benchmarks/HorizonBench, benchmarks/Locomo-Plus,
benchmarks/LongMemEval, benchmarks/PersonaMem (record schemas inspected directly).
Web sources: MultiWOZ (arXiv 1810.00278, github budzianowski/multiwoz, 2.1 data.json
and 2.2 dialogues inspected), HaluMem (arXiv 2511.03506, github MemTensor/HaluMem,
stage4_1 data + eval code inspected).

## Verdict matrix

| benchmark | current value | superseded value | explicit carrier | implicit labelled | target relation | turn timestamp | provenance |
|---|---|---|---|---|---|---|---|
| HorizonBench | yes | partial | no | no | yes (attr-level) | no (session) | LLM-sim |
| LoCoMo-Plus | no | no | no | no | partial | no (session) | LLM-sim |
| LongMemEval-KU | partial | no | partial | no | no | no (session) | mixed |
| PersonaMem | yes | partial | no | no | no | no (none) | LLM-sim |
| MultiWOZ | yes | no | partial | no | no | no | human WOZ |
| HaluMem | yes | yes | no | no | yes | yes (interpolated) | LLM-sim |

## Key evidence per cell (condensed from agent reports)

### HorizonBench
- current: `correct_letter` + `preference_evolution.current_attributes` agree (verified example).
- superseded: `distractor_letter` = pre-evolution value, populated in 76/100, 43/50, 27/30
  of evolved items (~75-90%), empty string otherwise. Hence partial.
- carrier: generation pipeline tags `preference_expressed` per turn but the tag does not
  survive to the released schema (`id, generator, user_id, conversation, correct_letter,
  options, has_evolved, preference_domain, distractor_letter, preference_evolution`).
- implicit: `trigger` in {evolution_event, contextual_evolution} = causal channel, not
  expression type; `--expression-type {explicit,implicit,natural}` is a corpus-level
  generation knob (v13_explicit / v14_implicit), never a per-item field.
- relation: `changed_attributes.{attr} = {original, current}` + `evolution_history[]`
  (event, date). Attribute-level old->new, not turn-to-turn.
- timestamp: dated sessions (~12 turns/session), turns carry only {role, content}.
- provenance: `generator` in {sonnet-4.5, o3, gemini-3-flash}; LLM-authored.

### LoCoMo-Plus
- Schema is only `relation_type, cue_dialogue, trigger_query, time_gap, model_name,
  scores, ranks, final_similarity_score`; no value/state semantics at all.
- Cognitive items have "no reference answer" (data README). `relation_type`
  (causal/state/goal/value) types a cue-trigger pair, not an old->new pairing.
- Design is implicit-by-construction (trigger prompt demands semantic distance) with no
  explicit subset. Base LoCoMo turns: session-level `session_N_date_time` only.
- Provenance: LLM-generated cues/triggers (gpt-4o-mini etc. per record) + human filtering;
  base LoCoMo is machine-human pipeline (arXiv 2402.17753).

### LongMemEval-KU (78 items)
- current: gold `answer` exists but carries no current/superseded tag; at least 2 KU items'
  gold IS the pre-update value (question asks for it). Hence partial.
- superseded: old value sits untagged in session text; the old-value turn and the new-value
  turn both carry the identical `has_answer: true`.
- carrier: `has_answer` per turn marks evidence, does not isolate the update-performing turn.
- relation: `answer_session_ids` is an unordered 2-list; direction inferable only from
  session dates (ascending in 78/78 by authoring accident, not by contract).
- timestamp: session-level `haystack_dates` only; our own consuming script
  (`longmemeval_retrieve_baf.mjs:175-182`) fabricates per-turn timestamps to force ordering.
- provenance: evidence sessions Llama-3-70B self-chat, ~70% human-edited; haystack mixes
  real ShareGPT/UltraChat logs. Hence mixed.

### PersonaMem
- current: `correct_answer` labels the current preference.
- superseded: outdated-preference text is embedded in `all_options` distractors but nothing
  distinguishes outdated from merely-irrelevant distractors. Hence partial.
- carrier: turns carry only {role, content}; `end_index_in_shared_context` is a slicing
  index (our script's "gold turn proxy" is a self-invented heuristic).
- timestamp: none at any granularity in released JSONL (generation-time MM/DD/YYYY event
  stamps not carried into release).
- provenance: GPT-4o pipeline (paper quote: "modular data curation pipeline powered by GPT-4o").

### MultiWOZ (2.0-2.4)
- current: per-turn cumulative belief state (`metadata.<domain>.semi` / `frames[].state`).
- superseded: overwritten value not retained anywhere (verified caribbean->turkish in
  MUL2395); recoverable only by diffing turn states.
- carrier: `fail_info`/`fail_book` scripted goal-change metadata, dialogue-level,
  2.0/2.1 only (dropped in 2.2+), compliance not guaranteed; no *-Change dialogue act:
  the change-executing turn is tagged generic `Inform`. Hence partial.
- relation: slot-key identity only, and not even stable (ely migrates departure->destination
  in MUL2105).
- timestamp: none (turn_id is a sequence index).
- provenance: human-human Wizard-of-Oz, ~1249 AMT workers. The only human dialogue of the six.

### HaluMem
- current: `memory_content` with `is_update: True` = updated memory (eval labels it
  "Target Memory for Update").
- superseded: `original_memories` list retains pre-update text (eval labels it
  "Original Memory Content").
- carrier: no turn link; update points stamped with session end timestamp + event-flow
  index (`assign_memory_point_timestamps`), no evidence_turn field anywhere in repo.
- implicit: `is_update` is a flat boolean, no explicit/implicit subtype.
- relation: old->new pairing via `original_memories` embedded verbatim in the new record
  (consumed exactly so by `evaluation_for_update_memory`); text copy, not an ID link.
- timestamp: per-turn second-granularity `timestamp` + `dialogue_turn` index; interpolated
  algorithmically (Dirichlet gap allocation) between session start/end.
- provenance: persona-driven GPT-4o pipeline, human role = QC/annotation only.

## Headline observations carried into the paper

1. The explicit-carrier column is empty across all six: no benchmark releases an
   utterance-level label of the supersession act. The two nearest fields (LongMemEval
   `has_answer`, MultiWOZ `fail_info`) do not isolate it.
2. The implicit-vs-explicit distinction is likewise absent in all six releases
   (HorizonBench has it only as a corpus-level generation setting).
3. HaluMem is the strongest neighbor (5/7: current, superseded, relation, turn timestamps,
   minus carrier and implicit split). HorizonBench second (current, partial superseded,
   attribute-level relation).
4. MultiWOZ is the only human-produced dialogue corpus of the six, and it lacks the
   supersession side entirely.
5. "No surveyed public benchmark jointly provides all fields" survives the audit for
   every row.
