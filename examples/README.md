# Transect examples

Runnable examples. Each script scans the demo eval log in this
directory, builds the dataframes, renders the HTML report, and
brings up the Scout viewer.

## The demo eval log

`logs/house_price_demo.eval` is a dummy transcript of a Kaggle-style
house-price regression run: a lead engineer agent explores the data,
hands off three subtasks to sub-agents (EDA, an alternative model
family, a final review), blends the models, and submits.

## Prerequisites

    pip install -e .              # from the repo root
    pip install anthropic openai  # judge-model provider SDKs
    export ANTHROPIC_API_KEY=...  # every example
    export OPENAI_API_KEY=...     # cohort example only

## What a Transect run does

`transect(logs, spec, ...)` runs the whole Transect pipeline:

1. Scout reads every transcript under `logs` (each sample of each
   `.eval` file). A run triages one sample: `sample=` picks it (it
   is required when the log carries several) and `epochs=` picks
   the epoch (an int, a list, `"all"` for one report per epoch, or
   `None` for the earliest successful one).
2. The structural scanners run on every transcript, with no LLM
   calls: per-turn token usage and context-window size, context
   compactions, mid-run human interventions, and sub-agent activity
   lanes.
3. With `judge_models` set, the judged scanners run as well: the
   run is segmented into labelled, narrated phases against the
   spec's phase vocabulary, and every spawned sub-agent is
   classified against the spec's sub-agent labels. `k_rolls`
   repeats each judgement and takes the majority vote across the
   rolls; passing several `judge_models` forms a cohort whose
   majority vote decides instead; `verify=True` sends doubtful
   judgements (low confidence, or low agreement across rolls) to a
   stronger verifier model for a second opinion that can overturn
   the label.
4. Scanner results are written to `scans_dir` and projected into
   pandas dataframes, one per surface, with typed columns and a
   schema version (`results.frames()`).
5. The HTML report is rendered from the dataframes and the Scout
   viewer starts for browsing the underlying transcripts.
   `load(scans_dir)` reloads a finished scan later without
   rescanning; `render(results)` re-renders the report.

## 1. Solo k-roll judge with a verifier

    python examples/transect_kroll.py

One judge model labels each phase and sub-agent three times; the
majority vote across its own rolls decides. A second, stronger
verifier model re-checks the judgements flagged as doubtful (low
confidence or low agreement across rolls).

## 2. Multi-model cohort

    python examples/transect_cohort.py

Three different judge models each label once; the majority vote
across the cohort decides. No verifier in this shape: the cohort is
the second opinion. The report carries the per-member ballots
alongside the voted labels.

## 3. Custom layer

    python examples/transect_custom_layer.py

The custom-interface worked example: a user-defined judged layer
(one research-skill label per orchestrator reasoning turn) injected
via `extra_layers`. Its scanner is built on `cohort_llm_scanner`
over the `reasoning_turns` loader with its own closed vocabulary
(including the `none_of_the_above` escape hatch); the layer adds a
badge-marked report section (markdown + turn band), `skill=` chips
and a filter on the phase cards, a "Group by: skill" option in the
Token spend section, and its own entity block in the reliability
audit. The full authoring recipe is the `add-a-layer` skill.

## The spec

`spec.yaml` is the spec every example uses: the phase vocabulary and
sub-agent labels were authored against the demo log's transcript,
and its `extra` block carries the custom-layer example's
research-skill rubric. JSON and YAML are equivalent
(`transect.spec.load_spec`).

## $0 dry run

Example 1 or 2 with `judge_models=None` (edit the script) runs the
structural surfaces only: token timeline, context flushes, human
interventions, sub-agent lanes. No API key, no LLM calls. The
custom-layer example has no $0 shape.
