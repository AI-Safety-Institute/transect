# Transect examples

Runnable examples. Each script scans the demo eval log in this
directory, renders the HTML report, and brings up the Scout
viewer.

`logs/house_price_demo.eval` is a dummy transcript of a
Kaggle-style house-price regression run: a lead engineer agent
explores the data, hands off three subtasks to sub-agents (EDA, an
alternative model family, a final review), blends the models, and
submits.

Every example needs `ANTHROPIC_API_KEY`; the cohort example also
needs `OPENAI_API_KEY` (with each provider's SDK installed).

## 1. Solo k-roll judge with a verifier

    python examples/transect_kroll.py

One judge model labels each phase and sub-agent three times; the
majority vote across its own rolls decides. A second, stronger
verifier model re-checks the judgements flagged as doubtful (low
confidence or low agreement across rolls).

## 2. Multi-model cohort

    python examples/transect_cohort.py

Three different judge models each label once; the majority vote
across the cohort decides. No verifier in this shape: the cohort
is the second opinion. The report carries the per-member ballots
alongside the voted labels.

## 3. Custom layer

    python examples/transect_custom_layer.py

A user-defined judged layer (one research-skill label per
orchestrator reasoning turn) injected via `extra_layers`: its own
badge-marked report section, phase-card chips and filter, a
"Group by: skill" option in Token spend, and an entity block in
the reliability audit.

## Sample and epoch selection

A run scans one sample: `sample=` picks it (required when the log
carries several) and `epochs=` picks the attempt: an int, a list,
`"all"` for one report per epoch, or `None` for the earliest
successful one.

## The spec

`spec.yaml` is the spec every example uses: the phase vocabulary
and sub-agent labels were authored against the demo log, and its
`extra` block carries the custom-layer example's research-skill
rubric.

## $0 dry run

Example 1 or 2 with `judge_models=None` (edit the script) runs the
structural scanners only: no API key, no LLM calls. The
custom-layer example has no $0 shape.
