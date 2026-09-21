---
name: transect-diagnostics
description: >
  Use when a Transect reliability audit flags concerns, when inspecting
  disagreements or verifier revisions, or when planning a controlled change
  to a rubric, judge roster, sampling settings or label vocabulary.
---

# Diagnose before changing the judging setup

Start from the exact stored scan and its `scan_status`, then inspect source
examples behind the signal. A completed scan can contain failed judgements.
An apparent improvement in agreement can reflect a changed rubric, easier cases,
missing votes or response-cache replay. It does not establish better labels.

Use [procedures.md](references/procedures.md) for dataframe recipes. The sibling
[using-transect](../using-transect/SKILL.md) covers running the pipeline and its
[intake recipe](../using-transect/references/adapting-evaluations.md) covers new
evaluations. Repository examples require a source checkout; they are not installed
alongside these skills.

## Read the quantity and its denominator

- Check recorded model names, rolls, verifier settings and coverage first.
  `mockllm/*` means scripted mechanics, not empirical judge performance.
- Solo output supplies stated confidence, not a consistency estimate.
- K-roll agreement describes repeated votes under the configured procedure.
  Temperature zero does not guarantee deterministic output. Cache replay is not
  a new draw; high agreement is not evidence of correctness.
- Cohort agreement summarizes recorded votes. Alpha and AC1 use different
  chance models; neither is a bound on true agreement or accuracy. Inspect label
  prevalence and disagreements rather than diagnosing their cause from one number.
- A verifier relabel is an applied revision, not proof the original label was
  wrong. Same-model verification uses a different prompt and is self-revision,
  not unchanged-procedure repeatability or a lower bound on another model's review.
- Review rates use original completed review units. Inspect selected, completed
  and missing outcomes separately. Historical stores may lack original phase
  review records; merged display phases cannot reconstruct them.
- Wilson and normal-approximation intervals rely on sampling assumptions.
  Dependent turns, shared batches and correlated judges can violate them; nominal
  coverage is not established by the formula. Later phase chunks share prior
  consensus hints, so their votes are not independent end-to-end segmentations.
- The audit's 0.80/0.66/0.20 flag thresholds are conventions, not validated cutoffs.

## Work through the evidence

1. Locate the flagged units and read the transcript evidence, original labels,
   explanations, final labels and available votes. Check refusals and missingness
   before interpreting a low or high rate. A random-sample relabel deserves source
   review; it is not automatically a detected error.
2. Distinguish a software/configuration problem from an unclear decision rule,
   a legitimate convention and a model mistake. More judges or a larger model
   need not fix missing evidence or an unspecified rule.
3. Form a concrete hypothesis. If changing a label definition, state which source
   distinction the new wording should capture. If changing models or settings,
   preserve the other conditions and verify the effective runtime configuration.
4. Use development cases for calibration. Keep a ledger of cases seen while
   tuning; freeze the revision before evaluating appropriately selected untouched
   cases. Agreement or better fit on selected disagreements is not generalization.
   Expert adjudication is needed for stronger correctness claims.
5. Before a paid comparison, size the actual calls and obtain the user's required
   spending approval. Include batching, context history, rolls, narration, verifier
   selection/chunks, retries and outputs. No generic transcript-length cost formula
   or historical approval establishes the next run's budget.

## Sampling and runtime controls

Build a judge with Inspect's `get_model(..., config=GenerateConfig(...))` and pass
it as `judge_models`. Available settings depend on the provider/model; verify they
are supported and inspect actual recorded requests before claiming a change took
effect. Do not assume lower temperature, more reasoning, or more judges improves
validity. Custom Scout runners must also bind their scan-level configuration;
constructing a model object alone does not demonstrate the effective concurrency
or output limit at the runtime boundary.

## Caches, repeats and recovery

`transect()` starts another scan; `load(results.scan_location)` reloads that exact
store without judging. A new output directory or virtual environment does not
isolate model responses. Keep old stores and response caches for provenance.
For a fresh-response comparison, set a new empty `INSPECT_CACHE_DIR` **before the
Python process starts**, then record it and verify dispatch/replay evidence:

```bash
INSPECT_CACHE_DIR=/path/to/new-comparison-cache uv run python your_runner.py
```

`cache=False` is not a top-level `transect()` argument. Inspect scanner authors
can control their own generation calls; distinguish those from the packaged
scanner behavior. Provider prompt-cache reuse is different from a local response
replay and may still be charged.

Budget conservatively rather than assuming all or none of a run will replay:

- Rubric changes affect the prompts that consume those fields; some other calls
  may be unchanged. `Spec.context` reaches phase prompts, not subagent prompts.
- Additional rolls/members can reuse unchanged calls, but phase prompts in later
  chunks depend on earlier consensus; changed history can invalidate those calls.
- A different verifier can alter later narration and final labels as well as its
  own calls. Increasing phase selection can recompose verifier batch prompts.
- Persisted errors can coexist with outer completion. Preserve the failed store,
  inspect item coverage, and rerun into a separate store when needed. Do not assume
  a nominally complete Scout scan can resume failed inner helper items.

Before comparing runs, verify input identity, effective Spec/settings, unit
alignment and coverage. A changed prompt can return the same label; identical
labels alone prove neither cache replay nor a wiring failure. Differences alone
prove neither improvement nor stochastic noise. Check the actual request/cache
provenance and the source evidence.

## Label-name sensitivity

`transect.diagnostics.scramble_spec` renames labels, keeps definition order fixed and preserves
text. Text can contain label-name references, so first inspect prompt coherence.
Compare any renamed condition with unchanged-setup fresh-response repeats; account
for ordering and label prevalence. One comparison cannot distinguish definition
reading from name anchoring. There is no universal `1 / number_of_labels` chance
baseline for imbalanced labels. Keep these artifacts clearly marked diagnostic,
with the mapping and provenance retained; never present neutral labels as findings.
