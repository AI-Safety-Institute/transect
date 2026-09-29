---
name: using-transect
description: >
  Use when helping someone run or understand Transect:
  triaging an agent-eval transcript, authoring or iterating a spec,
  choosing judge settings (solo / k-rolls / cohort / verifier),
  explaining what a Transect run is doing, reading a Transect report or
  interpreting any of its sections (the reliability audit included),
  re-rendering from a stored scan,
  or doing custom pandas analysis on the frames. Covers the default
  path end to end; reliability-driven judge-setup refinement and
  deep failure diagnosis live in the sibling transect-diagnostics skill.
---

# Using Transect

`transect` triages one long agent-eval transcript into one HTML report
file (single file, but its charts load from a CDN - it reads online)
plus pandas frames: judged decision phases and sub-agent
classifications on top of a structural ($0) record of tokens, context
flushes, human interventions, and sub-agent activity. Orientation:
[README](https://github.com/AI-Safety-Institute/transect/blob/main/README.md)
(user pitch, install, frames diagram) and
[AGENTS](https://github.com/AI-Safety-Institute/transect/blob/main/AGENTS.md)
(architecture, contracts).

Guide the user spec-first, in this order. Ask what the eval is and
what they want to learn from the run; their knowledge goes into the
spec, and a spec drafted from the log alone decides what the judges
look for without them. Then run a $0 structural pass
(`judge_models=None`, section 2) and read its report together: the
sub-agent spawn prompts, the human interventions and the eval setup
are the raw material for a vocabulary nobody knows before looking.
Draft the phases and sub-agent labels from both, show the draft, and
get the user's confirmation before any judged run. The run is one
call; the report is where the reading happens. Iterating and
re-running is the normal loop, not a failure - and it has two dials:
the spec (vocabulary and descriptions, when labels land wrong) and
the judge setup (regime, verifier, models - when the reliability
audit flags disagreement or inconsistency). Spec iteration is guided
here; judge-setup iteration off reliability signals is the
transect-diagnostics skill's own subject.

For a new evaluation or a domain-specific adaptation, work through
section 8 (Adapting to a new evaluation) before choosing labels or
custom layers.

## 1. The spec

A YAML/JSON file (or `transect.Spec`) holding the vocabularies the judges
classify against. The worked example is
[examples/spec.yaml](https://github.com/AI-Safety-Institute/transect/blob/main/examples/spec.yaml):

- `phases`: expected activity phases, each `label` + `description`
  (the description is rendered into the judge's rubric - write it as
  you would brief a colleague). Mark one phase `ops: true` for
  operational overhead (delegation mechanics, housekeeping); without
  one a reserved `ops` bucket is appended. `none_of_the_above` is
  always offered too - an escape hatch beats a forced wrong label.
- `subagent_labels`: expected sub-agent roles, same label +
  description shape.
- `context`: additional context for phase prompts. It is not passed to the
  built-in sub-agent classifier or automatically to custom scanners.
- `extra`: opaque configuration for your own code. A custom layer must read
  it and explicitly construct its question, vocabulary or display data;
  adding keys does not create an analysis surface.

An empty `phases` disables segmentation; empty `subagent_labels`
disables classification (each warns at run time). Bare strings are
sugar for `{label: ...}` - but in typed code use
`Spec.model_validate({...})`, the sugar fails static checkers.

## 2. Running

```python
import transect

results = transect.transect(
    logs="logs/",  # .eval file/dir, or OpenClaw .jsonl
    spec="spec.yaml",
    sample="my-sample",  # required when the log has several
    judge_models="anthropic/claude-sonnet-4-6",
    scans_dir="scans/",
)
```

Key arguments and what they mean:

| Argument | Meaning |
|---|---|
| `logs` | Inspect `.eval` log(s) read natively; OpenClaw `.jsonl` is imported into a transcript database first |
| `sample`, `epochs` | one run triages one sample; `epochs="all"` scans every epoch (one report file each), `None` picks the earliest successful epoch (the earliest overall when none succeeded) |
| `judge_models=None` | Built-in structural extraction without LLM calls; judged phases and sub-agent classifications are absent. Separately configured custom scanners may have their own model calls |
| one model | solo judge; cheapest judged run |
| one model + `k_rolls=3` | the same judge repeated k times, majority vote per turn - a self-consistency reading. Each roll has separately scoped requests and cache keys; account for fresh requests in each roll |
| a list of models | multi-model cohort, majority vote; inspect member coverage and agreement without assuming model errors are independent |
| `verify` | second-round verifier re-reviews doubtful labels - confidence < 0.6, k-roll agreement < 0.6, or a wedge (a <= 2-turn phase between same-label neighbours) - plus a deterministic random spot-check of max(5%, 3) phases; `None` = auto: on for solo/k-roll, off for cohort (the vote is the correction mechanism; forcing it on with a cohort raises) |
| `verify_sample` | the share of judged units (phases and sub-agent spans) the verifier additionally spot-checks at random - `None` = the default 5% (phases floored at 3), `0.0` disables, `1.0` reviews everything |
| `verifier_model` | defaults to the judge model: self-revision under a different review prompt. A different model changes the reviewer; it does not guarantee independent errors or greater correctness |
| `scans_dir` | parent for new scan stores and retained OpenClaw snapshots; `load()` reads a saved scan |
| `viewer` | spawns a Scout viewer and wires the report's deep links; with no TTY (coding agent, CI) it detaches and prints its URL |
| `report_path`, `open_report`, `title` | where the report lands (default `<scans_dir>/report.html`), whether to open it in a browser, and its title. `viewer=True` and `open_report=True` are the defaults - in a coding agent or script, pass both as `False` |
| `extra_layers` | user-injected `Layer` additions (own judged classification, sections, tags, audit block) - authoring them is the add-a-layer skill's subject |
| `section_order` | report section order: listed sections first in the given order, unlisted follow in default order, the audit always last. Keys are `transect.report.SECTION_KEYS` plus each custom layer's name; also on `render()` |

Judge requests use inspect-ai's response cache. An identical request with
a matching cache entry can replay; a repeated invocation does not guarantee
zero spend. Each k-roll has its own cache scope. Record reused versus fresh
responses separately when interpreting repeats. Provider SDKs are not Transect
dependencies - the user installs their own (`pip install anthropic`)
and exports the matching API key.

What a run does, in order: ingest (+ OpenClaw import if needed) ->
sample/epoch selection (before any judge spend) -> Scout scan
(structural scanners always; `decision_phases` and
`subagent_classification` when judges are set) -> frames -> report +
viewer. Explain the run to the user in those terms; the scan is where
time and money go.

How the two judge scanners work, when someone asks what is happening
under the hood: `subagent_classification` is a thin factory over the
generic `cohort_llm_scanner` - each span is one independent item, one
question, one label+confidence answer, and the generic layer supplies
the solo/k-roll/cohort regimes, voting, verifier, and `label_source`
bookkeeping. `decision_phases` implements the same regime concepts
itself rather than using that layer, because segmentation is
stateful: each judge call sees a chunk of turn digests plus the
running last-phase context and answers segment boundaries and labels
together; phases are then stitched across chunks, and a verifier
overturn re-stitches the partition. A per-item generic layer cannot
express that, so the two scanners share the vocabulary and member
machinery (triggers, label sources, roll-scoped caching) but not the
execution shape. Extension rule of thumb: a per-item labelling task
builds on `cohort_llm_scanner`; a stateful/sequence task uses
`decision_phases` as the reference implementation.

Size paid work from the actual prepared requests, including batch boundaries,
phase narration, verifier selection, each model/roll and allowed retries.
Input size, output limits and provider prices determine each call's estimate;
turn counts alone do not supply a spending bound (section 8 has the
per-call ledger).

## 3. Re-rendering and stored scans

Inspect `results.scan_status` on both new and reloaded results. Its `outer_complete`
describes scanner execution, while its per-scanner records describe completed
vs total transcripts, recorded errors and custom results not mounted
into the report. It records execution only: a cleanly executed scan can still
carry failed or refused judgements, which surface in the reliability audit.
The HTML status block remains visible even when a failed section is empty; it
covers the whole scan, including in epoch-specific reports. Preserve it in custom
reports and do not claim that a successfully returned object means all work passed.

`transect.load(results.scan_location)` rebuilds `TransectResults` from that exact store
with no scanning and no spend; `transect.render(results,
report_path=..., title=..., viewer=False, open_report=False)`
re-renders the report (same viewer/open defaults as `transect` - turn
both off when running non-interactively). Use this for report iteration, changed titles, or reading
someone else's scan. Call `transect()` again to apply scanner or importer changes;
it creates a new scan even when `scans_dir` is unchanged. OpenClaw inputs are
reparsed into a new retained database in `scans_dir/transcript_snapshots/`, so
edited content cannot silently reuse an older imported transcript. Keep those
databases for source viewing; to reclaim disk, delete only
snapshots no kept scan references (`load(scan_location).transcripts_location`
names a scan's snapshot) - the scan still loads afterwards, but renders
without excerpts and viewer links. Supply only one file for each transcript
identity in an invocation; duplicates raise before scanning. Identical judge
requests can still replay from inspect-ai's separate response cache. For exact
reloads, pass the saved `results.scan_location` rather than the parent directory;
a parent selects its newest scan by modification time. Partial loads still
require readable structural tables and valid mounted custom-frame contracts.

For an independently fresh response-cache run, point `INSPECT_CACHE_DIR` at
a new empty directory before starting the process. Preserve existing caches
and scans for comparison. The public `transect()` call has no `cache=False`
argument; a new scan and a cold model-response cache are different choices.

A run that dies part-way (an OOM kill, a provider outage, Ctrl+C) leaves a
store that `load()` reports as incomplete; `transect()` never resumes it.
Rerun the same call unchanged. The structural scanners are cheap, and every
judge call that completed replays from the response cache, so the spend is
roughly the calls that were in flight or never started. Two conditions: keep
`INSPECT_CACHE_DIR` where it was (a fresh directory makes the rerun pay in
full), and change nothing that alters the requests - spec vocabulary, judge
models, `k_rolls`, chunking - because a changed request is a new request.
Wall-clock is paid again in full: the transcript is reparsed and each judged
scanner re-dispatches every item, cached or not. The dead store holds no
results for a judged scanner that had not finished its transcript (Scout
records a scanner's results per transcript, once its loader finishes), so
there is nothing partial to salvage from it; the cache is the only carrier.

Do not assume cross-version stores are row-comparable: an importer upgrade
can change span/sample identities and the span count for the same source.
Check versions, schemas and unit alignment before attributing differences to
agent behaviour. If the comparison requires new scans under one version,
retain the earlier stores and size any fresh judge calls first.

## 4. Reading the report

Read [references/report-guide.md](references/report-guide.md) before
walking a user through a report - it covers every section top to
bottom (what it shows, how to read it, what to check), the
reliability audit's thresholds, and the interpretation discipline.
The one-line version: everything judged is descriptive and carries
its provenance (who decided, confidence, agreement); grey always
means "not judged", never "fine"; absence is stated, never faked.

## 5. Quick triage (common issues)

| Symptom | Cause / fix |
|---|---|
| No phases / sub-agent sections in the report | the Phase timeline section states the reason in place: no judge configured (`judge_models=None`, the console warning names the declared vocabularies), or the judge ran and produced no phases - then inspect `scan_status` for failed or missing requested work |
| No agreement strip under the phase band | a solo judge (one model, `k_rolls=1`) has no per-turn agreement to show; the caption in its place says so - `k_rolls` > 1 or a cohort brings the strip back |
| `sample= required` error | multi-sample log; the error lists the available ids |
| "values outside the declared vocabulary ... coerced to NaN" warning on `load()` | the store's recorded values do not match the current scanner schema - run a new scan, retain the old one, and account for any fresh judge calls |
| Loud "none joined this transcript's lanes" warning in the report | span identity mismatch between scan and render - treat as a bug, not cosmetics |
| Charts render blank | the report's charts are CDN-loaded - it is an online document; check network |
| Scanner change has no effect | `load()` reads saved results; call `transect()` for a new scan, and check the separate model-response cache when judging |
| Label definitions expandable says the definitions are not recorded in the store | the rubric is embedded at scan time - run a new scan to record it and retain the old store |
| OpenClaw run: no score/success, task name looks like a filename, spans drawn as ticks, sub-agent spend "no data" | expected source gaps (the telemetry never records them), stated honestly in the report - not bugs |
| Viewer link dead after a run in a coding agent | expected on a TTY (viewer dies with the process); without a TTY it detaches - use the printed URL |
| Run died part-way (OOM kill, Ctrl+C, provider outage) | rerun the same call unchanged with the same `INSPECT_CACHE_DIR`: completed judge calls replay from the response cache, only in-flight and unstarted calls are paid; the old store loads as incomplete and holds nothing for a judged scanner that did not finish |

Anything deeper belongs to the `transect-diagnostics` skill: reading the
reliability metrics and provenance to refine the judge setup (regime,
verifier, models, k), judge disagreement and refusal patterns, store
internals, chart debugging.

## 6. Custom analysis on the frames

`results.frames()` returns plain pandas DataFrames; no extra tooling
is needed - write ordinary pandas in a script or notebook. (If the
Meridian inspect-skills py-repl is installed, it works on these frames
too, but nothing here requires it.)

- Every frame carries the identity prefix (`sample_id`, `task_set`,
  `epoch`, `transcript_id`, `agent`) - the universal join key - plus
  `schema_version`.
- The column reference is each `transect.frames` module's docstring
  (also browsable as the
  [frame modules](https://github.com/AI-Safety-Institute/transect/tree/main/src/transect/frames));
  the README diagram maps the joins.
- Frames: `token_timeline`, `flushes`, `interventions`,
  `lane_activity`, `transcript_info` (structural);
  `phases`, `phase_turns`, `turn_groups`, `phase_turn_votes`,
  `subagent_votes`, `label_definitions` (judged); `subagents` is
  both - every span's structural facts on any run, label columns
  filled only when judged.
- Two column gotchas: `phases.n_turns` counts the reasoning turns the
  judge saw, judged and filled (fewer than the `turn_start..turn_end`
  width - tool-only and sub-agent turns in range are not counted); `new_work` sums are
  new-content tokens, not billable cost.
- `flushes` preserves compaction `role`, full `metadata`, `strategy`,
  `messages_before`, `messages_after`, and `trigger`. For Inspect `.eval`
  logs it also carries `compaction_prompt` (the summarization prompt as
  the model saw it) and `compaction_nudge` (the save-to-memory warning
  issued before compaction). The nudge only exists when the run had a
  `memory` tool and the strategy's memory warning on; native provider
  compaction records no prompt. Unrecorded text is missing, including on
  OpenClaw (whose export has no prompt or nudge) and synthesized drops.
  The configured template is `transcript_info.compaction_prompt`; the
  report uses it only when no observed prompt is available. These fields
  require a fresh scan; old stores cannot recover them by re-rendering.
- `transcript_info.compaction` retains the recorded strategy configuration,
  including its threshold when available. Re-rendering shows absolute
  thresholds as a toggleable dotted line on the context-window chart.
  A percentage without a recorded token count appears only in Core setup;
  missing thresholds add no row, line, or toggle. The eval's total token
  limit and the observed size before a flush are not compaction thresholds.

Starter recipes:

```python
f = transect.load(results.scan_location).frames()

# token spend per phase label
f["phases"].groupby("phase").new_work_tokens.sum().sort_values()

# turns where the cohort split (low per-turn agreement), with labels
t = f["phase_turns"]
t[t.judge_agreement < 0.7][["turn", "phase", "judge_agreement", "basis"]]

# each judge's own ballot on a contested turn
v = f["phase_turn_votes"]
v[v.turn == 41][["model", "roll", "phase", "confidence"]]

# verifier overturns with before/after labels
p = f["phases"]
p[p.overturned.fillna(False)][
    ["phase_index", "original_label", "phase", "verifier_trigger"]
]

# sub-agent spend by classified role
f["subagents"].groupby("label").new_work.sum()
```

For source reading, use the report's excerpts and Scout-viewer deep links;
frames are not a complete transcript reconstruction, but they do carry
source text (prompts, compaction prompts and nudges, interventions, delegated task text, judge
explanations) - check the fields before sharing an export.

## 7. Extending

A user addition - a judged classification over their own vocabulary,
extra sections, phase-card tags, a spend grouping, an audit block -
is a custom `Layer` via `extra_layers`; the authoring recipe is the
add-a-layer skill (and the README's "Custom layers" section).
Presenting results in a UI of your own (a dashboard, a paper
figure, an external deliverable) is the custom-ui skill's subject -
the reliability and provenance rules travel with the data.
Changing the package's built-ins requires a source checkout: follow
[AGENTS: Extending](https://github.com/AI-Safety-Institute/transect/blob/main/AGENTS.md#extending)
and [CONTRIBUTING](https://github.com/AI-Safety-Institute/transect/blob/main/CONTRIBUTING.md).

## 8. Adapting to a new evaluation

Intake first: record which transcript, sample and epoch are selected, and
distinguish original task material (task definition, scaffold
configuration, native scorer) from previously generated specs, labels and
reports. Agree which sources are open for development and which are held
out, and keep held-out outcomes and intended solutions out of prompt
iteration. Inspect the source's actual schema before deriving counts or
captions - a native scalar, an event list and a checkpoint roster measure
different things - and preserve native values and their units.

Map each requested output to its actual consumer, with a check there:

| Requirement | Implementation route | Check at the consumer |
|---|---|---|
| Describe activity phases | `Spec.phases`, descriptions and phase-only `Spec.context` | Capture the prepared phase prompt and inspect source-associated labels |
| Classify delegated tasks | `Spec.subagent_labels` | Inspect the prepared span task and its resulting classification; global context is not injected |
| Add another judged facet | Custom loader and `cohort_llm_scanner` question | Verify unit identity, question text, vocabulary and explicit failure statuses |
| Show a native outcome | Mechanical extraction or a data-only `Layer` | Compare displayed values and units with the selected native record |
| Present a custom view | Typed layer blocks or a separate UI over frames | Check joins, coverage, provenance and exact source-link destinations |

Choose the unit from the evidence: multiple actions in one turn need
separate identities or an explicit aggregation rule, and a single primary
label on a compound action hides secondary methods from an exact-label
filter.

Start from a fresh, empty spec and the structural pass described at the
top, not an inherited vocabulary; derive the labels from the task, the
review question and what that pass shows, with descriptions and escape
categories, and confirm them with the user before judging.

Calibrate before paying for labels: capture prepared prompts with a
deterministic model substitute to verify the rubric and context reach the
intended consumer; use negative controls for dropped, repeated and
reordered batch answers; check per-unit joins and
`no_answer`/error/refusal states, not only row counts. Build a fixed
development set with source-grounded reference judgments and keep
held-out material out of the tuning loop. An agent's assertion of success
is not an observed result; confidence, agreement and a populated report
do not demonstrate correctness.

Before a paid run, prepare a per-call ledger - scanner, unit identities,
model and roll, prepared input size, output cap, provider prices, allowed
retries; narration and verifier counts depend on earlier labels, so state
bounds - and get the user's spending approval. Verify effective model
settings at the provider boundary; Scout scan settings can override model
defaults. After each staged run, check `results.scan_status` and item
statuses (section 3) and read the report in a browser.
