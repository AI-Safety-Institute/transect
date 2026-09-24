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
[README.md](../../../README.md) (user pitch, install, frames diagram),
[AGENTS.md](../../../AGENTS.md) (architecture, contracts).

Guide the user spec-first: their knowledge of the eval goes into the
spec; the run is one call; the report is where the reading happens.
Iterating and re-running is the normal loop, not a failure - and it
has two dials: the spec (vocabulary and descriptions, when labels
land wrong) and the judge setup (regime, verifier, models - when the
reliability audit flags disagreement or inconsistency). Spec
iteration is guided here; judge-setup iteration off reliability
signals is the transect-diagnostics skill's own subject.

## 1. The spec

A YAML/JSON file (or `transect.Spec`) holding the vocabularies the judges
classify against - see `examples/spec.yaml` for a complete worked
example:

- `phases`: expected activity phases, each `label` + `description`
  (the description is rendered into the judge's rubric - write it as
  you would brief a colleague). Mark one phase `ops: true` for
  operational overhead (delegation mechanics, housekeeping); without
  one a reserved `ops` bucket is appended. `none_of_the_above` is
  always offered too - an escape hatch beats a forced wrong label.
- `subagent_labels`: expected sub-agent roles, same label +
  description shape.
- `context`: free-text task context injected into every judge prompt.

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
| `judge_models=None` | $0 structural-only run - zero LLM calls. The report then has exactly: token telemetry, human interventions, sub-agent activity (spans listed, unclassified), and the audit; phase timeline, cards, and spend need judges |
| one model | solo judge; cheapest judged run |
| one model + `k_rolls=3` | the same judge repeated k times, majority vote per turn - buys a self-consistency reading (per-turn agreement). Each roll is a real, separate API call (roll-scoped cache keys), so judge cost multiplies by k |
| a list of models | multi-model cohort, majority vote - independent judges, the strongest reliability signal |
| `verify` | second-round verifier re-reviews doubtful labels - confidence < 0.6, k-roll agreement < 0.6, or a wedge (a <= 2-turn phase between same-label neighbours) - plus a deterministic random spot-check of max(5%, 3) phases; `None` = auto: on for solo/k-roll, off for cohort (the vote is the correction mechanism; forcing it on with a cohort raises) |
| `verify_sample` | the share of judged units (phases and sub-agent spans) the verifier additionally spot-checks at random - `None` = the default 5% (phases floored at 3), `0.0` disables, `1.0` reviews everything |
| `verifier_model` | defaults to the judge model - the audit then honestly calls it a self-consistency check, not an independent second opinion; for independence set a different model, at least as capable as the judge (stronger, or peer-capability from another provider) |
| `scans_dir` | where the scan store lands; results replay from here |
| `viewer` | spawns a Scout viewer and wires the report's deep links; with no TTY (coding agent, CI) it detaches and prints its URL |
| `report_path`, `open_report`, `title` | where the report lands (default `<scans_dir>/report.html`), whether to open it in a browser, and its title. `viewer=True` and `open_report=True` are the defaults - in a coding agent or script, pass both as `False` |
| `extra_layers` | user-injected `Layer` additions (own judged classification, sections, tags, audit block) - authoring them is the add-a-layer skill's subject |
| `section_order` | report section order: listed sections first in the given order, unlisted follow in default order, the audit always last. Keys are `transect.report.SECTION_KEYS` plus each custom layer's name; also on `render()` |

Judge calls are cached (inspect's model cache), so a re-run over the
same inputs replays instead of re-spending - each k-roll replays its
own roll's answer, never another's. Provider SDKs are not Transect
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

Rough judge-cost arithmetic (for "what will this run cost me"):
segmentation is about one judge call per 40 reasoning-turn digests,
plus one narrator call per stitched phase; sub-agent classification
is one call per spawned span; every k-roll or cohort member is a full
extra pass over all of that; the verifier adds only the doubtful +
spot-checked phases, reviewed 8 per call.

## 3. Re-rendering and stored scans

`transect.load(scans_dir)` rebuilds `TransectResults` from a finished store
with no scanning and no spend; `transect.render(results,
report_path=..., title=..., viewer=False, open_report=False)`
re-renders the report (same viewer/open defaults as `transect` - turn
both off when running non-interactively). Use this for report iteration, changed titles, or reading
someone else's scan. A scanner code change does not take effect on an
existing store - scan into a fresh `scans_dir` to see it. The same
goes for OpenClaw ingestion: the transcript database inside
`scans_dir` dedupes by transcript id, so a same-dir re-run reuses the
already-imported transcript - an importer change also needs a fresh
dir.

Two stores scanned under different Transect versions are not
row-comparable: an importer upgrade can change span and sample
identities and even the sub-agent span count for the same source
file. Read cross-version differences as pipeline provenance, not
agent behaviour; for a real comparison, re-scan both sources fresh
under one version.

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
| No phases / sub-agent sections in the report | `judge_models` was not set (the run warned), or the spec declares no `phases` / `subagent_labels` |
| `sample= required` error | multi-sample log; the error lists the available ids |
| "values outside the declared vocabulary ... coerced to NaN" warning on `load()` | the store's recorded values do not match the current scanner schema - re-scan into a fresh dir ($0 for structural, judge calls replay from cache) |
| Loud "classification joined zero lanes" warning in the report | span identity mismatch between scan and render - treat as a bug, not cosmetics |
| Charts render blank | the report's charts are CDN-loaded - it is an online document; check network |
| Scanner change has no effect | results replayed from the existing `scans_dir` - use a fresh one |
| Label definitions expandable says the definitions are not recorded in the store | the rubric is embedded at scan time - re-scan into a fresh dir to record it |
| OpenClaw run: no score/success, task name looks like a filename, spans drawn as ticks, sub-agent spend "no data" | expected source gaps (the telemetry never records them), stated honestly in the report - not bugs |
| Viewer link dead after a run in a coding agent | expected on a TTY (viewer dies with the process); without a TTY it detaches - use the printed URL |

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
- The column reference is each frame module's docstring
  (`src/transect/frames/<name>.py`); the README's mermaid diagram maps how
  the frames join.
- Frames: `token_timeline`, `flushes`, `interventions`,
  `lane_activity`, `transcript_info` (structural);
  `phases`, `phase_turns`, `turn_groups`, `phase_turn_votes`,
  `subagent_votes`, `label_definitions` (judged); `subagents` is
  both - every span's structural facts on any run, label columns
  filled only when judged.
- Two column gotchas: `phases.n_turns` counts judged reasoning turns
  only (fewer than the `turn_start..turn_end` width - tool-only and
  sub-agent turns in range are not counted); `new_work` sums are
  new-content tokens, not billable cost.
- `flushes` preserves compaction `role`, full `metadata`, `strategy`,
  `messages_before`, `messages_after`, and `trigger`. For Inspect `.eval`
  logs it also carries `compaction_prompt` (summary-call input),
  `compaction_nudge` (save-context warning), and `compaction_resume`
  (instruction after a summary or Anthropic native compaction block).
  A native summary is output, not the prompt that produced it.
  Unrecorded or unidentifiable text is
  missing, including on OpenClaw and synthesized drops. The separately
  recorded template is `transcript_info.compaction_prompt`; the report
  uses it only when no observed prompt is available. These fields
  require a fresh scan; old stores cannot recover them by re-rendering.
- `transcript_info.compaction` retains the recorded strategy configuration,
  including its threshold when available. Re-rendering shows absolute
  thresholds as a toggleable dotted line on the context-window chart.
  A percentage without a recorded token count appears only in Core setup;
  missing thresholds add no row, line, or toggle. The eval's total token
  limit and the observed size before a flush are not compaction thresholds.

Starter recipes:

```python
f = transect.load("scans/").frames()

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

When a question needs transcript text (not just labels), point the
user at the report's phase-card excerpts and Scout-viewer deep links.
Frames carry selected text (setup prompts, compaction prompts/nudges,
and interventions), not the full message history.

## 7. Extending

A user addition - a judged classification over their own vocabulary,
extra sections, phase-card tags, a spend grouping, an audit block -
is a custom `Layer` via `extra_layers`; the authoring recipe is the
add-a-layer skill (and the README's "Custom layers" section).
Presenting results in a UI of your own (a dashboard, a paper
figure, an external deliverable) is the custom-ui skill's subject -
the reliability and provenance rules travel with the data.
Changing the package's own built-ins follows the wiring points in
AGENTS.md ("Extending"); workflow in CONTRIBUTING.md.
