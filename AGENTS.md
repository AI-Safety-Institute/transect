# AGENTS.md

Guidance for coding agents working in this repository. Read this
end-to-end before your first substantive change; [README.md](README.md)
is the user-facing overview, this file is the working knowledge that
does not fit there. Design history lives in a private development
repository - this repo is self-contained, so a design decision you
need to understand is either stated here, in a docstring, or up for
re-derivation; there is no external doc to hunt for.

## Pull requests

- Title PRs as Conventional Commits (`<type>: <description>`), and use
  the same convention for commit messages on PR branches. We
  squash-merge, so the PR title becomes the commit message on `main`.
- `feat:` / `fix:` are for user-facing changes; `docs:`, `refactor:`,
  `chore:`, `test:`, `ci:` for the rest.
- Versioning is manual semver (no PyPI): see
  [CONTRIBUTING.md](CONTRIBUTING.md) for the bump -> tag -> release
  ritual and how users pin a version.

## What this package is

`transect` triages long agent-eval transcripts into one HTML
report plus tidy pandas frames (see README.md for the full pitch). Two
kinds of extraction, deliberately separated: **structural** ($0, no API
key - token timeline, context flushes, human interventions, sub-agent
activity, transcript info, eval setup) and **judged** (LLM - decision
phases and sub-agent classification, run solo, k-roll, or cohort,
with an optional second-round verifier). Code and docs both say
"structural" - grep for `structural`.

## Architecture

Data flow: `transect()` -> ingest + select -> Scout scan -> results
store -> frames -> report.

| Layer | Where | Owns |
|---|---|---|
| API | `src/transect/api.py` | `transect()` / `load()` / `render()`, viewer spawn, epoch loop, and the OpenClaw import driver (`_openclaw_files` / `_import_openclaw`) |
| Spec | `src/transect/spec.py` | `Spec` / `Phase` / `load_spec` - the user's eval vocabulary |
| Selection | `src/transect/selection.py` | sample + epoch selection before any judge spend |
| Ingestion | `src/transect/ingestion/` | the vendored OpenClaw telemetry parser only - the import logic that drives it lives in `api.py` (`.eval` logs are read natively by Scout) |
| Scanners | `src/transect/scanners/` | Scout scanners; each scanner's `value` dict is the store schema, documented in that scanner's docstring |
| Diagnostics | `src/transect/diagnostics.py` | scrambled-vocabulary helpers for the reliability iteration loop (see the transect-diagnostics skill) |
| Skills | `src/transect/skills.py` + `.claude/skills/` | the shipped Claude Code skills; the wheel carries them as `transect/_skills` (pyproject force-include) and `python -m transect.skills install` copies them into a project's `.claude/skills/` |
| Frames | `src/transect/frames/` | one module per frame; pandas projections of stored scan results. Each frame module's docstring is the column contract - one entry per column; adding a column means updating it |
| Reliability | `src/transect/reliability.py` | the reliability statistics (agreement coefficients, intervals, verifier rates, per-label/per-member stats) - the public analysis surface, consumed by the report |
| Report | `src/transect/report/` | charts (inspect-viz), text sections (Jinja), render orchestrator, reliability audit (report-only copy layer in `report/reliability.py`: flags, tiers, display formatting) |

`load()` + `render()` rebuild a report from a stored scan without
re-scanning - keep everything the report needs inside the scan value.
`transect()` starts a new scan; identical judge requests can reuse inspect-ai's
response cache. OpenClaw JSONL inputs are reparsed into a separate retained
`scans_dir/transcript_snapshots/` database on each invocation. Preserve these
databases for source viewing. To reclaim disk, delete only snapshot
directories no kept scan references (`load(scan_location).transcripts_location`
names the one a scan reads); a scan whose snapshot is gone still loads, but
renders without transcript excerpts and viewer links. Duplicate transcript identities within one
invocation are rejected before scanning. Never delete old stores to make a
scanner change take effect; run a new scan and retain both results for comparison.

## Load-bearing contracts

- **Frames are this repo's own contract.** Builders read columns
  directly, never behind an `in frame.columns` test; a missing column
  is a contract breach and should raise loudly. Empty frames still
  carry the full documented column set. Every frame has the identity
  prefix (`sample_id`, `task_set`, `epoch`, `transcript_id`, `agent`)
  plus `schema_version` - bump `SCHEMA_VERSION` in `frames/common.py`
  on any breaking column change (rename, removal, semantics change;
  a purely additive column does not need one).
- **Scanners decide, frames project.** Final answer facts
  (`label_source`, vote agreement, verifier outcome) are stamped by the
  scanner into the stored value; a frame never re-derives a decision.
  The scan value also records the resolved `label_vocab` - the
  vocabulary contract the frames' categoricals are built from. A store
  missing expected fields needs a re-scan, not a fallback.
- **One vocabulary source.** Enum-like frame columns derive from the
  scanners' own `Literal` types via `typing.get_args` - never restate
  a value list.
- **Vendored tree.** `src/transect/ingestion/openclaw_telemetry_hal/` is a
  mirror of inspect_scout's worked example
  (`examples/sources/openclaw_telemetry_hal/` in
  [meridianlabs-ai/inspect_scout](https://github.com/meridianlabs-ai/inspect_scout))
  and is ruff-excluded. Never edit it here; fix upstream and re-vendor.
  The vendored `__init__.py` docstring records the pinned upstream
  commit and the only permitted local deltas (its own provenance
  docstring and `populate_db.py`'s usage docstring).
- **Honest absence.** A value the source never recorded renders as "no
  data" / an explicit note - never a fabricated zero, never a silently
  missing section. Unclassified spans are listed as unclassified;
  a classifier that joined zero spans warns loudly.
- **Execution and coverage.** `results.scan_status` is reconstructed from the
  stored scan on every load. It records execution only: completed vs total
  transcripts and recorded errors per requested scanner;
  judgement quality belongs to the reliability audit. The report always shows
  this run-wide status outside optional sections. Keep it in custom UIs,
  including when an epoch filter hides other scan transcripts.

## Report code: read before touching

- The module docstring of `src/transect/report/charts.py` is a
  browser-verified constraints ledger for inspect-viz/Mosaic/Plot
  (tooltip channels, mark ordering, scale identity, axis suppression,
  …). Most failures it documents are silent in Python and blank the
  chart in the browser. Read it before writing or reviewing any chart
  code; the long docstrings in `report/` encode these findings and
  should not be trimmed.
- `embed.py`'s pixel constants are real-browser measurements; the
  comments say what was measured and how to re-measure.
- Verify report changes in a real browser, not only via tests: chart
  regressions typically produce zero Python errors. The $0 loop: run
  `transect()` on the committed demo log
  (`examples/logs/house_price_demo.eval`) with no `judge_models`, or
  re-render a stored scan via `load()` + `render()`. Charts are
  CDN-loaded - a report (and the playwright test) is dead offline.
  Playwright lives in the `ui-test` dependency group
  (`uv run --group ui-test playwright install chromium`, then
  `uv run --group ui-test pytest`).

## Working practices

- **Comments state constraints the code cannot show** - no narration,
  no implementation history. No all-caps emphasis words in prose,
  comments, or docstrings (prompt strings exempt). No em dashes and no
  `--` dashes anywhere - use plain punctuation (a spaced single hyphen
  is fine).
- **Honest framing.** Judged output is descriptive, not validated:
  headlines/summaries are narrator output, phase labels are judge
  labels with stated confidence and agreement. Don't write copy (or
  docstrings) that upgrades a judged surface into a ground truth.
- **Typing.** mypy and pyright must both pass; `make typecheck` runs
  them (and `make check` includes it). `Spec`'s bare-string sugar
  (`Spec(phases=["setup"])`) fails static checks by design - typed
  code and tests use `Spec.model_validate({...})`.
- **Tests.** Plain pytest under `tests/` (`make test`); target is a
  small, heavily parametrized suite (~100 tests) with one-sentence
  docstrings and expressive names. Judged paths are tested
  deterministically with mockllm: scripted judges and the
  content-routing `demo_judge()` live in `tests/helpers.py`.
  `tests/conftest.py` neuters inspect's on-disk model cache for every
  test. Committed fixture logs each name their regenerator script in
  the skip/fail message; note the demo log is shared by examples and
  tests, so regenerating it changes pinned test values. The committed
  judged scan stores (`tests/fixtures/demo_scan/` k-roll + verifier;
  `tests/fixtures/demo_scan_cohort/` dissenting cohort - both made by
  `tests/fixtures/generate_demo_scan.py`) must be regenerated after
  any scanner value-schema change or an inspect-scout store-format
  change - the stored-scan report test reads them as-is.
- **Costs.** The test suite and the demo-log path are $0 (mockllm / no
  judges). `examples/transect_*.py` call real provider models - keys and
  spend. Don't "verify" a change by burning judge calls when the
  deterministic loop covers it.
- **Model caching + refusals (live runs).** Scanners call
  `generate(cache=True)`; refused judge calls retry under a separate
  cache scope (`src/transect/scanners/phases_common.py`).
- **Ecosystem APIs.** Before writing against inspect_ai /
  inspect_scout / inspect_viz APIs, consult the package's own docs
  (each publishes an `llms.txt`) rather than trusting memory - these
  APIs move fast. In Claude Code, the Meridian inspect-skills plugin
  automates this routing; installing it is a user action (agents
  cannot run slash commands), so suggest the user runs, once:
  `/plugin marketplace add meridianlabs-ai/inspect-skills` then
  `/plugin install inspect-skills@meridian`.
- **Dependencies.** Runtime deps stay minimal; provider SDKs
  (anthropic/openai) belong to the `dev` group, never to runtime
  dependencies - the README instructs users to install their own
  provider. New runtime deps need discussion.
- `make sync` / `format` / `lint` / `test` / `check` / `hooks` are the
  canonical entry points. Note pre-commit runs the writing variants
  (`ruff check --fix`, `ruff format`) - it mutates the commit rather
  than failing it; `make lint` is the check-only form.

## Extending

New task types land as custom layers first:
`transect(..., extra_layers=[Layer(...)])` carries a user's scanner,
frame, typed section blocks, phase-card tags, and audit declaration
(README "Custom layers"; the authoring recipe is the shipped
add-a-layer skill; `examples/transect_custom_layer.py` is the worked
example). Layers that prove themselves across enough usage are
expected to graduate into built-in surfaces - the package grows from
observed patterns, not speculatively. A new built-in touches the
four wiring points: scanner registration in `api.py`
(`_scanners()`), a frame builder module in `frames/` plus a field on
`TransectResults` (`frames/results.py`), section assembly in
`report/render.py` (keyed for `section_order`), and a Jinja partial
under `report/templates/`.

A custom judged scanner joins the reliability machinery by conforming
to the judge-identity contract: stamp a judge block as
`value["judge"]` on every result - scanners built on
`transect.cohort_llm_scanner` get this automatically; a from-scratch
scanner (the unsupported escape hatch) builds it with
`transect.scanners.cohort.judge_setup` - and have the frame
project it with `transect.judge_identity(value)`. That yields
the standard judge columns (`judge_regime`, `n_models`, `k_rolls`,
`verifier_armed`, `verifier_same_model`), which
`transect.reliability.detect_regime` and the report's audit read directly.
Every `transect.reliability` function checks its required columns upfront
and raises a `KeyError` naming the missing ones and the fix; the
per-function contract is stated in each public function's docstring, and the conformance path is pinned end to end
by `tests/test_reliability.py`'s custom-judged-layer test.
Workflow and release flow are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Keep the docs true

A change to any user-facing contract - frame columns, the `transect()` /
`load()` / `render()` signatures, report sections, spec fields -
is not done until the docs that state that contract are re-checked:
this file, README.md, and the skills under `.claude/skills/`. In
particular, README.md's mermaid diagram of the frames shows the
joins plus a representative subset of each frame's columns (the
full column contract is each frame module's docstring): update it
when a join or a charted column changes, and chart a new column
when it is load-bearing for reading the report. It is also a good
first map of how the frames connect when you are orienting.


Review units: phase `verifier_reviews` retains original selected units before
display merging; `transect.reliability.review_units` is the review-population
read. `verifier_selected` is record presence, `verifier_completed` the
usable-verdict flag rates condition on. Mock stores must be regenerated after
scanner value-schema changes (no real provider calls needed). Public docs and
fixtures must not embed local private paths.

Narration groups: preserve each supplied title/gist only on its exact inclusive
range when the groups form a complete, non-overlapping phase partition. Invalid
partitions use a neutral whole-phase group; never clamp or extend factual prose
to repair coordinates. Preserve complete nonblank headlines and summaries
without character clipping, and stamp `narration_group_status`; `complete`
checks a partition, not factual correctness.
