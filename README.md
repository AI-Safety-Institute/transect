<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/transect-logo-dark.svg">
    <img src="docs/images/transect-logo.svg" alt="Transect logo" width="140">
  </picture>
</p>

# Transect

Transcript analysis for long-horizon agent evaluations, built on
[Inspect Scout](https://meridianlabs-ai.github.io/inspect_scout/).

As agent runs stretch to hundreds of pages, the range of what a
reviewer can reliably infer about the model's behaviour narrows.
Pointing an LLM judge at the transcript only moves the problem:
judged labels cannot be taken at face value either. Transect is
built around that fact. Every classification it produces carries
its provenance and reliability, so a reviewer always knows how
much weight a label can bear.

```python
from transect import transect

results = transect(
    "logs/",  # Inspect .eval log(s), or OpenClaw .jsonl
    "spec.yaml",  # your phase + sub-agent vocabulary
    judge_models="anthropic/claude-sonnet-4-6",
)
# Transect report  -> scans/report.html
# Scout viewer  -> http://127.0.0.1:PORT  (deep links into the transcript)
```

Transect places a run on one navigable turn-based timeline:
structural scanners extract recorded events (token use, context
compactions, human interventions, sub-agent activity) at no model
cost, judged scanners classify behaviour against your own
vocabulary, and a self-contained HTML report presents both. The
same surfaces land as pandas dataframes, and everything
extends: custom scanners, custom report layers, or a fully custom
UI over the frames.

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Reading the report](#reading-the-report)
- [Working with the dataframes](#working-with-the-dataframes)
- [Custom layers](#custom-layers)
- [Cost](#cost)
- [Development](#development)

## Prerequisites

- Python 3.12+
- The SDK and an API key for each judge-model provider you use:

  ```bash
  pip install anthropic
  export ANTHROPIC_API_KEY=your-anthropic-api-key

  pip install openai
  export OPENAI_API_KEY=your-openai-api-key
  ```

  Any [Inspect AI model
  provider](https://inspect.aisi.org.uk/providers.html) works; a
  judged run without the provider's package installed fails.

- Transcripts: Inspect `.eval` log files, or OpenClaw `.jsonl`
  telemetry exports.

## Installation

Install straight from GitHub:

```bash
uv pip install git+https://github.com/AI-Safety-Institute/transect.git
# or: pip install git+https://github.com/AI-Safety-Institute/transect.git
```

Or clone for a local install (the runnable examples live in the
repository, not the package):

```bash
git clone https://github.com/AI-Safety-Institute/transect
cd transect
uv sync
# or: pip install -e .
```

If you work with Claude Code, the package ships four skills - a
guide to running the default pipeline and reading the report
(`using-transect`), the reliability iteration loop (`transect-diagnostics`),
the custom-layer authoring recipe (`add-a-layer`), and the
custom-presentation recipe (`custom-ui`). Install them into your
project once (re-run after upgrading the package):

```bash
python -m transect.skills install   # copies them into ./.claude/skills/
```

## Getting started

The repository ships a runnable worked example in `examples/`:

```bash
python examples/transect_kroll.py    # one judge, 3 rolls + verifier
python examples/transect_cohort.py   # three-model judge cohort
```

`examples/README.md` walks through both.

### Concepts

- **Spec**: one YAML/JSON file describing an evaluation family,
  reusable across its runs: phase vocabulary, sub-agent labels,
  free task context for the judges.
- **Scanners**: per-transcript extractors, run by Scout. Structural
  scanners are free and always on; judged scanners run when
  `judge_models` is set and the Spec declares their vocabulary.
- **Judge regimes**: one model solo, one model rolled `k_rolls`
  times, or a multi-model cohort - votes decided by majority - plus
  an optional verifier that re-checks doubtful judgements.
- **Frames**: the results as pandas dataframes.
- **Report + viewer**: a self-contained HTML report, deep-linking
  into a local Scout viewer for the full transcripts.

### Triaging your own transcripts

Two built-in judged scanners need your input, declared in the
spec. `decision_phases` segments the run into contiguous phases:
it needs the phase vocabulary you expect the run to move through,
each phase described in your own words. `subagent_classification`
labels every spawned sub-agent from its delegation instructions:
it needs the sub-agent roles you expect. Free task context
gives the phase judge background on the eval itself. The better
your descriptions, the better the labelling. A minimal spec:

```yaml
context: >
  An open-ended AI R&D task: the agent is given a research brief
  and has to implement the described method, run experiments with
  it, and write up its findings. It can delegate subtasks to
  sub-agents.

phases:
  - label: planning
    description: Understanding the brief and choosing an approach.
  - label: implementation
    description: Writing and debugging the method and experiment code.
  - label: experimentation
    description: Running experiments and analysing their results.
  - label: writeup
    description: Writing the findings up into the report.

subagent_labels:
  - label: literature_survey
    description: Surveys prior work relevant to the brief.
  - label: experiment_run
    description: Runs a training or evaluation experiment.
  - label: result_review
    description: Checks results and drafts before they are used.
```

`none_of_the_above` and an operational bucket are always
available to the judges.

> [!TIP]
> The shipped skills help here (see [Installation](#installation)):
> `using-transect` drafts a first spec from your description of the
> eval. Expect to iterate, a rubric is rarely right first time:
> read the report's reliability audit, refine your labels and
> descriptions, and re-run; `transect-diagnostics` walks that loop.

With the spec written, one call runs the whole pipeline:

```python
from transect import transect

results = transect(
    "logs/my_run.eval",
    "spec.yaml",
    sample="my-sample",  # which sample from the log to scan
    epochs=1,  # which epoch(s); "all" = report per epoch
    judge_models="anthropic/claude-sonnet-4-6",
    k_rolls=3,
    verify=True,
    scans_dir="scans/my_run",  # where the scan results land
)
```

This extracts the structural surfaces, has the judge label every
turn and sub-agent against your spec (three rolls each, majority
vote, with a verifier re-checking doubtful judgements), writes the
results to `scans_dir`, renders the HTML report, and starts the
viewer.

`load(scans_dir)` re-reads a finished scan without rescanning (and
without API calls); `render(results)` re-renders the report.

## Reading the report

![The Transect report on a CRUX AI R&D run: 71-phase timeline with the per-turn agreement strip (hovered: member votes, agreement with its denominator, and label provenance), human interventions, and token telemetry](docs/images/transect_report.png)

Top to bottom, everything on a shared turn axis:

- **Eval setup**: model, scaffold, verbatim prompts, limits, run summary.
- **Phase timeline**: the phase band plus the per-turn judge-agreement strip.
- **Human interventions**: mid-run operator messages and console inputs.
- **Token telemetry**: per-turn token measures and context size, compactions marked.
- **Sub-agent activity**: one swimlane per spawned sub-agent, with its classified role.
- **Token spend**: tokens by phase, by sub-agent, or by custom tag family.
- **Phase cards**: one expandable card per phase: label, narration, excerpts, reliability.
- **Custom layers**: user layers' own sections; `section_order` rearranges the report.
- **Reliability & provenance audit**: per judged surface, how every label was produced.

### Building your own UI

The shipped report is the reference rendering: build your own
presentation over the frames. The reliability and provenance
discipline travels with the data, not with our HTML.

> [!TIP]
> The recipe and checklist are the `custom-ui` skill.

## Working with the dataframes

For custom analysis beyond the generated report, everything the
scanners extracted is exposed as pandas dataframes on the results
object, ready for a notebook or your own Python:

```python
frames = results.frames()  # dict of name -> DataFrame
frames["transcript_info"]  # one row per transcript: task, model, outcome, setup
frames["token_timeline"]  # per-turn token usage + context size
frames["flushes"]  # context compactions, tokens before/after
frames["interventions"]  # mid-run human interventions
frames["lane_activity"]  # per-turn sub-agent tool activity
frames["phases"]  # stitched phases + reliability
frames["phase_turns"]  # per-turn phase attribution
frames["turn_groups"]  # narrated turn groups within each phase
frames["phase_turn_votes"]  # per-judge per-turn votes
frames["subagents"]  # one row per sub-agent span, classification when judged
frames["subagent_votes"]  # per-judge votes behind each label
frames["label_definitions"]  # the label rubric the judges classified against
```

Every frame carries the identity prefix (`sample_id`, `task_set`,
`epoch`, `transcript_id`, `agent`) and `schema_version`; per-column
semantics live in the corresponding `transect/frames/*.py` docstring.
They are plain pandas: filter, join, and plot as usual.

### How the frames relate

Each box below is one frame, with its granularity in the header;
the edges are the within-transcript join keys.

```mermaid
erDiagram
    transcript_info ||--o{ token_timeline : "transcript_id"
    transcript_info ||--o{ phases : "transcript_id"
    transcript_info ||--o{ subagents : "transcript_id"
    token_timeline ||--o| phase_turns : "turn"
    token_timeline ||--o{ flushes : "turn"
    token_timeline ||--o{ interventions : "turn"
    token_timeline ||--o{ lane_activity : "turn"
    subagents ||--o{ lane_activity : "agent_span_id"
    subagents ||--o{ subagent_votes : "agent_span_id"
    phases ||--o{ phase_turns : "phase_index"
    phases ||--o{ turn_groups : "phase_index"
    phase_turns ||--o{ phase_turn_votes : "turn"
    label_definitions ||--o{ phases : "label"
    label_definitions ||--o{ subagents : "label"

    transcript_info["transcript_info (one row per transcript)"] {
        string transcript_id PK
        string sample_id
        string task_name
        string model
        bool success
        float wallclock_seconds
        int message_count
        int total_tokens
        string error "None unless the run errored"
        string limit "the terminating limit, if one"
        string scaffold_prompt
        bool header_available "False on OpenClaw imports"
        int message_limit "None = not set / not found"
        string system_prompt "verbatim; the long prompts sit last"
    }
    token_timeline["token_timeline (one row per model turn)"] {
        int turn PK
        string agent_span_id FK "sub-agent lane; NA on the main lane"
        int output_tokens
        int new_work
        int context
    }
    flushes["flushes (one row per context compaction)"] {
        int turn FK
        string type
        int tokens_before
        int tokens_after
    }
    interventions["interventions (one row per human intervention)"] {
        int turn FK
        string channel
        string content
    }
    lane_activity["lane_activity (one row per (turn, sub-agent span))"] {
        int turn FK
        string agent_span_id FK
        int tool_calls
        float busy_seconds
        int span_end_turn
    }
    phases["phases (one row per stitched phase)"] {
        int phase_index PK
        string phase FK
        int turn_start
        int turn_end
        string headline
        float confidence
        float judge_agreement
        string confidence_source
        bool verifier_reviewed
        bool overturned
    }
    phase_turns["phase_turns (one row per turn (judged surface))"] {
        int turn PK
        int phase_index FK
        string phase
        string basis
        float confidence
        float judge_agreement
    }
    phase_turn_votes["phase_turn_votes (one row per (judge member, turn))"] {
        int turn FK
        string model
        int roll
        string phase
        float confidence
    }
    turn_groups["turn_groups (one row per narrated turn group)"] {
        int phase_index FK
        int group_index
        int turn_start
        int turn_end
        string title
        string gist
    }
    subagents["subagents (one row per classified sub-agent)"] {
        string agent_span_id PK
        string agent_lane
        string label FK
        float confidence
        string label_source
        bool verifier_reviewed
        bool overturned
    }
    subagent_votes["subagent_votes (one row per (judge member, span))"] {
        string agent_span_id FK
        string model
        int roll
        string label
        float confidence
        string status
    }
    label_definitions["label_definitions (one row per (surface, label))"] {
        string surface PK "phases / subagents / a custom layer's name"
        string label PK
        string description
        bool ops
    }
```

Custom layers sit alongside these: each layer's frame mounts at
`results.layer_frames[name]`, and declared tag families land in
`results.turn_tags`.

### Reliability analysis

`transect.reliability` exposes the functions behind the report's
"Reliability & provenance audit" section for your own analysis
over the frames:

```python
from transect import reliability

reliability.detect_regime(frames["phases"])  # solo / k-roll / cohort + verifier state
reliability.cohort_agreement(frames["phase_turn_votes"], "turn", "phase")
# CohortAgreement(percent_agreement=0.67, alpha=0.57, ac1=0.65, n=1424)
reliability.relabel_rate(frames["phases"])  # verifier re-labels, overall + per label
reliability.spot_check_overturns(frames["phases"])  # overturns among random spot-checks
reliability.member_coverage(
    frames["phase_turn_votes"], "basis", "judged", ("refusal", "no_answer", "missing_turn")
)  # usable judgements per (model, roll), with miss reasons
reliability.label_stats(
    frames["phase_turns"][frames["phase_turns"].basis == "judged"],  # decided units
    frames["phase_turn_votes"],  # the member ballots behind them
    frames["phases"],  # the verifier review records
    "turn",  # unit column
    "phase",  # label column
)  # per-label confidence / agreement / provenance breakdown
```

> [!IMPORTANT]
> These are reliability measures (repeatability, agreement,
> coverage), never correctness: high agreement can be judges
> sharing a bias.

Custom scanners built on `transect.cohort_llm_scanner` can use
these reliability functions out of the box.

## Custom layers

`extra_layers` injects your own additions into a run. The runnable
worked example is `examples/transect_custom_layer.py`; the shape:

```python
from inspect_scout import scanner
from transect import Layer, cohort_llm_scanner, reasoning_turns, transect, turns_frame
from transect.report import Markdown, TurnBand

SKILLS = {
    "hypothesis": "Proposing a new idea, mechanism, or approach to test.",
    # ...
    "none_of_the_above": "No research-skill reasoning in this turn.",
}


@scanner(loader=reasoning_turns())  # the units to judge: one item per reasoning turn
def research_skills(judge_models=None):
    return cohort_llm_scanner(
        question=QUESTION,  # the closed-vocab judge prompt
        answer=list(SKILLS),
        models=judge_models,
        vocabulary=SKILLS,  # recorded as the layer's rubric
    )


results = transect(
    logs,
    spec,
    judge_models=...,
    extra_layers=[
        Layer(
            name="research_skills",
            # a judged classification over your own vocabulary
            scanner=research_skills,
            # its dataframe, mounted at results.layer_frames[name]
            frame=turns_frame,
            # phase-card chips + a token-spend grouping
            # (tag family -> frame column)
            tags={"skill": "label"},
            # an entity block in the reliability audit
            # ((unit column, label column))
            audit=("turn", "label"),
            # the layer's own report section
            section=[
                Markdown("#### One skill label per reasoning turn"),
                TurnBand(label="label"),
            ],
        )
    ],
)
results.layer_frames["research_skills"]  # the layer's own frame
results.turn_tags  # tag families, per turn
```

Each `Layer` field is one surface, all optional:

- **scanner** - `cohort_llm_scanner` inherits the judge regimes,
  voting, verifier, and batching; structural scanners are $0.
- **frame** - post-processes the scan results into the layer's
  dataframe (`turns_frame`: one judged row per turn), or injects a
  ready dataframe with your own data.
- **section / tags / audit** - typed report blocks, phase-card
  chips plus the spend grouping, and an audit entity block.

> [!TIP]
> The full authoring recipe is the shipped `add-a-layer` skill.

## Cost

Structural extraction is free (no LLM calls): without
`judge_models`, or without the matching Spec vocabulary, only the
structural scanners run and the whole scan costs nothing. Judged
surfaces cost roughly (turns + sub-agents) x judges x rolls calls
per transcript; the demo example costs cents.

Batching is the cost lever on long runs: `reasoning_turns(batch=N)`
with `cohort_llm_scanner(batch=True)` judges N units per call,
cutting a layer's classification calls. Each batched call returns
one answer per unit, so votes are still counted per unit and the
verifier still reviews individual units.

Two caches sit at different levels. The scan store (`scans_dir`)
holds a finished scan's results: `load()` rebuilds the frames and
the report from it without any model calls. Beneath it, inspect-ai
keeps a machine-global response cache (shared across projects and
venvs): a new `transect()` call always re-runs the scan, but any
judge call identical to a cached one replays from disk, so
repeating an unchanged analysis costs almost nothing. For a
stability study or a genuinely cold run, bypass or clear the
response cache:

```bash
INSPECT_CACHE_DIR=$(mktemp -d) python my_analysis.py  # bypass, one run
inspect cache clear                                   # or wipe it
```

## Development

```bash
uv sync        # install the package + dev dependencies
make lint      # ruff check + format check
make typecheck # mypy + pyright
make test      # pytest
make check     # all of the above
```

Tests are plain pytest under `tests/`. One test loads the rendered
report in a real browser to catch chart errors; it needs the
optional `ui-test` dependency group (not installed by default) and
skips without it:

```bash
uv run --group ui-test playwright install chromium  # one-time
uv run --group ui-test pytest
```

If you work on this repo with Claude Code (or another coding agent),
see [AGENTS.md](AGENTS.md). For work against the wider Inspect
ecosystem's APIs, the
[Meridian inspect-skills plugin](https://github.com/meridianlabs-ai/inspect-skills)
keeps a coding agent on current API docs. Install it once with
`/plugin marketplace add meridianlabs-ai/inspect-skills` then
`/plugin install inspect-skills@meridian`.

## Acknowledgements

Built on the Inspect ecosystem: [Inspect
AI](https://inspect.aisi.org.uk/) (eval framework and logs),
[Inspect Scout](https://meridianlabs-ai.github.io/inspect_scout/)
(transcript scanning), and
[Inspect Viz](https://meridianlabs-ai.github.io/inspect_viz/)
(report charts).

## Citation

A paper describing the method is forthcoming.
