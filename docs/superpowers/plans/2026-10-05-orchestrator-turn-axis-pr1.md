# Orchestrator Turn Axis, PR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `turn` mean orchestrator turn everywhere (scanners, frames, judge digests, charts), place sub-agent spans on that axis by wall-clock, and keep the report rendering; this is PR 1 of the spec's four.

**Architecture:** One enumeration (`helpers.orchestrator_turns`) defines the axis; every scanner numbers by it and the all-lane iterator becomes internal. The token-timeline scanner stores per-turn timestamps and a per-span record; a small pure module (`frames/spine.py`) maps timestamps onto turn cells; the subagents frame carries the derived coordinates and every other frame drops sub-agent rows from the turn axis. The report reads the new columns with its existing charts.

**Tech Stack:** Python 3.12, inspect_ai 0.3.268, inspect_scout 0.5.3, pandas, inspect_viz, pytest (mockllm, no API calls), uv, ruff, mypy + pyright.

**Spec:** `docs/superpowers/specs/2026-10-05-orchestrator-turn-axis-design.md`

## Global Constraints

- No backward compatibility with existing scan stores; committed fixture stores are regenerated (spec, preamble).
- `turn` is the 0-based ordinal of the orchestrator's model events with output, in event order; nothing else is called a turn and no all-lane index exists anywhere (spec §1).
- Main lane = the span Scout's `timeline_messages(..., depth=1)` would scan; an ambiguous or empty tree raises `ValueError` naming the candidate spans; no fallback (spec §1).
- `pos(t) = m - 0.5 + (t - t_m) / (t_(m+1) - t_m)`; last cell's right edge = last call's `completed`, else `timestamp` + median cell width; clamp to `-0.5` and the axis right edge; a zero-width cell is skipped (a time at its instant belongs to the next non-empty cell); wall-clock `timestamp`, never working time (spec §3).
- Phase membership for sub-agent counts and delegated spend uses `spawn_turn`, never the timestamp anchor (spec §3).
- `SCHEMA_VERSION` bumps; `pyproject.toml` bumps minor (`0.1.16` -> `0.2.0`) because `src/transect` changes and the frame contract changes (spec §4, §8; CI's version-bump job).
- Repo style: comments state constraints only, no em dashes, no `--`, no all-caps emphasis; mypy and pyright both pass; `make check` is the gate and its exit status must never be masked behind a pipe (AGENTS.md; memory `transect-ci-workflow-lessons`).
- Tests: plain pytest, parametrised, one-sentence docstrings, deterministic (mockllm), $0 (AGENTS.md).
- Frames read columns directly, never behind `in frame.columns`; empty frames carry the full column set (AGENTS.md).

## Review Focus

1. **A transcript whose orchestrator never ran a model call** (an errored sample, or a timeline whose root holds two solver agents). Expected: the scan records a per-transcript error naming the spans, the report's scan status shows it, nothing renders an empty axis as if it were a run. Pinned in Task 1 (ambiguity/empty raise) and Task 11 (scan status surfaces the error).
2. **Sub-agent activity after the orchestrator's last turn** (a background agent still running at submit). Expected: the span clamps to the axis's right edge and its tooltip says activity continued after the last orchestrator turn, no crash, no off-canvas bar. Pinned in Task 4 (`position` clamps) and Task 5 (`after_last` flag on the frame).
3. **Events without timestamps** (a hand-built transcript, a store written by an importer that drops them). Expected: `position_source == "event_order"`, spans draw as ticks at `spawn_turn`, nothing divides by a missing value. Pinned in Task 4 and Task 5.
4. **An Inspect handoff transcript's message thread carrying sub-agent assistant messages** (every `.eval` with handoffs). Expected: interventions still land on the orchestrator turn whose input saw them, not shifted by the sub-agent's messages. Pinned in Task 7.
5. **Two orchestrator model calls with identical timestamps** (cached generates, replayed outputs). Expected: the zero-width cell is skipped, a time at that instant lands at the next cell's left edge, and later cells are unaffected. Pinned in Task 4.

---

### Task 1: The orchestrator enumeration and strict main-lane resolution

**Files:**
- Modify: `src/transect/scanners/helpers.py:30-37` (`model_turns`), `:182-240` (`main_lane_id`, `main_span`)
- Modify: `pyproject.toml:3` (version)
- Test: `tests/test_turn_axis.py` (create)

**Interfaces:**
- Produces: `orchestrator_turns(transcript) -> Iterator[tuple[int, ModelEvent, list[ToolCall]]]` yielding `(turn, event, tool_calls)` for main-lane model events with output, in event order.
- Produces: `all_model_turns(transcript) -> Iterator[tuple[Any, list[ToolCall]]]` (the renamed `model_turns`, every lane, no numbering; internal use only).
- Produces: `main_span(transcript) -> TimelineSpan` now raising `ValueError` when the resolved span has no direct model events, or when a container has two or more non-utility agent children carrying model events.
- Removes: `main_lane_id` (callers switch to `main_span(transcript).id`).

- [ ] **Step 1: Bump the version**

In `pyproject.toml` change `version = "0.1.16"` to `version = "0.2.0"`.

- [ ] **Step 2: Write the failing tests**

```python
"""The orchestrator turn axis: one enumeration defines turn numbers."""

import pytest
from helpers import StubTranscript, agent_span, model_turn

from transect.scanners.helpers import all_model_turns, main_span, orchestrator_turns


def handoff_shape():
    """Lead agent span holding two sub-agent spans: the .eval handoff shape."""
    return [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead 0"),
                *agent_span("A", "eda", inner=[model_turn("a0"), model_turn("a1")], parent_id="R"),
                model_turn("lead 1"),
                *agent_span("B", "rev", inner=[model_turn("b0")], parent_id="R"),
                model_turn("lead 2"),
            ],
        )
    ]


def test_orchestrator_turns_number_main_lane_turns_contiguously():
    """Sub-agent turns consume no turn numbers."""
    turns = list(orchestrator_turns(StubTranscript(handoff_shape())))
    assert [(t, e.output.message.text) for t, e, _ in turns] == [
        (0, "lead 0"),
        (1, "lead 1"),
        (2, "lead 2"),
    ]


def test_all_model_turns_sees_every_lane_unnumbered():
    """The internal iterator yields six events, in event order."""
    texts = [e.output.message.text for e, _ in all_model_turns(StubTranscript(handoff_shape()))]
    assert texts == ["lead 0", "a0", "a1", "lead 1", "b0", "lead 2"]


def test_a_bare_transcript_is_its_own_orchestrator():
    """Span-less model events form the main lane."""
    events = [model_turn("x"), model_turn("y")]
    assert [t for t, _, _ in orchestrator_turns(StubTranscript(events))] == [0, 1]


@pytest.mark.parametrize(
    ("events", "match"),
    [
        (
            [
                *agent_span("A", "one", inner=[model_turn("a")]),
                *agent_span("B", "two", inner=[model_turn("b")]),
            ],
            r"two top-level agents.*one.*two",
        ),
        ([*agent_span("A", "one", inner=[])], r"no model turns"),
    ],
    ids=["two-top-level-agents", "no-model-turns"],
)
def test_main_span_refuses_an_ambiguous_or_empty_tree(events, match):
    """No single orchestrator means a loud error, never an empty axis."""
    with pytest.raises(ValueError, match=match):
        main_span(StubTranscript(events))
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_turn_axis.py -q`
Expected: ImportError on `all_model_turns` / `orchestrator_turns`.

- [ ] **Step 4: Implement**

In `helpers.py` rename `model_turns` to `all_model_turns` with docstring "Yield (model event, its tool calls) for every model turn in every lane, in event order. Internal: the turn axis is `orchestrator_turns`." Add:

```python
def orchestrator_turns(
    transcript: Transcript | Any,
) -> Iterator[tuple[int, Any, list[ToolCall]]]:
    """Yield ``(turn, model event, tool calls)`` for the orchestrator's turns.

    The orchestrator is the main lane (`main_span`); ``turn`` is the
    0-based ordinal of its model events with output, in event order.
    This enumeration is the turn axis: every scanner, frame and chart
    numbers by it, and a custom scanner must too.
    """
    main = main_span(transcript)
    main_ids = {
        id(item.event)
        for item in main.content
        if isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
    }
    turn = 0
    for event, calls in all_model_turns(transcript):
        if id(event) not in main_ids:
            continue
        yield turn, event, calls
        turn += 1
```

Replace `main_lane_id` with nothing (delete it). In `main_span`, replace the walk's two exits:

```python
    span = timeline.root
    while True:
        if any(
            isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
            for item in span.content
        ):
            return span
        children = [
            item
            for item in span.content
            if isinstance(item, TimelineSpan)
            and item.span_type == "agent"
            and not item.utility
            and _holds_model_events(item)
        ]
        if len(children) == 1:
            span = children[0]
            continue
        if not children:
            raise ValueError(
                f"transcript has no model turns under its main span {span.name!r}; "
                "there is no orchestrator lane to number"
            )
        names = ", ".join(repr(c.name) for c in children)
        raise ValueError(
            f"transcript has {len(children)} top-level agents ({names}) and no "
            "single orchestrator lane; transect numbers one orchestrator's turns"
        )
```

with

```python
def _holds_model_events(span: TimelineSpan) -> bool:
    """Whether any model event with output lives in this span's subtree."""
    for item in span.content:
        if isinstance(item, TimelineEvent):
            if isinstance(item.event, ModelEvent) and item.event.output:
                return True
        elif _holds_model_events(item):
            return True
    return False
```

Docstring of `main_span`: "Resolve the orchestrator's timeline span: the span Scout's `timeline_messages(..., depth=1)` would scan - the outermost non-utility span holding a direct model event, container spans (the synthetic root, a solvers wrapper) being transparent. Raises ValueError when no such span exists or two or more do."

Update every importer of `model_turns` / `main_lane_id` (`scanners/base.py`, `scanners/phases.py`, `report/excerpts.py`) to the new names for now; later tasks change their bodies.

- [ ] **Step 5: Run the new tests and the suite**

Run: `uv run pytest tests/test_turn_axis.py -q` then `uv run pytest -q -x`
Expected: new tests pass; other suites may still pass since bodies are unchanged except the raise path.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/transect/scanners/helpers.py src/transect/scanners/base.py src/transect/scanners/phases.py src/transect/report/excerpts.py tests/test_turn_axis.py
git commit -m "feat: orchestrator_turns defines the turn axis; main span resolution is strict"
```

---

### Task 2: token_timeline stores orchestrator turns, lane turns, timestamps and a span record

**Files:**
- Modify: `src/transect/scanners/base.py:22-62` (`token_timeline`), `:340-434` (`_span_ends`, `_lane_activity`, `_non_model_events`, `_sub_agent_span`)
- Test: `tests/test_scanners_mechanical.py:21-112`

**Interfaces:**
- Produces: `value["timeline"]` entries `{turn: int|None, lane_turn: int, agent_lane, agent_span_id, n_tool_calls, timestamp: str|None, completed: str|None, <usage fields>}`.
- Produces: `value["spans"]` entries `{agent_span_id, agent_lane, spawn_turn: int, first_at, last_at, end_at, end_recorded: bool, event_order_end_turn: int|None}`.
- Produces: `value["lane_activity"]` entries as today with `turn` = orchestrator turn preceding the tool event (`max(count - 1, 0)`).
- Produces: `_orchestrator_count_before(transcript) -> Iterator[tuple[int, Event]]` yielding `(orchestrator turns so far, event)` for every non-orchestrator-model event (replaces `_non_model_events`; used by Tasks 6 and 7).

- [ ] **Step 1: Rewrite the mechanical tests for the new value**

Replace the five token-timeline tests with:

```python
def test_token_timeline_numbers_orchestrator_turns_and_lane_turns():
    """Main-lane turns take 0,1,2; a sub-agent's turns take no turn but
    count within their own lane; timestamps ride along."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead", usage=usage(100, 10)),
                *agent_span(
                    "C", "eda", inner=[model_turn("s0"), model_turn("s1")], parent_id="R"
                ),
                model_turn("lead again", usage=usage(300, 30)),
            ],
        ),
    ]
    value = run_scan(token_timeline(), events).value
    rows = [(e["turn"], e["lane_turn"], e["agent_lane"]) for e in value["timeline"]]
    assert rows == [(0, 0, None), (None, 0, "eda"), (None, 1, "eda"), (1, 1, None)]
    assert value["timeline"][0]["input_tokens"] == 100
    assert value["timeline"][1]["input_tokens"] is None
    assert all(isinstance(e["timestamp"], str) for e in value["timeline"])


def test_span_record_anchors_spawn_and_end_by_orchestrator_turn():
    """Each sub-agent span records the orchestrator turn before its begin
    and before its end, its activity timestamps, and a recorded end."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead"),
                *agent_span("C", "eda", inner=[model_turn("sub"), tool_event("t", span_id="C")], parent_id="R"),
                model_turn("wrap"),
            ],
        ),
    ]
    (span,) = run_scan(token_timeline(), events).value["spans"]
    assert (span["agent_span_id"], span["agent_lane"]) == ("C", "eda")
    assert (span["spawn_turn"], span["event_order_end_turn"]) == (0, 0)
    assert span["end_recorded"] is True
    assert span["first_at"] <= span["last_at"] <= span["end_at"]


def test_a_span_begun_before_any_orchestrator_turn_anchors_at_zero():
    """A sub-agent spawned before the first main turn still gets a turn."""
    events = [
        *agent_span("C", "eda", inner=[model_turn("sub")]),
        model_turn("lead"),
    ]
    (span,) = run_scan(token_timeline(), events).value["spans"]
    assert span["spawn_turn"] == 0


def test_lane_activity_counts_tool_events_per_orchestrator_turn():
    """Tool-only sub-agents surface in lane_activity, anchored to the
    orchestrator turn that preceded the tool event."""
    events = [
        model_turn("orchestrator"),
        *agent_span("A", "worker", inner=[tool_event("t1", span_id="A")]),
    ]
    value = run_scan(token_timeline(), events).value
    (row,) = value["lane_activity"]
    assert (row["agent_lane"], row["turn"], row["tool_calls"]) == ("worker", 0, 1)


def test_lane_activity_excludes_the_main_lane_and_folded_spawn_calls():
    (unchanged body)
```

Keep the existing exclusion test as is. Drop `test_span_ends_record_sub_agent_completions_only` (its fact moves to the span record test).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_scanners_mechanical.py -q -k "token_timeline or span_record or lane_activity or anchors_at_zero"`
Expected: KeyError `lane_turn` / `spans`.

- [ ] **Step 3: Implement the scanner**

```python
@scanner(events=["model", "span_begin", "span_end", "tool"])
def token_timeline() -> Scanner[Transcript]:
    """Per-model-turn token usage on the orchestrator turn axis, plus
    the sub-agent span record.

    value = {"timeline": [...], "spans": [...], "lane_activity": [...]}.

    ``timeline``: one entry per model turn in any lane, in event order:
    ``turn`` (the orchestrator ordinal; None on a sub-agent turn),
    ``lane_turn`` (0-based within the turn's own lane), ``agent_lane`` /
    ``agent_span_id`` (None on the orchestrator), ``n_tool_calls``,
    ``timestamp`` / ``completed`` (ISO strings; None when unrecorded),
    and the ModelUsage fields (None = not reported, never 0).

    ``spans``: one entry per sub-agent span: ``spawn_turn`` (the
    orchestrator turn preceding the span_begin in event order, 0 when
    none does), ``first_at`` / ``last_at`` (first and last model or tool
    event timestamps), ``end_at`` and ``end_recorded`` (the span_end's
    timestamp; OpenClaw synthesises ends, so recorded is False there),
    ``event_order_end_turn`` (the orchestrator turn preceding the
    span_end; None when the span never closed).

    ``lane_activity``: tool events inside sub-agent spans per
    (orchestrator turn preceding the event, span): calls, busy time,
    first start.
    """

    async def execute(transcript: Transcript) -> Result:
        spans = {e.id: e for e in transcript.events if e.event == "span_begin"}
        main = main_span(transcript)
        main_id = main.id
        main_events = {
            id(item.event)
            for item in main.content
            if isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
        }
        timeline: list[dict[str, Any]] = []
        lane_counts: dict[str, int] = {}
        turn = 0
        for event, calls in all_model_turns(transcript):
            agent_span = _sub_agent_span(spans, getattr(event, "span_id", None), main_id)
            lane_key = agent_span.id if agent_span else "__main__"
            lane_turn = lane_counts.get(lane_key, 0)
            lane_counts[lane_key] = lane_turn + 1
            is_main = id(event) in main_events
            usage = event.output.usage
            entry: dict[str, Any] = {
                "turn": turn if is_main else None,
                "lane_turn": lane_turn,
                "agent_lane": agent_span.name if agent_span else None,
                "agent_span_id": agent_span.id if agent_span else None,
                "n_tool_calls": len(calls),
                "timestamp": _iso(getattr(event, "timestamp", None)),
                "completed": _iso(getattr(event, "completed", None)),
            }
            for field in _USAGE_FIELDS:
                entry[field] = getattr(usage, field, None) if usage else None
            timeline.append(entry)
            if is_main:
                turn += 1
        return Result(
            value={
                "timeline": cast(JsonValue, timeline),
                "spans": cast(JsonValue, _span_records(transcript, spans, main_id, main_events)),
                "lane_activity": cast(JsonValue, _lane_activity(transcript, spans, main_id, main_events)),
            },
            explanation=f"{turn} orchestrator turns, {len(timeline) - turn} sub-agent turns",
        )

    return execute
```

A model event that is neither in the main span nor in a sub-agent span (an init or scorer call on a `.eval`) gets `turn=None` and no lane: it is off the axis, which is the honest reading. Note that in the docstring.

Helpers:

```python
def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _orchestrator_count_before(
    transcript: Transcript, main_events: set[int]
) -> Iterator[tuple[int, Any]]:
    """Yield (orchestrator turns so far, event) for every event that is
    not itself an orchestrator model turn."""
    count = 0
    for event in transcript.events:
        if event.event == "model" and event.output and id(event) in main_events:
            count += 1
            continue
        yield count, event


def _span_records(transcript, spans, main_id, main_events) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for before, event in _orchestrator_count_before(transcript, main_events):
        kind = event.event
        if kind == "span_begin":
            if getattr(event, "type", None) != "agent" or event.id == main_id:
                continue
            records[event.id] = {
                "agent_span_id": event.id,
                "agent_lane": event.name,
                "spawn_turn": max(before - 1, 0),
                "first_at": None,
                "last_at": None,
                "end_at": None,
                "end_recorded": False,
                "event_order_end_turn": None,
            }
            continue
        if kind == "span_end":
            record = records.get(getattr(event, "id", None))
            if record is not None:
                record["end_at"] = _iso(getattr(event, "timestamp", None))
                record["end_recorded"] = transcript.source_type != "openclaw"  # see note
                record["event_order_end_turn"] = max(before - 1, 0)
            continue
        if kind not in ("model", "tool"):
            continue
        if kind == "tool" and getattr(event, "agent_span_id", None) is not None:
            continue  # folded spawn call: the orchestrator's
        agent_span = _sub_agent_span(spans, getattr(event, "span_id", None), main_id)
        if agent_span is None or agent_span.id not in records:
            continue
        record = records[agent_span.id]
        started = getattr(event, "timestamp", None)
        finished = getattr(event, "completed", None) or started
        if started is not None and (record["first_at"] is None or started.isoformat() < record["first_at"]):
            record["first_at"] = started.isoformat()
        if finished is not None and (record["last_at"] is None or finished.isoformat() > record["last_at"]):
            record["last_at"] = finished.isoformat()
    return list(records.values())
```

Note on `end_recorded`: keep the existing rule from `frames/subagents.py` (`agent != "openclaw"`) but move it here: the OpenClaw import's transcripts have `source_type == "openclaw"`? Check what `Transcript.source_type` the importer stamps (grep `source_type` in `ingestion/openclaw_telemetry_hal/transcripts.py`); if it is not distinguishable there, read `transcript.metadata`/`agent` the way `identity()` does and record `end_recorded = agent != "openclaw"`. Pin the rule in the test below once determined. Nested spans (a sub-agent inside a sub-agent) are attributed via `nearest_agent_span`, so a nested span gets its own record and its parent does not absorb its events.

`_lane_activity` keeps its body but iterates `_orchestrator_count_before(transcript, main_events)` and anchors at `max(before - 1, 0)`. Delete `_span_ends` and `_non_model_events`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_scanners_mechanical.py -q`
Expected: the flush and intervention tests may now fail (they used `_non_model_events`); switch those scanners' call sites to `_orchestrator_count_before(transcript, main_events)` with `main_events` computed once per scanner the same way (Tasks 6 and 7 finish their semantics). Token tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/transect/scanners/base.py tests/test_scanners_mechanical.py
git commit -m "feat: token_timeline numbers orchestrator turns and records sub-agent spans"
```

---

### Task 3: token_timeline and lane_activity frames

**Files:**
- Modify: `src/transect/frames/token_timeline.py` (docstring, columns, dtypes), `src/transect/frames/lane_activity.py` (span end from `spans`), `src/transect/frames/common.py:27` (`SCHEMA_VERSION = "0.2"`)
- Test: `tests/test_frames.py:149-163`, `:258-277`

**Interfaces:**
- Produces: `token_timeline` columns `[*IDENTITY_COLS, turn (Int64, NA on sub-agent rows), lane_turn (Int64), agent_lane, agent_span_id, timestamp (string), n_tool_calls, <token fields>, <derived>]`.
- Produces: `lane_activity.span_end_turn` from `spans[].event_order_end_turn`.

- [ ] **Step 1: Update the pinned demo tests**

```python
def test_token_timeline_numbers_orchestrator_turns_only(demo_results):
    """Ten orchestrator turns take 0..9; the seven sub-agent turns carry
    no turn but keep their lane and token views; totals are unchanged."""
    timeline = demo_results.token_timeline
    main = timeline[timeline.turn.notna()].sort_values("turn")
    assert main.turn.tolist() == list(range(10))
    assert timeline.turn.isna().sum() == 7
    assert set(timeline.agent_lane.dropna()) == {"eda", "alt_model", "reviewer"}
    spots = main.set_index("turn")
    assert (spots.loc[0, "new_work"], spots.loc[0, "context"]) == (1020, 926)
    # turn 6 is the summarization call; turn 7 runs on the compacted window
    assert (spots.loc[6, "new_work"], spots.loc[6, "context"]) == (3588, 2667)
    assert (spots.loc[7, "new_work"], spots.loc[7, "context"]) == (1491, 1228)
    assert (spots.loc[9, "new_work"], spots.loc[9, "context"]) == (2005, 1761)
    assert timeline.new_work.sum() == 27221
    eda = timeline[timeline.agent_lane == "eda"]
    assert eda.lane_turn.tolist() == [0, 1]
```

and

```python
def test_lane_activity_frame_anchors_tool_activity_to_orchestrator_turns(demo_results):
    """Each sub-agent's tool events collapse onto the orchestrator turn
    that spawned it (the handoff turn); the main lane never appears."""
    expected = pd.DataFrame(
        [(3, "eda", 2, 3), (5, "alt_model", 2, 5), (8, "reviewer", 3, 8)],
        columns=["turn", "agent_lane", "tool_calls", "span_end_turn"],
    )
    got = demo_results.lane_activity.sort_values("turn")
    pd.testing.assert_frame_equal(
        got[expected.columns].reset_index(drop=True), expected, check_dtype=False
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_frames.py -q -k "orchestrator_turns_only or anchors_tool_activity"`

- [ ] **Step 3: Implement**

`token_timeline.py`: add `"lane_turn"` and `"timestamp"` to the row and the column list (after `agent_span_id`), docstring entries:

```
- turn: 0-based orchestrator turn; NA on a sub-agent's own turns (they
  are off the axis) and on init/scorer calls.
- lane_turn: 0-based ordinal within the turn's own lane.
- timestamp: the model call's start, ISO 8601; None when unrecorded.
```

dtypes: `df = df.astype({"turn": "Int64", "lane_turn": "Int64", "timestamp": "string"})`. `_derive_token_views` merges on `["transcript_id", "turn"]`; change the merge key to a positional index: add `derived_rows` with the row's original index (`t.name`) and merge on it, since sub-agent rows share `turn = NA`. Keep per-lane grouping.

`lane_activity.py`: `end_of = {s["agent_span_id"]: s["event_order_end_turn"] for s in value.get("spans") or []}`; docstring `turn`: "0-based orchestrator turn preceding the tool event".

`common.py`: `SCHEMA_VERSION = "0.2"`.

- [ ] **Step 4: Run the frame tests**

Run: `uv run pytest tests/test_frames.py -q -k "token_timeline or lane_activity or identity_and_schema"`
Expected: pass (other demo-pinned tests still red until later tasks).

- [ ] **Step 5: Commit**

```bash
git add src/transect/frames/token_timeline.py src/transect/frames/lane_activity.py src/transect/frames/common.py tests/test_frames.py
git commit -m "feat: token_timeline frame carries orchestrator turns, lane turns and timestamps"
```

---

### Task 4: `frames/spine.py`: timestamps onto turn cells

**Files:**
- Create: `src/transect/frames/spine.py`
- Test: `tests/test_turn_axis.py` (append)

**Interfaces:**
- Produces: `cells(starts: list[datetime], last_completed: datetime | None) -> list[tuple[float, float]]` returning per orchestrator turn `(t_m, t_right)` as POSIX seconds.
- Produces: `position(t: datetime, cells: list[tuple[float, float]]) -> float`.
- Produces: `coordinates(record: dict, cells) -> dict` with keys `start_pos, end_pos, anchor_turn, end_turn, position_source, after_last`.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timedelta

from transect.frames import spine

T0 = datetime(2026, 1, 1, 10, 0, 0)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def test_cells_cover_each_turn_until_the_next_call_starts():
    """Cell m runs from call m's start to call m+1's start; the last cell
    ends at the last call's completion."""
    got = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    assert got == [(at(0).timestamp(), at(100).timestamp()), (at(100).timestamp(), at(150).timestamp()), (at(150).timestamp(), at(170).timestamp())]


def test_last_cell_without_completion_uses_the_median_width():
    """No completion recorded: the last cell is median-width wide."""
    got = spine.cells([at(0), at(100), at(150)], last_completed=None)
    assert got[-1] == (at(150).timestamp(), at(225).timestamp())


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (at(-5), -0.5),
        (at(0), -0.5),
        (at(50), 0.0),
        (at(100), 0.5),
        (at(125), 1.0),
        (at(160), 2.0),
        (at(999), 2.5),
    ],
    ids=["before-first", "first-start", "mid-first", "second-start", "mid-second", "mid-last", "after-last"],
)
def test_position_interpolates_within_a_cell_and_clamps_outside(t, expected):
    """A timestamp maps to m - 0.5 plus its fraction of cell m."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    assert spine.position(t, cells) == pytest.approx(expected)


def test_zero_width_cells_place_at_their_left_edge():
    """Two calls at the same instant: the empty cell swallows nothing."""
    cells = spine.cells([at(0), at(0), at(100)], last_completed=at(120))
    assert spine.position(at(0), cells) == pytest.approx(-0.5)
    assert spine.position(at(50), cells) == pytest.approx(2.0)


def test_coordinates_follow_timestamps_when_present():
    """A span's box runs from its first activity to its recorded end."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    record = {"spawn_turn": 0, "first_at": at(20).isoformat(), "last_at": at(90).isoformat(), "end_at": at(110).isoformat(), "end_recorded": True, "event_order_end_turn": 1}
    got = spine.coordinates(record, cells)
    assert got["position_source"] == "timestamp"
    assert (got["anchor_turn"], got["end_turn"]) == (0, 1)
    assert got["start_pos"] == pytest.approx(-0.3)
    assert got["end_pos"] == pytest.approx(0.7)
    assert got["after_last"] is False


def test_coordinates_fall_back_to_event_order_without_timestamps():
    """Missing timestamps collapse the span onto its spawn turn."""
    cells = spine.cells([at(0), at(100)], last_completed=None)
    record = {"spawn_turn": 1, "first_at": None, "last_at": None, "end_at": None, "end_recorded": False, "event_order_end_turn": None}
    got = spine.coordinates(record, cells)
    assert got == {"start_pos": 1.0, "end_pos": 1.0, "anchor_turn": 1, "end_turn": 1, "position_source": "event_order", "after_last": False}


def test_activity_after_the_last_turn_clamps_and_is_flagged():
    """A background agent outliving the orchestrator clamps to the edge."""
    cells = spine.cells([at(0), at(100)], last_completed=at(120))
    record = {"spawn_turn": 1, "first_at": at(105).isoformat(), "last_at": at(500).isoformat(), "end_at": None, "end_recorded": False, "event_order_end_turn": None}
    got = spine.coordinates(record, cells)
    assert got["end_pos"] == pytest.approx(1.5)
    assert got["end_turn"] == 1
    assert got["after_last"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_turn_axis.py -q`

- [ ] **Step 3: Implement**

```python
"""The orchestrator turn axis as a time scale.

Orchestrator turn ``m`` occupies the cell ``[m - 0.5, m + 0.5]``, which
stands for the wall-clock interval from the start of model call ``m``
to the start of call ``m + 1`` (the call plus whatever tool execution
followed it). Sub-agent activity is placed on the axis by mapping its
timestamps into those cells, so a span's bar shows which orchestrator
turns were active while it ran, not a turn count of its own.
"""

from datetime import datetime
from statistics import median
from typing import Any

Cells = list[tuple[float, float]]


def cells(starts: list[datetime], last_completed: datetime | None) -> Cells:
    """Per-turn ``(start, right edge)`` in POSIX seconds.

    The last cell's right edge is the last call's completion when
    recorded, else its start plus the median cell width (the start
    itself when there is one turn and no completion).
    """
    s = [t.timestamp() for t in starts]
    out: Cells = [(s[i], s[i + 1]) for i in range(len(s) - 1)]
    if not s:
        return out
    if last_completed is not None and last_completed.timestamp() > s[-1]:
        right = last_completed.timestamp()
    else:
        widths = [b - a for a, b in out if b > a]
        right = s[-1] + (median(widths) if widths else 0.0)
    out.append((s[-1], right))
    return out


def position(t: datetime, cells: Cells) -> float:
    """Map a timestamp onto the axis; clamps to its two edges."""
    if not cells:
        return -0.5
    ts = t.timestamp()
    if ts <= cells[0][0]:
        return -0.5
    if ts >= cells[-1][1]:
        return len(cells) - 0.5
    for m, (start, right) in enumerate(cells):
        if start <= ts < right:
            width = right - start
            return m - 0.5 + ((ts - start) / width if width > 0 else 0.0)
        if ts < start:  # inside a zero-width cell's shadow
            return m - 0.5
    return len(cells) - 0.5


def coordinates(record: dict[str, Any], cells: Cells) -> dict[str, Any]:
    """A span record's axis coordinates (see `frames.subagents`)."""
    spawn = int(record["spawn_turn"])
    first_at = _parse(record.get("first_at"))
    end_raw = record.get("end_at") if record.get("end_recorded") else record.get("last_at")
    end_at = _parse(end_raw) or _parse(record.get("last_at"))
    if cells and first_at is not None and end_at is not None:
        start_pos = position(first_at, cells)
        end_pos = max(position(end_at, cells), start_pos)
        return {
            "start_pos": start_pos,
            "end_pos": end_pos,
            "anchor_turn": _cell_of(start_pos, cells),
            "end_turn": _cell_of(end_pos, cells),
            "position_source": "timestamp",
            "after_last": end_at.timestamp() > cells[-1][1],
        }
    end_turn = record.get("event_order_end_turn")
    return {
        "start_pos": float(spawn),
        "end_pos": float(spawn),
        "anchor_turn": spawn,
        "end_turn": int(end_turn) if end_turn is not None else spawn,
        "position_source": "event_order",
        "after_last": False,
    }


def _cell_of(pos: float, cells: Cells) -> int:
    return min(max(int(pos + 0.5), 0), len(cells) - 1)


def _parse(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None
```

Check the `end_turn` in the fallback: the spec says `end_turn = event_order_end_turn`; keep `spawn` when it is None.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_turn_axis.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/transect/frames/spine.py tests/test_turn_axis.py
git commit -m "feat: map timestamps onto orchestrator turn cells"
```

---

### Task 5: subagents frame carries the axis coordinates

**Files:**
- Modify: `src/transect/frames/subagents.py` (docstring, `subagents_df`, `_structural_spine`, `_SPINE_DTYPES`)
- Modify: `src/transect/frames/__init__.py` if it re-exports names
- Test: `tests/test_frames.py:165-196`, `:346-372`

**Interfaces:**
- Consumes: `spine.cells`, `spine.coordinates`; `token_timeline` rows (`turn`, `timestamp`, `completed` is not in the frame: read the last call's completion from the raw `timeline` entry instead, see below).
- Produces: `subagents_df(results, token_timeline_raw: pd.DataFrame, lane_activity, token_timeline)` where `token_timeline_raw` is the raw scanner results table (the `spans` record lives in the value). Columns replace `span_start_turn/span_last_turn/span_end_turn/span_end_recorded` with `spawn_turn (Int64), anchor_turn (Int64), end_turn (Int64), start_pos (float64), end_pos (float64), position_source (string), end_recorded (bool), after_last (bool), started_at (string), ended_at (string)`.

- [ ] **Step 1: Rewrite the demo pin**

```python
def test_subagent_spans_are_placed_on_the_orchestrator_axis(demo_results):
    """Each handoff span anchors in its transfer turn's cell by wall
    clock, spawns at that turn, and keeps its per-span spend."""
    subagents = demo_results.frames()["subagents"]
    assert subagents.label.isna().all()
    expected = pd.DataFrame(
        [
            ("eda", 3, 3, 3, "timestamp", True, 2, 1750.0),
            ("alt_model", 5, 5, 5, "timestamp", True, 2, 2701.0),
            ("reviewer", 8, 8, 8, "timestamp", True, 3, 6242.0),
        ],
        columns=["agent_lane", "spawn_turn", "anchor_turn", "end_turn", "position_source", "end_recorded", "tool_calls", "new_work"],
    )
    pd.testing.assert_frame_equal(subagents[expected.columns].reset_index(drop=True), expected, check_dtype=False)
    # the box sits inside its cell: between the transfer turn's start and the next turn's
    for row in subagents.itertuples():
        assert row.anchor_turn - 0.5 <= row.start_pos < row.end_pos <= row.end_turn + 0.5
    assert subagents.after_last.tolist() == [False, False, False]
    timeline = demo_results.token_timeline
    lane_sums = timeline.groupby("agent_lane").new_work.sum()
    assert dict(zip(subagents.agent_lane, subagents.new_work, strict=True)) == dict(lane_sums)
```

Adjust `test_subagents_frame_maps_ballots_and_the_vote` to pass the extra argument (`pd.DataFrame()` for the raw timeline) and `test_frame_builders_tolerate_empty_results` likewise.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_frames.py -q -k subagent`

- [ ] **Step 3: Implement**

`_structural_spine(token_timeline_raw, lane_activity, token_timeline)`:

1. Build per-transcript cells from the raw `timeline` entries with `turn is not None`: `starts = [parse(e["timestamp"]) ...]` in turn order (skip when any timestamp is None, which makes `cells = []` and every span event-order); `last_completed = parse(last["completed"])`.
2. For each raw `spans` entry: `coords = spine.coordinates(entry, cells)`; row = identity + `agent_span_id, agent_lane, spawn_turn, **coords, end_recorded, started_at=first_at, ended_at=end_at`.
3. Join the tool rollups from `lane_activity` (`tool_calls`, `busy_seconds`) and the token rollups from `token_timeline` sub-agent rows (as today), on `(transcript_id, agent_span_id)`, outer; a span present only in lane_activity (tool-only, no span record) cannot happen since the span record comes from span_begin events; keep the join outer anyway and fill `position_source="event_order"` from `spawn_turn = lane_activity.turn.min()` for a row with no record, so a store from a partial scan still projects.
4. Sort by `(transcript_id, start_pos, agent_span_id)`.

Dtypes: `spawn_turn/anchor_turn/end_turn` Int64; `start_pos/end_pos` float64; `position_source/started_at/ended_at` string; `end_recorded/after_last` bool (default False).

`subagents_df` signature gains `token_timeline_raw: pd.DataFrame | None = None` as the second argument; `api.load` passes `raw`. Docstring column list:

```
- spawn_turn: the orchestrator turn preceding the span's begin event
  (the turn that issued the spawn when the source links them, the
  importer's file-order anchor otherwise). Phase membership for the
  sub-agent count and the delegated spend uses this.
- anchor_turn / end_turn: the orchestrator turns whose cells hold the
  span's first activity and its end (frames.spine).
- start_pos / end_pos: the same as axis positions (cell m is
  [m - 0.5, m + 0.5]); the swimlane box.
- position_source: timestamp (positions follow wall clock) or
  event_order (no usable timestamps: the span sits at spawn_turn).
- end_recorded: the source recorded the span's end (False on OpenClaw,
  whose ends are synthesised at last activity).
- after_last: activity continued after the last orchestrator turn; the
  box is clamped to the axis edge.
- started_at / ended_at: ISO timestamps of first activity and end.
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_frames.py -q -k subagent`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/transect/frames/subagents.py src/transect/api.py tests/test_frames.py
git commit -m "feat: subagents frame places spans on the orchestrator axis"
```

---

### Task 6: Flushes on the orchestrator axis with a lane column

**Files:**
- Modify: `src/transect/scanners/base.py` (`context_flush`), `src/transect/frames/flushes.py`
- Test: `tests/test_scanners_mechanical.py:115-148`, `tests/test_frames.py:198-256`, `:473-560`

**Interfaces:**
- Produces: flush entries gain `agent_span_id` (None on the orchestrator); `turn` counts orchestrator turns before the event.
- Produces: `flushes_df` column `agent_span_id` (string) after `turn`; inference and synthesis over orchestrator rows (`token_timeline.turn.notna()`).

- [ ] **Step 1: Tests**

Mechanical: add

```python
def test_a_sub_agent_compaction_is_kept_with_its_lane():
    """A compaction inside a sub-agent span is a flush row on the
    orchestrator axis that names its lane; the orchestrator's has none."""
    events = [
        model_turn("lead"),
        *agent_span("C", "eda", inner=[model_turn("s0"), CompactionEvent(type="summary", source="inspect", tokens_before=10, tokens_after=5, span_id="C"), model_turn("s1")]),
        CompactionEvent(type="summary", source="inspect", tokens_before=20, tokens_after=8),
        model_turn("lead 2"),
    ]
    value = run_scan(context_flush(), events).value
    assert [(f["turn"], f["agent_span_id"]) for f in value["flushes"]] == [(1, "C"), (1, None)]
```

(check `CompactionEvent`'s constructor fields against the existing flush test and adapt.) Frames: demo pin changes `turn: 11` to `turn: 7`; the two synthesis tests add `"turn": range(10)` as orchestrator rows (already `agent_span_id None`) and gain a case:

```python
def test_sub_agent_lanes_never_synthesize_flushes():
    """A context reset inside a sub-agent lane is that lane's own
    business; the orchestrator axis synthesises nothing from it."""
    timeline = pd.DataFrame({
        "transcript_id": ["tr1"] * 5,
        "turn": [0, 1, None, None, 2],
        "context": [1000, 1100, 900, 100, 1200],
        "agent_span_id": [None, None, "A", "A", None],
        "agent_lane": [None, None, "a", "a", None],
    }).astype({"turn": "Int64"})
    frame = flushes_df(pd.DataFrame(), timeline)
    assert len(frame) == 0
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Scanner: compute `main_events` as in Task 2 and iterate `_orchestrator_count_before`; each entry adds `"agent_span_id": (span.id if (span := _sub_agent_span(spans, event.span_id, main_id)) else None)`. `compaction_texts` pairing is unchanged.

Frame: `columns` gains `"agent_span_id"` after `"turn"`; per transcript `orchestrator = per_turn[per_turn.turn.notna()]`; inference reads `orchestrator[(orchestrator.turn >= f["turn"]) & orchestrator.context.notna()]`; synthesis runs `_synthesized_drops(orchestrator, nearby)` once per transcript (drop the lane groupby, `lane_series`, `main_lane_of`). Docstring: "- agent_span_id: the sub-agent lane the compaction happened in; None on the orchestrator. Charts and card tags read orchestrator rows only."

- [ ] **Step 4: Run** `uv run pytest tests/test_scanners_mechanical.py tests/test_frames.py -q -k flush`

- [ ] **Step 5: Commit** `git commit -m "feat: flushes live on the orchestrator axis and name their lane"`

---

### Task 7: Interventions on the orchestrator axis

**Files:**
- Modify: `src/transect/scanners/base.py` (`human_intervention`)
- Test: `tests/test_scanners_mechanical.py:150-420` (existing operator/input tests), `tests/test_frames.py:227-256`

**Interfaces:**
- Produces: `turn` = the first orchestrator turn whose input saw the message; input/approval events anchor by orchestrator count.

- [ ] **Step 1: Tests**

Add to the mechanical tests:

```python
def test_sub_agent_assistant_messages_do_not_advance_the_axis():
    """A handoff appends the sub-agent's messages to the thread; an
    operator note after them lands on the next orchestrator turn."""
    note = ChatMessageUser(content="steer", source="operator", id="note")
    task = ChatMessageUser(content="task", id="task")
    lead0 = model_turn("lead 0", input=[task])
    sub0 = model_turn("sub 0")
    lead1 = model_turn("lead 1", input=[task, note])
    events = [lead0, *agent_span("C", "eda", inner=[sub0]), lead1]
    messages = [task, ChatMessageAssistant(content="lead 0", id=lead0.output.message.id), ChatMessageAssistant(content="sub 0", id=sub0.output.message.id), note, ChatMessageAssistant(content="lead 1", id=lead1.output.message.id)]
    value = run_scan(human_intervention(), events, messages).value
    assert [(i["turn"], i["channel"]) for i in value["interventions"]] == [(1, "operator")]
```

Demo pins: `(11, "operator"), (16, "input_event")` become `(7, "operator"), (9, "input_event")`; the shared-axis test's `== 11` becomes `== 7` with `turns` filtered to main-lane events (use `orchestrator_turns` on a `StubTranscript(sample.events)` or compute the main span the same way).

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

In `human_intervention.execute`: `main = main_span(transcript)`; `main_events` set; `first_seen`/`turn_of_output` built with `for turn, event, _calls in orchestrator_turns(transcript)`; `sub_outputs = {e.output.message.id for e, _ in all_model_turns(transcript) if id(e) not in main_events and e.output.message.id}`; in the thread walk, `if message.role == "assistant": if message.id in sub_outputs: continue` before the advance; `summary_flushes` and the event loop use `_orchestrator_count_before(transcript, main_events)`. Docstring: "turn (the orchestrator turn the intervention precedes)".

- [ ] **Step 4: Run** `uv run pytest tests/test_scanners_mechanical.py tests/test_frames.py -q -k "intervention or operator or input or approval or axis"`

- [ ] **Step 5: Commit** `git commit -m "feat: interventions anchor to orchestrator turns"`

---

### Task 8: decision_phases numbers orchestrator turns

**Files:**
- Modify: `src/transect/scanners/phases.py` (`turn_digests`, `execute` n_turns, `_span_anchors`)
- Test: `tests/test_decision_phases.py`, `tests/test_turn_axis.py` (append)

**Interfaces:**
- Produces: digests with `turn` = orchestrator ordinal; `value["turns"]` covers `range(n_orchestrator_turns)`.

- [ ] **Step 1: Test**

```python
def test_digests_number_orchestrator_turns_and_fold_delegations():
    """Digest indices are contiguous orchestrator ordinals; a span's
    delegation line folds into the eligible turn before its begin."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("plan"),
                model_turn("delegate"),
                *agent_span("A", "eda", inner=[model_turn("a0"), model_turn("a1")], metadata={"task": "survey data"}, parent_id="R"),
                model_turn("wrap"),
            ],
        )
    ]
    digests = turn_digests(StubTranscript(events))
    assert [d.turn for d in digests] == [0, 1, 2]
    assert digests[1].delegations == ["survey data"]
```

and a scan-level check that `value["turns"]` has three rows (reuse `test_every_turn_gets_a_row_with_its_basis`'s harness with the same events).

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

`turn_digests`: replace the `enumerate(model_turns(...))` loop with `for turn, event, calls in orchestrator_turns(transcript)` (drop the `main_models` check). `_span_anchors`: count orchestrator turns (`n_model` increments only on main events; build `main_events` via `main_span` once). `execute`: `n_turns = sum(1 for _ in orchestrator_turns(transcript))`; explanation string "digest turns ({n_turns} orchestrator turns)". Docstrings accordingly.

- [ ] **Step 4: Run** `uv run pytest tests/test_decision_phases.py tests/test_turn_axis.py tests/test_narration_ranges.py tests/test_narration_preservation.py tests/test_phase_review_units.py tests/test_batching.py -q`

- [ ] **Step 5: Commit** `git commit -m "feat: phase digests and dense turns number orchestrator turns"`

---

### Task 9: phases frame: orchestrator spend, delegated spend, spawn count

**Files:**
- Modify: `src/transect/frames/phases.py` (`phases_df` signature, `_new_work_rollup`, new `_delegated_rollup`), `src/transect/api.py:266-300` (build order), `src/transect/report/sections.py:1031-1110` (phase cards read the new columns for the existing tags; chips are PR 4)
- Test: `tests/test_frames.py` (append)

**Interfaces:**
- Produces: `phases_df(results, phase_turns, token_timeline, subagents)` with new columns `delegated_new_work_tokens (Float64)` and `n_subagents (Int64)`.

- [ ] **Step 1: Test**

```python
def test_phase_rollups_split_orchestrator_and_delegated_spend():
    """Orchestrator spend sums the phase's own turns; delegated spend
    sums the spans spawned inside the phase; the count follows spawns."""
    phase_turns = pd.DataFrame({"transcript_id": ["t"] * 4, "turn": [0, 1, 2, 3], "phase_index": [0, 0, 1, 1]})
    timeline = pd.DataFrame({"transcript_id": ["t"] * 6, "turn": [0, 1, None, None, 2, 3], "new_work": [10, 20, 500, 600, 30, 40]}).astype({"turn": "Int64"})
    subagents = pd.DataFrame({"transcript_id": ["t", "t"], "agent_span_id": ["A", "B"], "spawn_turn": [1, 3], "new_work": [1100.0, None]})
    phases = phases_df(raw_phases_two_phases(), phase_turns=phase_turns, token_timeline=timeline, subagents=subagents)
    assert phases.new_work_tokens.tolist() == [30, 70]
    assert phases.delegated_new_work_tokens.tolist()[0] == 1100
    assert pd.isna(phases.delegated_new_work_tokens.tolist()[1])
    assert phases.n_subagents.tolist() == [1, 1]
```

Build `raw_phases_two_phases()` from the existing `raw_phases` fixture pattern in that file (two phases with ranges 0-1 and 2-3).

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

`_new_work_rollup`: unchanged logic now naturally orchestrator-only because sub-agent rows have `turn` NA and the merge key is `turn`; add `token_timeline[token_timeline.turn.notna()]` explicitly and say so in the docstring. New:

```python
def _delegated_rollup(phases, subagents):
    """Per-phase delegated spend and spawn count: sub-agent spans whose
    spawn_turn lies in the phase's range. Spend is NA when no member
    span carries usage (tool-only lanes), never 0."""
```

Vectorised: for each phase row, `members = subagents[(subagents.transcript_id == tid) & (subagents.spawn_turn >= start) & (subagents.spawn_turn <= end)]`; `n = len(members)`; `spend = members.new_work.sum(min_count=1)`. Columns appended after `new_work_tokens`; docstring entries:

```
- new_work_tokens: orchestrator new-work summed over the turns the
  dense map assigns to the phase (tool-only turns included).
- delegated_new_work_tokens: new-work of the sub-agent spans spawned
  inside the phase (spawn_turn in range); NA when none carries usage.
- n_subagents: sub-agent spans spawned inside the phase.
```

`api.load`: build `subagents` before `phases`; pass it. `sections.phase_cards`: the `N sub-agent(s)` tag reads `p.n_subagents` (drop the `lanes` parameter and `spawn_turns`), keep the single spend tag reading `new_work_tokens` and relabel it "orchestrator new-work tokens" (the delegated chip is PR 4). `render.py` drops `lanes=tool_lanes` from the call.

- [ ] **Step 4: Run** `uv run pytest tests/test_frames.py tests/test_report_integration.py -q -k "phase or report"` (report tests still read the old stores until Task 12; expect failures there, not here).

- [ ] **Step 5: Commit** `git commit -m "feat: phases carry orchestrator and delegated spend separately"`

---

### Task 10: Excerpts and the report on the new axis

**Files:**
- Modify: `src/transect/report/excerpts.py` (`_turn_excerpts`, `_tool_call_counts`, `Excerpt.lane` doc), `src/transect/report/render.py` (`_n_turns` sites, `one` filtering, `_span_lanes`, `_subagent_section`, swimlane inputs), `src/transect/report/sections.py` (`span_titles`), `src/transect/report/lanes_layout.py` (`pack_lanes` takes `(span_id, x0, x1, lane_name)` tuples), `src/transect/report/charts.py` (`swimlanes` mode flag per span), `src/transect/report/templates/subagent_notes.html.j2`, `src/transect/api.py:641` (`_n_turns`)
- Test: `tests/test_report_integration.py`, `tests/test_turn_axis.py` (append: excerpt rows)

**Interfaces:**
- Consumes: `subagents` coordinates (Task 5), `token_timeline.turn` nullable (Task 3).
- Produces: `pack_lanes(spans: list[tuple[Any, float, float, str]], label_of, min_footprint)`; `PackedLanes.rows` unchanged shape.

- [ ] **Step 1: Tests**

```python
def test_excerpts_cover_orchestrator_turns_only(demo_transcript):
    """Excerpt rows exist for orchestrator turns 0..9 at most and none
    of them names a sub-agent lane."""
    found = _turn_excerpts(demo_transcript)
    assert set(found) <= set(range(10))
    assert {e.lane for e in found.values()} == {"orchestrator"}
```

(`demo_transcript`: a fixture reading the demo log through `transcripts_from(...).reader()` with events; add to `tests/conftest.py` if absent.) Report integration: the mechanical scenarios keep passing; add to `_assert` for "demo-with-subagents" that the HTML contains `turn 9` and not `turn 16`.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

- `excerpts.py`: both functions iterate `orchestrator_turns(transcript)`; `lane` is always the orchestrator (keep the field so the template is untouched in PR 1; PR 4 removes the label). Module docstring: "Same turn axis as everything else: `orchestrator_turns`."
- `api._n_turns`: `token_timeline.dropna(subset=["turn"]).groupby("transcript_id").turn.max().add(1)`.
- `render.py`: `main = one[one.turn.notna()]` and pass `main` to `charts.token_stack`, `interventions_chart`, `has_derived_token_views`, `token_measures_coincide`, `draws_threshold`, tool_counts; `my_flushes` for charts and cards filtered to `agent_span_id.isna()`; `n_turns = int(main.turn.max()) + 1`.
- `_span_lanes(subagents)`: returns `[(span_id, start_pos, end_pos, lane_name, position_source)]` from the subagents frame; the section renders when `len(subagents) > 0` (a `.eval` with no spans has no rows; the previous "more than one lane" gate becomes "any span").
- `lanes_layout.pack_lanes`: accept `(span_id, x0, x1, lane_name)`; sort by `x0`; the overlap test and footprint unchanged.
- `_subagent_section`: `has_end_markers = subagents.end_recorded.any()`; boxes when `position_source == "timestamp"` (pass a per-row `boxed` flag into `rows` as a sixth element and have `charts.swimlanes` draw a box or a tick per row from it; keep the global `span_ends_recorded` only for the `×` markers); `end_markers` from `end_pos` where `end_recorded`; `spawn_rows` sort by `spawn_turn`.
- `sections.span_titles`: the `turns` cell reads `f"{anchor_turn}–{end_turn}"` plus " (wall-clock extent)" when `position_source == "timestamp"`, " (spawn turn only)" otherwise; add an `after_last` note cell "activity continued after the last orchestrator turn" when set (add the field to `SPAN_TIP_FIELDS` only if it renders in every mode; otherwise fold the text into the `turns` cell).
- `subagent_notes.html.j2`: the how-to-read line: "Box extent is the span's wall-clock activity mapped onto the orchestrator turns active at the time, not a count of its own turns; {{ glyph }} marks a recorded span completion; a thin tick marks a span whose source recorded no timestamps."
- `custom.SectionContext.n_turns` receives the orchestrator count (it already comes from `_n_turns`).

Read `charts.py`'s module docstring before touching `swimlanes` (tooltip channel rules; mark ordering).

- [ ] **Step 4: Run** `uv run pytest tests/test_report_integration.py tests/test_turn_axis.py -q -k "mechanical or excerpt"`; then render the demo in a browser: `uv run python -c "from transect import transect; transect('examples/logs/house_price_demo.eval', spec='examples/spec.yaml', scans_dir='/tmp/.../scans', viewer=False)"` and open the report; confirm the band spans 0..9, the three swimlane boxes sit inside turns 3, 5 and 8, and the token chart shows ten bars.

- [ ] **Step 5: Commit** `git commit -m "feat: report reads the orchestrator axis; swimlanes drawn by wall-clock"`

---

### Task 11: Scan status surfaces an unresolvable orchestrator

**Files:**
- Test: `tests/test_scan_status_integration.py` (append)

- [ ] **Step 1: Test**

Build an `.eval`-free path: run `_run` on a tiny transcript database made by Scout's `transcripts_from` writer from two top-level agent spans (or monkeypatch `main_span` to raise for one transcript), assert `results.scan_status.errors` names the transcript and the message contains "no single orchestrator lane", and that `render` still writes a report whose scan status section shows the error. If building a store is heavy, monkeypatching `transect.scanners.helpers.main_span` for the demo log is acceptable.

- [ ] **Step 2: Run, implement nothing unless the error is swallowed, run again, commit** `git commit -m "test: an ambiguous orchestrator lane surfaces in scan status"`

---

### Task 12: The parallel sub-agents fixture

**Files:**
- Create: `tests/fixtures/generate_parallel_eval.py`, `tests/fixtures/logs/<stamp>_parallel-subagents_<id>.eval`
- Test: `tests/test_turn_axis.py` (append), `tests/conftest.py` (fixture path)

**Interfaces:**
- Produces: a mockllm `deepagent(background=True)` run with two background sub-agents (`scout_a`, `scout_b`) whose rewritten timestamps overlap each other and span orchestrator turns 1 to 3.

- [ ] **Step 1: Write the generator**

Shape (adapt to what `deepagent` actually emits; inspect the log after a first run):

```python
"""Generate the parallel sub-agents fixture: a deepagent run with two
background sub-agents working while the orchestrator continues.

Run from the repo root:

    uv run python tests/fixtures/generate_parallel_eval.py
"""
from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.agent import deepagent
from inspect_ai.agent._deepagent.subagent import Subagent
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import tool

MOCK = "mockllm/model"


@tool
def bash():
    async def execute(cmd: str) -> str:
        """Pretend shell.

        Args:
            cmd: command
        """
        return f"ran {cmd}"
    return execute


def route(input, tools, tool_choice, config) -> ModelOutput:
    system = next((m.text for m in input if m.role == "system"), "")
    turns = sum(1 for m in input if m.role == "assistant")
    if "scout_a" in system or "You survey" in system:
        return [ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "ls a"}), ModelOutput.from_content(MOCK, "a done")][turns]
    if "scout_b" in system or "You test" in system:
        return [ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "ls b"}), ModelOutput.from_content(MOCK, "b done")][turns]
    script = [
        ModelOutput.for_tool_calls(MOCK, [("agent", {"name": "scout_a", "prompt": "survey", "background": True}), ("agent", {"name": "scout_b", "prompt": "test", "background": True})]),
        ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "work 1"}),
        ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "work 2"}),
        ModelOutput.for_tool_call(MOCK, "agent_wait", {}),
        ModelOutput.for_tool_call(MOCK, "submit", {"answer": "DONE"}),
    ]
    return script[min(turns, len(script) - 1)]
```

Then `deepagent(tools=[bash()], subagents=[Subagent(name="scout_a", description="surveys", prompt="You survey the repo."), Subagent(name="scout_b", description="tests", prompt="You test the build.")], memory=False, todo_write=False, compaction=None, background=True, model=get_model(MOCK, custom_outputs=route, memoize=False))`. Check `ModelOutput.for_tool_calls` exists; otherwise build the two-call assistant message by hand. Check the exact `agent` and `agent_wait` tool schemas in the installed Inspect (`inspect_ai/agent/_deepagent/`) before scripting them.

Timestamp rewrite (`stretch_wallclock` variant): assign the orchestrator's events a schedule of 60 s per model call; give each sub-agent span's events timestamps starting 15 s after the spawn call and spaced 40 s, so both spans run during orchestrator turns 1 to 3; set `completed` 10 s after each start; set `span_end` timestamps at each span's last event plus 5 s. Write to `tests/fixtures/logs/` with a stable name.

- [ ] **Step 2: Generate, then pin**

```python
def test_parallel_fixture_spans_overlap_on_the_orchestrator_axis(parallel_results):
    """Two background sub-agents spawned at turn 0 run across turns 1 to 3
    and overlap each other, so they pack into two rows."""
    spans = parallel_results.subagents.sort_values("agent_lane")
    assert spans.spawn_turn.tolist() == [0, 0]
    assert (spans.anchor_turn >= 0).all() and (spans.end_turn >= 2).all()
    a, b = spans.itertuples()
    assert max(a.start_pos, b.start_pos) < min(a.end_pos, b.end_pos)  # overlap
    assert spans.position_source.tolist() == ["timestamp", "timestamp"]
```

Add the fixture `parallel_results` to `tests/conftest.py` the way `demo_results` is built. Add the fixture to `tests/test_report_integration.py`'s `SCENARIOS` as `"parallel-subagents"` with the expected sections including "Sub-agent activity".

- [ ] **Step 3: Run, commit** `git commit -m "test: parallel sub-agents fixture pins wall-clock placement"`

---

### Task 13: Regenerate the judged stores and re-pin the stored-scan report test

**Files:**
- Modify: `tests/fixtures/generate_demo_scan.py` if its planted-signal checks read turn numbers; regenerate `tests/fixtures/demo_scan/`, `tests/fixtures/demo_scan_cohort/`
- Test: `tests/test_report_integration.py:60-120` (`_STORE_CONTENT` strings mentioning turn ranges), `tests/test_frames.py` judged pins (`raw_phases` ranges), `tests/test_reliability.py` if it pins turns

- [ ] **Step 1:** `uv run python tests/fixtures/generate_demo_scan.py` from the repo root; read the script's planted-signal check output.
- [ ] **Step 2:** `uv run pytest tests/test_report_integration.py tests/test_frames.py tests/test_reliability.py -q`; update pinned strings (turn ranges now within 0..9, e.g. "turns 0–3" style) to the regenerated values after verifying each by reading the frame, never by copying failure output blindly.
- [ ] **Step 3:** Commit `git commit -m "test: regenerate the judged demo stores on the orchestrator axis"`

---

### Task 14: Docs

**Files:**
- Modify: `CLAUDE.md` (the AGENTS.md file; add a load-bearing contract bullet), `README.md:231-235` and the mermaid block, `.claude/skills/custom-ui/SKILL.md:58`, `.claude/skills/using-transect/SKILL.md:253`, `.claude/skills/add-a-layer/SKILL.md:271-281`

- [ ] **Step 1: AGENTS.md bullet** under Load-bearing contracts:

```
- **Orchestrator turn axis.** `turn` is the 0-based ordinal of the
  orchestrator's (main lane's) model turns, from
  `scanners.helpers.orchestrator_turns`; sub-agent turns are off the
  axis and are placed on it by wall-clock (`frames/spine.py`), never
  numbered. A custom scanner numbers turns with the same helper. The
  main lane is the span Scout would scan at depth 1; an ambiguous tree
  raises rather than guessing.
```

- [ ] **Step 2: README**: "Top to bottom, everything on a shared turn axis" becomes "everything on the orchestrator's turn axis; sub-agents appear as swimlanes placed by wall-clock"; the frames list line for `lane_activity`; the mermaid `token_timeline` box (`int turn "orchestrator turn; NA on sub-agent rows"`, add `int lane_turn`), the `subagents` box (add `int spawn_turn`, `float start_pos`, `float end_pos`, `string position_source`), the `flushes` box (add `string agent_span_id`), the `phases` box (add `int delegated_new_work_tokens`, `int n_subagents`), and the `lane_activity` box's `turn` comment.
- [ ] **Step 3: Skills**: custom-ui grain row: "`turn` (0-based orchestrator turn; sub-agent turns are not on the axis)"; using-transect line 253 wording; add-a-layer §4: one sentence "Number turns with `transect.scanners.helpers.orchestrator_turns(transcript)`; every lane's model events would misalign with the axis" (the public export and the fail-loud rule are PR 3).
- [ ] **Step 4:** `uv run pytest tests/test_skills.py -q`; commit `git commit -m "docs: the orchestrator turn axis"`

---

### Task 15: Gate, review, PR

- [ ] **Step 1:** `make check > /tmp/check.log 2>&1; echo "status=$?"`; fix until status is 0. Never pipe the status away.
- [ ] **Step 2:** Browser check (playwright test already ran); open the demo and parallel reports once more by eye.
- [ ] **Step 3:** Dispatch two clean-context review agents over `git diff main...HEAD`: a correctness reviewer with the spec and AGENTS.md contracts, and an adversarial tester with the Review Focus list. Fix what they find; rerun `make check`.
- [ ] **Step 4:** Push the branch and open the PR titled `feat: orchestrator turns are the turn axis` with a body summarising the contract, the frame changes, and the fixture; link #50 and #49.
