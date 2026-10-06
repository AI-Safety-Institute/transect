"""The orchestrator turn axis: one enumeration defines turn numbers."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from helpers import (
    PHASES_SPEC,
    StubTranscript,
    agent_span,
    model_turn,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
)
from inspect_ai.log import read_eval_log

from transect.api import _run
from transect.frames import spine
from transect.report import charts
from transect.report.excerpts import _tool_call_counts, _turn_excerpts
from transect.report.lanes_layout import pack_lanes
from transect.report.render import _span_lanes
from transect.scanners.helpers import (
    all_model_turns,
    main_span,
    orchestrator_turns,
    subagent_span_begins,
)
from transect.scanners.phases import decision_phases, turn_digests
from transect.spec import Spec

T0 = datetime(2026, 1, 1, 10, 0, 0)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def handoff_shape():
    """Lead agent span holding two sub-agent spans: the .eval handoff shape."""
    return [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead 0"),
                *agent_span(
                    "A",
                    "eda",
                    inner=[model_turn("a0"), model_turn("a1")],
                    parent_id="R",
                ),
                model_turn("lead 1"),
                *agent_span("B", "rev", inner=[model_turn("b0")], parent_id="R"),
                model_turn("lead 2"),
            ],
        )
    ]


def delegating_shape():
    """Lead plans, delegates (a two-turn sub-agent), then wraps up."""
    return [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("plan"),
                model_turn("delegate"),
                *agent_span(
                    "A",
                    "eda",
                    inner=[model_turn("a0"), model_turn("a1")],
                    metadata={"task": "survey data"},
                    parent_id="R",
                ),
                model_turn("wrap"),
            ],
        )
    ]


@pytest.fixture(scope="module")
def parallel_results(parallel_logs, tmp_path_factory):
    """One $0 run over the committed parallel sub-agents log."""
    return _run(
        logs=str(parallel_logs),
        spec=Spec(),
        scans_dir=str(tmp_path_factory.mktemp("scans")),
    )


def test_orchestrator_turns_number_the_main_lane_while_all_lanes_stay_visible():
    """Sub-agent turns consume no turn numbers, yet the internal iterator
    still yields every lane's turn in event order."""
    transcript = StubTranscript(handoff_shape())
    numbered = [
        (t, e.output.message.text) for t, e, _ in orchestrator_turns(transcript)
    ]
    assert numbered == [(0, "lead 0"), (1, "lead 1"), (2, "lead 2")]
    texts = [e.output.message.text for e, _ in all_model_turns(transcript)]
    assert texts == ["lead 0", "a0", "a1", "lead 1", "b0", "lead 2"]


@pytest.mark.parametrize(
    ("events", "turns"),
    [([model_turn("x"), model_turn("y")], [0, 1]), ([], [])],
    ids=["span-less-turns", "no-events"],
)
def test_a_span_less_transcript_is_its_own_orchestrator(events, turns):
    """Span-less model events form the main lane; no events is an empty
    lane, not an error."""
    assert [t for t, _, _ in orchestrator_turns(StubTranscript(events))] == turns


@pytest.mark.parametrize(
    ("events", "match"),
    [
        (
            [
                *agent_span("A", "one", inner=[model_turn("a")]),
                *agent_span("B", "two", inner=[model_turn("b")]),
            ],
            r"2 top-level agents.*'one'.*'two'",
        ),
        ([*agent_span("A", "one", inner=[])], r"no model turns"),
    ],
    ids=["two-top-level-agents", "no-model-turns"],
)
def test_main_span_refuses_a_transcript_without_one_orchestrator(events, match):
    """No single orchestrator, or an orchestrator that never called a
    model, is a loud error rather than a guessed or vanished axis."""
    with pytest.raises(ValueError, match=match):
        main_span(StubTranscript(events))


def test_a_wrapper_span_around_the_orchestrator_is_not_a_sub_agent():
    """An agent span that only contains the orchestrator is a container:
    not a sub-agent to record, place or classify."""
    events = [
        *agent_span(
            "outer",
            "wrapper",
            inner=[
                *agent_span(
                    "inner",
                    "lead",
                    inner=[model_turn("lead 0"), model_turn("lead 1")],
                    parent_id="outer",
                )
            ],
        )
    ]
    transcript = StubTranscript(events)
    assert main_span(transcript).id == "inner"
    begins, _ = subagent_span_begins(transcript, main_span(transcript))
    assert begins == []
    assert [t for t, _, _ in orchestrator_turns(transcript)] == [0, 1]


@pytest.mark.parametrize(
    ("last_completed", "last_cell"),
    [(at(170), (at(150), at(170))), (None, (at(150), at(225)))],
    ids=["completion-recorded", "median-width"],
)
def test_cells_run_from_each_call_to_the_next(last_completed, last_cell):
    """Cell m runs from call m's start to call m+1's start; the last cell
    ends at the last call's completion, or is median-width wide without
    one."""
    got = spine.cells([at(0), at(100), at(150)], last_completed=last_completed)
    assert got == [
        (at(0).timestamp(), at(100).timestamp()),
        (at(100).timestamp(), at(150).timestamp()),
        tuple(t.timestamp() for t in last_cell),
    ]


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
    ids=[
        "before-first",
        "first-start",
        "mid-first",
        "second-start",
        "mid-second",
        "mid-last",
        "after-last",
    ],
)
def test_position_interpolates_within_a_cell_and_clamps_outside(t, expected):
    """A timestamp maps to m - 0.5 plus its fraction of cell m."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    assert spine.position(t, cells) == pytest.approx(expected)


def test_a_zero_width_cell_is_skipped():
    """Two calls at one instant: a time there belongs to the next real cell."""
    cells = spine.cells([at(0), at(0), at(100)], last_completed=at(120))
    assert spine.position(at(0), cells) == pytest.approx(0.5)
    assert spine.position(at(50), cells) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        (
            {
                "spawn_turn": 0,
                "started_at": at(20).isoformat(),
                "ended_at": at(110).isoformat(),
                "event_order_end_turn": 1,
            },
            {"anchor_turn": 0, "end_turn": 1, "turn_source": "timestamp"},
        ),
        (
            {
                "spawn_turn": 1,
                "started_at": None,
                "ended_at": None,
                "event_order_end_turn": None,
            },
            {"anchor_turn": 1, "end_turn": 1, "turn_source": "event_order"},
        ),
    ],
    ids=["timestamps", "event-order"],
)
def test_span_turns_follow_timestamps_and_fall_back_to_the_spawn_turn(
    placement, expected
):
    """With timestamps the anchor and end turns are the cells holding the
    span's first activity and its end; without them the span sits at
    its spawn turn."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    assert spine.span_turns(cells, **placement) == expected


def test_turn_of_clamps_to_the_axis():
    """A time after the last cell belongs to the last turn."""
    cells = spine.cells([at(0), at(100)], last_completed=at(120))
    assert spine.turn_of(at(500), cells) == 1
    assert spine.turn_of(at(-5), cells) == 0


@pytest.mark.parametrize(
    ("starts", "completed", "n_cells"),
    [
        ([at(0).isoformat(), at(100).isoformat()], at(130).isoformat(), 2),
        ([at(0).isoformat(), None], None, 0),
        ([at(100).isoformat(), at(0).isoformat()], None, 0),
        ([], None, 0),
    ],
    ids=["usable", "missing-stamp", "skewed", "no-turns"],
)
def test_clock_requires_a_complete_non_decreasing_series(starts, completed, n_cells):
    """An incomplete or backwards clock yields no cells at all."""
    assert len(spine.clock(starts, completed)) == n_cells


def test_decision_phases_number_orchestrator_turns_and_fold_delegations():
    """Digest indices and the per-turn surface are contiguous orchestrator
    ordinals; a span's delegation line folds into the eligible turn
    before its begin."""
    digests = turn_digests(StubTranscript(delegating_shape()))
    assert [d.turn for d in digests] == [0, 1, 2]
    assert digests[1].delegations == ["survey data"]
    judge = scripted_judge(seg_answer(seg(0, 2, "setup", 0.9)))
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False),
        delegating_shape(),
    ).value
    assert [t["turn"] for t in value["turns"]] == [0, 1, 2]
    assert [t["basis"] for t in value["turns"]] == ["judged"] * 3


def test_excerpts_cover_orchestrator_turns_only(demo_log):
    """Excerpt rows and tool counts exist for orchestrator turns 0..9 at
    most and no row names a sub-agent lane."""
    sample = read_eval_log(str(demo_log), resolve_attachments=True).samples[0]
    transcript = StubTranscript(sample.events)
    found = _turn_excerpts(transcript)
    assert set(found) <= set(range(10))
    assert {e.lane for e in found.values()} == {"orchestrator"}
    assert set(_tool_call_counts(transcript)) <= set(range(10))


def test_parallel_fixture_frames_number_five_turns_and_overlapping_spans(
    parallel_results,
):
    """Seven interleaved sub-agent turns (subagent_a's summarization call
    included) take no turn numbers; the two background sub-agents
    spawned at turn 0 run across turns 0 to 3 by wall clock and overlap
    each other, and subagent_a's compaction sits in its own lane."""
    timeline = parallel_results.token_timeline
    assert timeline[timeline.turn.notna()].turn.tolist() == [0, 1, 2, 3, 4]
    assert int(timeline.turn.isna().sum()) == 7
    lane_a = timeline[timeline.agent_lane == "subagent_a"]
    assert lane_a.lane_turn.tolist() == [0, 1, 2, 3]
    (flush,) = parallel_results.flushes.itertuples()
    assert (flush.source, flush.lane_turn, flush.turn) == ("inspect", 2, 2)
    subagent_a = parallel_results.subagents.query("agent_lane == 'subagent_a'")
    assert flush.agent_span_id == subagent_a.agent_span_id.item()
    spans = parallel_results.subagents.sort_values("agent_lane")
    assert spans.agent_lane.tolist() == ["subagent_a", "subagent_b"]
    assert spans.spawn_turn.tolist() == [0, 0]
    assert spans.anchor_turn.tolist() == [0, 0]
    assert spans.end_turn.tolist() == [3, 3]
    assert spans.turn_source.tolist() == ["timestamp", "timestamp"]
    assert spans.end_recorded.tolist() == [True, True]
    a, b = spans.itertuples()
    assert max(a.started_at, b.started_at) < min(a.ended_at, b.ended_at)


def test_overlapping_spans_draw_as_overlapping_boxes_in_two_rows(parallel_results):
    """The render maps both spans' timestamps onto the axis as boxes that
    overlap, and the greedy packer gives them separate sub-lanes."""
    timeline = parallel_results.token_timeline
    lanes = _span_lanes(parallel_results.subagents, timeline[timeline.turn.notna()])
    assert all(span.boxed for span in lanes)
    a, b = lanes
    assert max(a.x0, b.x0) < min(a.x1, b.x1)
    assert not (a.before_first or a.after_last or b.before_first or b.after_last)
    packed = pack_lanes(lanes, lambda _sid: "sub-agents")
    assert sorted(row[1] for row in packed.rows) == [0, 1]
    assert packed.ylabels == ["sub-agents (2)"]


@pytest.mark.parametrize(
    ("n_turns", "expected"),
    [(10, 0.06), (1000, 3.75)],
    ids=["short-axis", "long-axis"],
)
def test_the_box_floor_is_a_pixel_sliver(n_turns, expected):
    """The minimum box is six pixels at the axis's pixel-per-turn rate,
    never a whole turn."""
    assert charts.span_min_box_width(n_turns) == pytest.approx(expected)


def test_a_box_at_the_axis_edge_is_shifted_left_not_drawn_off_canvas():
    """A span clamped at the last turn ends exactly at the axis edge."""
    rows = pd.DataFrame(
        {"x_start": [9.49, 2.0], "width": [0.0, 0.5], "boxed": [True, False]}
    )
    geo = charts.span_geometry(rows, n_turns=10)
    assert geo.x2.iloc[0] == pytest.approx(9.5)
    assert geo.x1.iloc[0] == pytest.approx(9.5 - charts.span_min_box_width(10))
    # a tick keeps its position and gets a hover footprint around it
    assert geo.x.iloc[1] == 2.0
    assert geo.hx2.iloc[1] - geo.hx1.iloc[1] == pytest.approx(
        charts.span_min_box_width(10)
    )
