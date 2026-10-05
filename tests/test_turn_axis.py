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
    texts = [
        e.output.message.text
        for e, _ in all_model_turns(StubTranscript(handoff_shape()))
    ]
    assert texts == ["lead 0", "a0", "a1", "lead 1", "b0", "lead 2"]


def test_a_bare_transcript_is_its_own_orchestrator():
    """Span-less model events form the main lane."""
    events = [model_turn("x"), model_turn("y")]
    assert [t for t, _, _ in orchestrator_turns(StubTranscript(events))] == [0, 1]


def test_main_span_refuses_two_top_level_agents():
    """No single orchestrator means a loud error, never a guessed axis."""
    events = [
        *agent_span("A", "one", inner=[model_turn("a")]),
        *agent_span("B", "two", inner=[model_turn("b")]),
    ]
    with pytest.raises(ValueError, match=r"2 top-level agents.*'one'.*'two'"):
        main_span(StubTranscript(events))


def test_a_transcript_without_events_is_an_empty_lane():
    """A message history with no event stream has zero turns, no error."""
    assert list(orchestrator_turns(StubTranscript([]))) == []


def test_events_without_a_model_turn_are_refused():
    """A sample that errored before its first call is a scan error, not
    a silently vanished transcript."""
    events = [*agent_span("A", "one", inner=[])]
    with pytest.raises(ValueError, match=r"no model turns"):
        main_span(StubTranscript(events))


from transect.scanners.helpers import subagent_span_begins  # noqa: E402


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


# --- frames.spine: timestamps onto turn cells ---------------------------------

from datetime import datetime, timedelta  # noqa: E402

from transect.frames import spine  # noqa: E402

T0 = datetime(2026, 1, 1, 10, 0, 0)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def test_cells_cover_each_turn_until_the_next_call_starts():
    """Cell m runs from call m's start to call m+1's start; the last cell
    ends at the last call's completion."""
    got = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    assert got == [
        (at(0).timestamp(), at(100).timestamp()),
        (at(100).timestamp(), at(150).timestamp()),
        (at(150).timestamp(), at(170).timestamp()),
    ]


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


def test_span_turns_follow_timestamps_when_present():
    """The anchor and end turns are the cells holding the span's first
    activity and its end."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    turns = spine.span_turns(
        cells,
        spawn_turn=0,
        started_at=at(20).isoformat(),
        ended_at=at(110).isoformat(),
        event_order_end_turn=1,
    )
    assert turns == {
        "anchor_turn": 0,
        "end_turn": 1,
        "turn_source": "timestamp",
    }


def test_span_turns_fall_back_to_event_order_without_timestamps():
    """Missing timestamps put the span at its spawn turn."""
    cells = spine.cells([at(0), at(100)], last_completed=None)
    turns = spine.span_turns(
        cells, spawn_turn=1, started_at=None, ended_at=None, event_order_end_turn=None
    )
    assert turns == {
        "anchor_turn": 1,
        "end_turn": 1,
        "turn_source": "event_order",
    }


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


# --- decision_phases on the orchestrator axis --------------------------------

from helpers import PHASES_SPEC, run_scan, scripted_judge, seg, seg_answer  # noqa: E402

from transect.scanners.phases import decision_phases, turn_digests  # noqa: E402


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


def test_digests_number_orchestrator_turns_and_fold_delegations():
    """Digest indices are contiguous orchestrator ordinals; a span's
    delegation line folds into the eligible turn before its begin."""
    digests = turn_digests(StubTranscript(delegating_shape()))
    assert [d.turn for d in digests] == [0, 1, 2]
    assert digests[1].delegations == ["survey data"]


def test_dense_turns_cover_orchestrator_turns_only():
    """The per-turn surface has one row per orchestrator turn."""
    judge = scripted_judge(seg_answer(seg(0, 2, "setup", 0.9)))
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False),
        delegating_shape(),
    ).value
    assert [t["turn"] for t in value["turns"]] == [0, 1, 2]
    assert [t["basis"] for t in value["turns"]] == ["judged"] * 3


# --- report excerpts ----------------------------------------------------------

from pathlib import Path  # noqa: E402

from inspect_ai.log import read_eval_log  # noqa: E402

from transect.report.excerpts import _tool_call_counts, _turn_excerpts  # noqa: E402

DEMO_LOG = Path(__file__).parents[1] / "examples" / "logs" / "house_price_demo.eval"


def test_excerpts_cover_orchestrator_turns_only():
    """Excerpt rows and tool counts exist for orchestrator turns 0..9 at
    most and no row names a sub-agent lane."""
    sample = read_eval_log(str(DEMO_LOG), resolve_attachments=True).samples[0]
    transcript = StubTranscript(sample.events)
    found = _turn_excerpts(transcript)
    assert set(found) <= set(range(10))
    assert {e.lane for e in found.values()} == {"orchestrator"}
    assert set(_tool_call_counts(transcript)) <= set(range(10))


# --- the parallel sub-agents fixture -----------------------------------------

from transect.api import _run  # noqa: E402
from transect.report.lanes_layout import pack_lanes  # noqa: E402
from transect.report.render import _span_lanes  # noqa: E402
from transect.spec import Spec  # noqa: E402


@pytest.fixture(scope="module")
def parallel_results(tmp_path_factory):
    """One $0 run over the committed parallel sub-agents log."""
    logs = Path(__file__).parent / "fixtures" / "parallel_logs"
    if not any(logs.glob("*.eval")):
        pytest.skip(
            "fixture missing - run: "
            "uv run python tests/fixtures/generate_parallel_eval.py"
        )
    return _run(
        logs=str(logs), spec=Spec(), scans_dir=str(tmp_path_factory.mktemp("scans"))
    )


def test_parallel_fixture_numbers_five_orchestrator_turns(parallel_results):
    """Four interleaved sub-agent turns take no turn numbers."""
    timeline = parallel_results.token_timeline
    assert timeline[timeline.turn.notna()].turn.tolist() == [0, 1, 2, 3, 4]
    assert int(timeline.turn.isna().sum()) == 4
    assert timeline[timeline.agent_lane == "scout_a"].lane_turn.tolist() == [0, 1]


def test_parallel_fixture_spans_overlap_on_the_orchestrator_axis(parallel_results):
    """Two background sub-agents spawned at turn 0 run across turns 0 to 3
    by wall clock and overlap each other."""
    spans = parallel_results.subagents.sort_values("agent_lane")
    assert spans.agent_lane.tolist() == ["scout_a", "scout_b"]
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
    assert sorted(row[0] for row in packed.rows) == [0, 1]
    assert packed.ylabels == ["sub-agents (2)"]


# --- swimlane geometry ----------------------------------------------------------

import pandas as pd  # noqa: E402

from transect.report import charts  # noqa: E402


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
