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


def test_a_transcript_without_model_events_is_an_empty_lane():
    """Nothing to number: zero orchestrator turns, no error."""
    events = [*agent_span("A", "one", inner=[])]
    assert list(orchestrator_turns(StubTranscript(events))) == []
    assert list(orchestrator_turns(StubTranscript([]))) == []


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


def test_zero_width_cells_place_at_their_left_edge():
    """Two calls at the same instant: the empty cell swallows nothing."""
    cells = spine.cells([at(0), at(0), at(100)], last_completed=at(120))
    assert spine.position(at(0), cells) == pytest.approx(0.5)
    assert spine.position(at(50), cells) == pytest.approx(1.0)


def test_coordinates_follow_timestamps_when_present():
    """A span's box runs from its first activity to its recorded end."""
    cells = spine.cells([at(0), at(100), at(150)], last_completed=at(170))
    record = {
        "spawn_turn": 0,
        "first_at": at(20).isoformat(),
        "last_at": at(90).isoformat(),
        "end_at": at(110).isoformat(),
        "end_recorded": True,
        "event_order_end_turn": 1,
    }
    got = spine.coordinates(record, cells)
    assert got["position_source"] == "timestamp"
    assert (got["anchor_turn"], got["end_turn"]) == (0, 1)
    assert got["start_pos"] == pytest.approx(-0.3)
    assert got["end_pos"] == pytest.approx(0.7)
    assert got["after_last"] is False


def test_an_unrecorded_end_falls_back_to_last_activity():
    """Without a recorded end the box closes at the last observed event."""
    cells = spine.cells([at(0), at(100)], last_completed=at(120))
    record = {
        "spawn_turn": 0,
        "first_at": at(10).isoformat(),
        "last_at": at(50).isoformat(),
        "end_at": at(90).isoformat(),
        "end_recorded": False,
        "event_order_end_turn": 0,
    }
    assert spine.coordinates(record, cells)["end_pos"] == pytest.approx(0.0)


def test_coordinates_fall_back_to_event_order_without_timestamps():
    """Missing timestamps collapse the span onto its spawn turn."""
    cells = spine.cells([at(0), at(100)], last_completed=None)
    record = {
        "spawn_turn": 1,
        "first_at": None,
        "last_at": None,
        "end_at": None,
        "end_recorded": False,
        "event_order_end_turn": None,
    }
    assert spine.coordinates(record, cells) == {
        "start_pos": 1.0,
        "end_pos": 1.0,
        "anchor_turn": 1,
        "end_turn": 1,
        "position_source": "event_order",
        "after_last": False,
    }


def test_activity_after_the_last_turn_clamps_and_is_flagged():
    """A background agent outliving the orchestrator clamps to the edge."""
    cells = spine.cells([at(0), at(100)], last_completed=at(120))
    record = {
        "spawn_turn": 1,
        "first_at": at(105).isoformat(),
        "last_at": at(500).isoformat(),
        "end_at": None,
        "end_recorded": False,
        "event_order_end_turn": None,
    }
    got = spine.coordinates(record, cells)
    assert got["end_pos"] == pytest.approx(1.5)
    assert got["end_turn"] == 1
    assert got["after_last"] is True


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
