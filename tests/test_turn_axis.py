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
def test_main_span_refuses_an_ambiguous_or_empty_tree(events, match):
    """No single orchestrator means a loud error, never an empty axis."""
    with pytest.raises(ValueError, match=match):
        main_span(StubTranscript(events))
