"""Typed section blocks: field validation and frame-shape validation."""

from typing import Any, cast

import pandas as pd
import pytest

from transect import Layer
from transect.report import (
    Component,
    EventMarks,
    Markdown,
    SpanLanes,
    Table,
    TurnBand,
    TurnChart,
)
from transect.report.blocks import validate_section

PER_TURN = pd.DataFrame(
    {
        "transcript_id": ["tr1"] * 3,
        "turn": [0, 1, 2],
        "risk": [0.1, 0.7, 0.4],
        "why": ["reads docs", "downloads binary", "sandboxes it"],
    }
)
SPANS = pd.DataFrame({"start_turn": [1], "end_turn": [3], "n_calls": [23]})


def test_a_full_section_validates_against_its_frames():
    """The happy path: blocks read the layer's own frame by default, a
    built-in frame by name, or their own explicit DataFrame."""
    section = [
        Markdown("Judge-scored risk per turn. *Prototype.*"),
        TurnChart(y="risk", kind="bar"),
        TurnBand(label="why"),
        Table(columns=["turn", "risk", "why"]),
        SpanLanes(start="start_turn", end="end_turn", label="n_calls", frame=SPANS),
        Component(lambda ctx: None),
    ]
    layer = Layer(name="tool_risk", frame=PER_TURN, section=section)
    validate_section(layer, PER_TURN)  # raises on any offence


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (lambda: TurnChart(y="risk", kind="pie"), "kind"),
        (lambda: TurnChart(y="nope"), "nope"),
        (lambda: TurnChart(y="why"), "numeric"),
        (lambda: TurnBand(label="risk"), "string"),
        (lambda: EventMarks(label="flush"), "turn"),  # frame below has no turn
        (lambda: Table(columns=["turn", "nope"]), "nope"),
        (lambda: Markdown(cast(Any, 42)), "text"),
        (lambda: Component(cast(Any, "not callable")), "callable"),
        (lambda: SpanLanes(start="start_turn", end="gone", label="n_calls"), "gone"),
    ],
    ids=[
        "chart-kind-unknown",
        "chart-column-absent",
        "chart-y-not-numeric",
        "band-label-not-string",
        "marks-need-a-turn-column",
        "table-column-absent",
        "markdown-not-text",
        "component-not-callable",
        "spanlanes-column-absent",
    ],
)
def test_each_malformed_block_is_rejected_naming_the_offence(block, match):
    """Errors name the layer, the block, and the offending field."""
    frame = SPANS if match in ("turn", "gone") else PER_TURN
    layer = Layer(name="tool_risk", frame=frame, section=[block()])
    with pytest.raises(ValueError, match=match) as err:
        validate_section(layer, frame)
    assert "tool_risk" in str(err.value)


def test_frame_reading_blocks_refuse_a_layer_without_a_frame():
    """A frame-reading block on a frameless layer must carry its own
    (precomputed) DataFrame - there is nothing to default to."""
    layer = Layer(name="ctxless", section=[TurnChart(y="risk")])
    with pytest.raises(ValueError, match="no frame"):
        validate_section(layer, None)


def test_sections_validate_on_the_run(layered_run):
    """The shared $0 run mounts a layer with a section: validation ran
    at the entry point and the blocks survived to the results."""
    results, _ = layered_run
    (layer,) = [x for x in results.extra_layers if x.name == "turn_chars"]
    assert [type(block) for block in layer.section] == [
        Markdown,
        TurnChart,
        TurnBand,
    ]
