"""Typed report blocks: what a user layer's section is made of.

Every viz block takes an optional ``frame`` override (default: the
layer's own frame). At render an override is cut to the transcript
when it carries ``transcript_id``, and broadcasts as-is otherwise.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from transect.tags import _taggable

_CHART_KINDS = ("bar", "line", "step", "area")


@dataclass(frozen=True)
class TurnChart:
    """A numeric per-turn series on the orchestrator turn axis."""

    y: str
    kind: str = "bar"
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class TurnBand:
    """A unit-height coloured band from a categorical per-turn column
    (the phase-band shape)."""

    label: str
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class SpanLanes:
    """Swimlane rectangles from one row per span."""

    start: str
    end: str
    label: str
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class EventMarks:
    """Dashed vertical rules at event turns (the flush shape);
    ``label`` is the legend text."""

    label: str
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class Table:
    """A sortable table; ``columns`` selects and orders, None = all."""

    columns: list[str] | None = None
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class Markdown:
    """Markdown prose (HTML escaped, headings demoted); ``text`` may
    be a string or a callable of the section context."""

    text: str | Callable[[Any], str]


@dataclass(frozen=True)
class Component:
    """The escape hatch: ``fn(ctx)`` returns an inspect-viz Component;
    ``ctx`` is the `SectionContext` (transcript_id, the layer's frame
    slice, n_turns)."""

    fn: Callable[[Any], Any]


BLOCK_TYPES = (TurnChart, TurnBand, SpanLanes, EventMarks, Table, Markdown, Component)


def validate_blocks_static(layer) -> None:
    """Section validation, run before anything is spent."""
    for i, block in enumerate(layer.section or []):
        where = f"layer {layer.name!r} section[{i}]"
        if not isinstance(block, BLOCK_TYPES):
            names = ", ".join(t.__name__ for t in BLOCK_TYPES)
            raise ValueError(
                f"{where}: {type(block).__name__} is not a section block "
                f"(one of {names}; raw HTML is deliberately not accepted)"
            )
        if isinstance(block, TurnChart) and block.kind not in _CHART_KINDS:
            raise ValueError(
                f"{where}: unknown chart kind {block.kind!r} "
                f"(one of {', '.join(_CHART_KINDS)})"
            )
        if isinstance(block, Markdown) and not (
            isinstance(block.text, str) or callable(block.text)
        ):
            raise ValueError(f"{where}: Markdown text must be a string or a callable")
        if isinstance(block, Component) and not callable(block.fn):
            raise ValueError(f"{where}: Component fn must be callable")


def validate_section(
    layer,
    own_frame: pd.DataFrame | None,
) -> None:
    """Full section validation: the static half plus every
    frame-reading block's shape against the frame it will read."""
    validate_blocks_static(layer)
    for i, block in enumerate(layer.section or []):
        where = f"layer {layer.name!r} section[{i}] {type(block).__name__}"
        if isinstance(block, (Markdown, Component)):
            continue
        frame = _resolve(block.frame, own_frame, where)
        if isinstance(block, (TurnChart, TurnBand, EventMarks)):
            _need(frame, "turn", where)
        if isinstance(block, TurnChart):
            _need(frame, block.y, where)
            if not pd.api.types.is_numeric_dtype(frame[block.y]):
                raise ValueError(f"{where}: column {block.y!r} must be numeric")
        elif isinstance(block, TurnBand):
            _need(frame, block.label, where)
            # same closed-vocabulary fence as a tag family (tags.py)
            if not _taggable(frame[block.label]):
                raise ValueError(
                    f"{where}: column {block.label!r} must be string/categorical"
                )
        elif isinstance(block, SpanLanes):
            for column in (block.start, block.end, block.label):
                _need(frame, column, where)
        elif isinstance(block, Table) and block.columns is not None:
            for column in block.columns:
                _need(frame, column, where)


def _resolve(
    ref: pd.DataFrame | None,
    own_frame: pd.DataFrame | None,
    where: str,
) -> pd.DataFrame:
    if isinstance(ref, pd.DataFrame):
        return ref
    if own_frame is None:
        raise ValueError(f"{where}: the layer has no frame to read")
    return own_frame


def _need(frame: pd.DataFrame, column: str, where: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{where}: column {column!r} is not in the frame")
