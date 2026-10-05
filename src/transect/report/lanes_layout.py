"""Greedy row-packing for the sub-agent swimlanes chart.

Concurrent same-label sub-agent spans would visually overlap on a shared
turn axis; this module packs them into sub-lane rows, one row-block per
classification label. The drawing is `charts.swimlanes`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The cap on a lane name surfaced in the swimlanes chart. Both this
# module's own `rows` and `sections.span_titles`' richer "lane" tooltip
# row read this one constant, so the two cannot drift onto different
# limits for what is conceptually one value. Not unbounded: tippy caps
# the popover at 350px wide whatever the content length, so a longer name
# only buys more wrapped lines, and `charts._SWIMLANE_LANE_ROW_PX`'s
# tooltip-fit floor is measured against this exact cap.
LANE_NAME_MAX_CHARS = 120


def truncate_lane_name(name: str) -> str:
    """One lane name, capped at `LANE_NAME_MAX_CHARS` - an ellipsis
    marks an actual truncation, never appended to a name that already
    fit.

    Codepoint-safe, not grapheme-safe: a name with a decomposed accent
    that lands exactly on the cut can be split base-from-mark. Real lane
    names are plain ASCII task labels, so this is an accepted limitation
    rather than an observed problem."""
    return (
        name if len(name) <= LANE_NAME_MAX_CHARS else name[:LANE_NAME_MAX_CHARS] + "…"
    )


@dataclass(frozen=True)
class SpanGeometry:
    """One sub-agent span on the orchestrator axis, as the swimlane
    draws it: ``x0``..``x1`` is its wall-clock extent mapped through
    `frames.spine.position` when ``boxed``, else both are the spawn
    turn (a tick). ``before_first`` / ``after_last`` say the span's
    activity ran past the axis's left or right edge, where the box is
    clamped and the tooltip says so.
    """

    span_id: Any
    lane_name: str
    x0: float
    x1: float
    boxed: bool
    before_first: bool = False
    after_last: bool = False


@dataclass
class PackedLanes:
    """Row layout for one transcript's sub-agent swimlanes.

    ``rows``: one ``(row_y, x_start, width, label, lane_name, boxed)``
    tuple per placed span - the bar geometry for `charts.swimlanes`
    (``boxed``: draw the span's extent as a box; a tick otherwise).
    ``span_row``: one ``(span_id, row_y)`` tuple per placed span, for
    joining completion markers onto their row. ``yticks``/``ylabels``:
    one entry per label row-block - the block's midpoint y and a
    "label (n)" text.
    """

    rows: list[tuple]
    span_row: list[tuple]
    yticks: list[float]
    ylabels: list[str]


def pack_lanes(
    lanes: list[SpanGeometry],
    label_of: Callable[[Any], str],
    min_footprint: float = 0.0,
) -> PackedLanes:
    """Greedily pack ``lanes`` into swimlane rows, one row-block per label.

    ``label_of`` maps a span_id to its classification label (callers
    fall back to "unclassified" for spans with no judged label).

    Row-blocks are ordered by sorted label name; within a block, spans
    are placed first-fit ordered by start position - a span opens a new
    sub-lane only when every existing sub-lane in the block is still
    busy past its start.

    ``min_footprint`` is the minimum horizontal extent, in turn units, a
    span occupies for packing purposes only. ``0.0`` packs by each span's
    own observed extent, which is what `charts.swimlanes`'
    proportional-width mode wants. Uniform-width mode passes a non-zero
    footprint (`charts.swimlane_min_footprint`), because every span there
    renders at the same fixed width regardless of its real (often
    zero-duration) extent, so two spans a handful of turns apart would
    render as touching or overlapping marks even though their raw x0/x1
    extents do not overlap at all.

    Only the running sub-lane occupancy (``lane_last_end``) is inflated,
    never the ``width`` stored in ``rows`` - that stays each span's true
    ``x1 - x0``, since `charts.span_geometry` applies the drawn floor
    itself. The overlap test (``x0 > end``) is unchanged.
    """
    by_label: dict[str, list[SpanGeometry]] = {}
    for span in lanes:
        by_label.setdefault(label_of(span.span_id), []).append(span)

    rows: list[tuple] = []  # (y, x_start, width, label, lane_name, boxed)
    span_row: list[tuple] = []  # (span_id, y) for completion markers
    y = 0
    yticks: list[float] = []
    ylabels: list[str] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda span: (span.x0, str(span.span_id)))
        lane_last_end: list[float] = []
        row_base = y
        for span in group:
            x0, x1 = span.x0, span.x1
            footprint_end = max(x1, x0 + min_footprint)
            placed = next(
                (li for li, end in enumerate(lane_last_end) if x0 > end), None
            )
            if placed is None:
                placed = len(lane_last_end)
                lane_last_end.append(footprint_end)
            else:
                lane_last_end[placed] = footprint_end
            rows.append(
                (
                    row_base + placed,
                    x0,
                    x1 - x0,
                    label,
                    truncate_lane_name(span.lane_name),
                    span.boxed,
                )
            )
            span_row.append((span.span_id, row_base + placed))
        n_sublanes = max(1, len(lane_last_end))
        yticks.append(row_base + (n_sublanes - 1) / 2)
        ylabels.append(f"{label} ({len(group)})")
        y = row_base + n_sublanes

    return PackedLanes(rows=rows, span_row=span_row, yticks=yticks, ylabels=ylabels)
