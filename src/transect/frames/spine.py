"""The orchestrator turn axis as a time scale.

Orchestrator turn ``m`` occupies the wall-clock interval from the start
of model call ``m`` to the start of call ``m + 1`` (the call plus
whatever tool execution followed it): its cell. A sub-agent's activity
is related to the axis by asking which cells its timestamps fall in,
so a span's extent reads as "the orchestrator turns active while it
ran", never as a turn count of its own. The frames keep the integer
answers (`span_turns`); the report maps the same timestamps to
fractional positions (`position`) when it draws.
"""

from datetime import datetime
from itertools import pairwise
from statistics import median
from typing import Any

Cells = list[tuple[float, float]]
"""Per orchestrator turn, ``(start, right edge)`` in POSIX seconds."""


def clock(starts: list[Any], last_completed: Any) -> Cells:
    """The turn cells from the orchestrator's recorded call times, or no
    cells at all when the clock is unusable.

    ``starts`` are each call's start in turn order and
    ``last_completed`` the last call's completion, as ISO strings (or
    None where unrecorded). A usable clock is complete and
    non-decreasing; a missing stamp or a skew backwards would place
    spans at the wrong turns, so it yields ``[]`` and callers fall back
    to event order.
    """
    stamps = [parse(value) for value in starts]
    times = [t for t in stamps if t is not None]
    if not times or len(times) != len(stamps):
        return []
    if any(a > b for a, b in pairwise(times)):
        return []
    return cells(times, parse(last_completed))


def cells(starts: list[datetime], last_completed: datetime | None) -> Cells:
    """Build the turn cells from the orchestrator's call start times.

    The last cell's right edge is the last call's completion when it is
    recorded and later than the start, else the start plus the median
    cell width (the start itself with one turn and no completion).
    """
    if not starts:
        return []
    s = [t.timestamp() for t in starts]
    out: Cells = list(pairwise(s))
    if last_completed is not None and last_completed.timestamp() > s[-1]:
        right = last_completed.timestamp()
    else:
        widths = [b - a for a, b in out if b > a]
        right = s[-1] + (median(widths) if widths else 0.0)
    out.append((s[-1], right))
    return out


def position(t: datetime, cells: Cells) -> float:
    """Map a timestamp onto the axis.

    Inside cell ``m`` the position is ``m - 0.5`` plus the fraction of
    the cell elapsed (turn ``m`` is drawn over ``[m - 0.5, m + 0.5]``);
    a zero-width cell is skipped. Times before the first call clamp to
    ``-0.5``, times at or after the last cell's right edge to the
    axis's right edge.
    """
    if not cells:
        return -0.5
    ts = t.timestamp()
    if ts < cells[0][0]:
        return -0.5
    if ts >= cells[-1][1]:
        return len(cells) - 0.5
    for m, (start, right) in enumerate(cells):
        if start <= ts < right:
            return m - 0.5 + (ts - start) / (right - start)
    return len(cells) - 0.5


def turn_of(t: datetime, cells: Cells) -> int:
    """The orchestrator turn whose cell holds a timestamp, clamped to
    the axis."""
    return min(max(int(position(t, cells) + 0.5), 0), len(cells) - 1)


def span_turns(
    cells: Cells,
    *,
    spawn_turn: int,
    started_at: Any,
    ended_at: Any,
    event_order_end_turn: Any,
) -> dict[str, Any]:
    """A span's ``anchor_turn`` / ``end_turn`` / ``turn_source`` (the
    `frames.subagents` columns).

    With a usable clock and both ISO timestamps, the turns are the cells
    holding the span's first activity and its end. Otherwise the span
    sits at its spawn turn, ending at the event-order end when one was
    recorded, and ``turn_source`` says so.
    """
    first = parse(started_at)
    last = parse(ended_at)
    if cells and first is not None and last is not None:
        anchor = turn_of(first, cells)
        return {
            "anchor_turn": anchor,
            "end_turn": max(turn_of(last, cells), anchor),
            "turn_source": "timestamp",
        }
    return {
        "anchor_turn": spawn_turn,
        "end_turn": spawn_turn
        if event_order_end_turn is None
        else int(event_order_end_turn),
        "turn_source": "event_order",
    }


def parse(value: Any) -> datetime | None:
    """ISO text to datetime; None for anything else."""
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None
