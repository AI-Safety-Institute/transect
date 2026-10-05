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
"""Per orchestrator turn, ``(start, right edge)`` in POSIX seconds."""


def cells(starts: list[datetime], last_completed: datetime | None) -> Cells:
    """Build the turn cells from the orchestrator's call start times.

    The last cell's right edge is the last call's completion when it is
    recorded and later than the start, else the start plus the median
    cell width (the start itself with one turn and no completion).

    Args:
        starts: Each orchestrator model call's start, in turn order.
        last_completed: The last call's completion time, if recorded.
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
    """Map a timestamp onto the axis.

    Inside cell ``m`` the position is ``m - 0.5`` plus the fraction of
    the cell elapsed; a zero-width cell is skipped. Times before the
    first call clamp to ``-0.5``, times at or after the last cell's
    right edge to the axis's right edge.
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


def coordinates(record: dict[str, Any], cells: Cells) -> dict[str, Any]:
    """A span record's axis coordinates (the columns `frames.subagents`
    documents under ``start_pos`` .. ``after_last``).

    With usable timestamps the box runs from the span's first activity
    to its recorded end (its last activity when no end was recorded).
    Without them, or without any cells, the span sits at its spawn turn
    as a point and ``position_source`` says so.
    """
    spawn = int(record["spawn_turn"])
    first_at = _parse(record.get("first_at"))
    end_at = _parse(record.get("end_at") if record.get("end_recorded") else None)
    end_at = end_at or _parse(record.get("last_at"))
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
    """The turn whose cell holds an axis position."""
    return min(max(int(pos + 0.5), 0), len(cells) - 1)


def _parse(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None
