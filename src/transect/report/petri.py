"""The target conversation section of a Petri report.

The stored branch tree draws as one row per trajectory, each target
call a block coloured by the auditor phase that drove it, surviving
calls solid and discarded ones muted; a block links to the call's
card. The cards follow the surviving conversation in order, grouped by
auditor phase, with the discarded attempts folded at the fork they
left from. Each card shows the reply capped like a phase-card excerpt,
or whole with ``full_text`` on, and links to its model event in the
Scout viewer.
"""

import json
from itertools import pairwise
from typing import Any

import pandas as pd
from markupsafe import Markup

from transect.petri.conversation import Branch, Conversation, TargetCall
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.excerpts import TEXT_CHARS

_UNIT = 30  # px per target call along a row
_ROW = 44  # px per branch row
_LEFT = 52  # px for the row labels
_tpl: Any = jinja_env().get_template("petri.html.j2").module


def target_section(
    conversation: Conversation | None,
    note: str | None,
    phase_turns: pd.DataFrame,
    colors: dict[str, tuple[str, str | None]],
    idx: int,
    link,
    full_text: bool,
) -> Markup:
    """The section for one transcript: ``idx`` scopes every id on a
    multi-transcript page, ``link`` is the viewer deep-link builder or
    None, ``colors`` the report's phase colours. Without a conversation
    the section states ``note``, the reason none could be read."""
    if conversation is None:
        return sections.section("Target conversation", [_tpl.unavailable(note)])
    prefix = f"petri-{idx}"
    phase_of = {
        int(turn): (int(index), str(label))
        for turn, index, label in zip(
            phase_turns.turn, phase_turns.phase_index, phase_turns.phase, strict=True
        )
        if pd.notna(index)
    }
    cards = {
        call.number: _card(call, phase_of, colors, idx, prefix, link, full_text)
        for call in conversation.calls
    }
    diagram = (
        _tpl.diagram(_diagram(conversation, cards, phase_of, idx))
        if conversation.branches
        else None
    )
    return sections.section(
        "Target conversation",
        [
            _tpl.intro(conversation, phase_of != {}),
            diagram,
            _tpl.cards(_entries(conversation, cards), cards),
        ],
    )


def _card(
    call: TargetCall,
    phase_of: dict[int, tuple[int, str]],
    colors: dict[str, tuple[str, str | None]],
    idx: int,
    prefix: str,
    link,
    full_text: bool,
) -> dict[str, Any]:
    phase = phase_of.get(call.auditor_turn) if call.auditor_turn is not None else None
    text = call.text if full_text else " ".join(call.text.split())
    reasoning = call.reasoning if full_text else " ".join(call.reasoning.split())
    truncated = not full_text and (
        len(text) > TEXT_CHARS or len(reasoning) > TEXT_CHARS
    )
    return {
        "id": f"{prefix}-call-{call.number}",
        "number": call.number,
        "attempt": call.attempt,
        "lane_turn": call.lane_turn,
        "auditor_turn": call.auditor_turn,
        "model": call.model,
        "phase_index": phase[0] if phase else None,
        "phase": phase[1] if phase else None,
        "phase_href": f"#phase-{idx}-{phase[0]}" if phase else None,
        "color": colors[phase[1]][0] if phase and phase[1] in colors else None,
        "text": text[:TEXT_CHARS] if not full_text else text,
        "reasoning": reasoning[:TEXT_CHARS] if not full_text else reasoning,
        "truncated": truncated,
        "full_text": full_text,
        "tools": [
            {
                "name": tool.name,
                "arguments": json.dumps(
                    tool.arguments, ensure_ascii=False, default=str
                ),
                "result": tool.result,
                "error": tool.error,
            }
            for tool in call.tools
        ],
        "open_link": link(call.event_id) if link and call.event_id else None,
    }


def _entries(conversation: Conversation, cards: dict[int, dict]) -> list[dict]:
    """The cards' reading order: surviving calls grouped by phase, with
    the discarded attempts folded in at the fork they left from; every
    call in recorded order, grouped by phase, when no path is known."""
    surviving = conversation.surviving
    if surviving is None:
        return _phase_groups([c.number for c in conversation.calls], cards, True)
    alternatives = _alternatives(conversation)
    entries: list[dict] = []
    group: list[int] = []
    for position, number in enumerate([*surviving, None]):
        if alternatives.get(position):
            entries.extend(_phase_groups(group, cards, True))
            group = []
            entries.append(
                {
                    "kind": "alternatives",
                    "after": surviving[position - 1] if position else None,
                    "branches": [
                        {
                            **attempt,
                            "groups": _phase_groups(attempt["calls"], cards, False),
                        }
                        for attempt in alternatives[position]
                    ],
                }
            )
        if number is not None:
            group.append(number)
    entries.extend(_phase_groups(group, cards, True))
    if alternatives.get(None):
        entries.append(
            {
                "kind": "alternatives",
                "after": None,
                "unplaced": True,
                "branches": [
                    {**attempt, "groups": _phase_groups(attempt["calls"], cards, False)}
                    for attempt in alternatives[None]
                ],
            }
        )
    return entries


def _phase_groups(numbers: list[int], cards: dict[int, dict], surviving) -> list[dict]:
    """Consecutive calls of one auditor phase, in order."""
    groups: list[dict] = []
    for number in numbers:
        card = cards[number]
        if not groups or groups[-1]["phase_index"] != card["phase_index"]:
            groups.append(
                {
                    "kind": "phase",
                    "phase_index": card["phase_index"],
                    "phase": card["phase"],
                    "phase_href": card["phase_href"],
                    "color": card["color"],
                    "surviving": surviving,
                    "calls": [],
                }
            )
        groups[-1]["calls"].append(number)
    return groups


def _alternatives(conversation: Conversation) -> dict[int | None, list[dict]]:
    """Discarded calls by branch, keyed by their position in the surviving
    path: the index of the first surviving call after the fork they left
    from. A path branch's own calls past its child's fork count too, and
    a fork past the path's departure folds at the departure. A branch
    whose fork point is unrecorded is keyed None."""
    surviving = conversation.surviving or []
    by_number = {b.number: b for b in conversation.branches}
    path: list[Branch] = []  # endpoint branch up to the root
    branch = next((b for b in conversation.branches if surviving[-1] in b.calls), None)
    while branch is not None:
        path.append(branch)
        branch = by_number[branch.parent] if branch.parent is not None else None
    before = {b.number: _path_start(by_number, b) for b in path}
    # how many of each path branch's calls the path keeps
    extent = {path[0].number: path[0].calls.index(surviving[-1]) + 1} if path else {}
    for child, parent in pairwise(path):
        extent[parent.number] = child.fork_after or 0
    out: dict[int | None, list[dict]] = {}

    def attach(position: int | None, branch: Branch, calls: list[int]) -> None:
        if calls:
            out.setdefault(position, []).append(
                {
                    "number": branch.number,
                    "rollback_turn": branch.rollback_turn,
                    "calls": calls,
                }
            )

    for branch in path:
        kept = extent[branch.number]
        attach(before[branch.number] + kept, branch, branch.calls[kept:])
    for branch in conversation.branches:
        if branch.number in before:
            continue
        climb, fork = branch, branch.fork_after
        while climb.parent is not None and climb.parent not in before:
            climb = by_number[climb.parent]
            fork = climb.fork_after
        departure = climb.parent
        if departure is None or fork is None:
            attach(None, branch, branch.calls)
        else:
            attach(
                before[departure] + min(fork, extent[departure]), branch, branch.calls
            )
    return {
        position: sorted(attempts, key=lambda a: a["number"])
        for position, attempts in out.items()
    }


def _path_start(by_number: dict[int, Branch], branch: Branch) -> int:
    """How many surviving calls precede a path branch's own calls."""
    start = 0
    while branch.parent is not None:
        start += branch.fork_after or 0
        branch = by_number[branch.parent]
    return start


def _diagram(
    conversation: Conversation,
    cards: dict[int, dict],
    phase_of: dict[int, tuple[int, str]],
    idx: int,
) -> dict[str, Any]:
    """Row geometry for the tree: each branch's start and its calls' x
    positions, in target-call units."""
    surviving = set(conversation.surviving or [])
    starts: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for branch in conversation.branches:
        start = 0
        if branch.parent is not None:
            start = starts[branch.parent] + (branch.fork_after or 0)
        starts[branch.number] = start
        rows.append(
            {
                "number": branch.number,
                "parent_row": next(
                    (i for i, r in enumerate(rows) if r["number"] == branch.parent),
                    None,
                ),
                "unresolved": branch.parent is not None and branch.fork_after is None,
                "x0": start * _UNIT,
                "x1": (start + len(branch.calls)) * _UNIT,
                "rollback_turn": branch.rollback_turn,
                "rollback_href": f"#phase-{idx}-{phase_of[branch.rollback_turn][0]}"
                if branch.rollback_turn in phase_of
                else None,
                "calls": [
                    {
                        "x": (start + i) * _UNIT,
                        "card": cards[number],
                        "surviving": conversation.surviving is not None
                        and number in surviving,
                    }
                    for i, number in enumerate(branch.calls)
                ],
            }
        )
    return {
        "rows": rows,
        "left": _LEFT,
        "unit": _UNIT,
        "row_height": _ROW,
        "width": _LEFT + max(r["x1"] for r in rows) + _UNIT,
        "height": len(rows) * _ROW + 24,
        "path_known": conversation.surviving is not None,
    }
