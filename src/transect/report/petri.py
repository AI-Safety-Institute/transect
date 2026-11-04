"""The target side of a Petri report: the Target conversation section
and the exchange cards the phase cards host.

The section draws the stored branch tree as one row per trajectory:
a start circle, the target calls as blocks numbered in recorded order
(surviving in their phase colour, discarded grey), and under them a
coloured band per auditor phase labelled with the phase number; a
child row forks at its parent's fork point, where the circle links to
the rollback's phase card. Each exchange card holds what the auditor
sent and the target's reply, capped like a phase-card excerpt or whole
with ``full_text`` on, with a link to the reply's model event in the
Scout viewer. On a judged run the cards sit inside the phase card of
the auditor turn that drove them, and a marker sits in the phase card
of the turn that rolled back; calls no judged phase covers, and every
call on a structural run, list in the section in recorded order.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from markupsafe import Markup

from transect.petri.conversation import Branch, Conversation, TargetCall
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.excerpts import TEXT_CHARS

_UNIT = 46  # px per target call along a row
_GAP = 24  # px between blocks, where a circle (r=5) sits with 7 px clear each side
_ROW = 50  # px per trajectory row
_LEFT = 40  # px for the row labels
_TOP = 44  # px above the first row: the axis caption and a segment label
_CAPTION = 200  # px the axis caption needs (measured at 0.68rem in Chromium)
_tpl: Any = jinja_env().get_template("petri.html.j2").module


@dataclass
class TargetReport:
    """The Target conversation section and the exchange block each
    phase card hosts, by phase index."""

    section: Markup
    card_blocks: dict[int, Markup] = field(default_factory=dict)


def target_report(
    conversation: Conversation | None,
    note: str | None,
    phase_turns: pd.DataFrame,
    colors: dict[str, tuple[str, str | None]],
    idx: int,
    link,
    full_text: bool,
) -> TargetReport:
    """The target report for one transcript: ``idx`` scopes every id on a
    multi-transcript page, ``link`` is the viewer deep-link builder or
    None, ``colors`` the report's phase colours. Without a conversation
    the section states ``note``, the reason none could be read."""
    if conversation is None:
        return TargetReport(
            sections.section("Target conversation", [_tpl.unavailable(note)])
        )
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
    # each call under the phase of the auditor turn that drove it; a
    # rollback marker under the phase of the turn that rolled back
    by_phase: dict[int | None, list[int]] = {}
    for call in conversation.calls:
        phase = (
            phase_of.get(call.auditor_turn) if call.auditor_turn is not None else None
        )
        by_phase.setdefault(phase[0] if phase else None, []).append(call.number)
    openers: dict[int | None, list[Branch]] = {}
    for branch in conversation.branches:
        if branch.calls and branch.rollback_turn is not None:
            phase = phase_of.get(branch.rollback_turn)
            openers.setdefault(phase[0] if phase else None, []).append(branch)
    blocks = {
        index: _tpl.exchanges(
            _items(conversation, numbers, openers.get(index, [])),
            cards,
            "Target exchanges",
        )
        for index, numbers in by_phase.items()
        if index is not None
    }
    unhosted = by_phase.get(None, [])
    listed = (
        _tpl.exchanges(
            _items(conversation, unhosted, openers.get(None, [])),
            cards,
            "Target exchanges without a judged phase" if blocks else "Target exchanges",
            open=not blocks,
        )
        if unhosted
        else None
    )
    diagram = (
        _tpl.diagram(_diagram(conversation, cards, phase_of, idx))
        if conversation.branches
        else None
    )
    section = sections.section(
        "Target conversation",
        [_tpl.intro(conversation, phase_of != {}, bool(blocks)), diagram, listed],
    )
    return TargetReport(section, blocks)


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

    def shown(text: str) -> str:
        return text if full_text else " ".join(text.split())[:TEXT_CHARS]

    text, reasoning = shown(call.text), shown(call.reasoning)
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
        "prompts": [shown(prompt) for prompt in call.prompts],
        "prompt_truncated": not full_text
        and any(len(" ".join(p.split())) > TEXT_CHARS for p in call.prompts),
        "text": text,
        "reasoning": reasoning,
        "truncated": not full_text
        and any(
            len(" ".join(t.split())) > TEXT_CHARS for t in (call.text, call.reasoning)
        ),
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


def _items(
    conversation: Conversation, numbers: list[int], openers: list[Branch]
) -> list[dict]:
    """The exchange list for the calls ``numbers``, in recorded order:
    each call flagged when it is off the surviving path, and a rollback
    marker for each trajectory in ``openers`` before the first listed
    call at or after the trajectory's first call (at the end when none
    follows)."""
    surviving = conversation.surviving
    markers: list[dict[str, Any]] = [
        {
            "kind": "rollback",
            "turn": branch.rollback_turn,
            "trajectory": branch.number,
            "parent": branch.parent,
            "fork_after": branch.fork_after,
            "first_call": branch.calls[0],
        }
        for branch in sorted(openers, key=lambda b: b.calls[0])
    ]
    items: list[dict] = []
    for number in numbers:
        while markers and markers[0]["first_call"] <= number:
            items.append(markers.pop(0))
        items.append(
            {
                "kind": "call",
                "number": number,
                "discarded": surviving is not None and number not in surviving,
            }
        )
    return items + markers


def _diagram(
    conversation: Conversation,
    cards: dict[int, dict],
    phase_of: dict[int, tuple[int, str]],
    idx: int,
) -> dict[str, Any]:
    """Row geometry for the tree, in pixels from the row origin: each
    trajectory's start, its circle, its blocks and its phase segments
    (consecutive calls of one phase), plus the legend's phase labels in
    order of first appearance."""
    surviving = set(conversation.surviving or [])
    starts: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for branch in conversation.branches:
        start = 0
        if branch.parent is not None:
            start = starts[branch.parent] + (branch.fork_after or 0)
        starts[branch.number] = start
        calls: list[dict[str, Any]] = [
            {
                "x": (start + i) * _UNIT + _GAP,
                "card": cards[number],
                "discarded": conversation.surviving is not None
                and number not in surviving,
            }
            for i, number in enumerate(branch.calls)
        ]
        segments: list[dict[str, Any]] = []
        for call in calls:
            card = call["card"]
            if card["phase_index"] is None:
                continue
            if segments and segments[-1]["phase_index"] == card["phase_index"]:
                segments[-1]["x1"] = call["x"] + _UNIT - _GAP
                segments[-1]["n"] += 1
            else:
                segments.append(
                    {
                        "phase_index": card["phase_index"],
                        "phase": card["phase"],
                        "color": card["color"],
                        "href": card["phase_href"],
                        "x0": call["x"],
                        "x1": call["x"] + _UNIT - _GAP,
                        "n": 1,
                    }
                )
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
                "circle_x": start * _UNIT + _GAP // 2,
                "rollback_turn": branch.rollback_turn,
                "rollback_href": f"#phase-{idx}-{phase_of[branch.rollback_turn][0]}"
                if branch.rollback_turn in phase_of
                else None,
                "calls": calls,
                "segments": segments,
            }
        )
    legend: dict[str, str | None] = {}
    for row in rows:
        for segment in row["segments"]:
            legend.setdefault(segment["phase"], segment["color"])
    return {
        "rows": rows,
        "legend": [{"phase": phase, "color": color} for phase, color in legend.items()],
        "left": _LEFT,
        "top": _TOP,
        "unit": _UNIT,
        "gap": _GAP,
        "row_height": _ROW,
        "width": _LEFT + max(_CAPTION, max(r["x1"] for r in rows) + _UNIT),
        "height": _TOP + (len(rows) - 1) * _ROW + 18,
        "path_known": conversation.surviving is not None,
    }
