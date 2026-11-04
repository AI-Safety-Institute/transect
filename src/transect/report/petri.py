"""The target side of a Petri report: the Target conversation section
and the exchange blocks the phase cards host.

The section draws the stored branch tree as one row per trajectory:
a start circle, the target calls as blocks numbered in recorded order
(surviving in their phase colour, discarded grey), and under them a
coloured band per auditor phase labelled with the phase number; a
child row forks at its parent's fork point, where the circle links to
the rollback's phase card. An exchange block lists, in recorded order,
each auditor turn that drove a target call (its text, reasoning and
tool calls, once) with one card per call under it (what the auditor
put before the target, the reply, its tool calls), each part capped
like a phase-card excerpt or whole with ``full_text`` on, each model
and tool event linked into the Scout viewer, and a marker where a
rollback opened a trajectory. On a judged run the blocks sit inside the
phase card of the auditor turn that drove the calls, the marker in the
phase card of the turn that rolled back; calls and markers no judged
phase covers, and every one on a structural run, list in the section.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from markupsafe import Markup

from transect.petri.conversation import AuditorTurn, Branch, Conversation, TargetCall
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.excerpts import TEXT_CHARS

_UNIT = 46  # px per target call along a row
_GAP = 24  # px between blocks, where a circle (r=5) sits with 7 px clear each side
_ROW = 50  # px per trajectory row
_LEFT = 40  # px for the row labels
_TOP = 44  # px above the first row: the axis caption and a segment label
_CAPTION = 200  # px the axis caption needs (measured at 0.68rem in Chromium)
_ARGS_CHARS = 120  # a tool call's arguments on its one-line card entry
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
    the section states ``note``, the reason none could be read. A call
    goes under the phase of the auditor turn that drove it, a rollback
    marker under the phase of the turn that rolled back; either without
    a judged phase goes to the section."""
    if conversation is None:
        return TargetReport(
            sections.section("Target conversation", [_tpl.unavailable(note)])
        )
    phase_of = {
        int(turn): (int(index), str(label))
        for turn, index, label in zip(
            phase_turns.turn, phase_turns.phase_index, phase_turns.phase, strict=True
        )
        if pd.notna(index)
    }
    surviving = (
        set(conversation.surviving) if conversation.surviving is not None else None
    )
    cards = {
        call.number: _card(call, surviving, phase_of, colors, idx, link, full_text)
        for call in conversation.calls
    }
    auditors = {
        turn: _auditor(auditor, link, full_text)
        for turn, auditor in conversation.auditor_turns.items()
    }
    by_phase: dict[int | None, list[int]] = {}
    for card in cards.values():
        by_phase.setdefault(card["phase_index"], []).append(card["number"])
    openers: dict[int | None, list[Branch]] = {}
    for branch in conversation.branches:
        if branch.parent is None or not branch.calls:
            continue
        phase = (
            phase_of.get(branch.rollback_turn)
            if branch.rollback_turn is not None
            else None
        )
        openers.setdefault(phase[0] if phase else None, []).append(branch)
    blocks = {
        index: _tpl.exchanges(
            _items(
                by_phase.get(index, []), openers.get(index, []), cards, auditors, link
            ),
            "Target exchanges",
        )
        for index in sorted(key for key in {*by_phase, *openers} if key is not None)
    }
    listed = None
    if by_phase.get(None) or openers.get(None):
        listed = _tpl.exchanges(
            _items(
                by_phase.get(None, []), openers.get(None, []), cards, auditors, link
            ),
            "Target exchanges without a judged phase" if blocks else "Target exchanges",
            open=not blocks,
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
    surviving: set[int] | None,
    phase_of: dict[int, tuple[int, str]],
    colors: dict[str, tuple[str, str | None]],
    idx: int,
    link,
    full_text: bool,
) -> dict[str, Any]:
    """One target call as its exchange card and diagram block read it:
    its phase (that of the auditor turn that drove it), whether the
    surviving path discarded it (None when no path is known), what the
    auditor put before it and its reply, each part capped unless
    ``full_text``, each event linked through ``link``."""
    phase = phase_of.get(call.auditor_turn) if call.auditor_turn is not None else None
    return {
        **_call(call, link, full_text),
        "id": f"petri-{idx}-call-{call.number}",
        "number": call.number,
        "attempt": call.attempt,
        "lane_turn": call.lane_turn,
        "auditor_turn": call.auditor_turn,
        "phase_index": phase[0] if phase else None,
        "phase": phase[1] if phase else None,
        "phase_href": f"#phase-{idx}-{phase[0]}" if phase else None,
        "color": colors[phase[1]][0] if phase and phase[1] in colors else None,
        "discarded": call.number not in surviving if surviving is not None else None,
        "prompts": [
            {
                "kind": prompt.kind,
                "text": _shown(prompt.text, full_text),
                "href": _href(link, prompt.event_id),
            }
            for prompt in call.prompts
        ],
        "prompt_truncated": _cut(full_text, *(p.text for p in call.prompts)),
    }


def _auditor(auditor: AuditorTurn, link, full_text: bool) -> dict[str, Any]:
    """One auditor turn as the exchange block heads it: its text,
    reasoning and tool calls, the prompt tools left to the cards."""
    return {**_call(auditor, link, full_text), "turn": auditor.turn}


def _call(call: Any, link, full_text: bool) -> dict[str, Any]:
    """The parts every model call renders: text, reasoning, message
    count, tool calls, viewer link."""
    return {
        "text": _shown(call.text, full_text),
        "reasoning": _shown(call.reasoning, full_text),
        "truncated": _cut(full_text, call.text, call.reasoning),
        "n_input": call.n_input,
        "model": call.model,
        "full_text": full_text,
        "args_chars": _ARGS_CHARS,
        "href": _href(link, call.event_id),
        "tools": [
            {
                "name": tool.name,
                "arguments": json.dumps(
                    tool.arguments, ensure_ascii=False, default=str
                ),
                "result": tool.result,
                "error": tool.error,
                "href": _href(link, tool.event_id),
            }
            for tool in call.tools
        ],
    }


def _items(
    numbers: list[int],
    openers: list[Branch],
    cards: dict[int, dict],
    auditors: dict[int, dict],
    link,
) -> list[dict]:
    """The exchange list for the calls ``numbers``, in recorded order:
    consecutive calls of one auditor turn grouped under that turn, and
    a rollback marker for each trajectory in ``openers`` before the
    first listed call at or after the trajectory's first call (at the
    end when none follows), linked to the rollback's tool event."""
    markers: list[dict[str, Any]] = [
        {
            "kind": "rollback",
            "turn": branch.rollback_turn,
            "trajectory": branch.number,
            "parent": branch.parent,
            "fork_after": branch.fork_after,
            "first_call": branch.calls[0],
            "href": _href(link, branch.rollback_event_id),
        }
        for branch in sorted(openers, key=lambda b: b.calls[0])
    ]
    items: list[dict[str, Any]] = []
    for number in numbers:
        while markers and markers[0]["first_call"] <= number:
            items.append(markers.pop(0))
        card = cards[number]
        turn = card["auditor_turn"]
        if items and items[-1]["kind"] == "turn" and items[-1]["turn"] == turn:
            items[-1]["calls"].append(card)
        else:
            items.append(
                {
                    "kind": "turn",
                    "turn": turn,
                    "auditor": auditors.get(turn) if turn is not None else None,
                    "calls": [card],
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
    starts: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for branch in conversation.branches:
        start = 0
        if branch.parent is not None:
            start = starts[branch.parent] + (branch.fork_after or 0)
        starts[branch.number] = start
        calls: list[dict[str, Any]] = [
            {"x": (start + i) * _UNIT + _GAP, "card": cards[number]}
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
        "height": _TOP + (len(rows) - 1) * _ROW + 30,
        "path_known": conversation.surviving is not None,
    }


def _shown(text: str, full_text: bool) -> str:
    """A text part as a card shows it: whole, or whitespace-flattened
    and capped at the excerpt length."""
    return text if full_text else " ".join(text.split())[:TEXT_CHARS]


def _cut(full_text: bool, *texts: str) -> bool:
    """Whether capping cut any of ``texts``."""
    return not full_text and any(
        len(" ".join(text.split())) > TEXT_CHARS for text in texts
    )


def _href(link, event_id: str | None) -> str | None:
    """The viewer link of an event, None without a builder or an id."""
    return link(event_id) if link and event_id else None
