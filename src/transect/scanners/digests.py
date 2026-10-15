"""Turn digests: what every judge reads per orchestrator turn.

`turn_digests` is the one builder of the lines the phase judge, its
verifier, the narrator and the `digest_turns` loader read; one
`DigestSettings` governs all of them. Digests are prompt-side only and
never stored.
"""

import json
from bisect import bisect_right
from typing import Any

from inspect_ai.tool import ToolCall
from pydantic import BaseModel, Field

from transect.scanners.helpers import (
    Lanes,
    message_reasoning,
    span_task_text,
    strip_subagent_scaffold,
)

_OPENCLAW_FAILED_TURN_PREFIX = "[assistant turn failed"  # placeholder assistant turn
# inspect_ai's handoff tools answer with this boilerplate, not with output
_HANDOFF_RESULT_PREFIX = "Successfully transferred to"

# recorded tool outputs: tool events by call id with their event position,
# and tool messages by call id (see _tool_outputs)
_ToolOutputs = tuple[
    dict[str, list[tuple[int, str, bool]]], dict[str, tuple[str, bool]]
]


class DigestSettings(BaseModel, frozen=True):
    """What every judge reads per orchestrator turn: the character cap
    of each digest part and whether turns whose only content is tool
    calls are judged. The phase judge, its verifier, the narrator and
    the `digest_turns` loader share one object, so every reader sees
    the same lines; Scout records it with the scanner's arguments. A
    part over its cap keeps its start and its end (`clip`); a cap of 0
    leaves the part out.
    """

    text_chars: int = Field(default=300, ge=0)  # turn text, reasoning, each delegation
    final_text_chars: int = Field(default=4000, ge=0)  # the last digest turn with text
    tool_call_chars: int = Field(default=200, ge=0)  # each call's JSON arguments
    tool_result_chars: int = Field(default=0, ge=0)  # each call's recorded output
    tool_only_turns: bool = True  # judge turns whose only content is tool calls


class ToolDigest(BaseModel):
    """One tool call of a digest turn, as shown to the judge."""

    function: str
    arguments: str = ""  # capped JSON arguments; "" when not shown
    result: str = ""  # capped output or error message; "" when not shown
    error: bool = False  # the call recorded an error (only when output is shown)


class Digest(BaseModel):
    """One orchestrator turn, as shown to the judge."""

    turn: int
    text: str = ""
    reasoning: str = ""  # the turn's reasoning-block text, when recorded
    tool_calls: list[ToolDigest] = []
    delegations: list[str] = []
    event_id: str | None = None  # source model event uuid (viewer anchor)


def clip(text: str, cap: int) -> str:
    """Bound a string to ``cap`` characters, keeping its start and its
    end around a ``[... N chars ...]`` marker: a message's closing lines
    often carry its conclusion.

    Args:
        text: The string to bound.
        cap: Maximum characters kept, marker excluded.

    Returns:
        The text unchanged when within the cap, or when the clipped form
        would be no shorter; otherwise its first ``cap // 2`` and last
        ``cap - cap // 2`` characters, whitespace at the cut dropped,
        with a marker between them counting every character they do not
        cover ("" for a cap of 0).
    """
    if len(text) <= cap:
        return text
    if cap <= 0:
        return ""
    head = text[: cap // 2].rstrip()
    tail = text[len(text) - (cap - cap // 2) :].lstrip()
    marker = f"[... {len(text) - len(head) - len(tail)} chars ...]"
    clipped = " ".join(part for part in (head, marker, tail) if part)
    return clipped if len(clipped) < len(text) else text


def digest_line(d: Digest) -> str:
    """Render one digest as a single prompt line: index, reasoning
    behind a [THINKING] marker, text, the tool calls, then each
    delegation behind a [DELEGATES] marker.

    Tool calls render as ``[CALL] name(arguments)`` followed by
    ``[RESULT]`` or ``[ERROR]`` output when the digest carries
    arguments or output, otherwise as the bare ``(tools: ...)`` names."""
    parts = [f"{d.turn}:"]
    if d.reasoning:
        parts.append(f"[THINKING] {d.reasoning}".replace("\n", " "))
    if d.text:
        parts.append(d.text.replace("\n", " "))
    if any(c.arguments or c.result or c.error for c in d.tool_calls):
        parts.extend(_call_part(c) for c in d.tool_calls)
    elif d.tool_calls:
        parts.append(f"(tools: {', '.join(c.function for c in d.tool_calls)})")
    for goal in d.delegations:
        parts.append(f"[DELEGATES] {goal}".replace("\n", " "))
    return " ".join(parts)


def turn_digests(transcript: Any, digest: DigestSettings | None = None) -> list[Digest]:
    """Build one digest per orchestrator turn that has content to judge.

    Digests are a sparse selection over the orchestrator's turns
    (`helpers.orchestrator_turns`, the turn axis): a turn is eligible
    when it carries visible text, reasoning-block content (when the
    source records it), a delegation, or (with
    ``digest.tool_only_turns``) a tool call. Provider-failure
    placeholder turns never are. Sub-agent activity lives in child
    spans and is represented only by delegation lines, folded in at
    the last eligible turn preceding each span_begin.

    A tool call's output is the first tool event with its id after the
    turn, else the tool message that answered the call (sources
    without tool events). A spawning call, one whose arguments carry
    the ``task``/``prompt`` a delegation line shows, and a handoff
    call, whose output is transfer boilerplate, render as bare names.

    A part over its cap keeps its start and its end (`clip`). The last
    digest turn with text keeps up to ``final_text_chars`` of it.

    Args:
        transcript: The transcript to digest (Scout ``Transcript`` or
            any object with compatible ``events``/``messages``).
        digest: The caps and the tool-only-turn switch
            (`DigestSettings`); ``None`` is the defaults.

    Returns:
        ``Digest`` records in turn order.
    """
    settings = digest or DigestSettings()
    text_chars = settings.text_chars
    lanes = Lanes(transcript)
    outputs = _tool_outputs(transcript) if settings.tool_result_chars else None
    by_turn: dict[int, Digest] = {}

    def _digest(turn: int) -> Digest:
        if turn not in by_turn:
            by_turn[turn] = Digest(turn=turn)
        return by_turn[turn]

    eligible_turns: list[int] = []  # orchestrator, non-failed: anchor targets
    final: tuple[int, str] | None = None  # last digest turn with text, uncapped
    for turn, ev, calls in lanes.turns():
        event: Any = ev
        if _is_failed_turn(event):
            continue  # provider-failure placeholder: not a turn to judge
        eligible_turns.append(turn)
        message = event.output.message
        text = (message.text or "").strip() if message else ""
        reasoning = message_reasoning(message)
        delegations: list[str] = []
        if not lanes.begins:  # span-less sources: delegations ride tool-call args
            for call in calls:
                goal = _spawn_goal(call)
                if goal:
                    delegations.append(clip(goal, text_chars))
        if not (text or reasoning or delegations) and not (
            calls and settings.tool_only_turns
        ):
            continue  # nothing classifiable in this turn
        record = _digest(turn)
        record.text = clip(text, text_chars)
        record.reasoning = clip(reasoning, text_chars)
        if text:
            final = (turn, text)
        record.tool_calls = [
            _tool_digest(call, outputs, lanes.position(event), settings)
            for call in calls
        ]
        record.delegations.extend(delegations)
        record.event_id = getattr(event, "uuid", None)

    # Sub-agent-span delegations, snapped to the nearest digest-eligible
    # orchestrator turn at or before the span_begin.
    for span, raw_anchor in _span_anchors(lanes):
        if not eligible_turns:
            continue
        position = bisect_right(eligible_turns, raw_anchor) - 1
        anchor = eligible_turns[position] if position >= 0 else eligible_turns[0]
        text = span_task_text(span, lanes.first_models.get(span.id))[0]
        goal = text.splitlines()[0].strip() if text else str(span.name)
        _digest(anchor).delegations.append(clip(goal, text_chars))

    if final is not None:
        turn, text = final
        by_turn[turn].text = clip(text, max(settings.final_text_chars, text_chars))
    return [by_turn[turn] for turn in sorted(by_turn)]


def _span_anchors(lanes: Lanes) -> list[tuple[Any, int]]:
    """Each sub-agent span_begin with its raw anchor: the last
    orchestrator turn preceding it in event order (0 when none does, as
    `token_timeline`'s ``spawn_turn``)."""
    return [
        (event, max(before - 1, 0))
        for before, event in lanes.events_before_turn()
        if event.event == "span_begin" and event.id in lanes.sub_ids
    ]


def _is_failed_turn(event: Any) -> bool:
    """Detect an OpenClaw provider-failure placeholder turn: the scaffold
    writes a fake assistant turn ("[assistant turn failed ...]") when a
    provider call dies."""
    message = event.output.message
    text = (message.text or "").strip().lower() if message else ""
    return text.startswith(_OPENCLAW_FAILED_TURN_PREFIX)


def _spawn_goal(call: ToolCall) -> str:
    """The delegation a spawning call carries: its ``task``/``prompt``
    argument behind its label, scaffold stripped; "" for other calls."""
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    task = arguments.get("task") or arguments.get("prompt")
    if not task:
        return ""
    label = arguments.get("label") or arguments.get("description")
    return (f"[{label}] " if label else "") + strip_subagent_scaffold(str(task))


def _tool_digest(
    call: ToolCall,
    outputs: _ToolOutputs | None,
    position: int,
    settings: DigestSettings,
) -> ToolDigest:
    """One tool call as a digest shows it, within the per-call caps.
    ``position`` is the calling model event's index in the event list;
    ``outputs`` is None when output is switched off."""
    spawning = bool(_spawn_goal(call))
    arguments = ""
    if settings.tool_call_chars and call.arguments and not spawning:
        arguments = json.dumps(call.arguments, ensure_ascii=False, default=str)
    result, error = ("", False)
    if outputs is not None and not spawning:
        result, error = _tool_output(outputs, call.id, position)
        if result.startswith(_HANDOFF_RESULT_PREFIX):
            result = ""
    return ToolDigest(
        function=call.function,
        arguments=clip(arguments, settings.tool_call_chars),
        result=clip(result, settings.tool_result_chars),
        error=error,
    )


def _tool_outputs(transcript: Any) -> _ToolOutputs:
    """The transcript's recorded tool outputs: tool events keyed by call
    id with their event position, and tool messages keyed by call id as
    the fallback for sources without tool events (``(text, is_error)``,
    whitespace-flattened)."""
    events: dict[str, list[tuple[int, str, bool]]] = {}
    messages: dict[str, tuple[str, bool]] = {}
    for message in transcript.messages or []:
        call_id = getattr(message, "tool_call_id", None)
        if getattr(message, "role", None) != "tool" or not call_id:
            continue
        error = getattr(message, "error", None)
        text = error.message if error is not None else (message.text or "")
        messages.setdefault(call_id, (_flat(text), error is not None))
    for position, event in enumerate(transcript.events):
        if getattr(event, "event", None) != "tool" or not event.id:
            continue
        error = event.error
        text = error.message if error is not None else _result_text(event.result)
        events.setdefault(event.id, []).append(
            (position, _flat(text), error is not None)
        )
    return events, messages


def _tool_output(
    outputs: _ToolOutputs, call_id: str | None, position: int
) -> tuple[str, bool]:
    """The output of the call ``call_id`` made at event ``position``: the
    first tool event with that id after the turn, so a reused id
    (OpenClaw sessions concatenated into one lane) resolves to its own
    event; else the answering tool message; a call without an id gets
    none."""
    if not call_id:
        return "", False
    events, messages = outputs
    for event_position, text, error in events.get(call_id, []):
        if event_position > position:
            return text, error
    return messages.get(call_id, ("", False))


def _flat(text: str) -> str:
    """Whitespace-flattened text."""
    return " ".join(text.split())


def _result_text(result: Any) -> str:
    """A tool event's result as text: strings as recorded, the text of
    content blocks, other scalars by ``str``; non-text blocks are left out."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return " ".join(
            text for item in result if (text := getattr(item, "text", None))
        )
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return "" if result is None else str(result)


def _call_part(call: ToolDigest) -> str:
    """One tool call's part of a digest line."""
    part = f"[CALL] {call.function}"
    if call.arguments:
        part += f"({call.arguments})"
    if call.result or call.error:
        part += f" [{'ERROR' if call.error else 'RESULT'}] {call.result}".rstrip()
    return part.replace("\n", " ")
