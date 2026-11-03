"""The target's side of a Petri audit: every target call, the stored
branch tree and the surviving conversation.

Branches come from the audit's stored ``target`` timeline, one span per
trajectory with ``branched_from`` naming the anchor it forked from. The
surviving conversation is the transcript's final message list: the last
target reply in it is the endpoint, and the path runs from there back
through each branch's fork point. Nothing is inferred when either is
missing; `Conversation.note` says what is absent.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.event import TimelineEvent, TimelineSpan

from transect.scanners.helpers import (
    Lanes,
    is_model_turn,
    message_reasoning,
    turn_message,
)

_BRANCH_NAME = re.compile(r"^branch (\d+)$")
_ROLLBACK_TOOLS = frozenset({"rollback_conversation", "restart_conversation"})


@dataclass(frozen=True)
class ToolUse:
    """One tool call a target reply made, with its recorded result."""

    name: str
    arguments: dict[str, Any]
    result: str | None
    error: str | None


@dataclass(frozen=True)
class TargetCall:
    """One target model call.

    ``number`` counts target calls in event order from 1; ``attempt``
    is the trajectory (role lane) the call ran in, ``lane_turn`` its
    ordinal there; ``auditor_turn`` is the orchestrator turn that drove
    it (the auditor turn preceding it in event order).
    """

    number: int
    attempt: int
    lane_turn: int
    auditor_turn: int
    event_id: str | None
    message_id: str | None
    model: str | None
    text: str
    reasoning: str
    tools: list[ToolUse] = field(default_factory=list)


@dataclass(frozen=True)
class Branch:
    """One trajectory in the stored target tree.

    ``fork_after`` is how many of the parent's calls precede the fork
    (None on the root, or when the parent holds no matching anchor);
    ``rollback_turn`` is the auditor turn whose rollback or restart
    opened the branch, when a tool call precedes its branch event.
    """

    number: int
    span_id: str
    parent: int | None
    fork_after: int | None
    calls: list[int]
    rollback_turn: int | None


@dataclass(frozen=True)
class Conversation:
    """The target's calls, the stored branches and the surviving path.

    ``surviving`` lists the call numbers of the final conversation in
    order, None when the tree or the endpoint is unavailable; ``note``
    then says why.
    """

    calls: list[TargetCall]
    branches: list[Branch]
    surviving: list[int] | None
    note: str | None = None

    @property
    def discarded(self) -> list[int]:
        """The calls off the surviving path, in event order."""
        kept = set(self.surviving or [])
        return [c.number for c in self.calls if c.number not in kept]


def target_conversation(transcript: Any) -> Conversation | None:
    """The target conversation of a Petri transcript, read from its events,
    stored timelines and final messages; None when no target call exists.

    Args:
        transcript: A Scout transcript read with its messages, its
            timelines, and its model, span, tool, anchor and branch
            events.
    """
    lanes = Lanes(transcript)
    attempts = {begin.id: n for n, begin in enumerate(lanes.role_begins, 1)}
    results = {e.id: e for e in lanes.events if e.event == "tool"}
    calls: list[TargetCall] = []
    lane_turns: dict[str, int] = {}
    for before, event in lanes.events_before_turn():
        lane = lanes.lane_of(event) if is_model_turn(event) else None
        if lane is None or lane.id not in attempts:
            continue
        message = turn_message(event)
        lane_turn = lane_turns.get(lane.id, 0)
        lane_turns[lane.id] = lane_turn + 1
        calls.append(
            TargetCall(
                number=len(calls) + 1,
                attempt=attempts[lane.id],
                lane_turn=lane_turn,
                auditor_turn=max(before - 1, 0),
                event_id=getattr(event, "uuid", None),
                message_id=getattr(message, "id", None),
                model=getattr(event, "model", None),
                text=(message.text or "") if message else "",
                reasoning=message_reasoning(message) if message else "",
                tools=[
                    ToolUse(
                        name=call.function,
                        arguments=dict(call.arguments),
                        result=_result_text(results.get(call.id)),
                        error=_error_text(results.get(call.id)),
                    )
                    for call in (message.tool_calls if message else None) or []
                ],
            )
        )
    if not calls:
        return None
    timeline = _target_timeline(transcript)
    if timeline is None:
        return Conversation(calls, [], None, "no target timeline is stored")
    branches = _branches(timeline, calls, lanes)
    placed = sorted(n for b in branches for n in b.calls)
    if placed != [c.number for c in calls]:
        return Conversation(
            calls, [], None, "the stored target tree does not hold every target call"
        )
    surviving, note = _surviving(branches, calls, getattr(transcript, "messages", []))
    return Conversation(calls, branches, surviving, note)


def _target_timeline(transcript: Any) -> Any | None:
    """The stored target branch tree: the timeline named ``target`` or
    ``<audit_name>-target`` whose root is a branch span."""
    found = [
        t
        for t in getattr(transcript, "timelines", None) or []
        if (t.name == "target" or t.name.endswith("-target"))
        and t.root.span_type == "branch"
    ]
    return found[0] if len(found) == 1 else None


def _branches(timeline: Any, calls: list[TargetCall], lanes: Lanes) -> list[Branch]:
    """Project the stored tree: each span's calls in order, its fork point
    in the parent, and the auditor turn that opened it."""
    call_of_event = {c.event_id: c.number for c in calls}
    turn_of_call = {
        call.id: turn
        for turn, _event, tool_calls in lanes.turns()
        for call in tool_calls
    }
    opener: dict[str, int] = {}
    pending: int | None = None
    for event in lanes.events:
        if event.event == "tool" and event.function in _ROLLBACK_TOOLS:
            pending = turn_of_call.get(event.id)
        elif event.event == "branch":
            if pending is not None:
                opener[event.span_id] = pending
            pending = None
    branches: list[Branch] = []

    def walk(span: TimelineSpan, parent: Branch | None) -> None:
        items = [item for item in span.content if isinstance(item, TimelineEvent)]
        own = [
            call_of_event[item.event.uuid]
            for item in items
            if item.event.event == "model" and item.event.uuid in call_of_event
        ]
        fork_after = None
        if parent is not None and span.branched_from is not None:
            fork_after = _calls_before_anchor(
                timeline_span(parent.span_id), span.branched_from, call_of_event
            )
        match = _BRANCH_NAME.match(span.name)
        branch = Branch(
            number=int(match.group(1)) if match else len(branches) + 1,
            span_id=span.id,
            parent=parent.number if parent else None,
            fork_after=fork_after,
            calls=own,
            rollback_turn=opener.get(span.id),
        )
        branches.append(branch)
        for child in span.branches:
            walk(child, branch)

    spans: dict[str, TimelineSpan] = {}

    def index(span: TimelineSpan) -> None:
        spans[span.id] = span
        for child in span.branches:
            index(child)

    def timeline_span(span_id: str) -> TimelineSpan:
        return spans[span_id]

    index(timeline.root)
    walk(timeline.root, None)
    return sorted(branches, key=lambda b: b.number)


def _calls_before_anchor(
    span: TimelineSpan, anchor_id: str, call_of_event: dict[Any, int]
) -> int | None:
    """How many target calls in a span precede its anchor ``anchor_id``;
    None when the span records no such anchor."""
    count = 0
    for item in span.content:
        if not isinstance(item, TimelineEvent):
            continue
        event = item.event
        if event.event == "anchor" and event.anchor_id == anchor_id:
            return count
        if event.event == "model" and event.uuid in call_of_event:
            count += 1
    return None


def _surviving(
    branches: list[Branch], calls: list[TargetCall], messages: list[Any]
) -> tuple[list[int] | None, str | None]:
    """The surviving path: the final message list's last target reply,
    then each ancestor branch up to its fork point."""
    by_message = {c.message_id: c.number for c in calls if c.message_id}
    endpoint = next(
        (by_message[m.id] for m in reversed(messages) if m.id in by_message), None
    )
    if endpoint is None:
        return None, "the final conversation holds no target reply"
    by_number = {b.number: b for b in branches}
    branch = next(b for b in branches if endpoint in b.calls)
    path = branch.calls[: branch.calls.index(endpoint) + 1]
    while branch.parent is not None:
        fork = branch.fork_after
        if fork is None:
            return None, f"branch {branch.number} has no fork point in its parent"
        branch = by_number[branch.parent]
        path = branch.calls[:fork] + path
    return path, None


def _result_text(event: Any) -> str | None:
    if event is None or event.result is None:
        return None
    result = event.result
    return result if isinstance(result, str) else str(result)


def _error_text(event: Any) -> str | None:
    error = getattr(event, "error", None)
    return str(getattr(error, "message", error)) if error else None
