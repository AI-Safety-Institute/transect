"""The target's side of a Petri audit: every target call, the stored
branch tree and the surviving conversation.

Branches come from the audit's stored ``target`` timeline, one span per
trajectory with ``branched_from`` naming the anchor it forked from. The
surviving conversation is the transcript's final message list: the last
target reply in it is the endpoint, and the path runs from there back
through each branch's fork point. Nothing is inferred when either is
missing; `Conversation.note` says what is absent.
"""

from dataclasses import dataclass, field
from typing import Any

from inspect_ai.event import TimelineEvent, TimelineSpan

from transect.scanners.digests import _result_text
from transect.scanners.helpers import (
    Lanes,
    is_model_turn,
    message_reasoning,
    turn_message,
)

_ROLLBACK_TOOLS = frozenset({"rollback_conversation", "restart_conversation"})
_SEND_TOOL = "send_message"  # the auditor tool whose ``message`` the target reads


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
    it (the auditor turn preceding it in event order), None when none
    did; ``prompts`` are the ``send_message`` texts the auditor sent
    since the previous target call or rollback, in order, empty when
    none is recorded.
    """

    number: int
    attempt: int
    lane_turn: int
    auditor_turn: int | None
    event_id: str | None
    message_id: str | None
    model: str | None
    text: str
    reasoning: str
    prompts: list[str] = field(default_factory=list)
    tools: list[ToolUse] = field(default_factory=list)


@dataclass(frozen=True)
class Branch:
    """One trajectory in the stored target tree, numbered by creation
    order.

    ``fork_after`` is how many of the parent's calls precede the fork: 0
    for a restart, None on the root or when the parent holds no matching
    anchor; ``rollback_turn`` is the auditor turn whose rollback or
    restart tool call opened the trajectory's span.
    """

    number: int
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

    @property
    def attempts(self) -> int:
        """How many trajectories ran a target call."""
        return max(c.attempt for c in self.calls)


def target_conversation(transcript: Any) -> Conversation | None:
    """The target conversation of a Petri transcript, read from its events,
    stored timelines and final messages; None when no target call exists.

    Args:
        transcript: A Scout transcript read with its messages, its
            timelines, and its model, span, tool, anchor and branch
            events.
    """
    lanes = Lanes(transcript)
    timeline, note = _target_timeline(transcript)
    # one numbering for trajectories: the tree's spans by creation order,
    # then any role lane the tree does not hold
    order = {e.id: i for i, e in enumerate(lanes.events) if e.event == "span_begin"}
    tree = _tree_spans(timeline.root) if timeline is not None else {}
    ranked = sorted(tree, key=lambda span_id: order.get(span_id, len(order)))
    numbers = {span_id: n for n, span_id in enumerate(ranked, 1)}
    for begin in lanes.role_begins:
        numbers.setdefault(begin.id, len(numbers) + 1)
    lane_ids = {begin.id for begin in lanes.role_begins}
    results = {e.id: e for e in lanes.events if e.event == "tool"}
    calls: list[TargetCall] = []
    lane_turns: dict[str, int] = {}
    sent: list[str] = []
    for before, event in lanes.events_before_turn():
        if event.event == "tool":
            if event.function == _SEND_TOOL:
                arguments = event.arguments if isinstance(event.arguments, dict) else {}
                if arguments.get("message"):
                    sent.append(str(arguments["message"]))
            elif event.function in _ROLLBACK_TOOLS:
                sent = []
        lane = lanes.lane_of(event) if is_model_turn(event) else None
        if lane is None or lane.id not in lane_ids:
            continue
        message = turn_message(event)
        lane_turn = lane_turns.get(lane.id, 0)
        lane_turns[lane.id] = lane_turn + 1
        calls.append(
            TargetCall(
                number=len(calls) + 1,
                attempt=numbers[lane.id],
                lane_turn=lane_turn,
                auditor_turn=before - 1 if before else None,
                event_id=getattr(event, "uuid", None),
                message_id=getattr(message, "id", None),
                model=getattr(event, "model", None),
                text=(message.text or "") if message else "",
                reasoning=message_reasoning(message) if message else "",
                prompts=sent,
                tools=[
                    ToolUse(
                        name=call.function,
                        arguments=dict(call.arguments),
                        result=_result_text(results[call.id].result)
                        if call.id in results
                        else None,
                        error=_error_text(results.get(call.id)),
                    )
                    for call in (message.tool_calls if message else None) or []
                ],
            )
        )
        sent = []
    if not calls:
        return None
    if timeline is None:
        return Conversation(calls, [], None, note)
    branches = _branches(timeline, tree, calls, lanes, numbers)
    placed = sorted(n for b in branches for n in b.calls)
    if placed != [c.number for c in calls]:
        return Conversation(
            calls, [], None, "the stored target tree does not hold every target call"
        )
    surviving, note = _surviving(branches, calls, getattr(transcript, "messages", []))
    return Conversation(calls, branches, surviving, note)


def _target_timeline(transcript: Any) -> tuple[Any | None, str | None]:
    """The stored target branch tree (the timeline named ``target`` or
    ``<audit_name>-target`` whose root is a branch span), or why none
    can be used."""
    found = [
        t
        for t in getattr(transcript, "timelines", None) or []
        if (t.name == "target" or t.name.endswith("-target"))
        and t.root.span_type == "branch"
    ]
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, "no target timeline is stored"
    names = ", ".join(t.name for t in found)
    return None, f"several target timelines are stored ({names})"


def _tree_spans(span: TimelineSpan) -> dict[str, TimelineSpan]:
    """Every span of a branch tree by id."""
    spans = {span.id: span}
    for child in span.branches:
        spans.update(_tree_spans(child))
    return spans


def _branches(
    timeline: Any,
    spans: dict[str, TimelineSpan],
    calls: list[TargetCall],
    lanes: Lanes,
    numbers: dict[str, int],
) -> list[Branch]:
    """Project the stored tree: each span's calls in order, its fork point
    in the parent, and the auditor turn whose rollback or restart opened
    its span."""
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
        elif event.event == "span_begin" and event.id in spans:
            if pending is not None:
                opener[event.id] = pending
            pending = None
        elif is_model_turn(event):
            pending = None
    branches: list[Branch] = []

    def walk(span: TimelineSpan, parent: TimelineSpan | None) -> None:
        own = [
            call_of_event[item.event.uuid]
            for item in span.content
            if isinstance(item, TimelineEvent)
            and item.event.event == "model"
            and item.event.uuid in call_of_event
        ]
        fork_after = None
        if parent is not None:
            fork_after = (
                _calls_before_anchor(parent, span.branched_from, call_of_event)
                if span.branched_from
                else 0
            )
        branches.append(
            Branch(
                number=numbers[span.id],
                parent=numbers[parent.id] if parent is not None else None,
                fork_after=fork_after,
                calls=own,
                rollback_turn=opener.get(span.id),
            )
        )
        for child in span.branches:
            walk(child, span)

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


def _error_text(event: Any) -> str | None:
    error = getattr(event, "error", None)
    return str(getattr(error, "message", error)) if error else None
