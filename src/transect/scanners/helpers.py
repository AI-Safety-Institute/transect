"""Shared helpers used by more than one scanner.

The turn axis lives here: `orchestrator_turns` numbers the orchestrator's
model turns, `Lanes` resolves which lane any event belongs to, and the
span text helpers read what a sub-agent was asked and what it did.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from inspect_ai.event import ModelEvent, TimelineEvent, TimelineSpan, timeline_build
from inspect_ai.model import ChatMessageAssistant, ContentReasoning
from inspect_ai.tool import ToolCall

# OpenClaw spawn-prompt scaffold markers
_OPENCLAW_SPAWN_CONTEXT_PREFIX = "[subagent context]"
_OPENCLAW_SPAWN_TASK_MARKER = "[subagent task]"
_OPENCLAW_SPAWN_ROLE_PHRASE = "you are running as a subagent"


@dataclass
class SpanActivity:
    """What a sub-agent span actually did (recorded activity).

    ``turn_texts``: the span's own model-turn texts, in order (empty
    on tool-events-only sources).
    ``tool_counts``: tool calls by function name, first-seen order."""

    turn_texts: list[str] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)


class Lanes:
    """The lanes of one transcript, resolved once.

    ``main`` is the orchestrator's span (`main_span`); its model turns of
    the orchestrator role (`orchestrator_role`) are the axis. ``begins``
    are the lane span_begin events (`subagent_span_begins`, the shared
    definition): sub-agent spans, plus the conversation spans of any
    other model role the main span interleaves (an auditor driving a
    target: each target trajectory is a lane). ``first_models`` is each
    lane's first model event. ``events`` is the transcript's event list.
    Every scanner that attributes events to lanes starts from one of
    these.
    """

    def __init__(self, transcript: Any) -> None:
        self.main = main_span(transcript)
        self.events: list[Any] = list(transcript.events)
        self._transcript = transcript
        self._spans = {e.id: e for e in self.events if e.event == "span_begin"}
        main_models = [
            item.event
            for item in self.main.content
            if isinstance(item, TimelineEvent) and is_model_turn(item.event)
        ]
        self.orchestrator_role = orchestrator_role(
            main_models, getattr(transcript, "model", None)
        )
        self._main_events = frozenset(
            id(event)
            for event in main_models
            if getattr(event, "role", None) == self.orchestrator_role
        )
        self.begins = subagent_span_begins(
            transcript, self.main, self.orchestrator_role
        )
        self.sub_ids = frozenset(begin.id for begin in self.begins)
        self.first_models: dict[str, Any] = {}
        for event in self.events:
            if is_model_turn(event):
                sub = self.sub_agent_of(event)
                if sub is not None:
                    self.first_models.setdefault(sub.id, event)
        self._positions = {id(e): i for i, e in enumerate(self.events)}

    def is_main(self, event: Any) -> bool:
        """Whether a model event is one of the orchestrator's own turns."""
        return id(event) in self._main_events

    def sub_agent_of(self, event: Any) -> Any | None:
        """The lane span_begin an event belongs to: the nearest enclosing
        lane span (`nearest_lane_span`). None when the event is the
        orchestrator's (a folded spawn call, ``agent_span_id`` set,
        included), a utility or wrapper span's, or off the axis (an init
        or scorer call)."""
        if getattr(event, "agent_span_id", None) is not None:
            return None
        return nearest_lane_span(
            self._spans, getattr(event, "span_id", None), self.sub_ids
        )

    def position(self, event: Any) -> int:
        """An event's index in the transcript's event list."""
        return self._positions[id(event)]

    def turns(self) -> Iterator[tuple[int, Any, list[ToolCall]]]:
        """Yield ``(turn, model event, tool calls)`` for the orchestrator's
        turns: `orchestrator_turns` over these resolved lanes."""
        turn = 0
        for event, calls in all_model_turns(self._transcript):
            if self.is_main(event):
                yield turn, event, calls
                turn += 1

    def events_before_turn(self) -> Iterator[tuple[int, Any]]:
        """Yield ``(turn, event)`` for every event that is not an
        orchestrator model turn, ``turn`` being the orchestrator turn
        the event precedes: the axis position of a span begin, a tool
        call or a compaction."""
        turn = 0
        for event in self.events:
            if is_model_turn(event) and self.is_main(event):
                turn += 1
                continue
            yield turn, event


def orchestrator_role(main_models: list[Any], transcript_model: Any) -> str | None:
    """The model role on the axis when the main span's calls carry
    several: the role that acts through tools (the most tool calls
    requested across its turns - an auditor driving a target through
    send_message / rollback tools, an agent beside a critic), else the
    role-less calls (the agent's own ``generate()``), else the role whose
    calls ran the transcript's recorded model, else the first call's
    role. A span of one role keeps that role."""
    roles: list[str | None] = []
    calls_by_role: dict[str | None, int] = {}
    for event in main_models:
        role = getattr(event, "role", None)
        if role not in roles:
            roles.append(role)
        message = event.output.message if event.output is not None else None
        n_calls = len(getattr(message, "tool_calls", None) or []) if message else 0
        calls_by_role[role] = calls_by_role.get(role, 0) + n_calls
    if len(roles) <= 1:
        return roles[0] if roles else None
    ranked = sorted(roles, key=lambda role: calls_by_role[role], reverse=True)
    if calls_by_role[ranked[0]] > 0 and (
        len(ranked) == 1 or calls_by_role[ranked[0]] > calls_by_role[ranked[1]]
    ):
        return ranked[0]
    if None in roles:
        return None
    if transcript_model is not None:
        for event in main_models:
            if getattr(event, "model", None) == transcript_model:
                return getattr(event, "role", None)
    return roles[0]


def is_model_turn(event: Any) -> TypeGuard[ModelEvent]:
    """Whether an event is a model turn on some lane. Every model event is,
    a failed generate's empty placeholder output included: Inspect never
    records a model event without an output object, so the None test only
    guards hand-built events. Every lane count and axis position derives
    from this one predicate, so no two surfaces can disagree about what a
    turn is."""
    return isinstance(event, ModelEvent) and event.output is not None


def turn_message(event: ModelEvent) -> ChatMessageAssistant | None:
    """The assistant message a model turn produced, or None when its
    output records no choices. ``ModelOutput.message`` indexes
    ``choices[0]`` and raises on an empty list; a log written by a
    converter can carry such a model event (an output-less closing
    event, a placeholder), and it is still a turn (`is_model_turn`)."""
    return event.output.message if event.output.choices else None


def all_model_turns(transcript: Any) -> Iterator[tuple[Any, list[ToolCall]]]:
    """Yield (model event, its tool calls) for every model turn in every
    lane, in event order; unnumbered, the turn axis is `orchestrator_turns`."""
    for event in transcript.events:
        if not is_model_turn(event):
            continue
        message = turn_message(event)
        yield event, (message.tool_calls or []) if message else []


def orchestrator_turns(transcript: Any) -> Iterator[tuple[int, Any, list[ToolCall]]]:
    """Yield ``(turn, model event, tool calls)`` for the orchestrator's turns.

    The orchestrator is the main lane (`main_span`); ``turn`` is the
    0-based ordinal of its model events with an output, in event order,
    whether or not that output records a choice: read a turn's message
    through `turn_message`, never ``event.output.message``. This
    enumeration is the turn axis: every scanner, frame and chart numbers
    by it, and a custom scanner must too.

    Args:
        transcript: A Scout ``Transcript`` or any object with
            compatible ``events`` and ``timelines``.
    """
    return Lanes(transcript).turns()


def main_span(transcript: Any) -> TimelineSpan:
    """Resolve the orchestrator's timeline span.

    The span Scout's ``timeline_messages(..., depth=1)`` would scan: the
    outermost non-utility span holding a direct model event, container
    spans (the synthetic root, a solvers wrapper) being transparent.
    Built from the event stream (`timeline_build`), never from a stored
    timeline: a scaffold's stored timelines are views of its own (one
    per model role, or a branch tree of a rolled-back conversation),
    not the lane tree over every call this transcript recorded.

    Args:
        transcript: The transcript whose timeline is resolved.

    Returns:
        The ``TimelineSpan`` whose direct content is the main lane.

    A transcript with no events at all (a message history without its
    event stream) resolves to its root: an empty orchestrator lane with
    zero turns, the honest reading of a source that recorded no calls.

    Raises:
        ValueError: When the transcript has events but no model turn
            under a single agent span - a sample that errored before its
            first call, two or more top-level agents, or model calls only
            inside utility spans. There is then no orchestrator lane to
            number; the error is recorded per transcript and shown in
            the scan status section.
    """
    span = timeline_build(transcript.events).root
    while True:
        if any(
            isinstance(item, TimelineEvent) and is_model_turn(item.event)
            for item in span.content
        ):
            return span
        children = [
            item
            for item in span.content
            if isinstance(item, TimelineSpan)
            and item.span_type == "agent"
            and not item.utility
            and _holds_model_events(item)
        ]
        if len(children) == 1:
            span = children[0]
            continue
        if not children:
            if not transcript.events:
                return span  # no events at all: an empty lane
            raise ValueError(
                f"transcript has no model turns under its main span {span.name!r}; "
                "there is no orchestrator lane to number"
            )
        names = ", ".join(repr(child.name) for child in children)
        raise ValueError(
            f"transcript has {len(children)} top-level agents ({names}) and no "
            "single orchestrator lane; transect numbers one orchestrator's turns"
        )


def _holds_model_events(span: TimelineSpan) -> bool:
    """Whether a model event with output lives anywhere in the subtree."""
    for item in span.content:
        if isinstance(item, TimelineEvent):
            if is_model_turn(item.event):
                return True
        elif _holds_model_events(item):
            return True
    return False


def subagent_span_begins(
    transcript: Any, main: TimelineSpan, orchestrator_role: str | None = None
) -> list[Any]:
    """The lane span_begin events, in event order.

    The shared definition of "lane" (decision_phases, the token_timeline
    span record and the agent_spans loader must agree). Two kinds:

    - a sub-agent: an agent-type span_begin that is neither the main
      lane's own span, nor an ancestor of it (a wrapper the timeline
      treats as a container), nor a timeline utility span
      (auto-classified helper calls);
    - another role's conversation: a span that directly holds a model
      turn whose role is not ``orchestrator_role`` (an auditing
      scaffold's target trajectories, one span per attempt), again
      excluding the main span and its ancestors, and anything under
      Inspect's ``init`` and ``scorers`` phase spans (a scorer's judge
      calls are off the axis, not a lane).

    Discovery is span_begin-based: metadata-only spawn spans are pruned
    from the timeline tree, which supplies only the utility exclusion.

    Args:
        transcript: The transcript whose events are scanned.
        main: The main lane's span (see ``main_span``).
        orchestrator_role: The axis role (see ``orchestrator_role``).

    Returns:
        The lane ``span_begin`` events in event order.
    """
    utility_ids = _utility_ids(main)
    by_id = {e.id: e for e in transcript.events if e.event == "span_begin"}
    ancestors: set[str] = set()
    parent = getattr(by_id.get(main.id), "parent_id", None)
    while parent is not None and parent not in ancestors:
        ancestors.add(parent)
        parent = getattr(by_id.get(parent), "parent_id", None)
    conversation_ids = {
        event.span_id
        for event in transcript.events
        if is_model_turn(event)
        and event.span_id is not None
        and getattr(event, "role", None) != orchestrator_role
        and not _under_phase_span(by_id, event.span_id)
    }
    return [
        e
        for e in transcript.events
        if e.event == "span_begin"
        and e.id != main.id
        and e.id not in ancestors
        and e.id not in utility_ids
        and (getattr(e, "type", None) == "agent" or e.id in conversation_ids)
    ]


_PHASE_SPAN_TYPES = frozenset({"init", "scorers", "scorer"})


def _under_phase_span(spans: dict[str, Any], span_id: str | None) -> bool:
    """Whether a span sits in Inspect's init or scoring phase."""
    seen: set[str] = set()
    current = span_id
    while current is not None and current in spans and current not in seen:
        seen.add(current)
        if getattr(spans[current], "type", None) in _PHASE_SPAN_TYPES:
            return True
        current = getattr(spans[current], "parent_id", None)
    return False


def _utility_ids(main: TimelineSpan) -> set[str]:
    """The ids of utility spans in the tree below ``main``."""
    utility_ids: set[str] = set()

    def _walk(span: TimelineSpan) -> None:
        for item in span.content:
            if not isinstance(item, TimelineSpan):
                continue
            if item.utility:
                utility_ids.add(item.id)
            _walk(item)

    _walk(main)
    return utility_ids


def nearest_lane_span(
    spans: dict[str, Any], span_id: str | None, lane_ids: frozenset[str]
) -> Any | None:
    """Walk up the span tree to the nearest lane span: the first ancestor
    that is a lane (in ``lane_ids``) or an agent-type span, returned when
    it is a lane and None otherwise - an event inside a utility agent
    span nested in a lane is the utility span's, not the lane's."""
    seen: set[str] = set()
    current = span_id
    while current is not None and current in spans and current not in seen:
        seen.add(current)
        span = spans[current]
        if current in lane_ids:
            return span
        if getattr(span, "type", None) == "agent":
            return None
        current = getattr(span, "parent_id", None)
    return None


def span_task_text(span: Any, model_event: Any) -> tuple[str, str]:
    """What an agent span was asked to do, with provenance.

    Tried in order:

    1. The spawn ``task``/``prompt`` recorded in the span-begin metadata.
    2. The last non-empty non-tool message handed into the span's first
       model call. Tool messages there are transfer boilerplate
       ("Successfully transferred to <agent>."), not task text; the
       delegation rides the assistant message before them.
    3. Neither -> empty text (a span name alone is not task text).
    """
    metadata = getattr(span, "metadata", None) or {}
    raw = metadata.get("task") or metadata.get("prompt")
    if raw:
        return strip_subagent_scaffold(str(raw)), "spawn_prompt"
    if model_event is not None:
        handoff = next(
            (
                m.text.strip()
                for m in reversed(getattr(model_event, "input", None) or [])
                if getattr(m, "text", None)
                and m.text.strip()
                and getattr(m, "role", None) != "tool"
            ),
            "",
        )
        if handoff:
            return " ".join(handoff.split()), "handoff_input"
    return "", "span_name_only"


def strip_subagent_scaffold(prompt: str) -> str:
    """Drop a spawn prompt's leading OpenClaw scaffold lines."""
    lines = prompt.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if (
            not stripped
            or stripped.startswith(_OPENCLAW_SPAWN_CONTEXT_PREFIX)
            or stripped == _OPENCLAW_SPAWN_TASK_MARKER
            or _OPENCLAW_SPAWN_ROLE_PHRASE in stripped
        ):
            start = i + 1
            continue
        break
    kept = " ".join(" ".join(lines[start:]).split())
    return kept or " ".join(prompt.split())


def span_activity(lanes: Lanes, span_id: str) -> SpanActivity:
    """Collect one sub-agent span's recorded activity (`Lanes.sub_agent_of`
    attribution: nested tool spans roll up)."""
    activity = SpanActivity()
    for event in lanes.events:
        if is_model_turn(event):
            sub = lanes.sub_agent_of(event)
            if sub is not None and sub.id == span_id:
                message = turn_message(event)
                text = getattr(message, "text", None) if message else None
                if isinstance(text, str) and text.strip():
                    activity.turn_texts.append(" ".join(text.split()))
        elif getattr(event, "event", None) == "tool":
            sub = lanes.sub_agent_of(event)
            if sub is not None and sub.id == span_id:
                name = str(getattr(event, "function", None) or "tool")
                activity.tool_counts[name] = activity.tool_counts.get(name, 0) + 1
    return activity


def message_reasoning(message: Any) -> str:
    """Join a message's reasoning-block text, in block order.

    A redacted block falls back to its summary, as does a block whose
    provider reports only a summary; blocks with neither readable field
    are skipped. "" for a plain-string message or one without
    reasoning blocks. (``ContentReasoning.text`` is not used: it wraps
    the text in replay-oriented ``<think>`` tags and drops a
    summary-only unredacted block.)

    Args:
        message: An assistant chat message (or None).

    Returns:
        The whitespace-flattened reasoning text, or "".
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, ContentReasoning):
            continue
        text = block.reasoning if not block.redacted else ""
        text = text.strip() or (block.summary or "").strip()
        if text:
            parts.append(" ".join(text.split()))
    return " ".join(parts)


def capped_lines(lines: list[str], cap: int) -> list[str]:
    """Bound a line list: head+tail with an elision marker.

    Args:
        lines: The rendered lines.
        cap: Maximum lines kept (marker excluded).

    Returns:
        The lines unchanged when under the cap; otherwise the first
        cap//2 and last cap-cap//2 with a ``[... N turns elided ...]``
        marker between them.
    """
    if len(lines) <= cap:
        return lines
    head, tail = cap // 2, cap - cap // 2
    elided = len(lines) - head - tail
    return [*lines[:head], f"[... {elided} turns elided ...]", *lines[-tail:]]
