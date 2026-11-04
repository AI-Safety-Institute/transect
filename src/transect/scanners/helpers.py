"""Shared helpers used by more than one scanner.

The turn axis lives here: `orchestrator_turns` numbers the orchestrator's
model turns, `Lanes` resolves which lane any event belongs to (a
sub-agent span, or another model role's conversation span), and the
span text helpers read what a sub-agent was asked and what it did.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard

from inspect_ai.event import ModelEvent, TimelineEvent, TimelineSpan, timeline_build
from inspect_ai.model import ChatMessageAssistant, ContentReasoning
from inspect_ai.tool import ToolCall

from transect.petri import is_petri

# OpenClaw spawn-prompt scaffold markers
_OPENCLAW_SPAWN_CONTEXT_PREFIX = "[subagent context]"
_OPENCLAW_SPAWN_TASK_MARKER = "[subagent task]"
_OPENCLAW_SPAWN_ROLE_PHRASE = "you are running as a subagent"

LaneKind = Literal["subagent", "role"]
"""What kind of lane a span is: a sub-agent span, or the conversation
span of another model role (`Lanes`)."""

# Inspect's init and scoring phase spans: a model call under them is off
# the axis and in no lane
_PHASE_SPAN_TYPES = frozenset({"init", "scorers", "scorer"})

# Spans whose model calls never drive an audit: the phase spans above and
# a tool span (a tool that calls a model is a helper call)
_OFF_AXIS_SPAN_TYPES = _PHASE_SPAN_TYPES | {"tool"}


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

    ``main`` is the orchestrator's span (`main_span`); its own model
    turns of ``axis_role`` (`axis_role`) are the orchestrator turns.
    ``begins`` are the sub-agent span_begin events (`subagent_span_begins`,
    the shared definition), ``role_begins`` the conversation spans of
    other model roles (`role_span_begins`), ``lane_ids`` both, and
    `lane_kind` tells them apart. ``first_models`` is each lane's first
    model event. ``events`` is the transcript's event list. Every
    scanner that attributes events to lanes starts from one of these.
    """

    def __init__(self, transcript: Any) -> None:
        self.main = main_span(transcript)
        self.events: list[Any] = list(transcript.events)
        self._transcript = transcript
        self._spans = {e.id: e for e in self.events if e.event == "span_begin"}
        self._positions = {id(e): i for i, e in enumerate(self.events)}
        in_main = {
            id(item.event)
            for item in self.main.content
            if isinstance(item, TimelineEvent) and is_model_turn(item.event)
        }
        main_models = [e for e in self.events if id(e) in in_main]
        self.axis_role = axis_role(main_models)
        self._main_events = frozenset(
            id(e) for e in main_models if getattr(e, "role", None) == self.axis_role
        )
        self.begins, self.first_models = subagent_span_begins(transcript, self.main)
        self.role_begins = role_span_begins(transcript, self.main, self.axis_role)
        self.sub_ids = frozenset(begin.id for begin in self.begins)
        self.lane_ids = self.sub_ids | frozenset(b.id for b in self.role_begins)
        # a role lane is no tree span: its first turn comes from the events
        for event in self.events:
            lane = self.lane_of(event) if is_model_turn(event) else None
            if lane is not None and lane.id not in self.sub_ids:
                self.first_models.setdefault(lane.id, event)

    def is_main(self, event: Any) -> bool:
        """Whether a model event is one of the orchestrator's own turns."""
        return id(event) in self._main_events

    def lane_of(self, event: Any) -> Any | None:
        """The lane span_begin an event belongs to: its nearest enclosing
        lane span (`nearest_lane_span`). None when the event is the
        orchestrator's (a folded spawn call, ``agent_span_id`` set,
        included), a utility or wrapper span's, or off the axis (an init
        or scorer call, another role's call in the orchestrator's own
        span)."""
        if getattr(event, "agent_span_id", None) is not None or self.is_main(event):
            return None
        return nearest_lane_span(
            self._spans, getattr(event, "span_id", None), self.lane_ids
        )

    def lane_kind(self, span_id: str) -> LaneKind:
        """What kind of lane a lane span is.

        Raises:
            KeyError: When ``span_id`` is not a lane.
        """
        if span_id not in self.lane_ids:
            raise KeyError(f"{span_id!r} is not a lane span")
        return "subagent" if span_id in self.sub_ids else "role"

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


def axis_role(main_models: list[Any]) -> str | None:
    """The model role on the axis: the role that opens the main span's
    conversation and makes its first tool call. A span of one role keeps
    that role, None being the role-less default model. The driven side
    of a scaffold cannot act before it is driven, so the opening turn
    and the first tool call agree on the driver; when they disagree
    there is no single orchestrator and the transcript raises rather
    than guessing. Turns of another role in the main span are off the
    axis unless a span of their own makes them a lane
    (`role_span_begins`).

    Args:
        main_models: The main span's direct model turns, in event order.

    Returns:
        The axis role; None when the main span has no turns.

    Raises:
        ValueError: When the opening turn's role and the first
            tool-calling turn's role differ.
    """
    if not main_models:
        return None
    opening = getattr(main_models[0], "role", None)
    first_tool = next(
        (
            getattr(event, "role", None)
            for event in main_models
            if (message := turn_message(event)) is not None and message.tool_calls
        ),
        opening,
    )
    if first_tool != opening:
        raise ValueError(
            "no single orchestrator role: the main span opens with a "
            f"{opening!r} turn but its first tool call is a {first_tool!r} turn"
        )
    return opening


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
            compatible ``events``.
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
    not the lane tree over every call this transcript recorded, and
    only some scanners receive them.

    On a Petri audit (`transect.petri.is_petri`) the auditor owns the
    axis. That span is kept whenever the auditor already drives it, the
    plain inspect_petri shape; when another role's turn opens it (an
    audit embedded in a seed-design solver, whose writer calls the model
    first) or it names no single orchestrator, the main lane narrows to
    the auditor's own span (`_auditor_span`): the built span when the
    timeline kept it, otherwise the auditor's events and child spans
    projected out of the span the timeline flattened them into, so its
    utility helpers and sub-agents stay wrapped as on the plain shape.

    Args:
        transcript: The transcript whose timeline is resolved.

    Returns:
        The ``TimelineSpan`` whose direct content is the main lane.

    A transcript with no events at all (a message history without its
    event stream) resolves to its root: an empty orchestrator lane with
    zero turns, the honest reading of a source that recorded no calls.

    Raises:
        ValueError: When the timeline has no single orchestrator lane:
            no model turn at all under the main span but events exist,
            two or more top-level agents, or model calls only inside
            utility spans; and on a Petri audit when unrelated spans
            each hold auditor turns (`_petri_auditor_id`), independent
            audits that raise rather than combine. There is then no
            orchestrator lane to number; the error is recorded per
            transcript and shown in the scan status section.
    """
    root = timeline_build(transcript.events).root
    auditor_id = _petri_auditor_id(transcript)
    if auditor_id is None:
        return _depth_one_span(root, transcript.events)
    try:
        span: TimelineSpan | None = _depth_one_span(root, transcript.events)
    except ValueError:
        span = None
    if span is not None and _span_axis(span) == "auditor":
        return span
    return _auditor_span(root, auditor_id, transcript.events)


def _depth_one_span(root: TimelineSpan, events: list[Any]) -> TimelineSpan:
    """The span at depth 1 of a built timeline (`main_span`): descend
    through a single non-utility agent child until a span holds a
    direct model turn."""
    span = root
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
            if not events:
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
    transcript: Any, main: TimelineSpan
) -> tuple[list[Any], dict[str, Any]]:
    """Sub-agent span_begin events + each span's first model event.

    The shared definition of "sub-agent" (decision_phases, the
    token_timeline span record and the agent_spans loader must agree):
    an agent-type span_begin that is neither the main lane's own span,
    nor an ancestor of it (a wrapper the timeline treats as a
    container), nor a timeline utility span (auto-classified helper
    calls). Discovery is span_begin-based: metadata-only spawn spans
    are pruned from the timeline tree, which supplies only the utility
    exclusion and each surviving span's first model event.

    Args:
        transcript: The transcript whose events are scanned.
        main: The main lane's span (see ``main_span``).

    Returns:
        ``(span_begins, first_models)`` in event order.
    """
    first_models, utility_ids = _span_details(main)
    by_id = {e.id: e for e in transcript.events if e.event == "span_begin"}
    ancestors = _ancestor_ids(by_id, main.id)
    begins = [
        e
        for e in transcript.events
        if e.event == "span_begin"
        and getattr(e, "type", None) == "agent"
        and e.id != main.id
        and e.id not in ancestors
        and e.id not in utility_ids
    ]
    return begins, first_models


def role_span_begins(
    transcript: Any, main: TimelineSpan, axis_role: str | None
) -> list[Any]:
    """The conversation spans of other model roles, in event order: the
    second kind of lane (`Lanes`).

    A span that directly holds model turns of other roles than
    ``axis_role`` and none of the axis role (an auditing scaffold's
    target trajectories, one span per attempt). Not a lane: the main
    span and its ancestors, agent-type spans and anything below one (a
    sub-agent owns the spans inside it, a utility agent is no lane),
    tool spans (a tool that calls a model is a helper call), and
    anything under Inspect's init and scoring phase spans. Another
    role's calls outside such a span are off the axis.

    Args:
        transcript: The transcript whose events are scanned.
        main: The main lane's span (see ``main_span``).
        axis_role: The role on the axis (see ``axis_role``).

    Returns:
        The role lanes' ``span_begin`` events in event order.
    """
    by_id = {e.id: e for e in transcript.events if e.event == "span_begin"}
    own = {main.id} | _ancestor_ids(by_id, main.id)
    conversation_ids: set[str] = set()
    axis_span_ids: set[str] = set()
    for event in transcript.events:
        if not is_model_turn(event) or event.span_id is None:
            continue
        if getattr(event, "role", None) == axis_role:
            axis_span_ids.add(event.span_id)
        elif not any(
            getattr(span, "type", None) in _PHASE_SPAN_TYPES
            for span in _lineage(by_id, event.span_id)
        ):
            conversation_ids.add(event.span_id)
    return [
        e
        for e in transcript.events
        if e.event == "span_begin"
        and e.id in conversation_ids
        and e.id not in axis_span_ids
        and e.id not in own
        and getattr(e, "type", None) not in ("agent", "tool")
        and not any(
            getattr(span, "type", None) == "agent" and span.id not in own
            for span in _lineage(by_id, e.parent_id)
        )
    ]


def nearest_lane_span(
    spans: dict[str, Any], span_id: str | None, lane_ids: frozenset[str]
) -> Any | None:
    """Walk up the span tree to the nearest lane span: the first ancestor
    that is a lane (in ``lane_ids``) or an agent-type span, returned when
    it is a lane and None otherwise - an event inside a utility agent
    span nested in a lane is the utility span's, not the lane's."""
    for span in _lineage(spans, span_id):
        if span.id in lane_ids:
            return span
        if getattr(span, "type", None) == "agent":
            return None
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
    """Collect one sub-agent span's recorded activity (`Lanes.lane_of`
    attribution: nested tool spans roll up)."""
    activity = SpanActivity()
    for event in lanes.events:
        if is_model_turn(event):
            sub = lanes.lane_of(event)
            if sub is not None and sub.id == span_id:
                message = turn_message(event)
                text = getattr(message, "text", None) if message else None
                if isinstance(text, str) and text.strip():
                    activity.turn_texts.append(" ".join(text.split()))
        elif getattr(event, "event", None) == "tool":
            sub = lanes.lane_of(event)
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


def _span_details(main: TimelineSpan) -> tuple[dict[str, Any], set[str]]:
    """Each descendant span's first model event with output, and the ids
    of utility spans, from the tree below ``main``."""
    first_models: dict[str, Any] = {}
    utility_ids: set[str] = set()

    def _walk(span: TimelineSpan) -> None:
        for item in span.content:
            if not isinstance(item, TimelineSpan):
                continue
            if item.utility:
                utility_ids.add(item.id)
            for sub in item.content:
                if isinstance(sub, TimelineEvent) and is_model_turn(sub.event):
                    first_models.setdefault(item.id, sub.event)
                    break
            _walk(item)

    _walk(main)
    return first_models, utility_ids


def _lineage(spans: dict[str, Any], span_id: str | None) -> Iterator[Any]:
    """The span_begin events from ``span_id`` up to the root."""
    seen: set[str] = set()
    current = span_id
    while current is not None and current in spans and current not in seen:
        seen.add(current)
        yield spans[current]
        current = getattr(spans[current], "parent_id", None)


def _ancestor_ids(spans: dict[str, Any], span_id: str) -> set[str]:
    """The ids of a span's ancestors in the recorded span tree."""
    parent = getattr(spans.get(span_id), "parent_id", None)
    return {span.id for span in _lineage(spans, parent)}


def _petri_auditor_id(transcript: Any) -> str | None:
    """The span driving a Petri audit: the outermost span holding the
    auditor's model turns, read from the events' own span ids. Calls
    under Inspect's phase spans or a tool span and span-less calls are
    off the axis and do not count, and a span nested in another auditor
    span (a helper the auditor runs) belongs to that audit. None when
    the run is not a Petri audit or the auditor made no such call.

    Raises:
        ValueError: When two or more unrelated spans hold auditor
            turns: independent audits that no single axis can number.
    """
    metadata = getattr(transcript, "metadata", None) or {}
    if not is_petri(metadata.get("model_roles")):
        return None
    by_id = {e.id: e for e in transcript.events if e.event == "span_begin"}
    ids: list[str] = []
    for event in transcript.events:
        if (
            not is_model_turn(event)
            or getattr(event, "role", None) != "auditor"
            or event.span_id is None
            or event.span_id in ids
            or any(
                getattr(span, "type", None) in _OFF_AXIS_SPAN_TYPES
                for span in _lineage(by_id, event.span_id)
            )
        ):
            continue
        ids.append(event.span_id)
    outer = [i for i in ids if not (_ancestor_ids(by_id, i) & set(ids))]
    if len(outer) > 1:
        names = ", ".join(repr(getattr(by_id.get(i), "name", i)) for i in outer)
        raise ValueError(
            f"Petri transcript holds {len(outer)} auditor spans ({names}) and no "
            "single audit to number; select one audit before scanning rather "
            "than merging independent conversations"
        )
    return outer[0] if outer else None


def _span_axis(span: TimelineSpan) -> str | None:
    """The axis role of a span's direct model turns (`axis_role`); None
    when they name no single orchestrator."""
    models = [
        item.event
        for item in span.content
        if isinstance(item, TimelineEvent) and is_model_turn(item.event)
    ]
    try:
        return axis_role(models)
    except ValueError:
        return None


def _auditor_span(
    root: TimelineSpan, auditor_id: str, events: list[Any]
) -> TimelineSpan:
    """The auditor's own span as the main lane (`main_span`): the built
    span when the timeline kept it (an agent-type auditor), otherwise a
    span of the same identity whose content is projected out of the span
    the timeline flattened the auditor's into: the events and child
    spans (utility wrappers, sub-agents) the recorded span tree places
    under the auditor, in timeline order."""
    found = _find_span(root, auditor_id)
    if found is not None:
        return found
    by_id = {e.id: e for e in events if e.event == "span_begin"}
    begin = by_id.get(auditor_id)
    if begin is None:
        raise ValueError(
            f"Petri auditor span {auditor_id!r} has no recorded span_begin"
        )
    holder = _holder_of(root, auditor_id)
    if holder is None:
        raise ValueError(
            f"Petri auditor span {begin.name!r} holds no model turn in the "
            "built timeline"
        )

    def under(span_id: str | None) -> bool:
        return any(span.id == auditor_id for span in _lineage(by_id, span_id))

    content: list[Any] = []
    for item in holder.content:
        if isinstance(item, TimelineEvent):
            if under(getattr(item.event, "span_id", None)):
                content.append(item)
        elif under(getattr(_first_event(item), "span_id", None)):
            content.append(item)
    return TimelineSpan(
        id=auditor_id, name=begin.name, span_type=begin.type, content=content
    )


def _find_span(span: TimelineSpan, span_id: str) -> TimelineSpan | None:
    """The span with ``span_id`` in a built tree, branches included."""
    if span.id == span_id:
        return span
    children = [item for item in span.content if isinstance(item, TimelineSpan)]
    for child in [*children, *span.branches]:
        found = _find_span(child, span_id)
        if found is not None:
            return found
    return None


def _holder_of(span: TimelineSpan, span_id: str) -> TimelineSpan | None:
    """The built span directly holding an event the recorded span
    ``span_id`` owns: where the timeline flattened that span into."""
    for item in span.content:
        if isinstance(item, TimelineEvent) and item.event.span_id == span_id:
            return span
    for item in span.content:
        if isinstance(item, TimelineSpan):
            found = _holder_of(item, span_id)
            if found is not None:
                return found
    return None


def _first_event(span: TimelineSpan) -> Any | None:
    """The first event anywhere under a built span, in timeline order."""
    for item in span.content:
        if isinstance(item, TimelineEvent):
            return item.event
        found = _first_event(item)
        if found is not None:
            return found
    return None
