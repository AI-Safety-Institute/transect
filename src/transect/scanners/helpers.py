"""Shared helpers used by more than one scanner.

The turn axis lives here: `orchestrator_turns` numbers the orchestrator's
model turns, `Lanes` resolves which lane any event belongs to, and the
span text helpers read what a sub-agent was asked and what it did.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from inspect_ai.event import ModelEvent, TimelineEvent, TimelineSpan, timeline_build
from inspect_ai.model import ContentReasoning
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

    ``main`` is the orchestrator's span (`main_span`). ``begins`` are the
    sub-agent span_begin events (`subagent_span_begins`, the shared
    definition) and ``first_models`` each one's first model event.
    ``events`` is the transcript's event list. Every scanner that
    attributes events to lanes starts from one of these.
    """

    def __init__(self, transcript: Any) -> None:
        self.main = main_span(transcript)
        self.begins, self.first_models = subagent_span_begins(transcript, self.main)
        self.events: list[Any] = list(transcript.events)
        self._transcript = transcript
        self.sub_ids = frozenset(begin.id for begin in self.begins)
        self._main_events = frozenset(
            id(item.event)
            for item in self.main.content
            if isinstance(item, TimelineEvent) and is_model_turn(item.event)
        )
        self._spans = {e.id: e for e in self.events if e.event == "span_begin"}

    def is_main(self, event: Any) -> bool:
        """Whether a model event is one of the orchestrator's own turns."""
        return id(event) in self._main_events

    def sub_agent_of(self, event: Any) -> Any | None:
        """The sub-agent span_begin an event belongs to: its nearest
        enclosing agent span, when that is a sub-agent. None when the
        event is the orchestrator's (a folded spawn call, ``agent_span_id``
        set, included), a utility or wrapper span's, or off the axis (an
        init or scorer call)."""
        if getattr(event, "agent_span_id", None) is not None:
            return None
        span = nearest_agent_span(self._spans, getattr(event, "span_id", None))
        return span if span is not None and span.id in self.sub_ids else None

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


def is_model_turn(event: Any) -> TypeGuard[ModelEvent]:
    """Whether an event is a model turn on some lane. Every model event is,
    a failed generate's empty placeholder output included: Inspect never
    records a model event without an output object, so the None test only
    guards hand-built events. Every lane count and axis position derives
    from this one predicate, so no two surfaces can disagree about what a
    turn is."""
    return isinstance(event, ModelEvent) and event.output is not None


def all_model_turns(transcript: Any) -> Iterator[tuple[Any, list[ToolCall]]]:
    """Yield (model event, its tool calls) for every model turn in every
    lane, in event order; unnumbered, the turn axis is `orchestrator_turns`."""
    for event in transcript.events:
        if not is_model_turn(event):
            continue
        message = event.output.message
        yield event, (message.tool_calls or []) if message else []


def orchestrator_turns(transcript: Any) -> Iterator[tuple[int, Any, list[ToolCall]]]:
    """Yield ``(turn, model event, tool calls)`` for the orchestrator's turns.

    The orchestrator is the main lane (`main_span`); ``turn`` is the
    0-based ordinal of its model events with output, in event order.
    This enumeration is the turn axis: every scanner, frame and chart
    numbers by it, and a custom scanner must too.

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
    Uses the transcript's stored timeline when present, else builds one
    in place - importer/database sources arrive with ``timelines`` empty.

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
    timelines = getattr(transcript, "timelines", None)
    timeline = timelines[0] if timelines else timeline_build(transcript.events)
    span = timeline.root
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
    ancestors: set[str] = set()
    parent = getattr(by_id.get(main.id), "parent_id", None)
    while parent is not None and parent not in ancestors:
        ancestors.add(parent)
        parent = getattr(by_id.get(parent), "parent_id", None)
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


def nearest_agent_span(spans: dict[str, Any], span_id: str | None) -> Any | None:
    """Walk up the span tree to the nearest agent-type span."""
    seen: set[str] = set()
    current = span_id
    while current is not None and current in spans and current not in seen:
        seen.add(current)
        span = spans[current]
        if getattr(span, "type", None) == "agent":
            return span
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
                message = event.output.message
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
