"""Shared helpers used by more than one scanner."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.event import ModelEvent, TimelineEvent, TimelineSpan, timeline_build
from inspect_ai.model import ContentReasoning
from inspect_ai.tool import ToolCall
from inspect_scout import Transcript

# OPENCLAW-SPECIFIC spawn-prompt scaffold markers.
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


def model_turns(transcript: Transcript) -> Iterator[tuple[Any, list[ToolCall]]]:
    """Yield (model event, its tool calls) per model turn, in event order.

    Shared across scanners: enumerate() over this is the turn axis."""
    for event in transcript.events:
        if event.event != "model" or not event.output:
            continue
        message = event.output.message
        yield event, (message.tool_calls or []) if message else []


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
        text = (text or block.summary or "").strip()
        if text:
            parts.append(" ".join(text.split()))
    return " ".join(parts)


def span_activity(
    transcript: Transcript, span_id: str, spans: dict[str, Any]
) -> SpanActivity:
    """Collect one sub-agent span's recorded activity.

    Model turns and tool events are attributed via
    ``nearest_agent_span`` (nested tool spans roll up); the folded
    spawn call is the orchestrator's, and is excluded.
    """
    activity = SpanActivity()
    for event in transcript.events:
        if isinstance(event, ModelEvent) and event.output:
            agent = nearest_agent_span(spans, getattr(event, "span_id", None))
            if agent is not None and agent.id == span_id:
                message = event.output.message
                text = getattr(message, "text", None) if message else None
                if isinstance(text, str) and text.strip():
                    activity.turn_texts.append(" ".join(text.split()))
        elif getattr(event, "event", None) == "tool":
            if getattr(event, "agent_span_id", None) is not None:
                continue  # folded spawn call
            agent = nearest_agent_span(spans, getattr(event, "span_id", None))
            if agent is not None and agent.id == span_id:
                name = str(getattr(event, "function", None) or "tool")
                activity.tool_counts[name] = activity.tool_counts.get(name, 0) + 1
    return activity


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


def main_lane_id(transcript: Any) -> str | None:
    """The main lane's span id, or None when it cannot be resolved.

    Args:
        transcript: The transcript whose main lane is resolved.

    Returns:
        The main span's id, or None.
    """
    try:
        return main_span(transcript).id
    except Exception:
        return None


def main_span(transcript: Any) -> TimelineSpan:
    """Resolve the main lane's timeline span.

    Uses the transcript's stored timeline when present, else builds one
    in place - Scout 0.4.45 only auto-builds timelines on the .eval
    path, so importer/database sources arrive with ``timelines`` empty.

    ``timeline_build`` gives an agent-centric tree: OpenClaw root-lane
    orchestrators become a synthetic "main" root holding their events;
    .eval logs get init/solvers/scorers partitioning with the solver
    agent as root. When the root holds no model events of its own and
    has exactly one non-utility agent child (a solo agent wrapped in a
    bare span, no phase spans), that child is the main lane.

    Args:
        transcript: The transcript whose timeline is resolved.

    Returns:
        The ``TimelineSpan`` whose direct content is the main lane.
    """
    timelines = getattr(transcript, "timelines", None)
    timeline = timelines[0] if timelines else timeline_build(transcript.events)
    span = timeline.root
    while True:
        has_models = any(
            isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
            for item in span.content
        )
        if has_models:
            return span
        children = [
            item
            for item in span.content
            if isinstance(item, TimelineSpan)
            and item.span_type == "agent"
            and not item.utility
        ]
        if len(children) != 1:
            return span
        span = children[0]


def subagent_span_begins(
    transcript: Any, main: TimelineSpan
) -> tuple[list[Any], dict[str, Any]]:
    """Sub-agent span_begin events + each span's first model event.

    The shared definition of "sub-agent" (decision_phases and the
    agent_spans loader must agree): an agent-type span_begin that is
    neither the main lane's own span nor a timeline utility span
    (auto-classified helper calls). Discovery stays span_begin-based -
    metadata-only spawn spans are pruned from the timeline tree, so the
    tree cannot own it; the tree supplies the utility exclusion and the
    first model event per surviving span (``span_task_text``'s handoff
    fallback).

    Args:
        transcript: The transcript whose events are scanned.
        main: The main lane's span (see ``main_span``).

    Returns:
        ``(span_begins, first_models)`` in event order.
    """
    first_models, utility_ids = _span_details(main)
    begins = [
        e
        for e in transcript.events
        if e.event == "span_begin"
        and getattr(e, "type", None) == "agent"
        and e.id != main.id
        and e.id not in utility_ids
    ]
    return begins, first_models


def _span_details(main: TimelineSpan) -> tuple[dict[str, Any], set[str]]:
    """Collect per-span details from the tree below the main span.

    Args:
        main: The main lane's timeline span.

    Returns:
        ``(first_models, utility_ids)`` - each descendant span's first
        model event, and the ids of utility spans.
    """
    first_models: dict[str, Any] = {}
    utility_ids: set[str] = set()

    def _walk(span: TimelineSpan) -> None:
        for item in span.content:
            if not isinstance(item, TimelineSpan):
                continue
            if item.utility:
                utility_ids.add(item.id)
            for sub in item.content:
                if (
                    isinstance(sub, TimelineEvent)
                    and isinstance(sub.event, ModelEvent)
                    and sub.event.output
                ):
                    first_models.setdefault(item.id, sub.event)
                    break
            _walk(item)

    _walk(main)
    return first_models, utility_ids
