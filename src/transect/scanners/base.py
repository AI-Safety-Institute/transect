"""Structural (zero-LLM) Scout scanners: the default path."""

import json
from collections import Counter
from typing import Any, cast

from inspect_ai.log import read_eval_log
from inspect_scout import Result, Scanner, Transcript, scanner
from pydantic import JsonValue

from transect.scanners.compaction import compaction_texts
from transect.scanners.helpers import main_lane_id, model_turns, nearest_agent_span

# inspect-ai ModelUsage attribute names, used verbatim as dataframe columns
_USAGE_FIELDS = (
    "output_tokens",
    "input_tokens",
    "total_tokens",
    "input_tokens_cache_read",
    "input_tokens_cache_write",
    "reasoning_tokens",
)


@scanner(events=["model", "span_begin", "span_end", "tool"])
def token_timeline() -> Scanner[Transcript]:
    """Per-model-turn token usage, in turn order.

    value = {"timeline": [entry, ...]} with one entry per model turn.
    Tokens: None = not reported (never 0).

    Each entry carries its agent lane (agent_lane, agent_span_id):
    handoff sub-agents get their own agent spans.
    """

    async def execute(transcript: Transcript) -> Result:
        spans = {e.id: e for e in transcript.events if e.event == "span_begin"}
        main_id = main_lane_id(transcript)
        lane_activity = _lane_activity(transcript, spans, main_id)
        span_ends = _span_ends(transcript, spans, main_id)

        timeline: list[dict[str, Any]] = []
        for turn, (event, calls) in enumerate(model_turns(transcript)):
            usage = event.output.usage
            entry: dict[str, Any] = {"turn": turn, "n_tool_calls": len(calls)}
            for field in _USAGE_FIELDS:
                entry[field] = getattr(usage, field, None) if usage else None
            agent_span = _sub_agent_span(
                spans, getattr(event, "span_id", None), main_id
            )
            entry["agent_lane"] = agent_span.name if agent_span else None
            entry["agent_span_id"] = agent_span.id if agent_span else None
            timeline.append(entry)
        return Result(
            value={
                "timeline": cast(JsonValue, timeline),
                "lane_activity": cast(JsonValue, lane_activity),
                "span_ends": cast(JsonValue, span_ends),
            },
            explanation=f"{len(timeline)} model turns",
        )

    return execute


@scanner(messages="all")
def eval_setup() -> Scanner[Transcript]:
    """The run's setup facts for the report's intro section.

    value = one dict per transcript:

    - ``system_prompt`` / ``task_message``: the verbatim initial
      prompts - the first system message and the first user (task)
      message the model actually saw; None where the source recorded
      none (OpenClaw has no system message).
    - ``agent_args``: the agent scaffold's full arguments as run
      (``Transcript.agent_args``: scaffold prompt, tool roster,
      attempts, submit, compaction, truncation, approval), None when
      the source recorded none.
    - ``compaction_prompt``: the configured Inspect compaction prompt
      template from agent_args.compaction.prompt, not a resolved default.
    - ``task_args`` / ``generate_config`` / ``model_roles``: from the
      importer's per-sample transcript metadata; None where absent.
    - ``header``: the source log's header subset (eval config incl.
      limits, sandbox, dataset, scorers, task file + version), read
      here from ``transcript.source_uri`` (header-only, memoised per
      source file). None when the source is not an eval log (OpenClaw
      import) - the report renders "data not found".
    """
    header_of: dict[str, dict[str, Any] | None] = {}

    async def execute(transcript: Transcript) -> Result:
        system_prompt = next(
            (m.text for m in transcript.messages if m.role == "system"), None
        )
        task_message = next(
            (m.text for m in transcript.messages if m.role == "user"), None
        )
        meta = transcript.metadata or {}
        header = None
        if transcript.source_type == "eval_log" and transcript.source_uri:
            uri = transcript.source_uri
            if uri not in header_of:
                header_of[uri] = _eval_header(uri)
            header = header_of[uri]
        value = {
            "system_prompt": system_prompt,
            "task_message": task_message,
            "agent_args": transcript.agent_args,
            "task_args": meta.get("task_args"),
            "generate_config": meta.get("generate_config"),
            "model_roles": meta.get("model_roles"),
            "header": header,
            "compaction_prompt": None,
        }
        if transcript.source_type == "eval_log":
            compaction = (transcript.agent_args or {}).get("compaction")
            if isinstance(compaction, dict):
                prompt = compaction.get("prompt")
                if isinstance(prompt, str):
                    value["compaction_prompt"] = prompt
        return Result(value=cast(JsonValue, value), explanation="eval setup")

    return execute


@scanner(messages=["user", "assistant"], events=["model", "compaction"])
def context_flush() -> Scanner[Transcript]:
    """Context-window compactions (flushes), from explicit compaction events.

    value = {"flushes": [entry, ...]} with one entry per compaction:
    turn (count of model turns preceding the flush), type, source,
    tokens_before, tokens_after, role, metadata - recorded as the event
    reports them (optional facts remain None). Inspect eval logs also
    carry compaction_prompt (the summarization call's formatted prompt)
    and compaction_nudge (the pre-compaction memory warning), both as
    the model saw them; see scanners/compaction.py for how they are
    located. Text the source never recorded stays None, including on
    OpenClaw imports, whose export carries neither.
    """

    async def execute(transcript: Transcript) -> Result:
        flushes: list[dict[str, Any]] = []
        texts = iter(compaction_texts(transcript))
        for turn, event in _non_model_events(transcript):
            if event.event != "compaction":
                continue
            flushes.append(
                {
                    "turn": turn,
                    "type": event.type,
                    "source": event.source,
                    "tokens_before": event.tokens_before,
                    "tokens_after": event.tokens_after,
                    "role": event.role,
                    "metadata": event.model_dump(mode="json")["metadata"],
                    "compaction_prompt": None,
                    "compaction_nudge": None,
                    **next(texts, {}),
                }
            )
        return Result(
            value={"flushes": cast(JsonValue, flushes)},
            explanation=f"{len(flushes)} compaction(s)",
        )

    return execute


@scanner(messages="all", events=cast("list[Any]", ["model", "input", "approval"]))
def human_intervention() -> Scanner[Transcript]:
    """Mid-run human interactions. Detection is structural, via inspect's
    ChatMessage.source field and its human-facing events - scaffold-
    generated user messages (handoff boundaries, react continue-prompts)
    can never match.

    Two shapes, told apart by ``initiator``:

    - human-initiated (``prompt``/``outcome`` None): the human wrote to
      the running agent unprompted. Channels "operator" (user message
      with source="operator": ACP steering, OpenClaw inbound messages)
      and "input" (source="input").
    - agent-initiated: the run put something to a human and recorded
      the reply. Channel "input_event" is an InputEvent: from
      ask_user / request_input it carries the question as ``prompt``,
      the ``outcome`` (accepted / declined / cancelled) and the answer
      as ``content`` (structured fields as "name: value" lines, else
      the recorded text minus the question); a console input_screen
      recording has no separate question, so ``prompt`` is None and
      the recording is the content. Channel "approval" is an
      ApprovalEvent decided by inspect's built-in human approver
      (approver name "human", ACP-routed approvals included; a custom
      approver registered under another name is not counted):
      ``prompt`` is the tool call put to them, ``outcome`` the
      decision, ``content`` the explanation. A "modify" decision
      records the decision, not the modified call.

    The first user message that arrived on a human channel (operator
    or input) is the task prompt, never an intervention.

    value = {"interventions": [entry, ...]}: turn (the model turn the
    intervention precedes, on the shared event axis), channel, initiator,
    prompt, content, outcome.
    """

    async def execute(transcript: Transcript) -> Result:
        interventions: list[dict[str, Any]] = []

        def entry(turn, channel, initiator, content, prompt=None, outcome=None):
            interventions.append(
                {
                    "turn": turn,
                    "channel": channel,
                    "initiator": initiator,
                    "prompt": prompt,
                    "content": content,
                    "outcome": outcome,
                }
            )

        # Human messages live in the history, not the event stream, so their
        # turn is the event turn of the assistant message before them plus
        # one. Counting assistant messages instead would drift by one after
        # every summary compaction: the summarization call is a model turn
        # whose output never enters the history.
        turn_of_output = {
            event.output.message.id: turn
            for turn, (event, _calls) in enumerate(model_turns(transcript))
            if event.output.message is not None
        }
        next_turn = 0
        seen_task_prompt = False
        for message in transcript.messages:
            if message.role == "assistant":
                # an assistant message no event recorded (a history without
                # its event stream) still advances the count by one
                recorded = turn_of_output.get(message.id)
                next_turn = next_turn + 1 if recorded is None else recorded + 1
                continue
            if message.role != "user":
                continue
            source = getattr(message, "source", None)
            if source not in ("operator", "input"):
                continue
            if not seen_task_prompt:
                seen_task_prompt = True
                continue
            entry(next_turn, source, "human", (message.text or "").strip())

        for turn, event in _non_model_events(transcript):
            if event.event == "input":
                prompt = (getattr(event, "message", None) or "").strip() or None
                entry(
                    turn,
                    "input_event",
                    "agent",
                    _input_answer(event, prompt),
                    prompt,
                    getattr(event, "outcome", None),
                )
            elif event.event == "approval" and event.approver == "human":
                call = event.call
                arguments = (
                    json.dumps(call.arguments, ensure_ascii=False, default=str)
                    if call.arguments
                    else ""
                )
                entry(
                    turn,
                    "approval",
                    "agent",
                    (event.explanation or "").strip(),
                    f"{call.function}({arguments})",
                    event.decision,
                )

        interventions.sort(key=lambda i: i["turn"])
        return Result(
            value={"interventions": cast(JsonValue, interventions)},
            explanation=f"{len(interventions)} intervention(s)",
        )

    return execute


def _input_answer(event: Any, prompt: str | None) -> str:
    """The human's side of an InputEvent: the structured answer when one
    was recorded (nested values as JSON), else the recorded text with the
    question (which inspect prepends to it) removed. Empty on a declined
    or cancelled request: the text is only inspect's marker line, and
    ``outcome`` carries the fact."""
    content = getattr(event, "content", None)
    if isinstance(content, dict) and content:
        return "\n".join(
            f"{name}: {_json_text(value)}" for name, value in content.items()
        )
    if getattr(event, "outcome", None) in ("declined", "cancelled"):
        return ""
    text = (event.input or "").strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :].strip()
    return text


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _span_ends(
    transcript: Transcript, spans: dict[str, Any], main_id: str | None = None
) -> list[dict[str, Any]]:
    """Recorded agent-span completions: one {agent_span_id, turn} per
    span_end event whose span is a sub-agent span (the main lane's own
    span excluded). Turn anchor = the initiating model turn (0-based).
    """
    ends: list[dict[str, Any]] = []
    for turns_before, event in _non_model_events(transcript):
        if event.event != "span_end":
            continue
        span_id = getattr(event, "id", None)
        if span_id is None or span_id == main_id:
            continue
        span = spans.get(span_id)
        if span is None or getattr(span, "type", None) != "agent":
            continue
        # the initiating turn on the 0-based axis
        ends.append({"agent_span_id": span.id, "turn": max(turns_before - 1, 0)})
    return ends


def _lane_activity(
    transcript: Transcript, spans: dict[str, Any], main_id: str | None = None
) -> list[dict[str, Any]]:
    """Tool activity per (model turn, agent span): calls, busy time, start.

    Counts tool events inside any sub-agent span (the main lane's own
    span excluded); sub-agents whose activity is tool-events-only (no
    model turns of their own, e.g. OpenClaw schema-B) show up only
    here. Turn anchor = the initiating model turn (0-based).
    """
    hits: Counter[tuple[int, str, str]] = Counter()
    busy_ms: Counter[tuple[int, str, str]] = Counter()
    started_at: dict[tuple[int, str, str], Any] = {}
    for turns_before, event in _non_model_events(transcript):
        if event.event != "tool":
            continue
        if getattr(event, "agent_span_id", None) is not None:
            continue  # folded spawn call: the orchestrator's, not the lane's
        agent_span = _sub_agent_span(spans, getattr(event, "span_id", None), main_id)
        if agent_span is None:
            continue
        # the initiating turn on the 0-based axis
        key = (max(turns_before - 1, 0), agent_span.name, agent_span.id)
        hits[key] += 1
        started = getattr(event, "timestamp", None)
        completed = getattr(event, "completed", None)
        if started is not None:
            if key not in started_at or started < started_at[key]:
                started_at[key] = started
            if completed is not None:
                busy_ms[key] += (completed - started).total_seconds() * 1000
    rows = []
    for key in sorted(hits):
        turn, lane, span_id = key
        rows.append(
            {
                "turn": turn,
                "agent_lane": lane,
                "agent_span_id": span_id,
                "tool_calls": hits[key],
                # busy time from per-call durations; None when never reported
                "busy_seconds": (
                    round(busy_ms[key] / 1000.0, 1) if key in busy_ms else None
                ),
                "started_at": (
                    started_at[key].isoformat() if key in started_at else None
                ),
            }
        )
    return rows


def _non_model_events(transcript: Transcript):
    """Yield (n_model_turns_before, event) for every non-model event."""
    turns = 0
    for event in transcript.events:
        if event.event == "model" and event.output:
            turns += 1
            continue
        yield turns, event


def _sub_agent_span(spans: dict[str, Any], span_id, main_id):
    """The nearest enclosing sub-agent span; None on/above the main lane."""
    span = nearest_agent_span(spans, span_id)
    return None if span is None or span.id == main_id else span


def _eval_header(uri: str) -> dict[str, Any] | None:
    """One source log's header subset for the intro (header-only read,
    samples never parsed); None when the header cannot be read."""
    try:
        log = read_eval_log(uri, header_only=True)
    except Exception:
        return None
    e = log.eval
    return {
        "task_file": e.task_file,
        "task_version": e.task_version,
        "config": e.config.model_dump(exclude_none=True),
        "sandbox": e.sandbox.type if e.sandbox else None,
        "dataset": e.dataset.model_dump(exclude_none=True) if e.dataset else None,
        "scorers": (
            [score.name for score in (log.results.scores or [])] if log.results else []
        ),
    }
