"""Structural (zero-LLM) Scout scanners: the default path."""

import json
from typing import Any, cast

from inspect_ai.log import read_eval_log
from inspect_scout import Result, Scanner, Transcript, scanner
from pydantic import JsonValue

from transect.scanners.compaction import compaction_texts
from transect.scanners.helpers import Lanes, all_model_turns, is_model_turn

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
    """Per-model-turn token usage on the orchestrator turn axis, plus the
    sub-agent span record.

    value = {"timeline": [...], "spans": [...], "lane_activity": [...]}.

    ``timeline``: one entry per model turn in any lane, in event order:
    ``turn`` (the orchestrator ordinal; None on a sub-agent turn and on
    an init/scorer call, which are off the axis), ``lane_turn`` (0-based
    within the turn's own lane; None on an off-axis call, which is in no
    lane), ``agent_lane`` / ``agent_span_id`` (None on the orchestrator),
    ``n_tool_calls``, ``timestamp`` / ``completed``
    (ISO strings; None when unrecorded), and the ModelUsage fields
    (None = not reported, never 0).

    ``spans``: one entry per sub-agent span (`helpers.subagent_span_begins`'
    definition): ``agent_span_id`` / ``agent_lane`` (the span's id and
    name), ``spawn_turn`` (the orchestrator turn preceding the
    span_begin in event order; 0 when none does), ``first_at`` /
    ``last_at`` (the span's first and last model or tool event, start
    and completion), ``end_at`` and ``end_recorded`` (the span_end's
    timestamp; an OpenClaw import synthesises its ends at last activity,
    so recorded is False there), ``event_order_end_turn`` (the
    orchestrator turn preceding the span_end; None when never closed).

    ``lane_activity``: tool events inside sub-agent spans, one entry
    ``{turn, agent_span_id, agent_lane, tool_calls, busy_seconds,
    started_at}`` per (orchestrator turn preceding the event, span).
    """

    async def execute(transcript: Transcript) -> Result:
        lanes = Lanes(transcript)
        timeline: list[dict[str, Any]] = []
        lane_counts: dict[str, int] = {}
        for event, calls in all_model_turns(transcript):
            sub = lanes.sub_agent_of(event)
            is_main = lanes.is_main(event)
            # an off-axis call (init/scorer/utility: neither orchestrator
            # nor sub-agent) belongs to no lane and counts in none
            lane_key = sub.id if sub else "__main__" if is_main else None
            lane_turn = None
            if lane_key is not None:
                lane_turn = lane_counts.get(lane_key, 0)
                lane_counts[lane_key] = lane_turn + 1
            usage = event.output.usage
            entry: dict[str, Any] = {
                "turn": lane_turn if is_main else None,
                "lane_turn": lane_turn,
                "agent_lane": sub.name if sub else None,
                "agent_span_id": sub.id if sub else None,
                "n_tool_calls": len(calls),
                "timestamp": _iso(getattr(event, "timestamp", None)),
                "completed": _iso(getattr(event, "completed", None)),
            }
            for field in _USAGE_FIELDS:
                entry[field] = getattr(usage, field, None) if usage else None
            timeline.append(entry)
        # only Inspect records span ends; an OpenClaw import synthesises
        # them at last activity
        ends_recorded = transcript.source_type == "eval_log"
        turns = lane_counts.get("__main__", 0)
        return Result(
            value={
                "timeline": cast(JsonValue, timeline),
                "spans": cast(JsonValue, _span_records(lanes, ends_recorded)),
                "lane_activity": cast(JsonValue, _lane_activity(lanes)),
            },
            explanation=(
                f"{turns} orchestrator turns, {len(timeline) - turns} other turns"
            ),
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
        prompt = None
        if transcript.source_type == "eval_log":
            if transcript.source_uri:
                uri = transcript.source_uri
                if uri not in header_of:
                    header_of[uri] = _eval_header(uri)
                header = header_of[uri]
            compaction = (transcript.agent_args or {}).get("compaction")
            prompt = compaction.get("prompt") if isinstance(compaction, dict) else None
        value = {
            "system_prompt": system_prompt,
            "task_message": task_message,
            "agent_args": transcript.agent_args,
            "task_args": meta.get("task_args"),
            "generate_config": meta.get("generate_config"),
            "model_roles": meta.get("model_roles"),
            "header": header,
            "compaction_prompt": prompt if isinstance(prompt, str) else None,
        }
        return Result(value=cast(JsonValue, value), explanation="eval setup")

    return execute


@scanner(
    messages=["user", "assistant"],
    # span events: the orchestrator lane is resolved from the timeline
    events=["model", "compaction", "span_begin", "span_end"],
)
def context_flush() -> Scanner[Transcript]:
    """Context-window compactions (flushes), from explicit compaction events.

    value = {"flushes": [entry, ...]} with one entry per compaction:
    turn (the orchestrator turn the flush precedes, i.e. the first
    post-flush orchestrator turn), agent_span_id (the sub-agent lane the
    compaction happened in; None on the orchestrator), lane_turn (the
    first post-flush turn of the compacted lane itself: equal to turn on
    the orchestrator, the sub-agent's own lane ordinal otherwise), type,
    source, tokens_before, tokens_after, role, metadata - recorded as
    the event reports them (optional facts remain None). Inspect eval logs also
    carry compaction_prompt (the summarization call's formatted prompt)
    and compaction_nudge (the pre-compaction memory warning), both as
    the model saw them; see scanners/compaction.py for how they are
    located. Text the source never recorded stays None, including on
    OpenClaw imports, whose export carries neither.
    """

    async def execute(transcript: Transcript) -> Result:
        flushes: list[dict[str, Any]] = []
        texts = iter(compaction_texts(transcript))
        lanes = Lanes(transcript)
        lane_counts: dict[str, int] = {}
        for turn, event in lanes.events_before_turn():
            if is_model_turn(event):
                sub = lanes.sub_agent_of(event)
                if sub is not None:
                    lane_counts[sub.id] = lane_counts.get(sub.id, 0) + 1
                continue
            if event.event != "compaction":
                continue
            lane = lanes.sub_agent_of(event)
            flushes.append(
                {
                    "turn": turn,
                    "agent_span_id": lane.id if lane is not None else None,
                    "lane_turn": lane_counts.get(lane.id, 0) if lane else turn,
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


@scanner(
    messages="all",
    # span events: the orchestrator lane is resolved from the timeline
    events=cast(
        "list[Any]",
        ["model", "input", "approval", "compaction", "span_begin", "span_end"],
    ),
)
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

    value = {"interventions": [entry, ...]}: turn (the orchestrator turn
    the intervention precedes), channel, initiator, prompt, content,
    outcome.
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

        # Human messages live in the history, not the event stream. The log
        # records which orchestrator call first saw each message: the first
        # orchestrator model event whose input carries its id (or merged it
        # into a combined message) is the turn it precedes. A message no
        # input records (one planted into the history, a history without
        # its event stream) falls back to the history's footprints of
        # orchestrator turns.
        lanes = Lanes(transcript)
        first_seen: dict[str, int] = {}
        turn_of_output: dict[str, int] = {}
        # every non-orchestrator output (sub-agent, init, scorer): an Inspect
        # handoff appends a sub-agent's to the parent thread, where they
        # must not advance the orchestrator axis
        sub_outputs = {
            event.output.message.id
            for _, event in lanes.events_before_turn()
            if is_model_turn(event) and event.output.message.id
        }
        for turn, event, _calls in lanes.turns():
            output_id = event.output.message.id
            for seen in event.input:
                combined = (getattr(seen, "metadata", None) or {}).get("combined_from")
                for seen_id in (seen.id, *(combined or [])):
                    if seen_id is not None:
                        first_seen.setdefault(seen_id, turn)
            if output_id is not None:
                # first occurrence wins: a cached generate replays an output
                turn_of_output.setdefault(output_id, turn)
        # summary flushes by the turn they precede (as context_flush records
        # them): the fallback footprint for a summary message no input saw
        summary_flushes = [
            turn
            for turn, event in lanes.events_before_turn()
            if event.event == "compaction" and event.type == "summary"
        ]
        next_turn = 0
        seen_task_prompt = False
        for message in transcript.messages:
            if message.role == "assistant":
                if message.id in sub_outputs:
                    continue
                # an orchestrator assistant message no event recorded still
                # advances the axis by one; a repeated id never moves it back
                recorded = turn_of_output.get(message.id or "")
                next_turn = max(next_turn + 1, 0 if recorded is None else recorded + 1)
                continue
            if message.role != "user":
                continue
            seen = first_seen.get(message.id or "")
            if (getattr(message, "metadata", None) or {}).get("summary"):
                # a compaction summary stands for the summarizer turn before
                # it; unseen, it takes the first summary flush ahead of the
                # axis (a flush at or behind it belongs to an earlier window,
                # and trim, edit and native flushes never qualify by type)
                if seen is not None:
                    next_turn = max(next_turn, seen)
                else:
                    ahead = next((t for t in summary_flushes if t > next_turn), None)
                    if ahead is not None:
                        next_turn = ahead
                continue
            source = getattr(message, "source", None)
            if source not in ("operator", "input"):
                continue
            if not seen_task_prompt:
                seen_task_prompt = True
                continue
            if seen is not None:
                next_turn = max(next_turn, seen)
            entry(
                seen if seen is not None else next_turn,
                source,
                "human",
                (message.text or "").strip(),
            )

        for turn, event in lanes.events_before_turn():
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


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _span_records(lanes: Lanes, ends_recorded: bool) -> list[dict[str, Any]]:
    """One record per sub-agent span: spawn and end anchors in
    orchestrator turns, activity timestamps (see `token_timeline`)."""
    sub_ids = lanes.sub_ids
    records: dict[str, dict[str, Any]] = {}
    for before, event in lanes.events_before_turn():
        kind = event.event
        if kind == "span_begin":
            if event.id not in sub_ids:
                continue
            records[event.id] = {
                "agent_span_id": event.id,
                "agent_lane": event.name,
                "spawn_turn": max(before - 1, 0),
                "first_at": None,
                "last_at": None,
                "end_at": None,
                "end_recorded": False,
                "event_order_end_turn": None,
            }
            continue
        if kind == "span_end":
            record = records.get(getattr(event, "id", ""))
            if record is not None:
                record["end_at"] = _iso(getattr(event, "timestamp", None))
                record["end_recorded"] = ends_recorded
                record["event_order_end_turn"] = max(before - 1, 0)
            continue
        if kind not in ("model", "tool"):
            continue
        sub = lanes.sub_agent_of(event)
        if sub is None or sub.id not in records:
            continue
        record = records[sub.id]
        started = _iso(getattr(event, "timestamp", None))
        finished = _iso(getattr(event, "completed", None)) or started
        if started is not None and (
            record["first_at"] is None or started < record["first_at"]
        ):
            record["first_at"] = started
        if finished is not None and (
            record["last_at"] is None or finished > record["last_at"]
        ):
            record["last_at"] = finished
    return list(records.values())


def _lane_activity(lanes: Lanes) -> list[dict[str, Any]]:
    """Tool activity per (orchestrator turn preceding the event, sub-agent
    span): calls, busy time, start. Sub-agents whose activity is
    tool-events-only (no model turns of their own, as some OpenClaw
    exports record them) show up only here."""
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for before, event in lanes.events_before_turn():
        if event.event != "tool":
            continue
        sub = lanes.sub_agent_of(event)
        if sub is None:
            continue
        row = rows.setdefault(
            (max(before - 1, 0), sub.id),
            {
                "turn": max(before - 1, 0),
                "agent_lane": sub.name,
                "agent_span_id": sub.id,
                "tool_calls": 0,
                # busy time from per-call durations; None when never reported
                "busy_seconds": None,
                "started_at": None,
            },
        )
        row["tool_calls"] += 1
        started = getattr(event, "timestamp", None)
        completed = getattr(event, "completed", None)
        if started is None:
            continue
        if row["started_at"] is None or started < row["started_at"]:
            row["started_at"] = started
        if completed is not None:
            busy = (completed - started).total_seconds()
            row["busy_seconds"] = (row["busy_seconds"] or 0.0) + busy
    for row in rows.values():
        row["started_at"] = _iso(row["started_at"])
        if row["busy_seconds"] is not None:
            row["busy_seconds"] = round(row["busy_seconds"], 1)
    return [rows[key] for key in sorted(rows)]


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
