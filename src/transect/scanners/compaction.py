"""Recorded Inspect compaction prompts, without importing runtime defaults.

Inspect marks summary messages with metadata.summary. Matching their
summary body to a model call's output identifies the summarization call
without guessing from an arbitrary user message before a compaction.
Memory warnings have no metadata tag, so recognition uses Inspect's
recorded warning prefix and excludes human-sourced messages.
Anthropic native compaction instead leaves a data block and a resume
message in the first subsequent model input. Its summary is output,
not evidence of the provider's unrecorded summarization prompt.
"""

import asyncio
from itertools import chain

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log_sample
from inspect_ai.model import ChatMessage
from inspect_scout import Transcript

_MEMORY_WARNING_PREFIX = (
    "Context compaction approaching. Use memory() to save concise notes on:\n"
)


async def compaction_texts(transcript: Transcript) -> list[dict[str, str | None]]:
    """Text in compaction-event order, scoped to Inspect eval logs."""
    if transcript.source_type != "eval_log":
        return []
    events, messages = transcript.events, transcript.messages
    if not any(event.event == "compaction" for event in events):
        return []
    # Scout can leave attachment references in pooled model input. Resolve
    # the selected sample at scan time, never depend on the log at render.
    if transcript.source_uri and any(
        "attachment://" in text
        for text in chain(
            (message.text for message in messages),
            (
                message.text
                for event in events
                if event.event == "model"
                for message in event.input
            ),
            (event.output.completion for event in events if event.event == "model"),
        )
    ):
        sample = await asyncio.to_thread(
            read_eval_log_sample,
            transcript.source_uri,
            id=transcript.task_id,
            epoch=transcript.task_repeat or 1,
            resolve_attachments=True,
        )
        events, messages = sample.events, sample.messages

    summaries: dict[str, set[str]] = {}

    def collect_summary(message: ChatMessage) -> None:
        if message.role != "user" or not (message.metadata or {}).get("summary"):
            return
        _, opening, rest = message.text.partition("<summary>\n")
        body, closing, resume = rest.rpartition("\n</summary>")
        if opening and closing:
            summaries.setdefault(body, set()).add(resume.strip())

    for message in messages:
        collect_summary(message)
    for event in events:
        if event.event == "model":
            for message in event.input:
                collect_summary(message)

    last_model: dict[str | None, ModelEvent] = {}
    warnings: dict[str | None, str] = {}
    seen_warnings: set[tuple[str | None, str]] = set()
    pending_resume: dict[str | None, dict[str, str | None]] = {}
    texts = []
    for event in events:
        lane = event.span_id
        if event.event == "model":
            if lane in pending_resume:
                pending_resume.pop(lane)["compaction_resume"] = _native_resume(
                    event.input
                )
            last_model[lane] = event
            for message in event.input:
                if (
                    message.role == "user"
                    and message.source not in ("input", "operator")
                    and message.text.startswith(_MEMORY_WARNING_PREFIX)
                ):
                    key = (lane, message.id or message.text)
                    if key not in seen_warnings:
                        warnings[lane] = message.text
                        seen_warnings.add(key)
        elif event.event == "compaction":
            pending_resume.pop(lane, None)
            prompt = resume = None
            candidate = last_model.pop(lane, None)
            if (
                event.source == "inspect"
                and event.type == "summary"
                and candidate is not None
                and candidate.output
                and candidate.output.completion
                and candidate.output.completion in summaries
                and candidate.input
                and candidate.input[-1].role == "user"
                and candidate.input[-1].source not in ("input", "operator")
                and "attachment://" not in candidate.input[-1].text
            ):
                prompt = candidate.input[-1].text
                continuations = summaries[candidate.output.completion]
                if len(continuations) == 1:
                    resume = next(iter(continuations)) or None
            text = {
                "compaction_prompt": prompt,
                "compaction_nudge": warnings.pop(lane, None),
                "compaction_resume": resume,
            }
            texts.append(text)
            if event.source == "inspect" and event.type == "summary" and resume is None:
                pending_resume[lane] = text
    return texts


def _native_resume(messages: list[ChatMessage]) -> str | None:
    """The recorded user instruction immediately after a native summary block."""
    conversation = [message for message in messages if message.role != "system"]
    if len(conversation) != 2:
        return None
    summary, resume = conversation
    if (
        summary.role != "assistant"
        or isinstance(summary.content, str)
        or resume.role != "user"
        or resume.source in ("input", "operator")
        or "attachment://" in resume.text
    ):
        return None
    for content in summary.content:
        if content.type == "data":
            metadata = content.data.get("compaction_metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("type") == "anthropic_compact"
                and metadata.get("content")
            ):
                return resume.text or None
    return None
