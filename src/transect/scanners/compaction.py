"""Recorded Inspect compaction prompt and nudge text, never runtime defaults.

Inspect records neither text on the ``CompactionEvent``. Both are only
recoverable from the model events around it:

- The summarization call (``CompactionSummary``, also the fallback path
  of ``CompactionAuto``) is a ``generate()`` emitted immediately before
  its compaction event, in the same span; its final input message is the
  formatted summary prompt. That position is how Inspect's compaction
  handler sequences the two events, not a guess about message content.
  Native provider compaction emits the same event type with no such call,
  so the candidate must also look like a summarizer: its output never
  enters the conversation history, while a react agent appends every
  output of its own. When a summary overshoots the threshold Inspect
  summarizes again before the one event; only the last pass is located.
- The pre-compaction memory warning has no metadata tag and never enters
  the transcript's message history - it lives only in model-event inputs.
  It is recognised by Inspect's fixed opening line. The eval-log message
  pool gives identical warnings one shared id, so the same text recurring
  before a later flush is a fresh warning, not a retained one.

Inspect's post-compaction continuation lines are hardcoded constants and
are not extracted.
"""

from typing import Any

from inspect_scout import Transcript

_MEMORY_WARNING_PREFIX = "Context compaction approaching. Use memory() to save"


def compaction_texts(transcript: Transcript) -> list[dict[str, str | None]]:
    """One ``{compaction_prompt, compaction_nudge}`` dict per compaction event,
    in event order; empty for non-Inspect sources or transcripts without
    compaction events."""
    if transcript.source_type != "eval_log":
        return []
    events, messages = transcript.events, transcript.messages
    if not any(event.event == "compaction" for event in events):
        return []
    # Both texts are user messages Inspect showed the model without adding
    # them to the conversation history, so a message that is in the history
    # (the task input, an operator note, the tail of a native compaction
    # request) is never one of them.
    history_ids = {message.id for message in messages}
    texts: list[dict[str, str | None]] = []
    last_model: dict[str | None, Any] = {}
    nudge: dict[str | None, str] = {}
    for event in events:
        lane = event.span_id
        if event.event == "model":
            last_model[lane] = event
            for message in event.input:
                if (
                    message.role == "user"
                    and message.id not in history_ids
                    and message.text.startswith(_MEMORY_WARNING_PREFIX)
                ):
                    nudge[lane] = message.text
        elif event.event == "compaction":
            prompt = None
            summarization = last_model.pop(lane, None)
            if (
                event.source == "inspect"
                and event.type == "summary"
                and summarization is not None
                and summarization.input
                and summarization.input[-1].role == "user"
                and summarization.input[-1].id not in history_ids
                and summarization.output
                and summarization.output.message.id not in history_ids
            ):
                prompt = summarization.input[-1].text
            texts.append(
                {"compaction_prompt": prompt, "compaction_nudge": nudge.pop(lane, None)}
            )
    return texts
