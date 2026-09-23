"""Inspect compaction text stays attributable from scan through stored report."""

import pandas as pd
import pytest
from helpers import StubTranscript, model_turn, run_item
from inspect_ai.event import CompactionEvent
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentData,
)
from inspect_scout import Transcript
from test_frames import raw_row

from transect.frames.flushes import flushes_df
from transect.frames.transcript_info import transcript_info_df
from transect.scanners.base import context_flush, eval_setup

PROMPT = (
    "Summarize the work completed so far. Include the user's request, decisions, "
    "changed files, errors, and remaining tasks.\n"
    "Keep <constraints> & next steps.\n"
    "Do not invent results or treat attempted actions as completed."
)
NUDGE = (
    "Context compaction approaching. Use memory() to save concise notes on:\n"
    "- Key decisions\n- Next steps"
)
RESUME = "Continue from the saved state.\nRespect <constraints>."
SUMMARY = "Finished the baseline; validation remains."
METADATA = {
    "strategy": "CompactionSummary",
    "messages_before": 8,
    "messages_after": 3,
    "trigger": "forced",
    "custom": {"preserved": True},
}


def compaction_events():
    """The recorded shapes emitted by Inspect's summary and memory paths."""
    warning = ChatMessageUser(content=NUDGE)
    summary = ChatMessageUser(
        content=(
            "[CONTEXT COMPACTION SUMMARY]\n\n"
            f"<summary>\n{SUMMARY}\n</summary>\n\n{RESUME}"
        ),
        metadata={"summary": True},
    )
    return [
        model_turn("saved notes", span_id="lead", input=[warning]),
        model_turn(
            SUMMARY,
            span_id="lead",
            input=[warning, ChatMessageUser(content=PROMPT)],
        ),
        model_turn("unrelated sub-agent", span_id="worker"),
        CompactionEvent(
            type="summary",
            source="inspect",
            span_id="lead",
            role="solver",
            tokens_before=900,
            tokens_after=200,
            metadata=METADATA,
        ),
        model_turn("resumed", span_id="lead", input=[summary]),
    ]


@pytest.mark.parametrize("source_type", ["eval_log", "openclaw"])
def test_compaction_records_text_only_for_inspect(source_type):
    """Role and metadata survive projection while text extraction is eval-only."""
    transcript = StubTranscript(compaction_events())
    transcript.source_type = source_type
    result = run_item(context_flush(), transcript)
    frame = flushes_df(
        pd.DataFrame([raw_row(result)]),
        pd.DataFrame(
            columns=["transcript_id", "turn", "agent_span_id", "agent_lane", "context"]
        ),
    )
    row = frame.iloc[0]
    assert (row.turn, row.role, row.strategy, row.trigger) == (
        3,
        "solver",
        "CompactionSummary",
        "forced",
    )
    assert row.metadata == METADATA
    assert (row.messages_before, row.messages_after) == (8, 3)
    assert str(frame.messages_after.dtype) == "Int64"
    assert (row.compaction_prompt, row.compaction_nudge, row.compaction_resume) == (
        (PROMPT, NUDGE, RESUME) if source_type == "eval_log" else (None, None, None)
    )


@pytest.mark.parametrize("missing", ["summary", "human_warning", "wrong_lane"])
def test_compaction_does_not_guess_unrecorded_text(missing):
    """Unmatched summaries, human messages, and other lanes cannot supply text."""
    events = compaction_events()
    if missing == "summary":
        events[-1].input = []
    elif missing == "human_warning":
        events[0].input[0].source = "input"
    else:
        events[3].span_id = "elsewhere"
    row = run_item(context_flush(), StubTranscript(events)).value["flushes"][0]
    if missing == "human_warning":
        assert row["compaction_nudge"] is None
        assert row["compaction_prompt"] == PROMPT
    else:
        assert row["compaction_prompt"] is None
        assert row["compaction_resume"] is None
    if missing == "wrong_lane":
        assert row["compaction_nudge"] is None


def test_compaction_does_not_reuse_an_old_warning():
    """A warning retained in later model histories belongs to its first flush."""
    events = compaction_events()
    events.extend([events[1], events[3]])
    rows = run_item(context_flush(), StubTranscript(events)).value["flushes"]
    assert [row["compaction_nudge"] for row in rows] == [NUDGE, None]


@pytest.mark.parametrize(
    "case",
    ["recorded", "openclaw", "wrong_lane", "human", "no_summary", "stale_summary"],
)
def test_native_compaction_extracts_only_the_recorded_resume(case):
    """Native summary output identifies the resume nudge, never a prompt."""
    resume = ChatMessageUser(content="Please continue working.")
    if case == "human":
        resume.source = "input"
    summary = ChatMessageAssistant(
        content=[
            ContentData(
                data={
                    "compaction_metadata": {
                        "type": "anthropic_compact",
                        "content": SUMMARY,
                    }
                }
            )
        ]
    )
    if case == "no_summary":
        summary.content = "Ordinary assistant response."
    messages = [ChatMessageSystem(content="System instructions."), summary, resume]
    if case == "stale_summary":
        messages.append(ChatMessageAssistant(content="Already resumed."))
    flush = CompactionEvent(
        type="summary",
        source="inspect",
        span_id="lead",
        metadata={**METADATA, "strategy": "CompactionNative"},
    )
    events = [
        model_turn("working", span_id="lead"),
        flush,
        model_turn("other lane", span_id="worker"),
        model_turn(
            "resuming",
            span_id="elsewhere" if case == "wrong_lane" else "lead",
            input=messages,
        ),
        flush.model_copy(),
        model_turn("no recorded resume", span_id="lead"),
    ]
    transcript = StubTranscript(events)
    if case == "openclaw":
        transcript.source_type = "openclaw"
    rows = run_item(context_flush(), transcript).value["flushes"]
    assert [row["compaction_resume"] for row in rows] == [
        resume.text if case == "recorded" else None,
        None,
    ]
    assert all(row["compaction_prompt"] is None for row in rows)
    assert all(row["compaction_nudge"] is None for row in rows)


@pytest.mark.parametrize("source_type", ["eval_log", "openclaw"])
def test_configured_compaction_template_is_not_a_runtime_default(source_type):
    """Only an explicitly configured Inspect template is stored and projected."""
    for args, expected in (({}, None), ({"compaction": {"prompt": PROMPT}}, PROMPT)):
        result = run_item(
            eval_setup(),
            Transcript(transcript_id="t", source_type=source_type, agent_args=args),
        )
        assert result.value["compaction_prompt"] == (
            expected if source_type == "eval_log" else None
        )
        raw = {**raw_row(result), "transcript_source_type": source_type}
        info = transcript_info_df(pd.DataFrame([raw]))
        assert info.compaction_prompt.iloc[0] == result.value["compaction_prompt"]


def test_empty_flushes_keep_the_new_columns():
    """No events still yields the complete nullable compaction contract."""
    result = run_item(context_flush(), StubTranscript([]))
    frame = flushes_df(
        pd.DataFrame([raw_row(result)]),
        pd.DataFrame(
            columns=["transcript_id", "turn", "agent_span_id", "agent_lane", "context"]
        ),
    )
    assert frame.empty
    assert {
        "compaction_prompt",
        "compaction_nudge",
        "compaction_resume",
        "metadata",
    } <= set(frame)
    assert str(frame.messages_before.dtype) == "Int64"
