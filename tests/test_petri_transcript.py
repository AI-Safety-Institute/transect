"""Complete Petri source text and explicit source-identity failures."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from inspect_ai.event import ToolEvent
from inspect_ai.model import ChatMessageUser, ContentReasoning, ContentText
from test_scanners_mechanical import _auditor_scaffold

from transect.report.petri.messages import _content
from transect.report.petri.transcript import build_payload, is_petri, read_petri


def test_payload_preserves_roles_text_tools_and_redaction():
    """Payloads retain complete text and tools while excluding opaque reasoning."""
    transcript = _auditor_scaffold()
    event = next(e for e in transcript.events if getattr(e, "role", None) == "auditor")
    event.input = [ChatMessageUser(content="context " * 10000)]
    event.output.message.content = [
        ContentText(text="output " * 10000),
        ContentReasoning(reasoning="SECRET", redacted=True),
    ]
    tool_id = event.output.message.tool_calls[0].id
    transcript.events.append(
        ToolEvent(
            id=tool_id,
            function="send_message",
            arguments={},
            result=42,
        )
    )
    payload = build_payload(transcript, "tid")
    assert is_petri(transcript)
    assert payload["target_model"] == "m/target"
    assert [t["role"] for t in payload["turns"]] == [
        "auditor",
        "target",
        "auditor",
        "target",
        "auditor",
        "target",
    ]
    output = payload["messages"][payload["turns"][0]["output"]]
    assert output["text"] == "output " * 10000
    assert output["reasoning"] == ""
    assert output["tools"][0]["result"] == "42"
    assert (
        payload["messages"][payload["turns"][0]["inputs"][0]]["text"]
        == "context " * 10000
    )
    assert [t["lane_turn"] for t in payload["turns"] if t["role"] == "target"] == [
        0,
        1,
        0,
    ]
    assert _content(42) == ("42", "")


def test_missing_source_retains_calls_without_claiming_a_surviving_path(monkeypatch):
    """A missing original source retains Scout text with an explicit absence note."""
    transcript = _auditor_scaffold()
    transcript.transcript_id = "tid"
    transcript.task_id, transcript.task_repeat = "sample", 2
    transcript.source_uri = "missing.eval"
    transcript.model_dump = lambda **kwargs: {"events": [], "timelines": []}
    reader = AsyncMock(side_effect=FileNotFoundError("missing.eval"))
    monkeypatch.setattr(
        "transect.report.petri.transcript.read_eval_log_sample_async", reader
    )
    result = asyncio.run(read_petri(transcript))
    assert result.tree["status"] == "unavailable"
    assert len(result.payload["turns"]) == 6
    assert "Original Petri sample unavailable" in result.note
    reader.assert_awaited_once_with(
        "missing.eval", id="sample", epoch=2, resolve_attachments=True
    )
    for event in transcript.events:
        if getattr(event, "role", None) == "auditor":
            event.role = "agent"
    assert asyncio.run(read_petri(transcript)) is None


@pytest.mark.parametrize("mismatch", [False, True])
def test_original_sample_identity_is_checked_before_using_final_store(
    monkeypatch, mismatch
):
    """A replaced source cannot silently supply another sample's final context."""
    from types import SimpleNamespace

    transcript = _auditor_scaffold()
    transcript.transcript_id = "tid"
    transcript.task_id, transcript.task_repeat = "sample", 2
    transcript.source_uri = "original.eval"
    transcript.model_dump = lambda **kwargs: {"events": [], "timelines": []}
    original = SimpleNamespace(
        id="other" if mismatch else "sample",
        epoch=2,
        events=transcript.events,
        model=transcript.model,
        model_dump=transcript.model_dump,
    )
    monkeypatch.setattr(
        "transect.report.petri.transcript.read_eval_log_sample_async",
        AsyncMock(return_value=original),
    )
    result = asyncio.run(read_petri(transcript))
    assert len(result.payload["turns"]) == 6
    if mismatch:
        assert "no longer matches" in result.note
    else:
        assert result.note is None
        assert result.payload["epoch"] == 2
