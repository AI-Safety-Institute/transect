"""Normalize complete recorded messages without exposing redacted reasoning."""

import json
from typing import Any


def _content(content: Any) -> tuple[str, str]:
    """Keep readable text and recorded summaries; never render opaque reasoning."""
    if isinstance(content, dict) and "type" in content:
        content = [content]
    if isinstance(content, str):
        return content, ""
    if isinstance(content, (dict, int, float, bool)) or (
        isinstance(content, list) and any(not isinstance(b, dict) for b in content)
    ):
        return json.dumps(content, ensure_ascii=False, indent=2), ""
    text, reasoning = [], []
    for block in content or []:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "reasoning":
            summary = block.get("summary")
            if summary:
                reasoning.append(
                    summary if isinstance(summary, str) else json.dumps(summary)
                )
            elif not block.get("redacted"):
                reasoning.append(block.get("reasoning", ""))
        else:
            text.append(f"[{block.get('type', 'non-text')} content recorded]")
    return "\n".join(text), "\n".join(reasoning)


def _message(message: dict[str, Any]) -> dict[str, Any]:
    text, reasoning = _content(message.get("content"))
    return {
        "id": message.get("id"),
        "role": message.get("role"),
        "text": text,
        "reasoning": reasoning,
        "tool_call_id": message.get("tool_call_id"),
        "tools": [
            {
                "id": call.get("id"),
                "name": call.get("function"),
                "arguments": call.get("arguments", {}),
            }
            for call in message.get("tool_calls") or []
        ],
    }


def _resolve_attachments(value: Any, attachments: Any) -> Any:
    if isinstance(value, str) and value.startswith("attachment://"):
        key = value.removeprefix("attachment://")
        if key not in attachments:
            raise ValueError(f"Unresolved attachment: {value}")
        return attachments[key]
    if isinstance(value, dict):
        return {
            key: _resolve_attachments(item, attachments) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_attachments(item, attachments) for item in value]
    return value
