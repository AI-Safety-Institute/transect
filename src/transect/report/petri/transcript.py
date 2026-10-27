"""Read complete Petri conversations and their exact saved target tree."""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from inspect_ai.log import read_eval_log_sample_async

from transect.report.petri.messages import _content, _message
from transect.report.petri.tree import extract_tree
from transect.scanners.helpers import Lanes, is_model_turn


@dataclass(frozen=True)
class PetriData:
    """Retained conversation, ancestry and an optional source availability note."""

    payload: dict[str, Any]
    tree: dict[str, Any]
    note: str | None = None


def is_petri(transcript: Any) -> bool:
    """Recognize the explicit auditor/target role contract, not task-name prose."""
    roles = {getattr(e, "role", None) for e in transcript.events if is_model_turn(e)}
    return {"auditor", "target"}.issubset(roles)


async def read_petri(transcript: Any) -> PetriData | None:
    """Use the original sample for its final store; a missing source is explicit.

    Scout retains events and timelines, but not EvalSample.store. If the original
    log is unavailable, those retained events still support the complete viewer
    and branch diagram; no final surviving path is asserted without the store.
    """
    if not is_petri(transcript):
        return None
    source = getattr(transcript, "source_uri", None)
    sample = transcript
    note = None
    if getattr(transcript, "source_type", None) == "eval_log" and source:
        try:
            original = await read_eval_log_sample_async(
                source,
                id=transcript.task_id,
                epoch=transcript.task_repeat or 1,
                resolve_attachments=True,
            )
            if (
                str(original.id) != str(transcript.task_id)
                or original.epoch != (transcript.task_repeat or 1)
                or _signature(original) != _signature(transcript)
            ):
                raise ValueError(
                    "Original sample no longer matches retained model calls"
                )
            sample = original
        except Exception as exc:
            note = f"Original Petri sample unavailable: {type(exc).__name__}: {exc}"
    payload = build_payload(sample, transcript.transcript_id)
    raw = sample.model_dump(mode="json")
    try:
        tree = extract_tree(raw, payload)
    except (KeyError, ValueError) as exc:
        tree = {"status": "unavailable", "branches": [], "reason": str(exc)}
    return PetriData(payload, tree, note)


def build_payload(sample: Any, transcript_id: str) -> dict[str, Any]:
    """Pool exact inputs and outputs, retaining all readable text and tool results."""
    lanes = Lanes(sample)
    messages: list[dict[str, Any]] = []
    pool: dict[str, int] = {}
    turns: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    attempts = {span.id: i for i, span in enumerate(lanes.begins, 1)}
    results = {e.id: e for e in sample.events if e.event == "tool"}

    def add(raw: dict[str, Any]) -> int:
        # Match complete messages: a reused ID with edited text is not authored
        # target output. Generated attribution is stamped separately below.
        key = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        if key not in pool:
            pool[key] = len(messages)
            messages.append(_message(raw))
        return pool[key]

    for event in sample.events:
        if not is_model_turn(event) or event.role not in ("auditor", "target"):
            continue
        lane = lanes.sub_agent_of(event)
        if not lanes.is_main(event) and lane is None:
            continue
        role = event.role
        raw = (
            event.output.message.model_dump(mode="json")
            if event.output.choices
            else {"content": "[No output recorded]", "role": "assistant"}
        )
        output = add(raw)
        inputs = [add(m.model_dump(mode="json")) for m in event.input or []]
        role_counts[role] += 1
        lane_turn = lane_counts[lane.id] if lane else None
        if lane:
            lane_counts[lane.id] += 1
        turns.append(
            {
                "number": len(turns) + 1,
                "role": role,
                "role_turn": role_counts[role] + (role == "auditor"),
                "model": event.model,
                "output": output,
                "inputs": inputs,
                "attempt": attempts.get(lane.id) if lane else None,
                "lane_turn": lane_turn,
                "citations": [],
            }
        )
        messages[output]["generated"] = True
        for tool in messages[output]["tools"]:
            result = results.get(tool["id"])
            if result:
                # Tool results can be structured values as well as content lists.
                value = result.model_dump(mode="json")
                tool["result"] = _content(value.get("result"))[0]
                tool["error"] = value.get("error")
    return {
        "sample_id": getattr(sample, "id", getattr(sample, "task_id", None)),
        "epoch": getattr(sample, "epoch", getattr(sample, "task_repeat", 1)),
        "transcript_id": transcript_id,
        "messages": messages,
        "turns": turns,
        "target_model": next(
            (t["model"] for t in turns if t["role"] == "target"), None
        ),
    }


def _signature(transcript: Any) -> list[tuple[Any, ...]]:
    return [
        (e.uuid, e.role, e.output.message.id if e.output.choices else None)
        for e in transcript.events
        if is_model_turn(e) and e.role in ("auditor", "target")
    ]
