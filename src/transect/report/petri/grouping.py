"""Map retained calls to the saved auditor phases and rollback markers."""

from collections import Counter
from typing import Any

import pandas as pd


def group_turns(payload: Any, timeline: Any, phases: Any, index: Any = 0) -> Any:
    """Partition every conversation call exactly once, using stored phase labels."""
    conversation = timeline.loc[timeline.turn.notna() | timeline.agent_span_id.notna()]
    if len(conversation) != len(payload["turns"]):
        raise ValueError("Conversation and token timeline have different populations")
    if phases.turn.duplicated().any():
        raise ValueError("Duplicate auditor turn in phase assignments")
    labels = {
        int(row.turn): int(row.phase_index)
        for row in phases.itertuples()
        if pd.notna(row.phase) and pd.notna(row.phase_index)
    }
    groups: dict[str, dict[str, Any]] = {}
    preceding_phase = None
    for expected, (call, telemetry) in enumerate(
        zip(payload["turns"], conversation.itertuples(), strict=True), 1
    ):
        auditor = pd.notna(telemetry.turn)
        if call["number"] != expected or call["role"] != (
            "auditor" if auditor else "target"
        ):
            raise ValueError(
                "Conversation roles/order disagree with the token timeline"
            )
        if auditor:
            preceding_phase = labels.get(int(telemetry.turn))
        card = (
            f"phase-{index}-{preceding_phase}"
            if preceding_phase is not None
            else "unclassified-transcript"
        )
        group = groups.setdefault(card, {"card": card, "turns": [], "roles": Counter()})
        group["turns"].append(expected)
        group["roles"][call["role"]] += 1
    return list(groups.values())


def add_continuations(payload: Any, groups: Any) -> Any:
    """Mark confirmed rollback boundaries, without inferring target ancestry.

    Boundaries follow the auditor output containing the tool call. Several
    tools can execute within that output; this is a recorded-order grouping,
    not a reconstruction of the target context used by each nested call.
    """
    owners = {number: group for group in groups for number in group["turns"]}
    for group in groups:
        group["segments"] = []
    continuation = 1
    for turn in payload["turns"]:
        number = turn["number"]
        segments = owners[number]["segments"]
        if (
            not segments
            or segments[-1]["continuation"] != continuation
            or segments[-1]["turns"][-1] != number - 1
            or segments[-1]["rollbacks"]
        ):
            segments.append(
                {"continuation": continuation, "turns": [], "rollbacks": []}
            )
        segment = segments[-1]
        segment["turns"].append(number)
        if turn["role"] != "auditor":
            continue
        for index, tool in enumerate(payload["messages"][turn["output"]]["tools"]):
            if tool["name"] != "rollback_conversation":
                continue
            arguments = tool.get("arguments")
            destination = (
                arguments.get("message_id") if isinstance(arguments, dict) else None
            )
            result = tool.get("result")
            if tool.get("error") is not None:
                status = "failed"
            elif (
                isinstance(destination, str)
                and isinstance(result, str)
                and result.strip() == f"Rolled back to message_id {destination}."
            ):
                status = "confirmed"
            else:
                status = "unconfirmed"
            event = {
                "turn": number,
                "tool_index": index,
                "destination": destination,
                "status": status,
            }
            if status == "confirmed":
                continuation += 1
                event["next_continuation"] = continuation
            segment["rollbacks"].append(event)
