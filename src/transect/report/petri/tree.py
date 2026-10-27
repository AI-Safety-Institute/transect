"""Project Inspect's retained target tree into a clickable report overview.

No ancestry or scanner decisions are inferred here. Parent branches and fork
anchors come from the saved Inspect timeline; flags come from saved judgments.
Optional evidence annotations supply locations without changing the saved tree.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from transect.report.petri.path import final_path

CACHE_VERSION = 2


def retained_tree(payload: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    """Read the saved timeline directly, avoiding full model-event hydration."""
    source = Path(payload.get("source_file") or "")
    if not source.is_file():
        return {
            "status": "unavailable",
            "reason": "Original Inspect log is unavailable.",
            "branches": [],
        }
    stat = source.stat()
    key = [
        CACHE_VERSION,
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        payload["sample_id"],
        payload["epoch"],
        _identity(payload),
    ]
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved.get("key") == key:
            return saved["tree"]
    # Inspect installs zstd/multiframe support for its .eval ZIP archives.
    from inspect_ai._util import zipfile as _zip_support  # noqa: F401

    with zipfile.ZipFile(source) as archive:
        sample = json.loads(
            archive.read(
                f"samples/{payload['sample_id']}_epoch_{payload['epoch']}.json"
            )
        )
    if sample["id"] != payload["sample_id"] or sample["epoch"] != payload["epoch"]:
        raise ValueError("Inspect sample identity does not match the report")
    tree = extract_tree(sample, payload)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"key": key, "tree": tree}, separators=(",", ":")))
    temporary.replace(cache_path)
    return tree


def extract_tree(sample: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Keep precise saved anchors, target calls, and the auditor rollback link."""
    timeline = next(
        (t for t in sample.get("timelines", []) if t.get("name") == "target"), None
    )
    if timeline is None:
        return {
            "status": "unavailable",
            "reason": "No target tree was retained in this Inspect log.",
            "branches": [],
        }
    events = sample["events"]
    models = [
        e
        for e in events
        if e["event"] == "model"
        and e.get("output") is not None
        and e.get("role") in ("auditor", "target")
    ]
    if len(models) != len(payload["turns"]):
        raise ValueError("Inspect model events do not match the retained conversation")
    event_turns = {}
    tool_turns = {}
    for event, turn in zip(models, payload["turns"], strict=True):
        choices = event["output"].get("choices") or []
        mid = choices[0]["message"].get("id") if choices else None
        output = payload["messages"][turn["output"]]
        if event["role"] != turn["role"] or mid != output.get("id"):
            raise ValueError(
                "Inspect model outputs/order do not match the retained conversation"
            )
        event_turns[event["uuid"]] = turn["number"]
        if turn["role"] == "auditor":
            for tool in output["tools"]:
                if tool.get("id"):
                    tool_turns[tool["id"]] = (turn["number"], tool)
    indexed = {e["uuid"]: e for e in events if e.get("uuid")}
    order = {e["uuid"]: i for i, e in enumerate(events) if e.get("uuid")}
    rollbacks, pending = {}, None
    for event in events:
        if event["event"] == "tool" and event.get("function") in (
            "rollback_conversation",
            "restart_conversation",
        ):
            mapped = tool_turns.get(event.get("id"))
            pending = None
            if mapped:
                number, tool = mapped
                args = tool.get("arguments") or {}
                destination = args.get("message_id") if isinstance(args, dict) else None
                result = tool.get("result")
                confirmed = (
                    tool.get("error") is None
                    and isinstance(destination, str)
                    and isinstance(result, str)
                    and result.strip() == f"Rolled back to message_id {destination}."
                )
                if confirmed:
                    pending = {
                        "turn": number,
                        "destination": destination,
                        "kind": event["function"],
                    }
        elif event["event"] == "branch":
            rollbacks[event["span_id"]] = pending
            pending = None
        elif event["event"] == "model" and event.get("role") == "auditor":
            pending = None  # Never carry an unmatched tool call into a later retry.

    def content_events(node: Any) -> Any:
        for item in node.get("content", []):
            if item["type"] == "event":
                yield indexed[item["event"]]
            elif item["type"] == "span":
                yield from content_events(item)

    branches = []
    seen_turns = []

    def walk(node: Any, parent: Any = None) -> Any:
        points: list[dict[str, Any]] = []
        anchors: dict[str, int] = {}
        refs = list(content_events(node))
        for event in refs:
            if event["event"] == "anchor":
                # An anchor after a model call names that output. Other anchors
                # represent supplied context between calls and get their own point.
                anchor = event["anchor_id"]
                if points and points[-1].get("message_id") == anchor:
                    anchors[anchor] = len(points) - 1
                else:
                    anchors[anchor] = len(points)
                    points.append({"kind": "context", "message_id": anchor})
            elif event["event"] == "model" and event.get("role") == "target":
                turn = event_turns[event["uuid"]]
                seen_turns.append(turn)
                points.append(
                    {
                        "kind": "target",
                        "turn": turn,
                        "message_id": payload["messages"][
                            payload["turns"][turn - 1]["output"]
                        ].get("id"),
                    }
                )
        branch = {
            "id": node["id"],
            "name": node.get("name", "Branch"),
            "parent": parent,
            "anchor": node.get("branched_from"),
            "points": points,
            "anchors": anchors,
            "rollback": rollbacks.get(node["id"]),
            "order": min((order[e["uuid"]] for e in refs), default=len(events)),
        }
        branches.append(branch)
        for child in node.get("branches", []):
            walk(child, node["id"])

    walk(timeline["root"])
    expected = [t["number"] for t in payload["turns"] if t["role"] == "target"]
    if sorted(seen_turns) != expected:
        raise ValueError(
            "Inspect target tree must contain each retained target call exactly once"
        )
    tree: dict[str, Any] = {
        "status": "ok",
        "branches": sorted(branches, key=lambda b: b["order"]),
    }
    tree["final_path"] = final_path(
        tree, sample.get("store", {}).get("target_messages")
    )
    return tree


def build_diagram(
    payload: dict[str, Any],
    groups: list[dict[str, Any]],
    tree: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Group exact target-tree calls into clickable phase sections."""
    evidence = evidence or {"status": "unavailable", "turns": []}
    if tree["status"] != "ok":
        return {**tree, "evidence": evidence}
    cards = {number: group["card"] for group in groups for number in group["turns"]}
    final = tree.get("final_path", {})
    surviving = set(final.get("turns", []))
    limits = final.get("limits", {})
    by_id = {b["id"]: b for b in tree["branches"]}
    positions: dict[str, list[float]] = {}
    starts: dict[str, float] = {}
    resolving: set[str] = set()

    def layout(branch: Any) -> Any:
        bid = branch["id"]
        if bid in starts:
            return
        if bid in resolving:
            raise ValueError("Cycle in Inspect branch tree")
        resolving.add(bid)
        start = 0.0
        parent = by_id.get(branch["parent"])
        if parent:
            layout(parent)
            index = parent["anchors"].get(branch["anchor"])
            if index is not None:
                start = positions[parent["id"]][index]
        starts[bid] = start
        points = []
        for point in branch["points"]:
            start += 1.0 if point["kind"] == "target" else 0.35
            points.append(round(start, 4))
        positions[bid] = points
        resolving.remove(bid)

    for branch in tree["branches"]:
        layout(branch)
    rows = []
    for index, branch in enumerate(tree["branches"]):
        sections: list[dict[str, Any]] = []
        for i, point in enumerate(branch["points"]):
            if point["kind"] != "target":
                continue
            number = point["turn"]
            card = cards[number]
            kept = number in surviving if final.get("status") == "ok" else None
            if (
                not sections
                or sections[-1]["card"] != card
                or sections[-1]["surviving"] != kept
            ):
                sections.append(
                    {
                        "card": card,
                        "surviving": kept,
                        "turns": [],
                        "start": positions[branch["id"]][i] - 0.8,
                    }
                )
            section = sections[-1]
            section["turns"].append(number)
            section["end"] = positions[branch["id"]][i] + 0.15
        for section in sections:
            section["flag"] = evidence["status"] == "located" and bool(
                set(evidence["turns"]) & set(section["turns"])
            )
        parent = by_id.get(branch["parent"])
        resolved = bool(parent and branch["anchor"] in parent["anchors"])
        rows.append(
            {
                "id": branch["id"],
                "number": index + 1,
                "parent": branch["parent"] if resolved else None,
                "unresolved": bool(branch["parent"] and not resolved),
                "start": starts[branch["id"]],
                "end": max(positions[branch["id"]] or [starts[branch["id"]] + 0.5]),
                "rollback": branch["rollback"],
                "sections": sections,
                "surviving": branch["id"] in limits,
                "surviving_end": positions[branch["id"]][limits[branch["id"]]]
                if branch["id"] in limits
                else None,
                "target_calls": sum(len(s["turns"]) for s in sections),
            }
        )
    return {
        "status": "ok",
        "branches": rows,
        "evidence": evidence,
        "final_path_status": final.get("status", "unavailable"),
        "target_calls": sum(b["target_calls"] for b in rows),
        "unresolved": sum(b["unresolved"] for b in rows),
    }


def _identity(payload: Any) -> Any:
    return hashlib.sha256(
        json.dumps(
            [
                (t["role"], payload["messages"][t["output"]].get("id"))
                for t in payload["turns"]
            ]
        ).encode()
    ).hexdigest()
