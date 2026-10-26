"""Project the final saved target context onto Inspect's exact conversation tree.

The endpoint comes from store.target_messages, not the last model event. Walking
its ancestors retains pre-compaction history and cuts every parent at its fork
anchor. No model calls or new classifications are involved.
"""

from collections import Counter, defaultdict
from typing import Any


def final_path(tree: dict[str, Any], messages: Any) -> dict[str, Any]:
    """Resolve the saved endpoint through exact ancestors, or explain absence."""

    def unavailable(reason: Any) -> Any:
        return {"status": "unavailable", "reason": reason, "turns": [], "limits": {}}

    if not isinstance(messages, list):
        return unavailable(
            "The final target context was not retained; recorded call order is shown."
        )
    locations = defaultdict(list)
    branches = {b["id"]: b for b in tree["branches"]}
    for branch in branches.values():
        for index, point in enumerate(branch["points"]):
            if point.get("message_id"):
                locations[point["message_id"]].append((branch["id"], index))
    # Context-only endpoints matter: the run can end immediately after rollback.
    endpoint = next(
        (locations[m["id"]] for m in reversed(messages) if m.get("id") in locations), []
    )
    if len(endpoint) != 1:
        return unavailable(
            "The final target endpoint cannot be uniquely located in "
            "the saved tree; recorded call order is shown."
        )
    bid, index = endpoint[0]
    limits, chain = {}, []
    while True:
        if bid in limits:
            return unavailable("The saved target ancestry contains a cycle.")
        limits[bid] = index
        branch = branches[bid]
        chain.append(branch)
        if branch["parent"] is None:
            break
        parent = branches.get(branch["parent"])
        if parent is None or branch["anchor"] not in parent["anchors"]:
            return unavailable(
                "A final-path parent anchor is missing; recorded call order is shown."
            )
        bid, index = parent["id"], parent["anchors"][branch["anchor"]]
    points = [p for b in reversed(chain) for p in b["points"][: limits[b["id"]] + 1]]
    surviving_ids = {p.get("message_id") for p in points if p["kind"] == "target"}
    stored_ids = {m.get("id") for m in messages}
    generated_ids = {
        p.get("message_id")
        for b in branches.values()
        for p in b["points"]
        if p["kind"] == "target"
    }
    if (stored_ids & generated_ids) - surviving_ids:
        return unavailable(
            "The final context disagrees with the saved tree ancestry; "
            "recorded call order is shown."
        )
    return {
        "status": "ok",
        "turns": [p["turn"] for p in points if p["kind"] == "target"],
        "limits": limits,
        "endpoint": endpoint[0],
    }


def presentation(
    payload: dict[str, Any], groups: list[dict[str, Any]], tree: dict[str, Any]
) -> dict[str, Any]:
    """Partition every retained call once; attach alternatives at divergence.

    Auditor calls are contextual evidence, not part of the target path. Each is
    stored once beside the next generated target call in recorded order (or in
    a closing auditor group). Target cards link to their preceding auditor call
    even when a single auditor output orchestrated several different branches.
    """
    final = tree.get("final_path", {})
    if tree["status"] != "ok" or final.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": final.get(
                "reason", tree.get("reason", "Final path unavailable.")
            ),
        }
    branches = {b["id"]: b for b in tree["branches"]}
    limits = final["limits"]
    kept = final["turns"]
    kept_set = set(kept)
    phases = {n: g["card"] for g in groups for n in g["turns"]}
    branch_numbers = {b["id"]: i for i, b in enumerate(tree["branches"], 1)}
    prefix_counts = {}
    count = 0
    # limits were inserted endpoint-to-root; traverse in conversation order.
    for bid in reversed(limits):
        for i, point in enumerate(branches[bid]["points"][: limits[bid] + 1]):
            count += point["kind"] == "target"
            prefix_counts[(bid, i)] = count
    alternatives = defaultdict(list)
    for branch in tree["branches"]:
        turns = [
            p["turn"]
            for p in branch["points"]
            if p["kind"] == "target" and p["turn"] not in kept_set
        ]
        if not turns:
            continue
        bid = branch["id"]
        attachment = None
        if bid in limits:
            attachment = (bid, limits[bid])
        else:
            current, seen = branch, set()
            while current["id"] not in limits:
                if current["id"] in seen:
                    return {
                        "status": "unavailable",
                        "reason": "A discarded branch has cyclic ancestry.",
                    }
                seen.add(current["id"])
                parent = branches.get(current["parent"])
                if parent is None or current["anchor"] not in parent["anchors"]:
                    return {
                        "status": "unavailable",
                        "reason": "A discarded branch has no exact "
                        "divergence anchor; recorded call order is shown.",
                    }
                attachment = (
                    parent["id"],
                    min(
                        parent["anchors"][current["anchor"]],
                        limits.get(parent["id"], float("inf")),
                    ),
                )
                current = parent
        if attachment is None:
            raise ValueError("Discarded branch has no divergence point")
        alternatives[prefix_counts[attachment]].append(
            {
                "branch": branch_numbers[bid],
                "turns": turns,
                "parent": branch_numbers.get(branch["parent"]),
                "rollback": branch["rollback"],
            }
        )

    result_groups: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    target_owners: dict[int, dict[str, Any]] = {}

    def make_groups(numbers: Any, surviving: Any, branch: Any = None) -> Any:
        made: list[dict[str, Any]] = []
        for number in numbers:
            phase = phases[number]
            if not made or made[-1]["phase"] != phase:
                group = {
                    "card": f"target-phase-{len(result_groups)}",
                    "phase": phase,
                    "surviving": surviving,
                    "branch": branch,
                    "target_turns": [],
                    "auditor_turns": [],
                    "turns": [],
                }
                result_groups.append(group)
                made.append(group)
            made[-1]["target_turns"].append(number)
            target_owners[number] = made[-1]
        return [g["card"] for g in made]

    start = 0
    for boundary in sorted(alternatives):
        entries.extend(
            {"kind": "phase", "card": card}
            for card in make_groups(kept[start:boundary], True)
        )
        attempts = alternatives[boundary]
        cards = [
            card
            for a in attempts
            for card in make_groups(a["turns"], False, a["branch"])
        ]
        turns = [n for a in attempts for n in a["turns"]]
        entries.append(
            {
                "kind": "alternatives",
                "after_turn": kept[boundary - 1] if boundary else None,
                "cards": cards,
                "turns": turns,
                "branches": attempts,
                "attempts": len(
                    {payload["turns"][n - 1].get("attempt") for n in turns}
                ),
            }
        )
        start = boundary
    entries.extend(
        {"kind": "phase", "card": card} for card in make_groups(kept[start:], True)
    )
    # A closing audit may contain no target response, but must remain readable.
    waiting, preceding, auditor_context = [], None, {}
    for turn in payload["turns"]:
        n = turn["number"]
        if turn["role"] == "auditor":
            waiting.append(n)
            preceding = n
        else:
            auditor_context[str(n)] = preceding
            target_owners[n]["auditor_turns"].extend(waiting)
            waiting = []
    if waiting:
        group = {
            "card": f"target-phase-{len(result_groups)}",
            "phase": phases[waiting[0]],
            "surviving": None,
            "branch": None,
            "target_turns": [],
            "auditor_turns": waiting,
            "turns": [],
        }
        result_groups.append(group)
        entries.append({"kind": "phase", "card": group["card"]})
    for group in result_groups:
        group["turns"] = sorted(group["target_turns"] + group["auditor_turns"])
        group["roles"] = dict(
            Counter(payload["turns"][n - 1]["role"] for n in group["turns"])
        )
    all_calls = [n for g in result_groups for n in g["turns"]]
    if sorted(all_calls) != [t["number"] for t in payload["turns"]]:
        raise ValueError(
            "Target path projection must retain every model call exactly once"
        )
    return {
        "status": "ok",
        "groups": result_groups,
        "entries": entries,
        "surviving_turns": kept,
        "discarded_turns": sorted(set(target_owners) - kept_set),
        "auditor_context": auditor_context,
    }
