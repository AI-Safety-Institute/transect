"""Exact Petri ancestry preserves discarded actions and pre-compaction history."""

import unittest

from transect.report.petri.path import final_path, presentation
from transect.report.petri.tree import build_diagram, extract_tree


def fixture():
    messages = [{"id": f"m{i}", "text": "", "tools": []} for i in range(1, 6)]
    messages[2]["tools"] = [
        {
            "id": "rollback",
            "name": "rollback_conversation",
            "arguments": {"message_id": "M1"},
            "result": "Rolled back to message_id M1.",
        }
    ]
    messages[3]["tools"] = [
        {"name": "bash", "arguments": {"command": "recorded command"}}
    ]
    payload = {
        "sample_id": "sample",
        "epoch": 1,
        "messages": messages,
        "turns": [
            {
                "number": i,
                "role": "auditor" if i in (1, 3) else "target",
                "output": i - 1,
            }
            for i in range(1, 6)
        ],
    }

    def model(i):
        return {
            "uuid": f"e{i}",
            "event": "model",
            "role": payload["turns"][i - 1]["role"],
            "output": {"choices": [{"message": {"id": f"m{i}"}}]},
        }

    def anchor(uuid, mid):
        return {"uuid": uuid, "event": "anchor", "anchor_id": mid}

    def node(bid, refs, branches=(), parent_anchor=None):
        return {
            "type": "span",
            "id": bid,
            "name": bid,
            "span_type": "branch",
            "content": [{"type": "event", "event": ref} for ref in refs],
            "branches": list(branches),
            "branched_from": parent_anchor,
        }

    events = [
        model(1),
        anchor("a0", "setup"),
        model(2),
        anchor("a2", "m2"),
        model(3),
        {
            "uuid": "rb",
            "event": "tool",
            "function": "rollback_conversation",
            "id": "rollback",
        },
        {
            "uuid": "b2-start",
            "event": "branch",
            "span_id": "b2",
            "from_anchor": "setup",
        },
        model(4),
        anchor("a4", "m4"),
        {"uuid": "b3-start", "event": "branch", "span_id": "b3", "from_anchor": "m2"},
        model(5),
        {"uuid": "b4-start", "event": "branch", "span_id": "b4", "from_anchor": "m4"},
    ]
    b4 = node("b4", ["b4-start"], parent_anchor="m4")
    b2 = node("b2", ["b2-start", "e4", "a4"], [b4], "setup")
    b3 = node("b3", ["b3-start", "e5"], parent_anchor="m2")
    root = node("b1", ["a0", "e2", "a2"], [b2, b3])
    sample = {
        "id": "sample",
        "epoch": 1,
        "events": events,
        "timelines": [{"name": "target", "root": root}],
    }
    groups = [
        {"card": "phase-0-0", "turns": [1, 2]},
        {"card": "phase-0-1", "turns": [3, 4, 5]},
    ]
    assessment = {"status": "located", "turns": [4], "quote": "recorded command"}
    return sample, payload, groups, assessment


class TargetPathTests(unittest.TestCase):
    def prepare(self, *ids):
        sample, payload, groups, assessment = fixture()
        sample["store"] = {"target_messages": [{"id": mid} for mid in ids]}
        for turn in payload["turns"]:
            turn["attempt"] = turn["number"]
        tree = extract_tree(sample, payload)
        return payload, groups, assessment, tree

    def test_final_context_overrides_last_model_and_empty_branch(self):
        payload, groups, _, tree = self.prepare("setup", "m4")
        # m5 was generated later, but the final context returned to m4.
        self.assertEqual(tree["final_path"]["turns"], [4])
        path = presentation(payload, groups, tree)
        self.assertEqual(path["surviving_turns"], [4])
        self.assertEqual(path["discarded_turns"], [2, 5])
        alternative = next(e for e in path["entries"] if e["kind"] == "alternatives")
        self.assertIsNone(alternative["after_turn"])
        self.assertEqual(alternative["turns"], [2, 5])

    def test_final_rollback_can_remove_every_target_call(self):
        payload, groups, _, tree = self.prepare("setup")
        path = presentation(payload, groups, tree)
        self.assertEqual(path["surviving_turns"], [])
        self.assertEqual(path["discarded_turns"], [2, 4, 5])

    def test_compacted_final_context_retains_ancestor_history(self):
        payload, groups, _, tree = self.prepare("new-summary", "m5")
        self.assertEqual(tree["final_path"]["turns"], [2, 5])
        path = presentation(payload, groups, tree)
        self.assertEqual(path["surviving_turns"], [2, 5])
        self.assertEqual(path["discarded_turns"], [4])

    def test_every_call_once_and_auditor_context_can_cross_paths(self):
        payload, groups, _, tree = self.prepare("m2", "m5")
        path = presentation(payload, groups, tree)
        self.assertEqual(
            sorted(n for g in path["groups"] for n in g["turns"]), [1, 2, 3, 4, 5]
        )
        self.assertEqual(path["auditor_context"], {"2": 1, "4": 3, "5": 3})
        owner = next(g for g in path["groups"] if 3 in g["auditor_turns"])
        self.assertFalse(owner["surviving"])
        self.assertEqual(owner["target_turns"], [4])
        # Original phase 1 crosses a discarded and surviving continuation.
        self.assertEqual(
            [g["surviving"] for g in path["groups"] if g["phase"] == "phase-0-1"],
            [False, True],
        )

    def test_discarded_quote_keeps_its_flag(self):
        payload, groups, assessment, tree = self.prepare("m2", "m5")
        diagram = build_diagram(payload, groups, tree, assessment)
        flagged = [s for b in diagram["branches"] for s in b["sections"] if s["flag"]]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["turns"], [4])
        self.assertFalse(flagged[0]["surviving"])

    def test_final_path_missing_or_conflicting_context_is_explicit(self):
        for ids in [("unknown",), ("m4", "m5")]:
            with self.subTest(ids=ids):
                payload, groups, _, tree = self.prepare(*ids)
                self.assertEqual(tree["final_path"]["status"], "unavailable")
                self.assertEqual(
                    presentation(payload, groups, tree)["status"], "unavailable"
                )
        _, _, _, tree = self.prepare("m5")
        self.assertEqual(final_path(tree, None)["status"], "unavailable")

    def test_missing_anchor_never_attaches_by_chronology(self):
        payload, groups, _, tree = self.prepare("m5")
        next(b for b in tree["branches"] if b["id"] == "b2")["anchor"] = "missing"
        self.assertEqual(presentation(payload, groups, tree)["status"], "unavailable")
        next(b for b in tree["branches"] if b["id"] == "b3")["anchor"] = "missing"
        self.assertEqual(final_path(tree, [{"id": "m5"}])["status"], "unavailable")

    def test_parent_prefix_splits_one_phase_at_the_exact_rollback(self):
        payload, groups, _, tree = self.prepare("m5")
        # Reuse the simple fixture's generated calls with a fork after m2.
        root = tree["branches"][0]
        b2 = next(b for b in tree["branches"] if b["id"] == "b2")
        root["points"].extend(b2["points"])
        root["anchors"]["m4"] = len(root["points"]) - 1
        b2["points"], b2["anchors"] = [], {}
        # The empty b4 must now branch from the moved anchor in root.
        next(b for b in tree["branches"] if b["id"] == "b4")["parent"] = root["id"]
        tree["final_path"] = final_path(tree, [{"id": "m5"}])
        groups = [{"card": "phase-0-0", "turns": [1, 2, 3, 4, 5]}]
        path = presentation(payload, groups, tree)
        self.assertEqual(path["surviving_turns"], [2, 5])
        self.assertEqual(
            [e["kind"] for e in path["entries"]], ["phase", "alternatives", "phase"]
        )
        self.assertEqual(path["entries"][1]["after_turn"], 2)
        diagram = build_diagram(payload, groups, tree)
        root_sections = diagram["branches"][0]["sections"]
        self.assertEqual(
            [(s["turns"], s["surviving"]) for s in root_sections],
            [([2], True), ([4], False)],
        )
