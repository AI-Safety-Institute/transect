"""Narration prose stays attached to exactly the ranges the narrator supplied."""

import pandas as pd
import pytest
from helpers import (
    PHASES_SPEC,
    StubTranscript,
    group,
    model_turn,
    narrate_answer,
    narrative,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
)
from inspect_ai.model import ContentReasoning, ContentText

from transect.frames import phase_turn_votes_df, phases_df
from transect.frames.turn_groups import turn_groups_df
from transect.report import sections
from transect.report.excerpts import (
    COMPACTION_NOTE,
    Excerpt,
    _turn_excerpts,
    mark_compaction_turns,
)
from transect.scanners.phases import decision_phases
from transect.scanners.phases_common import StitchedPhase, TurnGroup
from transect.scanners.phases_narrate import validate_turn_groups


def phase(end=9, count=10):
    return StitchedPhase(
        phase="setup",
        turn_start=0,
        turn_end=end,
        n_turns=count,
        confidence=0.9,
        min_confidence=0.9,
        explanation="Synthetic phase",
    )


@pytest.mark.parametrize(
    "ranges",
    [
        [(2, 3)],  # incomplete beginning and end
        [(0, 1), (4, 9)],  # internal gap
        [(0, 5), (4, 9)],  # overlap
        [(0, 9), (0, 9)],  # duplicate
        [(-2, 9)],  # before phase
        [(0, 12)],  # after phase
        [(4, 2)],  # reversed
        [(12, 15)],  # entirely outside
    ],
)
def test_invalid_partition_cannot_relocate_factual_prose(ranges):
    """An invalid partition falls back without extending, clamping or guessing prose."""
    proposed = [
        TurnGroup(turn_start=a, turn_end=b, title="Read configuration", gist="Read it.")
        for a, b in ranges
    ]
    before = [g.model_dump() for g in proposed]
    actual = validate_turn_groups(proposed, phase(), "Setup")
    assert [g.model_dump() for g in actual] == [
        {"turn_start": 0, "turn_end": 9, "title": "Setup", "gist": ""}
    ]
    assert [g.model_dump() for g in proposed] == before


@pytest.mark.parametrize("end,count", [(0, 1), (9, 2), (9, 10)])
def test_valid_partition_preserves_exact_claim_range_even_with_sparse_digests(
    end, count
):
    """A small digest count does not discard a valid range's title and gist."""
    proposed = [
        TurnGroup(turn_start=0, turn_end=end, title="Reviewed notes", gist="Read.")
    ]
    assert validate_turn_groups(proposed, phase(end, count), "Setup") == proposed


def test_valid_out_of_order_partition_only_reorders_groups():
    """Sorting a valid partition preserves both ends and their associated prose."""
    first = TurnGroup(turn_start=0, turn_end=3, title="Read", gist="Read config.")
    second = TurnGroup(turn_start=4, turn_end=9, title="Plan", gist="Planned a test.")
    assert validate_turn_groups([second, first], phase(), "Setup") == [first, second]


def test_reasoning_turns_are_excerpted_and_rendered_as_thinking_lines():
    """A reasoning-bearing turn gets an excerpt even with no visible
    text, and the card renders the reasoning as its own muted line."""
    events = [
        model_turn(
            [
                ContentReasoning(reasoning="weigh the options"),
                ContentText(text="Going with option B."),
            ]
        ),
        model_turn([ContentReasoning(reasoning="silent deliberation")]),
        model_turn(""),
    ]
    excerpts = _turn_excerpts(StubTranscript(events))
    assert sorted(excerpts) == [0, 1]
    assert (excerpts[0].reasoning, excerpts[0].text) == (
        "weigh the options",
        "Going with option B.",
    )
    assert (excerpts[1].reasoning, excerpts[1].text) == ("silent deliberation", "")
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.9)),
        narrate_answer(narrative(0, groups=[group(0, 1)])),
    )
    raw = run_scan(decision_phases(PHASES_SPEC, judge, verify=False), events[:2])
    frame = pd.DataFrame([{"value": raw.value, "transcript_id": "synthetic"}])
    html = str(
        sections.phase_cards(
            phases_df(frame),
            turn_groups_df(frame),
            phase_turn_votes_df(pd.DataFrame()),
            pd.DataFrame(),
            [],
            [],
            {},
            None,
            "cards",
            "phase",
            excerpts=excerpts,
        )
    )
    assert html.count("[thinking]") == 2
    assert "silent deliberation" in html


def test_neutral_fallback_reaches_the_report_without_unsupported_group_prose():
    """Rendered source excerpts remain available without a relocated factual gist."""
    judge = scripted_judge(
        seg_answer(seg(0, 9, "setup", 0.9)),
        narrate_answer(
            narrative(0, groups=[group(2, 3, title="Unsupported range claim")])
        ),
    )
    result = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False),
        [model_turn(f"Reasoning {i}") for i in range(10)],
    )
    raw = pd.DataFrame([{"value": result.value, "transcript_id": "synthetic"}])
    expected = [{"turn_start": 0, "turn_end": 9, "title": "Setup", "gist": ""}]
    assert result.value["phases"][0]["turn_groups"] == expected
    frame = turn_groups_df(raw)
    assert (
        frame[["turn_start", "turn_end", "title", "gist"]].to_dict("records")
        == expected
    )
    excerpts = {
        i: Excerpt(i, "orchestrator", "", f"Source evidence {i}", False)
        for i in (0, 2, 9)
    }
    html = str(
        sections.phase_cards(
            phases_df(raw),
            turn_groups_df(raw),
            phase_turn_votes_df(pd.DataFrame()),
            pd.DataFrame(),
            [],
            [],
            {},
            None,
            "cards",
            "phase",
            excerpts=mark_compaction_turns(excerpts, [2, 5]),
        )
    )
    assert "Unsupported range claim" not in html
    assert "Source evidence 9" in html
    assert "Source evidence 0" in html
    # turn 2 is a summarization call and says so; turn 5 has no excerpt to mark
    assert html.count(COMPACTION_NOTE) == 1
