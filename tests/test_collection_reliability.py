"""Collection measurements preserve sample weights and original judging units."""

import dataclasses

import pandas as pd
import pytest
from test_phase_review_units import reviewed_frame
from test_report_index import runs as runs

from transect.report.collection import collection_reliability


@pytest.fixture
def same_setup(runs):
    """Two runs reuse unit numbers while sharing one recorded judge setup."""
    frames = {}
    for name, frame in runs.frames().items():
        first = frame[frame.transcript_id == "run-0"].copy()
        second = first.copy()
        second["transcript_id"] = "run-1"
        second["sample_id"] = "sample-1"
        frames[name] = pd.concat([first, second], ignore_index=True)
    return dataclasses.replace(runs, **frames)


def test_collection_pools_original_ballots_with_transcript_scoped_units(same_setup):
    """Shared turn numbers across runs remain distinct units in pooled statistics."""
    row = next(r for r in collection_reliability(same_setup) if r["name"] == "Phases")
    assert (row["records"], row["scope"], row["cohort"].n) == (2, 2, 60)
    assert row["paired_units"] == 20
    assert (row["valid"], row["recorded"], row["expected"]) == (60, 60, 60)
    assert row["mean"].n == 10
    assert row["balanced"].samples == 2


def test_single_valid_ballot_does_not_count_as_collection_agreement(same_setup):
    """A lone ballot remains coverage evidence but cannot measure agreement."""
    votes = same_setup.phase_turn_votes
    remove = (votes.transcript_id == "run-0") & (votes.turn == 5) & (votes.roll > 0)
    same_setup.phase_turn_votes = votes[~remove]
    row = next(r for r in collection_reliability(same_setup) if r["name"] == "Phases")
    assert row["mean"].n == 9
    assert row["cohort"].n == 57
    assert row["paired_units"] == 19
    assert (row["valid"], row["expected"]) == (58, 60)


def test_attributed_turns_do_not_enter_collection_ballot_denominators(same_setup):
    """Turns never shown to the judge cannot dilute agreement or ballot coverage."""
    same_setup.phase_turns.loc[same_setup.phase_turns.turn == 9, "basis"] = "attributed"
    row = next(r for r in collection_reliability(same_setup) if r["name"] == "Phases")
    assert (row["units"], row["eligible"], row["expected"]) == (20, 18, 54)
    assert (row["recorded"], row["cohort"].n) == (54, 54)
    assert row["paired_units"] == 18


def test_collection_keeps_judge_setups_separate(runs):
    """K-roll and cohort results remain separate even within the same scanner."""
    rows = [r for r in collection_reliability(runs) if r["name"] == "Phases"]
    assert len(rows) == 2
    assert all(row["records"] == row["scope"] == 1 for row in rows)
    assert "k-rolls" in rows[0]["setup"]
    assert "cohort" in rows[1]["setup"]


def test_collection_retains_original_reviews_and_failed_verdicts(same_setup):
    """Verifier denominators count original review units rather than display phases."""
    frames = [
        reviewed_frame(transcript_id=tid, missing=(0, 2))[0]
        for tid in ("run-0", "run-1")
    ]
    same_setup.phases = pd.concat(frames, ignore_index=True)
    row = next(r for r in collection_reliability(same_setup) if r["name"] == "Phases")
    assert row["verifier"]["selected"] == 6
    assert row["verifier"]["completed"] == 2
    assert row["verifier"]["failed"] == 4
    assert row["verifier"]["random"].of == 2


def test_duplicate_original_ballots_fail_loudly(same_setup):
    """A copied ballot cannot inflate collection agreement or recorded coverage."""
    votes = same_setup.phase_turn_votes
    same_setup.phase_turn_votes = pd.concat([votes, votes.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate Phases original ballots"):
        collection_reliability(same_setup)
