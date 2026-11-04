"""Collection reliability pools the per-epoch frames per judge setup."""

import dataclasses

import pandas as pd
import pytest
from test_phase_review_units import reviewed_frame

from transect.report.collection import collection_reliability


@pytest.fixture
def same_setup(epoch_runs):
    """Two epochs from the same store, so they share one judge setup and
    reuse unit numbers."""
    frames = {}
    for name, frame in epoch_runs.frames().items():
        first = frame[frame.transcript_id == "run-0"].copy()
        second = first.copy()
        second["transcript_id"] = "run-1"
        second["epoch"] = 2
        frames[name] = pd.concat([first, second], ignore_index=True)
    return dataclasses.replace(epoch_runs, **frames)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda r: None,
            {
                "scope": 2,
                "seen": 20,
                "mean_n": 10,
                "cohort_n": 60,
                "ballots": 60,
                "expected": 60,
            },
        ),
        (
            lambda r: _drop_rolls_of_turn_5(r),
            {
                "scope": 2,
                "seen": 20,
                "mean_n": 10,
                "cohort_n": 57,
                "ballots": 58,
                "expected": 60,
            },
        ),
        (
            lambda r: _attribute_turn_9(r),
            {
                "scope": 2,
                "seen": 18,
                "mean_n": 8,
                "cohort_n": 54,
                "ballots": 54,
                "expected": 54,
            },
        ),
    ],
    ids=["pooled", "lone ballot", "attributed turn"],
)
def test_denominators_follow_the_frames_and_the_retained_ballots(
    same_setup, mutate, expected
):
    """Units pool with transcript-scoped keys, the mean reads the frames'
    own agreement, and only ballots on units the judge saw count."""
    mutate(same_setup)
    row = _row(same_setup, "Phases")
    assert {
        "scope": row["scope"],
        "seen": row["seen"],
        "mean_n": row["mean"].n,
        "cohort_n": row["cohort"].n,
        "ballots": row["ballots"],
        "expected": row["expected"],
    } == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Phases", [("solo + k-rolls", 30, 30), ("cohort of", 30, 30)]),
        ("Sub-agents", [("solo + k-rolls", 9, 9), ("cohort of", 9, 9)]),
    ],
    ids=["phases", "sub-agents"],
)
def test_each_setup_pools_its_own_epochs(epoch_runs, name, expected):
    """A k-roll epoch and a cohort epoch get separate rows, each with the
    ballot count its own setup expects, on every built-in surface."""
    rows = [r for r in collection_reliability(epoch_runs) if r["name"] == name]
    assert [(r["scope"], r["ballots"], r["expected"]) for r in rows] == [
        (1, ballots, expected) for _, ballots, expected in expected
    ]
    assert all(
        setup in r["setup"] for (setup, _, _), r in zip(expected, rows, strict=True)
    )


@pytest.mark.parametrize(
    ("mutate", "rows"),
    [
        (lambda r: _add_label(r, "run-0", "extra_label"), 2),
        (lambda r: _reverse_labels(r, "run-1"), 1),
        (lambda r: _unjudge(r), 1),
    ],
    ids=[
        "a vocabulary difference",
        "the same vocabulary reordered",
        "no setup recorded",
    ],
)
def test_vocabulary_and_setup_absence_decide_the_grouping(same_setup, mutate, rows):
    """A different label vocabulary splits the epochs, the same labels in
    another order do not, and epochs without a recorded setup pool into
    one row that states the absence."""
    mutate(same_setup)
    found = [r for r in collection_reliability(same_setup) if r["name"] == "Phases"]
    assert len(found) == rows
    if not found[0]["available"]:
        assert found[0]["reason"] == "no judged units" and found[0]["scope"] == 2


def test_verifier_counts_keep_the_original_review_units(same_setup):
    """Verifier denominators count the retained review units, failed
    verdicts apart, across the pooled epochs."""
    same_setup.phases = pd.concat(
        [
            reviewed_frame(transcript_id=tid, missing=(0, 2))[0]
            for tid in ("run-0", "run-1")
        ],
        ignore_index=True,
    )
    verifier = _row(same_setup, "Phases")["verifier"]
    assert (verifier["selected"], verifier["completed"]) == (6, 2)
    assert verifier["random"].of == 2


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda r: _duplicate(r, "phase_turn_votes"),
            "duplicate Phases original ballots",
        ),
        (lambda r: _duplicate(r, "phase_turns"), "duplicate Phases judgement units"),
        (lambda r: _mixed_setup(r), "several judge setups"),
    ],
    ids=["duplicate ballot", "duplicate unit", "mixed setup in one epoch"],
)
def test_a_frame_breaking_its_contract_yields_a_stated_row(same_setup, mutate, reason):
    """A duplicate unit or ballot, or several setups within one epoch,
    turn the surface into a row that names the breach instead of a
    pooled number."""
    mutate(same_setup)
    row = _row(same_setup, "Phases")
    assert row["available"] is False and reason in row["reason"]


def _row(results, name):
    """The first collection row of a surface."""
    return next(r for r in collection_reliability(results) if r["name"] == name)


def _drop_rolls_of_turn_5(results):
    """Leave turn 5 of the first epoch with a single ballot."""
    votes = results.phase_turn_votes
    results.phase_turn_votes = votes[
        ~((votes.transcript_id == "run-0") & (votes.turn == 5) & (votes.roll > 0))
    ]


def _attribute_turn_9(results):
    """Mark turn 9 of both epochs as never shown to the judge."""
    results.phase_turns.loc[results.phase_turns.turn == 9, "basis"] = "attributed"


def _add_label(results, transcript_id, label):
    """Record one more phase label in one epoch's vocabulary."""
    definitions = results.label_definitions
    extra = (
        definitions[
            (definitions.transcript_id == transcript_id)
            & (definitions.surface == "phases")
        ]
        .head(1)
        .assign(label=label)
    )
    results.label_definitions = pd.concat([definitions, extra], ignore_index=True)


def _reverse_labels(results, transcript_id):
    """Record one epoch's phase vocabulary in reverse order."""
    definitions = results.label_definitions
    own = (definitions.transcript_id == transcript_id) & (
        definitions.surface == "phases"
    )
    results.label_definitions = pd.concat(
        [definitions[~own], definitions[own].iloc[::-1]], ignore_index=True
    )


def _unjudge(results):
    """Strip every phase judgement, as a scan whose judge never answered."""
    results.phases = results.phases.iloc[:0]
    results.phase_turns = results.phase_turns.iloc[:0]
    results.phase_turn_votes = results.phase_turn_votes.iloc[:0]


def _duplicate(results, frame):
    """Append a copy of the frame's first row."""
    current = getattr(results, frame)
    setattr(results, frame, pd.concat([current, current.iloc[:1]], ignore_index=True))


def _mixed_setup(results):
    """Give one phase of the second epoch a different roll count."""
    results.phases.loc[results.phases.index[-1], "k_rolls"] = 9
