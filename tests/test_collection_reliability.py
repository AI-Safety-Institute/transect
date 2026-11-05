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


def _drop_rolls_of_turn_5(results):
    """Leave turn 5 of the first epoch with a single ballot."""
    votes = results.phase_turn_votes
    results.phase_turn_votes = votes[
        ~((votes.transcript_id == "run-0") & (votes.turn == 5) & (votes.roll > 0))
    ]


def _attribute_turn_9(results):
    """Mark turn 9 of both epochs as never shown to the judge."""
    results.phase_turns.loc[results.phase_turns.turn == 9, "basis"] = "attributed"


def _refuse_one_span_ballot(results):
    """Record one sub-agent ballot as a refusal."""
    results.subagent_votes.loc[results.subagent_votes.index[0], "status"] = "refusal"


def _solo(results):
    """Record the phase judge as a single model with one roll."""
    results.phases[["judge_regime", "n_models", "k_rolls"]] = ["solo", 1, 1]


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


def _forget_setup(results):
    """Keep the judged units but lose the judge identity, as an errored
    judge call records it."""
    results.phases["judge_regime"] = None


def _duplicate(frame):
    """A mutation appending a copy of the frame's first row."""

    def mutate(results):
        current = getattr(results, frame)
        setattr(
            results, frame, pd.concat([current, current.iloc[:1]], ignore_index=True)
        )

    return mutate


def _mixed_setup(results):
    """Give one phase of the second epoch a different roll count."""
    results.phases.loc[results.phases.index[-1], "k_rolls"] = 9


def _row(results, name):
    """The first collection row of a surface."""
    return next(r for r in collection_reliability(results) if r["name"] == name)


def _numbers(row):
    """The counts a row shows, None for the ones a non-voting row lacks."""
    return {
        "scope": row["scope"],
        "seen": row["seen"],
        "mean_n": row["mean"].n if "mean" in row else None,
        "cohort_n": row["cohort"].n if "cohort" in row else None,
        "ballots": row.get("ballots"),
        "expected": row.get("expected"),
    }


@pytest.mark.parametrize(
    ("surface", "mutate", "expected"),
    [
        ("Phases", lambda r: None, (2, 20, 10, 60, 60, 60)),
        ("Phases", _drop_rolls_of_turn_5, (2, 20, 10, 57, 58, 60)),
        ("Phases", _attribute_turn_9, (2, 18, 8, 54, 54, 54)),
        ("Phases", _solo, (2, 20, None, None, None, None)),
        ("Sub-agents", lambda r: None, (2, 6, 6, 18, 18, 18)),
        ("Sub-agents", _refuse_one_span_ballot, (2, 6, 6, 17, 17, 18)),
    ],
    ids=[
        "pooled",
        "lone ballot",
        "attributed turn",
        "single judge",
        "spans",
        "refused ballot",
    ],
)
def test_denominators_follow_the_frames_and_the_retained_ballots(
    same_setup, surface, mutate, expected
):
    """Units pool with transcript-scoped keys, the mean reads the frames'
    own agreement, only usable ballots on the units judging attempted
    count, and a single judge has no agreement numbers at all."""
    mutate(same_setup)
    scope, seen, mean_n, cohort_n, ballots, expected_ballots = expected
    assert _numbers(_row(same_setup, surface)) == {
        "scope": scope,
        "seen": seen,
        "mean_n": mean_n,
        "cohort_n": cohort_n,
        "ballots": ballots,
        "expected": expected_ballots,
    }


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
    ("mutate", "rows", "scope", "reason"),
    [
        (lambda r: _add_label(r, "run-0", "extra_label"), 2, 1, None),
        (lambda r: _reverse_labels(r, "run-1"), 1, 2, None),
        (_unjudge, 1, 2, "no judged units"),
        (_forget_setup, 1, 2, "no judge setup recorded"),
    ],
    ids=[
        "a vocabulary difference",
        "the same vocabulary reordered",
        "no judgements",
        "no setup recorded",
    ],
)
def test_vocabulary_and_setup_absence_decide_the_grouping(
    same_setup, mutate, rows, scope, reason
):
    """A different label vocabulary splits the epochs, the same labels in
    another order do not, and epochs without judgements or without a
    recorded setup pool into one row that states the absence."""
    mutate(same_setup)
    found = [r for r in collection_reliability(same_setup) if r["name"] == "Phases"]
    assert len(found) == rows
    assert (found[0].get("reason"), found[0]["scope"]) == (reason, scope)


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
        (_duplicate("phase_turn_votes"), "duplicate Phases original ballots"),
        (_duplicate("phase_turns"), "duplicate Phases judgement units"),
        (_mixed_setup, "several judge setups"),
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
