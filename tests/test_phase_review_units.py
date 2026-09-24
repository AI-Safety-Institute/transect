"""Verifier populations retain original phase units through display merging."""

import asyncio

import pandas as pd
import pytest
from helpers import PHASES_SPEC, model_turn, run_scan, scripted_judge, seg, seg_answer
from inspect_ai.model import get_model

import transect.scanners.phases_verify as phase_verifier
from transect import reliability
from transect.frames.phases import phases_df
from transect.scanners.cohort import judge_setup
from transect.scanners.phases import decision_phases
from transect.scanners.phases_cohort import ConsensusJudgement
from transect.scanners.phases_common import Digest, stitch_phases
from transect.scanners.phases_verify import (
    VerifierAudit,
    _apply_verdicts,
    _ReviewAttempt,
    _Verdict,
)
from transect.spec import Spec


def reviewed_frame(*, reasons=None, missing=(), transcript_id="t1"):
    """Run the real merge and frame projection over three original reviews."""
    rows = [
        ConsensusJudgement(turn=i, phase=label, confidence=0.9, basis="judged")
        for i, label in enumerate(["A", "B", "A"])
    ]
    selected = reasons or {i: "random_sample" for i in range(3)}
    attempts = {
        i: _ReviewAttempt(trigger=trigger)
        if i in missing
        else _ReviewAttempt(
            trigger=trigger,
            verdict=_Verdict(
                phase_index=i, phase="A", confidence=0.9, explanation="fixture"
            ),
            status="ok",
        )
        for i, trigger in selected.items()
    }
    phases, audit = _apply_verdicts(
        rows,
        stitch_phases(rows),
        attempts,
        VerifierAudit(n_random_sample=list(selected.values()).count("random_sample")),
        "m",
    )
    value = {
        "phases": [phase.model_dump() for phase in phases],
        "phase_names": ["A", "B"],
        "verifier": audit.model_dump(),
        "judge_models": ["m"],
        "judge": judge_setup(["m"], 1, verifier_armed=True, verifier_model="m"),
    }
    return phases_df(
        pd.DataFrame([{"transcript_id": transcript_id, "value": value}])
    ), value


def scalar_frame(**columns):
    """A minimal scalar (sub-agent/custom) reviewed row."""
    return pd.DataFrame(
        {
            "verifier_selected": [True],
            "overturned": [True],
            "original_label": ["A"],
            "verifier_trigger": ["random_sample"],
            **columns,
        }
    )


def test_merged_display_phase_retains_three_original_review_units():
    """One resulting A phase retains two unchanged A reviews and one B relabel."""
    frame, value = reviewed_frame()
    assert len(frame) == 1
    assert len(value["phases"][0]["verifier_reviews"]) == 3
    measured = reliability.relabel_rate(frame)
    assert (measured.overall.count, measured.overall.of) == (1, 3)
    assert {
        label: (rate.count, rate.of) for label, rate in measured.by_label.items()
    } == {
        "A": (0, 2),
        "B": (1, 1),
    }
    assert reliability.spot_check_overturns(frame).of == 3


def test_original_triggers_survive_merge():
    """A doubt-triggered relabel cannot become a random-sample overturn."""
    frame, _ = reviewed_frame(
        reasons={0: "random_sample", 1: "low_confidence", 2: "random_sample"}
    )
    measured = reliability.spot_check_overturns(frame)
    assert (measured.count, measured.of) == (0, 2)


def test_label_statistics_read_original_review_population():
    """Per-label review statistics keep labels merged out of display phases."""
    frame, _ = reviewed_frame()
    decided = pd.DataFrame(
        {
            "transcript_id": ["t1"],
            "turn": [0],
            "phase": ["A"],
            "judge_agreement": [None],
            "confidence": [0.9],
            "label_source": ["verifier"],
        }
    )
    votes = pd.DataFrame(columns=["transcript_id", "turn", "phase"])
    stats = reliability.label_stats(decided, votes, frame, "turn", "phase")
    assert {
        stat.label: (stat.relabelled.count, stat.relabelled.of) for stat in stats
    } == {
        "A": (0, 2),
        "B": (1, 1),
    }


def test_selected_missing_verdicts_are_recorded_without_diluting_rate():
    """Selected phases without answers remain visible outside completed denominators."""
    frame, value = reviewed_frame(missing=(0, 2))
    records = value["phases"][0]["verifier_reviews"]
    assert [record["review"]["status"] for record in records] == [
        "no_answer",
        "ok",
        "no_answer",
    ]
    units = reliability.review_units(frame)
    assert len(units) == 3
    assert int(units.verifier_completed.sum()) == 1
    measured = reliability.relabel_rate(frame)
    assert (measured.overall.count, measured.overall.of) == (1, 1)


def test_review_identity_is_scoped_to_transcript_and_deduplicated():
    """Repeated display associations count once while distinct transcripts add."""
    first, _ = reviewed_frame(transcript_id="t1")
    second, _ = reviewed_frame(transcript_id="t2")
    duplicate = pd.concat([first, first], ignore_index=True)
    assert reliability.relabel_rate(duplicate).overall.of == 3
    combined = pd.concat([first, second], ignore_index=True)
    assert reliability.relabel_rate(combined).overall.of == 6


def test_missing_review_list_contributes_no_units():
    """A phase row whose review list is missing yields no review population."""
    frame, _ = reviewed_frame()
    frame["verifier_reviews"] = None
    assert reliability.relabel_rate(frame).overall.of == 0
    assert not len(reliability.review_units(frame))


@pytest.mark.parametrize("failure", ["error", "refusal", "no_answer"])
def test_explicit_scalar_failures_do_not_dilute_completed_relabel_rate(failure):
    """Failed attempts do not dilute one usable scalar overturn's denominator."""
    frame = pd.DataFrame(
        {
            "verifier_selected": [True] * 10,
            "overturned": [True] + [False] * 9,
            "original_label": ["A"] * 10,
            "verifier_trigger": ["random_sample"] * 10,
            "verifier_status": ["ok"] + [failure] * 9,
            "verifier_label": ["B"] + [None] * 9,
        }
    )
    measured = reliability.relabel_rate(frame).overall
    assert (measured.count, measured.of) == (1, 1)
    assert reliability.spot_check_overturns(frame).of == 1


@pytest.mark.parametrize(
    "status, label, expected",
    [(None, "B", 1), (None, None, 0), (None, "", 0), ("ok", None, 0)],
)
def test_scalar_completion_requires_a_usable_recorded_verdict(status, label, expected):
    """Without an explicit completion column, the frames' one rule applies."""
    frame = scalar_frame(verifier_status=[status], verifier_label=[label])
    assert reliability.relabel_rate(frame).overall.of == expected


@pytest.mark.parametrize("outcome", ["refusal", "no_answer"])
def test_verifier_chunk_outcome_reaches_original_unit(outcome, monkeypatch):
    """Recorded missing verdicts retain the actual chunk outcome without guessing."""

    async def missing_verdict(*args, **kwargs):
        return None, outcome

    monkeypatch.setattr(phase_verifier, "call_judge", missing_verdict)
    rows = [ConsensusJudgement(turn=0, phase="A", confidence=0.4, basis="judged")]
    phases, audit = asyncio.run(
        phase_verifier.verify_phases(
            get_model("mockllm/model"),
            Spec.model_validate({"phases": ["A", "B"]}),
            task_prompt="",
            digests=[Digest(turn=0, text="fixture")],
            digest_judgements=rows,
            phases=stitch_phases(rows),
            phase_names=["A", "B"],
            chunk=5,
            cache=False,
            sample=0,
        )
    )
    assert audit.n_no_verdict == 1
    assert phases[0].verifier is None
    assert phases[0].verifier_reviews[0].review.status == outcome


def test_weak_differing_verdict_is_completed_without_applied_relabel():
    """A below-threshold differing verdict counts as completed but not overturned."""
    rows = [ConsensusJudgement(turn=0, phase="A", confidence=0.4, basis="judged")]
    phases, audit = _apply_verdicts(
        rows,
        stitch_phases(rows),
        {
            0: _ReviewAttempt(
                trigger="low_confidence",
                verdict=_Verdict(
                    phase_index=0, phase="B", confidence=0.5, explanation="weak"
                ),
                status="ok",
            )
        },
        VerifierAudit(n_low_confidence=1),
        "m",
    )
    record = phases[0].verifier_reviews[0].review
    assert record.status == "ok" and record.overturned is False
    assert audit.n_weak_relabel == 1 and audit.n_relabelled == 0


@pytest.mark.parametrize("verify", [False, True])
def test_scanner_stamps_known_empty_reviews_when_none_selected(verify):
    """Fresh output records an empty list for off or untriggered verification."""
    value = run_scan(
        decision_phases(
            PHASES_SPEC,
            scripted_judge(seg_answer(seg(0, 0, "setup"))),
            verify=verify,
            verify_sample=0,
            narrate=False,
        ),
        [model_turn("fixture")],
    ).value
    assert value["phases"][0]["verifier_reviews"] == []


def test_review_ledger_survives_actual_parquet_round_trip(tmp_path):
    """Arrow's ndarray representation retains the original three-review population."""
    frame, _ = reviewed_frame()
    path = tmp_path / "phases.parquet"
    frame.to_parquet(path)
    restored = pd.read_parquet(path)
    assert len(reliability.review_units(restored)) == 3
    rate = reliability.relabel_rate(restored).overall
    assert (rate.count, rate.of) == (1, 3)
