"""Verifier populations retain original phase units through display merging."""

import asyncio
import copy

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
from transect.scanners.phases_verify import VerifierAudit, _apply_verdicts, _Verdict
from transect.spec import Spec


def reviewed_frame(*, reasons=None, missing=(), transcript_id="t1"):
    """Run the real merge and frame projection over three original reviews."""
    rows = [
        ConsensusJudgement(turn=i, phase=label, confidence=0.9, basis="judged")
        for i, label in enumerate(["A", "B", "A"])
    ]
    selected = reasons or {i: "random_sample" for i in range(3)}
    verdicts = {
        i: _Verdict(phase_index=i, phase="A", confidence=0.9, explanation="fixture")
        for i in selected
        if i not in missing
    }
    phases, audit = _apply_verdicts(
        rows,
        stitch_phases(rows),
        verdicts,
        selected,
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
    """Legacy row defaults leave status, label and completion provenance absent."""
    return pd.DataFrame(
        {
            "verifier_reviewed": [True],
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
    units, unavailable = reliability.review_units(frame)
    assert unavailable is None and len(units) == 3
    assert units.verifier_reviewed.all()
    assert int(units.verifier_completed.sum()) == 1
    measured = reliability.relabel_rate(frame)
    assert (measured.overall.count, measured.overall.of) == (1, 1)


def test_historical_missing_units_are_unknown_even_when_mixed_with_new_data():
    """Historical representative reviews cannot provide original-unit denominators."""
    current, value = reviewed_frame()
    old = copy.deepcopy(value)
    for phase in old["phases"]:
        phase.pop("verifier_reviews", None)
    historical = phases_df(pd.DataFrame([{"transcript_id": "old", "value": old}]))
    for frame in (historical, pd.concat([current, historical], ignore_index=True)):
        rate = reliability.relabel_rate(frame).overall
        assert rate.rate is None and rate.unavailable_reason
        assert reliability.spot_check_overturns(frame).unavailable_reason


def test_review_identity_is_scoped_to_transcript_and_deduplicated():
    """Repeated display associations count once while distinct transcripts add."""
    first, _ = reviewed_frame(transcript_id="t1")
    second, _ = reviewed_frame(transcript_id="t2")
    duplicate = pd.concat([first, first], ignore_index=True)
    assert reliability.relabel_rate(duplicate).overall.of == 3
    combined = pd.concat([first, second], ignore_index=True)
    assert reliability.relabel_rate(combined).overall.of == 6


def test_known_empty_reviews_are_distinct_from_historical_unknown():
    """A known empty list states that no original phase reviews were selected."""
    frame, _ = reviewed_frame()
    frame["verifier_reviews"] = [[]]
    measured = reliability.relabel_rate(frame).overall
    assert measured.of == 0 and measured.unavailable_reason is None


@pytest.mark.parametrize("failure", ["error", "refusal", "no_answer"])
def test_explicit_scalar_failures_do_not_dilute_completed_relabel_rate(failure):
    """Failed attempts do not dilute one usable scalar overturn's denominator."""
    frame = pd.DataFrame(
        {
            "verifier_reviewed": [True] * 10,
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


@pytest.mark.parametrize("with_phase_column", [False, True])
def test_legacy_scalar_frames_without_status_keep_their_existing_contract(
    with_phase_column,
):
    """Legacy rows stay scalar even with a custom label column named phase."""
    frame = pd.DataFrame(
        {
            "verifier_reviewed": [True, True],
            "overturned": [True, False],
            "original_label": ["A", "A"],
            "verifier_trigger": ["random_sample"] * 2,
        }
    )
    if with_phase_column:
        frame["phase"] = ["A", "A"]
    measured = reliability.relabel_rate(frame).overall
    assert (measured.count, measured.of) == (1, 2)
    assert measured.unavailable_reason is None


def test_conflicting_duplicate_original_reviews_raise():
    """Conflicting associations cannot silently choose one original review outcome."""
    frame, _ = reviewed_frame()
    other = frame.copy(deep=True)
    other["verifier_reviews"] = [copy.deepcopy(frame.verifier_reviews.iloc[0])]
    other.verifier_reviews.iloc[0][0]["review"]["overturned"] = True
    with pytest.raises(ValueError, match="conflicting"):
        reliability.review_units(pd.concat([frame, other], ignore_index=True))


@pytest.mark.parametrize("label, expected", [("B", 1), (None, 0), ("", 0)])
def test_scalar_missing_status_requires_a_usable_recorded_verdict(label, expected):
    """A legacy null status is usable only when its verdict label survived."""
    frame = scalar_frame(
        verifier_selected=[True], verifier_status=[None], verifier_label=[label]
    )
    assert reliability.relabel_rate(frame).overall.of == expected


def test_explicit_ok_without_label_is_not_a_completed_verdict():
    """An ok status alone cannot create a usable verifier answer."""
    frame = scalar_frame(
        verifier_selected=[True], verifier_status=["ok"], verifier_label=[None]
    )
    units, reason = reliability.review_units(frame)
    assert reason is None and len(units) == 1
    assert not units.verifier_completed.any()
    assert reliability.relabel_rate(frame).overall.of == 0


def test_historical_verifier_off_does_not_imply_lost_review_units():
    """An explicitly unarmed historical verifier has no review population."""
    frame, _ = reviewed_frame()
    frame["verifier_reviews"] = None
    frame["verifier_armed"] = False
    frame["verifier_reviewed"] = False
    rate = reliability.relabel_rate(frame).overall
    assert rate.of == 0 and rate.unavailable_reason is None


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
    assert phases[0].verifier_reviews[0].review["status"] == outcome


def test_weak_differing_verdict_is_completed_without_applied_relabel():
    """A below-threshold differing verdict counts as completed but not overturned."""
    rows = [ConsensusJudgement(turn=0, phase="A", confidence=0.4, basis="judged")]
    phases, audit = _apply_verdicts(
        rows,
        stitch_phases(rows),
        {0: _Verdict(phase_index=0, phase="B", confidence=0.5, explanation="weak")},
        {0: "low_confidence"},
        VerifierAudit(n_low_confidence=1),
        "m",
    )
    record = phases[0].verifier_reviews[0].review
    assert record["status"] == "ok" and record["overturned"] is False
    assert audit.n_weak_relabel == 1 and audit.n_relabelled == 0


def test_unknown_review_population_marks_each_label_rate_unavailable():
    """Label statistics cannot silently narrow a mixed historical population."""
    frame, _ = reviewed_frame()
    frame["verifier_reviews"] = None
    decided = pd.DataFrame(
        {
            "turn": [0],
            "phase": ["A"],
            "judge_agreement": [None],
            "confidence": [0.9],
            "label_source": ["single_judge"],
        }
    )
    votes = pd.DataFrame(columns=["turn", "phase"])
    stats = reliability.label_stats(decided, votes, frame, "turn", "phase")
    assert stats[0].relabelled.unavailable_reason
    assert stats[0].spot_checked.unavailable_reason


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


def test_historical_export_without_review_list_column_is_unavailable():
    """Dropping the new column cannot turn old display phases into review units."""
    frame, _ = reviewed_frame()
    historical = frame.drop(columns=["verifier_reviews", "verifier_completed"])
    rate = reliability.relabel_rate(historical).overall
    assert rate.rate is None and rate.unavailable_reason
    assert reliability.spot_check_overturns(historical).unavailable_reason


def test_review_ledger_survives_actual_parquet_round_trip(tmp_path):
    """Arrow's ndarray representation retains the original three-review population."""
    frame, _ = reviewed_frame()
    path = tmp_path / "phases.parquet"
    frame.to_parquet(path)
    restored = pd.read_parquet(path)
    units, unavailable = reliability.review_units(restored)
    assert unavailable is None and len(units) == 3
    rate = reliability.relabel_rate(restored).overall
    assert (rate.count, rate.of) == (1, 3)


def test_mixed_scalar_contracts_do_not_silently_exclude_legacy_rows():
    """Concatenation cannot erase a legacy overturn through newly nullable columns."""
    legacy = scalar_frame()
    modern = scalar_frame(
        overturned=[False], verifier_status=["ok"], verifier_label=["A"]
    )
    assert reliability.relabel_rate(legacy).overall.count == 1
    combined = pd.concat([legacy, modern], ignore_index=True)
    units, unavailable = reliability.review_units(combined)
    assert unavailable and len(units) == 2
    assert pd.isna(units.verifier_completed.iloc[0])
    rate = reliability.relabel_rate(combined).overall
    assert rate.rate is None and rate.unavailable_reason


@pytest.mark.parametrize("completion", [False, pd.NA])
def test_nullable_explicit_completion_is_not_overwritten_by_review_presence(completion):
    """False means incomplete, while absent completion provenance stays unknown."""
    frame = scalar_frame(
        verifier_reviewed=pd.Series([True], dtype="boolean"),
        verifier_completed=pd.Series([completion], dtype="boolean"),
    )
    units, unavailable = reliability.review_units(frame)
    rate = reliability.relabel_rate(frame).overall
    if completion is False:
        assert not units.verifier_completed.iloc[0]
        assert unavailable is None and rate.of == 0
    else:
        assert pd.isna(units.verifier_completed.iloc[0]) and unavailable
        assert rate.unavailable_reason


@pytest.mark.parametrize("failure", ["error", "refusal", "no_answer"])
def test_explicit_completion_cannot_override_recorded_failure(failure):
    """A contradictory completion flag cannot promote an explicit failed verdict."""
    frame = scalar_frame(
        verifier_reviewed=pd.Series([True], dtype="boolean"),
        verifier_completed=pd.Series([True], dtype="boolean"),
        verifier_status=[failure],
        verifier_label=["B"],
    )
    assert reliability.relabel_rate(frame).overall.of == 0


@pytest.mark.parametrize("completion", [False, True])
def test_explicit_completion_remains_authoritative_with_usable_raw_verdict(completion):
    """A usable raw label cannot promote an explicitly incomplete review."""
    frame = scalar_frame(
        verifier_completed=pd.Series([completion], dtype="boolean"),
        overturned=[completion],
        verifier_status=["ok"],
        verifier_label=["B"],
    )
    rate = reliability.relabel_rate(frame).overall
    assert rate.of == int(completion) and rate.unavailable_reason is None
