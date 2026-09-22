"""Agreement coefficients and the reliability flags."""

from collections.abc import AsyncIterator

import pandas as pd
import pytest
from helpers import vote_output
from inspect_ai.model import ChatMessageUser, get_model
from inspect_scout import Loader, Transcript, loader, scanner as scout_scanner

import transect
from transect.frames.common import judge_identity
from transect.reliability import (
    NO_MEAN,
    NO_RATE,
    NO_REGIME,
    CohortAgreement,
    Mean,
    Rate,
    Regime,
    RelabelRate,
    cohort_agreement,
    detect_regime,
    gwet_ac1_nominal,
    krippendorff_alpha_nominal,
    label_stats,
    member_coverage,
    provenance_shares,
    rate,
    relabel_rate,
    spot_check_overturns,
    wilson_interval,
)
from transect.report.reliability import build_flags
from transect.scanners.cohort import judge_setup

NO_COHORT = CohortAgreement(None, None, None, 0)
NO_RELABEL = RelabelRate(NO_RATE, {})
KROLL = Regime("k_roll", 1, 3, ("m",), True, "m", True)


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ([["a", "a"], ["b", "b"]], 1.0),
        ([["a", "a", "b"]] * 9, -0.2),
        ([["a", "a"], ["a", "a"]], None),
        ([["a"]], None),
    ],
)
def test_gwet_ac1_hand_computed_cases(units, expected):
    """AC1 matches hand-computed values; a single observed category or
    no pairable unit is None, never a fabricated 1.0."""
    ac1, _ = gwet_ac1_nominal(units)
    if expected is None:
        assert ac1 is None
    else:
        assert ac1 == pytest.approx(expected)


def test_alpha_and_ac1_bracket_a_skewed_run():
    """On prevalence-skewed data alpha reads below AC1 - the bracket
    the audit renders both ends of."""
    units = [["a", "a", "b"]] * 9
    alpha, _ = krippendorff_alpha_nominal(units)
    ac1, _ = gwet_ac1_nominal(units)
    assert alpha is not None and ac1 is not None
    assert alpha < ac1 < 0


@pytest.mark.parametrize(
    ("mean", "expected"),
    [
        (0.67, "below 0.80"),
        (0.98, "above 0.95"),
        (0.90, None),
        # the boundaries themselves: 0.80 is healthy, 0.95 is not yet
        # suspicious - both comparisons are strict
        (0.80, None),
        (0.95, None),
    ],
)
def test_kroll_self_consistency_flags_low_and_suspicious_high(mean, expected):
    """K-roll self-consistency flags amber below 0.80 and above 0.95;
    the band between stays silent."""
    flags = build_flags(
        KROLL, Mean(mean, None, mean, 10), NO_COHORT, NO_RELABEL, NO_RATE
    )
    if expected is None:
        assert flags == []
    else:
        (flag,) = flags
        assert flag.level == "amber"
        assert expected in flag.explanation
        assert flag.metric == "k-roll self-consistency (mean per-turn agreement)"


@pytest.mark.parametrize(
    ("alpha", "level"),
    # 0.66 and 0.80 pin the boundaries: amber starts at exactly 0.66
    # (red is strictly below), silence at exactly 0.80
    [(0.5, "red"), (0.7, "amber"), (0.9, None), (0.66, "amber"), (0.8, None)],
)
def test_cohort_alpha_flag_bands(alpha, level):
    """Cohort alpha flags red below 0.66, amber up to 0.80, clean above."""
    regime = Regime("cohort", 3, 1, ("a", "b", "c"), False, None, False)
    flags = build_flags(
        regime, NO_MEAN, CohortAgreement(0.8, alpha, 0.9, 30), NO_RELABEL, NO_RATE
    )
    if level is None:
        assert flags == []
    else:
        (flag,) = flags
        assert flag.level == level
        assert flag.metric.startswith("cohort inter-judge agreement")


@pytest.mark.parametrize(
    ("count", "of", "level"),
    # any unjudged unit flags amber; red starts at exactly 25%
    [(0, 8, None), (1, 8, "amber"), (2, 8, "red"), (8, 8, "red")],
)
def test_unjudged_units_flag_bands(count, of, level):
    """Any unjudged unit flags amber; a quarter or more of the surface red."""
    solo = Regime("solo", 1, 1, ("m",), False, None, False)
    flags = build_flags(
        solo, NO_MEAN, NO_COHORT, NO_RELABEL, NO_RATE, unjudged=rate(count, of)
    )
    if level is None:
        assert flags == []
    else:
        (flag,) = flags
        assert flag.level == level
        assert flag.metric == "unjudged units (no delivered judgement)"
        assert flag.value == f"{count} of {of} ({count / of:.0%})"


def test_relabel_and_spot_check_flags_with_same_model_caveat():
    """A >= 0.20 re-label rate flags overall and per label with the
    same-model caveat inlined; any random-sample overturn flags red."""
    relabel = RelabelRate(rate(2, 4), {"setup": rate(2, 2)})
    solo = Regime("solo", 1, 1, ("m",), True, "m", True)
    flags = build_flags(solo, NO_MEAN, NO_COHORT, relabel, rate(1, 2))
    metrics = [f.metric for f in flags]
    assert metrics == [
        "verifier re-label rate (overall)",
        "verifier re-label rate (setup)",
        "verifier spot-check overturns",
    ]
    assert [f.level for f in flags] == ["amber", "amber", "red"]
    assert "same model as the judge" in flags[0].remediation
    assert flags[2].value == "1 of 2 sampled"


def test_label_stats_cover_overturned_away_and_minority_labels():
    """A label with zero decided units still gets a column when the
    verifier examined it under its original name; minority counts read
    votes against vote-decided labels only - a verifier-decided unit's
    ballots neither win nor lose."""

    decided = pd.DataFrame(
        [
            {
                "turn": 0,
                "phase": "b",
                "judge_agreement": 1.0,
                "confidence": 0.9,
                "label_source": "majority_vote",
            },
            {
                "turn": 1,
                "phase": "b",
                "judge_agreement": 0.5,
                "confidence": 0.8,
                "label_source": "majority_vote",
            },
            # verifier-decided: its ballots must not enter the minority
            # join on either side
            {
                "turn": 2,
                "phase": "c",
                "judge_agreement": None,
                "confidence": 0.9,
                "label_source": "verifier",
            },
        ]
    )
    votes = pd.DataFrame(
        [
            {"turn": 0, "phase": "b"},
            {"turn": 0, "phase": "a"},
            {"turn": 1, "phase": "b"},
            {"turn": 1, "phase": "b"},
            {"turn": 2, "phase": "b"},
        ]
    )
    entity = pd.DataFrame(
        [
            {
                "original_label": "a",
                "verifier_reviewed": True,
                "overturned": True,
                "verifier_trigger": "random_sample",
            }
        ]
    )
    stats = {
        st.label: st for st in label_stats(decided, votes, entity, "turn", "phase")
    }
    assert set(stats) == {"a", "b", "c"}
    assert stats["a"].n == 0
    assert (stats["a"].relabelled.count, stats["a"].relabelled.of) == (1, 1)
    assert (stats["a"].spot_checked.count, stats["a"].spot_checked.of) == (1, 1)
    assert (stats["a"].minority.count, stats["a"].minority.of) == (1, 1)
    assert stats["b"].n == 2
    assert stats["b"].agreement.mean == 0.75
    # turn 2's ballot for "b" is excluded: the verifier decided there
    assert (stats["b"].minority.count, stats["b"].minority.of) == (0, 3)
    assert stats["c"].minority == Rate(0, 0, None, None)


def test_member_coverage_reconciles_with_the_filled_basis():
    """produced + every counted miss reason == asked, with filled
    counted as a miss (the member answered but left the turn
    uncovered)."""

    members = pd.DataFrame(
        [
            {"model": "m", "roll": 0, "basis": "judged"},
            {"model": "m", "roll": 0, "basis": "judged"},
            {"model": "m", "roll": 0, "basis": "filled"},
            {"model": "m", "roll": 0, "basis": "refusal"},
        ]
    )
    (row,) = member_coverage(
        members, "basis", "judged", ("refusal", "no_answer", "missing_turn", "filled")
    )
    assert (row.coverage.count, row.coverage.of) == (2, 4)
    assert row.misses == {
        "refusal": 1,
        "no_answer": 0,
        "missing_turn": 0,
        "filled": 1,
    }
    assert row.coverage.count + sum(row.misses.values()) == row.coverage.of


def test_provenance_shares_count_decided_units_per_source():
    """Shares tally each classification's decided units by the label
    source that decided them, zero-filled over the source vocabulary."""

    decided = pd.DataFrame(
        [
            {"turn": 0, "phase": "b", "label_source": "majority_vote"},
            {"turn": 1, "phase": "b", "label_source": "verifier"},
            {"turn": 2, "phase": "a", "label_source": "majority_vote"},
        ]
    )
    shares = provenance_shares(decided, "phase", ["a", "b"])
    assert shares["majority_vote"] == {"a": 1, "b": 1}
    assert shares["verifier"] == {"a": 0, "b": 1}
    assert shares["single_judge"] == {"a": 0, "b": 0}


@pytest.mark.parametrize(
    ("models", "k_rolls", "regime"),
    [
        (["m"], 1, "solo"),
        (["m"], 3, "k_roll"),
        (["m", "n"], 1, "cohort"),
        (["m", "m", "n"], 1, "cohort"),
    ],
)
def test_judge_setup_classifies_the_regime(models, k_rolls, regime):
    """The stamped block names the regime from the deduplicated roster
    and the roll count; a disarmed verifier stamps no verifier model."""
    block = judge_setup(models, k_rolls, verifier_armed=False, verifier_model="v")
    assert block["regime"] == regime
    assert block["models"] == list(dict.fromkeys(models))
    assert block["n_models"] == len(dict.fromkeys(models))
    assert block["verifier_model"] is None
    assert block["verifier_same_model"] is False


def test_judge_setup_same_model_verifier_only_for_a_lone_judge():
    solo = judge_setup(["m"], 1, verifier_armed=True, verifier_model="m")
    assert solo["verifier_same_model"] is True
    assert solo["verifier_model"] == "m"
    cohort = judge_setup(["m", "n"], 1, verifier_armed=True, verifier_model="m")
    assert cohort["verifier_same_model"] is False


def test_judge_identity_projects_the_block_and_rejects_missing_blocks():
    block = judge_setup(["m"], 3, verifier_armed=True, verifier_model="m")
    cols = judge_identity({"judge": block})
    assert cols == {
        "judge_regime": "k_roll",
        "n_models": 1,
        "k_rolls": 3,
        "verifier_armed": True,
        "verifier_same_model": True,
    }
    with pytest.raises(KeyError, match="re-scan"):
        judge_identity({"label": "x"})


def test_detect_regime_reads_the_stamped_columns():
    """detect_regime is a column read: the regime comes off the judged
    rows' judge-identity columns, unjudged rows are ignored, and an
    unjudged frame is NO_REGIME."""
    frame = pd.DataFrame(
        [
            {
                "judge_models": None,
                "verifier_model": None,
                "judge_regime": None,
                "n_models": None,
                "k_rolls": None,
                "verifier_armed": None,
                "verifier_same_model": None,
            },
            {
                "judge_models": "m+n",
                "verifier_model": "v",
                "judge_regime": "cohort",
                "n_models": 2,
                "k_rolls": 1,
                "verifier_armed": True,
                "verifier_same_model": False,
            },
        ]
    )
    regime = detect_regime(frame)
    assert regime == Regime("cohort", 2, 1, ("m", "n"), True, "v", False)
    assert detect_regime(frame.iloc[:1]) == NO_REGIME
    assert detect_regime(frame.iloc[:0]) == NO_REGIME


def test_detect_regime_names_the_missing_columns_and_the_fix():
    """A frame without the judge-identity columns fails with the
    contract in the message, not a bare column KeyError."""
    with pytest.raises(KeyError, match="judge_identity"):
        detect_regime(pd.DataFrame({"label": ["a"]}))


@pytest.mark.parametrize(
    ("call", "missing"),
    [
        (lambda df: cohort_agreement(df, "turn", "phase"), "phase"),
        (lambda df: relabel_rate(df), "verifier_reviewed"),
        (lambda df: spot_check_overturns(df), "verifier_trigger"),
        (lambda df: member_coverage(df, "basis", "judged", ()), "model"),
        (lambda df: label_stats(df, df, df, "turn", "phase"), "judge_agreement"),
    ],
)
def test_statistics_name_their_column_contract_on_a_bare_frame(call, missing):
    """Every frame-consuming statistic fails a nonconforming frame with
    the missing columns and its contract, never a bare pandas error."""
    with pytest.raises(KeyError, match=missing):
        call(pd.DataFrame({"turn": [0]}))


def test_wilson_interval_needs_an_observation():
    """No interval without data: n = 0 is None, never a fabricated
    (0, 1)."""
    assert wilson_interval(0, 0) is None


def test_units_never_pool_across_transcripts():
    """Turn 5 of two transcripts is two judged units: agreement stays
    perfect when each transcript agrees internally, and the minority
    join never crosses transcripts."""
    votes = pd.DataFrame(
        {
            "transcript_id": ["a", "a", "b", "b"],
            "turn": [5, 5, 5, 5],
            "phase": ["setup", "setup", "experiment", "experiment"],
        }
    )
    result = cohort_agreement(votes, "turn", "phase")
    assert result.percent_agreement == 1.0
    decided = pd.DataFrame(
        {
            "transcript_id": ["a", "b"],
            "turn": [5, 5],
            "phase": ["setup", "experiment"],
            "judge_agreement": [1.0, 1.0],
            "confidence": [0.9, 0.9],
            "label_source": ["majority_vote", "majority_vote"],
        }
    )
    frame = pd.DataFrame(
        {
            "original_label": [None, None],
            "verifier_reviewed": [False, False],
            "overturned": [False, False],
            "verifier_trigger": [None, None],
        }
    )
    stats = {st.label: st for st in label_stats(decided, votes, frame, "turn", "phase")}
    # cross-transcript pooling would make each label lose 2 of 4 votes
    assert stats["setup"].minority.count == 0
    assert stats["experiment"].minority.count == 0


def test_detect_regime_reads_nullable_dtypes_and_later_verifier_rows():
    """The column read works on the frames' real nullable dtypes, reads
    k_roll off the stamp, and finds the verifier name on a later row."""
    frame = pd.DataFrame(
        {
            "judge_models": ["m", "m"],
            "verifier_model": [None, "v"],
            "judge_regime": ["k_roll", "k_roll"],
            "n_models": [1, 1],
            "k_rolls": [3, 3],
            "verifier_armed": [True, True],
            "verifier_same_model": [False, False],
        }
    ).astype(
        {
            "n_models": "Int64",
            "k_rolls": "Int64",
            "verifier_armed": "boolean",
            "verifier_same_model": "boolean",
        }
    )
    assert detect_regime(frame) == Regime("k_roll", 1, 3, ("m",), True, "v", False)


def test_judge_setup_default_model_path_stamps_an_empty_roster():
    """models=None resolves no name at factory time: empty roster,
    n_models still 1, no verifier name, never same-model."""
    block = judge_setup([], 1, verifier_armed=True, verifier_model=None)
    assert block["models"] == []
    assert block["n_models"] == 1
    assert block["regime"] == "solo"
    assert block["verifier_model"] is None
    assert block["verifier_same_model"] is False


def test_wilson_interval_matches_published_values():
    """Hand-computed pins: the textbook n=100, k=50 case (0.4038,
    0.5962) and the k=0 boundary; rate() carries the same interval.
    A z-formula typo would mis-state every CI in every audit."""
    interval = wilson_interval(50, 100)
    assert interval is not None
    assert (round(interval[0], 4), round(interval[1], 4)) == (0.4038, 0.5962)
    boundary = wilson_interval(0, 5)
    assert boundary is not None
    assert boundary[0] == 0.0
    assert boundary[1] == pytest.approx(0.4345, abs=5e-4)
    r = rate(2, 4)
    assert r.rate == 0.5
    assert r.ci == wilson_interval(2, 4)


def test_cohort_scanner_without_a_loader_judges_the_whole_transcript(
    demo_log, tmp_path
):
    """No loader: Scout's identity loader hands the whole transcript
    to the judge as one item - one call, one label per transcript,
    everything downstream (frame, ballots, audit) unchanged."""

    @scout_scanner(messages="all")
    def run_risk(judge_models=None):
        return transect.cohort_llm_scanner(
            question="Is this whole run risky?",
            answer=["risky", "safe"],
            models=judge_models,
        )

    doves = get_model(
        "mockllm/model", custom_outputs=[vote_output("safe", 0.9)], memoize=False
    )
    hawks = get_model(
        "mockllm/model2",
        custom_outputs=[vote_output("risky", 0.4, model="mockllm/model2")],
        memoize=False,
    )
    results = transect.transect(
        logs=str(demo_log.parent),
        spec=transect.Spec(),
        judge_models=[doves, hawks],
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="run_risk",
                scanner=run_risk,
                frame=transect.turns_frame,
                audit=("item", "label"),
            )
        ],
    )
    frame = results.layer_frames["run_risk"]
    assert len(frame) == 1  # the transcript is the unit
    assert list(frame.label) == ["safe"]  # confidence tie-break
    assert detect_regime(frame).kind == "cohort"
    assert len(transect.member_ballots(frame, "item")) == 2
    assert "<h4>run_risk" in open(results.report_paths[0]).read()


def test_member_ballots_renames_the_label_and_skips_memberless_rows():
    """The ballots' label lands in the caller's label_col so the frames
    join in label_stats; a row without a cohort block (solo, or
    mechanical) contributes nothing instead of crashing iteration."""
    frame = pd.DataFrame(
        [
            {
                "transcript_id": "t",
                "item": "i0",
                "members": [
                    {
                        "model": "m",
                        "roll": 0,
                        "label": "a",
                        "confidence": 0.9,
                        "explanation": None,
                        "status": "ok",
                        "error": None,
                    }
                ],
            },
            {"transcript_id": "t", "item": "i1", "members": float("nan")},
        ]
    )
    ballots = transect.member_ballots(frame, "item", "activity")
    assert len(ballots) == 1
    assert list(ballots.activity) == ["a"]
    assert "label" not in ballots.columns


def test_custom_judged_layer_conforms_to_the_reliability_contract(demo_log, tmp_path):
    """The supported custom judged path end to end."""

    @loader(messages="all")
    def halves() -> Loader[Transcript]:
        async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
            for k in (0, 1):
                yield Transcript(
                    transcript_id=f"item-{k}",
                    messages=[ChatMessageUser(content=f"half {k} of the run")],
                    metadata={"half": k},
                )

        return load

    @scout_scanner(loader=halves())
    def risk_judge(judge_models=None):
        return transect.cohort_llm_scanner(
            question="Is this half of the run risky?",
            answer=["risky", "safe"],
            models=judge_models,
            vocabulary={"risky": "could go wrong", "safe": "could not"},
        )

    # every item splits 1-1; risky wins each tie-break on confidence
    doves = get_model(
        "mockllm/model", custom_outputs=[vote_output("risky", 0.8)] * 2, memoize=False
    )
    hawks = get_model(
        "mockllm/model2",
        custom_outputs=[vote_output("safe", 0.2, model="mockllm/model2")] * 2,
        memoize=False,
    )
    # the factory form: the roster rides transect()'s own judge_models
    results = transect.transect(
        logs=str(demo_log.parent),
        spec=transect.Spec(),
        judge_models=[doves, hawks],
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="risk",
                scanner=risk_judge,
                frame=transect.turns_frame,
                audit=("item", "label"),
            )
        ],
    )
    frame = results.layer_frames["risk"]
    assert sorted(frame.half) == [0, 1]  # item metadata rode into the frame

    regime = detect_regime(frame)
    assert regime.kind == "cohort"
    assert regime.n_models == 2
    assert not regime.verifier_on

    votes = transect.member_ballots(frame, "item")
    assert len(votes) == 4  # 2 items x 2 members
    assert set(votes.columns) >= {"transcript_id", "item", "model", "roll", "label"}
    stats = label_stats(
        frame, votes, frame, "item", "label", vocabulary=["risky", "safe"]
    )
    assert [st.label for st in stats] == ["risky", "safe"]
    risky, safe = stats
    assert risky.n == 2 and safe.n == 0  # confidence tie-break went risky
    assert risky.agreement.mean == 0.5  # every unit split 1-1
    assert safe.minority.rate == 1.0  # safe only ever loses

    html = open(results.report_paths[0]).read()
    assert "<h4>risk" in html and "custom-layer-badge" in html
    assert "risk labelling" in html
    vocab = results.label_definitions
    assert list(vocab[vocab.surface == "risk"].label) == ["risky", "safe"]
