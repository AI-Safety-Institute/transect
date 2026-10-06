"""decision_phases end to end: segmentation, stitching, narration,
and the second-round verifier."""

import pytest
from helpers import (
    MODEL,
    PHASES_SPEC,
    REFUSED,
    StubTranscript,
    group,
    model_turn,
    narrate_answer,
    narrative,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
    verify_answer,
)
from inspect_ai.model import ContentReasoning, ContentText, get_model

from transect.scanners.phases import (
    decision_phases,
    project_phase_turns,
    system_prompt,
    turn_digests,
)
from transect.scanners.phases_common import (
    DigestJudgement,
    StitchedPhase,
    digest_line,
    is_labelled,
)
from transect.scanners.phases_verify import select_for_verify
from transect.spec import Spec


def turns(n):
    return [model_turn(f"reasoning turn {i}") for i in range(n)]


def test_system_prompt_carries_the_vocabulary_and_escape_hatches():
    """The judge sees every declared phase with its description, the
    task context, and always an ops bucket plus none_of_the_above."""
    system = system_prompt(PHASES_SPEC)
    assert "- setup: Environment preparation." in system
    assert "- experiment" in system
    assert "reproducing a systems paper" in system
    assert "ops" in system and "none_of_the_above" in system


def test_declared_ops_phase_replaces_the_reserved_bucket():
    """A spec phase flagged ops: true becomes the operational bucket;
    no extra reserved ops label is appended."""
    spec = Spec.model_validate(
        {"phases": [{"label": "coordination", "ops": True}, "experiment"]}
    )
    system = system_prompt(spec)
    assert "coordination" in system
    assert "- ops" not in system


def test_segments_sharing_a_label_stitch_into_one_phase():
    """Adjacent same-label segments merge; the phase records its turn
    range and the minimum member confidence."""
    judge = scripted_judge(
        seg_answer(
            seg(0, 1, "setup", 0.9),
            seg(2, 3, "setup", 0.7),
            seg(4, 5, "experiment", 0.8),
        ),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False), turns(6)
    ).value
    assert [(p["phase"], p["turn_start"], p["turn_end"]) for p in value["phases"]] == [
        ("setup", 0, 3),
        ("experiment", 4, 5),
    ]
    assert value["phases"][0]["min_confidence"] == 0.7


def test_every_turn_gets_a_row_with_its_basis():
    """The per-turn projection covers the whole range, distinguishing
    judged turns from filled gaps."""
    judge = scripted_judge(seg_answer(seg(0, 0, "setup", 0.9)))
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False), turns(2)
    ).value
    bases = {t["turn"]: t["basis"] for t in value["turns"]}
    assert bases[0] == "judged"
    assert len(bases) == 2


def test_reasoning_blocks_enter_the_digest_and_its_prompt_line():
    """Recorded reasoning rides the digest behind a [THINKING] marker,
    and a thinking-only turn is digest-eligible in its own right."""
    events = [
        model_turn(
            [
                ContentReasoning(reasoning="I should inspect the data first"),
                ContentText(text="Loading the csv."),
            ]
        ),
        model_turn([ContentReasoning(reasoning="try a different split")]),
        model_turn("plain text turn"),
    ]
    first, thinking_only, plain = turn_digests(StubTranscript(events))
    assert digest_line(first) == (
        "0: [THINKING] I should inspect the data first Loading the csv."
    )
    assert (thinking_only.turn, thinking_only.text) == (1, "")
    assert thinking_only.reasoning == "try a different split"
    assert (plain.reasoning, plain.text) == ("", "plain text turn")
    assert digest_line(plain) == "2: plain text turn"


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (ContentReasoning(reasoning="visible chain"), "visible chain"),
        (
            ContentReasoning(reasoning="payload", redacted=True, summary="the gist"),
            "the gist",
        ),
        (ContentReasoning(reasoning="", summary="summary only"), "summary only"),
        (ContentReasoning(reasoning="  \n ", summary="padded"), "padded"),
        (ContentReasoning(reasoning="payload", redacted=True), None),
    ],
)
def test_unreadable_reasoning_blocks_fall_back_or_drop_the_turn(block, expected):
    """Redacted or summary-only blocks fall back to readable text; a
    turn with no readable content yields no digest."""
    digests = turn_digests(StubTranscript([model_turn([block])]))
    if expected is None:
        assert digests == []
    else:
        assert [d.reasoning for d in digests] == [expected]


def test_a_thinking_only_turn_is_judged_not_attributed():
    """A turn carrying only reasoning blocks reaches the judge, so its
    row is judged rather than projection-attributed."""
    events = [
        model_turn("set up the environment"),
        model_turn([ContentReasoning(reasoning="now pick hyperparameters")]),
        model_turn("running the sweep"),
    ]
    judge = scripted_judge(seg_answer(seg(0, 2, "experiment", 0.9)))
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False), events
    ).value
    assert [t["basis"] for t in value["turns"]] == ["judged"] * 3


@pytest.mark.parametrize(
    ("phase_starts", "n_turns", "unjudged", "expected"),
    [
        # tool-only turns between two phases inherit the previous one
        ([0, 5], 8, (), [0, 0, 0, 0, 0, 1, 1, 1]),
        # a tool-only turn before a judged first digest joins the first phase
        ([1], 4, (), [0, 0, 0, 0]),
        # an unjudged digest turn between phases leaves it and the tool-only
        # turns after it in no phase
        ([0, 6], 8, (4,), [0, 0, 0, 0, None, None, 1, 1]),
        # a tool-only turn before a refused first digest takes that outcome
        ([3], 4, (1,), [None, None, None, 0]),
        ([], 3, (0,), [None, None, None]),
    ],
)
def test_projection_stops_inheriting_across_unjudged_turns(
    phase_starts, n_turns, unjudged, expected
):
    """A turn between phases inherits the previous one only when no unjudged
    digest turn separates them; the leading edge takes the first digest's
    outcome."""
    assert project_phase_turns(phase_starts, n_turns, unjudged) == expected


@pytest.mark.parametrize(
    ("basis", "phase", "expected"),
    [
        ("judged", "setup", True),
        ("filled", "setup", True),
        ("refusal", None, False),
    ],
)
def test_is_labelled_is_the_one_predicate_that_keeps_a_turn_inside_a_phase(
    basis, phase, expected
):
    """The stitcher closes a phase and the projection breaks inheritance on
    the same rows: those `is_labelled` rejects."""
    row = DigestJudgement(turn=0, phase=phase, confidence=0.5, basis=basis)
    assert is_labelled(row) is expected


def test_turns_before_a_refused_opening_chunk_belong_to_no_phase():
    """Refused reasoning turns and the tool-only turns after them carry no
    phase, so the frame and band cannot claim coverage the judge never gave."""
    events = [
        model_turn("reasoning 0"),
        model_turn("reasoning 1"),
        model_turn(""),  # tool-call-only: no digest
        model_turn(""),
        *[model_turn(f"reasoning {i}") for i in range(4, 8)],
    ]
    judge = scripted_judge(
        *[REFUSED] * 4,
        seg_answer(seg(4, 5, "experiment", 0.9)),
        seg_answer(seg(6, 7, "experiment", 0.9)),
    )
    value = run_scan(
        decision_phases(
            PHASES_SPEC, judge, chunk=2, verify=False, narrate=False, cache=False
        ),
        events,
    ).value
    assert [(p["turn_start"], p["turn_end"]) for p in value["phases"]] == [(4, 7)]
    by_turn = {t["turn"]: t for t in value["turns"]}
    assert [by_turn[t]["basis"] for t in range(4)] == ["refusal"] * 2 + [
        "unattributed"
    ] * 2
    assert all(by_turn[t]["phase_index"] is None for t in range(4))
    assert all(by_turn[t]["label_source"] is None for t in range(4))
    assert all(by_turn[t]["phase_index"] == 0 for t in range(4, 8))


def test_narrator_headlines_and_turn_groups_land_on_the_phase():
    judge = scripted_judge(
        seg_answer(seg(0, 4, "setup", 0.9)),
        narrate_answer(
            narrative(
                0,
                headline="Installed deps and configured the env",
                summary="The agent set things up.",
                groups=[group(0, 1), group(2, 4, title="config")],
            )
        ),
    )
    value = run_scan(decision_phases(PHASES_SPEC, judge, verify=False), turns(5)).value
    (phase,) = value["phases"]
    assert phase["headline"] == "Installed deps and configured the env"
    assert [(g["turn_start"], g["turn_end"]) for g in phase["turn_groups"]] == [
        (0, 1),
        (2, 4),
    ]
    assert value["narrator"]["ran"] is True


def test_failed_narration_falls_back_to_template_headlines():
    """A persistently refused narrator call (all in-call retries spent)
    degrades to per-phase template headlines instead of failing the scan."""
    judge = scripted_judge(seg_answer(seg(0, 1, "setup", 0.9)), *[REFUSED] * 4)
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, cache=False), turns(2)
    ).value
    (phase,) = value["phases"]
    assert phase["headline"] == "Setup (2 turns)"
    assert value["narrator"]["n_fallback"] == 1


def test_low_confidence_selects_the_verifier_and_a_confident_verdict_relabels():
    """A doubtful phase goes to review; the verifier's confident
    differing label rewrites it and the audit counts the relabel."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.4), seg(2, 3, "experiment", 0.9)),
        verify_answer(
            {
                "phase_index": 0,
                "phase": "experiment",
                "confidence": 0.9,
                "explanation": "clearly runs the experiment",
            }
        ),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=True, narrate=False), turns(4)
    ).value
    (phase,) = value["phases"]
    assert (phase["phase"], phase["turn_start"], phase["turn_end"]) == (
        "experiment",
        0,
        3,
    )
    audit = value["verifier"]
    assert audit["n_low_confidence"] == 1
    assert audit["n_relabelled"] == 1


def test_a_weak_verifier_verdict_never_relabels():
    """Below the confidence gate the differing verdict is recorded as
    weak; the judge's label stands."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.4)),
        verify_answer(
            {
                "phase_index": 0,
                "phase": "experiment",
                "confidence": 0.5,
                "explanation": "not sure",
            }
        ),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=True, narrate=False), turns(2)
    ).value
    (phase,) = value["phases"]
    assert phase["phase"] == "setup"
    audit = value["verifier"]
    assert audit["n_relabelled"] == 0 and audit["n_weak_relabel"] == 1
    (unit,) = phase["verifier_reviews"]
    assert unit["review"]["status"] == "ok" and unit["review"]["overturned"] is False


def test_confident_phases_still_get_a_random_sample_review():
    """With nothing doubtful the verifier reviews a fixed-seed random
    sample; a confirming verdict changes nothing and is audited."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.9)),
        verify_answer(
            {
                "phase_index": 0,
                "phase": "setup",
                "confidence": 0.9,
                "explanation": "confirmed",
            }
        ),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=True, narrate=False), turns(2)
    ).value
    audit = value["verifier"]
    assert audit["n_random_sample"] == 1
    assert audit["n_relabelled"] == 0
    (phase,) = value["phases"]
    assert phase["verifier"]["trigger"] == "random_sample"


def test_a_refused_verifier_chunk_leaves_phases_unreviewed():
    """A persistently refused verifier chunk (all in-call retries spent)
    is counted as no-verdict and the phase keeps its judged label."""
    judge = scripted_judge(seg_answer(seg(0, 1, "setup", 0.4)), *[REFUSED] * 4)
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=True, narrate=False, cache=False),
        turns(2),
    ).value
    (phase,) = value["phases"]
    assert phase["phase"] == "setup"
    assert value["verifier"]["n_no_verdict"] == 1
    assert phase["verifier_reviews"][0]["review"]["status"] == "refusal"


def test_k_rolls_vote_per_turn_and_record_agreement():
    """Three rolls majority-vote each turn; a dissenting roll lowers the
    phase's minimum agreement below 1."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.9)),
        seg_answer(seg(0, 1, "setup", 0.8)),
        seg_answer(seg(0, 0, "setup", 0.7), seg(1, 1, "experiment", 0.7)),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, k_rolls=3, verify=False, narrate=False),
        turns(2),
    ).value
    (phase,) = value["phases"]
    assert phase["phase"] == "setup"
    assert 0 < phase["min_agreement"] < 1


def test_out_of_vocabulary_answers_are_retried_never_invented():
    """A judge answer using an undeclared label is rejected and retried;
    the output only ever carries vocabulary labels."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "hallucinated_phase", 0.9)),
        seg_answer(seg(0, 1, "setup", 0.9)),
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False), turns(2)
    ).value
    assert [p["phase"] for p in value["phases"]] == ["setup"]


def test_spec_without_phases_is_rejected():
    with pytest.raises(ValueError):
        decision_phases(Spec(), MODEL)


def test_cohort_judges_vote_per_turn():
    """Two different models vote per turn; the majority decides and the
    votes block records both members."""
    a = get_model(
        MODEL,
        custom_outputs=[seg_answer(seg(0, 1, "setup", 0.9))],
        memoize=False,
    )
    b = get_model(
        "mockllm/model2",
        custom_outputs=[
            seg_answer(seg(0, 1, "setup", 0.8)).model_copy(
                update={"model": "mockllm/model2"}
            )
        ],
        memoize=False,
    )
    value = run_scan(
        decision_phases(PHASES_SPEC, [a, b], narrate=False), turns(2)
    ).value
    (phase,) = value["phases"]
    assert phase["phase"] == "setup"
    cohort = value["cohort"]
    assert cohort["agreement"]["n_members"] == 2
    assert [v["agreement"] for v in cohort["vote"]] == [1.0, 1.0]


@pytest.mark.parametrize(
    ("n_phases", "sample", "expected"),
    [(100, None, 5), (10, None, 3), (10, 0.5, 5), (4, 1.0, 4)],
)
def test_select_for_verify_sizes_the_random_sample(n_phases, sample, expected):
    """The random draw is max(5%, 3) of phases by default, round(share
    times n) for a float share, always capped at the phase count."""

    phases = [
        StitchedPhase(
            phase=f"p{k % 3}",
            turn_start=k * 5,
            turn_end=k * 5 + 4,
            n_turns=5,
            confidence=0.9,
            min_confidence=0.9,
            explanation="scripted",
        )
        for k in range(n_phases)
    ]
    reasons = select_for_verify(phases, sample=sample)
    assert len(reasons) == expected
    assert set(reasons.values()) <= {"random_sample"}


def test_verify_sample_outside_the_unit_interval_is_rejected():
    """A verify_sample outside [0, 1] raises at scanner build time."""
    with pytest.raises(ValueError, match="verify_sample"):
        decision_phases(PHASES_SPEC, scripted_judge(), verify=True, verify_sample=-0.1)
