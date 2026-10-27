"""Report consumers expose original review counts and completion."""

import pandas as pd
import pytest
from bs4 import BeautifulSoup
from helpers import (
    MODEL,
    PHASES_SPEC,
    REFUSED,
    model_turn,
    phase_turns_frame,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
)
from test_frames import raw_row
from test_phase_review_units import reviewed_frame

from transect.frames import (
    phase_turn_votes_df,
    phase_turns_df,
    phases_df,
    subagent_votes_df,
    subagents_df,
)
from transect.reliability import detect_regime
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.custom import _run_judge_cells
from transect.scanners.phases import decision_phases


def audit(frame):
    turns = pd.DataFrame(
        {
            "transcript_id": ["t1"] * 3,
            "turn": [0, 1, 2],
            "phase": ["A"] * 3,
            "basis": ["judged"] * 3,
            "judge_agreement": [None] * 3,
            "confidence": [0.9] * 3,
            "label_source": ["verifier"] * 3,
        }
    )
    return sections._entity_audit(
        "Phases",
        frame,
        phase_turn_votes_df(pd.DataFrame()),
        "turn",
        "phase",
        turns,
    )[0]


def test_phase_metadata_counts_original_reviews_after_merging():
    frame, _ = reviewed_frame()
    note = sections._verifier_notes(frame)
    assert note["selected"] == note["verified"] == 3
    assert note["overturned"] == 1
    module = jinja_env().get_template("notes.html.j2").module
    text = BeautifulSoup(
        str(module.phase_meta_line(1, "judge", "m", None, note, 0)), "html.parser"
    ).get_text()
    assert "1 of 3 judged original phase reviews relabelled (3 selected)" in text


def test_audit_shows_completion_coverage_beside_conditional_rate():
    frame, _ = reviewed_frame(missing=(0, 2))
    rows = {
        row["label"]: row
        for row in sections._span_verifier_rows(frame, unit_word="phase")
    }
    assert "3 random sample" in rows["Verifier selection"]["value"]
    assert "1 judged · 2 without usable verdict" in (rows["Verifier outcomes"]["value"])
    audit_rows = {row["label"]: row for row in audit(frame)["rows"]}
    assert "1/1 examined" in audit_rows["Verifier re-label rate (overall)"]["value"]


def test_unjudged_flag_excludes_attributed_turns_from_its_denominator():
    """Attributed turns the judge never saw do not dilute the unjudged share."""
    frame, _ = reviewed_frame()
    turns = pd.DataFrame(
        {
            "transcript_id": ["t1"] * 4,
            "turn": [0, 1, 2, 3],
            "phase": ["A", None, "A", "A"],
            "basis": ["judged", "refusal", "attributed", "attributed"],
            "judge_agreement": [None] * 4,
            "confidence": [0.9, None, None, None],
            "label_source": ["single_judge", None, None, None],
        }
    )
    _, flags = sections._entity_audit(
        "Phases", frame, phase_turn_votes_df(pd.DataFrame()), "turn", "phase", turns
    )
    (unjudged,) = [f for f in flags if f.metric.startswith("unjudged")]
    assert unjudged.value == "1 of 2 (50%)"
    assert unjudged.level == "red"


@pytest.mark.parametrize("failure", ["no_answer", "refusal", "missing_turn"])
@pytest.mark.parametrize("k_rolls", [1, 3])
def test_total_phase_judging_failure_keeps_the_judge_setup_and_audit(failure, k_rolls):
    """Failed judgments remain auditable even when no phase was produced."""
    failed = {
        "no_answer": seg_answer(),
        "refusal": REFUSED,
        "missing_turn": seg_answer(seg(20, 21, "setup")),
    }[failure]
    raw = pd.DataFrame(
        [
            raw_row(
                run_scan(
                    decision_phases(
                        PHASES_SPEC,
                        scripted_judge(*([failed] * 30)),
                        k_rolls=k_rolls,
                        cache=False,
                        narrate=False,
                    ),
                    [
                        model_turn("Planning."),
                        model_turn("Running."),
                        model_turn("Done."),
                    ],
                )
            )
        ]
    )
    phases = phases_df(raw)
    turns = phase_turns_df(raw)
    votes = phase_turn_votes_df(raw)
    assert phases.empty
    assert turns.basis.tolist() == [failure] * 3
    regime = detect_regime(turns)
    assert regime.models == (MODEL,)
    assert regime.k_rolls == k_rolls
    assert regime.verifier_on and regime.verifier_model == MODEL
    assert regime.verifier_same_model
    flags = sections.phase_reliability_flags(phases, votes, turns)
    (unjudged,) = [f for f in flags if f.metric.startswith("unjudged")]
    assert unjudged.value == "3 of 3 (100%)"
    assert unjudged.level == "red"
    text = _phase_audit_text(phases, turns, votes)
    assert "3 unjudged" in text
    assert "3 of 3 (100%)" in text
    assert "Pass judge_models" not in text
    assert "no judge ran" not in text


def test_no_digest_content_is_configured_without_attempted_judgments():
    """An empty-content run retains its setup without claiming failed attempts."""
    raw = pd.DataFrame(
        [
            raw_row(
                run_scan(
                    decision_phases(PHASES_SPEC, MODEL, narrate=False),
                    [model_turn(""), model_turn("")],
                )
            )
        ]
    )
    phases, turns, votes = phases_df(raw), phase_turns_df(raw), phase_turn_votes_df(raw)
    assert turns.basis.tolist() == ["unattributed"] * 2
    assert detect_regime(turns).models == (MODEL,)
    assert sections.phase_reliability_flags(phases, votes, turns) == []
    text = _phase_audit_text(phases, turns, votes)
    assert "0 judged" in text and "2 unattributed" in text
    assert "Pass judge_models" not in text


def test_partial_phase_judging_failure_retains_its_attempted_turn_denominator():
    """A successful phase does not dilute or remove an unanswered digest turn."""
    raw = pd.DataFrame(
        [
            raw_row(
                run_scan(
                    decision_phases(
                        PHASES_SPEC,
                        scripted_judge(
                            seg_answer(seg(0, 0, "setup")),
                            seg_answer(),
                            seg_answer(seg(2, 2, "experiment")),
                        ),
                        chunk=1,
                        cache=False,
                        verify=False,
                        narrate=False,
                    ),
                    [model_turn("Plan."), model_turn("Run."), model_turn("Review.")],
                )
            )
        ]
    )
    phases, turns, votes = phases_df(raw), phase_turns_df(raw), phase_turn_votes_df(raw)
    (unjudged,) = sections.phase_reliability_flags(phases, votes, turns)
    assert unjudged.value == "1 of 3 (33%)"
    assert unjudged.level == "red"


def test_empty_phase_result_retains_a_distinct_configured_verifier():
    """No phase output does not lose a distinct verifier's configured identity."""
    raw = pd.DataFrame(
        [
            raw_row(
                run_scan(
                    decision_phases(
                        PHASES_SPEC,
                        scripted_judge(seg_answer()),
                        verifier_model="mockllm/verifier",
                        cache=False,
                        narrate=False,
                    ),
                    [model_turn("Work.")],
                )
            )
        ]
    )
    regime = detect_regime(phase_turns_df(raw))
    assert regime.verifier_on and regime.verifier_model == "mockllm/verifier"
    assert not regime.verifier_same_model


def test_structural_only_scan_keeps_the_no_judge_note():
    """An absent phase scanner remains distinct from attempted failed judging."""
    raw = pd.DataFrame()
    turns = phase_turns_df(raw)
    assert detect_regime(turns).kind == "none"
    assert "Pass judge_models" in _phase_audit_text(
        phases_df(raw), turns, phase_turn_votes_df(raw)
    )


def test_zero_turn_projection_creates_no_attempts_or_synthetic_phase_rows():
    """An empty stored turn surface has no attempted judgment to flag as failed."""
    raw = pd.DataFrame(
        [
            {
                "transcript_id": "tr1",
                "value": {"turns": [], "phases": [], "phase_names": ["setup"]},
            }
        ]
    )
    assert phases_df(raw).empty and phase_turns_df(raw).empty
    assert phase_turns_df(raw).phase.cat.categories.tolist() == ["setup"]
    assert (
        sections.phase_reliability_flags(
            phases_df(raw), phase_turn_votes_df(raw), phase_turns_df(raw)
        )
        == []
    )


def _phase_audit_text(phases, turns, votes):
    html = sections.reliability_audit(
        phases,
        turns,
        subagents_df(pd.DataFrame()),
        subagent_votes_df(pd.DataFrame()),
        None,
        votes,
    )
    return BeautifulSoup(str(html), "html.parser").get_text(" ", strip=True)


def test_failed_review_is_not_described_as_an_unchanged_verdict():
    row = pd.Series(
        {
            "verifier_selected": True,
            "verifier_completed": False,
            "verifier_status": "error",
            "verifier_trigger": "random_sample",
            "verifier_label": None,
            "overturned": False,
        }
    )
    stored_row = next(pd.DataFrame([row]).itertuples(index=False))
    assert sections._span_verifier_cell(stored_row) == (
        "selected, no usable verdict (random sample)"
    )
    frame = pd.DataFrame([{**row.to_dict(), "turn": 0}])
    cells = _run_judge_cells(frame, [{"start": 0, "end": 0}])
    assert cells == [
        (
            "verifier",
            "verifier_text",
            ["1 selected · 0 judged · 1 without usable verdict · 0 overturned"],
        )
    ]


def _no_phase_turns():
    """Eight turns: a refused opening chunk, two unattributed tool-only
    turns, four judged turns."""
    return phase_turns_frame(
        ["refusal"] * 2 + ["unattributed"] * 2 + ["judged"] * 4, [None] * 4 + [0] * 4
    )


@pytest.mark.parametrize(
    ("bases", "expected"),
    [
        (
            ["refusal"] * 2 + ["unattributed"] * 2 + ["judged"] * 4,
            "8 total · 4 judged · 0 filled/attributed · 2 unjudged · 2 unattributed",
        ),
        # the unattributed bucket is omitted when empty
        (
            ["refusal"] * 2 + ["attributed"] * 2 + ["judged"] * 4,
            "8 total · 4 judged · 2 filled/attributed · 2 unjudged",
        ),
    ],
)
def test_turns_summary_counts_unattributed_turns_apart_from_covered_ones(
    bases, expected
):
    """Tool-only turns the projection left without a phase are not reported
    as covered alongside attributed turns that have one."""
    assert sections._turns_summary(phase_turns_frame(bases, [0] * 8)) == expected


def test_meta_line_greyed_count_matches_the_band():
    """The band greys every unattributed turn, so the caption counts them
    with the unjudged ones."""
    frame, _ = reviewed_frame()
    text = BeautifulSoup(
        str(sections.phase_meta_line(frame, _no_phase_turns())), "html.parser"
    ).get_text()
    assert "4 unjudged turn(s) greyed" in text
