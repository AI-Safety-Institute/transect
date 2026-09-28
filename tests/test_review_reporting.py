"""Report consumers expose original review counts and completion."""

import pandas as pd
from bs4 import BeautifulSoup
from test_phase_review_units import reviewed_frame

from transect.frames import phase_turn_votes_df
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.custom import _run_judge_cells


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
    assert "1 of 3 completed original phase reviews relabelled (3 selected)" in text


def test_audit_shows_completion_coverage_beside_conditional_rate():
    frame, _ = reviewed_frame(missing=(0, 2))
    rows = {
        row["label"]: row
        for row in sections._span_verifier_rows(frame, unit_word="phase")
    }
    assert "3 random sample" in rows["Verifier selection"]["value"]
    assert (
        "1 completed · 2 without usable verdict" in (rows["Verifier outcomes"]["value"])
    )
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
            ["1 selected · 0 completed · 1 without usable verdict · 0 overturned"],
        )
    ]
