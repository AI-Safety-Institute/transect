"""Report notes that state an absence or name a surface, so a reader
never has to guess why something is missing or what a chart element is."""

import html as html_mod
from pathlib import Path

import pandas as pd
import pytest

import transect
from transect import load, render
from transect.api import _run
from transect.report import charts, sections
from transect.scan_status import ScannerCoverage, ScanStatus

DEMO_SCAN = Path(__file__).parent / "fixtures" / "demo_scan"


@pytest.fixture(scope="module")
def demo_frames():
    return load(str(DEMO_SCAN)).frames()


def test_absent_setup_facts_read_as_not_recorded_by_source():
    """An eval-setup fact the source never recorded says so, instead of
    reading as a lookup transect failed at."""
    text = html_mod.unescape(str(sections.eval_setup_blocks(pd.DataFrame())))
    assert "not recorded by source" in text
    assert "data not found" not in text


@pytest.mark.parametrize(
    ("turn_total", "new_work", "coincide"),
    [
        ([100, 200, 300], [100, 200, 300], True),
        ([100, 200, 300], [100, 150, 300], False),
        ([None, None], [None, None], False),
    ],
)
def test_token_measures_coincide_only_when_every_turn_agrees(
    turn_total, new_work, coincide
):
    one = pd.DataFrame({"turn_total": turn_total, "new_work": new_work}).astype("Int64")
    assert charts.token_measures_coincide(one) is coincide


@pytest.mark.parametrize("coincide", [True, False])
def test_token_intro_names_coinciding_measures(coincide):
    """The measure selector has nothing to change on a cache-less source;
    the intro says so rather than leaving the toggle looking broken."""
    text = str(sections.token_intro(derived=True, coincide=coincide))
    assert ("per-turn total and per-turn new work are equal" in text) is coincide


def test_strip_caption_names_the_strip_and_rules_out_confidence(demo_frames):
    """A voting-regime caption leads with what the strip is, and says
    what it is not."""
    text = str(
        sections.agreement_strip_caption(
            demo_frames["phase_turns"], demo_frames["phases"]
        )
    )
    assert text.startswith('<p class="meta"><strong>Strip under the band')
    assert "judge agreement" in text
    assert "not confidence" in text


def test_solo_run_says_why_the_strip_is_absent(demo_frames):
    """One voter cannot agree with itself: the report states the strip
    is absent for that reason instead of silently dropping it."""
    phase_turns = demo_frames["phase_turns"].copy()
    phase_turns["judge_agreement"] = pd.NA
    phases = demo_frames["phases"].copy()
    phases["judge_regime"] = "solo"
    phases["k_rolls"] = 1
    text = str(sections.agreement_strip_caption(phase_turns, phases))
    assert "single judge" in text
    assert "stated confidence" in text


def test_strip_label_sits_in_the_band_margin(demo_frames):
    """The strip carries its own axis-side label, so the margin must
    leave room for it."""
    assert charts._BAND_MARGIN_LEFT >= 60
    components, _ = charts.phase_band(
        demo_frames["phase_turns"],
        demo_frames["phases"],
        {
            "ensembling": ("#111111", None),
            "validation_and_submission": ("#222222", None),
        },
        "phase-0",
        demo_frames["phase_turn_votes"],
    )
    assert len(components) == 3  # inputs, band, strip


@pytest.mark.parametrize(
    ("requested", "needle"),
    [
        (False, "judge_models"),
        (True, "Scan execution"),
    ],
)
def test_no_phases_note_names_the_reason(requested, needle):
    """No phases: either no judge was configured (say which argument),
    or the scanner ran and produced none (point at the execution block)."""
    scanners = [ScannerCoverage("token_timeline", True)]
    if requested:
        scanners.append(ScannerCoverage("decision_phases", True))
    text = str(sections.no_phases_note(ScanStatus(True, scanners)))
    assert needle in text


def test_structural_report_keeps_a_phase_timeline_section_with_the_note(
    demo_log, tmp_path
):
    """A $0 run renders the Phase timeline section as a note, not as a
    missing section; a judged run carries no such note."""
    judged = render(
        load(str(DEMO_SCAN)),
        report_path=str(tmp_path / "judged.html"),
        viewer=False,
        open_report=False,
    )
    assert "No judge was configured" not in Path(judged.report_paths[0]).read_text()
    structural = render(
        _run(logs=str(demo_log), spec=transect.Spec(), scans_dir=str(tmp_path / "s")),
        report_path=str(tmp_path / "structural.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(structural.report_paths[0]).read_text()
    assert "<h3>Phase timeline</h3>" in html
    assert "No judge was configured" in html


def test_card_filter_states_when_nothing_matches(tmp_path):
    """Filtering every card out shows a sentence, not blank space."""
    results = render(
        load(str(DEMO_SCAN)),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(results.report_paths[0]).read_text()
    assert "No phase matches the selected filters" in html


def test_no_judge_warning_names_the_declared_vocabularies(demo_log, tmp_path, capsys):
    """A spec with vocabularies and no judge model warns with the counts
    and the fix, so the absent sections are explained before the report."""
    spec = transect.Spec.model_validate(
        {"phases": ["a", "b"], "subagent_labels": ["x"]}
    )
    transect.transect(
        str(demo_log),
        spec,
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    out = capsys.readouterr().out
    assert "2 phases and 1 subagent_labels" in out
    assert "judge_models" in out
