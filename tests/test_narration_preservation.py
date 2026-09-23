"""Narration survives storage intact and reports explicit grouping fallback."""

from pathlib import Path

import pandas as pd
import pytest
from helpers import (
    PHASES_SPEC,
    group,
    model_turn,
    narrate_answer,
    narrative,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
)
from markupsafe import escape

from transect import load, render, transect
from transect.frames import phase_turn_votes_df, phases_df
from transect.frames.turn_groups import turn_groups_df
from transect.report import sections
from transect.scanners.phases import decision_phases

HEADLINE = (
    "  Reviewed "
    + "identifier_" * 25
    + "; did NOT execute <script>alert(1)</script>.  "
)
SUMMARY = (
    "  Inspected "
    + "configuration " * 35
    + '"literal_complete"; success was NOT confirmed.  '
)


def scan_narrative(items, *, narrate=True):
    judge = scripted_judge(seg_answer(seg(0, 3, "setup", 0.9)), narrate_answer(*items))
    return run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=narrate, cache=False),
        [model_turn(f"Reasoning {i}") for i in range(4)],
    ).value


def cards(raw):
    return str(
        sections.phase_cards(
            phases_df(raw),
            turn_groups_df(raw),
            phase_turn_votes_df(pd.DataFrame()),
            pd.DataFrame(),
            [],
            [],
            {},
            None,
            "cards",
            "phase",
        )
    )


@pytest.mark.parametrize(
    "items,narrate,status,note,fallback_count",
    [
        (
            [narrative(0, groups=[group(0, 3, title="Setup", gist="")])],
            True,
            "accepted",
            None,
            0,
        ),
        (
            [narrative(0, groups=[group(1, 2)])],
            True,
            "invalid_partition",
            "invalid group ranges",
            0,
        ),
        ([narrative(0)], True, "empty_groups", "no groups supplied", 0),
        ([], True, "no_narrative", "no usable narrative", 1),
        ([], False, "not_run", None, 0),
    ],
)
def test_group_status_is_stamped_and_fallback_visible(
    items, narrate, status, note, fallback_count
):
    """Neutral-looking accepted prose is not mistaken for missing or invalid groups."""
    value = scan_narrative(items, narrate=narrate)
    assert value["phases"][0]["narration_group_status"] == status
    assert value["narrator"]["n_fallback"] == fallback_count
    raw = pd.DataFrame([{"value": value, "transcript_id": "synthetic"}])
    assert phases_df(raw).iloc[0].narration_group_status == status
    html = cards(raw)
    if note:
        assert f"Neutral grouping: {note}." in html.split("</summary>", 1)[0]
    else:
        assert "Neutral grouping:" not in html


def test_blank_headline_keeps_template_without_changing_group_status():
    """A blank headline uses its template independently of group validation."""
    value = scan_narrative([narrative(0, " \n ", SUMMARY, [group(0, 3)])])
    assert value["phases"][0]["headline"] == "Setup (4 turns)"
    assert value["phases"][0]["narration_group_status"] == "accepted"
    assert value["phases"][0]["summary"] == SUMMARY


@pytest.fixture
def reloaded_narration(demo_log, tmp_path):
    """A mock-only stored scan re-rendered through the public API."""
    judge = scripted_judge(
        seg_answer(seg(0, 15, "setup", 0.9)),
        narrate_answer(narrative(0, HEADLINE, SUMMARY)),
    )
    result = transect(
        logs=str(demo_log),
        spec=PHASES_SPEC,
        judge_models=judge,
        verify=False,
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    loaded = load(result.scan_location)
    report = tmp_path / "reloaded.html"
    render(loaded, report_path=str(report), viewer=False, open_report=False)
    return result, loaded, report


def test_complete_narration_survives_public_load_and_render(reloaded_narration):
    """The real Parquet store and public reload preserve complete narrator output."""
    result, loaded, report = reloaded_narration
    assert len(loaded.phases) == 1
    assert loaded.phases.iloc[0].headline == HEADLINE
    assert loaded.phases.iloc[0].summary == SUMMARY
    assert loaded.phases.iloc[0].narration_group_status == "empty_groups"
    pd.testing.assert_frame_equal(result.phases, loaded.phases)
    html = Path(report).read_text()
    assert escape(HEADLINE.strip()) in html
    assert escape(SUMMARY) in html
    assert "<script>alert(1)</script>" not in html
    assert "Neutral grouping: no groups supplied." in html


def test_full_narration_and_fallback_note_in_browser(reloaded_narration):
    """Collapsed notice and complete escaped text stay readable at phone width."""
    playwright = pytest.importorskip("playwright.sync_api")
    _, _, report = reloaded_narration
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as err:
            pytest.skip(f"browser unavailable: {err}")
        for width in (390,):
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(report.as_uri(), wait_until="domcontentloaded")
            card = page.locator(".phase-cards > details").first
            assert card.get_attribute("open") is None
            note = card.locator(":scope > summary .narration-group-note")
            assert note.is_visible()
            assert note.inner_text() == "Neutral grouping: no groups supplied."
            assert HEADLINE.strip() in card.locator(":scope > summary").inner_text()
            card.locator(":scope > summary").click()
            assert card.locator(":scope > p").first.text_content() == SUMMARY
            assert card.locator("script").count() == 0
            assert card.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
            page.close()
        browser.close()
