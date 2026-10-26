"""Native Petri reports keep complete text and independent transcript scopes."""

import json
import re

import pandas as pd
import pytest
import test_petri_path

from transect import load
from transect.report.excerpts import TranscriptExtras
from transect.report.petri import assets_html, native_section
from transect.report.petri.transcript import PetriData
from transect.report.render import render_report


def data():
    """A branched transcript with one discarded continuation and complete text."""
    payload, _, _, tree = test_petri_path.TargetPathTests().prepare("m5")
    payload["target_model"] = "example/target"
    payload["transcript_id"] = "petri-test"
    for turn in payload["turns"]:
        turn.update(inputs=[], role_turn=turn["number"], lane_turn=0, citations=[])
    for message in payload["messages"]:
        message.update(
            role="assistant",
            text="complete text </script><b>unsafe</b>",
            reasoning="recorded thought",
        )
    return PetriData(payload, tree)


def config(document):
    return json.loads(re.search(r"data-petri-config>(.*?)</script>", document, re.S)[1])


@pytest.mark.parametrize("multiple", [False, True])
def test_native_config_scopes_all_links_and_escapes_untrusted_payload(multiple):
    """All generated card references are scoped, and payloads cannot close scripts."""
    phases = pd.DataFrame(columns=["turn", "phase", "phase_index"])
    document = str(native_section(data(), phases, 1, multiple=multiple))
    c = config(document)
    prefix = "petri-1-" if multiple else ""
    assert c["container_id"] == "phase-cards-1"
    assert c["payload_id"] == prefix + "petri-payload"
    assert c["path"]["surviving_turns"] == [2, 5]
    assert c["path"]["discarded_turns"] == [4]
    assert all(
        g["card"].startswith(prefix + "target-phase-") for g in c["path"]["groups"]
    )
    assert all(
        g["phase"] == prefix + "unclassified-transcript" for g in c["path"]["groups"]
    )
    assert "<b>unsafe</b>" not in document
    assert r"\u003c/script>" in document
    assert "worst_act" not in str(assets_html())


def test_native_render_auto_includes_petri_sections_and_keeps_non_petri_unchanged(
    monkeypatch,
):
    """The normal renderer supplies the Petri viewer, even without phase judgments."""
    results = load("tests/fixtures/demo_scan")
    identities = results.token_timeline.transcript_id.unique()
    monkeypatch.setattr(
        "transect.report.render.read_transcript_extras",
        lambda *_: {tid: TranscriptExtras(petri=data()) for tid in identities},
    )
    # Structural view: keep frame schemas, remove judged rows.
    results.phases = results.phases.iloc[:0]
    results.phase_turns = results.phase_turns.iloc[:0]
    document = render_report(results)
    assert "data-petri-config>" in document
    assert "Target conversation branches" in document
    assert document.index('id="branch-overview"') < document.index('id="phase-cards-0"')
    assert document.count("<!-- petri-assets -->") == 1
    monkeypatch.setattr("transect.report.render.read_transcript_extras", lambda *_: {})
    ordinary = render_report(results)
    assert "data-petri-config>" not in ordinary
    assert "<!-- petri-assets -->" not in ordinary
    assert 'name="viewport"' not in ordinary


@pytest.mark.parametrize("exact_path", [False, True])
def test_browser_petri_navigation_and_search_are_scoped(tmp_path, exact_path):
    """Branch links open lazy text and search stays local across transcripts."""
    playwright = pytest.importorskip("playwright.sync_api")
    phases = pd.DataFrame(columns=["turn", "phase", "phase_index"])
    parts = []
    for index in range(2):
        retained = data()
        if not exact_path:
            retained.tree["final_path"] = {
                "status": "unavailable",
                "reason": "No final context",
            }
        parts.append(
            '<article class="transcript">'
            f'<section id="petri-{index}-branch-overview"></section>'
            f'<section class="section"><h3>Target conversation</h3>'
            f'<div id="phase-cards-{index}"></div></section>'
            + str(native_section(retained, phases, index, multiple=True))
            + "</article>"
        )
    report = tmp_path / "petri.html"
    report.write_text(
        "<!DOCTYPE html><html><body>"
        + "".join(parts)
        + str(assets_html())
        + "</body></html>"
    )
    errors = []
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        playwright.expect(page.locator(".branch-viewport svg")).to_have_count(2)
        assert page.locator(".full-turn").count() == 0
        page.locator(
            '#petri-0-branch-overview svg a[href="#petri-0-turn-4"]'
        ).first.click()
        playwright.expect(page.locator("#petri-0-turn-4")).to_have_attribute("open", "")
        playwright.expect(page.locator("#petri-0-turn-4 .full-prose")).to_contain_text(
            "complete text"
        )
        page.locator("#petri-0-transcript-query").fill("recorded command")
        playwright.expect(
            page.locator("#petri-0-transcript-search-status")
        ).to_contain_text("1 matching calls")
        playwright.expect(page.locator("#petri-1-transcript-query")).to_have_value("")
        assert page.locator("#phase-cards-1 .full-turn").count() == 0
        assert page.locator("b").count() == 0
        assert errors == []
        page.screenshot(path=str(tmp_path / "petri.png"), full_page=True)
        browser.close()
