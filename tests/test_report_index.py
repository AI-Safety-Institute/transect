"""Multi-run indexes expose scoped reliability and link to detailed reports."""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from transect import Layer, load, render
from transect.report.index import render_index
from transect.scan_status import ScanError


@pytest.fixture
def runs(monkeypatch):
    """Two runs use distinct judge regimes without making provider requests."""
    sources = [
        load(f"tests/fixtures/{name}") for name in ("demo_scan", "demo_scan_cohort")
    ]
    combined = {}
    for name in sources[0].frames():
        frames = []
        for index, source in enumerate(sources):
            frame = source.frames()[name].copy()
            frame["transcript_id"] = f"run-{index}"
            frame["sample_id"] = f"sample-{index}"
            frames.append(frame)
        combined[name] = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr("transect.report.render.read_transcript_extras", lambda *_: {})
    return dataclasses.replace(sources[0], **combined)


def test_render_creates_separate_index_with_scoped_metrics_and_links(runs, tmp_path):
    """The default entry is a separate index whose metrics keep judge setups apart."""
    render(
        runs,
        report_path=str(tmp_path / "details.html"),
        viewer=False,
        open_report=False,
    )
    assert [Path(p).name for p in runs.report_paths] == [
        "details_index.html",
        "details_run001_epoch1.html",
        "details_run002_epoch1.html",
    ]
    document = BeautifulSoup(Path(runs.report_paths[0]).read_text(), "html.parser")
    rows = document.select(".runs-table tbody tr")
    assert len(rows) == 2
    assert [cell.get_text() for cell in document.select(".runs-table thead th")] == [
        "Report",
        "Scan status",
    ]
    assert all(len(row.find_all(["th", "td"])) == 2 for row in rows)
    assert all(row.find("td").get_text() == "Ran" for row in rows)
    assert [section["id"] for section in document.select("section.section")] == [
        "runs",
        "collection-reliability",
        "scan-status",
    ]
    collection = document.select(".collection-reliability-table tbody tr")
    assert len(collection) == 4
    assert "solo + k-rolls" in collection[0].get_text()
    assert "cohort of" in collection[1].get_text()
    for link in document.select(".runs-table tbody a"):
        filename, anchor = link["href"].split("#")
        details = BeautifulSoup((tmp_path / filename).read_text(), "html.parser")
        assert details.find(id=anchor) is not None
        assert details.select_one('a[href="details_index.html"]') is not None
        assert len(details.select(".transcript")) == 1
    assert document.find("script") is None
    assert document.find(id="scan-status") is not None


@pytest.mark.parametrize("source", ["eval_log", "openclaw_telemetry_hal"])
def test_structural_and_failed_runs_remain_in_index(runs, source):
    """Runs without judgment or structural rows stay visible without fake scores."""
    runs.phases = runs.phases.iloc[:0]
    runs.phase_turns = runs.phase_turns.iloc[:0]
    runs.subagents = runs.subagents.iloc[:0]
    runs.transcript_info["source_type"] = source
    runs.token_timeline = runs.token_timeline[
        runs.token_timeline.transcript_id == "run-0"
    ]
    runs.scan_status.errors.append(
        ScanError("token_timeline", "run-1", "recorded failure")
    )
    document = BeautifulSoup(
        render_index(runs, [("runs.html", runs)], "Index"), "html.parser"
    )
    rows = document.select(".runs-table tbody tr")
    assert len(rows) == 2
    assert rows[0].find("td").get_text() == "Ran"
    assert rows[1].find("td").get_text() == "Errors recorded"
    assert "recorded failure" in document.get_text()
    assert rows[1].find("a")["href"] == "runs.html#scan-status"
    assert "No heuristic flags" not in document.get_text()


@pytest.mark.parametrize("broken", [False, True])
def test_index_includes_custom_judged_layers_and_honest_audit_failures(runs, broken):
    """Custom layers use the audit contract and expose unavailable metrics."""
    frame = runs.subagents.copy()
    if broken:
        frame = frame.drop(columns="judge_regime")
    runs.extra_layers = [
        Layer(name="custom_judge", frame=frame, audit=("agent_span_id", "label"))
    ]
    runs.layer_frames = {"custom_judge": frame}
    document = BeautifulSoup(
        render_index(runs, [("runs.html", runs)], "Index"), "html.parser"
    )
    rows = [
        row
        for row in document.select(".collection-reliability-table tbody tr")
        if "custom_judge" in row.get_text()
    ]
    assert len(rows) == (1 if broken else 2)
    assert all(("Not audited" in row.get_text()) == broken for row in rows)


@pytest.mark.parametrize("missing_timeline", [False, True])
def test_index_links_all_epochs_and_opens_by_default(
    runs, tmp_path, monkeypatch, missing_timeline
):
    """An epoch-split report opens one index linking every epoch's detailed run."""
    for frame in runs.frames().values():
        frame["sample_id"] = "same-sample"
        frame["epoch"] = frame.transcript_id.map({"run-0": 1, "run-1": 2})
    if missing_timeline:
        runs.token_timeline = runs.token_timeline[
            runs.token_timeline.transcript_id == "run-0"
        ]
    opened = []
    monkeypatch.setattr("transect.api.webbrowser.open", opened.append)
    render(runs, report_path=str(tmp_path / "runs.html"), viewer=False)
    assert len(runs.report_paths) == 3
    assert opened == [Path(runs.report_paths[0]).as_uri()]
    document = BeautifulSoup(Path(runs.report_paths[0]).read_text(), "html.parser")
    links = {link["href"].split("#")[0] for link in document.select("tbody a")}
    assert {"runs_run001_epoch1.html", "runs_run002_epoch2.html"} <= links


def test_index_browser_navigation_and_escaping(runs, tmp_path):
    """A real browser follows report links and preserves the simple section order."""
    playwright = pytest.importorskip("playwright.sync_api")
    runs.transcript_info["model"] = "<script>unsafe()</script>"
    render(
        runs,
        report_path=str(tmp_path / "runs.html"),
        viewer=False,
        open_report=False,
        sensitivity="TEST MARKING",
    )
    errors = []
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path(runs.report_paths[0]).as_uri())
        playwright.expect(page.locator(".runs-table tbody tr")).to_have_count(2)
        assert page.locator("#collection-reliability details.definitions").count() == 1
        assert page.locator(".collection-reliability-table td[style]").count() == 20
        assert page.locator(
            ".collection-reliability-table thead th"
        ).all_text_contents()[:3] == [
            "Scanner / setup",
            "Valid / expected ballots",
            "Mean unit agreement",
        ]
        assert page.locator("script").count() == 0
        assert page.locator(".sensitivity-banner").count() == 2
        page.screenshot(path=str(tmp_path / "run-index.png"), full_page=False)
        assert errors == []
        page.locator(
            'tbody a[href="runs_run002_epoch1.html#transcript-0"]'
        ).first.click()
        playwright.expect(page.locator("#transcript-0")).to_be_visible()
        page.get_by_role("link", name="Run index", exact=False).click()
        playwright.expect(page.locator(".runs-table")).to_be_visible()
        assert page.locator("section.section").evaluate_all(
            "sections => sections.map(section => section.id)"
        ) == ["runs", "collection-reliability", "scan-status"]
        browser.close()
