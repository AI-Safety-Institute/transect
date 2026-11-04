"""The run index of a multi-epoch render: one report per epoch, an index
that links them with their scan status, and the collection reliability
pooled over the epochs."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from helpers import PHASES_SPEC, demo_judge

from transect import Layer, render, transect
from transect.report.render import render_index
from transect.scan_status import ScanError


def test_a_multi_epoch_render_writes_the_index_first_and_links_each_epoch(
    epoch_runs, tmp_path, monkeypatch
):
    """Each epoch gets its own report named by epoch, the index lists them
    with their status and links them by file, every report links back,
    the index opens by default, and recorded text is escaped."""
    opened = []
    monkeypatch.setattr("transect.api.webbrowser.open", opened.append)
    epoch_runs.transcript_info["model"] = "<script>unsafe()</script>"
    render(epoch_runs, report_path=str(tmp_path / "runs.html"), viewer=False)
    names = [Path(p).name for p in epoch_runs.report_paths]
    assert names == ["runs_index.html", "runs_epoch1.html", "runs_epoch2.html"]
    assert opened == [Path(epoch_runs.report_paths[0]).as_uri()]
    index = BeautifulSoup(Path(epoch_runs.report_paths[0]).read_text(), "html.parser")
    rows = index.select(".runs-table tbody tr")
    assert [row.select("td")[-1].get_text() for row in rows] == ["Ran", "Ran"]
    assert [row.find("a")["href"] for row in rows] == names[1:]
    assert index.find("script") is None and "unsafe()" in index.get_text()
    for name in names[1:]:
        report = BeautifulSoup((tmp_path / name).read_text(), "html.parser")
        assert report.select_one('a[href="runs_index.html"]') is not None
        assert len(report.select(".transcript")) == 1
    assert [s["id"] for s in index.select("section.section")] == [
        "runs",
        "collection-reliability",
        "scan-status",
    ]


@pytest.mark.parametrize(
    ("errored", "timeline"),
    [(True, True), (False, False)],
    ids=["error recorded", "no timeline rows"],
)
def test_an_errored_or_timeline_less_epoch_keeps_its_row(epoch_runs, errored, timeline):
    """A recorded scanner error marks the epoch's status; an epoch with no
    timeline rows still lists from its transcript_info row."""
    if errored:
        epoch_runs.scan_status.errors.append(
            ScanError("token_timeline", "run-1", "failed")
        )
    if not timeline:
        epoch_runs.token_timeline = epoch_runs.token_timeline[
            epoch_runs.token_timeline.transcript_id == "run-0"
        ]
    index = BeautifulSoup(
        render_index(epoch_runs, [("e1.html", "run-0"), ("e2.html", "run-1")], "Index"),
        "html.parser",
    )
    rows = index.select(".runs-table tbody tr")
    assert [row.select("td")[-1].get_text() for row in rows] == [
        "Ran",
        "Errors recorded" if errored else "Ran",
    ]
    assert rows[1].find("a")["href"] == "e2.html"
    assert ("failed" in index.get_text()) is errored


@pytest.mark.parametrize(
    ("audit", "frame", "rows", "text"),
    [
        (("agent_span_id", "label"), "subagents", 2, "cohort of"),
        (("agent_span_id", "label"), "broken", 1, "not audited"),
        (("agent_span_id", "label"), None, 1, "no retained frame"),
        (None, "subagents", 0, ""),
    ],
    ids=["audited", "contract breach", "no frame", "no audit declared"],
)
def test_custom_layers_join_the_collection_only_through_their_audit(
    epoch_runs, audit, frame, rows, text
):
    """A custom layer that declares an audit gets its rows, a failing
    one a stated reason, a frame-less one a stated absence, and one
    without an audit nothing."""
    layer_frame = {
        "subagents": epoch_runs.subagents,
        "broken": epoch_runs.subagents.drop(columns="judge_regime"),
        None: None,
    }[frame]
    epoch_runs.extra_layers = [
        Layer(name="custom_judge", frame=layer_frame, audit=audit)
    ]
    epoch_runs.layer_frames = (
        {"custom_judge": layer_frame} if layer_frame is not None else {}
    )
    index = BeautifulSoup(
        render_index(epoch_runs, [("e1.html", "run-0")], "Index"), "html.parser"
    )
    found = [
        row.get_text()
        for row in index.select(".collection-reliability-table tbody tr")
        if "custom_judge" in row.get_text()
    ]
    assert len(found) == rows
    assert text in " ".join(found)


def test_a_single_judge_over_the_epochs_pools_without_agreement(epochs_logs, tmp_path):
    """A solo judge through the real pipeline yields an index whose
    collection row says there is no agreement to measure."""
    results = transect(
        logs=str(epochs_logs),
        spec=PHASES_SPEC,
        sample="epoch-sample-1",
        epochs="all",
        judge_models=demo_judge(phase_label="setup"),
        verify=False,
        narrate=False,
        scans_dir=str(tmp_path / "scans"),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    assert [Path(p).name for p in results.report_paths] == [
        "report_index.html",
        "report_epoch1.html",
        "report_epoch2.html",
        "report_epoch3.html",
    ]
    index = BeautifulSoup(Path(results.report_paths[0]).read_text(), "html.parser")
    rows = [
        r.get_text() for r in index.select(".collection-reliability-table tbody tr")
    ]
    assert len(rows) == 1
    assert "Phases" in rows[0] and "single judge" in rows[0] and "3 epoch(s)" in rows[0]


def test_browser_follows_an_epoch_link_and_back(epoch_runs, tmp_path):
    """A real browser lands on the epoch report from the index and returns."""
    playwright = pytest.importorskip("playwright.sync_api")
    render(
        epoch_runs,
        report_path=str(tmp_path / "runs.html"),
        viewer=False,
        open_report=False,
    )
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path(epoch_runs.report_paths[0]).as_uri())
        page.locator('tbody a[href="runs_epoch2.html"]').click()
        playwright.expect(page.locator(".transcript")).to_have_count(1)
        page.get_by_role("link", name="Run index").click()
        playwright.expect(page.locator(".runs-table")).to_be_visible()
        assert errors == []
        browser.close()
