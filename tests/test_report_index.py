"""The run index of a multi-epoch render: one report per epoch, an index
that links them with their scan status, and the collection reliability
pooled over the epochs."""

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from transect import Layer, render
from transect.report.render import render_index
from transect.scan_status import ScanError


def test_a_multi_epoch_render_writes_the_index_first_and_links_each_epoch(
    epoch_runs, tmp_path, monkeypatch
):
    """Each epoch gets its own report named by epoch, the index lists them
    with their status and links them by quoted file name, every report
    links back, the index opens by default, and recorded text is escaped."""
    opened = []
    monkeypatch.setattr("transect.api.webbrowser.open", opened.append)
    epoch_runs.transcript_info["model"] = "<script>unsafe()</script>"
    render(epoch_runs, report_path=str(tmp_path / "my runs.html"), viewer=False)
    names = [Path(p).name for p in epoch_runs.report_paths]
    assert names == ["my runs_index.html", "my runs_epoch1.html", "my runs_epoch2.html"]
    assert opened == [Path(epoch_runs.report_paths[0]).as_uri()]
    index = BeautifulSoup(Path(epoch_runs.report_paths[0]).read_text(), "html.parser")
    rows = index.select(".runs-table tbody tr")
    assert [row.select("td")[-1].get_text() for row in rows] == ["Ran", "Ran"]
    assert [row.find("a")["href"] for row in rows] == [quote(n) for n in names[1:]]
    assert index.find("script") is None and "unsafe()" in index.get_text()
    for name in names[1:]:
        report = BeautifulSoup((tmp_path / name).read_text(), "html.parser")
        assert report.select_one(f'a[href="{quote(names[0])}"]') is not None
        assert len(report.select(".transcript")) == 1


def test_an_errored_epoch_lists_from_its_info_row_with_its_status(epoch_runs):
    """An epoch whose scanner errored, and so has no timeline rows, still
    lists from its transcript_info row with the status Errors recorded."""
    epoch_runs.scan_status.errors.append(ScanError("token_timeline", "run-1", "failed"))
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
        "Errors recorded",
    ]
    assert "epoch 2" in rows[1].find("a").get_text()


def test_a_transcript_without_an_epoch_is_refused(epoch_runs, tmp_path):
    """A multi-epoch render names the transcript_info gap instead of
    failing inside pandas."""
    epoch_runs.transcript_info.loc[0, "epoch"] = pd.NA
    with pytest.raises(ValueError, match="without an epoch"):
        render(epoch_runs, report_path=str(tmp_path / "r.html"), viewer=False)


@pytest.mark.parametrize(
    ("audit", "broken", "rows", "text"),
    [
        (("agent_span_id", "label"), False, 2, "cohort of"),
        (("agent_span_id", "label"), True, 1, "not audited"),
        (None, False, 0, ""),
    ],
    ids=["audited", "contract breach", "no audit declared"],
)
def test_custom_layers_join_the_collection_only_through_their_audit(
    epoch_runs, audit, broken, rows, text
):
    """A custom layer that declares an audit gets its rows, a failing one
    a stated reason, and one without an audit nothing."""
    frame = (
        epoch_runs.subagents.drop(columns="judge_regime")
        if broken
        else epoch_runs.subagents
    )
    epoch_runs.extra_layers = [Layer(name="custom_judge", frame=frame, audit=audit)]
    epoch_runs.layer_frames = {"custom_judge": frame}
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


def test_browser_follows_an_epoch_link_and_back(epoch_runs, tmp_path):
    """A real browser resolves the quoted file links from the index to an
    epoch report and back."""
    playwright = pytest.importorskip("playwright.sync_api")
    render(
        epoch_runs,
        report_path=str(tmp_path / "my runs.html"),
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
        page.locator("tbody a").nth(1).click()
        playwright.expect(page.locator(".transcript")).to_have_count(1)
        page.get_by_role("link", name="Run index").click()
        playwright.expect(page.locator(".runs-table")).to_be_visible()
        assert errors == []
        browser.close()
