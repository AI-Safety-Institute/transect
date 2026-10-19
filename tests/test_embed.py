"""Chart embedding: every chart section is a fragment of one page."""

from pathlib import Path

from transect import load, render


def test_report_carries_one_chart_runtime_and_no_iframes(tmp_path):
    """The report embeds the widget bundle once, with no per-section iframe."""
    scans = Path(__file__).parent / "fixtures" / "demo_scan"
    results = render(
        load(str(scans)),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(results.report_paths[0]).read_text()

    assert "<iframe" not in html
    assert html.count('id="iv-esm"') == 1
