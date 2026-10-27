"""A simple run index followed by collection reliability and execution coverage."""

from urllib.parse import quote

from markupsafe import Markup

from transect.frames import TransectResults
from transect.report import sections
from transect.report._jinja import jinja_env
from transect.report.collection import collection_reliability
from transect.report.colors import ramp_anchors
from transect.report.render import (
    _brand_asset,
    _mine,
    _package_version,
    _transcript_title,
)
from transect.report.style import _STYLE


def render_index(
    results: TransectResults,
    reports: list[tuple[str, TransectResults]],
    title: str,
    sensitivity: str | None = None,
) -> str:
    """List all report runs without transcript reads or new judge requests.

    Report names are relative filenames next to the index. Run status reflects
    recorded scan execution, not label quality or the source eval score.
    Collection reliability and coverage follow the report links.
    """
    entries = []
    for filename, view in reports:
        timeline = view.token_timeline
        ordered = (
            timeline[["agent", "sample_id", "epoch", "transcript_id"]]
            .drop_duplicates()
            .sort_values(["agent", "sample_id", "epoch"])
            .transcript_id.tolist()
        )
        identities = list(
            dict.fromkeys([*ordered, *view.transcript_info.transcript_id])
        )
        for tid in identities:
            info = _mine(view.transcript_info, tid)
            one = _mine(timeline, tid)
            position = ordered.index(tid) if tid in ordered else None
            href = quote(filename) + (
                f"#transcript-{position}" if position is not None else "#scan-status"
            )
            errors = [
                e.message for e in results.scan_status.errors if e.transcript_id == tid
            ]
            failed = bool(errors) or info.scan_error.notna().any()
            ran = len(one) > 0 or len(info) > 0
            entries.append(
                {
                    "heading": _transcript_title(one if len(one) else info, info),
                    "href": href,
                    "status": "Errors recorded"
                    if failed
                    else "Ran"
                    if ran
                    else "No scan data",
                }
            )
    low, mid, high = ramp_anchors()
    return (
        jinja_env()
        .get_template("index.html.j2")
        .render(
            title=title,
            transcripts=entries,
            n_transcripts=len(entries),
            collection=collection_reliability(results),
            ramp={"low": low, "mid": mid, "high": high},
            index_page=True,
            scan_location=results.scan_location,
            scan_status_view=sections.scan_status_view(results.scan_status),
            style=Markup(_STYLE),
            sensitivity=sensitivity,
            logo_uri=_brand_asset("logo.svg", "image/svg+xml"),
            favicon_uri=_brand_asset("favicon-32.png", "image/png"),
            package_version=_package_version(),
        )
    )
