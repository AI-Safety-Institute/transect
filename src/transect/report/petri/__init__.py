"""Native Petri branch diagrams and complete surviving-conversation reports.

The default renderer detects explicit auditor and target roles. Integrations
can also reuse build_config and the packaged assets with retained payloads.
Optional evidence and category annotations describe saved scanner results;
they never change the reconstructed conversation or underlying judgments.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from markupsafe import Markup

from transect.report.petri.grouping import add_continuations, group_turns
from transect.report.petri.path import presentation
from transect.report.petri.transcript import PetriData
from transect.report.petri.tree import build_diagram

ASSETS = Path(__file__).with_name("assets")
ASSET_NAMES = (
    "phase-transcripts.js",
    "phase-transcripts.css",
    "branch-diagram.js",
    "branch-diagram.css",
)


def build_config(
    payload: dict[str, Any],
    groups: list[dict[str, Any]],
    tree: dict[str, Any],
    *,
    index: int = 0,
    prefix: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a scanner-independent view; annotations locate saved evidence only.

    Evidence supports status, turns, quote, title, badge, description and href.
    Category annotations support title, labels (key/title pairs), turns (call
    number to category) and flagged (categories surfaced on folded attempts).
    Pass categories as config.categories or a script[data-petri-categories]
    inside the transcript. Both annotation types are optional. A target_label
    config override supplies a display name while retaining recorded model IDs.
    """
    path = presentation(payload, groups, tree)
    if prefix and path["status"] == "ok":
        for group in path["groups"]:
            group["card"] = prefix + group["card"]
        for entry in path["entries"]:
            if entry["kind"] == "phase":
                entry["card"] = prefix + entry["card"]
            else:
                entry["cards"] = [prefix + card for card in entry["cards"]]
    return {
        "groups": groups,
        "total": len(payload["turns"]),
        "path": path,
        "diagram": build_diagram(payload, groups, tree, evidence),
        "prefix": prefix,
        "container_id": f"phase-cards-{index}",
        "target_model": payload.get("target_model") or "Target model",
    }


def native_section(
    data: PetriData, phases: pd.DataFrame, index: int, *, multiple: bool
) -> Markup:
    """Embed complete pooled text as inert JSON, parsed only when a card opens."""
    prefix = f"petri-{index}-" if multiple else ""
    auditor = -1
    timeline = []
    for turn in data.payload["turns"]:
        if turn["role"] == "auditor":
            auditor += 1
        timeline.append(
            {
                "turn": auditor if turn["role"] == "auditor" else None,
                "agent_span_id": str(turn["attempt"])
                if turn["role"] == "target"
                else None,
            }
        )
    groups = group_turns(data.payload, pd.DataFrame(timeline), phases, index)
    # Unlabelled calls also need transcript-scoped anchors.
    for group in groups:
        if group["card"] == "unclassified-transcript":
            group["card"] = prefix + "unclassified-transcript"
    add_continuations(data.payload, groups)
    config = build_config(data.payload, groups, data.tree, index=index, prefix=prefix)
    config["payload_id"] = prefix + "petri-payload"
    config["branch_container_id"] = prefix + "branch-overview"
    config["source_note"] = data.note
    return Markup(
        '<script type="application/json" '
        f'id="{prefix}phase-transcript-config" data-petri-config>'
        + _encode(config)
        + "</script>"
        + f'<script type="application/json" id="{config["payload_id"]}">'
        + _encode(data.payload)
        + "</script>"
        + "<noscript><p>Enable JavaScript to browse the complete Petri conversation "
        "and branch diagram. Saved auditor phase assessments remain "
        "below.</p></noscript>"
    )


def assets_html() -> Markup:
    """Inline package assets once, so native reports need no separate server files."""
    styles = "\n".join(
        (ASSETS / name).read_text() for name in ASSET_NAMES if name.endswith(".css")
    )
    scripts = "\n".join(
        (ASSETS / name).read_text() for name in ASSET_NAMES if name.endswith(".js")
    )
    return Markup(
        "<!-- petri-assets --><style>"
        + styles
        + "</style><script>"
        + scripts
        + "</script><!-- /petri-assets -->"
    )


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")
