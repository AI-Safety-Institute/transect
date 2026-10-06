"""Report integration: render real scans of differently-shaped logs
and check the HTML comes out whole."""

import html as html_mod
from pathlib import Path

import pandas as pd
import pytest
from inspect_ai.log import read_eval_log, write_eval_log

import transect
from transect import load, render
from transect.api import _run
from transect.frames.flushes import flushes_df
from transect.report import charts, sections
from transect.report.embed import (
    _TIP_EXTRA_LINE_PX,
    _TIP_FIT_MARGIN,
    _TIP_SHORT_ROW_PX,
    _tip_floor,
    wrap_row_px,
)
from transect.report.lanes_layout import SpanGeometry
from transect.spec import Spec

SCENARIOS = {
    "demo-with-subagents": (
        "examples/logs/house_price_demo.eval",
        {},
        ["Eval setup", "Token telemetry", "Sub-agent activity", "Human interventions"],
    ),
    "plain-single-agent": (
        "tests/fixtures/logs",
        {"sample": "fixture-sample-1"},
        ["Eval setup", "Token telemetry"],
    ),
    "parallel-subagents": (
        "tests/fixtures/parallel_logs",
        {},
        ["Eval setup", "Token telemetry", "Sub-agent activity"],
    ),
    "openclaw-import": (
        "tests/fixtures/openclaw/mini_telemetry.jsonl",
        {},
        # no .eval header exists on an OpenClaw import: the setup
        # section renders its honest absences
        ["Eval setup", "not recorded by source", "Token telemetry"],
    ),
}

_STORE_CONTENT = {
    # each store's planted signals, as rendered copy; flags counted
    # twice (once inline in the entity's own section, once in the audit)
    "demo_scan": (
        ("k-roll self-consistency (mean per-turn agreement) = 0.67", 2),
        ("Flagged above 0.95", 2),
        ("verifier spot-check overturns = 1 of 2 sampled", 2),
        ("verifier re-label rate (model_development) = 100%", 2),
        ("Verifier selection</span>: 2 random sample", 1),
        ("Member coverage", 1),
        # the audit groups flags under the entity sub-heading
        ("Phase segmentation and labelling", 1),
        # the joined per-member ballots tooltip channel
        ("member votes", 1),
        # the flagged class-box with the pre-overturn label + confidence
        ("class-box-flagged", 1),
        ("verifier overturned (was model_development (0.90))", 1),
        # eval-setup absence wordings (header read; scaffold args recorded)
        ("scaffold default", 1),
        ("not set", 1),
        ("verifier: mockllm/model", 1),
        # the audit's per-classification maps
        ("Reliability map", 1),
        ("Provenance map", 1),
        ("appears as minority vote", 1),
        ("Chance-corrected self-consistency", 1),
        ("Labels (N = decided turns)", 2),
        ("What these rows mean", 2),
        ("Flags (overall scanner assessment)", 1),
    ),
    "demo_scan_cohort": (
        # the billed judge usage table: the scanner cell spans its three models
        ("Billed model usage", 1),
        ('<th rowspan="3">decision_phases</th>', 1),
        ("Gwet's AC1", 1),
        ("Flagged below 0.66", 2),
        ("Flagged between 0.66 and 0.80", 2),
        ("Sub-agent labelling", 1),
        ("member votes", 1),
        ("Reliability map", 1),
        ("Provenance map", 1),
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_mechanical_report_renders_whole(name, tmp_path):
    """A $0 scan of this log shape renders a report with its expected
    sections and no leaked error text."""
    logs, kwargs, sections = SCENARIOS[name]
    root = Path(__file__).parents[1]
    results = _run(
        logs=str(root / logs), spec=Spec(), scans_dir=str(tmp_path / "s"), **kwargs
    )
    results = render(
        results,
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(results.report_paths[0]).read_text()
    for section in sections:
        assert section in html
    # an OpenClaw import records no compaction configuration at all; the
    # parallel fixture is an Inspect log run with compaction disabled
    recorded = name not in ("openclaw-import", "parallel-subagents")
    if recorded:
        assert "Compaction nudge (before compaction)" in html
    else:
        assert "Compaction nudge" not in html
    assert "Traceback" not in html
    # the other Inspect logs record a compaction threshold (row + toggle)
    assert ("compaction threshold</span>" in html) == recorded
    assert ("Compaction threshold:" in html) == recorded
    assert len(html) > 20_000


@pytest.mark.parametrize("store", ["demo_scan", "demo_scan_cohort"])
def test_judged_report_renders_all_sections_from_a_stored_scan(store, tmp_path):
    """The committed judged demo scans (k-roll + verifier; dissenting
    cohort) render every report section, the audit, and each store's
    planted reliability signals - inline and in the audit."""

    scans = Path(__file__).parent / "fixtures" / store
    if not scans.exists():
        pytest.fail(
            "committed demo scan missing - regenerate: "
            "python tests/fixtures/generate_demo_scan.py"
        )
    results = render(
        load(str(scans)),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(results.report_paths[0]).read_text()
    for section in (
        "Eval setup",
        "Phase timeline",
        "Token telemetry",
        "Human interventions",
        "Sub-agent activity",
        "Token spend",
        "Phase cards",
        "Reliability",
    ):
        assert section in html
    assert "Traceback" not in html
    text = html_mod.unescape(html)
    for needle, at_least in _STORE_CONTENT[store]:
        assert text.count(needle) >= at_least, f"{needle!r} x{at_least} missing"
    # reading order: interventions sit directly below the phase timeline
    assert (
        text.index("Phase timeline")
        < text.index("Human interventions")
        < text.index("Token telemetry")
    )
    _assert_no_page_errors(results.report_paths[0])


def _assert_no_page_errors(
    report_path: str,
    min_frames: int = 6,
    setup_prompts: dict[str, str] | None = None,
    compaction_threshold: int | None = None,
) -> None:
    """Load the report in a real browser and require zero page errors
    (inspect-viz widget failures are console-only and blank charts
    silently). Skips without playwright or without the CDN."""
    playwright = pytest.importorskip("playwright.sync_api")
    errors: list[str] = []
    cdn_failures: list[str] = []
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as err:
            # an environment failure, not a content failure
            pytest.skip(f"browser unavailable: {err}")
        page = browser.new_page()
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on(
            "requestfailed",
            lambda request: (
                cdn_failures.append(request.url) if "cdn" in request.url else None
            ),
        )
        page.goto(Path(report_path).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        if setup_prompts:
            page.get_by_text("Core setup", exact=True).click()
            for label, text in setup_prompts.items():
                summary = page.locator("summary").filter(has_text=label)
                summary.click()
                content = summary.locator("..").locator(".prompt-verbatim")
                assert content.is_visible()
                assert content.inner_text() == text
        if compaction_threshold is not None:
            label = f"Compaction threshold: {compaction_threshold:,} tokens (dotted)"
            frame = next(f for f in page.frames if f.get_by_label(label).count())
            control = frame.get_by_label(label)
            rule = frame.locator('[stroke="#9467bd"][stroke-dasharray="2,3"] line')
            curve = frame.locator('[aria-label="line"][stroke="#4c78a8"]')
            playwright.expect(control).to_be_checked()
            playwright.expect(rule).to_have_count(1)
            playwright.expect(curve).to_have_count(1)
            # the curve stays drawn either way; its scale may follow the rule
            control.uncheck()
            playwright.expect(rule).to_have_count(0)
            playwright.expect(curve.locator("path")).to_have_count(1)
            control.check()
            playwright.expect(rule).to_have_count(1)
        n_frames = len(page.frames)
        browser.close()
    if cdn_failures:
        pytest.skip(f"inspect-viz CDN unreachable: {cdn_failures[0]}")
    assert errors == []
    assert n_frames >= min_frames


def test_recorded_compaction_threshold_survives_replay_and_toggles(demo_log, tmp_path):
    """A saved absolute threshold renders in Core setup and toggles a dotted rule."""
    log = read_eval_log(str(demo_log))
    # a threshold above the demo's context peak: the rule then sits clear of
    # the curve, so hiding it visibly changes the chart
    log.plan.steps[-1].params["compaction"] = {"type": "summary", "threshold": 4000}
    path = tmp_path / "threshold.eval"
    write_eval_log(log, str(path))
    scans = tmp_path / "scan"
    _run(logs=str(path), spec=Spec(), scans_dir=str(scans))
    path.unlink()
    results = load(str(scans))
    results.transcripts_location = None
    report = tmp_path / "report.html"
    render(results, report_path=str(report), viewer=False, open_report=False)
    html = report.read_text()
    assert "compaction threshold</span>" in html
    assert "4,000 tokens" in html
    assert "dotted purple = configured compaction threshold (4,000 tokens)" in html
    _assert_no_page_errors(str(report), min_frames=2, compaction_threshold=4000)


def test_tip_floor_budgets_one_or_two_wrapping_rows():
    """The tooltip-fit floor prices the single-line rows plus each
    wrapping row's own allowance; wrap_row_px caps at four extras."""

    assert _tip_floor(0, 40) == _TIP_FIT_MARGIN
    assert _tip_floor(5, 40) == 4 * _TIP_SHORT_ROW_PX + 40 + _TIP_FIT_MARGIN
    assert _tip_floor(5, 40, 30) == 3 * _TIP_SHORT_ROW_PX + 40 + 30 + _TIP_FIT_MARGIN
    assert wrap_row_px(10) == _TIP_SHORT_ROW_PX
    assert wrap_row_px(61) == _TIP_SHORT_ROW_PX + 2 * _TIP_EXTRA_LINE_PX
    assert wrap_row_px(10_000) == _TIP_SHORT_ROW_PX + 4 * _TIP_EXTRA_LINE_PX


def test_spend_bars_floor_covers_the_declared_tooltip_rows():
    """The spend chart's iframe floor follows its declared channels, so
    a short chart cannot clip its own tooltip."""

    _, height = charts.spend_bars(
        [
            {
                "label": "x",
                "value": 10,
                "color": "#123456",
                "share": None,
                "output": "5",
                "billable": "7",
            }
        ]
    )
    # single-line rows plus the bucket label's two-line allowance
    assert height >= _tip_floor(5, _TIP_SHORT_ROW_PX + 2 * _TIP_EXTRA_LINE_PX)


def test_interventions_chart_budgets_its_previews_not_a_generic_two_line_row():
    """The interventions iframe is floored at its own five rows, the two
    message previews priced as 60-character prose, so the chart no
    longer carries a hundred blank pixels under a 94px strip."""

    act = pd.DataFrame(
        {
            "turn": [3, 7],
            "channel": ["approval", "operator_message"],
            "outcome": ["approved", None],
            "prompt": ["may I delete the cache? " * 8, None],
            "content": ["yes, but keep the model artefacts " * 6, "stop and report"],
        }
    )
    _, height = charts.interventions_chart(act, n_turns=10)
    preview_px = wrap_row_px(charts._PREVIEW_CHARS)
    assert height == _tip_floor(5, preview_px, preview_px)
    assert len(charts._preview("x" * 100)) == charts._PREVIEW_CHARS + 1


def empty_flushes() -> pd.DataFrame:
    return flushes_df(pd.DataFrame(), pd.DataFrame(columns=["transcript_id"]))


def test_eval_setup_renders_container_values():
    """Container values in the bypass fields render as compact JSON
    (escaped); an empty container reads as unconfigured, not []."""
    info = pd.DataFrame(
        [
            {
                "header_available": True,
                "scaffold_prompt": ["<b>x</b>", "a & b"],
                "sandbox": ["docker", "compose.yaml"],
                "tools": [],
            }
        ],
        dtype=object,
    )
    html = str(sections.eval_setup_blocks(info, empty_flushes()))
    text = html_mod.unescape(html)
    assert '["<b>x</b>", "a & b"]' in text
    assert '["docker", "compose.yaml"]' in text
    assert "<b>x</b>" not in html  # escaped, not markup
    assert "[]" not in text


def test_run_intro_preserves_recorded_values():
    """The deterministic intro keeps recorded values verbatim (a score
    of "C" never lowercases) and picks the article by the agent name."""

    info = pd.DataFrame(
        [
            {
                "model": "m1",
                "task_name": "t1",
                "sample_id": "s1",
                "epoch": 2,
                "agent": "openclaw",
                "date": "2026-06-18T10:00:00",
                "message_count": 10,
                "wallclock_seconds": 61.0,
                "total_tokens": 1234,
                "score": "C",
                "success": None,
                "error": None,
                "limit": "message",
            }
        ]
    )
    text = str(sections.run_intro_line(info))
    assert "under an openclaw scaffold" in text
    assert "Scored C; ended at the message limit." in text


def test_custom_layer_section_renders_and_passes_the_browser(layered_run, tmp_path):
    """The layered $0 run's report carries the badge-marked custom
    section with its markdown, chart, and band."""

    results, _ = layered_run
    path = str(tmp_path / "layered.html")
    transect.render(results, report_path=path, viewer=False, open_report=False)
    html = open(path).read()
    assert "custom-layer-badge" in html
    assert "turn_chars" in html
    assert "an e2e fixture layer" in html  # markdown rendered
    # the chart+band embed sits inside the custom section itself (a
    # page-level iframe count would pass on the built-ins' alone)
    section_start = html.index('<h3>turn_chars <span class="custom-layer-badge">')
    section_end = html.index("<h3>", section_start + 1)
    assert html[section_start:section_end].count("<iframe") >= 1
    # the $0 mechanical report has fewer sections than the judged one
    # the default floor is calibrated for: main frame + 3 built-in
    # chart iframes + the custom section's one
    _assert_no_page_errors(path, min_frames=5)


def test_tag_chips_and_selectors_ride_the_phase_cards(tmp_path):
    """A tags layer replayed over the committed judged store: each card
    carries only its own turn range's provenance-marked family=value
    chips, and the control bar gains one filter selector per family."""

    scans = Path(__file__).parent / "fixtures" / "demo_scan"
    per_turn = pd.DataFrame({"turn": [0, 1, 7], "quality": ["good", "good", "poor"]})
    results = render(
        load(
            str(scans),
            extra_layers=[transect.Layer(name="review", frame=per_turn, tags=True)],
        ),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
    )
    html = Path(results.report_paths[0]).read_text()
    # the store's two phases span orchestrator turns 0-4 and 5-9: good
    # lands only on the first card, poor only on the second
    assert html.count('data-tags="|quality=good|"') == 1
    assert html.count('data-tags="|quality=poor|"') == 1
    assert html.count("user-chip") >= 2
    # the provenance title names the declaring layer (a bare "review"
    # match would be satisfied by the demo's reviewer sub-agent)
    assert "from layer 'review'" in html
    assert html.count('data-tag-family="quality"') == 1
    assert "By custom tag family" in html
    assert "quality (review)" in html
    assert html.count('data-spend-family="quality"') == 1


def test_section_order_rearranges_sections_and_rejects_unknown_keys(tmp_path):
    """section_order moves the named sections to the front in the given
    order, everything else follows in default order, and the audit
    stays last; an unknown key is refused naming the valid set."""

    scans = Path(__file__).parent / "fixtures" / "demo_scan"
    results = render(
        load(str(scans)),
        report_path=str(tmp_path / "report.html"),
        viewer=False,
        open_report=False,
        section_order=["token_spend", "phase_cards"],
    )
    html = Path(results.report_paths[0]).read_text()
    positions = [
        html.index(f"<h3>{title}</h3>")
        for title in ("Token spend", "Phase cards", "Eval setup", "Phase timeline")
    ]
    assert positions == sorted(positions)
    assert html.rindex("<h3>Reliability") > max(positions)
    with pytest.raises(ValueError, match="unknown section"):
        render(
            results,
            report_path=str(tmp_path / "r2.html"),
            viewer=False,
            open_report=False,
            section_order=["token_spendd"],
        )
    with pytest.raises(ValueError, match="repeats"):
        render(
            results,
            report_path=str(tmp_path / "r3.html"),
            viewer=False,
            open_report=False,
            section_order=["token_spend", "token_spend"],
        )
    # the same check guards the transect() entry point pre-spend: it fires
    # before the logs are even touched (the path does not exist)
    with pytest.raises(ValueError, match="unknown section"):
        transect.transect(
            "does-not-exist",
            Spec(),
            viewer=False,
            open_report=False,
            section_order=["nope"],
        )


def test_markdown_block_escapes_raw_html():
    """A Markdown block's text may interpolate transcript-derived
    strings, so raw HTML must land escaped, never live."""
    from transect.report import custom
    from transect.report.blocks import Markdown

    ctx = custom.SectionContext(transcript_id="t", frame=None, n_turns=0)
    html = str(custom._markdown(Markdown("<script>x</script> *ok*"), ctx))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "<em>ok</em>" in html


def test_tag_spend_bars_splits_new_work_with_an_untagged_bucket():
    """Tagged spend sums per label, the remainder lands in the
    untagged bucket, and shares stay against the full total; the
    empty cases return no bars rather than a zero chart."""
    one = pd.DataFrame({"turn": [0, 1, 2], "new_work": [100, 200, 300]})
    tags = pd.DataFrame({"turn": [0, 1], "skill": ["a", "a"]})
    bars = sections.tag_spend_bars(one, tags, "skill")
    assert {b["label"]: b["value"] for b in bars} == {"a": 300, "untagged": 300}
    assert all(b["share"] == "50% of new work" for b in bars)
    assert sections.tag_spend_bars(one, None, "skill") == []
    assert sections.tag_spend_bars(one, tags, "nope") == []


def test_layer_audit_block_skips_quietly_when_the_judge_never_ran():
    """An audit-declaring layer whose judged columns exist but whose
    judge never stamped a regime contributes no entity block; with no
    built-in judged surface either, the honest one-liner renders."""
    empty = pd.DataFrame()
    subagents = pd.DataFrame({"status": []})
    frame = pd.DataFrame({"turn": [0], "label": [None], "judge_regime": [None]})
    out = str(
        sections.reliability_audit(
            empty,
            empty,
            subagents,
            empty,
            None,
            phase_turn_votes=empty,
            layer_audits=[
                {"name": "x", "frame": frame, "unit_col": "turn", "label_col": "label"}
            ],
        )
    )
    assert "No judged surfaces" in out
    assert "x labelling" not in out


def test_structural_only_subagents_raise_no_unjudged_flag():
    """Spans a judge never saw are not flagged as an unjudged share."""
    frames = load(str(Path(__file__).parent / "fixtures" / "demo_scan")).frames()
    subagents = frames["subagents"].copy()
    for column in (
        "label",
        "confidence",
        "judge_regime",
        "judge_models",
        "n_models",
        "k_rolls",
        "verifier_armed",
        "verifier_same_model",
        "verifier_model",
    ):
        subagents[column] = None
    votes = frames["subagent_votes"].iloc[:0]
    assert sections.subagent_reliability_flags(subagents, votes) == []


def test_layer_audit_block_names_an_unauditable_frame_instead_of_crashing():
    """An arbitrary user frame that stamps a judge regime but lacks the
    audit's columns renders an honest not-audited block, never a crash."""
    empty = pd.DataFrame()
    subagents = pd.DataFrame({"status": []})
    frame = pd.DataFrame({"turn": [0], "judge_regime": ["solo"]})
    out = str(
        sections.reliability_audit(
            empty,
            empty,
            subagents,
            empty,
            None,
            phase_turn_votes=empty,
            layer_audits=[
                {"name": "x", "frame": frame, "unit_col": "turn", "label_col": "label"}
            ],
        )
    )
    assert "could not be audited" in out
    assert "KeyError" in out


@pytest.mark.parametrize(
    ("row", "needles"),
    [
        (  # human-initiated: the message alone
            {
                "channel": "operator",
                "initiator": "human",
                "content": "stop",
                "prompt": None,
                "outcome": None,
            },
            ["message (human):</span> stop"],
        ),
        (  # ask_user: question, answer, outcome on the header line
            {
                "channel": "input_event",
                "initiator": "agent",
                "content": "confirm: yes",
                "prompt": "Submit?",
                "outcome": "accepted",
            },
            [
                "· accepted",
                "asked (agent):</span> Submit?",
                "answered (human):</span> confirm: yes",
            ],
        ),
        (  # approval with no explanation: an explicit absence, not a blank
            {
                "channel": "approval",
                "initiator": "agent",
                "content": "",
                "prompt": "bash({})",
                "outcome": "reject",
            },
            ["asked (agent):</span> bash({})", "no answer recorded"],
        ),
        (  # console recording: agent-initiated, no separate question
            {
                "channel": "input_event",
                "initiator": "agent",
                "content": "y",
                "prompt": None,
                "outcome": None,
            },
            ["recorded (human):</span> y"],
        ),
        (  # a store scanned before the new columns existed
            {
                "channel": "input_event",
                "initiator": None,
                "content": "y",
                "prompt": None,
                "outcome": None,
            },
            ["recorded (human):</span> y"],
        ),
    ],
)
def test_intervention_list_labels_each_shape_by_its_initiator(row, needles):
    frame = pd.DataFrame([{"turn": 3, **row}])
    html = html_mod.unescape(str(sections.intervention_line(frame)))
    for needle in needles:
        assert needle in html, needle
    assert "message (human)" not in html or row["initiator"] == "human"


def test_span_tooltip_says_when_a_span_began_before_the_first_turn():
    """A span clamped at the axis's left edge does not claim turn 0 was
    active while it ran."""
    span = pd.DataFrame(
        {
            "agent_span_id": ["A"],
            "agent_lane": ["early"],
            "turn_source": ["timestamp"],
            "spawn_turn": [0],
            "anchor_turn": [0],
            "end_turn": [0],
            "label": [None],
            "label_source": [None],
            "confidence": [None],
            "judge_agreement": [None],
            "n_voting": [None],
            "n_members": [None],
            "verifier_selected": [False],
            "overturned": [False],
            "verifier_label": [None],
            "verifier_confidence": [None],
            "verifier_status": [None],
            "tool_calls": [1],
            "busy_seconds": [None],
        }
    )
    geometry = {"A": SpanGeometry("A", "early", -0.5, -0.5, True, before_first=True)}
    titles = sections.span_titles(span, {}, geometry)
    assert "began before the first orchestrator turn" in titles["A"]["turns"]
