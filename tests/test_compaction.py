"""Inspect compaction text stays attributable from a real log to the stored report."""

import pandas as pd
import pytest
from fixtures.generate_eval_log import (
    COMPACTION_INSTRUCTIONS,
    COMPACTION_THRESHOLD,
    NUDGE_PREFIX,
)
from helpers import StubTranscript, model_turn, run_item
from inspect_ai.event import CompactionEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from inspect_scout import Transcript
from test_frames import raw_row
from test_report_integration import _assert_no_page_errors

from transect import load, render
from transect.api import _run
from transect.frames.flushes import flushes_df
from transect.frames.transcript_info import transcript_info_df
from transect.report.sections import compaction_threshold, eval_setup_blocks
from transect.scanners.base import context_flush, eval_setup
from transect.spec import Spec


def test_recorded_prompt_and_nudge_survive_to_the_stored_report(fixture_logs, tmp_path):
    """Every summary flush carries the formatted prompt, nudges attach to the flush
    they preceded, and the cards render from the store without the source log."""
    scans = tmp_path / "scan"
    results = _run(
        logs=str(fixture_logs),
        sample="fixture-sample-1",
        spec=Spec(),
        scans_dir=str(scans),
    )
    flushes = results.flushes
    assert len(flushes) >= 2
    assert (flushes.strategy == "CompactionSummary").all()
    assert (flushes.trigger == "threshold").all()
    prompts = flushes.compaction_prompt
    assert prompts.notna().all()
    # instructions are substituted into the recorded prompt, the template
    # placeholder is not
    assert prompts.str.contains(COMPACTION_INSTRUCTIONS, regex=False).all()
    assert not prompts.str.contains("{addendums}", regex=False).any()
    assert "attachment://" not in "".join(prompts)
    template = results.transcript_info.compaction_prompt.iloc[0]
    assert "{addendums}" in template
    # flushes.turn is the first post-flush turn on the 0-based axis, so the
    # summarization call (whose input ends with the prompt) is the turn before
    log = read_eval_log(next(fixture_logs.glob("*.eval")), resolve_attachments=True)
    model_turns = [e for e in log.samples[0].events if e.event == "model" and e.output]
    for flush in flushes.itertuples():
        assert model_turns[flush.turn - 1].input[-1].text == flush.compaction_prompt
    # identical nudges share one pooled message id in the log, so each flush
    # must still get the warning issued in its own window
    nudges = flushes.compaction_nudge
    assert nudges.notna().all()
    assert nudges.str.startswith(NUDGE_PREFIX).all()

    reloaded = load(str(scans))
    pd.testing.assert_frame_equal(flushes, reloaded.flushes)
    reloaded.transcripts_location = None
    report = tmp_path / "report.html"
    render(reloaded, report_path=str(report), viewer=False, open_report=False)
    html = report.read_text()
    assert html.count("<summary>Compaction prompt (verbatim)</summary>") == 1
    assert "Keep &lt;paths&gt; &amp; decisions." in html  # instructions, escaped
    assert "Compaction prompt (configured template)" not in html
    assert html.index("Task message (verbatim") < html.index("Compaction prompt")
    _assert_no_page_errors(
        str(report),
        min_frames=2,
        setup_prompts={
            "Compaction prompt (verbatim)": prompts.iloc[0],
            "Compaction nudge (before compaction)": nudges.iloc[0],
        },
        compaction_threshold=COMPACTION_THRESHOLD,
    )


@pytest.mark.parametrize("case", ["openclaw", "history_message", "no_summary_call"])
def test_no_text_is_inferred_where_none_was_recorded(case):
    """Other sources, a history message before the flush, and non-summary
    compactions all leave the text columns None."""
    task = ChatMessageUser(content="Do the thing", source="input")
    flush = CompactionEvent(type="summary", source="inspect", span_id="lead")
    if case == "no_summary_call":
        flush.type = "edit"
    transcript = StubTranscript(
        [model_turn("working", span_id="lead", input=[task]), flush], messages=[task]
    )
    if case == "openclaw":
        transcript.source_type = "openclaw"
    (row,) = run_item(context_flush(), transcript).value["flushes"]
    assert (row["compaction_prompt"], row["compaction_nudge"]) == (None, None)


@pytest.mark.parametrize("source_type", ["eval_log", "openclaw"])
def test_configured_compaction_template_is_not_a_runtime_default(source_type):
    """An explicit Inspect template supplies the card when observed text is absent."""
    prompt = "Summarize.\nKeep <constraints>.\n{addendums}"
    for args, expected in (({}, None), ({"compaction": {"prompt": prompt}}, prompt)):
        result = run_item(
            eval_setup(),
            Transcript(transcript_id="t", source_type=source_type, agent_args=args),
        )
        assert result.value["compaction_prompt"] == (
            expected if source_type == "eval_log" else None
        )
        raw = {**raw_row(result), "transcript_source_type": source_type}
        info = transcript_info_df(pd.DataFrame([raw]))
        flushes = flushes_df(pd.DataFrame(), pd.DataFrame(columns=["transcript_id"]))
        html = str(eval_setup_blocks(info, flushes))
        assert ("Compaction prompt (configured template)" in html) == (
            source_type == "eval_log" and expected is not None
        )


@pytest.mark.parametrize(
    "setting, source_type, label, tokens",
    [
        ({"threshold": 120000}, "eval_log", "120,000 tokens", 120000),
        (
            {"threshold": 0.9},
            "eval_log",
            "90% of context window (token count not recorded)",
            None,
        ),
        (None, "eval_log", None, None),
        ({"threshold": "120000"}, "eval_log", None, None),
        ({"threshold": 120000}, "openclaw", None, None),
    ],
)
def test_compaction_threshold_uses_only_the_recorded_setting(
    setting, source_type, label, tokens
):
    """Stored settings retain their units; missing facts never become defaults."""
    result = run_item(
        eval_setup(),
        Transcript(
            transcript_id="t",
            source_type=source_type,
            agent_args={"compaction": setting},
        ),
    )
    info = transcript_info_df(
        pd.DataFrame([{**raw_row(result), "transcript_source_type": source_type}])
    )
    found = compaction_threshold(info)
    if label is None:
        assert found is None
    else:
        assert found is not None
        assert (found.label, found.tokens) == (label, tokens)
    flushes = flushes_df(pd.DataFrame(), pd.DataFrame(columns=["transcript_id"]))
    html = str(eval_setup_blocks(info, flushes))
    assert ("compaction threshold</span>" in html) == (label is not None)
    if label is not None:
        assert label in html
