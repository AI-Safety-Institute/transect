"""Stored execution failures remain visible through the public API and report."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest
from inspect_ai.model import ChatMessageUser, Model, ModelOutput, get_model
from inspect_scout import Loader, Transcript, loader, scan_results_df, scanner

import transect
import transect.api as api
from transect.scanners import base, helpers
from transect.scanners.cohort import batch_item_content


@pytest.fixture(autouse=True)
def only_local_models(monkeypatch):
    """Integration tests may call scripted mock models only."""
    original = Model.generate

    async def generate(self, *args, **kwargs):
        assert self.api.__class__.__module__.startswith(
            "inspect_ai.model._providers.mockllm"
        )
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)


def _status_element(path):
    html = Path(path).read_text()
    marker = '<section id="scan-status"'
    start = html.index(marker)
    end = html.index("</section>", start) + len("</section>")
    return ElementTree.fromstring(html[start:end])


def _status_text(path):
    """The section's text with whitespace collapsed away, for equality
    across renders that differ only in element boundaries."""
    return "".join("".join(_status_element(path).itertext()).split())


def test_failed_phase_scan_remains_visible_after_reload(demo_log, tmp_path):
    """A failed requested judge stays distinct from a scan with judging disabled."""

    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic judge failure <script>example()</script>")

    result = transect.transect(
        str(demo_log),
        transect.Spec.model_validate({"phases": ["work"]}),
        judge_models=get_model("mockllm/model", custom_outputs=fail, memoize=False),
        verify=False,
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    assert result.scan_status.outer_complete is False
    assert result.scan_status.has_failures
    phase = next(
        s for s in result.scan_status.scanners if s.scanner == "decision_phases"
    )
    assert phase.errors == 1
    assert result.phases.empty
    assert 'see <a href="#scan-status">' in Path(result.report_paths[0]).read_text()
    element = _status_element(result.report_paths[0])
    readable = " ".join(element.itertext())
    assert "decision_phases" in readable
    assert "Synthetic judge failure" in readable
    assert element.find(".//script") is None
    text = _status_text(result.report_paths[0])
    restored = transect.load(result.scan_location)
    assert restored.scan_status == result.scan_status
    transect.render(
        restored,
        report_path=str(tmp_path / "reload.html"),
        viewer=False,
        open_report=False,
        section_order=["phase_cards"],
    )
    assert _status_text(restored.report_paths[0]) == text
    empty = replace(restored, token_timeline=restored.token_timeline.iloc[:0])
    transect.render(
        empty,
        report_path=str(tmp_path / "no-timeline.html"),
        viewer=False,
        open_report=False,
    )
    assert _status_text(empty.report_paths[0]) == text


def test_failed_eval_setup_surfaces_its_recorded_error_on_run_and_reload(
    demo_log, tmp_path, monkeypatch
):
    """A persisted setup failure reaches both API callers before frame projection."""

    def fail_header(uri):
        raise RuntimeError("recorded setup failure")

    monkeypatch.setattr(base, "_eval_header", fail_header)
    scans = tmp_path / "scans"
    with pytest.raises(RuntimeError) as initial:
        transect.transect(
            str(demo_log),
            transect.Spec(),
            scans_dir=str(scans),
            viewer=False,
            open_report=False,
        )

    location = next(scans.glob("scan_id=*"))
    raw = scan_results_df(str(location))
    (error,) = raw.errors
    assert error.scanner == "eval_setup"
    assert raw.scanners["eval_setup"].value.isna().all()
    assert raw.summary.scanners["eval_setup"].results == 0
    assert "eval_setup" in str(initial.value)
    assert error.transcript_id in str(initial.value)
    assert error.error in str(initial.value)
    with pytest.raises(RuntimeError) as restored:
        transect.load(str(location))
    assert str(restored.value) == str(initial.value)


@loader(messages="all")
def three_units() -> Loader[Transcript]:
    """Supply exactly three synthetic units, independent of transcript length."""

    async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
        yield Transcript(
            transcript_id=f"{transcript.transcript_id}:units",
            messages=[
                ChatMessageUser(content=batch_item_content(["one", "two", "three"]))
            ],
            metadata={"items": [{"unit": 1}, {"unit": 2}, {"unit": 3}]},
        )

    return load


@scanner(loader=three_units())
def partial_batch(judge_models=None):
    """Use the shipped judged helper with a fixed three-unit loader."""
    return transect.cohort_llm_scanner(
        question="Choose a category",
        answer=["a", "b"],
        models=judge_models,
        batch=True,
        verify=False,
    )


def test_clean_execution_with_partial_labels_raises_no_alarm(demo_log, tmp_path):
    """Judgement-quality gaps are the audit's territory, not execution failures."""

    def answer(*args, **kwargs):
        return ModelOutput.for_tool_call(
            "mockllm/model",
            "answer",
            {
                "items": [
                    {
                        "item": 1,
                        "label": "a",
                        "confidence": 0.9,
                        "explanation": "first unit only",
                    }
                ],
                "explanation": "partial",
            },
        )

    layer = transect.Layer(name="custom_units", scanner=partial_batch)
    result = transect.transect(
        str(demo_log),
        transect.Spec(),
        judge_models=get_model("mockllm/model", custom_outputs=answer, memoize=False),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[layer],
    )
    assert result.scan_status.outer_complete is True
    status = next(
        s for s in result.scan_status.scanners if s.scanner == "partial_batch"
    )
    assert status.scanned_transcripts == 1 and status.missing_scans == 0
    assert not result.scan_status.has_failures
    assert 'see <a href="#scan-status">' not in Path(result.report_paths[0]).read_text()
    # Reload without the custom declaration still reports its execution.
    restored = transect.load(result.scan_location)
    unmounted = next(
        s for s in restored.scan_status.scanners if s.scanner == "partial_batch"
    )
    assert unmounted.mounted is False
    transect.render(
        restored,
        report_path=str(tmp_path / "unmounted.html"),
        viewer=False,
        open_report=False,
    )
    assert "not mounted" in " ".join(
        _status_element(restored.report_paths[0]).itertext()
    )


def test_status_load_preserves_custom_frame_columns(monkeypatch):
    """Custom frame functions get the scanner value and usage columns, never
    Scout's heavy columns (the item input, its pool, the scan's event log)."""
    read = api.scan_results_df
    seen = []

    def capture(*args, **kwargs):
        raw = read(*args, **kwargs)
        for table in raw.scanners.values():
            assert not {"input", "input_data", "scan_events"} & set(table.columns)
            assert {"value", "scan_model_usage"} <= set(table.columns)
        seen.append(True)
        return raw

    monkeypatch.setattr(api, "scan_results_df", capture)
    path = Path(__file__).parent / "fixtures" / "demo_scan"
    result = transect.load(str(path))
    assert seen
    assert result.scan_status.scanners


def test_an_unresolvable_orchestrator_lane_surfaces_as_a_scan_error(
    demo_log, tmp_path, monkeypatch
):
    """A transcript with no single orchestrator lane errors per scanner,
    is named in the scan status section, and never renders a guessed
    axis."""

    def ambiguous(transcript):
        raise ValueError(
            "transcript has 2 top-level agents ('one', 'two') and no single "
            "orchestrator lane; transect numbers one orchestrator's turns"
        )

    monkeypatch.setattr(helpers, "main_span", ambiguous)
    result = transect.transect(
        str(demo_log),
        transect.Spec(),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    assert result.scan_status.has_failures
    timeline = next(
        s for s in result.scan_status.scanners if s.scanner == "token_timeline"
    )
    assert timeline.errors == 1
    assert result.token_timeline.empty
    readable = " ".join(_status_element(result.report_paths[0]).itertext())
    assert "no single orchestrator lane" in readable
    assert "'one', 'two'" in readable
