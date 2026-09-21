"""Stored execution failures remain visible through the public API and report."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest
from inspect_ai.model import ChatMessageUser, Model, ModelOutput, get_model
from inspect_scout import Loader, Transcript, loader, scanner

import transect
import transect.api as api
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
    assert phase.execution_errors == 1
    assert result.phases.empty
    element = _status_element(result.report_paths[0])
    text = " ".join(element.itertext())
    assert "Partial analysis" in text
    assert "decision_phases" in text
    assert "Synthetic judge failure" in text
    assert element.find(".//script") is None
    restored = transect.load(result.scan_location)
    assert restored.scan_status == result.scan_status
    transect.render(
        restored,
        report_path=str(tmp_path / "reload.html"),
        viewer=False,
        open_report=False,
        section_order=["phase_cards"],
    )
    assert " ".join(_status_element(restored.report_paths[0]).itertext()) == text
    empty = replace(restored, token_timeline=restored.token_timeline.iloc[:0])
    transect.render(
        empty,
        report_path=str(tmp_path / "no-timeline.html"),
        viewer=False,
        open_report=False,
    )
    assert " ".join(_status_element(empty.report_paths[0]).itertext()) == text


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


@pytest.mark.parametrize("all_fail", [False, True])
def test_batch_coverage_does_not_trust_outer_completion(demo_log, tmp_path, all_fail):
    """Nested unanswered or errored units stay visible even after a completed scan."""

    def answer(*args, **kwargs):
        if all_fail:
            raise RuntimeError("Synthetic batch failure")
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
    assert status.observed_units == 3
    assert status.usable_units == (0 if all_fail else 1)
    assert status.failed_units == (3 if all_fail else 0)
    assert status.missing_units == (0 if all_fail else 2)
    assert result.scan_status.has_failures
    text = " ".join(_status_element(result.report_paths[0]).itertext())
    assert "Partial analysis" in text
    assert "partial_batch" in text
    # Reload without the custom declaration still reports its execution and coverage.
    restored = transect.load(result.scan_location)
    unmounted = next(
        s for s in restored.scan_status.scanners if s.scanner == "partial_batch"
    )
    assert unmounted.mounted is False
    assert unmounted.usable_units == status.usable_units
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
    """Status projection keeps the raw columns available to custom frame functions."""
    read = api.scan_results_df
    seen = []

    def capture(*args, **kwargs):
        raw = read(*args, **kwargs)
        for table in raw.scanners.values():
            assert "input" not in table.columns
            assert "scan_events" in table.columns
        seen.append(True)
        return raw

    monkeypatch.setattr(api, "scan_results_df", capture)
    path = Path(__file__).parent / "fixtures" / "demo_scan"
    result = transect.load(str(path))
    assert seen
    assert result.scan_status.scanners
