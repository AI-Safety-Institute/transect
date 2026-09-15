"""Transcript ingestion (.eval and OpenClaw .jsonl) and the
one-sample/epoch selection rules."""

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from transect.api import _openclaw_files, _run
from transect.selection import select_transcripts
from transect.spec import Spec


def info(
    task_id="s1", epoch=1, task_set="ts1", success=None, transcript_id=None
) -> Any:
    """A TranscriptInfo stand-in (duck-typed; Any keeps checkers off)."""
    return SimpleNamespace(
        task_id=task_id,
        task_repeat=epoch,
        task_set=task_set,
        success=success,
        transcript_id=transcript_id or f"{task_id}-e{epoch}-{task_set}",
    )


def test_single_sample_log_needs_no_sample_argument():
    """A one-sample log selects itself; nothing is filtered."""
    selection = select_transcripts([info()])
    assert selection.transcript_ids is None


def test_multi_sample_log_requires_sample_and_lists_the_ids():
    """With several samples present the error names every available id."""
    infos = [info("s1"), info("s2")]
    with pytest.raises(ValueError, match=r"pass sample=<id>.*'s1', 's2'"):
        select_transcripts(infos)


def test_unknown_sample_id_lists_the_available_ones():
    with pytest.raises(ValueError, match=r"unknown sample id 'nope'.*'s1'"):
        select_transcripts([info("s1")], sample="nope")


@pytest.mark.parametrize(
    ("epochs", "kept"),
    [
        (2, ["e2"]),
        ([1, 3], ["e1", "e3"]),
        ("all", None),
    ],
    ids=["single-epoch", "epoch-list", "all-epochs"],
)
def test_explicit_epoch_selection(epochs, kept):
    """epochs= keeps exactly the requested 1-based epochs; "all" keeps
    everything unfiltered."""
    infos = [info(epoch=e) for e in (1, 2, 3)]
    selection = select_transcripts(infos, epochs=epochs)
    if kept is None:
        assert selection.transcript_ids is None
    else:
        assert selection.transcript_ids == [f"s1-{e}-ts1" for e in kept]


def test_missing_epoch_errors_with_the_present_ones():
    with pytest.raises(ValueError, match=r"epoch\(s\) \[9\] not present.*\[1, 2\]"):
        select_transcripts([info(epoch=1), info(epoch=2)], epochs=9)


@pytest.mark.parametrize(
    ("successes", "kept_epoch"),
    [((False, True, True), 2), ((False, False, False), 1)],
    ids=["earliest-successful", "earliest-when-none-succeeded"],
)
def test_auto_epoch_prefers_the_earliest_success(successes, kept_epoch):
    """epochs=None on a multi-epoch sample keeps the earliest successful
    epoch, falling back to the earliest epoch outright."""
    infos = [
        info(epoch=e, success=s) for e, s in zip((1, 2, 3), successes, strict=True)
    ]
    selection = select_transcripts(infos)
    assert selection.transcript_ids == [f"s1-e{kept_epoch}-ts1"]


def test_same_sample_and_epoch_across_task_sets_is_refused():
    """The same (sample, epoch) in two task_sets is two different runs,
    never an implicit choice."""
    infos = [info(task_set="a"), info(task_set="b")]
    with pytest.raises(ValueError, match="2 task_sets"):
        select_transcripts(infos)


def test_empty_log_errors():
    with pytest.raises(ValueError, match="no transcripts"):
        select_transcripts([])


def test_openclaw_jsonl_imports_and_scans_like_an_eval(openclaw_log, tmp_path):
    """An OpenClaw telemetry file rides the importer into a transcript
    DB and produces the same mechanical frames as an .eval log.

    task_set stays None on import, task_name falls back to the file
    stem, and success stays honestly None (the importer never scores).
    """
    results = _run(
        logs=str(openclaw_log), spec=Spec(), scans_dir=str(tmp_path / "scans")
    )
    timeline = results.token_timeline
    assert timeline.transcript_id.nunique() == 1
    assert set(timeline.agent.unique()) == {"openclaw"}
    assert timeline.turn.nunique() == 4
    row = results.transcript_info.iloc[0]
    assert pd.isna(row.task_set)
    assert row.task_name == "mini_telemetry"
    assert row.success is None
    assert row.model is not None


def test_openclaw_reimport_is_idempotent(openclaw_log, tmp_path):
    """A second run over the same .jsonl must not duplicate transcripts
    in the DB."""
    first = _run(logs=str(openclaw_log), spec=Spec(), scans_dir=str(tmp_path / "scans"))
    second = _run(
        logs=str(openclaw_log), spec=Spec(), scans_dir=str(tmp_path / "scans")
    )
    pd.testing.assert_frame_equal(
        second.token_timeline.reset_index(drop=True),
        first.token_timeline.reset_index(drop=True),
    )


def test_mixing_openclaw_and_eval_inputs_raises(tmp_path):
    """One run ingests one source kind; a .jsonl directory mixed with an
    .eval file is refused."""
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "a.jsonl").write_text("")
    eval_log = tmp_path / "run.eval"
    eval_log.write_text("")
    with pytest.raises(ValueError, match="cannot mix"):
        _openclaw_files([str(telemetry), str(eval_log)])


def test_eval_log_scan_selects_the_requested_sample(fixture_logs, tmp_path):
    """A multi-sample .eval scans exactly the sample= transcript."""
    results = _run(
        logs=str(fixture_logs),
        spec=Spec(),
        scans_dir=str(tmp_path / "scans"),
        sample="fixture-sample-1",
    )
    assert set(results.token_timeline.sample_id) == {"fixture-sample-1"}
    assert (results.token_timeline.epoch == 1).all()
