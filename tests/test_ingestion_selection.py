"""Transcript ingestion (.eval and OpenClaw .jsonl) and the
one-sample/epoch selection rules."""

import asyncio
import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from inspect_scout import TranscriptContent, transcripts_from

from transect.api import _import_openclaw, _openclaw_files, _run, load
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
    """Unchanged input preserves identities and frames in independent snapshots."""
    first = _run(logs=str(openclaw_log), spec=Spec(), scans_dir=str(tmp_path / "scans"))
    second = _run(
        logs=str(openclaw_log), spec=Spec(), scans_dir=str(tmp_path / "scans")
    )
    pd.testing.assert_frame_equal(
        second.token_timeline.reset_index(drop=True),
        first.token_timeline.reset_index(drop=True),
    )
    assert first.transcripts_location != second.transcripts_location
    assert load(str(tmp_path / "scans")).scan_location == second.scan_location


async def _stored_transcripts(location):
    async with transcripts_from(location).reader() as reader:
        return [
            await reader.read(info, TranscriptContent(messages="all", events="all"))
            async for info in reader.index()
        ]


def test_reused_scan_directory_selects_only_current_inputs(openclaw_log, tmp_path):
    """A later invocation never selects a transcript imported by an earlier one."""
    other = tmp_path / "other.jsonl"
    other.write_text(
        openclaw_log.read_text().replace("sess-cron-0001", "sess-cron-0002")
    )
    scans = str(tmp_path / "scans")
    first = _run(str(openclaw_log), Spec(), scans_dir=scans)
    second = _run(str(other), Spec(), scans_dir=scans)
    assert (
        second.transcript_info.sample_id.tolist()
        != first.transcript_info.sample_id.tolist()
    )
    assert second.transcript_info.task_name.tolist() == ["other"]
    with pytest.raises(ValueError, match="unknown sample id"):
        _run(
            str(other),
            Spec(),
            scans_dir=scans,
            sample=first.transcript_info.sample_id.iloc[0],
        )


def test_changed_input_preserves_prior_scan_source(openclaw_log, tmp_path):
    """Edited inputs leave earlier scans' stored messages and events unchanged."""
    source = tmp_path / "input.jsonl"
    source.write_text(openclaw_log.read_text())
    scans = str(tmp_path / "scans")
    first = _run(str(source), Spec(), scans_dir=scans)
    original = asyncio.run(_stored_transcripts(first.transcripts_location))[0]
    source.write_text(
        source.read_text()
        .replace("Wrapping up.", "Updated conclusion.")
        .replace('"output": 80', '"output": 81')
    )
    second = _run(str(source), Spec(), scans_dir=scans)
    updated = asyncio.run(_stored_transcripts(second.transcripts_location))[0]
    assert updated.transcript_id == original.transcript_id
    assert updated.messages[-1].text == "Updated conclusion."
    assert original.messages[-1].text == "Wrapping up."
    assert (
        second.token_timeline.output_tokens.sum()
        == first.token_timeline.output_tokens.sum() + 1
    )
    restored = load(first.scan_location)
    stored = asyncio.run(_stored_transcripts(restored.transcripts_location))[0]
    assert stored.model_dump() == original.model_dump()
    pd.testing.assert_frame_equal(restored.token_timeline, first.token_timeline)


def test_conflicting_input_id_fails_before_scanning(
    openclaw_log, tmp_path, monkeypatch
):
    """Different files with the same identity cannot silently replace one another."""
    other = tmp_path / "conflict.jsonl"
    other.write_text(
        openclaw_log.read_text().replace("Wrapping up.", "Different conclusion.")
    )

    def unexpected_scan(**kwargs):
        pytest.fail("duplicate input must be rejected before scanning")

    monkeypatch.setattr("transect.api.scout_scan", unexpected_scan)
    with pytest.raises(ValueError, match="duplicate transcript id") as error:
        _run([str(openclaw_log), str(other)], Spec(), scans_dir=str(tmp_path / "scans"))
    assert str(openclaw_log) in str(error.value)
    assert str(other) in str(error.value)


@pytest.mark.parametrize("content", ["", "not json\n"])
def test_empty_import_cannot_reuse_previous_input(openclaw_log, tmp_path, content):
    """An empty or wholly malformed input never falls back to an earlier transcript."""
    source = tmp_path / "input.jsonl"
    source.write_text(openclaw_log.read_text())
    scans = str(tmp_path / "scans")
    _run(str(source), Spec(), scans_dir=scans)
    source.write_text(content)
    with pytest.raises(ValueError, match="no transcripts"):
        _run(str(source), Spec(), scans_dir=scans)


def test_new_import_preserves_existing_database(openclaw_log, tmp_path):
    """New snapshot files stay outside the original database's recursive search root."""
    scans = tmp_path / "scans"
    existing = scans / "transcripts"
    asyncio.run(_import_openclaw([openclaw_log], str(existing)))
    before = {
        str(p.relative_to(existing)): p.read_bytes()
        for p in existing.rglob("*.parquet")
    }
    from_existing = _run(str(existing), Spec(), scans_dir=str(scans))
    fresh = _run(str(openclaw_log), Spec(), scans_dir=str(scans))
    assert not Path(fresh.transcripts_location).is_relative_to(existing)
    assert {
        str(p.relative_to(existing)): p.read_bytes()
        for p in existing.rglob("*.parquet")
    } == before
    old = asyncio.run(
        _stored_transcripts(load(from_existing.scan_location).transcripts_location)
    )
    assert len(old) == 1
    assert old[0].messages[-1].text == "Wrapping up."


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


def test_vendored_importer_notice_and_docstring_pin_the_same_upstream_commit():
    """The third-party notice cites the commit the importer docstring was
    vendored from, so a re-vendor cannot update one and leave the other
    stale."""
    # transect.ingestion re-exports the loader function under the subpackage's
    # name, so the module itself must be resolved through importlib.
    vendored = importlib.import_module("transect.ingestion.openclaw_telemetry_hal")
    notice = (Path(__file__).parents[1] / "LICENSES" / "README.md").read_text()
    notice_sha = re.search(r"at commit `([0-9a-f]+)`", notice)
    docstring_sha = re.search(r"inspect_scout @ ([0-9a-f]+)", vendored.__doc__ or "")
    assert notice_sha and docstring_sha
    assert notice_sha.group(1) == docstring_sha.group(1)
