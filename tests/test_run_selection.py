"""One logical sample/epoch must not silently choose among physical inputs."""

import asyncio
import shutil

import pytest
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_scout import TranscriptInfo, transcripts_from

from transect.api import _run
from transect.selection import read_index, select_transcripts
from transect.spec import Spec


def info(tid, source=None, *, epoch=1, task_set="task", sample="sample", success=None):
    """A real index record with optional source identity and a visible origin."""
    return TranscriptInfo(
        transcript_id=tid,
        source_id=source,
        source_uri=f"logs/{tid}.eval",
        task_id=sample,
        task_set=task_set,
        task_repeat=epoch,
        success=success,
    )


@pytest.mark.parametrize("epochs", [None, "all", 1])
@pytest.mark.parametrize("sources", [("run-a", "run-b"), (None, None), ("run-a", None)])
def test_same_epoch_distinct_transcripts_are_ambiguous(epochs, sources):
    """Different transcript identities cannot become an arbitrary epoch choice."""
    records = [
        info(tid, source) for tid, source in zip(["a", "b"], sources, strict=True)
    ]
    with pytest.raises(ValueError) as caught:
        select_transcripts(records, epochs=epochs)
    message = str(caught.value)
    assert "sample" in message and "epoch 1" in message
    assert "a" in message and "b" in message
    assert "logs/a.eval" in message and "logs/b.eval" in message
    if None in sources:
        assert "independent" not in message


def test_repeated_transcript_identity_is_not_silently_deduplicated():
    """Repeated inputs are identified as duplicates, not invented separate runs."""
    record = info("same", "run")
    with pytest.raises(ValueError, match="repeat"):
        select_transcripts([record, record])


def test_distinct_sources_cannot_hide_behind_one_transcript_id():
    """Conflicting recorded sources remain ambiguous even if a UUID was reused."""
    with pytest.raises(ValueError, match="sources"):
        select_transcripts([info("same", "run-a"), info("same", "run-b")])


@pytest.mark.parametrize("sources", [("run-a", "run-b"), (None, None)])
def test_distinct_epochs_with_optional_sources_are_valid(sources):
    """Unique epochs remain valid, without inventing a meaning for missing sources."""
    records = [info("a", sources[0]), info("b", sources[1], epoch=2, success=True)]
    assert select_transcripts(records).transcript_ids == ["b"]
    assert select_transcripts(records, epochs="all").transcript_ids is None


def test_explicit_selection_does_not_validate_excluded_inputs():
    """An unambiguous requested sample and epoch is unaffected by other collisions."""
    records = [
        info("a", "run-a", task_set="first"),
        info("b", "run-b", task_set="second"),
        info("c", "run-a", epoch=2),
        info("d", "run-d", sample="other"),
        info("e", "run-e", sample="other"),
    ]
    assert select_transcripts(records, sample="sample", epochs=2).transcript_ids == [
        "c"
    ]


@pytest.mark.parametrize("epochs", [None, "all", 1])
def test_two_real_eval_runs_are_rejected_before_scanning(
    epochs, demo_log, tmp_path, monkeypatch
):
    """Two .eval files for the same task/sample/epoch fail before any scanner runs."""
    logs = tmp_path / "logs"
    logs.mkdir()
    for suffix in ["a", "b"]:
        log = read_eval_log(demo_log)
        log.eval.eval_id = f"eval-{suffix}"
        log.eval.run_id = f"run-{suffix}"
        log.eval.task_id = f"task-{suffix}"
        log.samples[0].uuid = f"sample-{suffix}"
        write_eval_log(log, logs / f"run-{suffix}.eval")
    records = asyncio.run(read_index(transcripts_from(str(logs))))
    assert len({row.source_id for row in records}) == 2
    assert len({row.task_set for row in records}) == 1
    monkeypatch.setattr(
        "transect.api.scout_scan",
        lambda **kwargs: pytest.fail("ambiguous selection must fail before scanning"),
    )
    with pytest.raises(ValueError, match="sources") as caught:
        _run(str(logs), Spec(), epochs=epochs, scans_dir=str(tmp_path / "scans"))
    assert "run-a.eval" in str(caught.value) and "run-b.eval" in str(caught.value)


def test_identical_eval_copies_are_reported_as_repeated_inputs(demo_log, tmp_path):
    """Byte-identical copies do not masquerade as two epochs or get silently chosen."""
    for name in ["a.eval", "b.eval"]:
        shutil.copy2(demo_log, tmp_path / name)
    records = asyncio.run(read_index(transcripts_from(str(tmp_path))))
    assert len(records) == 2
    assert len({row.transcript_id for row in records}) == 1
    with pytest.raises(ValueError, match="repeat"):
        select_transcripts(records)


def test_real_distinct_epochs_still_scan(epochs_logs, tmp_path):
    """A valid multi-epoch sample retains every explicitly requested epoch."""
    result = _run(
        str(epochs_logs),
        Spec(),
        sample="epoch-sample-1",
        epochs="all",
        scans_dir=str(tmp_path / "scans"),
    )
    assert sorted(result.token_timeline.epoch.unique()) == [1, 2, 3]
    assert result.token_timeline.transcript_id.nunique() == 3
    assert not result.scan_status.has_failures
