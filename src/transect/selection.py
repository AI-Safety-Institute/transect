"""Pre-scan transcript selection: one sample, chosen epochs."""

from dataclasses import dataclass, field
from typing import Literal

from inspect_scout import TranscriptInfo, Transcripts


@dataclass
class Selection:
    """The outcome of sample/epoch selection over a transcript index.

    ``transcript_ids`` is None when no filtering is needed (scan
    everything); ``notes`` carries user-facing selection messages."""

    transcript_ids: list[str] | None
    notes: list[str] = field(default_factory=list)


def select_transcripts(
    infos: list[TranscriptInfo],
    sample: str | None = None,
    epochs: int | list[int] | Literal["all"] | None = None,
) -> Selection:
    """Select one sample's transcripts, narrowed by epoch.

    Args:
        infos: The transcript index entries.
        sample: The sample id to select - exact match on
            ``TranscriptInfo.task_id``. Optional for a single-sample
            log; with several samples present it is required (the
            error lists the available ids). An unknown id raises.
        epochs: 1-based epoch(s) to keep; ``"all"`` keeps every
            epoch; None = auto - a single-epoch sample runs whole, a
            multi-epoch sample keeps its earliest successful epoch
            (the earliest epoch when none succeeded).

    Returns:
        The ``Selection``; ``transcript_ids`` preserves the index
        order and is None when nothing filtered.
    """
    if not infos:
        raise ValueError("the log contains no transcripts")
    available = list(dict.fromkeys(i.task_id or "" for i in infos))

    if sample is None:
        if len(available) > 1:
            raise ValueError(
                f"the log contains {len(available)} samples - a Transect "
                "run scans one sample at a time; pass sample=<id> with "
                f"one of: {available}"
            )
        kept = infos
        narrowed_sample = False
    else:
        if sample not in available:
            raise ValueError(
                f"unknown sample id {sample!r} - available samples: {available}"
            )
        kept = [i for i in infos if i.task_id == sample]
        narrowed_sample = len(kept) != len(infos)

    # the same (sample, epoch) from several task_sets is not an epoch
    # choice - refuse instead of silently keeping one run
    by_run: dict[tuple[str, int], set[str]] = {}
    for info in kept:
        key = (info.task_id or "", _epoch(info))
        by_run.setdefault(key, set()).add(info.task_set or "")
    duplicated = {key: sets for key, sets in by_run.items() if len(sets) > 1}
    if duplicated:
        (sample_id, epoch_n), task_sets = next(iter(duplicated.items()))
        raise ValueError(
            f"sample {sample_id!r} (epoch {epoch_n}) appears in "
            f"{len(task_sets)} task_sets {sorted(task_sets)} - a Transect "
            "run scans one run of one sample; point logs at a single "
            "task_set's log file"
        )

    notes: list[str] = []
    if epochs == "all":
        narrowed_epochs = False
    elif epochs is not None:
        wanted = [epochs] if isinstance(epochs, int) else list(epochs)
        present = sorted({_epoch(i) for i in kept})
        missing = [e for e in wanted if e not in present]
        if missing:
            raise ValueError(
                f"epoch(s) {missing} not present - the sample has epochs "
                f"{present} (1-based)"
            )
        kept = [i for i in kept if _epoch(i) in wanted]
        narrowed_epochs = True
    else:
        kept, notes, narrowed_epochs = _auto_epoch(kept)

    if not narrowed_sample and not narrowed_epochs:
        return Selection(transcript_ids=None, notes=notes)
    return Selection(transcript_ids=[i.transcript_id for i in kept], notes=notes)


async def read_index(transcripts: Transcripts) -> list[TranscriptInfo]:
    """Read the transcript index (metadata only)."""
    infos: list[TranscriptInfo] = []
    async with transcripts.reader() as reader:
        async for info in reader.index():
            infos.append(info)
    return infos


def _epoch(info: TranscriptInfo) -> int:
    return info.task_repeat if info.task_repeat is not None else 1


def _auto_epoch(
    infos: list[TranscriptInfo],
) -> tuple[list[TranscriptInfo], list[str], bool]:
    """epochs=None: the sample's earliest successful epoch.

    Returns (kept, notes, narrowed) - narrowed is False for a
    single-epoch sample (nothing to select; run whole, silently)."""
    if len(infos) == 1:
        return infos, [], False
    ordered = sorted(infos, key=_epoch)
    successful = [i for i in ordered if i.success is True]
    chosen = successful[0] if successful else ordered[0]
    tail = "(earliest successful)" if successful else "(earliest - no epoch succeeded)"
    note = f"running on epoch {_epoch(chosen)} of {len(infos)} {tail}"
    return [chosen], [note], True
