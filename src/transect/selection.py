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

    Raises:
        ValueError: When a selected sample/epoch has multiple physical
            inputs or repeated transcript identities. Explicit sample and
            epoch filters apply before this check; automatic epoch choice
            never resolves a collision by taking the first input.
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

    notes: list[str] = []
    narrowed_epochs = False
    if epochs is not None and epochs != "all":
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

    _validate_unique_runs(kept)
    if epochs is None:
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


def _validate_unique_runs(infos: list[TranscriptInfo]) -> None:
    """Require one physical input per selected sample/epoch before auto selection.

    A task_set names a task family, not a run. Missing source IDs do not
    establish independent runs; distinct transcript IDs still make selection
    ambiguous. Repeated identities are reported rather than deduplicated from
    metadata, which cannot establish that their contents are identical.
    """
    groups: dict[tuple[str, int], list[TranscriptInfo]] = {}
    for info in infos:
        groups.setdefault((info.task_id or "", _epoch(info)), []).append(info)
    for (sample_id, epoch_n), group in groups.items():
        if len(group) < 2:
            continue
        task_sets = {info.task_set or "" for info in group}
        sources = {
            source for info in group if (source := getattr(info, "source_id", None))
        }
        transcripts = {info.transcript_id for info in group}
        if len(task_sets) > 1:
            reason = f"{len(task_sets)} task_sets {sorted(task_sets)}"
        elif len(sources) > 1:
            reason = f"{len(sources)} sources {sorted(sources)}"
        elif len(transcripts) > 1:
            reason = "multiple transcript identities"
        else:
            reason = "repeated input for the same transcript identity"
        origins = "; ".join(
            f"{info.transcript_id!r} "
            f"(source_id={getattr(info, 'source_id', None)!r}, "
            f"source_uri={getattr(info, 'source_uri', None)!r})"
            for info in group
        )
        raise ValueError(
            f"sample {sample_id!r} (epoch {epoch_n}) has {reason}: {origins}. "
            "A Transect run scans one run of one sample; narrow logs to one "
            "input per epoch and remove repeated inputs."
        )


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
