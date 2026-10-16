"""Loaders for user layers.

`digest_turns` is the per-turn judged-layer shape: one loader item
per digest turn of the orchestrator (an orchestrator turn with content
to judge, rendered as the built-in phase judges read it), so a
`cohort_llm_scanner` built behind it judges the orchestrator agent one
turn at a time.
"""

from collections.abc import AsyncIterator

from inspect_ai.model import ChatMessageUser
from inspect_scout import Loader, Transcript, TranscriptContent, loader

from transect.scanners.cohort import batch_item_content
from transect.scanners.digests import (
    DigestSettings,
    digest_line,
    resolve_digest,
    turn_digests,
)


@loader(
    content=TranscriptContent(
        messages="all",
        events=["model", "span_begin", "span_end", "tool", "branch"],
        timeline=True,
    )
)
def digest_turns(
    batch: int = 1, digest: DigestSettings | None = None
) -> Loader[Transcript]:
    """One item per digest turn of the orchestrator.

    Each item's message is the turn's digest line: its reasoning
    (when the source records reasoning blocks), its own text, its tool
    calls, and each sub-agent spawn task, each part capped by
    ``digest``. The item's metadata carries ``{"turn": n}``.

    Args:
        batch: With ``batch=N``, one item per window of up to N
            consecutive turns instead - the shape
            ``cohort_llm_scanner(batch=True)`` judges in one call per
            member. The item packs its turns as ``[ITEM n]``-marked
            content and declares one fact dict per turn as
            ``metadata={"items": [...]}``, each carrying the turn
            number and the matching unbatched item id.
        digest: The caps on each part of the line and whether
            tool-call-only turns yield an item (`DigestSettings`);
            ``None`` is the defaults.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    digest = resolve_digest(digest)

    async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
        digests = turn_digests(transcript, digest)
        if batch == 1:
            for record in digests:
                yield Transcript(
                    # the parent id keeps item ids globally unique, so
                    # per-item seeds (the verifier spot-check draw) stay
                    # independent across transcripts and epochs
                    transcript_id=f"{transcript.transcript_id}:turn-{record.turn}",
                    messages=[ChatMessageUser(content=digest_line(record))],
                    metadata={"turn": record.turn},
                )
            return
        for start in range(0, len(digests), batch):
            window = digests[start : start + batch]
            yield Transcript(
                transcript_id=(
                    f"{transcript.transcript_id}:"
                    f"turns-{window[0].turn}-{window[-1].turn}"
                ),
                messages=[
                    ChatMessageUser(
                        content=batch_item_content([digest_line(d) for d in window])
                    )
                ],
                metadata={
                    "items": [
                        {
                            "turn": d.turn,
                            "id": f"{transcript.transcript_id}:turn-{d.turn}",
                        }
                        for d in window
                    ]
                },
            )

    return load
