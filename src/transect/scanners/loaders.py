"""Loaders for user layers.

`reasoning_turns` is the per-turn judged-layer shape: one loader item
per digest turn of the orchestrator (the turns `decision_phases` shows
its judge), so a `cohort_llm_scanner` built behind it judges the
orchestrator agent one turn at a time. The name predates tool calls in
the digests and is kept for compatibility.
"""

from collections.abc import AsyncIterator

from inspect_ai.model import ChatMessageUser
from inspect_scout import Loader, Transcript, TranscriptContent, loader

from transect.scanners.cohort import batch_item_content
from transect.scanners.phases import turn_digests
from transect.scanners.phases_common import DigestSettings, digest_line


@loader(
    content=TranscriptContent(
        messages="all",
        events=["model", "span_begin", "span_end", "tool", "branch"],
        timeline=True,
    )
)
def reasoning_turns(
    batch: int = 1, digest: DigestSettings | None = None
) -> Loader[Transcript]:
    """One item per digest turn of the orchestrator.

    Each item's message is the turn's digest line, as
    `decision_phases` shows it: its reasoning (when the source records
    reasoning blocks), its own text, its tool calls, and each sub-agent
    spawn task, behind the markers `DIGEST_MARKERS` explains - a judge
    prompt over these items should carry that legend. The item's
    metadata carries ``{"turn": n}``.

    Args:
        batch: With ``batch=N``, one item per window of up to N
            consecutive turns instead - the shape
            ``cohort_llm_scanner(batch=True)`` judges in one call per
            member. The item packs its turns as ``[ITEM n]``-marked
            content and declares one fact dict per turn as
            ``metadata={"items": [...]}``, each carrying the turn
            number and the matching unbatched item id.
        digest: What each item carries (`DigestSettings`, as on
            `decision_phases`); ``None`` is the defaults.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

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
