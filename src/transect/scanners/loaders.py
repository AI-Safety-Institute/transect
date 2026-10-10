"""Loaders for user layers.

`reasoning_turns` is the per-turn judged-layer shape: one loader item
per digest turn of the orchestrator (the turns `decision_phases` shows
its judge), so a `cohort_llm_scanner` built behind it judges the
orchestrator agent one turn at a time. The name predates tool calls
and user messages in the digests and is kept for compatibility.
"""

from collections.abc import AsyncIterator

from inspect_ai.model import ChatMessageUser
from inspect_scout import Loader, Transcript, TranscriptContent, loader

from transect.scanners.cohort import batch_item_content
from transect.scanners.phases import turn_digests
from transect.scanners.phases_common import (
    FINAL_TEXT_CHARS,
    SNIPPET_CHARS,
    TOOL_CALL_CHARS,
    TOOL_RESULT_CHARS,
    USER_CHARS,
    check_digest_caps,
    digest_line,
)


@loader(
    content=TranscriptContent(
        messages="all",
        events=["model", "span_begin", "span_end", "tool", "branch"],
        timeline=True,
    )
)
def reasoning_turns(
    batch: int = 1,
    snippet_chars: int = SNIPPET_CHARS,
    final_text_chars: int = FINAL_TEXT_CHARS,
    tool_call_chars: int = TOOL_CALL_CHARS,
    tool_result_chars: int = TOOL_RESULT_CHARS,
    user_chars: int = USER_CHARS,
    tool_only_turns: bool = True,
) -> Loader[Transcript]:
    """One item per digest turn of the orchestrator.

    Each item's message is the turn's digest line, as
    `decision_phases` shows it: the user messages the turn received,
    its reasoning (when the source records reasoning blocks), its own
    text, its tool calls with their arguments and recorded output,
    and each sub-agent spawn task. The item's metadata carries
    ``{"turn": n}``. The digest settings mean what they mean on
    `decision_phases`.

    Args:
        batch: With ``batch=N``, one item per window of up to N
            consecutive turns instead - the shape
            ``cohort_llm_scanner(batch=True)`` judges in one call per
            member. The item packs its turns as ``[ITEM n]``-marked
            content and declares one fact dict per turn as
            ``metadata={"items": [...]}``, each carrying the turn
            number and the matching unbatched item id.
        snippet_chars: Per-digest cap on turn text, reasoning, and
            each delegation.
        final_text_chars: Cap on the text of the last digest turn with
            text; never below ``snippet_chars``.
        tool_call_chars: Per-call cap on the JSON arguments; 0 shows
            tool names only.
        tool_result_chars: Per-call cap on the recorded output or
            error message; 0 leaves outputs out.
        user_chars: Per-message cap on user messages; 0 leaves them out.
        tool_only_turns: Keep turns whose only content is tool calls.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    check_digest_caps(
        snippet_chars=snippet_chars,
        final_text_chars=final_text_chars,
        tool_call_chars=tool_call_chars,
        tool_result_chars=tool_result_chars,
        user_chars=user_chars,
    )

    async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
        digests = turn_digests(
            transcript,
            snippet_chars=snippet_chars,
            final_text_chars=final_text_chars,
            tool_call_chars=tool_call_chars,
            tool_result_chars=tool_result_chars,
            user_chars=user_chars,
            tool_only_turns=tool_only_turns,
        )
        if batch == 1:
            for digest in digests:
                yield Transcript(
                    # the parent id keeps item ids globally unique, so
                    # per-item seeds (the verifier spot-check draw) stay
                    # independent across transcripts and epochs
                    transcript_id=f"{transcript.transcript_id}:turn-{digest.turn}",
                    messages=[ChatMessageUser(content=digest_line(digest))],
                    metadata={"turn": digest.turn},
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
