"""Transcript excerpts, spawn prompts and tool counts for the report.

Phase-card turn excerpts are fetched at render time from the Scout transcript
store. Other frames can already contain source text: setup/task prompts, operator
interventions and subagent task text, as well as generated explanations. Neither
dataframes nor rendered reports should be treated as text-free exports.

This module preserves:

- **Same turn axis as everything else.** `_turn_excerpts` iterates
  `helpers.orchestrator_turns`, the one enumeration every scanner
  numbers by, so turn *n* here is turn *n* on the charts, in the phase
  ranges, and in the narrator's turn groups. Sub-agent turns are not
  on that axis and never become excerpt rows: the cards show what the
  judges read.
- **Bounded page size.** A real run's groups span more than a thousand
  turns, so the whole store is never inlined: only turns inside a turn
  group, at `TEXT_CHARS` each, under a per-card budget
  (`card_excerpts`), with the remainder counted.

Excerpt text is agent/user content: untrusted. It is carried as plain
``str`` and rendered through the templates' autoescape, never as
``Markup`` - see `templates/phase_cards.html.j2`.
"""

import asyncio
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from inspect_scout import TranscriptContent, transcripts_from

from transect.scanners.helpers import (
    Lanes,
    message_reasoning,
    orchestrator_turns,
    span_task_text,
)

TEXT_CHARS = 400
"""Per-turn excerpt cap. The full text lives in the Scout viewer."""


CARD_BUDGET = 10
"""Excerpt turns per phase card, split across the card's turn groups."""


_TOOLS_SHOWN = 6
"""Tool names named per turn before the rest are counted."""


_ORCHESTRATOR = "orchestrator"
"""Lane name for a span-less model turn (see `Excerpt.lane`)."""


COMPACTION_NOTE = "compaction summary call (the summarizer, not the agent)"


@dataclass(frozen=True)
class Excerpt:
    """One model turn's excerpt, as a card renders it.

    Attributes:
        turn: The orchestrator turn number (the report's turn axis).
        lane: The turn's agent lane - the sub-agent span's name, or
            ``"orchestrator"`` for a span-less turn (the token
            timeline records the same lane as ``None`` there; a
            reader needs it named, and "orchestrator" is what the
            report's sub-agent section already calls that lane).
        tools: The turn's tool calls, as names ("" for a prose turn);
            the tail beyond `_TOOLS_SHOWN` is counted, not dropped.
        text: The turn's own text, whitespace-flattened and capped at
            `TEXT_CHARS`.
        truncated: Whether `text` or `reasoning` was cut - the card
            says so.
        note: A marker the card shows beside the lane, or ``None``;
            `mark_compaction_turns` sets it on turns whose "output" is
            Inspect's summarizer speaking, not the agent.
        reasoning: The turn's reasoning-block text when the source
            records it, same flattening and cap as `text`; the card
            renders it as a muted [thinking] line above the text.
    """

    turn: int
    lane: str
    tools: str
    text: str
    truncated: bool
    note: str | None = None
    reasoning: str = ""


@dataclass(frozen=True)
class SpawnPrompt:
    """One sub-agent span's own spawn prompt, as the sub-agent activity
    section's expandable renders it.

    Attributes:
        text: The task text. Read from the transcript it is the whole
            text; the judge's stored copy (the per-span fallback in
            `render._subagent_section`) carries the scanner's cap.
        truncated: Whether `text` was cut - only ever True on the stored
            copy.
    """

    text: str
    truncated: bool


@dataclass(frozen=True)
class TranscriptExtras:
    """One transcript's render-time store-derived extras: several
    derived views of the same already-read transcript, so no view costs
    a second store fetch (see `read_transcript_extras`).

    Attributes:
        excerpts: `_turn_excerpts` output - the phase cards' own
            per-turn transcript text.
        spawn_prompts: `_spawn_prompts` output - the sub-agent activity
            section's own per-span spawn-task text.
        tool_counts: `_tool_call_counts` output - the phase cards'
            tool-call sort key + tag.
    """

    excerpts: dict[int, Excerpt] = field(default_factory=dict)
    spawn_prompts: dict[str, SpawnPrompt] = field(default_factory=dict)
    tool_counts: dict[int, int] = field(default_factory=dict)


def mark_compaction_turns(
    excerpts: dict[int, Excerpt], turns: Iterable[int]
) -> dict[int, Excerpt]:
    """Note the turns that were summarization calls.

    A summary compaction's ``generate()`` is a model event like any other,
    so it holds a turn on the shared axis and its excerpt is the summary
    body. Unmarked, a card presents that as the agent pausing to recap.
    ``turns`` derives from the flushes frame (the turn before a flush with
    a recorded ``compaction_prompt`` is its summarization call), so the
    card's marker and the flush list can never disagree.
    """
    marked = dict(excerpts)
    for turn in turns:
        if turn in marked:
            marked[turn] = replace(marked[turn], note=COMPACTION_NOTE)
    return marked


def read_transcript_extras(
    location: str | None, transcript_ids: list[str]
) -> dict[str, TranscriptExtras]:
    """Read the transcript store once and derive every extra for each
    wanted transcript.

    Called once per render (`render.render_report`), so each
    transcript's store read happens exactly once no matter how many
    report surfaces consume its extras: `_read` derives every view from
    the one `Transcript` object each `reader.read` call produced.

    Args:
        location: ``TransectResults.transcripts_location`` - a
            transcript database directory or an eval-log path.
        transcript_ids: The transcripts the report renders. Anything
            else in the store is indexed but never read.

    Returns:
        ``{transcript_id: TranscriptExtras}``, with an entry for each
        wanted transcript the store could actually be read for - a
        transcript whose own read failed is absent from the result
        while the others keep their extras (`_read` guards each read
        separately). ``{}`` when there is no reachable store, when it
        holds none of these transcripts, or when the failure is
        store-level (indexing, opening) rather than one record's.
        Every failure prints its reason to stderr and leaves the
        affected report surfaces rendering without their extras - see
        the module docstring on honest absence.
    """
    if not location or not transcript_ids or not Path(location).exists():
        return {}
    try:
        return _run_sync(_read(location, set(transcript_ids)))
    # deliberately broad, and store-level only (`_read` guards each
    # transcript's own read): the excerpts are an addition to a report
    # that rendered without them for its whole life, so no store failure
    # - a moved store, an unreadable index, a source kind this reader
    # cannot open - should cost the reader the rest of their report.
    # Named on stderr rather than swallowed, so a store that should have
    # worked does not fail invisibly.
    except Exception as failure:
        print(
            f"WARNING: transcript excerpts unavailable from {location}: "
            f"{type(failure).__name__}: {failure}",
            file=sys.stderr,
        )
        return {}


def card_excerpts(
    excerpts: dict[int, Excerpt],
    ranges: list[tuple[int, int]],
    budget: int | None = None,
) -> list[tuple[list[Excerpt], int]]:
    """Pick one card's excerpts, group by group, under a page budget.

    The budget is split evenly across the card's own groups (2-5 in
    practice, the narrator's own range) rather than spent front to
    back, so a card's later groups are not starved by its first one.
    Every group keeps at least one excerpt even when there are more
    groups than budget - a dropdown that opens on nothing is worse
    than a thin one - so a card with more groups than `budget` may
    show one excerpt per group instead (the cap is a page-size bound,
    not a contract).

    Args:
        excerpts: `_turn_excerpts` output for this transcript.
        ranges: The card's groups, as inclusive ``(turn_start,
            turn_end)`` pairs, in order.
        budget: Excerpt turns for the whole card; ``None`` reads
            `CARD_BUDGET` at call time (so the page-size bound is one
            constant, adjustable in one place).

    Returns:
        One ``(shown, n_more)`` pair per range, in the same order -
        always, so a caller zips it against its own groups without a
        length branch; a range with nothing to show gets ``([], 0)``.
        ``n_more`` counts the range's *excerpt-bearing* turns that did
        not fit - turns with no text or reasoning of their own were
        never candidates and are not counted (the card's tag line
        already reports the phase's turn counts).
    """
    if not ranges:
        return []
    if not excerpts:
        return [([], 0) for _ in ranges]
    turns = sorted(excerpts)
    per_group = max(1, (CARD_BUDGET if budget is None else budget) // len(ranges))
    selected = []
    for turn_start, turn_end in ranges:
        lo = bisect_left(turns, turn_start)
        hi = bisect_right(turns, turn_end)
        candidates = turns[lo:hi]
        shown = [excerpts[t] for t in candidates[:per_group]]
        selected.append((shown, len(candidates) - len(shown)))
    return selected


async def _read(location: str, wanted: set[str]) -> dict[str, TranscriptExtras]:
    """Index the store, read the wanted transcripts, derive every extra
    from each.

    ``events`` is the only content asked for: model events for the turns
    and their text, span events for the lane tree and each sub-agent
    span's spawn-task metadata. Reading messages as well would
    double the read for nothing - the turn axis is an event axis.

    Each transcript's read is guarded on its own, so the result holds
    every transcript that could be read and omits only those that could
    not (each named on stderr).
    """
    content = TranscriptContent(events=["model", "span_begin", "span_end"])
    found: dict[str, TranscriptExtras] = {}
    async with transcripts_from(location).reader() as reader:
        infos = [info async for info in reader.index() if info.transcript_id in wanted]
        for info in infos:
            # per-transcript, not per-batch: one unreadable record (a
            # truncated write, a schema this reader cannot parse) costs
            # that transcript its extras and no other's. A single
            # guard around the whole loop would silently downgrade every
            # transcript on the page because of one bad record - the
            # grain the module docstring's honesty claim is about.
            try:
                transcript = await reader.read(info, content)
                found[info.transcript_id] = TranscriptExtras(
                    excerpts=_turn_excerpts(transcript),
                    spawn_prompts=_spawn_prompts(transcript),
                    tool_counts=_tool_call_counts(transcript),
                )
            except Exception as failure:
                print(
                    f"WARNING: transcript excerpts unavailable for "
                    f"{info.transcript_id}: {type(failure).__name__}: {failure}",
                    file=sys.stderr,
                )
    return found


def _turn_excerpts(transcript: Any, text_chars: int = TEXT_CHARS) -> dict[int, Excerpt]:
    """Excerpt every orchestrator turn of one transcript that carries
    text or reasoning-block content.

    Turns with neither (content-free tool-call-only turns) get no
    excerpt but still consume their turn number, since the axis is
    `helpers.orchestrator_turns`' own enumeration - the module docstring
    has why that sharing is load-bearing.

    Provider-failure placeholder turns (the OpenClaw
    ``[assistant turn failed ...]`` text) are excerpted like any other
    turn: they are what the transcript actually records, and a reader
    of a group that contains one is better served seeing it than
    seeing the turn silently missing. (The phases scanner excludes
    them from its *classification* digests, a different question.)

    Args:
        transcript: The transcript to read (anything with ``events``).
        text_chars: Per-turn character cap.

    Returns:
        ``{turn: Excerpt}`` for the turns carrying text or reasoning.
    """
    found: dict[int, Excerpt] = {}
    for turn, event, calls in orchestrator_turns(transcript):
        message = event.output.message
        raw = (getattr(message, "text", None) or "") if message else ""
        text = " ".join(raw.split())
        reasoning = message_reasoning(message)
        if not text and not reasoning:
            continue
        found[turn] = Excerpt(
            turn=turn,
            lane=_ORCHESTRATOR,
            tools=_tools_label(calls),
            text=text[:text_chars],
            truncated=len(text) > text_chars or len(reasoning) > text_chars,
            reasoning=reasoning[:text_chars],
        )
    return found


def _spawn_prompts(transcript: Any) -> dict[str, SpawnPrompt]:
    """Each sub-agent span's own spawn prompt, in full, keyed by
    ``agent_span_id``. Uncapped: the expandable exists for a human to
    read the whole task. The judge's own capped copy fills in per span
    when this read yields nothing for it (no store, or a span whose
    metadata carries no task text).

    Reads directly off each ``span_begin`` event already present on
    ``transcript.events`` - the same object `_turn_excerpts` reads, so
    this adds no second store fetch. Extraction delegates to
    `transect.scanners.helpers.span_task_text`, the same function the
    `agent_spans` loader uses to decide what a span's classifier prompt
    is, so this expandable can never show a different task text than the
    one the sub-agent was classified against.

    The spans and each one's first model event (the handoff-input
    fallback `span_task_text` reads when the span's metadata carries no
    task) come from `helpers.Lanes`, the same resolution the loader
    uses. A span with no task text from either source is simply absent
    from the result, the same honest absence a model turn with no text
    gets.

    Args:
        transcript: The transcript to read (anything with ``events``).

    Returns:
        ``{agent_span_id: SpawnPrompt}`` for the spans with task text.
    """
    lanes = Lanes(transcript)
    found: dict[str, SpawnPrompt] = {}
    for begin in lanes.begins:
        text, _source = span_task_text(begin, lanes.first_models.get(begin.id))
        if text:
            found[begin.id] = SpawnPrompt(text=text, truncated=False)
    return found


def _tool_call_counts(transcript: Any) -> dict[int, int]:
    """Per-turn tool-call counts on the same `helpers.orchestrator_turns`
    axis the excerpts use - every orchestrator turn, text-bearing or not
    (a tool-only turn has calls but no excerpt), so the phase cards can
    sum a turn range for their tool-call sort and tag.

    Descriptive, not an audited total: it counts the calls the
    orchestrator's own model turns record; a sub-agent's tool calls are
    its own (`frames.lane_activity`, the subagents frame).
    """
    return {
        turn: len(calls)
        for turn, _event, calls in orchestrator_turns(transcript)
        if calls
    }


def _run_sync(coroutine):
    """Run one coroutine from this sync render path.

    `asyncio.run` is the plain case - a script, a pytest test - and is
    what `api._import_openclaw`/`api._run` use for their own async
    reads. It refuses outright when a loop is already running in this
    thread, which is the notebook case (an `transect.render()` call in a
    Jupyter cell) and the sync-API-driver case (Playwright's sync API, as in
    `tests/test_report_integration.py`), so that case gets its own loop on a
    worker thread rather than losing the excerpts to an environment
    detail. A coroutine is not bound to a loop until it is awaited, so
    handing this one to another thread is safe.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _tools_label(calls: list) -> str:
    """One turn's tool calls as names: the first `_TOOLS_SHOWN`, then a
    count of the rest (a long tool run is bounded, never silently
    shortened)."""
    names = [str(getattr(call, "function", None) or "tool") for call in calls]
    if len(names) <= _TOOLS_SHOWN:
        return ", ".join(names)
    rest = len(names) - _TOOLS_SHOWN
    return f"{', '.join(names[:_TOOLS_SHOWN])} (+{rest} more)"
