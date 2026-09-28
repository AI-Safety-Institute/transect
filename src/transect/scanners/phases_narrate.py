"""Narratives for decision_phases: headline + summary + turn groups."""

from collections.abc import Mapping, Sequence
from itertools import pairwise

from inspect_ai.model import CachePolicy, Model
from inspect_scout import AnswerStructured
from pydantic import BaseModel, Field, ValidationError, create_model

from transect.scanners.helpers import capped_lines
from transect.scanners.phases_common import (
    EVIDENCE_LINES,
    Digest,
    NarrationGroupStatus,
    StitchedPhase,
    TurnGroup,
    call_judge,
    context_blocks,
    digest_line,
    gather_judge_calls,
    humanise_phase,
)
from transect.spec import Spec

_NARRATE_PER = 6  # phases per narrative call

_NARRATE_HEAD = (
    "You are narrating each phase of an autonomous agent's work. For "
    "each PHASE shown you get its label and its turns (one line per "
    "turn, prefixed with the turn index). Produce per phase:\n"
    "1. HEADLINE: ONE plain-English sentence (<=16 words) stating what "
    "the agent DID in the phase — concrete and specific (name the "
    "method/experiment/section if shown), NOT a quality judgement.\n"
    "2. SUMMARY: 2-3 sentences (<=400 chars) describing what the agent "
    "did, referencing the phase label where relevant. Descriptive and "
    "concrete, NOT a quality judgement.\n"
    "3. GROUPS: a small number (2-5; just 1 if the phase is tiny) of "
    "CONTIGUOUS sub-sections partitioning the phase's turns, each a "
    "short title + one-line gist, in order, together covering ALL the "
    "phase's turns.\n"
)


class _PhaseNarrative(BaseModel):
    """One phase's narrative.

    JUDGE-FACING: the Field descriptions render in the answer() tool."""

    phase_index: int = Field(description="The phase index being narrated, as shown.")
    headline: str = Field(description="One sentence (<=16 words): what the agent did.")
    summary: str = Field(
        description="2-3 sentences (<=400 chars) describing the phase."
    )
    groups: list[TurnGroup] = Field(
        description=(
            "2-5 contiguous sub-sections (1 if tiny) partitioning the "
            "phase's turns, in order."
        )
    )


class NarratorAudit(BaseModel):
    """The narrator's audit."""

    ran: bool = True
    n_fallback: int = 0
    narrator_model: str | None = None


async def narrate_phases(
    judge: Model,
    spec: Spec,
    *,
    task_prompt: str,
    digests: Sequence[Digest],
    phases: list[StitchedPhase],
    cache: bool | CachePolicy,
) -> NarratorAudit:
    """Narrate the phases: headline + summary + turn groups, in place.

    Mutates ``phases`` in place.

    Args:
        judge: The judge model.
        spec: Supplies the user context block.
        task_prompt: The agent's task prompt ("" omits the block).
        digests: All turn digests (evidence lines).
        phases: The phases to narrate; mutated in place.
        cache: Judge-call caching.

    Returns:
        The counts-only narrator audit.
    """
    audit = NarratorAudit()
    if not phases:
        return audit
    system = narrate_system_prompt(spec, task_prompt)
    answer = _narrate_answer_spec()
    ids_chunks = [
        list(range(offset, min(offset + _NARRATE_PER, len(phases))))
        for offset in range(0, len(phases), _NARRATE_PER)
    ]
    by_turn = {d.turn: d for d in digests}
    results = await gather_judge_calls(
        call_judge(
            judge, answer, system, _narrate_user_prompt(phases, ids, by_turn), cache
        )
        for ids in ids_chunks
    )
    narratives: dict[int, _PhaseNarrative] = {}
    for ids, (value, status) in zip(ids_chunks, results, strict=True):
        if status != "ok":
            continue  # whole chunk falls back
        raw = (value or {}).get("narratives")
        for item in raw if isinstance(raw, list) else []:
            try:
                parsed = _PhaseNarrative.model_validate(item)
            except ValidationError:
                continue
            if parsed.phase_index in ids and parsed.phase_index not in narratives:
                narratives[parsed.phase_index] = parsed
    for k, p in enumerate(phases):
        default_title = humanise_phase(p.phase)
        narrative = narratives.get(k)
        if narrative is None:
            p.narration_group_status = "no_narrative"
            audit.n_fallback += 1
            p.headline = f"{default_title} ({p.n_turns} turns)"
            p.summary = ""
            p.turn_groups = validate_turn_groups([], p, default_title)
            continue
        p.headline = narrative.headline.strip() or (
            f"{default_title} ({p.n_turns} turns)"
        )
        p.summary = narrative.summary.strip()
        p.turn_groups, p.narration_group_status = _validated_groups(
            narrative.groups, p, default_title
        )
    return audit


def narrate_system_prompt(spec: Spec, task_prompt: str = "") -> str:
    """Render the narrator system prompt.

    Head + the two standard context blocks + the answer contract.

    Args:
        spec: Supplies the user context block.
        task_prompt: The agent's own task prompt ("" omits the block).

    Returns:
        The complete narrator system prompt string.
    """
    prompt = _NARRATE_HEAD
    prompt += context_blocks(spec, task_prompt)
    prompt += (
        "\nReport by calling the answer() tool: 'narratives' is one "
        "entry per PHASE shown, each with phase_index (the id shown), "
        "headline, summary, and groups (each with turn_start and "
        "turn_end — inclusive turn indices as shown — plus title and "
        "gist)."
    )
    return prompt


def validate_turn_groups(
    groups: Sequence[TurnGroup],
    phase: StitchedPhase,
    default_title: str,
) -> list[TurnGroup]:
    """Preserve an exact partition, or use a neutral whole-phase fallback.

    Args:
        groups: The judge's proposed groups.
        phase: The phase being partitioned.
        default_title: Title for the whole-phase fallback group.

    Returns:
        The validated groups, in turn order.
    """
    return _validated_groups(groups, phase, default_title)[0]


def _validated_groups(
    groups: Sequence[TurnGroup], phase: StitchedPhase, default_title: str
) -> tuple[list[TurnGroup], NarrationGroupStatus]:
    """Return the partition and its outcome from the same validation decision."""
    lo, hi = phase.turn_start, phase.turn_end
    whole = [TurnGroup(turn_start=lo, turn_end=hi, title=default_title, gist="")]
    if not groups:
        return whole, "empty_groups"
    ordered = sorted(groups, key=lambda group: group.turn_start)
    if (
        ordered[0].turn_start != lo
        or ordered[-1].turn_end != hi
        or any(group.turn_start > group.turn_end for group in ordered)
        or any(
            right.turn_start != left.turn_end + 1 for left, right in pairwise(ordered)
        )
    ):
        return whole, "invalid_partition"
    return [
        TurnGroup(
            turn_start=group.turn_start,
            turn_end=group.turn_end,
            title=group.title.strip() or f"Turns {group.turn_start}+",
            gist=group.gist.strip(),
        )
        for group in ordered
    ], "complete"


def _narrate_answer_spec() -> AnswerStructured:
    """Build the narrator's answer schema."""
    narrate_chunks = create_model(
        "_NarrateChunks",
        narratives=(
            list[_PhaseNarrative],
            Field(description="One narrative per phase shown."),
        ),
        # declared so Scout does not auto-add its stock explanation field
        explanation=(
            str,
            Field(description="One or two sentences on your narration."),
        ),
    )
    return AnswerStructured(type=narrate_chunks)


def _narrate_user_prompt(
    phases: Sequence[StitchedPhase],
    ids: Sequence[int],
    by_turn: Mapping[int, Digest],
) -> str:
    """Render one narrator chunk: a block per phase.

    Args:
        phases: All phases.
        ids: The phase indices in this chunk.
        by_turn: All turn digests, keyed by turn.

    Returns:
        The chunk user prompt.
    """
    blocks = []
    for k in ids:
        p = phases[k]
        members = capped_lines(
            [
                digest_line(by_turn[turn])
                for turn in range(p.turn_start, p.turn_end + 1)
                if turn in by_turn
            ],
            EVIDENCE_LINES,
        )
        blocks.append(
            f"PHASE {k} [{p.phase}, {p.n_turns} turns]:\n" + "\n".join(members)
        )
    return "Narrate these phases:\n\n" + "\n\n".join(blocks)
