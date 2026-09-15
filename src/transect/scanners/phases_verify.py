"""Second-round verifier for decision_phases."""

import asyncio
import random
from collections.abc import Sequence
from typing import Literal

from inspect_ai.model import CachePolicy, Model
from inspect_scout import AnswerStructured
from pydantic import BaseModel, Field, ValidationError, create_model

from transect.scanners.cohort import _SPOT_DEFAULT, LOWCONF, Trigger, VerifierReview
from transect.scanners.helpers import capped_lines
from transect.scanners.phases_cohort import ConsensusJudgement
from transect.scanners.phases_common import (
    EVIDENCE_LINES,
    Digest,
    StitchedPhase,
    call_judge,
    context_blocks,
    digest_line,
    resolve_phases,
    stitch_phases,
    vocab_lines,
)
from transect.spec import Spec

_SAMPLE_MIN = 3  # random sample size: max(_SPOT_DEFAULT share, 3) of phases
_SAMPLE_SEED = 20260616  # deterministic random-sample draw

_VERIFY_HEAD = (
    "You are a second-round reviewer checking phase labels assigned to an "
    "autonomous agent's turns. For each PHASE shown, decide the SINGLE "
    "correct phase label given its turns and its neighbours. Labels:\n"
)


class _Verdict(BaseModel):
    """One phase's verifier verdict.

    JUDGE-FACING: the Field descriptions render in the answer() tool."""

    phase_index: int = Field(description="The PHASE index being reviewed, as shown.")
    phase: str  # judge-facing wording lives on the _VocabVerdict override
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your 0.0-1.0 certainty in the label you return.",
    )
    explanation: str = Field(description="Why (<=14 words).")


class VerifierAudit(BaseModel):
    """The verifier's counts-only audit (the value's ``verifier`` block).

    Not judge-facing. One stable shape: when the verifier did not run
    the value carries this model with ``ran=False`` and zero counts."""

    ran: bool = True
    n_low_confidence: int = 0
    n_low_agreement: int = 0
    n_wedge: int = 0
    n_random_sample: int = 0
    n_no_verdict: int = 0
    n_relabelled: int = 0
    n_weak_relabel: int = 0
    n_random_sample_relabelled: int = 0
    verifier_model: str | None = None


def select_for_verify(
    phases: Sequence[StitchedPhase],
    sample: float | None = None,
) -> dict[int, Trigger]:
    """Select phases for the second-round verifier.

    Triggers:

    - "low_confidence": ``min_confidence`` < 0.6
    - "low_agreement": ``min_agreement`` < 0.6 (k-roll consensus
      instability; solo phases carry None and never trigger).
    - "wedge": <=2-turn phase between two phases sharing one other
      label.
    - "random_sample": a share of the phases, stratified by label,
      drawn from un-triggered phases with a fixed seed.

    Args:
        phases: The phases, in turn order.
        sample: The share of phases spot-checked. ``None`` = the
            default max(5%, 3); a float = round(sample * n_phases).
            Capped at the phase count.

    Returns:
        phase index -> trigger, for every selected phase.
    """
    reasons: dict[int, Trigger] = {}
    for k, p in enumerate(phases):
        if p.min_confidence < LOWCONF:
            reasons[k] = "low_confidence"
            continue
        if p.min_agreement is not None and p.min_agreement < LOWCONF:
            reasons[k] = "low_agreement"
            continue
        prev_l = phases[k - 1].phase if k > 0 else None
        next_l = phases[k + 1].phase if k + 1 < len(phases) else None
        if (
            p.n_turns <= 2
            and prev_l == next_l
            and prev_l is not None
            and prev_l != p.phase
        ):
            reasons[k] = "wedge"
    if sample is None:
        n_sample = max(_SAMPLE_MIN, round(_SPOT_DEFAULT * len(phases)))
    else:
        n_sample = round(sample * len(phases))
    n_sample = min(n_sample, len(phases))
    rng = random.Random(_SAMPLE_SEED)
    by_label: dict[str, list[int]] = {}
    for k, p in enumerate(phases):
        if k not in reasons:
            by_label.setdefault(p.phase, []).append(k)
    sampled: set[int] = set()
    pool_size = sum(len(v) for v in by_label.values())
    labels_cycle = sorted(by_label)
    while len(sampled) < min(n_sample, pool_size) and labels_cycle:
        for label in list(labels_cycle):
            pool = [k for k in by_label[label] if k not in sampled]
            if not pool:
                labels_cycle.remove(label)
                continue
            sampled.add(rng.choice(pool))
            if len(sampled) >= n_sample:
                break
    for k in sampled:
        reasons[k] = "random_sample"
    return reasons


async def verify_phases(
    judge: Model,
    spec: Spec,
    *,
    task_prompt: str,
    digests: Sequence[Digest],
    digest_judgements: list[ConsensusJudgement],
    phases: list[StitchedPhase],
    phase_names: Sequence[str],
    chunk: int,
    cache: bool | CachePolicy,
    sample: float | None = None,
) -> tuple[list[StitchedPhase], VerifierAudit]:
    """Run the second-round verifier and apply its overturns.

    Selected phases are reviewed in chunks; a differing label is
    applied only at verifier confidence >= 0.6 (a weak signal never
    rewrites a phase). Repairs relabel the phase's judgement rows -
    label, confidence, and explanation all taken from the verifier -
    then the partition is re-stitched. Mutates ``digest_judgements``
    in place.

    Args:
        judge: The judge model.
        spec: Supplies phase definitions + user context.
        task_prompt: The agent's task prompt ("" omits the block).
        digests: All turn digests.
        digest_judgements: Digest phase judgements; relabelled in place.
        phases: The stitched phases before verification.
        phase_names: The resolved phase names (answer schema).
        chunk: Phases per verifier call.
        cache: Judge-call caching.
        sample: The spot-checked share (see `select_for_verify`).

    Returns:
        ``(phases, audit)`` - the (possibly re-stitched) phases carrying
        per-phase verifier flags, and the counts-only audit block.
    """
    reasons = select_for_verify(phases, sample=sample)
    selected = sorted(reasons)
    triggers = list(reasons.values())
    audit = VerifierAudit(
        n_low_confidence=triggers.count("low_confidence"),
        n_low_agreement=triggers.count("low_agreement"),
        n_wedge=triggers.count("wedge"),
        n_random_sample=triggers.count("random_sample"),
    )
    if not selected:
        return phases, audit
    system = verify_system_prompt(spec, task_prompt)
    answer = _verify_answer_spec(phase_names)
    verdicts: dict[int, _Verdict] = {}
    ids_chunks = [
        selected[offset : offset + chunk] for offset in range(0, len(selected), chunk)
    ]
    # throttled by inspect's own connection limit
    results = await asyncio.gather(
        *[
            call_judge(
                judge, answer, system, _verify_user_prompt(phases, ids, digests), cache
            )
            for ids in ids_chunks
        ]
    )
    for ids, (value, status) in zip(ids_chunks, results, strict=True):
        if status != "ok":
            continue  # whole chunk stays no-verdict
        raw_verdicts = (value or {}).get("verdicts")
        for item in raw_verdicts if isinstance(raw_verdicts, list) else []:
            try:
                verdict = _Verdict.model_validate(item)
            except ValidationError:
                continue
            if verdict.phase_index in ids and verdict.phase_index not in verdicts:
                verdicts[verdict.phase_index] = verdict
    return _apply_verdicts(
        digest_judgements, phases, verdicts, reasons, audit, str(judge)
    )


def verify_system_prompt(spec: Spec, task_prompt: str = "") -> str:
    """Render the verifier system prompt.

    Args:
        spec: Supplies the phase definitions and the user context.
        task_prompt: The agent's own task prompt ("" omits the block).

    Returns:
        The complete verifier system prompt string.
    """
    phase_defs, _ = resolve_phases(spec)
    labels_line = ", ".join(p.label for p in phase_defs)
    prompt = _VERIFY_HEAD + vocab_lines(phase_defs)
    prompt += context_blocks(spec, task_prompt)
    prompt += (
        "\nReport your verdicts by calling the answer() tool: 'verdicts' "
        "is one entry per PHASE shown, each with phase_index (the PHASE "
        "index shown), "
        f"phase (EXACTLY one of: {labels_line} — return the current label "
        "to confirm it, a different one to correct it), confidence (your "
        "0.0-1.0 certainty in the label YOU return), and an explanation of "
        "<=14 words."
    )
    return prompt


def _verify_answer_spec(phase_names: Sequence[str]) -> AnswerStructured:
    """Build the verifier's answer schema.

    Args:
        phase_names: The judged phase names.

    Returns:
        The ``AnswerStructured`` spec for the verifier's answer() call.
    """
    vocab_verdict = create_model(
        "_VocabVerdict",
        __base__=_Verdict,
        phase=(
            Literal[tuple(phase_names)],
            Field(
                description=(
                    "The correct phase label: the current one to confirm, "
                    "a different one to correct."
                )
            ),
        ),
    )
    verify_chunks = create_model(
        "_VerifyChunks",
        verdicts=(
            list[vocab_verdict],  # type: ignore[valid-type]
            Field(description="One verdict per PHASE shown."),  # type: ignore[valid-type]
        ),
        # declared so Scout does not auto-add its stock explanation field
        explanation=(
            str,
            Field(description="One or two sentences on your review."),
        ),
    )
    return AnswerStructured(type=verify_chunks)


def _verify_user_prompt(
    phases: Sequence[StitchedPhase],
    ids: Sequence[int],
    digests: Sequence[Digest],
) -> str:
    """Render one verifier chunk: a block per selected phase.

    Args:
        phases: All stitched phases (neighbour labels come from here).
        ids: The phase indices in this chunk.
        digests: All turn digests (the evidence lines).

    Returns:
        The chunk user prompt.
    """
    by_turn = {d.turn: d for d in digests}
    blocks = []
    for k in ids:
        p = phases[k]
        prev_l = phases[k - 1].phase if k > 0 else "(none)"
        next_l = phases[k + 1].phase if k + 1 < len(phases) else "(none)"
        members = capped_lines(
            [
                digest_line(by_turn[turn])
                for turn in range(p.turn_start, p.turn_end + 1)
                if turn in by_turn
            ],
            EVIDENCE_LINES,
        )
        lines = "\n".join(members)
        blocks.append(
            f"PHASE {k} [current={p.phase} confidence={p.confidence} "
            f"prev={prev_l} next={next_l}]:\n{lines}"
        )
    return "Review these phases:\n\n" + "\n\n".join(blocks)


def _apply_verdicts(
    digest_judgements: list[ConsensusJudgement],
    phases: list[StitchedPhase],
    verdicts: dict[int, _Verdict],
    reasons: dict[int, Trigger],
    audit: VerifierAudit,
    verifier_model: str | None = None,
) -> tuple[list[StitchedPhase], VerifierAudit]:
    """Apply verifier verdicts: repairs, re-stitch, flags, audit counts.

    A differing label is applied only at verifier confidence >= 0.6.
    Applied overturns rewrite the phase's judgement in place; the
    review lands on the phase as a ``VerifierReview``
    (``StitchedPhase.verifier``) - original label/confidence always
    recorded, ``overturned`` True when the relabel applied.

    Args:
        digest_judgements: Phase judgements for digests; relabelled in place.
        phases: The pre-verification phases.
        verdicts: phase index -> verdict entry.
        reasons: phase index -> selection trigger, for every selected.
        audit: The audit to fill.

    Returns:
        ``(phases, audit)`` with flags attached.
    """
    selected = sorted(reasons)
    audit.n_no_verdict = len(selected) - len(verdicts)
    checked_ranges: list[tuple[int, int, VerifierReview]] = []
    applied_any = False
    for k in selected:
        verdict = verdicts.get(k)
        if verdict is None:
            continue
        p = phases[k]
        trigger = reasons[k]
        differs = verdict.phase != p.phase
        applied = differs and verdict.confidence >= LOWCONF
        if differs and not applied:
            audit.n_weak_relabel += 1
        explanation = verdict.explanation.strip()[:120]
        if applied:
            audit.n_relabelled += 1
            if trigger == "random_sample":
                audit.n_random_sample_relabelled += 1
            applied_any = True
            for row in digest_judgements:
                if p.turn_start <= row.turn <= p.turn_end and row.basis in (
                    "judged",
                    "filled",
                ):
                    row.phase = verdict.phase
                    row.confidence = verdict.confidence
                    row.explanation = explanation
                    row.source = "verifier"
                    row.confidence_pm = 0.0
                    row.agreement = None
        checked_ranges.append(
            (
                p.turn_start,
                p.turn_end,
                VerifierReview(
                    trigger=trigger,
                    original_label=p.phase,
                    original_confidence=p.confidence,
                    original_explanation=p.explanation,
                    verifier_label=verdict.phase,
                    verifier_confidence=verdict.confidence,
                    verifier_explanation=explanation,
                    verifier_model=verifier_model,
                    overturned=applied,
                ),
            )
        )
    result = stitch_phases(digest_judgements) if applied_any else phases
    for p in result:
        hits = [
            review
            for start, end, review in checked_ranges
            if start <= p.turn_end and p.turn_start <= end  # ranges overlap
        ]
        if not hits:
            continue
        # after a merge the phase must report the verdict that rewrote it
        p.verifier = next((r for r in hits if r.overturned), hits[0])
        if p.verifier.overturned:
            p.confidence_spread = 0.0
            p.confidence_source = "verifier"
    return result, audit
