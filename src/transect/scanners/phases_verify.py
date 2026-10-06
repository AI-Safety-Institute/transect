"""Second-round verifier for decision_phases."""

import random
from collections.abc import Mapping, Sequence
from typing import Literal

from inspect_ai.model import CachePolicy, Model
from inspect_scout import AnswerStructured
from pydantic import BaseModel, Field, ValidationError, create_model

from transect.scanners.cohort import _SPOT_DEFAULT, LOWCONF, Trigger, VerifierReview
from transect.scanners.phases_cohort import ConsensusJudgement
from transect.scanners.phases_common import (
    Digest,
    PhaseReview,
    StitchedPhase,
    call_judge,
    context_blocks,
    gather_judge_calls,
    phase_evidence,
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

    Judge-facing: the Field descriptions render in the answer() tool."""

    phase_index: int = Field(description="The PHASE index being reviewed, as shown.")
    phase: str  # judge-facing wording lives on the _VocabVerdict override
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your 0.0-1.0 certainty in the label you return.",
    )
    explanation: str = Field(description="Why (<=14 words).")


class _ReviewAttempt(BaseModel):
    """One selected phase's review attempt: its trigger plus outcome.

    ``verdict`` is set exactly when ``status`` is "ok"."""

    trigger: Trigger
    verdict: _Verdict | None = None
    status: Literal["ok", "no_answer", "refusal"] = "no_answer"


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
        original review units and representative verifier flags, and the
        counts-only audit block.
    """
    attempts = {
        k: _ReviewAttempt(trigger=trigger)
        for k, trigger in select_for_verify(phases, sample=sample).items()
    }
    selected = sorted(attempts)
    triggers = [attempt.trigger for attempt in attempts.values()]
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
    ids_chunks = [
        selected[offset : offset + chunk] for offset in range(0, len(selected), chunk)
    ]
    by_turn = {d.turn: d for d in digests}
    # throttled by inspect's own connection limit
    results = await gather_judge_calls(
        call_judge(
            judge, answer, system, _verify_user_prompt(phases, ids, by_turn), cache
        )
        for ids in ids_chunks
    )
    for ids, (value, status) in zip(ids_chunks, results, strict=True):
        if status != "ok":
            for k in ids:
                attempts[k].status = status
            continue  # whole chunk stays no-verdict
        raw_verdicts = (value or {}).get("verdicts")
        for item in raw_verdicts if isinstance(raw_verdicts, list) else []:
            try:
                verdict = _Verdict.model_validate(item)
            except ValidationError:
                continue
            attempt = attempts.get(verdict.phase_index)
            if verdict.phase_index in ids and attempt and attempt.verdict is None:
                attempt.verdict = verdict
                attempt.status = "ok"
    return _apply_verdicts(digest_judgements, phases, attempts, audit, str(judge))


def verify_system_prompt(spec: Spec, task_prompt: str) -> str:
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
    by_turn: Mapping[int, Digest],
) -> str:
    """Render one verifier chunk: a block per selected phase.

    Args:
        phases: All stitched phases (neighbour labels come from here).
        ids: The phase indices in this chunk.
        by_turn: All turn digests (the evidence lines), keyed by turn.

    Returns:
        The chunk user prompt.
    """
    blocks = []
    for k in ids:
        p = phases[k]
        prev_l = phases[k - 1].phase if k > 0 else "(none)"
        next_l = phases[k + 1].phase if k + 1 < len(phases) else "(none)"
        blocks.append(
            f"PHASE {k} [current={p.phase} confidence={p.confidence} "
            f"prev={prev_l} next={next_l}]:\n{phase_evidence(p, by_turn)}"
        )
    return "Review these phases:\n\n" + "\n\n".join(blocks)


def _apply_verdicts(
    digest_judgements: list[ConsensusJudgement],
    phases: list[StitchedPhase],
    attempts: dict[int, _ReviewAttempt],
    audit: VerifierAudit,
    verifier_model: str | None,
) -> tuple[list[StitchedPhase], VerifierAudit]:
    """Apply verifier verdicts: repairs, re-stitch, flags, audit counts.

    A differing label is applied only at verifier confidence >= 0.6.
    Applied overturns rewrite the phase's judgement in place; the
    original units survive in ``StitchedPhase.verifier_reviews``.
    The singular ``verifier`` is representative display provenance;
    it must not be counted as the whole review population.

    Args:
        digest_judgements: Phase judgements for digests; relabelled in place.
        phases: The pre-verification phases.
        attempts: phase index -> review attempt, for every selected phase.
        audit: The audit to fill.
        verifier_model: The verifier's model name, stamped on every review.

    Returns:
        ``(phases, audit)`` with flags attached.
    """
    audit.n_no_verdict = sum(attempt.status != "ok" for attempt in attempts.values())
    reviews: list[PhaseReview] = []
    applied_any = False
    for k in sorted(attempts):
        attempt = attempts[k]
        verdict = attempt.verdict
        p = phases[k]
        review = VerifierReview(
            trigger=attempt.trigger,
            original_label=p.phase,
            original_confidence=p.confidence,
            original_explanation=p.explanation,
            verifier_model=verifier_model,
        )
        if verdict is None:
            review.status = attempt.status
        else:
            differs = verdict.phase != p.phase
            applied = differs and verdict.confidence >= LOWCONF
            if differs and not applied:
                audit.n_weak_relabel += 1
            explanation = verdict.explanation.strip()[:120]
            if applied:
                audit.n_relabelled += 1
                if attempt.trigger == "random_sample":
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
            review.verifier_label = verdict.phase
            review.verifier_confidence = verdict.confidence
            review.verifier_explanation = explanation
            review.overturned = applied
        reviews.append(
            PhaseReview(
                original_phase_index=k,
                turn_start=p.turn_start,
                turn_end=p.turn_end,
                review=review,
            )
        )
    result = stitch_phases(digest_judgements) if applied_any else phases
    for p in result:
        p.verifier_reviews = [
            review
            for review in reviews
            if review.turn_start <= p.turn_end and p.turn_start <= review.turn_end
        ]
        hits = [
            review.review
            for review in p.verifier_reviews
            if review.review.status == "ok"
        ]
        if not hits:
            continue
        # after a merge the phase must report the verdict that rewrote it
        p.verifier = next((r for r in hits if r.overturned), hits[0])
        if p.verifier.overturned:
            p.confidence_spread = 0.0
            p.confidence_source = "verifier"
    return result, audit
