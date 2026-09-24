"""Shared pieces of the decision_phases scanner family."""

from collections.abc import Sequence
from typing import Literal

from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
)
from inspect_scout import AnswerStructured, RefusalError, generate_answer
from pydantic import BaseModel, Field, JsonValue

from transect.scanners.cohort import LabelSource, VerifierReview, ci_half_width
from transect.spec import Phase, Spec

SNIPPET_CHARS = 300  # per-digest text cap shown to the judge

# per-phase evidence cap (digest lines shown to a judge)
EVIDENCE_LINES = 200

_CONTEXT_GUARD = (
    "Use it to recognise phases; do NOT invent moves the turns do not show."
)

_DEFAULT_OPS_PHASE = Phase(
    label="ops",
    description="Operational, setup, coordination, or plumbing work",
    ops=True,
)

_NONE_PHASE = Phase(
    label="none_of_the_above",
    description=(
        "Activity that fits none of the declared phases and is not operational work"
    ),
)


class Digest(BaseModel):
    """One reasoning-bearing turn, as shown to the judge."""

    turn: int
    text: str = ""
    tools: list[str] = []
    delegations: list[str] = []
    event_id: str | None = None  # source model event uuid (viewer anchor)


Basis = Literal["judged", "filled", "no_answer", "refusal", "missing_turn"]
TurnBasis = Literal[
    "judged", "filled", "attributed", "no_answer", "refusal", "missing_turn"
]


class DigestJudgement(BaseModel):
    """One judge's statement about one digest turn."""

    turn: int
    phase: str | None = None
    confidence: float | None = None
    explanation: str | None = None
    basis: Basis


class TurnGroup(BaseModel):
    """One narrative sub-section of a phase.

    JUDGE-FACING: the Field descriptions render in the answer() tool."""

    turn_start: int = Field(
        description="First turn index of this group (inclusive, as shown)."
    )
    turn_end: int = Field(
        description="Last turn index of this group (inclusive, as shown)."
    )
    title: str = Field(description="Short section title (a few words).")
    gist: str = Field(description="One-line gist of the section.")


class PhaseReview(BaseModel):
    """One selected original phase, preserved independently of display merging.

    A missing verdict retains the original facts with status
    ``no_answer`` or ``refusal`` and no verifier fields. This record
    is not judge-facing.
    """

    original_phase_index: int
    turn_start: int
    turn_end: int
    review: VerifierReview


NarrationGroupStatus = Literal[
    "complete", "invalid_partition", "empty_groups", "no_narrative", "not_run"
]


class StitchedPhase(BaseModel):
    """One phase: consecutive same-label segments merged across chunk boundaries."""

    phase: str = Field(description="The label shared by the merged judgements.")
    turn_start: int = Field(description="First digest turn of the phase (inclusive).")
    turn_end: int = Field(description="Last digest turn of the phase (inclusive).")
    n_turns: int = Field(
        description="Number of digest turns in the phase (the "
        "reasoning-bearing turns the judge saw)."
    )
    confidence: float = Field(
        description="Mean over member judgements (fills included, at 0.3)."
    )
    min_confidence: float = Field(
        description="Minimum over members - the verifier's selection signal."
    )
    min_agreement: float | None = Field(
        default=None,
        description=(
            "Minimum per-turn vote agreement over members carrying one "
            "(voting regimes); None for solo rows."
        ),
    )
    confidence_spread: float = Field(
        default=0.0,
        description=(
            "95%-CI half-width of the phase's per-turn confidences "
            "0.0 after a verifier overturn."
        ),
    )
    judge_agreement: float | None = Field(
        default=None,
        description=(
            "Mean per-turn vote agreement over members; None for solo judges."
        ),
    )
    confidence_source: LabelSource = Field(
        default="single_judge",
        description="Whose confidence the phase carries.",
    )
    explanation: str = Field(
        description=("The first contributing segment's explanation.")
    )
    verifier: VerifierReview | None = Field(
        default=None,
        description=(
            "A representative completed review overlapping this phase; None "
            "when no completed verdict overlaps. Original selected units "
            "live in verifier_reviews."
        ),
    )
    verifier_reviews: list[PhaseReview] = Field(
        default_factory=list,
        description=(
            "Original phase review units overlapping this displayed phase. "
            "An empty list means none selected. The singular verifier is "
            "representative only."
        ),
    )
    headline: str = Field(
        default="",
        description="Narrator: one sentence on what the agent did.",
    )
    summary: str = Field(
        default="",
        description="Narrator: 2-3 descriptive sentences ('' on fallback).",
    )
    turn_groups: list[TurnGroup] = Field(
        default_factory=list,
        description="Narrator: gapless partition of the phase's turn range.",
    )
    narration_group_status: NarrationGroupStatus = Field(
        default="not_run",
        description=(
            "Group partition outcome; complete describes partition "
            "coordinates, not factual correctness."
        ),
    )
    anchor_event_id: str | None = Field(
        default=None,
        description=("Viewer anchor: the first member model event's uuid."),
    )


def resolve_phases(spec: Spec) -> tuple[list[Phase], str]:
    """Resolve the phase definitions shown to the judge, guaranteeing
    an operational bucket and a none_of_the_above escape category.

    Args:
        spec: Supplies the user's phase definitions.

    Returns:
        ``(phases, ops_name)`` - the spec's phases, plus the reserved
        "ops" phase when none is flagged ``ops`` in the spec, plus
        the reserved "none_of_the_above" phase when the spec does
        not declare one itself.
    """
    phases = list(spec.phases)
    ops_name = next((p.label for p in phases if p.ops), None)
    if ops_name is None:  # unflagged "ops" is still OUR reserved name
        ops_name = next(
            (p.label for p in phases if p.label == _DEFAULT_OPS_PHASE.label),
            None,
        )
    if ops_name is None:
        phases.append(_DEFAULT_OPS_PHASE)
        ops_name = _DEFAULT_OPS_PHASE.label
    if all(p.label != _NONE_PHASE.label for p in phases):
        phases.append(_NONE_PHASE)
    return phases, ops_name


def stitch_phases(
    digest_judgements: Sequence[DigestJudgement],
) -> list[StitchedPhase]:
    """Stitch ok-labelled digest judgements into contiguous same-phase ranges.

    Consecutive judgements sharing a label merge into one phase, across
    chunk boundaries. An unjudged turn (basis refusal / no_answer /
    missing_turn) closes the current phase.

    Args:
        digest_judgements: Per-digest-turn judgements.

    Returns:
        ``StitchedPhase`` entries in turn order, each with:

        - ``phase``: the label shared by the merged judgements.
        - ``turn_start`` / ``turn_end``: inclusive digest-turn range.
        - ``n_turns``: digest turns in the phase (the reasoning-
          bearing turns the judge saw).
        - ``confidence``: mean over member judgements (fills included,
          at their 0.3).
        - ``min_confidence``: minimum over members - the verifier's
          selection signal.
        - ``explanation``: the first contributing segment's explanation
          (judge provenance; may predate a later verifier overturn).
    """
    phases: list[StitchedPhase] = []
    members: list[DigestJudgement] = []

    def _close() -> None:
        if not members:
            return
        # ok rows always carry confidence; `or 0.0` narrows the Optional
        confidences = [m.confidence or 0.0 for m in members]
        # getattr, not m.agreement: the field lives on ConsensusJudgement
        # (phases_cohort), which this module cannot import without a cycle
        agreements = [
            a for m in members if (a := getattr(m, "agreement", None)) is not None
        ]
        phases.append(
            StitchedPhase(
                phase=members[0].phase or "",
                turn_start=members[0].turn,
                turn_end=members[-1].turn,
                n_turns=len(members),
                confidence=round(sum(confidences) / len(confidences), 3),
                min_confidence=round(min(confidences), 3),
                min_agreement=round(min(agreements), 3) if agreements else None,
                confidence_spread=round(ci_half_width(confidences), 3),
                judge_agreement=(
                    round(sum(agreements) / len(agreements), 3) if agreements else None
                ),
                explanation=members[0].explanation or "",
            )
        )
        members.clear()

    for row in digest_judgements:
        if row.basis not in ("judged", "filled") or row.phase is None:
            _close()  # an unjudged row ends the current run
            continue
        if members and members[-1].phase != row.phase:
            _close()
        members.append(row)
    _close()
    return phases


def context_blocks(spec: Spec, task_prompt: str) -> str:
    """Render the two standard prompt context blocks.

    TASK CONTEXT (the agent's own task prompt) and ADDITIONAL CONTEXT
    (``spec.context``), each guarded against invention; "" when a
    block's source is empty.

    Args:
        spec: Supplies the user context.
        task_prompt: The agent's own task prompt ("" omits the block).

    Returns:
        The rendered blocks ("" when both sources are empty).
    """
    out = ""
    if task_prompt:
        out += (
            f"\nTASK CONTEXT (the task the agent was given; "
            f"{_CONTEXT_GUARD}):\n{task_prompt}\n"
        )
    if spec.context:
        out += (
            f"\nADDITIONAL CONTEXT (user-supplied; {_CONTEXT_GUARD}):\n{spec.context}\n"
        )
    return out


def vocab_lines(phase_defs: Sequence[Phase]) -> str:
    """Render the phase vocabulary as ``- name: description`` lines."""
    return "\n".join(
        f"- {p.label}: {p.description}" if p.description else f"- {p.label}"
        for p in phase_defs
    )


async def call_judge(
    judge: Model,
    answer: AnswerStructured,
    system: str,
    user: str,
    cache: bool | CachePolicy,
) -> tuple[dict[str, JsonValue] | None, Literal["ok", "no_answer", "refusal"]]:
    """Run one structured-answer judge call.

    Args:
        judge: The judge model.
        answer: The ``AnswerStructured`` spec.
        system: The system prompt.
        user: The user prompt.
        cache: Caching for the first attempt (the retry is uncached).

    Failure handling, per mode:

    - refusal: retried inside the call by Scout's own refusal loop
      (its default count, same as the sub-agent path).
    - no_answer (schema loop exhausted, or an empty answer): one
      retry under a separate cache scope if ``cache=True``.

    call_judge overrides the ``cache`` field in judge.config.

    Returns:
        ``(value, status)`` - the answer's raw value dict with status
        "ok" when every non-``explanation`` field is non-empty;
        otherwise ``(None, <last attempt's failure>)`` with
        "no_answer" (schema loop exhausted or an empty answer) or
        "refusal".
    """
    prompt: list[ChatMessage] = [
        ChatMessageSystem(content=system),
        ChatMessageUser(content=user),
    ]
    attempts: list[bool | CachePolicy] = [cache]
    retry = _retry_cache(cache)
    if retry is not None:
        attempts.append(retry)
    for attempt_cache in attempts:
        try:
            result = await generate_answer(
                prompt,
                answer,
                model=judge,
                config=judge.config.merge(GenerateConfig(cache=attempt_cache)),
            )
        except RefusalError:
            return None, "refusal"
        value = result.value if isinstance(result.value, dict) else {}
        payload = {k: v for k, v in value.items() if k != "explanation"}
        if payload and all(payload.values()):
            return value, "ok"
    return None, "no_answer"


def _retry_cache(cache: bool | CachePolicy) -> CachePolicy | None:
    """Cache policy for the second judge attempt.

    Args:
        cache: The caller's cache setting for the first attempt.

    Returns:
        The retry's CachePolicy, or None when caching is off.
    """
    if cache is False:
        return None
    if isinstance(cache, CachePolicy):
        return CachePolicy(
            expiry=cache.expiry,
            per_epoch=cache.per_epoch,
            scopes={**cache.scopes, "transect_retry": "1"},
        )
    return CachePolicy(scopes={"transect_retry": "1"})


def digest_line(d: Digest) -> str:
    """Render one digest as a single prompt line: index, text, tool
    names, then each delegation behind a [DELEGATES] marker."""
    parts = [f"{d.turn}:"]
    if d.text:
        parts.append(d.text.replace("\n", " "))
    if d.tools:
        parts.append(f"(tools: {', '.join(d.tools)})")
    for goal in d.delegations:
        parts.append(f"[DELEGATES] {goal}".replace("\n", " "))
    return " ".join(parts)


def humanise_phase(label: str) -> str:
    """A phase label as a human title: ``initial_recon`` -> ``Initial recon``."""
    return str(label).replace("_", " ").capitalize()
