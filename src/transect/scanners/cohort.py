"""Cohort / k-roll voting: scanner-agnostic pieces.

Two mutually exclusive voting regimes: several models each judging
once (cohort), or one model judging ``k_rolls`` times (k-roll). A
member's identity is ``(model name, roll index)``; roll indices are
0-based.

``cohort_llm_scanner`` is a user-facing helper for the common
scanner shape - one closed-vocab question per unit, answered by
``llm_scanner``.
"""

import asyncio
import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from inspect_ai.model import CachePolicy, ChatMessageUser, Model, get_model
from inspect_scout import (
    AnswerStructured,
    RefusalError,
    Result,
    Scanner,
    Transcript,
    llm_scanner,
)
from pydantic import BaseModel, Field, JsonValue, create_model

logger = logging.getLogger(__name__)

# every verifier selection trigger
Trigger = Literal["low_confidence", "low_agreement", "wedge", "random_sample"]

# who decided a label
LabelSource = Literal["single_judge", "majority_vote", "verifier"]

# outcome of one judge call / review
CallStatus = Literal["ok", "no_answer", "refusal", "error"]

# verifier threshold: the selection trigger (a stated
# confidence or vote agreement below it sends the unit to the
# second-round verifier) and the overturn gate (a verifier verdict
# below it never rewrites a label)
LOWCONF = 0.60


def judge_setup(
    models: Sequence[str],
    k_rolls: int,
    verifier_armed: bool,
    verifier_model: str | None,
) -> dict:
    """The canonical judge-identity block a judged scanner stamps as
    ``value["judge"]`` on every result.

    Consumed by `transect.reliability.detect_regime` at render time.

    On the ``models=None`` default-model path no name is resolvable
    at factory time: the block stamps an empty roster (``n_models``
    stays 1) and ``verifier_same_model`` False, and the frames read
    the judge name from the recorded model usage instead.
    """
    names = list(dict.fromkeys(models))
    regime = "cohort" if len(names) > 1 else ("k_roll" if k_rolls > 1 else "solo")
    same_model = bool(
        verifier_armed
        and verifier_model is not None
        and len(names) == 1
        and verifier_model == names[0]
    )
    return {
        "regime": regime,
        "models": names,
        "n_models": len(names) or 1,
        "k_rolls": k_rolls,
        "verifier_armed": verifier_armed,
        "verifier_model": verifier_model if verifier_armed else None,
        "verifier_same_model": same_model,
    }


# default verify_sample share, shared by both judged scanners; here it
# drives the floor-less per-item draw (`_spot_check`, seeded on
# _SPOT_SEED), decision_phases adds a floor of 3 on top of it
_SPOT_SEED = 20260831
_SPOT_DEFAULT = 0.05

# batch mode's per-unit marker
_BATCH_HEAD = re.compile(r"^\[ITEM (\d+)\]$", re.MULTILINE)
_BATCH_QUESTION_SUFFIX = (
    "\n\nThe content above holds several items, each headed by an "
    "[ITEM n] marker. Answer every marker: one entry per item, "
    "carrying its number."
)


@dataclass(frozen=True)
class Member:
    """One cohort member: a judge model at one roll index.

    ``model`` is kept as passed (a name or a ``Model`` carrying its
    GenerateConfig) - ``resolve()`` when the instance is needed."""

    model: str | Model
    name: str  # the model spec as given (str(model) for instances)
    roll: int  # 0-based; a model's first run is roll 0

    @property
    def key(self) -> tuple[str, int]:
        """The (name, roll) identity key."""
        return (self.name, self.roll)

    def resolve(self) -> Model:
        return self.model if isinstance(self.model, Model) else get_model(self.model)


class Ballot(BaseModel):
    """One member's answer for one judged unit."""

    label: str
    confidence: float | None = None  # tie-break input; None = no claim
    explanation: str | None = None


class UnitVote(BaseModel):
    """The vote outcome for one judged unit."""

    label: str
    agreement: float  # voters on the modal label / n_voting
    n_voting: int
    confidence: float | None = None  # modal-side mean (solo keeps exact)
    confidence_pm: float = 0.0  # modal-side 95%-CI half-width
    explanation: str | None = None  # first modal-side member's


@dataclass(frozen=True)
class _UnitDecision:
    """One judged unit's final answer after the vote and the optional
    second-round review."""

    vote: UnitVote | None
    verifier: dict | None
    label: str | None
    confidence: float | None
    explanation: str | None
    overturned: bool


class VerifierReview(BaseModel):
    """One second-round review of a doubtful judgement."""

    trigger: Trigger
    """Why this judgement was selected for review."""

    original_label: str
    """The judge's label before the review."""

    original_confidence: float | None = None
    """The judge's confidence before the review."""

    original_explanation: str | None = None
    """The judge's explanation before the review."""

    verifier_label: str | None = None
    """The verifier's own label; None when the call returned no verdict."""

    verifier_confidence: float | None = None
    """The verifier's stated confidence in its label."""

    verifier_explanation: str | None = None
    """The verifier's explanation."""

    verifier_model: str | None = None
    """The resolved verifier model name."""

    overturned: bool = False
    """The verifier's differing label was applied - only at verifier
    confidence >= 0.6 (a weak verdict is recorded, never applied)."""

    status: CallStatus = "ok"
    """The review call's outcome: ok / no_answer / refusal / error."""


def cohort_llm_scanner(
    question: str,
    answer: Sequence[str],
    models: str | Model | Sequence[str | Model] | None = None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verifier_model: str | Model | None = None,
    verify_sample: float | None = None,
    cache: bool | CachePolicy = True,
    vocabulary: Mapping[str, str] | Sequence[dict | str] | None = None,
    batch: bool = False,
) -> Scanner[Transcript]:
    """``llm_scanner`` with cohort / k-roll voting and a second-round
    verifier for closed-vocab classification scanners.

    Solo (no models, or one model with ``k_rolls=1``): one inner
    ``llm_scanner`` call with a structured answer spec - Result
    carries the label, the stated confidence, and the explanation.

    Voting regimes: one inner llm_scanner per member (roll-scoped
    caches), fanned out per judged unit. Members answer a structured
    ``{label, confidence, explanation}`` - the confidence feeds the
    tie-break and the vote's confidence stats. Labels are normalized
    snake_case before voting.

    Verifier: with ``verify`` on, a doubtful unit gets a second-round
    review by ``verifier_model``. Triggers: a final confidence below
    0.6 (``low_confidence`` - the judges' own doubt), or a k-roll
    vote agreement below 0.6 (``low_agreement`` - instability across
    rolls). An overturn is applied only at verifier confidence >= 0.6.

    Voting-regime Result contract (both regimes also stamp the loader
    item's own metadata as ``Result.metadata``):

    - ``answer``: the winning label (None when every member
      abstained; also carried as ``cohort.vote.label``).
    - ``explanation``: the first modal-side member's explanation.
    - ``value``: ``{confidence, label_source, status, label_vocab,
      judge_models, judge, "cohort": {ran, members, vote, agreement},
      "verifier": {...}}`` - ``label_vocab`` is the resolved closed
      vocabulary the judge classified against (label, description,
      reserved per entry); ``judge`` is the `judge_setup` identity
      block (regime, roster, roll count, verifier arming), stamped on
      the solo path too, which the frames project as the standard
      judge-identity columns.
      - the top-level fields are the final answer facts (the
      verifier's after an overturn); the cohort record keeps the
      pre-verify vote:

      - ``members``: ``[{model, roll, label, confidence,
        explanation, status, error}]`` - status is ok / refusal /
        error; ``error`` is the reason for an error status (the
        exception type and text of a failed judge call, None otherwise.
      - ``vote``: the full ``UnitVote`` record (dumped); None when
        nobody voted.
      - ``agreement``: a scalar when >= 2 members voted, else None.
      - ``verifier``: present only when the verifier is on.

        - ``{ran: false, model: <verifier name>}`` - nothing
          triggered (the armed stamp; ``model`` is None only on the
          ``models=None`` default-model path, where no name is
          resolvable at factory time).
        - after a review: ``{ran: true}`` + the ``VerifierReview``
          record ``{trigger, original_label / _confidence /
          _explanation, verifier_label / _confidence / _explanation,
          overturned, status}``.
        - ``overturned`` True: the verifier's label is the final
          ``answer``.

    Args:
        question: The llm_scanner question.
        answer: The closed label list.
        models: One model (solo) or several (cohort regime).
            ``None`` = llm_scanner's default model, solo only.
        k_rolls: Rolls of one model (k-roll regime); mutually
            exclusive with a multi-model list, and requires an
            explicit ``models``.
        verify: The second-round verifier. ``None`` = auto: on for a
            single model (solo and k-roll), off for a multi-model
            cohort, ``True`` forces on (raises with a cohort);
            ``False`` = off.
        verifier_model: Verifier judge; ``None`` uses the (first)
            model from ``models``.
        verify_sample: Random spot-check probability per item, on
            top of the doubt triggers - each un-triggered item is
            verifier-reviewed with this probability (deterministic
            per item, seeded on its transcript_id). ``None`` = the
            default 5% (the same rate as decision_phases' default,
            minus its min-3 floor - a floor needs a global view this
            per-item scanner does not have); ``0.0`` = doubt-only. A
            per-item draw, not an exact fraction.
        cache: Base cache setting; later rolls get ``{"roll": r}``
            scopes on top.
        vocabulary: The rubric to record.
        batch: Judge several units per call, any unit shape. The
            loader is responsible for yielding batch-shaped items:
            each packs its units as ``[ITEM n]``-marked content
            (1-based, `batch_item_content`) and declares one fact
            dict per unit, in content order, as
            ``metadata={"items": [...]}``. The facts (e.g.
            ``{"turn": 4}``) merge into that unit's result entry.
            `transect.reasoning_turns` with ``batch=N`` yields this shape
            for reasoning turns, with ids matching its unbatched items.

    Raises:
        TypeError: On a free-string (non-list) answer spec.
        ValueError: On invalid models/k_rolls (see
            ``cohort_members``), incl. ``k_rolls != 1`` with
            ``models=None``; on ``verify=True`` with a multi-model
            cohort (verifier XOR cohort).
    """
    if isinstance(answer, str):
        raise TypeError(
            "cohort_llm_scanner requires a closed label list as the "
            "answer spec - voting over open-vocab free strings is not "
            "defined"
        )
    labels = list(answer)
    vocabulary = _vocabulary_entries(vocabulary)
    multi_model = (
        models is not None and not isinstance(models, (str, Model)) and len(models) > 1
    )
    if verify is True and multi_model:
        raise ValueError(
            "verify=True with a multi-model cohort, the regimes are "
            "mutually exclusive (verifier XOR cohort): the majority "
            "vote is the cohort's correction mechanism"
        )
    verify_on = (not multi_model) if verify is None else bool(verify)
    if verify_sample is not None and not 0.0 <= verify_sample <= 1.0:
        raise ValueError(f"verify_sample must be in [0, 1], got {verify_sample}")
    spot_p = _SPOT_DEFAULT if verify_sample is None else verify_sample
    if models is None and k_rolls != 1:
        raise ValueError(
            "k_rolls requires an explicit models value - "
            "models=None is the solo default-model path"
        )
    members = cohort_members(models, k_rolls) if models is not None else []
    if members and verifier_model is None:
        verifier_model = members[0].model
    judge_identity = judge_setup(
        [m.name for m in members],
        k_rolls,
        verifier_armed=verify_on,
        verifier_model=_model_name(verifier_model),
    )
    if batch:
        return _batch_llm_scanner(
            question,
            labels,
            members,
            verify_on,
            verifier_model,
            verify_sample=spot_p,
            vocabulary=vocabulary,
            cache=cache,
            judge_identity=judge_identity,
        )
    if len(members) <= 1:
        return _verified_llm_scanner(
            question,
            labels,
            members[0].model if members else None,
            verify_on,
            verifier_model,
            verify_sample=spot_p,
            vocabulary=vocabulary,
            cache=cache,
            judge_identity=judge_identity,
        )
    spec = _structured_answer(labels)
    inner = [
        (member, _judge_scanner(question, spec, member.model, member.roll, cache))
        for member in members
    ]

    async def execute(item: Transcript) -> Result:
        results = await asyncio.gather(
            *[scan(item) for _, scan in inner], return_exceptions=True
        )
        _raise_fatal(results)
        records: list[dict[str, JsonValue]] = []
        ballots: list[Ballot] = []
        for (member, _), outcome in zip(inner, results, strict=True):
            label = confidence = explanation = None
            error: str | None = None
            status: CallStatus = "ok"
            if isinstance(outcome, (BaseException, list)):
                status, error = _call_failure(outcome)
            else:
                label, confidence, explanation = _answer_fields(outcome)
                if label is None:
                    status = "error"
                    error = "answered without a usable label"
            records.append(
                {
                    "model": member.name,
                    "roll": member.roll,
                    "label": label,
                    "confidence": confidence,
                    "explanation": explanation,
                    "status": status,
                    "error": error,
                }
            )
            if label is not None:
                ballots.append(
                    Ballot(label=label, confidence=confidence, explanation=explanation)
                )
        # only the k-roll regime reaches the verifier here
        decided = await _vote_and_verify(
            ballots,
            item,
            question,
            labels,
            verify_on,
            verifier_model,
            verify_sample=spot_p,
            cache=cache,
        )
        statuses = [str(record["status"]) for record in records]
        value: dict[str, JsonValue] = {
            "confidence": decided.confidence,
            "label_source": _label_source(decided, solo=False),
            "status": "ok" if decided.label else _failure_status(statuses),
            "label_vocab": cast(JsonValue, list(vocabulary or [])),
            "judge_models": list(dict.fromkeys(m.name for m in members)),
            "judge": cast(JsonValue, judge_identity),
            "cohort": _cohort_block(records, decided.vote),
        }
        if decided.verifier is not None:
            value["verifier"] = decided.verifier
        return Result(
            value=cast(JsonValue, value),
            answer=decided.label,
            label=decided.label,
            explanation=decided.explanation,
            metadata=item.metadata,
        )

    return execute


def batch_item_content(texts: Sequence[str]) -> str:
    """One batch item's content."""
    return "\n\n".join(f"[ITEM {i}]\n{text}" for i, text in enumerate(texts, start=1))


def vote_ballots(ballots: Sequence[Ballot]) -> UnitVote | None:
    """The per-unit vote.

    The modal label wins; a tie goes to the tied side with the higher
    mean confidence; a residual tie (equal counts and equal means)
    goes to the earliest member. A ``None`` return (nobody voted)
    means whatever the caller's abstention policy says.

    Args:
        ballots: The members' ballots, in member order.

    Returns:
        The ``UnitVote``, or None when there are no ballots.
        ``confidence`` is the mean over the modal side's stated
        confidences (a single stated value is kept exact).
    """
    if not ballots:
        return None
    sides: dict[str, list[tuple[int, Ballot]]] = {}
    for order, ballot in enumerate(ballots):
        sides.setdefault(ballot.label, []).append((order, ballot))
    label, side = max(sides.items(), key=_rank_side)
    stated = [b.confidence for _, b in side if b.confidence is not None]
    confidence = None
    if stated:
        confidence = (
            stated[0] if len(stated) == 1 else round(sum(stated) / len(stated), 3)
        )
    return UnitVote(
        label=label,
        agreement=round(len(side) / len(ballots), 3),
        n_voting=len(ballots),
        confidence=confidence,
        confidence_pm=round(ci_half_width(stated), 3) if stated else 0.0,
        explanation=side[0][1].explanation,
    )


def cohort_members(
    judge_models: str | Model | Sequence[str | Model],
    k_rolls: int = 1,
) -> list[Member]:
    """Validated member enumeration.

    Cohort regime = each model once (roll 0); k-roll regime = one
    model, rolls 0..k_rolls-1. ``Member.model`` is kept as given
    (resolution stays with the caller).

    Raises:
        ValueError: On an empty list; on duplicate models (repeated
            rolls are expressed via k_rolls); on a multi-model list
            with k_rolls > 1 (the regimes are mutually exclusive);
            on k_rolls < 1.
    """
    models: list[str | Model] = (
        [judge_models] if isinstance(judge_models, (str, Model)) else list(judge_models)
    )
    if not models:
        raise ValueError("at least one judge model is required")
    if k_rolls < 1:
        raise ValueError("k_rolls >= 1 is required")
    names = [m if isinstance(m, str) else str(m) for m in models]
    if len(set(names)) != len(names):
        raise ValueError(
            "duplicate judge models - repeated rolls of one model are "
            "expressed via k_rolls, not by repeating it in judge_models"
        )
    if len(models) > 1 and k_rolls > 1:
        raise ValueError(
            "cohort and k_rolls are mutually exclusive: use several "
            "models (cohort regime) OR one model with k_rolls > 1 "
            "(k-roll regime), not both"
        )
    if len(models) > 1:
        return [Member(model=m, name=names[i], roll=0) for i, m in enumerate(models)]
    first = models[0]
    if k_rolls > 1 and isinstance(first, Model) and first.config.temperature == 0:
        logger.warning(
            "k_rolls > 1 with temperature 0 - rolls will near-replay a "
            "single answer (like reading the cache N times); set a "
            "non-zero temperature on the judge Model for a meaningful "
            "k-roll study"
        )
    return [Member(model=models[0], name=names[0], roll=r) for r in range(k_rolls)]


def roll_cache(cache: bool | CachePolicy, roll: int) -> bool | CachePolicy:
    """Cache policy for one k-roll member's calls.

    Roll 0 keeps the caller's cache setting unchanged. Later rolls
    get a ``{"roll": <r>}`` cache scope so they hit the API afresh
    instead of replaying roll 0's cache entries (a fresh call only
    re-samples if the judge's temperature is non-zero - the caller's
    Model GenerateConfig responsibility). Cache off stays off -
    every roll hits the API.

    Args:
        cache: The caller's cache setting.
        roll: The member's roll index.

    Returns:
        The member's cache setting.
    """
    if roll == 0 or cache is False:
        return cache
    if isinstance(cache, CachePolicy):
        return CachePolicy(
            expiry=cache.expiry,
            per_epoch=cache.per_epoch,
            scopes={**cache.scopes, "roll": str(roll)},
        )
    return CachePolicy(scopes={"roll": str(roll)})


def ci_half_width(values: list[float]) -> float:
    """95%-CI half-width of ``values`` (sample sd); 0.0 below n=2."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return 1.96 * sd / len(values) ** 0.5


def _batch_llm_scanner(
    question: str,
    labels: list[str],
    members: list[Member],
    verify_on: bool,
    verifier_model: str | Model | None,
    verify_sample: float = 0.0,
    vocabulary: Sequence[dict] | None = None,
    cache: bool | CachePolicy = True,
    judge_identity: dict | None = None,
) -> Scanner[Transcript]:
    """The batch execution path: one structured answer per member
    call covering every marked unit of the item, split into per-unit
    votes. The verifier and the spot-check draw stay per unit."""
    spec = _structured_batch_answer(labels)
    batch_question = question + _BATCH_QUESTION_SUFFIX
    solo = len(members) <= 1
    inner: list[tuple[str | None, int, Scanner[Transcript]]] = (
        [
            (
                member.name,
                member.roll,
                _judge_scanner(batch_question, spec, member.model, member.roll, cache),
            )
            for member in members
        ]
        if members
        else [(None, 0, _judge_scanner(batch_question, spec, None, 0, cache))]
    )

    async def execute(item: Transcript) -> Result:
        declared = (item.metadata or {}).get("items")
        if not isinstance(declared, list) or not declared:
            raise ValueError(
                "cohort_llm_scanner(batch=True) items must declare their "
                "units as one fact dict per unit, in content order, e.g. "
                'metadata={"items": [{"turn": 4, "id": "<tid>:turn-4"}, '
                '{"turn": 5, "id": "<tid>:turn-5"}]} - '
                "transect.reasoning_turns with batch=N yields this shape"
            )
        units: list[dict] = [
            unit if isinstance(unit, dict) else {} for unit in declared
        ]
        content = "\n\n".join(
            text for m in item.messages if (text := getattr(m, "text", None))
        )
        unit_text = _split_batch_content(content)
        outcomes = await asyncio.gather(
            *[scan(item) for _, _, scan in inner], return_exceptions=True
        )
        _raise_fatal(outcomes)
        expected = set(range(1, len(units) + 1))
        parsed = [_member_batch_answers(outcome, expected) for outcome in outcomes]
        call_statuses = [status for _, status, _ in parsed]
        entries: list[dict[str, JsonValue]] = []
        for n, unit in enumerate(units, start=1):
            records: list[dict[str, JsonValue]] = []
            ballots: list[Ballot] = []
            for (name, roll, _), (answers, call_status, call_error) in zip(
                inner, parsed, strict=True
            ):
                ballot = answers.get(n)
                records.append(
                    {
                        "model": name,
                        "roll": roll,
                        "label": ballot.label if ballot else None,
                        "confidence": ballot.confidence if ballot else None,
                        "explanation": ballot.explanation if ballot else None,
                        "status": "no_answer"
                        if call_status == "ok" and ballot is None
                        else call_status,
                        "error": call_error,
                    }
                )
                if ballot is not None:
                    ballots.append(ballot)
            facts = {k: v for k, v in unit.items() if k != "id"}
            unit_item = Transcript(
                # the declared id keys the verifier view and the
                # spot-check seed; the fallback is batching-dependent
                transcript_id=str(unit.get("id") or f"{item.transcript_id}#item-{n}"),
                messages=[ChatMessageUser(content=unit_text.get(n) or content)],
                metadata=facts,
            )
            decided = await _vote_and_verify(
                ballots,
                unit_item,
                question,
                labels,
                verify_on,
                verifier_model,
                verify_sample=verify_sample,
                cache=cache,
            )
            if decided.label:
                unit_status = "ok"
            elif all(r["status"] == "no_answer" for r in records):
                unit_status = "no_answer"
            else:
                unit_status = _failure_status(call_statuses)
            # the unit's declared facts become the entry's own keys
            # (e.g. "turn"); the judged fields win any collision
            entry: dict[str, JsonValue] = {
                **facts,
                "label": decided.label,
                "confidence": decided.confidence,
                "label_source": _label_source(decided, solo),
                "status": unit_status,
                "explanation": decided.explanation,
            }
            if not solo:
                entry["cohort"] = _cohort_block(records, decided.vote)
            if decided.verifier is not None:
                entry["verifier"] = decided.verifier
            entries.append(entry)
        labelled = any(entry["label"] for entry in entries)
        value: dict[str, JsonValue] = {
            "status": "ok" if labelled else _failure_status(call_statuses),
            "label_vocab": cast(JsonValue, list(vocabulary or [])),
            "judge_models": list(dict.fromkeys(m.name for m in members)),
            "judge": cast(JsonValue, judge_identity),
            "items": cast(JsonValue, entries),
        }
        return Result(value=cast(JsonValue, value), metadata=item.metadata)

    return execute


def _member_batch_answers(
    outcome: Result | list | BaseException, expected: set[int]
) -> tuple[dict[int, Ballot], CallStatus, str | None]:
    """One member's batch call parsed to per-unit ballots."""
    if isinstance(outcome, (BaseException, list)):
        status, error = _call_failure(outcome)
        return {}, status, error
    value = outcome.value if isinstance(outcome.value, dict) else {}
    raw = value.get("items")
    if not isinstance(raw, list):
        return {}, "error", "answered without usable per-item entries"
    answers: dict[int, Ballot] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        n = entry.get("item")
        label = _normalize_label(entry.get("label"))
        if not isinstance(n, int) or n not in expected or n in answers:
            continue
        if label is None:
            continue
        stated = entry.get("confidence")
        raw_explanation = entry.get("explanation")
        answers[n] = Ballot(
            label=label,
            confidence=float(stated) if isinstance(stated, (int, float)) else None,
            explanation=raw_explanation if isinstance(raw_explanation, str) else None,
        )
    return answers, "ok", None


def _split_batch_content(text: str) -> dict[int, str]:
    """`batch_item_content` split back per unit ordinal."""
    parts = _BATCH_HEAD.split(text)
    return {
        int(n): body.strip() for n, body in zip(parts[1::2], parts[2::2], strict=True)
    }


def _structured_batch_answer(labels: list[str]) -> AnswerStructured:
    """The batch answer spec: one {item, label, confidence,
    explanation} entry per marked unit."""
    entry_model = create_model(
        "_CohortItemAnswer",
        item=(int, Field(description="The [ITEM n] marker's number.")),
        label=(
            Literal[tuple(labels)],
            Field(description="The single best-fitting label for this item."),
        ),
        confidence=(
            float,
            Field(ge=0.0, le=1.0, description="Your 0.0-1.0 certainty in the label."),
        ),
        explanation=(
            str,
            Field(description="Why this label (<=14 words)."),
        ),
    )
    batch_model = create_model(
        "_CohortBatchAnswer",
        items=(
            list[entry_model],  # type: ignore[valid-type]
            Field(description="One entry per [ITEM n] marker; answer every marker."),
        ),
    )
    return AnswerStructured(type=batch_model)


def _verified_llm_scanner(
    question: str,
    labels: list[str],
    model: str | Model | None,
    verify_on: bool,
    verifier_model: str | Model | None,
    verify_sample: float = 0.0,
    vocabulary: Sequence[dict] | None = None,
    cache: bool | CachePolicy = True,
    judge_identity: dict | None = None,
) -> Scanner[Transcript]:
    """The solo execution path: one llm_scanner call per item with an
    optional second-round verifier; ``judge_identity`` is the
    factory's `judge_setup` block, stamped verbatim."""
    inner = llm_scanner(
        question=question,
        answer=_structured_answer(labels),
        model=model,
        cache=cache,
    )

    async def execute(item: Transcript) -> Result:
        outcome = await inner(item)
        assert not isinstance(outcome, list)  # per-item inner
        value = outcome.value if isinstance(outcome.value, dict) else {}
        label, confidence, explanation = _answer_fields(outcome)
        final_label = label
        verifier: dict[str, JsonValue] | None = (
            {"ran": False, "model": _model_name(verifier_model)} if verify_on else None
        )
        if verify_on and label is not None:
            trigger: Trigger | None = _verify_trigger(confidence, None)
            if trigger is None and _spot_check(item, verify_sample):
                trigger = "random_sample"
            if trigger is not None:
                review = await _verify_judgement(
                    item,
                    question,
                    labels,
                    verifier_model,
                    cache,
                    label=label,
                    confidence=confidence,
                    explanation=explanation,
                    trigger=trigger,
                )
                verifier = {"ran": True, **review.model_dump()}
                if review.overturned:
                    final_label = review.verifier_label
                    explanation = review.verifier_explanation
                    value = {**value, "confidence": review.verifier_confidence}
        label_source: LabelSource | None = None
        if final_label:
            label_source = (
                "verifier"
                if verifier is not None and verifier.get("overturned")
                else "single_judge"
            )
        value = {
            **value,
            "label_source": label_source,
            "label_vocab": cast(JsonValue, list(vocabulary or [])),
            "judge": cast(JsonValue, judge_identity),
        }
        if verifier is not None:
            value["verifier"] = verifier
        return Result(
            value=cast(JsonValue, value),
            answer=final_label,
            label=final_label,
            explanation=explanation,
            metadata=item.metadata,
        )

    return execute


def _vocabulary_entries(
    vocabulary: Mapping[str, str] | Sequence[dict | str] | None,
) -> list[dict] | None:
    """The accepted vocabulary shapes normalised to entry dicts."""
    if vocabulary is None:
        return None
    if isinstance(vocabulary, Mapping):
        return [
            {"label": label, "description": description}
            for label, description in vocabulary.items()
        ]
    return [
        entry if isinstance(entry, dict) else {"label": str(entry)}
        for entry in vocabulary
    ]


def _spot_check(item: Transcript, verify_sample: float) -> bool:
    """Deterministic per-item spot-check draw: stable across re-runs
    (seeded on the item's transcript_id), independent across items."""
    if verify_sample <= 0:
        return False
    rng = random.Random(f"{_SPOT_SEED}:{item.transcript_id}")
    return rng.random() < verify_sample


def _model_name(model: str | Model | None) -> str | None:
    """The stamped judge-model name (the same str() the reviews use)."""
    return str(model) if model is not None else None


def _verify_trigger(
    confidence: float | None, agreement: float | None
) -> Literal["low_confidence", "low_agreement"] | None:
    """The verifier's selection rule."""
    if confidence is not None and confidence < LOWCONF:
        return "low_confidence"
    if agreement is not None and agreement < LOWCONF:
        return "low_agreement"
    return None


async def _verify_judgement(
    item: Transcript,
    question: str,
    labels: list[str],
    verifier_model: str | Model | None,
    cache: bool | CachePolicy,
    *,
    label: str,
    confidence: float | None,
    explanation: str | None,
    trigger: Trigger,
) -> VerifierReview:
    """One second-round review of a doubtful judgement.

    The verifier sees the same content plus the first-round
    answer, and re-answers with the same structured spec. A differing
    label overturns only at verifier confidence >= 0.6.

    Returns:
        The complete ``VerifierReview`` record (labels,
        confidences, and explanations).
    """
    review = (
        "You are a second-round reviewer. A first-round judge answered "
        "the question below about this content; decide the correct "
        "label yourself — return the earlier label if it holds, the "
        "correct one if not.\n\n"
        f"{question}\n\n"
        f"First-round answer: {label} (confidence {confidence}); "
        f"explanation: {explanation}"
    )
    result = VerifierReview(
        trigger=trigger,
        original_label=label,
        original_confidence=confidence,
        original_explanation=explanation,
        verifier_model=str(verifier_model) if verifier_model is not None else None,
    )
    scan = llm_scanner(
        question=review,
        answer=_structured_answer(labels),
        model=verifier_model,
        cache=cache,
    )
    try:
        outcome = await scan(item)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except RefusalError:
        result.status = "refusal"
        return result
    except Exception:
        result.status = "error"
        return result
    # per-item inner never produces a multi-result list
    if isinstance(outcome, list):
        result.status = "error"
        return result
    v_label, v_confidence, v_explanation = _answer_fields(outcome)
    if v_label is None:
        result.status = "error"
        return result
    result.verifier_label = v_label
    result.verifier_confidence = v_confidence
    result.verifier_explanation = v_explanation
    result.overturned = (
        v_label != label and v_confidence is not None and v_confidence >= LOWCONF
    )
    return result


async def _vote_and_verify(
    ballots: Sequence[Ballot],
    item: Transcript,
    question: str,
    labels: list[str],
    verify_on: bool,
    verifier_model: str | Model | None,
    verify_sample: float = 0.0,
    cache: bool | CachePolicy = True,
) -> _UnitDecision:
    """One unit's decision, shared by the per-item and batch paths:
    the ballots' vote, then the verifier's optional second round,
    with an overturn applied to the final facts."""
    vote = vote_ballots(ballots)
    label = vote.label if vote else None
    confidence = vote.confidence if vote else None
    explanation = vote.explanation if vote else None
    verifier: dict[str, JsonValue] | None = (
        {"ran": False, "model": _model_name(verifier_model)} if verify_on else None
    )
    if vote and verify_on:
        trigger: Trigger | None = _verify_trigger(
            vote.confidence,
            vote.agreement if vote.n_voting >= 2 else None,
        )
        if trigger is None and _spot_check(item, verify_sample):
            trigger = "random_sample"
        if trigger is not None:
            review = await _verify_judgement(
                item,
                question,
                labels,
                verifier_model,
                cache,
                label=vote.label,
                confidence=vote.confidence,
                explanation=vote.explanation,
                trigger=trigger,
            )
            verifier = {"ran": True, **review.model_dump()}
            if review.overturned:
                label = review.verifier_label
                confidence = review.verifier_confidence
                explanation = review.verifier_explanation
    overturned = bool(verifier and verifier.get("overturned"))
    return _UnitDecision(vote, verifier, label, confidence, explanation, overturned)


def _cohort_block(
    records: list[dict[str, JsonValue]], vote: UnitVote | None
) -> dict[str, JsonValue]:
    """The ``cohort`` record both voting paths stamp."""
    return {
        "ran": True,
        "members": cast(JsonValue, records),
        "vote": vote.model_dump() if vote else None,
        "agreement": vote.agreement if vote and vote.n_voting >= 2 else None,
    }


def _label_source(decided: _UnitDecision, solo: bool) -> LabelSource | None:
    """Who decided the unit's final label; None when nobody did."""
    if not decided.label:
        return None
    if decided.overturned:
        return "verifier"
    return "single_judge" if solo else "majority_vote"


def _failure_status(statuses: Sequence[str]) -> CallStatus:
    """The status when no label was decided."""
    return "refusal" if statuses and all(s == "refusal" for s in statuses) else "error"


def _raise_fatal(outcomes: Sequence[object]) -> None:
    """Cancellation/exit must cancel the scan."""
    for outcome in outcomes:
        if isinstance(outcome, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise outcome


def _judge_scanner(
    question: str,
    spec: AnswerStructured,
    model: str | Model | None,
    roll: int,
    cache: bool | CachePolicy,
) -> Scanner[Transcript]:
    """One member's inner llm_scanner, on its roll-scoped cache."""
    return llm_scanner(
        question=question,
        answer=spec,
        model=model,
        cache=roll_cache(cache, roll),
    )


def _structured_answer(labels: list[str]) -> AnswerStructured:
    """The answer spec: label + confidence."""
    answer_model = create_model(
        "_CohortAnswer",
        label=(
            Literal[tuple(labels)],
            Field(description="The single best-fitting label."),
        ),
        confidence=(
            float,
            Field(ge=0.0, le=1.0, description="Your 0.0-1.0 certainty in the label."),
        ),
        explanation=(
            str,
            Field(description="Why this label (<=14 words)."),
        ),
    )
    return AnswerStructured(type=answer_model)


def _normalize_label(raw: object) -> str | None:
    """snake_case label normalization."""
    if not (isinstance(raw, str) and raw.strip()):
        return None
    return "_".join(raw.strip().lower().split())


def _call_failure(outcome: list | BaseException) -> tuple[CallStatus, str | None]:
    """A failed judge call's (status, error)."""
    if isinstance(outcome, RefusalError):
        return "refusal", None
    if isinstance(outcome, BaseException):
        return "error", f"{type(outcome).__name__}: {outcome}"
    return "error", "unexpected multi-result answer shape"


def _answer_fields(outcome: Result) -> tuple[str | None, float | None, str | None]:
    """A Result's (label, confidence, explanation), the label
    normalized snake_case."""
    label = _normalize_label(outcome.label)
    if label is None:
        return None, None, None
    value = outcome.value if isinstance(outcome.value, dict) else {}
    stated = value.get("confidence")
    confidence = float(stated) if isinstance(stated, (int, float)) else None
    explanation = outcome.explanation if isinstance(outcome.explanation, str) else None
    return label, confidence, explanation


def _rank_side(item: tuple[str, list[tuple[int, Ballot]]]) -> tuple[int, float, int]:
    """Vote ordering: count, then mean confidence, then earliest member."""
    confidences = [b.confidence or 0.0 for _, b in item[1]]
    return (
        len(item[1]),
        sum(confidences) / len(confidences),
        -min(order for order, _ in item[1]),
    )
