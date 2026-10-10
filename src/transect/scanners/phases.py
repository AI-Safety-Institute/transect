"""decision_phases: phase segmentation of a run."""

import json
import logging
from bisect import bisect_right
from collections.abc import Collection, Sequence
from typing import Any, Literal, cast

from inspect_ai.model import CachePolicy, Model, get_model
from inspect_ai.tool import ToolCall
from inspect_scout import AnswerStructured, Result, Scanner, Transcript, scanner
from pydantic import BaseModel, Field, JsonValue, create_model

from transect.scanners.cohort import (
    LabelSource,
    cohort_members,
    judge_setup,
    resolve_verify,
    roll_cache,
)
from transect.scanners.helpers import (
    Lanes,
    message_reasoning,
    orchestrator_turns,
    span_task_text,
    strip_subagent_scaffold,
)
from transect.scanners.phases_cohort import (
    Cohort,
    CohortAgreement,
    CohortMember,
    ConsensusJudgement,
    TurnVote,
    vote_turns,
)
from transect.scanners.phases_common import (
    FINAL_TEXT_CHARS,
    SNIPPET_CHARS,
    TOOL_CALL_CHARS,
    TOOL_RESULT_CHARS,
    USER_CHARS,
    Digest,
    DigestJudgement,
    StitchedPhase,
    ToolDigest,
    TurnBasis,
    call_judge,
    check_digest_caps,
    clip,
    context_blocks,
    digest_line,
    gather_judge_calls,
    is_labelled,
    resolve_phases,
    stitch_phases,
    vocab_lines,
)
from transect.scanners.phases_narrate import NarratorAudit, narrate_phases
from transect.scanners.phases_verify import VerifierAudit, verify_phases
from transect.spec import Spec

logger = logging.getLogger(__name__)

TASK_PROMPT_CHARS = 2400  # agent-task-prompt context cap

_OPENCLAW_FAILED_TURN_PREFIX = "[assistant turn failed"  # placeholder assistant turn
_OPENCLAW_SCAFFOLD_USER_MESSAGES = ("[openclaw heartbeat poll]",)  # exact matches

_SYSTEM_HEAD = (
    "You are segmenting an autonomous agent's turns (its [THINKING] "
    "reasoning, its own text, its [CALL] tool calls with their outputs, "
    "AND the [DELEGATES] tasks it hands to sub-agents) into CONTIGUOUS "
    "PHASES of the task it is working on. Each phase is a run of "
    "consecutive turns doing ONE activity. Label each phase with the "
    "single best-fitting phase name:\n"
)

_RULES = (
    "\n\nRULES:\n"
    "1. Judge ONLY on the text shown. Decide the BOUNDARY and the LABEL "
    "together — a phase boundary is where the activity changes.\n"
    "2. Phases MUST be contiguous and cover EVERY turn index shown, in "
    "order, with no gaps and no overlaps. A long stretch of one activity "
    "is ONE phase; a brief detour is its own short phase.\n"
    "3. Operational / setup / coordination / plumbing turns belong in "
    "'{ops}' — do NOT force them into a substantive component.\n"
    "4. Delegation turns (marked [DELEGATES]) are the agent handing work "
    "to a sub-agent — classify the phase by the delegated task's purpose.\n"
    "5. [THINKING] text is the agent's internal reasoning — treat it as "
    "evidence of the turn's activity, same as its visible text.\n"
    "6. [CALL] is a tool call the agent made in that turn and [RESULT] or "
    "[ERROR] its recorded output — treat them as evidence of the turn's "
    "activity, same as its visible text.\n"
    "7. [USER] is a message the agent received just before that turn — "
    "context for the turn, NOT the agent's own activity.\n"
    "8. confidence is your 0.0-1.0 certainty in BOTH the boundary and the "
    "label of the phase.\n"
)


class _Segment(BaseModel):
    """One judged phase.

    Judge-facing: the Field descriptions render in the answer() tool."""

    turn_start: int = Field(description=("First turn index of this phase (inclusive)."))
    turn_end: int = Field(description=("Last turn index of this phase (inclusive)."))
    phase: str  # judge-facing wording lives on the _VocabSegment override
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=("Judge's 0.0-1.0 certainty in both the boundary and the label."),
    )
    explanation: str = Field(
        description=("Why this run of turns is this phase (<=14 words).")
    )


class PhaseTurn(BaseModel):
    """One row of the dense ``turns`` surface."""

    turn: int
    phase_index: int | None = None
    basis: TurnBasis
    label_source: LabelSource | None = None
    confidence: float | None = None
    confidence_pm: float | None = None  # 95%-CI half-width; 0 non-cohort
    agreement: float | None = None  # vote agreement; None solo/unvoted


@scanner(
    messages="all",
    events=["model", "span_begin", "span_end", "tool", "branch"],
    timeline=True,
)
def decision_phases(
    spec: Spec,
    judge_models: str | Model | Sequence[str | Model],
    k_rolls: int = 1,
    chunk: int = 40,
    snippet_chars: int = SNIPPET_CHARS,
    final_text_chars: int = FINAL_TEXT_CHARS,
    tool_call_chars: int = TOOL_CALL_CHARS,
    tool_result_chars: int = TOOL_RESULT_CHARS,
    user_chars: int = USER_CHARS,
    tool_only_turns: bool = True,
    cache: bool | CachePolicy = True,
    verify: bool | None = None,
    verify_chunk: int = 8,
    verify_sample: float | None = None,
    verifier_model: str | Model | None = None,
    narrate: bool = True,
) -> Scanner[Transcript]:
    """Segment a run into decision phases over the orchestrator's turns.

    The judge, the verifier and the narrator read the same digest
    lines (`turn_digests`), so the digest settings (``snippet_chars``,
    ``final_text_chars``, ``tool_call_chars``, ``tool_result_chars``,
    ``user_chars``, ``tool_only_turns``) apply to all three and change
    the judge requests, and with them the cache keys. A part over its
    cap keeps its start and its end (`phases_common.clip`).

    Three regimes (mutually exclusive):

    - solo: one model, ``k_rolls=1`` - single-judge path.
    - k-roll: one model, ``k_rolls>1`` - roll-vote drives the
      partition; the verifier composes on top.
    - cohort: per-turn majority vote drives the partition;
      the verifier is off.

    Args:
        spec: Supplies the required phase definitions
            (``spec.phases``) and the optional user context block
            (``spec.context``).
        judge_models: A name/``Model``, or a list of distinct models
            for the cohort regime.
        k_rolls: Rolls of one model; mutually exclusive with a
            multi-model list.
        chunk: Digests per segmentation call.
        snippet_chars: Per-digest cap on turn text, reasoning, and
            each delegation shown to the judge.
        final_text_chars: Cap on the text of the orchestrator's last
            turn with text, which is usually its report of the work;
            never below ``snippet_chars``, so 0 caps it like any
            other turn.
        tool_call_chars: Per-call cap on the JSON arguments shown with
            each tool call; 0 shows tool names only.
        tool_result_chars: Per-call cap on each tool call's recorded
            output or error message; 0 leaves outputs out.
        user_chars: Per-message cap on the user messages a turn
            received; 0 leaves them out.
        tool_only_turns: Judge turns whose only content is tool calls.
            ``False`` leaves them out of the digests, so they take
            their label by attribution.
        cache: Judge-call caching. The default ``True`` is inspect's
            standard on-disk response cache; ``False`` always hits
            the API.
        verify: Run the second-round verifier over the phases
            (selection, in precedence order: ``min_confidence`` < 0.6,
            ``min_agreement`` < 0.6 (k-roll vote instability), short
            wedge phases, plus a deterministic spot-check). Default
            ``None`` = auto:
            on for a single model (solo and k-roll), off for a
            multi-model cohort.
        verify_chunk: Phases per verifier call (1 = one call per phase).
        verify_sample: The share of phases the verifier spot-checks
            at random on top of the doubt triggers. None = the
            default max(5%, 3); 0.0 disables; 1.0 reviews every
            phase. Capped at the phase count.
        verifier_model: Judge for the verifier pass; ``None`` (default)
            uses the (first) judge model.
        narrate: Narrate the (verified) phases - headline + summary +
            turn groups per phase.

    Returns:
        A scanner whose Result value carries:

        - ``phase_names``: the spec's phase names plus the
          reserved "ops" bucket when none is declared.
        - ``phases``: the phase partition, one entry per contiguous
          same-label run:

          - ``phase``: the label.
          - ``turn_start`` / ``turn_end``: inclusive digest-turn range.
          - ``n_turns``: digest turns in the phase (the turns the
            judge saw).
          - ``confidence``: mean of the phase's per-turn confidences
            (filled turns included, at 0.3).
          - ``min_confidence``: minimum - the verifier's selection signal.
          - ``min_agreement``: minimum per-turn vote agreement over
            turns carrying one (voting regimes); None solo.
          - ``confidence_spread``: 95%-CI half-width of the same
            per-turn confidences; 0.0 after an overturn.
          - ``judge_agreement``: mean per-turn vote agreement
            (voting regimes); None solo.
          - ``confidence_source``: whose confidence the phase
            carries - single_judge / majority_vote / verifier.
          - ``explanation``: the first contributing segment's explanation.
          - ``verifier``: a representative second-round review (the
            ``VerifierReview`` record: trigger, original label /
            confidence / explanation, verifier label / confidence /
            explanation, verifier_model, overturned, status); None
            when no completed verdict overlaps the range. A merged display
            phase can contain several original review units.
          - ``verifier_reviews``: every original selected phase review
            overlapping this display phase, each with original_phase_index,
            turn_start, turn_end, and a nested review. Missing verdicts carry
            status no_answer or refusal. An empty list means none selected.
          - ``headline`` / ``summary``: complete narrator text, without
            character clipping; blank headlines use a template, missing
            narratives use a template headline and empty summary.
          - ``narration_group_status``: complete, invalid_partition,
            empty_groups, no_narrative, or not_run. Complete describes
            partition coordinates, not factual correctness.
          - ``turn_groups``: gapless partition of the phase's turn
            range - ``{turn_start, turn_end, title, gist}``.
          - ``anchor_event_id``: the first member model event's uuid
            - the Scout-viewer deep-link anchor (None when the
            source carries no event uuids).
        - ``turns``: the dense turn-granular surface, one row per
          orchestrator turn:

          - ``basis``: how the turn got its label:

            - "judged": digest turn covered by a judge segment (in
              the voting regimes: by at least one voting member).
            - "filled": digest turn no judge covered; inherits the
              previous (consensus) label at low confidence.
            - "attributed": no digest - a failed turn, a turn with
              no content, or (``tool_only_turns=False``) a
              tool-call-only turn; the judge never saw it, so by
              projection it takes the phase whose turn range
              contains it (or the nearest preceding phase, for
              turns in a gap).
            - "unattributed": no digest and no phase to take - an
              unjudged digest turn separates it from the previous
              phase, or the first digest turn was unjudged
              (`project_phase_turns`).
            - "refusal": its chunk's judge calls were refused on
              both the cached and the uncached attempt.
            - "no_answer": its chunk's judge calls never produced a
              schema-valid answer.
            - "missing_turn": chunk judged fine, but this turn was
              left uncovered before any label existed to inherit.

          - ``label_source``: which regime decided the label -
            "single_judge" / "majority_vote" / "verifier".
          - ``confidence``: the dominant-source confidence.
          - ``confidence_pm``: its 95%-CI half-width; 0 outside the
            cohort regime.
          - ``agreement``: the turn's cohort vote agreement.
        - ``verifier``: the second-round audit, counts only:

          - ``ran``: whether the verifier pass ran at all.
          - ``n_low_confidence``: selected because ``min_confidence``
            < 0.6 (includes any phase containing a gap-filled turn).
          - ``n_wedge``: selected as a <=2-turn phase wedged between
            two phases sharing one other label.
          - ``n_random_sample``: the deterministic random spot-check
            draw - max(5%, 3) of phases, stratified by label.
          - ``n_no_verdict``: selected phases the verifier never
            answered on (omitted id, refusal, no answer).
          - ``n_relabelled``: applied overturns (differing label at
            verifier confidence >= 0.6).
          - ``n_weak_relabel``: the verifier disagreed below 0.6 -
            recorded here, not applied.
          - ``n_low_agreement``: phases selected on vote instability.
          - ``n_random_sample_relabelled``: applied overturns among
            the random sample.
          - ``verifier_model``: the verifier's resolved model name.
        - ``narrator``: counts-only audit `{ran, n_fallback, narrator_model}``
          (``ran: false`` with a zero count when the narrator did not run).
        - ``phase_vocab``: the resolved rubric, one ``{label,
          description, ops, reserved}`` per judged phase (``reserved``:
          appended by transect, not declared in the spec).
        - ``judge_models``: the judge model names as given, deduplicated.
        - ``judge``: the `judge_setup` identity block (regime, roster,
          roll count, verifier arming) the frames project as the
          standard judge-identity columns.
        - ``cohort``: one stable shape (``ran: false`` with empty
          defaults outside the voting regimes):

          - ``members``: the complete raw record per member -
            ``{model, roll, turns: [{turn, phase, confidence,
            explanation, basis}]}``.
          - ``vote``: ``[{turn, phase, agreement, n_voting}]`` per
            voted turn.
          - ``agreement``: ``{n_members, n_models, k_rolls}`` - the
            cohort shape (per-turn agreement lives on the vote
            records; run-level means are derivable, not duplicated).

    Raises:
        ValueError: When ``spec.phases`` is empty (there is no
            open-vocabulary mode); when the model list and ``k_rolls``
            are combined or malformed; when ``verify=True`` meets a
            multi-model cohort.
    """
    if not spec.phases:
        raise ValueError(
            "decision_phases requires spec.phases - supply the phase "
            "definitions (there is no open-vocabulary mode)."
        )
    if chunk <= 0:
        raise ValueError("decision_phases requires chunk >= 1")
    if verify_chunk <= 0:
        raise ValueError("decision_phases requires verify_chunk >= 1")
    check_digest_caps(
        snippet_chars=snippet_chars,
        final_text_chars=final_text_chars,
        tool_call_chars=tool_call_chars,
        tool_result_chars=tool_result_chars,
        user_chars=user_chars,
    )
    members_spec = cohort_members(judge_models, k_rolls)
    factory_names = list(dict.fromkeys(m.name for m in members_spec))
    verify_on = resolve_verify(len(factory_names), verify, verify_sample)
    judge = judge_setup(
        factory_names,
        k_rolls,
        verifier_armed=verify_on,
        verifier_model=(
            (str(verifier_model) if verifier_model is not None else factory_names[0])
            if verify_on
            else None
        ),
    )
    if verify is None and len(factory_names) > 1:
        logger.info(
            "decision_phases: cohort regime - verifier auto-disabled "
            "(majority vote is the correction mechanism)"
        )
    phase_defs, ops_name = resolve_phases(spec)
    phase_names = [p.label for p in phase_defs]
    declared = {p.label for p in spec.phases}
    phase_vocab = [
        {
            "label": p.label,
            "description": p.description,
            "ops": p.label == ops_name,
            "reserved": p.label not in declared,
        }
        for p in phase_defs
    ]
    answer = _answer_spec(phase_names)

    async def execute(transcript: Transcript) -> Result:
        judges = [(member, member.resolve()) for member in members_spec]
        member_keys = [member.key for member in members_spec]
        n_members = len(members_spec)
        digests = turn_digests(
            transcript,
            snippet_chars=snippet_chars,
            final_text_chars=final_text_chars,
            tool_call_chars=tool_call_chars,
            tool_result_chars=tool_result_chars,
            user_chars=user_chars,
            tool_only_turns=tool_only_turns,
        )
        n_turns = sum(1 for _ in orchestrator_turns(transcript))
        task_prompt = agent_task_prompt(transcript)
        system = system_prompt(spec, task_prompt=task_prompt)
        digest_judgements: list[ConsensusJudgement] = []
        # (model name, roll) -> that member's per-turn rows, whole run
        member_rows: dict[tuple[str, int], list[DigestJudgement]] = {
            key: [] for key in member_keys
        }
        votes: list[TurnVote] = []
        last_phase: str | None = None
        for start in range(0, len(digests), chunk):
            chunk_digests = digests[start : start + chunk]
            user = chunk_user_prompt(chunk_digests, last_phase)
            answers = await gather_judge_calls(
                call_judge(judge, answer, system, user, roll_cache(cache, member.roll))
                for member, judge in judges
            )
            chunk_rows: dict[tuple[str, int], list[DigestJudgement]] = {}
            for (member, _), (value, status) in zip(judges, answers, strict=True):
                key = member.key
                if status != "ok":
                    chunk_rows[key] = [
                        DigestJudgement(turn=d.turn, basis=status)
                        for d in chunk_digests
                    ]
                    continue
                chunk_segments = _validate_segments(
                    cast(list[JsonValue], (value or {}).get("segments") or []),
                    first=chunk_digests[0].turn,
                    last=chunk_digests[-1].turn,
                )
                chunk_rows[key] = _project_chunk(
                    chunk_digests, chunk_segments, last_phase
                )
            consensus, chunk_votes = vote_turns(
                chunk_rows, order=member_keys, carry=last_phase
            )
            digest_judgements.extend(consensus)
            votes.extend(chunk_votes)
            for key, rows in chunk_rows.items():
                member_rows[key].extend(rows)
            tail = consensus[-1]
            if tail.basis in ("refusal", "no_answer"):
                # reset so the next chunk gets no false-adjacency hint
                last_phase = None
            else:
                last_phase = tail.phase or last_phase
        phases = stitch_phases(digest_judgements)
        audit = VerifierAudit(ran=False)
        if verify_on and phases:
            verifier = judges[0][1]
            if verifier_model is not None:
                verifier = (
                    verifier_model
                    if isinstance(verifier_model, Model)
                    else get_model(verifier_model)
                )
            phases, audit = await verify_phases(
                verifier,
                spec,
                task_prompt=task_prompt,
                digests=digests,
                digest_judgements=digest_judgements,
                phases=phases,
                phase_names=phase_names,
                chunk=verify_chunk,
                cache=cache,
                sample=verify_sample,
            )
            audit.verifier_model = str(verifier)
        if n_members > 1:
            for p in phases:
                if p.confidence_source != "verifier":
                    p.confidence_source = "majority_vote"
        anchor_of = {d.turn: d.event_id for d in digests}
        for p in phases:
            p.anchor_event_id = next(
                (
                    anchor_of[t]
                    for t in range(p.turn_start, p.turn_end + 1)
                    if anchor_of.get(t) is not None
                ),
                None,
            )
        narrator = NarratorAudit(ran=False)
        if narrate and phases:
            narrator = await narrate_phases(
                judges[0][1],
                spec,
                task_prompt=task_prompt,
                digests=digests,
                phases=phases,
                cache=cache,
            )
            narrator.narrator_model = factory_names[0]
        cohort = Cohort()
        if n_members > 1:
            cohort = Cohort(
                ran=True,
                members=[
                    CohortMember(model=name, roll=r, turns=member_rows[(name, r)])
                    for (name, r) in member_keys
                ],
                vote=votes,
                agreement=CohortAgreement(
                    n_members=n_members,
                    n_models=len(factory_names),
                    k_rolls=k_rolls,
                ),
            )
        explanation = (
            f"{len(phases)} phases over {len(digest_judgements)} "
            f"digest turns ({n_turns} orchestrator turns)"
        )
        if n_members > 1:
            explanation += f" · cohort of {n_members} members"
        return Result(
            value=cast(
                JsonValue,
                {
                    "phase_names": list(phase_names),
                    "phase_vocab": cast(JsonValue, phase_vocab),
                    "phases": [p.model_dump() for p in phases],
                    "turns": [
                        t.model_dump()
                        for t in _dense_turns(
                            digest_judgements, phases, n_turns, n_members
                        )
                    ],
                    "verifier": audit.model_dump(),
                    "narrator": narrator.model_dump(),
                    "judge_models": factory_names,
                    "judge": cast(JsonValue, judge),
                    "cohort": cohort.model_dump(),
                },
            ),
            explanation=explanation,
        )

    return execute


def turn_digests(
    transcript: Any,
    snippet_chars: int = SNIPPET_CHARS,
    final_text_chars: int = FINAL_TEXT_CHARS,
    tool_call_chars: int = TOOL_CALL_CHARS,
    tool_result_chars: int = TOOL_RESULT_CHARS,
    user_chars: int = USER_CHARS,
    tool_only_turns: bool = True,
) -> list[Digest]:
    """Build one digest per orchestrator turn that has content to judge.

    Digests are a sparse selection over the orchestrator's turns
    (`helpers.orchestrator_turns`, the turn axis): a turn is eligible
    when it carries visible text, reasoning-block content (when the
    source records it), a delegation, a user message, or (with
    ``tool_only_turns``) a tool call. Provider-failure placeholder
    turns never are. Sub-agent activity lives in child spans and is
    represented only by delegation lines, folded in at the last
    eligible turn preceding each span_begin.

    A user message belongs to the first orchestrator turn whose input
    carries it. The first turn's input is the task, which the judge
    prompts already carry, so it is not repeated, nor is a later copy
    of it; compaction summaries and OpenClaw scaffold polls are not
    user messages. A tool call's output is its tool event's result or
    error, else the tool message that answered the call; a call whose
    arguments already appear as a delegation line shows no arguments.

    A part over its cap keeps its start and its end (`clip`). The last
    digest turn with text keeps up to ``final_text_chars`` of it.

    Args:
        transcript: The transcript to digest (Scout ``Transcript`` or
            any object with compatible ``events``/``messages``).
        snippet_chars: Per-digest cap on turn text, on reasoning, and
            on each delegation string.
        final_text_chars: Cap on the text of the last digest turn with
            text; never below ``snippet_chars``.
        tool_call_chars: Per-call cap on the JSON arguments; 0 shows
            tool names only.
        tool_result_chars: Per-call cap on the recorded output or
            error message; 0 leaves outputs out.
        user_chars: Per-message cap on user messages; 0 leaves them out.
        tool_only_turns: Keep turns whose only content is tool calls.

    Returns:
        ``Digest`` records in turn order.
    """
    check_digest_caps(
        snippet_chars=snippet_chars,
        final_text_chars=final_text_chars,
        tool_call_chars=tool_call_chars,
        tool_result_chars=tool_result_chars,
        user_chars=user_chars,
    )
    lanes = Lanes(transcript)
    outputs = _tool_outputs(transcript) if tool_result_chars else {}
    user_by_turn = _user_messages_by_turn(lanes) if user_chars else {}
    by_turn: dict[int, Digest] = {}

    def _digest(turn: int) -> Digest:
        if turn not in by_turn:
            by_turn[turn] = Digest(turn=turn)
        return by_turn[turn]

    eligible_turns: list[int] = []  # orchestrator, non-failed: anchor targets
    final: tuple[int, str] | None = None  # last digest turn with text, uncapped
    for turn, ev, calls in lanes.turns():
        event: Any = ev
        if _is_failed_turn(event):
            continue  # provider-failure placeholder: not a turn to judge
        eligible_turns.append(turn)
        message = event.output.message
        text = (message.text or "").strip() if message else ""
        reasoning = message_reasoning(message)
        user = [clip(m, user_chars) for m in user_by_turn.get(turn, [])]
        delegations: list[str] = []
        delegating: set[str] = set()  # call ids already shown as delegations
        if not lanes.begins:  # span-less sources: delegations ride tool-call args
            for call in calls:
                arguments = call.arguments if isinstance(call.arguments, dict) else {}
                task = arguments.get("task") or arguments.get("prompt")
                if task:
                    label = arguments.get("label") or arguments.get("description")
                    task_text = strip_subagent_scaffold(str(task))
                    goal = (f"[{label}] " if label else "") + task_text
                    delegations.append(clip(goal, snippet_chars))
                    delegating.add(call.id)
        has_content = text or reasoning or delegations or user
        if not has_content and not (calls and tool_only_turns):
            continue  # nothing classifiable in this turn
        digest = _digest(turn)
        digest.user = user
        digest.text = clip(text, snippet_chars)
        digest.reasoning = clip(reasoning, snippet_chars)
        if text:
            final = (turn, text)
        digest.calls = [
            _tool_digest(
                call,
                outputs,
                tool_call_chars=0 if call.id in delegating else tool_call_chars,
                tool_result_chars=tool_result_chars,
            )
            for call in calls
        ]
        digest.delegations.extend(delegations)
        digest.event_id = getattr(event, "uuid", None)

    # Sub-agent-span delegations, snapped to the nearest digest-eligible
    # orchestrator turn at or before the span_begin.
    for span, raw_anchor in _span_anchors(lanes):
        if not eligible_turns:
            continue
        position = bisect_right(eligible_turns, raw_anchor) - 1
        anchor = eligible_turns[position] if position >= 0 else eligible_turns[0]
        text = span_task_text(span, lanes.first_models.get(span.id))[0]
        goal = text.splitlines()[0].strip() if text else str(span.name)
        _digest(anchor).delegations.append(clip(goal, snippet_chars))

    if final is not None:
        turn, text = final
        by_turn[turn].text = clip(text, max(final_text_chars, snippet_chars))
    return [by_turn[turn] for turn in sorted(by_turn)]


def agent_task_prompt(transcript: Any) -> str:
    """Extract the task the agent was given.

    Source-generic: the first non-scaffold user message - the sample
    input on .eval sources, the initial brief on OpenClaw imports.

    Args:
        transcript: The transcript whose messages are scanned.

    Returns:
        The whitespace-flattened task prompt, or "".
    """
    for message in transcript.messages or []:
        if getattr(message, "role", None) != "user":
            continue
        text = (getattr(message, "text", None) or "").strip()
        if not text:
            continue
        if text.lower() in _OPENCLAW_SCAFFOLD_USER_MESSAGES:
            continue  # known OpenClaw scaffold line (e.g. heartbeat poll)
        return " ".join(text.split())[:TASK_PROMPT_CHARS]
    return ""


def system_prompt(spec: Spec, task_prompt: str = "") -> str:
    """Render the segmentation system prompt.

    Phase definitions (names + descriptions) + rules + the two context
    blocks: task context (the prompt the agent was given) and
    additional context (``spec.context``, user-supplied prose).

    Args:
        spec: Supplies the phase definitions and the user context.
        task_prompt: The agent's own task prompt ("" omits the block).

    Returns:
        The complete system prompt string.
    """
    phase_defs, ops_name = resolve_phases(spec)
    labels_line = ", ".join(p.label for p in phase_defs)
    prompt = _SYSTEM_HEAD + vocab_lines(phase_defs) + _RULES.format(ops=ops_name)
    prompt += context_blocks(spec, task_prompt)
    prompt += (
        "\nReport the phases by calling the answer() tool: 'segments' is "
        "the ordered list of phases, each with turn_start and turn_end "
        "(inclusive turn indices as shown), phase (EXACTLY one of: "
        f"{labels_line}), confidence (0.0-1.0), and an explanation of <=14 "
        "words. The first phase starts at the first index shown, the "
        "last ends at the last index shown, each phase starting one "
        "after the previous ends."
    )
    return prompt


def chunk_user_prompt(chunk: Sequence[Digest], last_phase: str | None) -> str:
    """Render one chunk's user prompt.

    Args:
        chunk: The chunk's digests, in turn order.
        last_phase: The phase left open by the previous chunk (None on
            the first chunk or after a refused region - no hint line).

    Returns:
        Optional last-phase hint + digest lines prefixed with their
        turn indices.
    """
    first, last = chunk[0].turn, chunk[-1].turn
    hint = (
        f"Context: the phase ending just before turn {first} was "
        f"'{last_phase}'. If turn {first} continues that same activity, "
        "begin with that label.\n"
        if last_phase is not None
        else ""
    )
    lines = "\n".join(digest_line(d) for d in chunk)
    return f"{hint}Segment turns {first}..{last} into phases:\n{lines}"


def project_phase_turns(
    phase_starts: Sequence[int],
    n_turns: int,
    unjudged_turns: Collection[int] = (),
) -> list[int | None]:
    """Assign orchestrator turns (tool-call-only included) to phases.

    A phase start sets the current phase and an unjudged digest turn
    clears it, so a turn inside a phase belongs to it, a turn between
    phases inherits the previous one unless an unjudged digest turn
    separates them, and turns before the first digest turn take that
    turn's outcome.

    Args:
        phase_starts: Each phase's ``turn_start``, in phase order.
        n_turns: Number of orchestrator turns in the transcript.
        unjudged_turns: Digest turns `is_labelled` rejects - the rows
            `stitch_phases` closed a phase on, so never inside one.

    Returns:
        ``phase_index_of_turn`` of length ``n_turns`` (indices into the
        phases list, None where no phase claims the turn). All-None
        when there are no phases at all.
    """
    assignment: list[int | None] = [None] * n_turns
    if not phase_starts:
        return assignment
    events: list[tuple[int, int | None]] = sorted(
        [(start, index) for index, start in enumerate(phase_starts)]
        + [(turn, None) for turn in set(unjudged_turns)],
        key=lambda event: event[0],
    )
    current = events[0][1]
    position = 0
    for turn in range(n_turns):
        while position < len(events) and events[position][0] <= turn:
            current = events[position][1]
            position += 1
        assignment[turn] = current
    return assignment


def _answer_spec(phase_names: Sequence[str]) -> AnswerStructured:
    """Build the per-spec answer schema.

    Args:
        phase_names: The judged phase names (resolved, ops included).

    Returns:
        The ``AnswerStructured`` spec for ``generate_answer``.
    """
    vocab_segment = create_model(
        "_VocabSegment",
        __base__=_Segment,
        phase=(
            Literal[tuple(phase_names)],
            Field(description=("The phase label (one of the allowed phase names).")),
        ),
    )
    phase_chunks = create_model(
        "_PhaseChunks",
        segments=(
            list[vocab_segment],  # type: ignore[valid-type]
            Field(
                description=(  # type: ignore[valid-type]
                    "The contiguous phases covering every turn index shown, in order."
                )
            ),
        ),
        # declared so Scout does not auto-add its stock explanation field
        explanation=(
            str,
            Field(
                description=("One or two sentences on how you segmented these turns.")
            ),
        ),
    )
    return AnswerStructured(type=phase_chunks)


def _validate_segments(
    items: Sequence[JsonValue], first: int, last: int
) -> list[_Segment]:
    """Apply the semantic checks the answer schema cannot express.

    (Phase-name membership + confidence bounds are in the schema -
    see ``_answer_spec`` - where the retry loop enforces them.)

    1. range sanity - inverted ranges (turn_end < turn_start) and
       segments entirely outside this chunk drop.
    2. chunk-bounds clamping - the judge sometimes claims turns it
       was not shown.

    Args:
        items: ``segments`` entries from the validated answer() call.
        first: First turn index shown in this chunk's digest lines.
        last: Last turn index shown in this chunk's digest lines.

    Returns:
        Clamped ``_Segment`` models, in answer order.
    """
    segments: list[_Segment] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = cast(dict[str, Any], raw)
        s_idx, e_idx = int(item["turn_start"]), int(item["turn_end"])
        if e_idx < s_idx or e_idx < first or s_idx > last:
            continue  # inverted, or entirely outside this chunk
        segments.append(
            _Segment(
                turn_start=max(s_idx, first),
                turn_end=min(e_idx, last),
                phase=str(item["phase"]),
                confidence=float(item["confidence"]),
                explanation=str(item.get("explanation") or "").strip()[:120],
            )
        )
    return segments


def _project_chunk(
    chunk: Sequence[Digest],
    segments: list[_Segment],
    last_phase: str | None,
) -> list[DigestJudgement]:
    """Unpack the judge's range-form segments into per-turn rows.

    Args:
        chunk: The chunk's digests, in turn order.
        segments: The chunk's phase segments from the judge.
        last_phase: The phase left open by the previous chunk (fill
            seed for uncovered turns at the start of the chunk).

    Returns:
        One ``DigestJudgement`` per digest turn: basis "judged" or "filled"
        or basis "missing_turn" before any label exists.
    """
    rows: list[DigestJudgement] = []
    last_label = last_phase
    for d in chunk:
        segment = next(
            (seg for seg in segments if seg.turn_start <= d.turn <= seg.turn_end),
            None,
        )
        if segment is not None:
            last_label = segment.phase
            rows.append(
                DigestJudgement(
                    turn=d.turn,
                    phase=segment.phase,
                    confidence=segment.confidence,
                    explanation=segment.explanation,
                    basis="judged",
                )
            )
        elif last_label is not None:
            rows.append(
                DigestJudgement(
                    turn=d.turn,
                    phase=last_label,
                    confidence=0.3,
                    basis="filled",
                )
            )
        else:  # nothing labelled yet in the whole run
            rows.append(DigestJudgement(turn=d.turn, basis="missing_turn"))
    return rows


def _dense_turns(
    digest_judgements: list[ConsensusJudgement],
    phases: list[StitchedPhase],
    n_turns: int,
    n_members: int,
) -> list[PhaseTurn]:
    """Build the scanner result's turn-granular surface.

    Args:
        digest_judgements: Per-digest-turn (consensus) judgements.
        phases: The stitched phases (the projection targets).
        n_turns: Number of orchestrator turns in the transcript.
        n_members: Judge member count (decides the source labels).

    Returns:
        One ``PhaseTurn`` per turn; ``phase_index`` is
        `project_phase_turns`'s assignment.
    """
    assignment = project_phase_turns(
        [p.turn_start for p in phases],
        n_turns,
        [row.turn for row in digest_judgements if not is_labelled(row)],
    )
    row_of = {row.turn: row for row in digest_judgements}
    regime_label: LabelSource = "majority_vote" if n_members > 1 else "single_judge"
    out: list[PhaseTurn] = []
    for turn in range(n_turns):
        row = row_of.get(turn)
        verifier = row is not None and row.source == "verifier"
        confidence = row.confidence if row is not None else None
        out.append(
            PhaseTurn(
                turn=turn,
                phase_index=assignment[turn],
                basis=(
                    row.basis
                    if row is not None
                    else "attributed"
                    if assignment[turn] is not None
                    else "unattributed"
                ),
                label_source=(
                    "verifier"
                    if verifier
                    else regime_label
                    if assignment[turn] is not None
                    else None
                ),
                confidence=confidence,
                confidence_pm=(
                    (row.confidence_pm or 0.0)
                    if row is not None and confidence is not None
                    else None
                ),
                agreement=(
                    row.agreement if row is not None and n_members > 1 else None
                ),
            )
        )
    return out


def _span_anchors(lanes: Lanes) -> list[tuple[Any, int]]:
    """Each sub-agent span_begin with its raw anchor: the last
    orchestrator turn preceding it in event order (0 when none does, as
    `token_timeline`'s ``spawn_turn``)."""
    return [
        (event, max(before - 1, 0))
        for before, event in lanes.events_before_turn()
        if event.event == "span_begin" and event.id in lanes.sub_ids
    ]


def _is_failed_turn(event: Any) -> bool:
    """Detect an OpenClaw provider-failure placeholder turn: the scaffold
    writes a fake assistant turn ("[assistant turn failed ...]") when a
    provider call dies."""
    message = event.output.message
    text = (message.text or "").strip().lower() if message else ""
    return text.startswith(_OPENCLAW_FAILED_TURN_PREFIX)


def _tool_digest(
    call: ToolCall,
    outputs: dict[str, tuple[str, bool]],
    tool_call_chars: int,
    tool_result_chars: int,
) -> ToolDigest:
    """One tool call as a digest shows it, within the per-call caps."""
    arguments = ""
    if tool_call_chars and call.arguments:
        arguments = json.dumps(call.arguments, ensure_ascii=False, default=str)
    result, error = ("", False)
    if tool_result_chars:
        result, error = outputs.get(call.id, ("", False))
    return ToolDigest(
        function=call.function,
        arguments=clip(arguments, tool_call_chars),
        result=clip(result, tool_result_chars),
        error=error,
    )


def _tool_outputs(transcript: Any) -> dict[str, tuple[str, bool]]:
    """Each tool call's recorded output, keyed by call id: ``(text,
    is_error)``, whitespace-flattened. The tool event wins over the tool
    message that answered the call; the message covers sources without
    tool events."""
    outputs: dict[str, tuple[str, bool]] = {}
    for message in transcript.messages or []:
        call_id = getattr(message, "tool_call_id", None)
        if getattr(message, "role", None) != "tool" or not call_id:
            continue
        error = getattr(message, "error", None)
        text = error.message if error is not None else (message.text or "")
        outputs[call_id] = (" ".join(text.split()), error is not None)
    for event in transcript.events:
        if getattr(event, "event", None) != "tool" or not event.id:
            continue
        error = event.error
        text = error.message if error is not None else _result_text(event.result)
        outputs[event.id] = (" ".join(text.split()), error is not None)
    return outputs


def _result_text(result: Any) -> str:
    """A tool event's result as text: strings as recorded, the text of
    content blocks, other scalars by ``str``; non-text blocks are left out."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return " ".join(
            text for item in result if (text := getattr(item, "text", None))
        )
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return "" if result is None else str(result)


def _user_messages_by_turn(lanes: Lanes) -> dict[int, list[str]]:
    """User-message texts keyed by the orchestrator turn whose input
    carried them first, whitespace-flattened. The first non-failed turn's
    input is the task and yields nothing, and a later message repeating
    one of its texts (a compaction re-inserting the task) is skipped. A
    failed turn records nothing, so its new messages go to the next
    turn. A merged message (``combined_from``) counts only when one of
    its parts is new."""
    seen: set[str] = set()
    task_texts: set[str] = set()
    by_turn: dict[int, list[str]] = {}
    task_read = False
    for turn, event, _calls in lanes.turns():
        if _is_failed_turn(event):
            continue
        for message in event.input:
            key = message.id or f"{message.role}:{message.text}"
            parts = (message.metadata or {}).get("combined_from") or []
            new = key not in seen and (not parts or any(p not in seen for p in parts))
            seen.update([key, *parts])
            if not new or message.role != "user":
                continue
            text = " ".join((message.text or "").split())
            if not task_read:
                task_texts.add(text)
                continue
            if (message.metadata or {}).get("summary"):
                continue  # a compaction summary, not a message from the user
            if text in task_texts or text.lower() in _OPENCLAW_SCAFFOLD_USER_MESSAGES:
                continue
            if text:
                by_turn.setdefault(turn, []).append(text)
        task_read = True
    return by_turn
