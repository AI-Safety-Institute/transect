"""decision_phases: phase segmentation of a run."""

import logging
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any, Literal, cast

from inspect_ai.event import ModelEvent, TimelineEvent
from inspect_ai.model import CachePolicy, Model, get_model
from inspect_scout import AnswerStructured, Result, Scanner, Transcript, scanner
from pydantic import BaseModel, Field, JsonValue, create_model

from transect.scanners.cohort import (
    LabelSource,
    cohort_members,
    judge_setup,
    roll_cache,
)
from transect.scanners.helpers import (
    events_between_turns,
    main_span,
    model_turns,
    span_task_text,
    strip_subagent_scaffold,
    subagent_span_begins,
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
    SNIPPET_CHARS,
    Digest,
    DigestJudgement,
    StitchedPhase,
    TurnBasis,
    call_judge,
    context_blocks,
    digest_line,
    gather_judge_calls,
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

# Appended to the judged phases when the spec declares no operational bucket
_SYSTEM_HEAD = (
    "You are segmenting an autonomous agent's turns (its own text AND the "
    "[DELEGATES] tasks it hands to sub-agents) into CONTIGUOUS PHASES of "
    "the task it is working on. Each phase is a run of consecutive turns "
    "doing ONE activity. Label each phase with the single best-fitting "
    "phase name:\n"
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
    "5. confidence is your 0.0-1.0 certainty in BOTH the boundary and the "
    "label of the phase.\n"
)


class _Segment(BaseModel):
    """One judged phase.

    JUDGE-FACING: the Field descriptions render in the answer() tool."""

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
    cache: bool | CachePolicy = True,
    verify: bool | None = None,
    verify_chunk: int = 8,
    verify_sample: float | None = None,
    verifier_model: str | Model | None = None,
    narrate: bool = True,
) -> Scanner[Transcript]:
    """Segment a run into decision phases over reasoning-bearing turns.

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
        snippet_chars: Per-digest text cap shown to the judge.
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
          - ``n_turns``: digest turns in the phase (the reasoning-
            bearing turns the judge saw).
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
          model turn:

          - ``basis``: how the turn got its label:

            - "judged": digest turn covered by a judge segment (in
              the voting regimes: by at least one voting member).
            - "filled": digest turn no judge covered; inherits the
              previous (consensus) label at low confidence.
            - "attributed": no digest - tool-call-only / failed /
              sub-agent turns; the judge never saw it, so by
              projection it takes the phase whose turn range
              contains it (or the nearest preceding phase, for
              turns in a gap).
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
    members_spec = cohort_members(judge_models, k_rolls)
    n_models = len({member.name for member in members_spec})
    if verify is True and n_models > 1:
        raise ValueError(
            "verify=True is incompatible with a multi-model cohort - "
            "the majority vote is the correction mechanism there "
            "(verifier XOR cohort)"
        )
    verify_on = n_models == 1 if verify is None else verify
    if verify_sample is not None and not 0.0 <= verify_sample <= 1.0:
        raise ValueError(f"verify_sample must be in [0, 1], got {verify_sample}")
    factory_names = list(dict.fromkeys(m.name for m in members_spec))
    judge = judge_setup(
        factory_names,
        k_rolls,
        verifier_armed=verify_on,
        verifier_model=(
            (str(verifier_model) if verifier_model is not None else factory_names[0])
            if verify_on and factory_names
            else None
        ),
    )
    if verify is None and n_models > 1:
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
        names = list(dict.fromkeys(m.name for m in members_spec))
        member_keys = [member.key for member in members_spec]
        n_members = len(members_spec)
        digests = turn_digests(transcript, snippet_chars=snippet_chars)
        n_turns = sum(1 for _ in model_turns(transcript))
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
            # the verifier: verifier_model when given, else the
            # (first) judge
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
            narrator.narrator_model = names[0]
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
                    n_models=n_models,
                    k_rolls=k_rolls,
                ),
            )
        explanation = (
            f"{len(phases)} phases over {len(digest_judgements)} "
            f"digest turns ({n_turns} model turns)"
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
                    "judge_models": names,
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
) -> list[Digest]:
    """Build one digest per reasoning-bearing main-lane turn.

    Digests are a sparse selection over model turns. The main lane
    comes from the transcript's timeline. Sub-agent activity lives
    in child spans and is represented only by delegation lines,
    folded in at the last eligible turn preceding each span_begin.

    Args:
        transcript: The transcript to digest (Scout ``Transcript`` or
            any object with compatible ``events``/``messages``).
        snippet_chars: Per-digest cap on turn text and on each
            delegation string.

    Returns:
        ``Digest`` records in turn order.
    """
    main = main_span(transcript)
    main_models = {
        id(item.event)
        for item in main.content
        if isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
    }
    subagent_spans, first_models = subagent_span_begins(transcript, main)

    by_turn: dict[int, Digest] = {}

    def _digest(turn: int) -> Digest:
        if turn not in by_turn:
            by_turn[turn] = Digest(turn=turn)
        return by_turn[turn]

    eligible_turns: list[int] = []  # main-lane, non-failed (anchor targets)
    for turn, (ev, calls) in enumerate(model_turns(transcript)):
        event: Any = ev
        if id(event) not in main_models:
            continue  # sub-agent/init/scorer turn: not the main lane
        if _is_failed_turn(event):
            continue  # provider-failure placeholder: not reasoning
        eligible_turns.append(turn)
        message = event.output.message
        text = (message.text or "").strip() if message else ""
        delegations: list[str] = []
        if not subagent_spans:  # span-less sources: delegations ride tool-call args
            for call in calls:
                arguments = call.arguments if isinstance(call.arguments, dict) else {}
                task = arguments.get("task") or arguments.get("prompt")
                if task:
                    label = arguments.get("label") or arguments.get("description")
                    task_text = strip_subagent_scaffold(str(task))
                    goal = (f"[{label}] " if label else "") + task_text
                    delegations.append(goal[:snippet_chars])
        if not text and not delegations:
            continue  # tool-call-only turn: nothing classifiable
        digest = _digest(turn)
        digest.text = text[:snippet_chars]
        digest.tools = [c.function for c in calls]
        digest.delegations.extend(delegations)
        digest.event_id = getattr(event, "uuid", None)

    # Sub-agent-span delegations, snapped to the nearest digest-eligible
    # main-lane turn at or before the span_begin.
    for span, raw_anchor in _span_anchors(transcript, subagent_spans):
        if not eligible_turns:
            continue
        position = bisect_right(eligible_turns, raw_anchor) - 1
        anchor = eligible_turns[position] if position >= 0 else eligible_turns[0]
        text = span_task_text(span, first_models.get(span.id))[0]
        goal = _first_line(text) if text else str(span.name)
        _digest(anchor).delegations.append(goal[:snippet_chars])

    return [by_turn[turn] for turn in sorted(by_turn)]


def agent_task_prompt(transcript: Any, cap: int = TASK_PROMPT_CHARS) -> str:
    """Extract the task the agent was given.

    Source-generic: the first non-scaffold user message - the sample
    input on .eval sources, the initial brief on OpenClaw imports.

    Args:
        transcript: The transcript whose messages are scanned.
        cap: Maximum characters returned.

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
        return " ".join(text.split())[:cap]
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


def project_phase_turns(phase_starts: Sequence[int], n_turns: int) -> list[int | None]:
    """Assign every model turn (tool-call-only included) to a phase.

    A turn inside a phase belongs to it; a turn between phases inherits
    the previous one; turns before the first phase belong to the first.

    Args:
        phase_starts: Each phase's ``turn_start``, in phase order.
        n_turns: Total number of model turns in the transcript.

    Returns:
        ``phase_index_of_turn`` of length ``n_turns`` (indices into the
        phases list). All-None only when there are no phases at all.
    """
    assignment: list[int | None] = [None] * n_turns
    if not phase_starts:
        return assignment
    ordered = sorted(range(len(phase_starts)), key=lambda i: phase_starts[i])
    current = ordered[0]  # turns before the first phase -> the first
    position = 0
    for turn in range(n_turns):
        while position < len(ordered) and phase_starts[ordered[position]] <= turn:
            current = ordered[position]
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


class PhaseTurn(BaseModel):
    """One row of the dense ``turns`` surface."""

    turn: int
    phase_index: int | None = None
    basis: TurnBasis
    label_source: LabelSource | None = None
    confidence: float | None = None
    confidence_pm: float | None = None  # 95%-CI half-width; 0 non-cohort
    agreement: float | None = None  # vote agreement; None solo/unvoted


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
        n_turns: Total number of model turns in the transcript.
        n_members: Judge member count (decides the source labels).

    Returns:
        One ``PhaseTurn`` per turn. ``phase_index`` is the index into
        ``phases`` from the total-coverage projection.
    """
    assignment = project_phase_turns([p.turn_start for p in phases], n_turns)
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
                basis=row.basis if row is not None else "attributed",
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


def _span_anchors(transcript: Any, subagent_spans: list[Any]) -> list[tuple[Any, int]]:
    """Find the anchor turn for each sub-agent span.

    Args:
        transcript: The transcript whose event stream is walked.
        subagent_spans: The span_begin events to anchor.

    Returns:
        ``(span, anchor)`` pairs, where the raw anchor is the last
        model turn of any lane preceding the span_begin event.
    """
    wanted = {id(sp) for sp in subagent_spans}
    anchors: list[tuple[Any, int]] = []
    for turns_before, event in events_between_turns(transcript):
        if event.event == "span_begin" and id(event) in wanted:
            anchors.append((event, turns_before - 1))  # -1: nothing precedes
    return anchors


def _is_failed_turn(event: Any) -> bool:
    """Detect an OPENCLAW-SPECIFIC provider-failure placeholder turn.

    The OpenClaw scaffold writes a fake assistant turn ("[assistant turn
    failed ...]") when a provider call dies. Text match against the known
    2026-07 wording.

    Args:
        event: A model event.

    Returns:
        True when the turn is a content-free failure placeholder.
    """
    message = event.output.message if event.output else None
    text = (message.text or "").strip().lower() if message else ""
    return text.startswith(_OPENCLAW_FAILED_TURN_PREFIX)


def _first_line(text: str) -> str:
    """Return the first non-blank-stripped line of ``text``."""
    return text.splitlines()[0].strip() if text else text
