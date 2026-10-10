"""subagent_classification scanner: what was each sub-agent spawned to do.

Architecture: the ``agent_spans`` loader turns each sub-agent span
into a small synthetic Transcript - message one holds the span's
prepared task text, message two (when the span recorded any) a
digest of its actual activity; Scout then runs the classifier once
per span. By default the classifier judges the task text alone (the
digest message is stripped before the judge sees it); the
experimental ``activity`` knob switches to judging recorded
activity as well.

Closed vocab only: the spec must declare ``subagent_labels``.
"""

from collections.abc import AsyncIterator, Sequence

from inspect_ai.model import ChatMessage, ChatMessageUser, Model
from inspect_scout import Loader, Result, Scanner, Transcript, loader, scanner

from transect.scanners._labels import normalize_vocabulary
from transect.scanners.cohort import cohort_llm_scanner
from transect.scanners.helpers import (
    Lanes,
    SpanActivity,
    capped_lines,
    span_activity,
    span_task_text,
)
from transect.spec import Spec

NONE_OF_THE_ABOVE = "none_of_the_above"

_SPAWN_QUESTION = (
    "Above is the spawn task of ONE delegated sub-agent of an autonomous "
    "agent run. What was this sub-agent spawned to do?"
)

_ACTIVITY_QUESTION = (
    "Above are the spawn task of ONE delegated sub-agent of an autonomous "
    "agent run and, when recorded, a digest of its actual activity. What "
    "was this sub-agent's work? When the activity differs from the task, "
    "judge by what it actually did."
)

_TURN_EXCERPT_CHARS = 200  # per-turn excerpt bound in the activity digest
_ACTIVITY_LINES = 8  # digest turn excerpts kept (head+tail, middle elided)
_TOP_TOOLS = 6  # tool names shown on the summary line

_PROMPT_CHARS = 800  # bound per-span task text (loader-side preparation)


@loader(events=["span_begin", "span_end", "model", "tool"])
def agent_spans() -> Loader[Transcript]:
    """One synthetic Transcript per sub-agent span that has task text.

    Message one is the prepared task text; message two (only when
    the span recorded any activity) is the activity digest.
    """

    async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
        lanes = Lanes(transcript)
        for span in lanes.begins:  # event order
            task, source = span_task_text(span, lanes.first_models.get(span.id))
            task_truncated = len(task) > _PROMPT_CHARS
            task = task[:_PROMPT_CHARS]
            if not task:
                continue  # no classifier input -> no judge call, no row
            messages: list[ChatMessage] = [
                ChatMessageUser(content=f"name: {span.name}\ntask: {task}")
            ]
            digest = _activity_digest(span_activity(lanes, span.id))
            if digest:
                messages.append(ChatMessageUser(content=digest))
            yield Transcript(
                transcript_id=span.id,  # -> results input_ids (the join key)
                messages=messages,
                metadata={
                    "agent_span_id": span.id,
                    "agent_lane": span.name,
                    "span_task_source": source,
                    # the exact task text the judge saw
                    "span_task": task,
                    "span_task_truncated": task_truncated,
                },
            )

    return load


@scanner(loader=agent_spans())
def subagent_classification(
    spec: Spec,
    judge_models: str | Model | Sequence[str | Model] | None = None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verifier_model: str | Model | None = None,
    verify_sample: float | None = None,
    activity: bool = False,
) -> Scanner[Transcript]:
    """One label per agent span, via ``cohort_llm_scanner``.

    Args:
        spec: Supplies the closed ``subagent_labels`` vocabulary.
        judge_models: One model, or cohort (majority-vote).
        k_rolls: Repeated rolls of one model.
        verify: Second-round verifier for doubtful judgements.
        verifier_model: Verifier judge; ``None`` uses the judge model.
        verify_sample: Per-span random spot-check probability on top
            of the doubt triggers (see ``cohort_llm_scanner``);
            ``None`` = the default 5%, ``0.0`` = doubt-only.
        activity: (Experimental) Also show the judge(s) (and verifier) a
            digest of the span's recorded activity, and ask it to judge
            by what the sub-agent actually did (off by default).

    Raises:
        ValueError: When the spec declares no ``subagent_labels``.
    """
    if not spec.subagent_labels:
        raise ValueError("subagent_classification requires ``spec.subagent_labels``.")
    vocabulary = normalize_vocabulary(
        [
            {"label": entry.label, "description": entry.description, "reserved": False}
            for entry in spec.subagent_labels
        ]
    )
    if all(entry["label"] != NONE_OF_THE_ABOVE for entry in vocabulary):
        vocabulary.append(
            {
                "label": NONE_OF_THE_ABOVE,
                "description": "fits none of the labels above",
                "reserved": True,
            }
        )
    labels = [str(entry["label"]) for entry in vocabulary]
    definitions = [
        f"- {entry['label']}: {entry['description']}"
        for entry in vocabulary
        if not entry["reserved"] and entry["description"]
    ]
    question = _ACTIVITY_QUESTION if activity else _SPAWN_QUESTION
    if definitions:
        definitions.append(f"- {NONE_OF_THE_ABOVE}: fits none of the labels above.")
        question += "\n\nLabel definitions:\n" + "\n".join(definitions)
    scan = cohort_llm_scanner(
        question=question,
        answer=labels,
        models=judge_models,
        k_rolls=k_rolls,
        verify=verify,
        verifier_model=verifier_model,
        verify_sample=verify_sample,
        vocabulary=vocabulary,
    )
    if activity:
        return scan

    async def task_only(transcript: Transcript) -> Result | list[Result]:
        # drop the loader's digest message
        if len(transcript.messages) > 1:
            transcript = transcript.model_copy(
                update={"messages": transcript.messages[:1]}
            )
        return await scan(transcript)

    return task_only


def _activity_digest(activity: SpanActivity) -> str:
    """Render a span's recorded activity for the judge.

    One summary line (turn/tool-call totals; top tool names by count)
    plus every model-turn excerpt bounded per turn, the middle elided
    via ``capped_lines`` when the run is long.
    """
    n_turns = len(activity.turn_texts)
    n_tools = sum(activity.tool_counts.values())
    if not n_turns and not n_tools:
        return ""
    # only count what was recorded: "0 turns" would read as a contradiction
    counts = []
    if n_turns:
        counts.append(f"{n_turns} turn" + ("s" if n_turns != 1 else ""))
    if n_tools:
        counts.append(f"{n_tools} tool call" + ("s" if n_tools != 1 else ""))
    summary = "activity: " + ", ".join(counts)
    if activity.tool_counts:
        top = sorted(activity.tool_counts.items(), key=lambda kv: -kv[1])
        shown = ", ".join(f"{name} x{count}" for name, count in top[:_TOP_TOOLS])
        summary += f" ({shown})"
    excerpts = [f"- {text[:_TURN_EXCERPT_CHARS]}" for text in activity.turn_texts]
    return "\n".join([summary, *capped_lines(excerpts, _ACTIVITY_LINES)])
