"""Shared builders: stub transcripts, scripted judges, spec constants.

Every scripted judge is a mockllm with pre-baked structured answer()
calls, so no test makes an API call.
"""

import asyncio
import re
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.event import ModelEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from transect.spec import Spec

MODEL = "mockllm/model"
JSON_SPEC = str(Path(__file__).parent / "fixtures" / "specs" / "minimal.json")

PHASES_SPEC = Spec.model_validate(
    {
        "phases": [
            {"label": "setup", "description": "Environment preparation."},
            "experiment",
        ],
        "context": "The agent is reproducing a systems paper.",
    }
)
CLOSED = Spec.model_validate(
    {"subagent_labels": ["literature_survey", "experiment_run"]}
)

REFUSED = ModelOutput.from_content(MODEL, "", stop_reason="content_filter")


class StubTranscript:
    """Container fake: real inspect events, no stored timelines."""

    def __init__(self, events, messages=()):
        self.events = list(events)
        self.messages = list(messages)
        self.timelines = []
        self.source_type = "eval_log"
        self.source_uri = None


def model_turn(text, span_id=None, input=(), usage=None, error=None):
    output = ModelOutput.from_content(MODEL, text)
    if usage is not None:
        output.usage = usage
    return ModelEvent(
        model="m",
        input=list(input),
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output,
        span_id=span_id,
        error=error,
    )


def call_turn(function, arguments, call_id, text="", input=()):
    """A model turn that makes one tool call (optionally with text)."""
    return ModelEvent(
        model="m",
        input=list(input),
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.for_tool_call(
            MODEL, function, arguments, tool_call_id=call_id, content=text
        ),
    )


def tool_event(
    id, function="bash", span_id=None, agent_span_id=None, result="", error=None
):
    return ToolEvent(
        id=id,
        function=function,
        arguments={},
        result=result,
        error=error,
        span_id=span_id,
        agent_span_id=agent_span_id,
    )


def agent_span(id, name, inner=(), metadata=None, parent_id=None):
    """A bracketed agent span; direct inner model events get span_id stamped."""
    inner = list(inner)
    for event in inner:
        if isinstance(event, ModelEvent) and event.span_id is None:
            event.span_id = id
    return [
        SpanBeginEvent(
            id=id, parent_id=parent_id, type="agent", name=name, metadata=metadata
        ),
        *inner,
        SpanEndEvent(id=id),
    ]


def run_scan(scan_fn, events, messages=()) -> Any:
    return run_item(scan_fn, StubTranscript(events, messages))


def run_item(scan_fn, item) -> Any:
    """Run one scanner call; Any return - tests assert dynamically.

    (Scanner returns an Awaitable, which asyncio.run rejects
    statically; the wrapper makes a coroutine.)"""

    async def call():
        return await scan_fn(item)

    return asyncio.run(call())


def scripted_judge(*outputs, model=MODEL):
    return get_model(model, custom_outputs=list(outputs), memoize=False)


def vote_output(label, confidence=0.8, explanation="r", model=MODEL):
    return ModelOutput.for_tool_call(
        model,
        "answer",
        {"label": label, "confidence": confidence, "explanation": explanation},
    )


def voter(model, label, confidence=0.8, explanation="r"):
    return get_model(
        model,
        custom_outputs=[vote_output(label, confidence, explanation, model=model)],
        memoize=False,
    )


def seg(turn_start, turn_end, phase, confidence=0.9, explanation="x"):
    return {
        "turn_start": turn_start,
        "turn_end": turn_end,
        "phase": phase,
        "confidence": confidence,
        "explanation": explanation,
    }


def seg_answer(*segments):
    return ModelOutput.for_tool_call(
        MODEL, "answer", {"segments": list(segments), "explanation": "segmented"}
    )


def scripted_groups(prompt: str, phase_index: int) -> list[dict]:
    """Two contiguous turn groups splitting the phase's digest lines in the
    narrator prompt (one group when the phase is a single turn)."""
    block = re.search(
        rf"PHASE {phase_index} \[[^\]]*\]:\n((?:\d+:.*(?:\n|$))+)", prompt
    )
    assert block, f"phase {phase_index} block missing from the narrator prompt"
    turns = [int(n) for n in re.findall(r"^(\d+):", block.group(1), re.M)]
    first, last = turns[0], turns[-1]
    if first == last:
        return [group(first, last, title="Whole phase", gist="scripted")]
    mid = (first + last) // 2
    return [
        group(first, mid, title="Opening moves", gist="scripted"),
        group(mid + 1, last, title="Follow-through", gist="scripted"),
    ]


def narrative(phase_index, headline="did work", summary="Did the work.", groups=()):
    return {
        "phase_index": phase_index,
        "headline": headline,
        "summary": summary,
        "groups": list(groups),
    }


def group(turn_start, turn_end, title="part", gist="g"):
    return {
        "turn_start": turn_start,
        "turn_end": turn_end,
        "title": title,
        "gist": gist,
    }


def narrate_answer(*narratives):
    return ModelOutput.for_tool_call(
        MODEL, "answer", {"narratives": list(narratives), "explanation": "narrated"}
    )


def verify_answer(*verdicts):
    return ModelOutput.for_tool_call(
        MODEL, "answer", {"verdicts": list(verdicts), "explanation": "reviewed"}
    )


def demo_judge(model=MODEL, phase_label="model_development", subagent_labels=None):
    """A content-routing scripted judge for whole-pipeline runs."""
    labels = subagent_labels or {
        "eda": "data_analysis",
        "alt_model": "model_experimentation",
        "reviewer": "result_review",
    }

    def route(input, tools, tool_choice, config) -> ModelOutput:
        text = "\n".join(getattr(m, "text", "") or "" for m in input)
        if "Review these phases:" in text:
            return verify_answer(
                *(
                    {
                        "phase_index": int(k),
                        "phase": current,
                        "confidence": 0.9,
                        "explanation": "label holds",
                    }
                    for k, current in re.findall(r"PHASE (\d+) \[current=(\w+)", text)
                )
            )
        if "Narrate these phases:" in text:
            return narrate_answer(
                *(
                    narrative(
                        int(k),
                        headline=f"Scripted headline for {label}",
                        groups=scripted_groups(text, int(k)),
                    )
                    for k, label in re.findall(r"PHASE (\d+) \[(\w+),", text)
                )
            )
        segment_range = re.search(r"Segment turns (\d+)\.\.(\d+)", text)
        if segment_range:
            first, last = map(int, segment_range.groups())
            return seg_answer(seg(first, last, phase_label, 0.9))
        lane = re.search(r"name: (\w+)", text)
        if lane and "spawn task" in text:
            return vote_output(labels[lane.group(1)], 0.9, "scripted", model=model)
        raise AssertionError(f"unroutable judge request: {text[:200]}")

    return get_model(model, custom_outputs=route, memoize=False)


def phase_turns_frame(bases, phase_index):
    """A minimal phase_turns frame: one row per basis, with the columns the
    band geometry and the audit's turn counts read."""
    return pd.DataFrame(
        {
            "transcript_id": ["t1"] * len(bases),
            "turn": range(len(bases)),
            "phase_index": pd.array(phase_index, dtype="Int64"),
            "basis": list(bases),
            "judge_agreement": [None] * len(bases),
        }
    )


def petri_scaffold(roles=True, reply=None, reply_tools=()):
    """`auditor_scaffold` as a Petri audit: the model roles recorded (or
    not, with ``roles`` off), every target reply's text set to ``reply``
    when given and its tool calls to ``reply_tools``."""
    from inspect_ai.tool import ToolCall

    transcript = auditor_scaffold()
    if roles:
        transcript.metadata = {"model_roles": {"auditor": "a", "target": "t"}}
    for event in transcript.events:
        if getattr(event, "role", None) != "target":
            continue
        if reply is not None:
            event.output.message.content = reply
        event.output.message.tool_calls = [
            ToolCall(id=f"{name}-{n}", function=name, arguments={})
            for n, name in enumerate(reply_tools)
        ]
    return transcript


def auditor_scaffold(target_tools=False):
    """An auditor driving a target through tools: both roles' calls in one
    solver span with no agent-type span anywhere (the inspect_petri
    shape), the target attempted twice (one trajectory span per attempt),
    the auditor's own conversation compacted once, and a judge call under
    the scorers phase. With ``target_tools`` the target calls tools too,
    so Inspect's timeline leaves its calls beside the auditor's instead of
    wrapping them as utility helpers."""
    from inspect_ai.event import CompactionEvent
    from inspect_ai.model import ChatMessageSystem, ModelUsage
    from inspect_ai.tool import ToolCall

    auditor = [model_turn(f"auditor {n}") for n in range(3)]
    target = [model_turn(f"target {n}") for n in range(3)]
    for n, event in enumerate(auditor):
        event.role, event.model, event.span_id = "auditor", "m/auditor", "auditor"
        event.input = [ChatMessageSystem(content="You are the auditor.")]
        event.output.message.tool_calls = [
            ToolCall(id=f"call-{n}", function="send_message", arguments={})
        ]
    for n, event in enumerate(target):
        event.role, event.model = "target", "m/target"
        event.input = [ChatMessageSystem(content="You are a helpful assistant.")]
        if target_tools:
            event.output.message.tool_calls = [
                ToolCall(id=f"target-call-{n}", function="bash", arguments={})
            ]
    judge = model_turn("judge")
    judge.role, judge.model, judge.span_id = "judge", "m/judge", "judge"
    target[0].span_id = target[1].span_id = "attempt-1"
    target[2].span_id = "attempt-2"
    for event in (*auditor, *target):
        event.output.usage = ModelUsage(
            input_tokens=1000, output_tokens=10, total_tokens=1010
        )
    events = [
        SpanBeginEvent(id="solvers", parent_id=None, type="solvers", name="solvers"),
        SpanBeginEvent(id="solver", parent_id="solvers", type="solver", name="audit"),
        SpanBeginEvent(
            id="auditor", parent_id="solver", type="auditor", name="auditor"
        ),
        SpanBeginEvent(id="target", parent_id="solver", type="target", name="target"),
        auditor[0],
        SpanBeginEvent(
            id="attempt-1", parent_id="target", type="trajectory", name="trajectory"
        ),
        target[0],
        auditor[1],
        target[1],
        SpanEndEvent(id="attempt-1"),
        CompactionEvent(
            type="summary",
            source="inspect",
            tokens_before=5000,
            tokens_after=500,
            role="auditor",
        ),
        auditor[2],
        SpanBeginEvent(
            id="attempt-2", parent_id="target", type="trajectory", name="trajectory"
        ),
        target[2],
        SpanEndEvent(id="attempt-2"),
        SpanEndEvent(id="target"),
        SpanEndEvent(id="auditor"),
        SpanEndEvent(id="solver"),
        SpanEndEvent(id="solvers"),
        SpanBeginEvent(id="scorers", parent_id=None, type="scorers", name="scorers"),
        SpanBeginEvent(id="judge", parent_id="scorers", type="scorer", name="judge"),
        judge,
        SpanEndEvent(id="judge"),
        SpanEndEvent(id="scorers"),
    ]
    return StubTranscript(events)
