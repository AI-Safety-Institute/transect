"""Shared builders: stub transcripts, scripted judges, spec constants.

Every scripted judge is a mockllm with pre-baked structured answer()
calls, so no test makes an API call.
"""

import asyncio
import re
from pathlib import Path
from typing import Any

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


def model_turn(text, span_id=None, input=(), usage=None):
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
    )


def tool_event(id, function="bash", span_id=None, agent_span_id=None):
    return ToolEvent(
        id=id,
        function=function,
        arguments={},
        result="",
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
                    narrative(int(k), headline=f"Scripted headline for {label}")
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
