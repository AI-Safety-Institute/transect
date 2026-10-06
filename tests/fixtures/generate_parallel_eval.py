"""Generate the parallel sub-agents fixture (tests only).

Run from the repo root:

    uv run python tests/fixtures/generate_parallel_eval.py

Writes one .eval into tests/fixtures/parallel_logs/ (committed): a mockllm
``deepagent(background=True)`` run whose orchestrator dispatches two
sub-agents in the background in its first turn, keeps working for two
turns, waits for them, then submits. Inspect appends the sub-agents'
events as they happen, so their model turns interleave with the
orchestrator's in the event stream. subagent_a compacts once in its own
lane (a summary compaction with no memory nudge).
"""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast

from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.agent import deepagent, subagent
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    CompactionSummary,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import Tool, ToolCall, tool

from transect.scanners.helpers import main_span, nearest_agent_span

LOG_DIR = Path(__file__).parent / "parallel_logs"
MOCK = "mockllm/model"
STAMP = "2026-10-05T12-00-00+00-00"
FIXTURE = LOG_DIR / f"{STAMP}_parallel-subagents_mockdeepagent.eval"

# the sub-agent system prompts: the routing key for their model calls
PROMPTS = {
    "subagent_a": "You are subagent_a. Survey the repository layout.",
    "subagent_b": "You are subagent_b. Check that the build runs.",
}

# the deterministic schedule (seconds): orchestrator model calls start
# ORCH_PITCH apart; sub-agent events start SUB_OFFSET after the spawn
# call, SUB_PITCH apart, with subagent_b lagging subagent_a by SUB_LAG
ORCH_PITCH = 100
ORCH_DURATION = 30
SUB_OFFSET = 20
SUB_PITCH = 25
SUB_LAG = 15

# subagent_a compacts once, between its first and second tool call: its
# first tool result is a long listing that carries its context over the
# threshold (tokens), and the short summary brings it back under. The
# summarization call is routed by its prompt, so it consumes no script.
COMPACTION_THRESHOLD = 500
LISTING_FILES = 300
SUBAGENT_A_SUMMARY = "Listed subagent_a: many files, README still to read."


@tool
def bash() -> Tool:
    async def execute(cmd: str) -> str:
        """Pretend shell for fixture generation (no sandbox).

        Args:
            cmd: command to run
        """
        if cmd == "ls subagent_a":
            return "\n".join(f"module_{i}.py" for i in range(LISTING_FILES))
        return f"ran: {cmd}"

    return execute


def _calls(content: str, *calls: tuple[str, dict[str, Any]]) -> ModelOutput:
    """One assistant message carrying several tool calls."""
    return ModelOutput.from_message(
        ChatMessageAssistant(
            content=content,
            tool_calls=[
                ToolCall(id=f"call-{index}-{name}", function=name, arguments=args)
                for index, (name, args) in enumerate(calls)
            ],
            model=MOCK,
        )
    )


# sub-agent calls answered so far, by name: a compaction empties the
# history, so the script position cannot be read off the input
_SUB_CALLS: dict[str, int] = {}


def route(input, tools, tool_choice, config) -> ModelOutput:
    """Content-routed script: a summarization call by Inspect's prompt,
    sub-agents by their system prompt and call count, the orchestrator
    by how many of its own turns precede this call."""
    system = "\n".join(m.text for m in input if m.role == "system")
    if "tasked with summarizing conversations" in (input[-1].text or ""):
        return ModelOutput.from_message(
            ChatMessageAssistant(content=SUBAGENT_A_SUMMARY, model=MOCK)
        )
    turns = sum(1 for m in input if m.role == "assistant")
    for name, prompt in PROMPTS.items():
        if prompt in system:
            script = [
                ModelOutput.for_tool_call(MOCK, "bash", {"cmd": f"ls {name}"}),
                ModelOutput.for_tool_call(MOCK, "bash", {"cmd": f"cat {name}/README"}),
                ModelOutput.for_tool_call(MOCK, "submit", {"answer": f"{name} done"}),
            ]
            call = _SUB_CALLS.get(name, 0)
            _SUB_CALLS[name] = call + 1
            return script[min(call, len(script) - 1)]
    script = [
        _calls(
            "Fanning out two sub-agents in the background.",
            (
                "agent",
                {"subagent_type": "subagent_a", "prompt": "survey", "background": True},
            ),
            (
                "agent",
                {"subagent_type": "subagent_b", "prompt": "test", "background": True},
            ),
        ),
        ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "work 1"}),
        ModelOutput.for_tool_call(MOCK, "bash", {"cmd": "work 2"}),
        ModelOutput.for_tool_call(
            MOCK, "agent_wait", {"agent_ids": ["AGENT-1", "AGENT-2"], "mode": "all"}
        ),
        ModelOutput.for_tool_call(MOCK, "submit", {"answer": "DONE"}),
    ]
    return script[min(turns, len(script) - 1)]


@task
def parallel_subagents() -> Task:
    model = get_model(MOCK, custom_outputs=route, memoize=False)
    return Task(
        dataset=[Sample(input="Scout the repo and report.", target="DONE")],
        solver=deepagent(
            tools=[bash()],
            subagents=[
                subagent(
                    name=name,
                    description=f"{name} helper",
                    prompt=prompt,
                    tools=[bash()],
                    compaction=CompactionSummary(
                        threshold=COMPACTION_THRESHOLD, memory=False
                    )
                    if name == "subagent_a"
                    else None,
                )
                for name, prompt in PROMPTS.items()
            ],
            memory=False,
            todo_write=False,
            compaction=None,
            background=True,
            model=model,
        ),
    )


def stretch_wallclock(log: EvalLog) -> None:
    """Rewrite event timestamps so the two sub-agents overlap in wall
    clock while the orchestrator keeps turning.

    Orchestrator events keep their order at ORCH_PITCH per model call;
    each sub-agent span's events start SUB_OFFSET after its spawn (the
    orchestrator's first model call) and advance SUB_PITCH per event,
    the second span lagging the first by SUB_LAG. Completion times
    follow each start by ORCH_DURATION.
    """
    assert log.samples
    sample = log.samples[0]
    events = cast("list[Any]", sample.events)
    spans = {e.id: e for e in events if e.event == "span_begin"}
    main = main_span(SimpleNamespace(events=events, timelines=[]))
    clock = events[0].timestamp
    lane_elapsed: dict[str, float] = {}
    orchestrator_elapsed = 0.0
    spawn_at = 0.0
    for event in events:
        agent = nearest_agent_span(spans, getattr(event, "span_id", None))
        if event.event == "span_begin" and getattr(event, "type", None) == "agent":
            agent = spans.get(event.id)
        lane = agent.id if agent is not None and agent.id != main.id else "__main__"
        if lane == "__main__":
            if event.event == "model":
                orchestrator_elapsed += ORCH_PITCH
                spawn_at = spawn_at or orchestrator_elapsed
            elapsed = orchestrator_elapsed
        else:
            if lane not in lane_elapsed:
                lane_elapsed[lane] = spawn_at + SUB_OFFSET + SUB_LAG * len(lane_elapsed)
            else:
                lane_elapsed[lane] += SUB_PITCH
            elapsed = lane_elapsed[lane]
        event.timestamp = clock + timedelta(seconds=elapsed)
        event.working_start = elapsed
        if hasattr(event, "completed") and event.completed is not None:
            event.completed = event.timestamp + timedelta(seconds=ORCH_DURATION)
            event.working_time = float(ORCH_DURATION)
    total = max(orchestrator_elapsed, *lane_elapsed.values()) + ORCH_DURATION
    sample.total_time = total
    sample.working_time = total
    log.stats.completed_at = (clock + timedelta(seconds=total)).isoformat()


def write_fixture(path: Path = FIXTURE) -> None:
    """Run the mock eval in a scratch log dir, stretch its clock, and
    write the result to ``path``."""
    with TemporaryDirectory() as scratch:
        (written,) = inspect_eval(
            parallel_subagents(), model=MOCK, log_dir=scratch, display="plain"
        )
        log = read_eval_log(
            written.location.removeprefix("file://"), resolve_attachments=True
        )
    assert log.status == "success", log.status
    stretch_wallclock(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    write_fixture()
