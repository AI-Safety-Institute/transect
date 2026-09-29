"""Generate the minimal Inspect eval log used as the transect test fixture.

Run from the repo root:

    uv run python tests/fixtures/generate_eval_log.py

Writes one .eval into tests/fixtures/logs/ (committed). A 2-sample react()
run against mockllm with scripted tool calls (bash, think, submit), so tests
exercise real transcript structure without API keys, Docker, or network.
"""

import itertools
import shutil
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import CompactionSummary, ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import think, tool

LOG_DIR = Path(__file__).parent / "logs"
MODEL = "mockllm/model"
COMPACTION_THRESHOLD = 1000
COMPACTION_INSTRUCTIONS = "Keep <paths> & decisions."
NUDGE_PREFIX = "Context compaction approaching. Use memory() to save"

# one sample's scripted turns: work (enough to cross the compaction
# threshold twice), reflect, submit. Summarization calls consume outputs
# from the same script (one per flush), so the pattern repeats until every
# sample has submitted; the length keeps submit clear of those calls.
TURNS = [("bash", {"command": f"echo step {i} && ls /tmp"}) for i in range(15)] + [
    ("think", {"thought": "output looks right, submitting"}),
    ("submit", {"answer": "DONE"}),
]


@tool
def bash():
    async def execute(command: str) -> str:
        """Pretend shell for fixture generation (no sandbox).

        Args:
            command: command to run
        """
        return f"ran: {command}\n" + "output line\n" * 20

    return execute


@tool
def memory():
    async def execute(command: str, path: str) -> str:
        """Stand-in for Inspect's memory tool (no sandbox).

        Args:
            command: memory command
            path: memory file path
        """
        return "saved"

    return execute


def _scripted_outputs():
    for tool_name, arguments in itertools.cycle(TURNS):
        yield ModelOutput.for_tool_call(MODEL, tool_name, arguments, content="working")


def _check_compaction(location: str) -> None:
    """Fail loudly if the log lost the shape tests read from it."""
    log = read_eval_log(location, resolve_attachments=True)
    for sample in log.samples or []:
        assert sample.limit is None, f"{sample.id} hit {sample.limit.type} limit"
        flushes = nudged = 0
        seen_nudge = False
        for e in sample.events:
            if e.event == "model":
                seen_nudge = seen_nudge or any(
                    m.role == "user" and m.text.startswith(NUDGE_PREFIX)
                    for m in e.input
                )
            elif e.event == "compaction":
                flushes += 1
                nudged += seen_nudge
                seen_nudge = False
        print(f"{sample.id}: {flushes} compactions, {nudged} preceded by a nudge")
        assert flushes >= 2, f"{sample.id} needs at least two compactions"
        assert nudged >= 2, f"{sample.id} needs the nudge before two flushes"


if __name__ == "__main__":
    samples = [
        Sample(id="fixture-sample-1", input="Say DONE.", target="DONE"),
        Sample(id="fixture-sample-2", input="Also say DONE.", target="DONE"),
    ]
    task = Task(
        name="fixture_task",
        dataset=samples,
        solver=react(
            prompt="You are a test fixture. Use your tools, then submit.",
            tools=[bash(), think(), memory()],
            compaction=CompactionSummary(
                threshold=COMPACTION_THRESHOLD, instructions=COMPACTION_INSTRUCTIONS
            ),
        ),
        scorer=includes(),
        message_limit=200,
    )
    model = get_model(MODEL, custom_outputs=_scripted_outputs())
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    # max_samples=1 serializes the samples so the scripted outputs cannot
    # interleave across samples
    logs = inspect_eval(
        task, model=model, log_dir=str(LOG_DIR), max_samples=1, display="plain"
    )
    assert logs[0].status == "success", logs[0].status
    _check_compaction(logs[0].location)
    print(f"fixture log written to {LOG_DIR}")
