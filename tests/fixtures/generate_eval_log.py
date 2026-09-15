"""Generate the minimal Inspect eval log used as the transect test fixture.

Run from the repo root:

    uv run python tests/fixtures/generate_eval_log.py

Writes one .eval into tests/fixtures/logs/ (committed). A 2-sample react()
run against mockllm with scripted tool calls (bash, think, submit), so tests
exercise real transcript structure without API keys, Docker, or network.
"""

import shutil
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import think, tool

LOG_DIR = Path(__file__).parent / "logs"
MODEL = "mockllm/model"

# one sample's scripted turns: work, reflect, submit
TURNS = [
    ("bash", {"command": "echo hi && ls /tmp"}),
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
        return f"ran: {command}"

    return execute


def _scripted_outputs(n_samples: int) -> list[ModelOutput]:
    outputs = []
    for _ in range(n_samples):
        for tool_name, arguments in TURNS:
            output = ModelOutput.for_tool_call(MODEL, tool_name, arguments)
            output.usage = ModelUsage(
                input_tokens=100, output_tokens=40, total_tokens=140
            )
            outputs.append(output)
    return outputs


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
            tools=[bash(), think()],
        ),
        scorer=includes(),
    )
    model = get_model(MODEL, custom_outputs=_scripted_outputs(len(samples)))
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    # max_samples=1 serializes the samples so the scripted outputs cannot
    # interleave across samples
    logs = inspect_eval(
        task, model=model, log_dir=str(LOG_DIR), max_samples=1, display="plain"
    )
    assert logs[0].status == "success", logs[0].status
    print(f"fixture log written to {LOG_DIR}")
