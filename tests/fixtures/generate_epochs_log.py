"""Generate the multi-epoch Inspect eval log used as the transect test fixture.

Run from the repo root:

    uv run python tests/fixtures/generate_epochs_log.py

Writes one .eval into tests/fixtures/epochs_logs/ (committed). Two samples
x three epochs against mockllm:

    epoch-sample-1: epoch 1 FAILS, epochs 2..3 succeed -> auto picks 2
    epoch-sample-2: every epoch FAILS -> auto falls back to epoch 1
"""

import shutil
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import think, tool

LOG_DIR = Path(__file__).parent / "epochs_logs"
MODEL = "mockllm/model"
EPOCHS = 3

# (sample id, epoch, 1-based) -> should this run succeed?
OUTCOME = {
    ("epoch-sample-1", 1): False,
    ("epoch-sample-1", 2): True,
    ("epoch-sample-1", 3): True,
    ("epoch-sample-2", 1): False,
    ("epoch-sample-2", 2): False,
    ("epoch-sample-2", 3): False,
}


@tool
def bash():
    async def execute(command: str) -> str:
        """Pretend shell for fixture generation (no sandbox).

        Args:
            command: command to run
        """
        return f"ran: {command}"

    return execute


def _scripted_outputs() -> list[ModelOutput]:
    """One work/think/submit sequence per (sample, epoch) run."""
    outputs = []
    for (_, _), succeed in sorted(OUTCOME.items(), key=lambda kv: kv[0][::-1]):
        turns = [
            ("bash", {"command": "echo hi"}),
            ("think", {"thought": "checking"}),
            ("submit", {"answer": "DONE" if succeed else "NOPE"}),
        ]
        for tool_name, arguments in turns:
            output = ModelOutput.for_tool_call(MODEL, tool_name, arguments)
            output.usage = ModelUsage(
                input_tokens=100, output_tokens=40, total_tokens=140
            )
            outputs.append(output)
    return outputs


if __name__ == "__main__":
    samples = [
        Sample(id="epoch-sample-1", input="Say DONE.", target="DONE"),
        Sample(id="epoch-sample-2", input="Also say DONE.", target="DONE"),
    ]
    task = Task(
        name="epochs_fixture_task",
        dataset=samples,
        solver=react(
            prompt="You are a test fixture. Use your tools, then submit.",
            tools=[bash(), think()],
        ),
        scorer=includes(),
        epochs=EPOCHS,
    )
    model = get_model(MODEL, custom_outputs=_scripted_outputs())
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    # max_samples=1 serializes the runs so the scripted outputs cannot
    # interleave across (sample, epoch) pairs
    logs = inspect_eval(
        task, model=model, log_dir=str(LOG_DIR), max_samples=1, display="plain"
    )
    assert logs[0].status == "success", logs[0].status

    log = read_eval_log(logs[0].location)
    assert log.samples
    got = {
        (s.id, s.epoch): (s.scores or {})["includes"].value == "C" for s in log.samples
    }
    assert got == OUTCOME, f"outcome drift:\n  wanted {OUTCOME}\n  got    {got}"
    print(f"multi-epoch fixture log written to {LOG_DIR}")
    for key in sorted(got):
        print(f"  {key}: {'success' if got[key] else 'fail'}")
