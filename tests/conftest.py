from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from inspect_scout import Result, Scanner, Transcript, scanner
from pydantic import JsonValue

import transect
from transect.report import Markdown, TurnBand, TurnChart
from transect.scanners.helpers import orchestrator_turns

FIXTURES = Path(__file__).parent / "fixtures"
DEMO_LOG = Path(__file__).parents[1] / "examples" / "logs" / "house_price_demo.eval"
SPEC = Path(__file__).parents[1] / "examples" / "spec.yaml"


@scanner(messages="all", events=["model", "span_begin", "span_end", "tool"])
def turn_counter() -> Scanner[Transcript]:
    """$0 structural custom scanner: one per-turn value row per
    orchestrator turn (the turn axis), with a taggable string column."""

    async def execute(transcript: Transcript) -> Result:
        turns = [
            {
                "turn": turn,
                "chars": len(event.output.message.text or ""),
                "band": "early" if turn < 5 else "late",
            }
            for turn, event, _calls in orchestrator_turns(transcript)
        ]
        return Result(value=cast(JsonValue, {"turns": turns}))

    return execute


@scanner(messages="all", events=["model", "span_begin", "span_end", "tool"])
def message_counter() -> Scanner[Transcript]:
    """$0 structural custom scanner: one flat value per transcript."""

    async def execute(transcript: Transcript) -> Result:
        turns = [
            {"turn": turn} for turn, _event, _calls in orchestrator_turns(transcript)
        ]
        return Result(
            value=cast(
                JsonValue, {"n_messages": len(transcript.messages), "turns": turns}
            )
        )

    return execute


@pytest.fixture(scope="session")
def layered_run(tmp_path_factory):
    """One $0 run mounting all three layer shapes at once, with tags
    on the per-turn layer; shared by the layer and tag suites."""
    if not DEMO_LOG.exists():
        pytest.skip("demo log missing - run tests/fixtures/generate_demo_eval.py")
    carried = pd.DataFrame({"turn": [1, 2], "shell": ["pip", "pytest"]})
    scans = tmp_path_factory.mktemp("scans")
    results = transect.transect(
        logs=str(DEMO_LOG),
        spec=str(SPEC),
        scans_dir=str(scans),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="turn_chars",
                scanner=turn_counter(),
                frame=transect.turns_frame,
                tags=True,
                section=[
                    Markdown("Per-turn *chars*, an e2e fixture layer."),
                    TurnChart(y="chars"),
                    TurnBand(label="band"),
                ],
            ),
            transect.Layer(name="raw_counts", scanner=message_counter()),
            transect.Layer(name="shell", frame=carried, tags=True),
        ],
    )
    return results, scans


@pytest.fixture(autouse=True)
def no_model_cache(monkeypatch):
    """Neuter inspect's on-disk model cache for every test.

    Scanner code calls generate(cache=True); without this, mockllm
    replies would persist in the user-level cache and replay across
    tests/runs whenever prompts collide."""
    monkeypatch.setattr("inspect_ai.model._model.cache_fetch", lambda entry: None)
    monkeypatch.setattr(
        "inspect_ai.model._model.cache_store", lambda entry, output: None
    )


@pytest.fixture
def fixture_logs() -> Path:
    return _log_dir("logs", "tests/fixtures/generate_eval_log.py")


@pytest.fixture
def epochs_logs() -> Path:
    return _log_dir("epochs_logs", "tests/fixtures/generate_epochs_log.py")


@pytest.fixture
def demo_log() -> Path:
    """The committed examples demo eval: handoff sub-agents, a
    compaction, and two human interventions in one log."""
    path = Path(__file__).parents[1] / "examples" / "logs" / "house_price_demo.eval"
    if not path.exists():
        pytest.skip(
            "demo log missing - run: python tests/fixtures/generate_demo_eval.py"
        )
    return path


@pytest.fixture
def parallel_logs() -> Path:
    """The committed parallel sub-agents eval: a deep agent running two
    background sub-agents that overlap each other and several of its
    own turns."""
    return _log_dir("parallel_logs", "tests/fixtures/generate_parallel_eval.py")


@pytest.fixture
def openclaw_log() -> Path:
    path = FIXTURES / "openclaw" / "mini_telemetry.jsonl"
    if not path.exists():
        pytest.fail(
            "committed openclaw fixture missing - regenerate: "
            "python tests/fixtures/generate_openclaw_telemetry.py"
        )
    return path


def _log_dir(name: str, generator: str) -> Path:
    directory = FIXTURES / name
    if not any(directory.glob("*.eval")):
        pytest.skip(f"fixture log missing - run: uv run python {generator}")
    return directory
