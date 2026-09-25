"""Concurrent judge calls are all-or-nothing: a failure cancels its siblings."""

import asyncio

import pytest
from helpers import MODEL, StubTranscript, model_turn, scripted_judge, seg, seg_answer
from inspect_ai.model import get_model

from transect.scanners.phases import decision_phases
from transect.scanners.phases_common import gather_judge_calls
from transect.spec import Spec


class JudgeDown(RuntimeError):
    pass


def _pending() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def _sibling(started: asyncio.Event, log: list[str]) -> str:
    started.set()
    try:
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0)  # cleanup that itself yields
        log.append("sibling cleaned up")
    return "never"


async def _failing(started: asyncio.Event) -> str:
    await started.wait()
    raise JudgeDown("provider gone")


def test_failure_cancels_and_awaits_siblings_then_reraises_the_original():
    """The first failure surfaces as itself once every sibling has cleaned up."""
    log: list[str] = []

    async def run():
        started = asyncio.Event()
        with pytest.raises(JudgeDown, match="provider gone"):
            await gather_judge_calls([_sibling(started, log), _failing(started)])
        # checked inside the loop: asyncio.run's shutdown would otherwise
        # cancel a leaked sibling for us and hide the leak
        assert log == ["sibling cleaned up"]
        assert _pending() == []

    asyncio.run(run())


def test_results_keep_call_order():
    """Results come back in the order the calls were given, not completion order."""

    async def slow(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    async def run():
        return await gather_judge_calls([slow(1, 0.02), slow(2, 0.0), slow(3, 0.01)])

    assert asyncio.run(run()) == [1, 2, 3]


def test_cancelling_the_caller_cancels_every_call():
    """Cancellation from outside reaches the calls and propagates as cancellation."""
    log: list[str] = []

    async def run():
        started = asyncio.Event()
        task = asyncio.create_task(
            gather_judge_calls([_sibling(started, log), _sibling(started, log)])
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert log == ["sibling cleaned up", "sibling cleaned up"]
        assert _pending() == []

    asyncio.run(run())


def test_segmentation_member_error_fails_the_scan_cleanly():
    """A member call raising a non-refusal error surfaces from the scan as that
    error (Scout records it per transcript) with no judge task left pending."""

    def broken(input, tools, tool_choice, config):
        raise JudgeDown("quota")

    good = scripted_judge(*[seg_answer(seg(0, 2, "setup"))] * 4)
    bad = get_model(f"{MODEL}2", custom_outputs=broken, memoize=False)
    spec = Spec.model_validate({"phases": ["setup", "experiment"]})
    scan = decision_phases(spec, judge_models=[good, bad], verify=False)
    transcript = StubTranscript([model_turn(f"turn {i}") for i in range(3)])

    async def run():
        with pytest.raises(JudgeDown, match="quota"):
            await scan(transcript)
        assert _pending() == []

    asyncio.run(run())
