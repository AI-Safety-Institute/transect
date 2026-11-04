"""The synthetic nested audit is a usable, reproducible Inspect text fixture."""

import asyncio
import runpy
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_scout import TranscriptContent, transcripts_from

from transect.petri.conversation import target_conversation
from transect.scanners.digests import turn_digests
from transect.scanners.helpers import Lanes

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "nested_petri/nested_audit.json"


def test_nested_text_fixture_loads_with_the_recorded_auditor_axis_and_branches(
    tmp_path,
):
    """Scout reads the outer loop and inner audit without confusing their roles."""
    log = read_eval_log(FIXTURE)
    assert log.samples is not None and len(log.samples) == 1
    assert Counter(e.role for e in log.samples[0].events if e.event == "model") == {
        "contestant": 2,
        "auditor": 6,
        "target": 4,
        "classifier": 1,
    }

    compressed = tmp_path / "nested_audit.eval"
    write_eval_log(log, str(compressed))

    async def read():
        async with transcripts_from(str(compressed)).reader() as reader:
            infos = [info async for info in reader.index()]
            assert len(infos) == 1
            return await reader.read(
                infos[0],
                TranscriptContent(
                    messages="all",
                    events=[
                        "model",
                        "span_begin",
                        "span_end",
                        "tool",
                        "anchor",
                        "branch",
                    ],
                    timeline=True,
                ),
            )

    transcript = asyncio.run(read())
    lanes = Lanes(transcript)
    assert lanes.axis_role == "auditor"
    assert len(list(lanes.turns())) == 6
    assert [len(d.target_replies) for d in turn_digests(transcript)] == [
        0,
        1,
        1,
        1,
        1,
        0,
    ]
    conversation = target_conversation(transcript)
    assert conversation is not None
    assert len(conversation.calls) == 4
    assert len(conversation.branches) == 2
    assert sum(b.rollback_turn is not None for b in conversation.branches) == 1
    assert conversation.surviving is None
    assert conversation.note == "the final conversation holds no target reply"


def test_nested_text_fixture_regenerates_without_model_calls():
    """The committed text is exactly reproducible from the public scripted log."""
    module = runpy.run_path(str(FIXTURES / "generate_nested_petri_eval.py"))
    regenerated = module["build"]().model_dump_json(indent=2, exclude_none=True) + "\n"
    assert regenerated == FIXTURE.read_text()
