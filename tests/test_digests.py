"""What a digest line carries: tool calls with their arguments and, when
switched on, their recorded output; tool-call-only turns; spawning and
handoff calls; the settings every judge shares."""

import asyncio

import pytest
from helpers import (
    MODEL,
    PHASES_SPEC,
    StubTranscript,
    agent_span,
    model_turn,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
    tool_event,
)
from inspect_ai.event import ModelEvent
from inspect_ai.model import ChatMessageTool, ContentText, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCallError

from transect import DigestSettings, reasoning_turns
from transect.scanners.phases import decision_phases, system_prompt, turn_digests
from transect.scanners.phases_common import DIGEST_MARKERS, digest_line
from transect.scanners.phases_narrate import narrate_system_prompt
from transect.scanners.phases_verify import verify_system_prompt

RESULTS_ON = DigestSettings(tool_result_chars=200)
COMMAND = "cd repo && pytest -q tests"
OUTPUT = "2 failed, 40 passed in 1.2s"


def call_turn(function, arguments, call_id, text=""):
    """A model turn that makes one tool call (optionally with text)."""
    return ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.for_tool_call(
            MODEL, function, arguments, tool_call_id=call_id, content=text
        ),
    )


def lines(events, messages=(), digest=None):
    return [
        digest_line(d) for d in turn_digests(StubTranscript(events, messages), digest)
    ]


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (DigestSettings(), f'0: go [CALL] bash({{"cmd": "{COMMAND}"}})'),
        (
            DigestSettings(tool_call_chars=8),
            '0: go [CALL] bash({"cm [... 29 chars ...] ts"})',
        ),
        (RESULTS_ON, f'0: go [CALL] bash({{"cmd": "{COMMAND}"}}) [RESULT] {OUTPUT}'),
        (
            DigestSettings(tool_result_chars=3),
            f'0: go [CALL] bash({{"cmd": "{COMMAND}"}})'
            " [RESULT] 2 [... 24 chars ...] 2s",
        ),
        (DigestSettings(tool_call_chars=0), "0: go (tools: bash)"),
    ],
    ids=["default", "args-cap", "results-on", "result-cap", "names-only"],
)
def test_tool_calls_render_within_their_caps(settings, expected):
    """Arguments show by default and output on request, each clipped at
    both ends by its own cap; a cap of 0 leaves the part out."""
    events = [
        call_turn("bash", {"cmd": COMMAND}, "c1", text="go"),
        tool_event("c1", result=OUTPUT),
    ]
    assert lines(events, digest=settings) == [expected]


@pytest.mark.parametrize(
    ("events", "messages", "expected"),
    [
        (
            [
                call_turn("bash", {"cmd": "sleep 999"}, "c1", text="wait"),
                tool_event("c1", error=ToolCallError("timeout", "Command timed out")),
            ],
            [],
            ['0: wait [CALL] bash({"cmd": "sleep 999"}) [ERROR] Command timed out'],
        ),
        (
            [
                call_turn("bash", {"cmd": "ls"}, "c1", text="first"),
                tool_event("c1", result="a.py"),
                call_turn("bash", {"cmd": "ls"}, "c1", text="again"),
                tool_event("c1", result="b.py"),
            ],
            [],
            [
                '0: first [CALL] bash({"cmd": "ls"}) [RESULT] a.py',
                '1: again [CALL] bash({"cmd": "ls"}) [RESULT] b.py',
            ],
        ),
        (
            [call_turn("bash", {"cmd": "ls"}, "c1", text="list")],
            [ChatMessageTool(content="README.md", tool_call_id="c1")],
            ['0: list [CALL] bash({"cmd": "ls"}) [RESULT] README.md'],
        ),
        (
            [
                call_turn("read", {"path": "x"}, "c1", text="read"),
                tool_event("c1", result=[ContentText(text="line 1\nline 2")]),
            ],
            [],
            ['0: read [CALL] read({"path": "x"}) [RESULT] line 1 line 2'],
        ),
    ],
    ids=["error", "reused-id", "message-fallback", "content-blocks"],
)
def test_output_is_the_first_tool_event_after_the_turn_else_the_tool_message(
    events, messages, expected
):
    """An error renders behind [ERROR]; a reused call id resolves to the
    event after its own turn; a source without tool events falls back
    to the answering tool message; content blocks render their text."""
    assert lines(events, messages, digest=RESULTS_ON) == expected


@pytest.mark.parametrize(
    ("tool_only_turns", "digest_turns", "bases"),
    [(True, [0, 1], ["judged", "judged"]), (False, [0], ["judged", "attributed"])],
    ids=["kept", "dropped"],
)
def test_tool_call_only_turns_are_judged_unless_switched_off(
    tool_only_turns, digest_turns, bases
):
    """A turn with only a tool call gets a digest by default; switched
    off, it takes its label by attribution as before."""
    settings = DigestSettings(tool_only_turns=tool_only_turns)
    events = [
        model_turn("Starting."),
        call_turn("bash", {"cmd": "git push --no-verify"}, "c1"),
        tool_event("c1", result="pushed"),
    ]
    digests = turn_digests(StubTranscript(events), settings)
    assert [d.turn for d in digests] == digest_turns
    judge = scripted_judge(seg_answer(seg(0, digest_turns[-1], "setup", 0.9)))
    value = run_scan(
        decision_phases(
            PHASES_SPEC, judge, digest=settings, verify=False, narrate=False
        ),
        events,
    ).value
    assert [t["basis"] for t in value["turns"]] == bases


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [
                call_turn("spawn", {"task": "survey data"}, "c1", text="split"),
                tool_event("c1", result='{"status": "accepted"}'),
            ],
            "0: split (tools: spawn) [DELEGATES] survey data",
        ),
        (
            [
                call_turn(
                    "sessions_spawn", {"task": "survey data"}, "c1", text="split"
                ),
                tool_event("c1", result='{"status": "accepted"}'),
                *agent_span(
                    "s1",
                    "sub",
                    inner=[model_turn("working")],
                    metadata={"task": "survey data"},
                ),
            ],
            "0: split (tools: sessions_spawn) [DELEGATES] survey data",
        ),
        (
            [
                call_turn("transfer_to_eda", {}, "c1", text="hand off"),
                tool_event("c1", result="Successfully transferred to eda."),
            ],
            "0: hand off (tools: transfer_to_eda)",
        ),
    ],
    ids=["span-less", "span-based", "handoff"],
)
def test_spawning_and_handoff_calls_show_their_task_once(events, expected):
    """A spawn's task is the delegation line whether it comes from the
    call's arguments or the span, so the call shows neither arguments
    nor its acceptance output; a handoff's transfer boilerplate is not
    output either."""
    assert lines(events, digest=RESULTS_ON)[0] == expected


def test_every_judge_prompt_explains_the_markers():
    """The segmenter's rules name [CALL]; the verifier and the narrator
    carry the shared marker legend."""
    assert "[CALL]" in system_prompt(PHASES_SPEC)
    assert DIGEST_MARKERS in verify_system_prompt(PHASES_SPEC, "")
    assert DIGEST_MARKERS in narrate_system_prompt(PHASES_SPEC, "")


def test_a_negative_cap_is_rejected_naming_the_setting():
    """The settings object refuses a negative cap before any scan."""
    with pytest.raises(ValueError, match="tool_result_chars"):
        DigestSettings(tool_result_chars=-1)


def test_reasoning_turns_items_carry_the_same_digest_lines():
    """The per-turn loader yields the decision_phases digest line,
    honouring the same settings."""
    events = [
        call_turn("bash", {"cmd": "ls"}, "c1", text="list"),
        tool_event("c1", result="README.md"),
    ]
    transcript = StubTranscript(events)
    transcript.transcript_id = "t"  # type: ignore[attr-defined]

    async def items(digest=None):
        return [
            item.messages[0].text
            async for item in reasoning_turns(digest=digest)(transcript)  # type: ignore[arg-type]
        ]

    assert asyncio.run(items()) == ['0: list [CALL] bash({"cmd": "ls"})']
    assert asyncio.run(items(RESULTS_ON)) == [
        '0: list [CALL] bash({"cmd": "ls"}) [RESULT] README.md'
    ]
