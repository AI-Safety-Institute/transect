"""What a digest line carries: tool calls with their recorded output,
tool-call-only turns, and the user messages a turn received."""

import asyncio

import pytest
from helpers import (
    MODEL,
    PHASES_SPEC,
    StubTranscript,
    model_turn,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
)
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCallError

from transect import reasoning_turns
from transect.scanners.phases import decision_phases, system_prompt, turn_digests
from transect.scanners.phases_common import DIGEST_MARKERS, digest_line
from transect.scanners.phases_narrate import narrate_system_prompt
from transect.scanners.phases_verify import verify_system_prompt


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


def tool_result(call_id, result="", error=None):
    return ToolEvent(
        id=call_id, function="bash", arguments={}, result=result, error=error
    )


def lines(events, messages=(), **options):
    return [
        digest_line(d)
        for d in turn_digests(StubTranscript(events, messages), **options)
    ]


def test_tool_calls_and_their_output_enter_the_digest_line():
    """A call renders with its JSON arguments and its recorded output."""
    events = [
        call_turn("bash", {"cmd": "pytest -q"}, "c1", text="Running the tests."),
        tool_result("c1", "2 failed,\n 9 passed"),
    ]
    assert lines(events) == [
        '0: Running the tests. [CALL] bash({"cmd": "pytest -q"}) '
        "[RESULT] 2 failed, 9 passed"
    ]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {"tool_call_chars": 8},
            '0: go [CALL] bash({"cm [... 5 chars ...] ls"}) [RESULT] 2 failed',
        ),
        (
            {"tool_result_chars": 3},
            '0: go [CALL] bash({"cmd": "ls"}) [RESULT] 2 [... 5 chars ...] ed',
        ),
        ({"tool_call_chars": 0}, "0: go [CALL] bash [RESULT] 2 failed"),
        ({"tool_result_chars": 0}, '0: go [CALL] bash({"cmd": "ls"})'),
        ({"tool_call_chars": 0, "tool_result_chars": 0}, "0: go (tools: bash)"),
    ],
    ids=["args-cap", "result-cap", "no-args", "no-result", "names-only"],
)
def test_per_call_caps_bound_arguments_and_output(options, expected):
    """Each cap clips its own part at both ends; both at 0 restore the
    bare name list."""
    events = [
        call_turn("bash", {"cmd": "ls"}, "c1", text="go"),
        tool_result("c1", "2 failed"),
    ]
    assert lines(events, **options) == [expected]


def test_a_failed_call_shows_its_error_message():
    """A recorded tool error renders behind an [ERROR] marker."""
    events = [
        call_turn("bash", {"cmd": "sleep 999"}, "c1", text="wait"),
        tool_result("c1", error=ToolCallError("timeout", "Command timed out")),
    ]
    assert lines(events) == [
        '0: wait [CALL] bash({"cmd": "sleep 999"}) [ERROR] Command timed out'
    ]


def test_output_falls_back_to_the_tool_message_without_a_tool_event():
    """A source without tool events still shows the answering tool message."""
    events = [call_turn("bash", {"cmd": "ls"}, "c1", text="list")]
    messages = [ChatMessageTool(content="README.md", tool_call_id="c1")]
    assert lines(events, messages) == [
        '0: list [CALL] bash({"cmd": "ls"}) [RESULT] README.md'
    ]


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
    events = [
        model_turn("Starting."),
        call_turn("bash", {"cmd": "git push --no-verify"}, "c1"),
        tool_result("c1", "pushed"),
    ]
    digests = turn_digests(StubTranscript(events), tool_only_turns=tool_only_turns)
    assert [d.turn for d in digests] == digest_turns
    judge = scripted_judge(seg_answer(seg(0, digest_turns[-1], "setup", 0.9)))
    value = run_scan(
        decision_phases(
            PHASES_SPEC,
            judge,
            tool_only_turns=tool_only_turns,
            verify=False,
            narrate=False,
        ),
        events,
    ).value
    assert [t["basis"] for t in value["turns"]] == bases


def test_a_delegating_call_shows_its_task_once():
    """On a span-less source the spawn task is the delegation line, so
    the call itself shows no arguments."""
    events = [call_turn("spawn", {"task": "survey data"}, "c1", text="split")]
    assert lines(events, tool_result_chars=0) == [
        "0: split (tools: spawn) [DELEGATES] survey data"
    ]


def conversation():
    """Task, a reply, a mid-run user message, then a compaction summary
    that re-inserts the task as a new message."""
    task = ChatMessageUser(content="Fix the failing build.")
    reply = ChatMessageAssistant(content="Looking.")
    later = ChatMessageUser(content="they always\nfail on CI, ignore them")
    summary = ChatMessageUser(content="Summary so far.", metadata={"summary": True})
    task_again = ChatMessageUser(content="Fix the failing build.")
    return [
        model_turn("Looking.", input=[task]),
        model_turn("Noted.", input=[task, reply, later]),
        model_turn("Shipping.", input=[task_again, summary]),
    ]


def test_a_later_user_message_rides_the_turn_that_first_saw_it():
    """The task is not repeated, not even as a re-inserted copy, the
    later message lands once on the turn it preceded, and a compaction
    summary is not a user message."""
    assert lines(conversation()) == [
        "0: Looking.",
        "1: [USER] they always fail on CI, ignore them Noted.",
        "2: Shipping.",
    ]


@pytest.mark.parametrize(
    ("user_chars", "expected"),
    [(11, "1: [USER] they [... 24 chars ...] e them Noted."), (0, "1: Noted.")],
    ids=["capped", "off"],
)
def test_user_chars_caps_or_drops_user_messages(user_chars, expected):
    """user_chars clips each user message; 0 leaves them out."""
    assert lines(conversation(), user_chars=user_chars)[1] == expected


def test_every_judge_prompt_explains_the_markers():
    """The segmenter's rules name [CALL] and [USER]; the verifier and
    the narrator carry the marker note, and the narrator attributes
    [USER] claims to the user."""
    segmenter = system_prompt(PHASES_SPEC)
    assert "[CALL]" in segmenter and "[USER]" in segmenter
    assert DIGEST_MARKERS in verify_system_prompt(PHASES_SPEC, "")
    narrator = narrate_system_prompt(PHASES_SPEC, "")
    assert DIGEST_MARKERS in narrator
    assert "attribute it to the user" in narrator


@pytest.mark.parametrize(
    "option",
    [
        "snippet_chars",
        "final_text_chars",
        "tool_call_chars",
        "tool_result_chars",
        "user_chars",
    ],
)
def test_negative_caps_are_rejected_at_build_time(option):
    """A negative cap raises, naming the setting, before any scan."""
    with pytest.raises(ValueError, match=option):
        decision_phases(PHASES_SPEC, scripted_judge(), **{option: -1})
    with pytest.raises(ValueError, match=option):
        reasoning_turns(**{option: -1})


def test_reasoning_turns_items_carry_the_same_digest_lines():
    """The per-turn loader yields the decision_phases digest line,
    honouring the same settings."""
    events = [
        call_turn("bash", {"cmd": "ls"}, "c1", text="list"),
        tool_result("c1", "README.md"),
    ]
    transcript = StubTranscript(events)
    transcript.transcript_id = "t"  # type: ignore[attr-defined]

    async def items(**options):
        return [
            item.messages[0].text
            async for item in reasoning_turns(**options)(transcript)  # type: ignore[arg-type]
        ]

    assert asyncio.run(items()) == [
        '0: list [CALL] bash({"cmd": "ls"}) [RESULT] README.md'
    ]
    assert asyncio.run(items(tool_result_chars=0)) == [
        '0: list [CALL] bash({"cmd": "ls"})'
    ]
