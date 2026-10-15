"""What a digest line carries: tool calls with their arguments and, when
switched on, their recorded output; tool-call-only turns; spawning and
handoff calls; the settings every judge shares."""

import asyncio

import pytest
from helpers import (
    PHASES_SPEC,
    StubTranscript,
    agent_span,
    call_turn,
    model_turn,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
    tool_event,
)
from inspect_ai.model import ChatMessageTool, ContentText
from inspect_ai.tool import ToolCallError

from transect import DigestSettings, digest_turns
from transect.scanners.digests import digest_line, turn_digests
from transect.scanners.phases import decision_phases

RESULTS_ON = DigestSettings(tool_result_chars=200)
COMMAND = "cd repo && pytest -q tests"
OUTPUT = "2 failed, 40 passed in 1.2s"


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
        (
            [
                call_turn("bash", {"cmd": "ls"}, "c1", text="first"),
                call_turn("bash", {"cmd": "ls"}, "c1", text="again"),
                tool_event("c1", result="b.py"),
            ],
            [],
            [
                '0: first [CALL] bash({"cmd": "ls"})',
                '1: again [CALL] bash({"cmd": "ls"}) [RESULT] b.py',
            ],
        ),
        (
            [
                call_turn("bash", {"cmd": "ls"}, "c1", text="first"),
                call_turn("bash", {"cmd": "ls"}, "c1", text="again"),
            ],
            [
                ChatMessageTool(content="a.py", tool_call_id="c1"),
                ChatMessageTool(content="b.py", tool_call_id="c1"),
            ],
            [
                '0: first [CALL] bash({"cmd": "ls"})',
                '1: again [CALL] bash({"cmd": "ls"})',
            ],
        ),
    ],
    ids=[
        "error",
        "reused-id",
        "message-fallback",
        "content-blocks",
        "event-of-another-turn",
        "ambiguous-messages",
    ],
)
def test_output_is_the_tool_event_of_the_turn_else_the_one_answering_message(
    events, messages, expected
):
    """An error renders behind [ERROR]; a reused id resolves to the event
    inside its own turn and never to another turn's; without tool events
    the one answering message serves, and several answers serve none."""
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
                tool_event("c1", result='{"status": "accepted"}', agent_span_id="s1"),
                *agent_span(
                    "s1",
                    "sub",
                    inner=[model_turn("working")],
                    metadata={"task": "survey data"},
                ),
                call_turn("generate_image", {"prompt": "a cat"}, "c2", text="draw"),
                tool_event("c2", result="cat.png"),
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
    call's arguments or the span its tool event names, so the call shows
    neither arguments nor its acceptance output, while another call with
    a prompt argument keeps both; a handoff's transfer boilerplate is not
    output either."""
    rendered = lines(events, digest=RESULTS_ON)
    assert rendered[0] == expected
    if len(rendered) > 1:
        assert rendered[1] == (
            '1: draw [CALL] generate_image({"prompt": "a cat"}) [RESULT] cat.png'
        )


def test_settings_of_the_wrong_type_are_refused_at_build_time():
    """A loader or scanner built with a dict for digest fails before any scan."""
    with pytest.raises(TypeError, match="DigestSettings"):
        digest_turns(digest={"text_chars": 10})  # type: ignore[arg-type]


def test_digest_turns_items_carry_the_same_digest_lines():
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
            async for item in digest_turns(digest=digest)(transcript)  # type: ignore[arg-type]
        ]

    assert asyncio.run(items()) == ['0: list [CALL] bash({"cmd": "ls"})']
    assert asyncio.run(items(RESULTS_ON)) == [
        '0: list [CALL] bash({"cmd": "ls"}) [RESULT] README.md'
    ]
