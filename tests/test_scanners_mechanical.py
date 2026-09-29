"""The zero-LLM scanners: token_timeline, context_flush,
human_intervention - judged by what they extract from event streams."""

import pytest
from helpers import agent_span, model_turn, run_scan, tool_event
from inspect_ai.event import ApprovalEvent, CompactionEvent, InputEvent
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelUsage
from inspect_ai.tool import ToolCall

from transect.scanners.base import context_flush, human_intervention, token_timeline


def usage(input_tokens=100, output_tokens=10):
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def test_token_timeline_emits_one_entry_per_model_turn_with_usage():
    """Every model turn lands as one timeline entry carrying the
    provider-reported usage fields verbatim; None where unreported."""
    events = [
        model_turn("a", usage=usage(100, 10)),
        model_turn("b"),
        model_turn("c", usage=usage(300, 30)),
    ]
    value = run_scan(token_timeline(), events).value
    timeline = value["timeline"]
    assert [entry["turn"] for entry in timeline] == [0, 1, 2]
    assert timeline[0]["input_tokens"] == 100
    assert timeline[1]["input_tokens"] is None
    assert timeline[2]["output_tokens"] == 30


def test_token_timeline_attributes_turns_to_their_sub_agent_lane():
    """A sub-agent's own model turns carry its span; main-lane turns
    carry no lane, even when the lead agent has an agent span (handoff
    .eval shape)."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead"),
                *agent_span("C", "eda", inner=[model_turn("sub")], parent_id="R"),
                model_turn("lead again"),
            ],
        ),
    ]
    value = run_scan(token_timeline(), events).value
    lanes = [entry["agent_lane"] for entry in value["timeline"]]
    assert lanes == [None, "eda", None]


def test_lane_activity_counts_tool_events_per_sub_agent_turn():
    """Tool-only sub-agents (no model turns of their own) surface in
    lane_activity, anchored to the initiating model turn."""
    events = [
        model_turn("orchestrator"),
        *agent_span("A", "worker", inner=[tool_event("t1", span_id="A")]),
    ]
    value = run_scan(token_timeline(), events).value
    (row,) = value["lane_activity"]
    assert (row["agent_lane"], row["turn"], row["tool_calls"]) == ("worker", 0, 1)


def test_lane_activity_excludes_the_main_lane_and_folded_spawn_calls():
    """Neither the lead agent's own tools nor the orchestrator's folded
    spawn call (agent_span_id-tagged) become sub-agent activity."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead"),
                tool_event("t1", span_id="R"),
                *agent_span(
                    "C",
                    "eda",
                    inner=[
                        tool_event("t2", span_id="C"),
                        tool_event("t3", span_id="C", agent_span_id="C"),
                    ],
                    parent_id="R",
                ),
            ],
        ),
    ]
    value = run_scan(token_timeline(), events).value
    assert [(r["agent_lane"], r["tool_calls"]) for r in value["lane_activity"]] == [
        ("eda", 1)
    ]


def test_span_ends_record_sub_agent_completions_only():
    """span_ends anchors each sub-agent span's close to a turn; the main
    lane's own span end is not a completion marker."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead"),
                *agent_span("C", "eda", inner=[model_turn("sub")], parent_id="R"),
                model_turn("wrap"),
            ],
        ),
    ]
    value = run_scan(token_timeline(), events).value
    assert [(e["agent_span_id"], e["turn"]) for e in value["span_ends"]] == [("C", 1)]


def test_context_flush_records_compaction_events_at_their_turn():
    """A compaction event lands with the count of model turns before it
    and the event's own type/source/token fields."""
    events = [
        model_turn("a"),
        model_turn("b"),
        CompactionEvent(
            type="summary", source="inspect", tokens_before=900, tokens_after=200
        ),
        model_turn("c"),
    ]
    value = run_scan(context_flush(), events).value
    (flush,) = value["flushes"]
    assert flush == {
        "turn": 2,
        "type": "summary",
        "source": "inspect",
        "tokens_before": 900,
        "tokens_after": 200,
        "role": None,
        "metadata": None,
        "compaction_prompt": None,
        "compaction_nudge": None,
    }


def test_no_compaction_events_means_no_flushes():
    value = run_scan(context_flush(), [model_turn("a")]).value
    assert value["flushes"] == []


@pytest.mark.parametrize(
    ("source", "channel"),
    [("operator", "operator"), ("input", "input")],
)
def test_human_messages_become_interventions_by_source(source, channel):
    """Operator steering and post-task console input register with
    their channel and the model turn they precede (assistant messages are
    counted when no event stream records them)."""
    messages = [
        ChatMessageUser(content="the task", source="input"),
        ChatMessageAssistant(content="working"),
        ChatMessageUser(content="stop doing that", source=source),
    ]
    value = run_scan(human_intervention(), [], messages=messages).value
    (intervention,) = value["interventions"]
    assert intervention["channel"] == channel
    assert intervention["content"] == "stop doing that"
    assert intervention["turn"] == 1


@pytest.mark.parametrize(
    ("note_at", "turn"),
    [("before_summary", 1), ("after_summary", 2), ("after_blend", 3)],
)
def test_operator_messages_sit_on_the_event_turn_axis(note_at, turn):
    """A summarization call is a model turn with no assistant message in the
    history; its summary message stands for it, so an operator note before,
    right after, or later than that footprint precedes the turn the
    timeline says it does."""
    turns = [model_turn(text) for text in ("working", "recap", "blend", "final")]
    working, _recap, blend, final = turns
    note = ChatMessageUser(content="steer", source="operator")
    summary = ChatMessageUser(content="[SUMMARY] recap", metadata={"summary": True})
    flush = CompactionEvent(type="summary", source="inspect")
    events = [working, turns[1], flush, blend, final]
    history = {
        "before_summary": [note, summary, blend.output.message],
        "after_summary": [summary, note, blend.output.message],
        "after_blend": [summary, blend.output.message, note],
    }[note_at]
    messages = [
        ChatMessageUser(content="the task", source="input"),
        working.output.message,
        *history,
        final.output.message,
    ]
    value = run_scan(human_intervention(), events, messages=messages).value
    (intervention,) = value["interventions"]
    assert intervention["turn"] == turn


def test_the_first_input_message_is_the_task_not_an_intervention():
    messages = [ChatMessageUser(content="the task prompt", source="input")]
    value = run_scan(human_intervention(), [], messages=messages).value
    assert value["interventions"] == []


def test_an_operator_delivered_task_brief_is_not_an_intervention():
    """ACP/OpenClaw sources deliver the task prompt as an operator
    message: the transcript's first user message never counts,
    whichever channel carried it - later operator messages do."""
    messages = [
        ChatMessageUser(content="the task brief", source="operator"),
        ChatMessageAssistant(content="working"),
        ChatMessageUser(content="also check the logs", source="operator"),
    ]
    value = run_scan(human_intervention(), [], messages=messages).value
    (intervention,) = value["interventions"]
    assert intervention["content"] == "also check the logs"
    assert intervention["turn"] == 1


def test_a_scaffold_first_message_does_not_consume_the_task_slot():
    """The task-prompt slot belongs to the first human-channel message:
    a scaffold-built opener neither counts as an intervention nor
    absorbs the slot, so the human's first message after it is still
    the task prompt - and the second is real steering."""
    messages = [
        ChatMessageUser(content="scaffold-built task framing"),
        ChatMessageAssistant(content="working"),
        ChatMessageUser(content="the actual task", source="input"),
        ChatMessageAssistant(content="working more"),
        ChatMessageUser(content="stop doing that", source="input"),
    ]
    value = run_scan(human_intervention(), [], messages=messages).value
    (intervention,) = value["interventions"]
    assert intervention["content"] == "stop doing that"


def test_scaffold_user_messages_are_not_interventions():
    """Source-less user messages (react continue-prompts, handoff
    scaffolding) never count, whatever their text."""
    messages = [
        ChatMessageUser(content="the task", source="input"),
        ChatMessageUser(content="Please proceed to the next step."),
    ]
    value = run_scan(human_intervention(), [], messages=messages).value
    assert value["interventions"] == []


def test_input_events_are_interventions_on_their_own_channel():
    """A console InputEvent (no question of its own) lands as
    channel=input_event at the current turn, agent-initiated, with the
    recording as its content and no prompt or outcome."""
    events = [model_turn("asking"), InputEvent(input="yes, go ahead", input_ansi="")]
    value = run_scan(human_intervention(), events).value
    (intervention,) = value["interventions"]
    assert intervention["channel"] == "input_event"
    assert intervention["initiator"] == "agent"
    assert intervention["content"] == "yes, go ahead"
    assert intervention["prompt"] is None and intervention["outcome"] is None
    assert intervention["turn"] == 1


@pytest.mark.parametrize(
    ("event", "prompt", "content", "outcome"),
    [
        (  # ask_user as inspect records it: question prepended to the text
            InputEvent(
                input="Submit now?\n  confirm: yes",
                input_ansi="",
                message="Submit now?",
                outcome="accepted",
                content={"confirm": "yes"},
            ),
            "Submit now?",
            "confirm: yes",
            "accepted",
        ),
        (  # declined: inspect's marker line is not an answer, outcome carries it
            InputEvent(
                input="Submit now?\n[declined]",
                input_ansi="",
                message="Submit now?",
                outcome="declined",
            ),
            "Submit now?",
            "",
            "declined",
        ),
        (  # an older log: bare answer, question, no outcome recorded
            InputEvent(input="y", input_ansi="", message="Submit now?"),
            "Submit now?",
            "y",
            None,
        ),
    ],
)
def test_ask_user_events_separate_question_answer_and_outcome(
    event, prompt, content, outcome
):
    value = run_scan(human_intervention(), [model_turn("asking"), event]).value
    (intervention,) = value["interventions"]
    assert (intervention["prompt"], intervention["content"]) == (prompt, content)
    assert intervention["outcome"] == outcome


@pytest.mark.parametrize(
    ("approver", "n"),
    [("human", 1), ("auto", 0)],
)
def test_human_approvals_are_interventions_and_automatic_ones_are_not(approver, n):
    """A tool call decided by the human approver is an agent-initiated
    intervention carrying the call, the decision and the explanation."""
    call = ToolCall(id="c1", function="bash", arguments={"cmd": "rm -rf build"})
    events = [
        model_turn("running"),
        ApprovalEvent(
            message="cleaning up",
            call=call,
            approver=approver,
            decision="reject",
            explanation="too destructive",
        ),
    ]
    value = run_scan(human_intervention(), events).value
    assert len(value["interventions"]) == n
    if n:
        (intervention,) = value["interventions"]
        assert intervention["channel"] == "approval"
        assert intervention["prompt"] == 'bash({"cmd": "rm -rf build"})'
        assert intervention["outcome"] == "reject"
        assert intervention["content"] == "too destructive"
        assert intervention["turn"] == 1
