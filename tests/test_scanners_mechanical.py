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


def test_token_timeline_numbers_orchestrator_turns_and_lane_turns():
    """Main-lane turns take 0,1; a sub-agent's turns take no turn but
    count within their own lane; usage and timestamps ride along."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead", usage=usage(100, 10)),
                *agent_span(
                    "C",
                    "eda",
                    inner=[model_turn("s0"), model_turn("s1")],
                    parent_id="R",
                ),
                model_turn("lead again", usage=usage(300, 30)),
            ],
        ),
    ]
    value = run_scan(token_timeline(), events).value
    rows = [(e["turn"], e["lane_turn"], e["agent_lane"]) for e in value["timeline"]]
    assert rows == [(0, 0, None), (None, 0, "eda"), (None, 1, "eda"), (1, 1, None)]
    assert value["timeline"][0]["input_tokens"] == 100
    assert value["timeline"][1]["input_tokens"] is None
    assert value["timeline"][3]["output_tokens"] == 30
    assert all(isinstance(e["timestamp"], str) for e in value["timeline"])


def test_span_record_anchors_spawn_and_end_by_orchestrator_turn():
    """Each sub-agent span records the orchestrator turn before its begin
    and before its end, its activity timestamps, and a recorded end."""
    events = [
        *agent_span(
            "R",
            "react",
            inner=[
                model_turn("lead"),
                *agent_span(
                    "C",
                    "eda",
                    inner=[model_turn("sub"), tool_event("t", span_id="C")],
                    parent_id="R",
                ),
                model_turn("wrap"),
            ],
        ),
    ]
    (span,) = run_scan(token_timeline(), events).value["spans"]
    assert (span["agent_span_id"], span["agent_lane"]) == ("C", "eda")
    assert (span["spawn_turn"], span["event_order_end_turn"]) == (0, 0)
    assert span["end_recorded"] is True
    assert span["first_at"] <= span["last_at"] <= span["end_at"]


def test_a_span_begun_before_any_orchestrator_turn_anchors_at_zero():
    """A sub-agent spawned before the first main turn still gets a turn."""
    events = [
        *agent_span("C", "eda", inner=[model_turn("sub")]),
        model_turn("lead"),
    ]
    (span,) = run_scan(token_timeline(), events).value["spans"]
    assert span["spawn_turn"] == 0


def test_utility_spans_are_neither_recorded_nor_counted(monkeypatch):
    """A span outside the shared sub-agent definition (a timeline utility
    span) gets no span record and no lane activity."""
    import transect.scanners.base as base

    events = [
        model_turn("lead"),
        *agent_span(
            "U", "titler", inner=[model_turn("t"), tool_event("x", span_id="U")]
        ),
        *agent_span("C", "eda", inner=[model_turn("s"), tool_event("y", span_id="C")]),
    ]
    real = base.subagent_span_begins

    def only_eda(transcript, main):
        begins, firsts = real(transcript, main)
        return [b for b in begins if b.id == "C"], firsts

    monkeypatch.setattr(base, "subagent_span_begins", only_eda)
    value = run_scan(token_timeline(), events).value
    assert [s["agent_span_id"] for s in value["spans"]] == ["C"]
    assert [r["agent_span_id"] for r in value["lane_activity"]] == ["C"]


def test_lane_activity_counts_tool_events_per_orchestrator_turn():
    """Tool-only sub-agents surface in lane_activity, anchored to the
    orchestrator turn that preceded the tool event."""
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
        "agent_span_id": None,
        "type": "summary",
        "source": "inspect",
        "tokens_before": 900,
        "tokens_after": 200,
        "role": None,
        "metadata": None,
        "compaction_prompt": None,
        "compaction_nudge": None,
    }


def test_a_sub_agent_compaction_is_kept_with_its_lane():
    """A compaction inside a sub-agent span is a flush row on the
    orchestrator axis that names its lane; the orchestrator's has none."""
    events = [
        model_turn("lead"),
        *agent_span(
            "C",
            "eda",
            inner=[
                model_turn("s0"),
                CompactionEvent(
                    type="summary",
                    source="inspect",
                    tokens_before=10,
                    tokens_after=5,
                    span_id="C",
                ),
                model_turn("s1"),
            ],
        ),
        CompactionEvent(
            type="summary", source="inspect", tokens_before=20, tokens_after=8
        ),
        model_turn("lead 2"),
    ]
    value = run_scan(context_flush(), events).value
    assert [(f["turn"], f["agent_span_id"]) for f in value["flushes"]] == [
        (1, "C"),
        (1, None),
    ]


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


def _summary(text="stripped recap"):
    return ChatMessageUser(content=f"[SUMMARY]\n{text}", metadata={"summary": True})


def _flush(type="summary"):
    return CompactionEvent(type=type, source="inspect")


def _intervention_shape(name):
    """(events, history, expected turn) for one placement shape. Each model
    turn's ``input`` is what that call saw, as Inspect records it; a shape
    that omits inputs exercises the footprint fallback instead."""
    task = ChatMessageUser(content="the task", source="input")
    note = ChatMessageUser(content="steer", source="operator")
    summary = _summary()
    a0 = model_turn("working", input=[task])
    if name == "before_summary":
        # the note was drained before the compaction: the summarizer saw it
        recap = model_turn("recap", input=[task, a0.output.message, note])
        blend = model_turn("blend", input=[task, summary])
        return (
            [a0, recap, _flush(), blend],
            [task, a0.output.message, note, summary, blend.output.message],
            1,
        )
    if name in ("after_summary", "after_summary_unrecorded"):
        recap = model_turn("recap", input=[task, a0.output.message])
        seen = [task, summary, note] if name == "after_summary" else []
        blend = model_turn("blend", input=seen)
        return (
            [a0, recap, _flush(), blend],
            [task, a0.output.message, summary, note, blend.output.message],
            2,
        )
    if name == "after_blend":
        recap = model_turn("recap", input=[task, a0.output.message])
        blend = model_turn("blend", input=[task, summary])
        final = model_turn("final", input=[task, summary, blend.output.message, note])
        return (
            [a0, recap, _flush(), blend, final],
            [
                task,
                a0.output.message,
                summary,
                blend.output.message,
                note,
                final.output.message,
            ],
            3,
        )
    if name in ("trim_then_summary", "native_then_summary"):
        # an earlier flush of another kind (or a native one) must not stand
        # in for the summarizer; no inputs, so the fallback decides
        earlier = _flush("trim" if name == "trim_then_summary" else "summary")
        recap = model_turn("recap")
        blend = model_turn("blend")
        return (
            [a0, earlier, recap, _flush(), blend],
            [task, a0.output.message, summary, note, blend.output.message],
            2,
        )
    if name == "two_pass_summary":
        # a summary that overshoots is summarized again: two summarizer
        # turns, one flush, one message
        first, second = model_turn("too long"), model_turn("recap")
        blend = model_turn("blend", input=[task, summary])
        return (
            [a0, first, second, _flush(), blend],
            [task, a0.output.message, summary, note, blend.output.message],
            3,
        )
    if name == "replayed_output":
        # a cached generate replays an earlier output id at a later turn
        recap = model_turn("recap")
        replay = model_turn("replayed", input=[task, summary])
        replay.output = a0.output
        return (
            [a0, recap, _flush(), replay],
            [task, a0.output.message, summary, note, replay.output.message],
            2,
        )
    if name == "as_tool_lane_flush":
        # an as_tool sub-agent compacts inside the lead's tool call: its
        # turns are on the axis, its messages never enter the lead's history
        sub1, subsum, sub2 = (
            model_turn("sub"),
            model_turn("sub recap"),
            model_turn("sub"),
        )
        leadsum = model_turn("recap")
        a2 = model_turn("blend", input=[task, summary, note])
        events = [a0, sub1, subsum, _flush(), sub2, leadsum, _flush(), a2]
        return events, [task, a0.output.message, summary, note, a2.output.message], 5
    if name == "forced_after_threshold":
        # overflow recovery replaces the history with the new summary, so
        # the earlier flush and its summary are gone from the history
        sum1 = model_turn("recap one")
        first_summary = _summary("recap one")
        a2 = model_turn("more", input=[task, first_summary])
        sum2 = model_turn("recap two")
        a4 = model_turn("final", input=[task, summary, note])
        events = [a0, sum1, _flush(), a2, sum2, _flush(), a4]
        return events, [task, summary, note, a4.output.message], 4
    raise ValueError(name)


@pytest.mark.parametrize(
    "shape",
    [
        "before_summary",
        "after_summary",
        "after_summary_unrecorded",
        "after_blend",
        "trim_then_summary",
        "native_then_summary",
        "two_pass_summary",
        "replayed_output",
        "as_tool_lane_flush",
        "forced_after_threshold",
    ],
)
def test_operator_messages_sit_on_the_event_turn_axis(shape):
    """A human message precedes the first model turn whose input saw it;
    a summarization call (a turn with no assistant message in the history)
    never shifts that, whatever other compactions the run recorded."""
    events, history, turn = _intervention_shape(shape)
    (intervention,) = run_scan(human_intervention(), events, messages=history).value[
        "interventions"
    ]
    assert intervention["turn"] == turn


def test_sub_agent_assistant_messages_do_not_advance_the_axis():
    """A handoff appends the sub-agent's messages to the thread; an
    operator note after them lands on the next orchestrator turn."""
    note = ChatMessageUser(content="steer", source="operator", id="note")
    task = ChatMessageUser(content="task", source="input", id="task")
    lead0 = model_turn("lead 0", input=[task])
    sub0 = model_turn("sub 0")
    lead1 = model_turn("lead 1", input=[task, note])
    events = [lead0, *agent_span("C", "eda", inner=[sub0]), lead1]
    messages = [
        task,
        ChatMessageAssistant(content="lead 0", id=lead0.output.message.id),
        ChatMessageAssistant(content="sub 0", id=sub0.output.message.id),
        note,
        ChatMessageAssistant(content="lead 1", id=lead1.output.message.id),
    ]
    value = run_scan(human_intervention(), events, messages).value
    assert [(i["turn"], i["channel"]) for i in value["interventions"]] == [
        (1, "operator")
    ]


def test_an_unseen_note_after_sub_agent_messages_takes_the_next_orchestrator_turn():
    """With no input recording it, the note's footprint counts
    orchestrator assistant messages only."""
    note = ChatMessageUser(content="steer", source="operator", id="note")
    task = ChatMessageUser(content="task", source="input", id="task")
    lead0 = model_turn("lead 0", input=[task])
    sub0 = model_turn("sub 0")
    lead1 = model_turn("lead 1", input=[task])
    events = [lead0, *agent_span("C", "eda", inner=[sub0]), lead1]
    messages = [
        task,
        ChatMessageAssistant(content="lead 0", id=lead0.output.message.id),
        ChatMessageAssistant(content="sub 0", id=sub0.output.message.id),
        note,
        ChatMessageAssistant(content="lead 1", id=lead1.output.message.id),
    ]
    value = run_scan(human_intervention(), events, messages).value
    assert [i["turn"] for i in value["interventions"]] == [1]


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
