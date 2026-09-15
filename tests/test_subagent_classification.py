"""subagent_classification end to end: the agent_spans loader, the
activity digest, and the solo / k-roll / cohort / verifier regimes -
all judges scripted, no API calls."""

import asyncio
from typing import cast

import pytest
from helpers import (
    CLOSED,
    MODEL,
    StubTranscript,
    agent_span,
    model_turn,
    run_item,
    scripted_judge,
    tool_event,
    vote_output,
    voter,
)
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_scout import Transcript

import transect.scanners.cohort as cohort_mod
from transect.scanners.subagents import agent_spans, subagent_classification
from transect.spec import Spec

BOILER = (
    "[Subagent Context] You are running as a subagent (depth 1/1).\n"
    "\n[Subagent Task]\n\n"
)


def items(events):
    async def collect():
        stub = cast(Transcript, StubTranscript(events))
        return [item async for item in agent_spans()(stub)]

    return asyncio.run(collect())


def span_item(task="Survey related work", with_tool=False):
    inner = [tool_event("t1", function="web_search", span_id="A")] if with_tool else []
    events = [
        model_turn("orchestrator plans"),
        *agent_span("A", "lit", inner=inner, metadata={"task": task}),
    ]
    (item,) = items(events)
    return item


def test_loader_yields_one_item_per_tasked_sub_agent_span():
    """Each agent span with task text becomes one synthetic transcript
    carrying identity metadata; task-less spans yield nothing."""
    events = [
        model_turn("orchestrator plans"),
        *agent_span("A", "lit", metadata={"task": "Survey related work"}),
        *agent_span("B", "named-only"),
    ]
    (item,) = items(events)
    assert item.transcript_id == "A"
    assert item.metadata == {
        "agent_span_id": "A",
        "agent_lane": "lit",
        "span_task_source": "spawn_prompt",
        "span_task": "Survey related work",
        "span_task_truncated": False,
    }
    assert "task: Survey related work" in item.messages[0].text


def test_loader_excludes_the_main_lane_span():
    """A solo agent wrapped in its own span is not its own sub-agent."""
    events = agent_span(
        "MAIN",
        "react",
        [
            model_turn(
                "working",
                span_id="MAIN",
                input=[ChatMessageUser(content="Solve the task please")],
            )
        ],
    )
    assert items(events) == []


def test_loader_falls_back_to_the_handoff_input_for_the_task():
    """A span without a spawn prompt takes its task from the first
    model call's input (the handoff carrier)."""
    events = [
        model_turn("orchestrator"),
        SpanBeginEvent(id="B", parent_id=None, type="agent", name="checker"),
        SpanBeginEvent(id="CH", parent_id="B", type="generate", name="gen"),
        model_turn(
            "thinking",
            span_id="CH",
            input=[
                ChatMessageUser(content=""),
                ChatMessageUser(content="Verify the results please"),
            ],
        ),
        SpanEndEvent(id="CH"),
        SpanEndEvent(id="B"),
    ]
    (item,) = items(events)
    assert item.metadata["span_task_source"] == "handoff_input"
    assert "Verify the results please" in item.messages[0].text


def test_loader_strips_spawn_scaffold_and_bounds_task_text():
    """OpenClaw spawn boilerplate is stripped and the task text is
    bounded, so the judge sees the actual instruction."""
    item = span_item(task=BOILER + "Run the ablation sweep." * 100)
    text = item.messages[0].text
    assert "task: Run the ablation sweep." in text
    assert "[Subagent Context]" not in text
    assert len(text) < 900


def test_recorded_activity_rides_a_second_message():
    """An active span's digest is its own message (task text stays
    byte-stable in message one); silent spans get no second message."""
    active = span_item(with_tool=True)
    assert len(active.messages) == 2
    assert "activity: 1 tool call (web_search x1)" in active.messages[1].text
    silent = span_item()
    assert len(silent.messages) == 1


def test_digest_elides_the_middle_of_long_runs():
    """Turn excerpts are per-turn bounded and middle-elided, keeping the
    first and last turns."""
    turns = [model_turn(f"turn {i}: " + "x" * 300, span_id="A") for i in range(12)]
    events = [
        model_turn("root"),
        *agent_span("A", "lit", inner=turns, metadata={"task": "Survey"}),
    ]
    (item,) = items(events)
    digest = item.messages[1].text
    assert "activity: 12 turns" in digest
    assert "turn 0:" in digest and "turn 11:" in digest
    assert "turns elided" in digest
    assert "turn 5:" not in digest


def capture_judge_input(monkeypatch, factory):
    seen = {}

    def capturing(**kwargs):
        seen["question"] = kwargs["question"]

        async def execute(item):
            seen["content"] = "\n\n".join(m.text for m in item.messages)
            raise cohort_mod.RefusalError("captured")

        return execute

    monkeypatch.setattr(cohort_mod, "llm_scanner", capturing)
    try:
        run_item(factory(), span_item(with_tool=True))
    except cohort_mod.RefusalError:
        pass
    return seen


def test_activity_digest_is_stripped_by_default(monkeypatch):
    """Off by default: the judge gets the spawn-task question and never
    sees the digest."""
    seen = capture_judge_input(
        monkeypatch, lambda: subagent_classification(CLOSED, MODEL)
    )
    assert "spawned to do" in seen["question"]
    assert "activity:" not in seen["content"]


def test_activity_true_keeps_digest_and_switches_the_question(monkeypatch):
    """The experimental knob shows the digest and asks the judge to
    weigh what the sub-agent actually did."""
    seen = capture_judge_input(
        monkeypatch, lambda: subagent_classification(CLOSED, MODEL, activity=True)
    )
    assert "digest of its actual activity" in seen["question"]
    assert "activity: 1 tool call (web_search x1)" in seen["content"]


def test_closed_vocabulary_is_required():
    with pytest.raises(ValueError, match="subagent_labels"):
        subagent_classification(Spec(), MODEL)


def test_solo_judge_stamps_the_answer_contract():
    """A solo judgement promotes the label and records confidence,
    label_source, and the stable verifier block."""
    judge = voter(MODEL, "literature_survey", 0.85, "surveys prior work")
    result = run_item(subagent_classification(CLOSED, judge), span_item())
    assert result.label == "literature_survey"
    value = result.value
    assert value["confidence"] == 0.85
    assert value["label_source"] == "single_judge"
    assert value["verifier"] == {"ran": False, "model": MODEL}
    assert [entry["label"] for entry in value["label_vocab"]] == [
        "literature_survey",
        "experiment_run",
        "none_of_the_above",
    ]


def test_judges_can_answer_none_of_the_above():
    judge = voter(MODEL, "none_of_the_above", 0.6, "fits neither")
    result = run_item(
        subagent_classification(CLOSED, judge), span_item(task="Water the plants")
    )
    assert result.label == "none_of_the_above"


def test_cohort_majority_vote_decides_the_label():
    """Three models vote; the majority label wins and the decision is
    stamped majority_vote with each member's ballot kept."""
    members = [
        voter(MODEL, "literature_survey", 0.9),
        voter("mockllm/model2", "literature_survey", 0.7),
        voter("mockllm/model3", "experiment_run", 0.95),
    ]
    result = run_item(subagent_classification(CLOSED, members), span_item())
    assert result.answer == "literature_survey"
    assert result.value["label_source"] == "majority_vote"
    assert len(result.value["cohort"]["members"]) == 3


def test_k_rolls_vote_like_a_cohort_of_one_model():
    """Three rolls of one model majority-vote; a dissenting roll lowers
    agreement below 1."""
    judge = scripted_judge(
        vote_output("literature_survey", 0.9),
        vote_output("literature_survey", 0.8),
        vote_output("experiment_run", 0.7),
    )
    result = run_item(
        subagent_classification(CLOSED, judge, k_rolls=3, verify=False), span_item()
    )
    assert result.answer == "literature_survey"
    assert result.value["label_source"] == "majority_vote"
    assert 0 < result.value["cohort"]["agreement"] < 1


def test_a_refusing_member_is_recorded_and_outvoted():
    """One member refusing does not sink the vote; its status lands as
    refusal in the ballots."""
    refusal = ModelOutput.from_content(
        "mockllm/model2", "", stop_reason="content_filter"
    )
    refuser = get_model("mockllm/model2", custom_outputs=[refusal] * 5, memoize=False)
    members = [
        voter(MODEL, "literature_survey", 0.9),
        refuser,
        voter("mockllm/model3", "literature_survey", 0.8),
    ]
    result = run_item(subagent_classification(CLOSED, members), span_item())
    assert result.answer == "literature_survey"
    statuses = [m["status"] for m in result.value["cohort"]["members"]]
    assert statuses.count("refusal") == 1


def test_a_failing_member_call_records_its_reason():
    """A member judge call that raises (post-retry) lands as an error
    ballot carrying the exception type and text."""
    boom = get_model(
        "mockllm/model2",
        custom_outputs=[RuntimeError("upstream 500: bad gateway")] * 5,
        memoize=False,
    )
    members = [
        voter(MODEL, "literature_survey", 0.9),
        boom,
        voter("mockllm/model3", "literature_survey", 0.8),
    ]
    result = run_item(subagent_classification(CLOSED, members), span_item())
    assert result.answer == "literature_survey"
    failed = [m for m in result.value["cohort"]["members"] if m["status"] == "error"]
    assert len(failed) == 1
    # the contract is type name + text captured; which exception type
    # mockllm wraps the payload in is inspect-ai's own business
    error = failed[0]["error"]
    assert error and not error.startswith(":") and "bad gateway" in error
    ok = [m for m in result.value["cohort"]["members"] if m["status"] == "ok"]
    assert all(m["error"] is None for m in ok)


def test_low_confidence_triggers_the_verifier_and_the_gate_holds():
    """A doubtful solo judgement is re-reviewed; a confident verifier
    overturns the label, and the original is preserved."""
    judge = scripted_judge(
        vote_output("literature_survey", 0.4, "unsure"),
        vote_output("experiment_run", 0.9, "clearly experiments"),
    )
    result = run_item(subagent_classification(CLOSED, judge, verify=True), span_item())
    verifier = result.value["verifier"]
    assert verifier["ran"] and verifier["trigger"] == "low_confidence"
    assert verifier["overturned"] is True
    assert result.label == "experiment_run"
    assert verifier["original_label"] == "literature_survey"
    assert result.value["label_source"] == "verifier"


def test_a_weak_verifier_verdict_never_overturns():
    """Below the confidence gate the verifier's differing label is
    recorded but the original stands."""
    judge = scripted_judge(
        vote_output("literature_survey", 0.4, "unsure"),
        vote_output("experiment_run", 0.5, "also unsure"),
    )
    result = run_item(subagent_classification(CLOSED, judge, verify=True), span_item())
    assert result.label == "literature_survey"
    assert result.value["verifier"]["overturned"] is False


def test_a_confident_judgement_skips_the_verifier():
    judge = voter(MODEL, "literature_survey", 0.95)
    result = run_item(subagent_classification(CLOSED, judge, verify=True), span_item())
    assert result.value["verifier"] == {"ran": False, "model": MODEL}


def test_verify_sample_spot_checks_a_confident_judgement():
    """With verify_sample=1.0 even a confident span is verifier-reviewed
    under the random_sample trigger; a confirming verdict changes nothing."""
    judge = scripted_judge(
        vote_output("literature_survey", 0.95, "confident"),
        vote_output("literature_survey", 0.9, "confirmed"),
    )
    result = run_item(
        subagent_classification(CLOSED, judge, verify=True, verify_sample=1.0),
        span_item(),
    )
    verifier = result.value["verifier"]
    assert verifier["ran"] and verifier["trigger"] == "random_sample"
    assert verifier["overturned"] is False
    assert result.label == "literature_survey"


@pytest.mark.parametrize(("span_id", "reviewed"), [("O", True), ("F", False)])
def test_default_verify_sample_draws_five_percent_per_span(span_id, reviewed):
    """verify_sample=None uses the seeded 5% per-span default: span id
    O draws 0.003 and is reviewed, F draws 0.062 and is not."""
    outputs = [vote_output("literature_survey", 0.95, "confident")]
    if reviewed:
        outputs.append(vote_output("literature_survey", 0.9, "confirmed"))
    events = [
        model_turn("orchestrator plans"),
        *agent_span(span_id, "lit", metadata={"task": "Survey related work"}),
    ]
    (item,) = items(events)
    result = run_item(
        subagent_classification(CLOSED, scripted_judge(*outputs), verify=True), item
    )
    verifier = result.value["verifier"]
    assert verifier["ran"] is reviewed
    if reviewed:
        assert verifier["trigger"] == "random_sample"


def test_verify_sample_outside_the_unit_interval_is_rejected():
    """A verify_sample outside [0, 1] raises at scanner build time."""
    with pytest.raises(ValueError, match="verify_sample"):
        subagent_classification(
            CLOSED, scripted_judge(), verify=True, verify_sample=1.5
        )


def test_verify_with_a_cohort_is_rejected():
    with pytest.raises(ValueError, match="verifier XOR cohort"):
        subagent_classification(CLOSED, [MODEL, "mockllm/model2"], verify=True)
