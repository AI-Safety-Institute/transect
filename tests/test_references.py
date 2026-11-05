"""Native Scout citations survive the unbatched cohort wrapper and reload."""

import json
from pathlib import Path

import pytest
from helpers import run_item, scripted_judge, vote_output
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, get_model
from inspect_ai.scorer import Reference
from inspect_scout import Transcript, llm_scanner, scan_results_df, scanner

from transect import Layer, Spec, cohort_llm_scanner, load, turns_frame
from transect.api import _run
from transect.scanners.cohort import _merge_references, _structured_answer

LABELS = ["risky", "safe"]
QUESTION = "Classify the behaviour and cite the relevant [Mn] message IDs."


def cited_item():
    return Transcript(
        transcript_id="reference-repro",
        messages=[
            ChatMessageUser(id="task", content="Inspect the check."),
            ChatMessageAssistant(id="evidence", content="The test failed."),
        ],
    )


@pytest.mark.parametrize("regime", ["solo", "k_roll", "cohort"])
def test_wrapper_preserves_native_references(regime):
    """References retain source identity and deduplicate across all members."""
    output = vote_output("risky", explanation="Failure in [M2].")
    native = run_item(
        llm_scanner(
            question=QUESTION,
            answer=_structured_answer(LABELS),
            model=scripted_judge(output),
        ),
        cited_item(),
    )
    assert [r.id for r in native.references] == ["evidence"]
    kwargs = {"verify": False}
    if regime == "solo":
        models = scripted_judge(output)
    elif regime == "k_roll":
        models = scripted_judge(
            output, output, vote_output("safe", explanation="Task in [M1].")
        )
        kwargs["k_rolls"] = 3
    else:
        models = [
            scripted_judge(output),
            scripted_judge(output, model="mockllm/model2"),
            scripted_judge(
                vote_output("safe", explanation="Task in [M1]."), model="mockllm/model3"
            ),
        ]
    result = run_item(
        cohort_llm_scanner(question=QUESTION, answer=LABELS, models=models, **kwargs),
        cited_item(),
    )
    expected = list(native.references)
    if regime != "solo":
        expected.append(Reference(type="message", cite="[M1]", id="task"))
    assert result.references == expected
    assert result.answer == "risky"


@pytest.mark.parametrize("k_rolls", [1, 3])
@pytest.mark.parametrize("verifier_label", ["risky", "safe"])
def test_verifier_references_remain_in_native_result(k_rolls, verifier_label):
    """Verifier citations survive without expanding the stored review value."""
    outputs = [
        vote_output("risky", confidence=0.4, explanation="Failure in [M2].")
    ] * k_rolls
    outputs.append(
        vote_output(verifier_label, confidence=0.9, explanation="Task in [M1].")
    )
    result = run_item(
        cohort_llm_scanner(
            question=QUESTION,
            answer=LABELS,
            models=scripted_judge(*outputs),
            k_rolls=k_rolls,
            verify=True,
            verify_sample=0,
        ),
        cited_item(),
    )
    assert [(r.type, r.id) for r in result.references] == [
        ("message", "evidence"),
        ("message", "task"),
    ]
    assert "references" not in result.value["verifier"]
    assert result.answer == verifier_label


@scanner(messages="all")
def cited_behaviour():
    return cohort_llm_scanner(
        question=QUESTION,
        answer=LABELS,
        models=scripted_judge(vote_output("risky", explanation="See [M1].")),
        verify=False,
    )


def test_references_survive_the_store_and_layer_projection(demo_log, tmp_path):
    """Reload exposes Scout's stored message references in the layer item frame."""
    layer = Layer(name="cited_behaviour", scanner=cited_behaviour, frame=turns_frame)
    result = _run(
        str(demo_log), Spec(), scans_dir=str(tmp_path / "scans"), extra_layers=[layer]
    )
    raw = scan_results_df(result.scan_location).scanners["cited_behaviour"]
    expected = json.loads(raw.iloc[0].message_references)
    assert len(expected) == 1
    assert expected[0]["type"] == "message"
    assert result.layer_frames["cited_behaviour"].iloc[0].message_references == expected
    reloaded = load(result.scan_location, extra_layers=[layer])
    assert (
        reloaded.layer_frames["cited_behaviour"].iloc[0].message_references == expected
    )
    assert (Path(result.scan_location) / "_scan.json").is_file()


@pytest.mark.parametrize("explanation", ["No citation.", "Unknown [M999]."])
def test_absent_or_unknown_citations_do_not_create_references(explanation):
    """Scout's missing-reference result survives without invented source IDs."""
    result = run_item(
        cohort_llm_scanner(
            question=QUESTION,
            answer=LABELS,
            models=scripted_judge(vote_output("risky", explanation=explanation)),
            verify=False,
        ),
        cited_item(),
    )
    assert result.references == []
    assert result.answer == "risky"


def test_failed_verifier_keeps_first_round_references():
    """A failed review adds no reference and preserves the judge's citations."""
    judge = scripted_judge(
        vote_output("risky", confidence=0.4, explanation="Failure in [M2].")
    )

    def fail(input, tools, tool_choice, config):
        raise RuntimeError("verifier unavailable")

    result = run_item(
        cohort_llm_scanner(
            question=QUESTION,
            answer=LABELS,
            models=judge,
            verifier_model=get_model(
                "mockllm/reviewer", custom_outputs=fail, memoize=False
            ),
            verify=True,
            verify_sample=0,
        ),
        cited_item(),
    )
    assert [r.id for r in result.references] == ["evidence"]
    assert result.value["verifier"]["status"] == "error"
    assert result.answer == "risky"


def test_reference_identity_includes_type():
    """A message and an event with the same ID remain distinct citations."""
    first = Reference(type="message", cite="[M1]", id="shared")
    duplicate = Reference(type="message", cite="[M2]", id="shared")
    event = Reference(type="event", id="shared")
    assert _merge_references([[first], [duplicate, event]]) == [first, event]
