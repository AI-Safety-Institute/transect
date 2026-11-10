"""Native Scout citations survive the unbatched cohort wrapper and reload."""

import json
from pathlib import Path

import pytest
from helpers import run_item, scripted_judge, vote_output
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Reference
from inspect_scout import Result, Transcript, scan_results_df, scanner

from transect import Layer, Spec, cohort_llm_scanner, load, turns_frame
from transect.api import _run
from transect.scanners import cohort

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
    expected = [Reference(type="message", cite="[M2]", id="evidence")]
    if regime != "solo":
        expected.append(Reference(type="message", cite="[M1]", id="task"))
    assert result.references == expected
    assert result.answer == "risky"


@pytest.mark.parametrize("k_rolls", [1, 3])
@pytest.mark.parametrize(
    "review", ["risky", "safe", "invalid", "out_of_vocab", "exception"]
)
def test_verifier_references_remain_in_native_result(k_rolls, review, monkeypatch):
    """Only successful verifier citations join the first-round references."""
    outputs = [
        vote_output("risky", confidence=0.4, explanation="Failure in [M2].")
    ] * k_rolls
    if review in LABELS:
        outputs.append(vote_output(review, confidence=0.9, explanation="Task in [M1]."))
    else:
        original = cohort.llm_scanner

        def scanner_with_failed_review(*args, **kwargs):
            if kwargs["question"] == QUESTION:
                return original(*args, **kwargs)

            async def failed_review(item):
                if review == "exception":
                    raise RuntimeError("verifier unavailable")
                return Result(
                    value={},
                    label="unsupported" if review == "out_of_vocab" else None,
                    references=[Reference(type="message", cite="[M1]", id="task")],
                )

            return failed_review

        monkeypatch.setattr(cohort, "llm_scanner", scanner_with_failed_review)
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
    expected = [("message", "evidence")]
    if review in LABELS:
        expected.append(("message", "task"))
    assert [(r.type, r.id) for r in result.references] == expected
    verifier = result.value["verifier"]
    assert "references" not in verifier
    assert verifier["status"] == ("ok" if review in LABELS else "error")
    assert result.answer == (review if review in LABELS else "risky")


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
    with pytest.raises(KeyError, match="message_references"):
        turns_frame(raw.drop(columns="message_references"))
    expected = json.loads(raw.iloc[0].message_references)
    assert len(expected) == 1
    assert expected[0]["type"] == "message"
    assert result.layer_frames["cited_behaviour"].iloc[0].message_references == expected
    reloaded = load(result.scan_location, extra_layers=[layer])
    assert (
        reloaded.layer_frames["cited_behaviour"].iloc[0].message_references == expected
    )
    assert (Path(result.scan_location) / "_scan.json").is_file()
