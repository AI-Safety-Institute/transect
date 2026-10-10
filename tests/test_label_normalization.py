"""Closed labels keep the same keys in requests, stored rubrics and frames."""

import json
from pathlib import Path

import pandas as pd
import pytest
from helpers import run_item, scripted_judge, vote_output
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_scout import Transcript, scanner

from transect import Layer, load, reasoning_turns, turns_frame
from transect.api import _run
from transect.frames import label_definitions_df
from transect.scanners.cohort import batch_item_content, cohort_llm_scanner
from transect.scanners.subagents import subagent_classification
from transect.spec import Spec


def enum_labels(schema):
    """Find the closed label enum in a scalar or batch answer tool."""
    if isinstance(schema, dict):
        if schema.get("enum"):
            return schema["enum"]
        for value in schema.values():
            if found := enum_labels(value):
                return found
    return None


def following_judge(seen, *, review=False):
    """A scripted judge that answers within the actual tool's declared enum."""

    def answer(input, tools, tool_choice, config):
        schema = tools[0].parameters.model_dump()
        labels = enum_labels(schema)
        text = "\n".join(message.text for message in input)
        seen.append((labels, text))
        verifying = "second-round reviewer" in text
        fields = {
            "label": labels[1 if verifying else 0],
            "confidence": 0.3 if review and not verifying else 0.9,
            "explanation": "scripted",
        }
        value = (
            {"items": [{"item": n, **fields} for n in (1, 2)], "explanation": "batch"}
            if "items" in schema["properties"]
            else fields
        )
        return ModelOutput.for_tool_call("mockllm/model", "answer", value)

    return get_model("mockllm/model", custom_outputs=answer, memoize=False)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rolls", [1, 2])
def test_closed_label_keys_match_the_recorded_rubric(batch, rolls):
    """Scalar and batch votes use the same normalized keys as their rubric."""
    seen = []
    result = run_item(
        cohort_llm_scanner(
            "Classify the work",
            ["Research", "Result Review"],
            models=following_judge(seen),
            vocabulary={"Research": "Investigate", "Result Review": "Review"},
            verify=False,
            k_rolls=rolls,
            batch=batch,
        ),
        Transcript(
            transcript_id="units",
            messages=[ChatMessageUser(content=batch_item_content(["one", "two"]))],
            metadata={"items": [{"turn": 0}, {"turn": 1}]},
        ),
    )
    assert all(labels == ["research", "result_review"] for labels, _ in seen)
    assert [entry["label"] for entry in result.value["label_vocab"]] == [
        "research",
        "result_review",
    ]
    assert [entry["description"] for entry in result.value["label_vocab"]] == [
        "Investigate",
        "Review",
    ]
    units = result.value["items"] if batch else [{"label": result.label}]
    assert all(unit["label"] == "research" for unit in units)


@pytest.mark.parametrize("batch", [False, True])
def test_verifier_uses_the_same_closed_label_keys(batch):
    """The verifier can overturn using the same normalized answer vocabulary."""
    seen = []
    result = run_item(
        cohort_llm_scanner(
            "Classify the work",
            ["Research", "Result Review"],
            models=following_judge(seen, review=True),
            verify=True,
            verify_sample=0.0,
            batch=batch,
        ),
        Transcript(
            transcript_id="units",
            messages=[ChatMessageUser(content=batch_item_content(["one", "two"]))],
            metadata={"items": [{"turn": 0}, {"turn": 1}]},
        ),
    )
    assert all(labels == ["research", "result_review"] for labels, _ in seen)
    units = result.value["items"] if batch else [result.value]
    for unit in units:
        assert unit["verifier"]["original_label"] == "research"
        assert unit["verifier"]["verifier_label"] == "result_review"
        assert unit["verifier"]["overturned"]


@pytest.mark.parametrize(
    "labels", [["Research", "research"], ["a b", "a_b"], ["x", "x"], [" ", "x"]]
)
def test_invalid_closed_labels_fail_before_constructing_a_judge(labels, monkeypatch):
    """Blank or colliding labels cannot reach a model or become merged categories."""
    monkeypatch.setattr(
        "transect.scanners.cohort.llm_scanner",
        lambda **kwargs: pytest.fail("must reject labels before constructing a judge"),
    )
    with pytest.raises(ValueError, match="label"):
        cohort_llm_scanner("Classify", labels)


def test_colliding_rubric_entries_fail_before_constructing_a_judge(monkeypatch):
    """Normalization collisions in the recorded rubric also fail before judging."""
    monkeypatch.setattr(
        "transect.scanners.cohort.llm_scanner",
        lambda **kwargs: pytest.fail("must reject rubric before constructing a judge"),
    )
    with pytest.raises(ValueError, match="collid"):
        cohort_llm_scanner("Classify", ["a_b"], vocabulary={"A B": "one", "a_b": "two"})


@pytest.mark.parametrize("label", ["research", "Research", "Literature Survey"])
@pytest.mark.parametrize("rolls", [1, 2])
def test_subagent_labels_and_definitions_join_after_reload(
    label, rolls, demo_log, tmp_path
):
    """Accepted labels survive storage and join their authored definition."""
    spec = Spec.model_validate(
        {"subagent_labels": [{"label": label, "description": "Authored definition"}]}
    )
    before = spec.model_dump()
    seen = []
    result = _run(
        str(demo_log),
        spec,
        judge_models=following_judge(seen),
        verify=False,
        k_rolls=rolls,
        scans_dir=str(tmp_path / "scans"),
    )
    expected = label.lower().replace(" ", "_")
    restored = load(result.scan_location)
    for frames in [result, restored]:
        assert len(frames.subagents) == 3
        assert set(frames.subagents.label) == {expected}
        definitions = frames.label_definitions.query("surface == 'subagents'")
        joined = frames.subagents.merge(definitions, on=["transcript_id", "label"])
        assert len(joined) == 3
        assert set(joined.description) == {"Authored definition"}
    assert all(expected in labels for labels, _ in seen)
    assert all(f"- {expected}: Authored definition" in text for _, text in seen)
    assert spec.model_dump() == before


def test_legacy_vocabulary_reloads_without_rewriting_the_store(demo_log, tmp_path):
    """Old title-case rubrics join already-normalized answers without another scan."""
    result = _run(
        str(demo_log),
        Spec.model_validate({"subagent_labels": ["research"]}),
        judge_models=following_judge([]),
        verify=False,
        scans_dir=str(tmp_path / "scans"),
    )
    path = Path(result.scan_location) / "subagent_classification.parquet"
    table = pd.read_parquet(path)

    def legacy_value(cell):
        value = json.loads(cell)
        for entry in value["label_vocab"]:
            if entry["label"] == "research":
                entry["label"] = "Research"
        return json.dumps(value)

    table["value"] = table.value.map(legacy_value)
    table.to_parquet(path, index=False)
    original = path.read_bytes()
    restored = load(result.scan_location)
    assert set(restored.subagents.label) == {"research"}
    assert "research" in set(restored.label_definitions.label)
    assert "Research" not in set(restored.label_definitions.label)
    assert path.read_bytes() == original


@pytest.mark.parametrize("second", ["Research", "research"])
def test_legacy_rubrics_preserve_exact_duplicates_but_reject_aliases(second):
    """Exact repeated entries retain their first definition; aliases are ambiguous."""
    raw = pd.DataFrame(
        [
            {
                "transcript_id": "t",
                "value": {
                    "label_vocab": [
                        {"label": "Research", "description": "first"},
                        {"label": second, "description": "second"},
                    ]
                },
            }
        ]
    )
    if second == "research":
        with pytest.raises(ValueError, match="collid"):
            label_definitions_df(pd.DataFrame(), raw)
    else:
        frame = label_definitions_df(pd.DataFrame(), raw)
        assert frame.label.tolist() == ["research"]
        assert frame.description.tolist() == ["first"]


@pytest.mark.parametrize("judge", [None, {"regime": "solo", "models": ["custom"]}])
def test_custom_vocabulary_keeps_its_own_keys(judge):
    """A custom frame can keep case-sensitive keys even with judge metadata."""
    raw = pd.DataFrame(
        [
            {
                "transcript_id": "t",
                "value": {
                    "label_vocab": [
                        {"label": "Research", "description": "Custom key"},
                    ]
                },
            }
        ]
    )
    if judge is not None:
        raw.at[0, "value"]["judge"] = judge
    frame = label_definitions_df(pd.DataFrame(), pd.DataFrame(), {"custom": raw})
    assert frame.label.tolist() == ["Research"]


def test_explicit_fallback_alias_is_not_added_twice():
    """A normalized explicit fallback stays spec-declared rather than duplicated."""
    spec = Spec.model_validate({"subagent_labels": ["None Of The Above"]})
    result = run_item(
        subagent_classification(
            spec, scripted_judge(vote_output("none_of_the_above")), verify=False
        ),
        Transcript(transcript_id="span", messages=[ChatMessageUser(content="work")]),
    )
    assert result.value["label_vocab"] == [
        {"label": "none_of_the_above", "description": "", "reserved": False}
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_batched_layer_labels_join_definitions_after_reload(legacy, demo_log, tmp_path):
    """New batched keys join; legacy custom rubrics retain their recorded namespace."""

    @scanner(loader=reasoning_turns(batch=2))
    def label_units():
        return cohort_llm_scanner(
            "Classify the work",
            ["Research", "Result Review"],
            vocabulary={"Research": "Investigate", "Result Review": "Review"},
            models=following_judge([]),
            k_rolls=2,
            batch=True,
            verify=False,
        )

    layer = Layer(name="labels", scanner=label_units(), frame=turns_frame)
    result = _run(
        str(demo_log), Spec(), extra_layers=[layer], scans_dir=str(tmp_path / "scans")
    )
    if legacy:
        path = Path(result.scan_location) / "label_units.parquet"
        table = pd.read_parquet(path)

        def legacy_value(cell):
            value = json.loads(cell)
            for entry in value["label_vocab"]:
                entry["label"] = entry["label"].replace("_", " ").title()
            return json.dumps(value)

        table["value"] = table.value.map(legacy_value)
        table.to_parquet(path, index=False)
    restored = load(result.scan_location, extra_layers=[layer])
    frame = restored.layer_frames["labels"]
    assert len(frame) > 0 and set(frame.label) == {"research"}
    definitions = restored.label_definitions.query("surface == 'labels'")
    if legacy:
        assert definitions.label.tolist() == ["Research", "Result Review"]
        return
    joined = frame.merge(definitions, on=["transcript_id", "label"])
    assert len(joined) == len(frame) and set(joined.description) == {"Investigate"}
