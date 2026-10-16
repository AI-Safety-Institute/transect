from pathlib import Path

import pandas as pd
import pytest
from helpers import JSON_SPEC, demo_judge, run_item, scripted_judge, vote_output
from inspect_ai.model import ChatMessageUser
from inspect_scout import Transcript
from pydantic import ValidationError

from transect import load
from transect.api import _run
from transect.diagnostics import descramble, scramble_spec
from transect.scanners.phases_common import reserved_phase_labels, resolve_phases
from transect.scanners.subagents import subagent_classification
from transect.spec import Spec, load_spec

MINIMAL = {
    "context": "An AI R&D run.",
    "phases": [
        "planning",
        {"label": "experimentation", "description": "Runs experiments."},
        {"label": "coordination", "ops": True},
    ],
    "subagent_labels": ["experiment_run", {"label": "result_review"}],
}


def test_plain_string_labels_normalise_to_full_entries():
    """A bare "planning" string means {"label": "planning"} for phases
    and sub-agent labels alike."""
    spec = Spec.model_validate(MINIMAL)
    assert spec.phases[0].label == "planning"
    assert spec.phases[0].description == ""
    assert spec.subagent_labels[0].label == "experiment_run"


def test_ops_flag_marks_the_operational_phase():
    """A phase declared with ops: true carries the flag; others default off."""
    spec = Spec.model_validate(MINIMAL)
    assert [p.ops for p in spec.phases] == [False, False, True]


def test_ops_bucket_off_resolves_no_operational_phase():
    """With ops_bucket off no reserved ops phase is appended, no phase is
    the operational bucket, and the escape hatch no longer excludes
    operational work."""
    spec = Spec.model_validate({"phases": ["experiment"], "ops_bucket": False})
    phases, ops_name = resolve_phases(spec)
    assert [p.label for p in phases] == ["experiment", "none_of_the_above"]
    assert ops_name is None
    assert "operational" not in phases[-1].description


def test_ops_bucket_off_rejects_a_flagged_ops_phase():
    """A phase flagged ops contradicts ops_bucket: false and fails validation."""
    with pytest.raises(ValidationError, match="flagged ops"):
        Spec.model_validate(
            {"phases": [{"label": "coordination", "ops": True}], "ops_bucket": False}
        )


def test_ops_bucket_off_makes_a_phase_named_ops_an_ordinary_phase():
    """Without the bucket a declared phase called ops is selected by
    nothing, so it is renamed by scrambling and the switch survives."""
    spec = Spec.model_validate({"phases": ["ops", "experiment"], "ops_bucket": False})
    assert reserved_phase_labels(spec) == set()
    scrambled, maps = scramble_spec(spec, seed=7)
    assert maps["phases"]["ops"].startswith("phase_")
    assert scrambled.ops_bucket is False


@pytest.mark.parametrize(
    "content",
    [
        {"phasez": ["typo"]},
        {"phases": [{"label": "a", "descriptionz": "typo"}]},
        {"subagent_labels": [{"label": "a", "ops": True}]},
    ],
    ids=["top-level-typo", "phase-field-typo", "ops-on-subagent-label"],
)
def test_unknown_fields_are_rejected_loudly(content):
    """A misspelled field fails validation instead of silently dropping."""
    with pytest.raises(ValidationError):
        Spec.model_validate(content)


def test_extra_namespace_is_preserved_verbatim():
    """User content under extra survives validation untouched."""
    spec = Spec.model_validate({"extra": {"tool_risk": {"threshold": 3}}})
    assert spec.extra == {"tool_risk": {"threshold": 3}}


def test_empty_spec_is_valid_and_empty():
    """A spec with nothing declared loads with empty vocabularies."""
    spec = Spec.model_validate({})
    assert spec.phases == [] and spec.subagent_labels == []


def test_committed_example_specs_load():
    """The specs shipped for the tests and examples parse."""
    assert load_spec(JSON_SPEC).phases
    root = Path(__file__).parents[1]
    example = load_spec(root / "examples" / "spec.yaml")
    assert example.phases and example.subagent_labels


@pytest.mark.parametrize(
    "label, key",
    [
        ("Research", "research"),
        ("Result Review", "result_review"),
        (" review", "review"),
        ("", ""),
    ],
)
def test_subagent_label_must_be_its_own_key(label, key):
    """A sub-agent label spelled unlike its stored key is refused up
    front, naming the key to use."""
    with pytest.raises(ValidationError, match="must be its own key") as caught:
        Spec.model_validate({"subagent_labels": [label]})
    if key:
        assert f", {key!r}:" in str(caught.value)
    assert Spec.model_validate({"subagent_labels": ["result_review"]})


@pytest.mark.parametrize(
    "phases",
    [
        ["ops", "experiment"],
        [{"label": "coordination", "ops": True}, "experiment"],
        [
            {"label": "none_of_the_above", "description": "Authored fallback"},
            "experiment",
        ],
        ["ops", "none_of_the_above", "experiment"],
        ["ops", {"label": "coordination", "ops": True}, "experiment"],
        [{"label": "setup", "description": "prepare", "ops": True}, "experiment"],
        [],
    ],
)
def test_scramble_spec_preserves_the_resolved_phase_vocabulary(phases):
    """Scrambling renames labels only: definitions, flags, context and
    the resolved vocabulary differ from the original by the mapping alone,
    and descramble inverts it, passing unmapped and NaN values through."""
    original = Spec.model_validate({"phases": phases, "context": "ctx"})
    before, before_ops = resolve_phases(original)
    scrambled, maps = scramble_spec(original, seed=7)
    mapping = maps["phases"]
    after, after_ops = resolve_phases(scrambled)
    assert [p.model_dump() for p in after] == [
        {**p.model_dump(), "label": mapping.get(p.label, p.label)} for p in before
    ]
    assert after_ops == mapping.get(before_ops, before_ops)
    assert scrambled.context == "ctx"
    assert len(set(mapping.values())) == len(mapping)
    assert scramble_spec(original, seed=7)[1] == maps
    back = descramble(
        pd.Series([*(p.label for p in after), "unmapped", float("nan")]), mapping
    )
    assert back.iloc[:-2].tolist() == [p.label for p in before]
    assert back.iloc[-2] == "unmapped" and pd.isna(back.iloc[-1])


@pytest.mark.parametrize(
    "phases, kept",
    [
        (["ops", "experiment"], {"ops"}),
        (["ops", {"label": "coordination", "ops": True}], set()),
        ([{"label": "ops", "ops": True}, "experiment"], set()),
        (["none_of_the_above", "experiment"], {"none_of_the_above"}),
    ],
)
def test_only_phase_labels_resolved_by_name_keep_their_names(phases, kept):
    """A declared ops phase is renamed whenever a flag, not its name, selects
    the bucket; a declared none_of_the_above always keeps its name."""
    spec = Spec.model_validate({"phases": phases})
    mapping = scramble_spec(spec, seed=7)[1]["phases"]
    assert {label for label, token in mapping.items() if label == token} == kept


def test_scrambled_subagent_vocabulary_is_isomorphic_to_the_original():
    """The scanner's stored rubric descrambles to the original's, entry for
    entry: a declared fallback keeps its name and every other label is
    renamed, an ops-named role included."""
    spec = Spec.model_validate(
        {"subagent_labels": ["ops", "research", "none_of_the_above"]}
    )
    scrambled, maps = scramble_spec(spec, seed=7)
    mapping = maps["subagents"]
    assert mapping["none_of_the_above"] == "none_of_the_above"
    assert mapping["ops"].startswith("role_")

    def rubric(spec):
        result = run_item(
            subagent_classification(
                spec, scripted_judge(vote_output("none_of_the_above")), verify=False
            ),
            Transcript(transcript_id="span", messages=[ChatMessageUser(content="x")]),
        )
        return [(v["label"], v["reserved"]) for v in result.value["label_vocab"]]

    expected = rubric(spec)
    labels, flags = zip(*rubric(scrambled), strict=True)
    assert (
        list(zip(descramble(pd.Series(labels), mapping), flags, strict=True))
        == expected
    )


def test_scrambled_run_stores_an_isomorphic_phase_vocabulary(demo_log, tmp_path):
    """The full phase pipeline records the scrambled rubric, and both the run
    and its reload descramble to the original categories."""
    original = Spec.model_validate({"phases": ["ops", "experiment"]})
    scrambled, maps = scramble_spec(original, seed=7)
    result = _run(
        str(demo_log),
        scrambled,
        judge_models=demo_judge(phase_label=maps["phases"]["experiment"]),
        verify=False,
        scans_dir=str(tmp_path / "scans"),
    )
    expected = [p.label for p in resolve_phases(original)[0]]
    for frames in [result, load(result.scan_location)]:
        definitions = frames.label_definitions.query("surface == 'phases'")
        assert descramble(definitions.label, maps["phases"]).tolist() == expected
        assert set(descramble(frames.phases.phase, maps["phases"])) == {"experiment"}
