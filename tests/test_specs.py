from pathlib import Path

import pandas as pd
import pytest
from helpers import JSON_SPEC
from pydantic import ValidationError

from transect.diagnostics import descramble, scramble_spec
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


def test_scramble_spec_round_trips_and_preserves_definitions():
    """Scrambling renames labels only; descramble inverts them and
    passes reserved/NaN values through."""

    spec = Spec.model_validate(
        {
            "phases": [
                {"label": "setup", "description": "prepare", "ops": True},
                {"label": "experiment", "description": "run things"},
            ],
            "subagent_labels": [{"label": "reviewer", "description": "checks"}],
            "context": "ctx",
        }
    )
    scrambled, mapping = scramble_spec(spec, seed=7)
    assert {p.label for p in scrambled.phases} == {"phase_a", "phase_b"}
    assert [p.description for p in scrambled.phases] == ["prepare", "run things"]
    assert scrambled.phases[0].ops and scrambled.context == "ctx"
    assert scramble_spec(spec, seed=7)[1] == mapping
    tokens = pd.Series([mapping["phases"]["setup"], "none_of_the_above", float("nan")])
    back = descramble(tokens, mapping["phases"])
    assert back.iloc[0] == "setup"
    assert back.iloc[1] == "none_of_the_above"
    assert pd.isna(back.iloc[2])
