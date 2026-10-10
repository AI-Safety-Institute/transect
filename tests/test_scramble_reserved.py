"""Renaming labels preserves the effective vocabulary and operational role."""

import pandas as pd
import pytest
from helpers import demo_judge, run_item, scripted_judge, vote_output
from inspect_ai.model import ChatMessageUser
from inspect_scout import Transcript

from transect import load
from transect.api import _run
from transect.diagnostics import descramble, scramble_spec
from transect.scanners.phases_common import resolve_phases
from transect.scanners.subagents import subagent_classification
from transect.spec import Spec


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
        ["setup", "experiment"],
        [],
    ],
)
def test_scrambling_preserves_resolved_phase_semantics(phases):
    """The resolved definitions and selected ops role differ only by the mapping."""
    original = Spec.model_validate({"phases": phases, "context": "context"})
    snapshot = original.model_dump()
    before, before_ops = resolve_phases(original)
    scrambled, maps = scramble_spec(original, seed=7)
    after, after_ops = resolve_phases(scrambled)
    mapping = maps["phases"]
    expected = [
        {**phase.model_dump(), "label": mapping.get(phase.label, phase.label)}
        for phase in before
    ]
    assert [phase.model_dump() for phase in after] == expected
    assert after_ops == mapping.get(before_ops, before_ops)
    assert descramble(pd.Series([p.label for p in after]), mapping).tolist() == [
        p.label for p in before
    ]
    assert len(set(mapping.values())) == len(mapping)
    assert original.model_dump() == snapshot
    assert scramble_spec(original, seed=7)[1] == maps
    assert scrambled.context == original.context


def test_subagent_fallback_is_preserved_but_ops_is_an_ordinary_role():
    """Sub-agent fallback stays reserved while an ops-named role can be renamed."""
    spec = Spec.model_validate(
        {"subagent_labels": ["ops", "research", "none_of_the_above"]}
    )
    scrambled, maps = scramble_spec(spec, seed=7)
    mapping = maps["subagents"]
    assert mapping["none_of_the_above"] == "none_of_the_above"
    assert mapping["ops"].startswith("role_")
    result = run_item(
        subagent_classification(
            scrambled, scripted_judge(vote_output("none_of_the_above")), verify=False
        ),
        Transcript(transcript_id="span", messages=[ChatMessageUser(content="work")]),
    )
    vocab = result.value["label_vocab"]
    assert len(vocab) == 3
    assert descramble(pd.Series([v["label"] for v in vocab]), mapping).tolist() == [
        label.label for label in spec.subagent_labels
    ]
    assert not any(entry["reserved"] for entry in vocab)


@pytest.mark.parametrize("fallback", ["None Of The Above", " none_of_the_above "])
def test_subagent_fallback_spelling_is_preserved(fallback):
    """Fallback aliases retain the key used by closed-label normalization."""
    spec = Spec.model_validate({"subagent_labels": [fallback, "research"]})
    scrambled, maps = scramble_spec(spec, seed=7)
    assert maps["subagents"][fallback] == fallback
    assert scrambled.subagent_labels[0].label == fallback
    assert descramble(
        pd.Series([r.label for r in scrambled.subagent_labels]), maps["subagents"]
    ).tolist() == [fallback, "research"]


@pytest.mark.parametrize("special", ["ops", "none_of_the_above"])
def test_scrambled_run_and_reload_keep_the_same_categories(special, demo_log, tmp_path):
    """The full phase pipeline stores a vocabulary isomorphic to the original one."""
    original = Spec.model_validate({"phases": [special, "experiment"]})
    scrambled, maps = scramble_spec(original, seed=7)
    result = _run(
        str(demo_log),
        scrambled,
        judge_models=demo_judge(phase_label=maps["phases"]["experiment"]),
        verify=False,
        scans_dir=str(tmp_path / "scans"),
    )
    expected, _ = resolve_phases(original)
    restored = load(result.scan_location)
    for frames in [result, restored]:
        definitions = frames.label_definitions.query("surface == 'phases'")
        assert descramble(definitions.label, maps["phases"]).tolist() == [
            p.label for p in expected
        ]
        assert len(frames.phases) > 0
        assert set(descramble(frames.phases.phase, maps["phases"])) == {"experiment"}
