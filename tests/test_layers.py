"""The custom-layer injection surface: Layer validation, scan-batch
execution, and frame mounting at results.layer_frames."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import turn_counter
from inspect_scout import Result, Scanner, Transcript, scanner

import transect
from transect import Layer, load, turns_frame
from transect.layers import resolve_scanner_factories, validate_layers

SPEC = Path(__file__).parents[1] / "examples" / "spec.yaml"


def test_scanner_plus_frame_fn_mounts_a_tidy_frame(layered_run):
    """The full pipeline: the custom scanner joins the scan batch and
    its frame fn's output lands at results.layer_frames[name]."""
    results, _ = layered_run
    frame = results.layer_frames["turn_chars"]
    assert set(frame.columns) >= {"transcript_id", "turn", "chars"}
    assert len(frame) == 16  # one row per model turn of the demo log
    assert sorted(frame.turn) == list(range(16))
    # user frames never claim the package frames contract
    assert all("schema_version" not in f.columns for f in results.layer_frames.values())


def test_scanner_without_frame_fn_gets_the_generic_flatten(layered_run):
    """No frame fn: identity columns + the value's top-level keys,
    one row per result."""
    results, _ = layered_run
    frame = results.layer_frames["raw_counts"]
    assert len(frame) == 1
    assert "transcript_id" in frame.columns
    assert int(frame.iloc[0]["n_messages"]) > 0  # scalar projects
    assert isinstance(frame.iloc[0]["turns"], list)  # nested stays raw


def test_data_only_layer_mounts_verbatim(layered_run):
    """A ready DataFrame mounts as handed over: bring-your-own data."""
    results, _ = layered_run
    frame = results.layer_frames["shell"]
    assert list(frame.shell) == ["pip", "pytest"]


def test_load_remounts_layer_frames_from_the_store(layered_run, capsys):
    """load(scans_dir, extra_layers=...) replays layer frames from
    the stored scan (frame fns re-run, no scanning)."""
    results, scans = layered_run
    replayed = load(
        str(scans),
        extra_layers=[
            Layer(name="turn_chars", scanner=turn_counter(), frame=turns_frame)
        ],
    )
    pd.testing.assert_frame_equal(
        replayed.layer_frames["turn_chars"].reset_index(drop=True),
        results.layer_frames["turn_chars"].reset_index(drop=True),
    )
    assert "message_counter" in capsys.readouterr().out  # unclaimed, warned


def test_layer_scanner_missing_from_the_store_fails_loudly(layered_run):
    """Replaying a layer whose scanner never ran in the stored scan
    names the scanner rather than mounting an empty frame."""
    _, scans = layered_run

    @scanner(messages="all")
    def never_scanned() -> Scanner[Transcript]:
        async def execute(transcript: Transcript) -> Result:
            return Result(value={})

        return execute

    with pytest.raises(KeyError, match="never_scanned"):
        load(str(scans), extra_layers=[Layer(name="x", scanner=never_scanned())])


def test_empty_layer_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        transect.transect(
            logs=str(tmp_path / "never-touched"),
            spec=str(SPEC),
            scans_dir=str(tmp_path / "scans"),
            viewer=False,
            open_report=False,
            extra_layers=[Layer(name="hollow")],
        )


def _bare_cohort_scanner():
    return transect.cohort_llm_scanner(
        question="q", answer=["a", "b"], models="mockllm/model"
    )


@pytest.mark.parametrize(
    ("layers", "match"),
    [
        (
            lambda: [Layer(name="phases", frame=pd.DataFrame({"turn": [0]}))],
            "phases",
        ),
        (
            lambda: [
                Layer(name="a", frame=pd.DataFrame({"turn": [0]})),
                Layer(name="a", frame=pd.DataFrame({"turn": [1]})),
            ],
            "duplicate",
        ),
        (
            lambda: [
                Layer(
                    name="x",
                    scanner=turn_counter(),
                    frame=pd.DataFrame({"turn": [0]}),
                )
            ],
            "either",
        ),
        (lambda: [Layer(name="x", scanner=_bare_cohort_scanner())], "@scanner"),
        (
            lambda: [
                Layer(name="a", scanner=turn_counter()),
                Layer(name="b", scanner=turn_counter(), frame=turns_frame),
            ],
            "share the scanner",
        ),
        (
            lambda: [Layer(name="token_spend", frame=pd.DataFrame({"turn": [0]}))],
            "section key",
        ),
        (
            lambda: [
                Layer(
                    name="x",
                    frame=pd.DataFrame({"turn": [0]}),
                    tags=("skill",),  # type: ignore[arg-type]
                )
            ],
            "tags is a bool",
        ),
        (
            lambda: [
                Layer(
                    name="x",
                    frame=pd.DataFrame({"turn": [0]}),
                    audit=("turn", "label", "extra"),  # type: ignore[arg-type]
                )
            ],
            "unit_col, label_col",
        ),
        (
            lambda: [Layer(name="x", tags=True, audit=("turn", "label"))],
            "frame to audit",
        ),
    ],
    ids=[
        "builtin-name-collision",
        "duplicate-names",
        "scanner-with-ready-frame",
        "undecorated-scanner",
        "shared-scanner",
        "section-key-collision",
        "tags-tuple",
        "audit-three-columns",
        "audit-only-layer",
    ],
)
def test_validation_rules_reject_each_malformed_shape(layers, match):
    """One case per validate_layers rule (the empty-layer rule is
    covered at the transect() entry point above)."""
    with pytest.raises(ValueError, match=match):
        validate_layers(layers(), builtin_frames=("phases",))


def test_validate_audit_names_the_full_judged_contract():
    """audit is checked against everything the audit reads at load -
    declared columns plus the judged projection - so a mechanical or
    hand-built frame is refused before render, with the fix named."""
    from transect.layers import validate_audit

    layer = Layer(name="x", scanner=turn_counter(), audit=("turn", "label"))
    with pytest.raises(ValueError, match="judge_regime"):
        validate_audit(layer, pd.DataFrame({"turn": [0], "label": ["a"]}))
    with pytest.raises(ValueError, match="'turn'"):
        validate_audit(layer, pd.DataFrame({"label": ["a"]}))


def test_factory_scanner_resolution_threads_the_judge_args():
    """A factory-form layer scanner is instantiated with the subset of
    the run's judge arguments its signature declares; a factory taking
    judge_models with none passed fails loudly instead of silently
    scanning on the default model."""
    seen = {}

    def factory(judge_models, k_rolls=1):
        seen.update(judge_models=judge_models, k_rolls=k_rolls)
        return turn_counter()

    (resolved,) = resolve_scanner_factories(
        [Layer(name="x", scanner=factory)],
        {"judge_models": "m", "k_rolls": 3, "verify": True},
    )
    assert seen == {"judge_models": "m", "k_rolls": 3}  # verify not declared
    assert resolved.scanner is not None and resolved.scanner is not factory
    # an already-instantiated scanner passes through untouched
    instance = turn_counter()
    (kept,) = resolve_scanner_factories(
        [Layer(name="y", scanner=instance)], {"judge_models": None}
    )
    assert kept.scanner is instance
    with pytest.raises(ValueError, match="judge_models"):
        resolve_scanner_factories(
            [Layer(name="z", scanner=factory)], {"judge_models": None}
        )


def test_factory_receives_the_spec_and_its_scanner_args():
    """A factory declaring ``spec`` gets the loaded Spec; ``scanner_args``
    ride along as given; a key the factory cannot take, or one that
    shadows an injected argument, is refused before any scan."""
    seen = {}

    def factory(spec, rubric, judge_models=None, threshold=0.5):
        seen.update(spec=spec, rubric=rubric, threshold=threshold)
        return turn_counter()

    marker = object()
    resolve_scanner_factories(
        [
            Layer(
                name="x", scanner=factory, scanner_args={"rubric": "r", "threshold": 1}
            )
        ],
        {"judge_models": "m"},
        spec=marker,
    )
    assert seen == {"spec": marker, "rubric": "r", "threshold": 1}
    with pytest.raises(ValueError, match="not parameters"):
        resolve_scanner_factories(
            [Layer(name="x", scanner=factory, scanner_args={"rubric": "r", "nope": 1})],
            {"judge_models": "m"},
        )
    with pytest.raises(ValueError, match="shadow"):
        resolve_scanner_factories(
            [Layer(name="x", scanner=factory, scanner_args={"rubric": "r", "spec": 1})],
            {"judge_models": "m"},
        )
    with pytest.raises(ValueError, match="factory-form"):
        validate_layers(
            [Layer(name="y", scanner=turn_counter(), scanner_args={"rubric": "r"})]
        )


def test_turns_frame_explodes_the_turns_key():
    """The exported helper: identity + one row per value['turns']
    entry - per-turn layers need no hand-written frame fn."""
    result = SimpleNamespace(
        value={"turns": [{"turn": 0, "risk": 0.1}, {"turn": 1, "risk": 0.7}]},
        label=None,
        answer=None,
        explanation=None,
    )
    row = {
        "transcript_task_id": "s1",
        "transcript_task_set": "ts",
        "transcript_task_repeat": 1,
        "transcript_id": "tr1",
        "transcript_agent": "react",
        "value": json.dumps(result.value),
        "label": None,
        "answer": None,
        "explanation": None,
    }
    frame = turns_frame(pd.DataFrame([row]))
    assert len(frame) == 2
    assert list(frame.risk) == [0.1, 0.7]
    assert set(frame.transcript_id) == {"tr1"}
