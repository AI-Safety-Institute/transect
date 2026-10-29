"""Run provenance is stable across Spec spelling and survives reload."""

from hashlib import sha256
from pathlib import Path

from helpers import demo_judge

from transect import Layer, Spec, __version__, load, transect
from transect.provenance import (
    canonical_spec_json,
    custom_layer_identities,
    model_identity,
)


def test_new_scan_records_and_reloads_spec_and_run_provenance(demo_log, tmp_path):
    """The exact Spec source, its identity, effective settings and input
    hash are available after the scan and after a judge-free reload."""
    spec_path = demo_log.parents[2] / "examples" / "spec.yaml"
    results = transect(
        logs=str(demo_log),
        spec=str(spec_path),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )

    provenance = results.provenance
    assert provenance is not None
    assert provenance.spec_source == str(spec_path.resolve())
    assert provenance.spec_text == spec_path.read_text()
    assert provenance.spec_source_sha256 == sha256(spec_path.read_bytes()).hexdigest()
    assert provenance.spec_id == f"sha256:{provenance.spec_sha256}"
    assert provenance.settings["judge_models"] is None
    assert provenance.packages["transect"] == __version__
    assert provenance.inputs == {
        str(demo_log.resolve()): sha256(demo_log.read_bytes()).hexdigest()
    }

    loaded = load(results.scan_location)
    assert loaded.provenance == provenance


def test_spec_identity_tracks_content_not_mapping_order():
    """The canonical identity is stable across equivalent mappings and
    changes when a behavioural definition changes."""
    first = Spec.model_validate(
        {
            "phases": [{"label": "planning", "description": "Choose a method."}],
            "context": "One task.",
        }
    )
    reordered = Spec.model_validate(
        {
            "context": "One task.",
            "phases": [{"description": "Choose a method.", "label": "planning"}],
        }
    )
    changed = Spec.model_validate(
        {
            "phases": [{"label": "planning", "description": "Choose two methods."}],
            "context": "One task.",
        }
    )

    assert canonical_spec_json(first) == canonical_spec_json(reordered)
    assert canonical_spec_json(first) != canonical_spec_json(changed)


def test_judge_model_identity_records_constructor_settings():
    """A Model object records its resolved name, config, endpoint and args."""
    identity = model_identity(demo_judge())
    assert identity["model"] == "mockllm/model"
    assert identity["config"] == {}
    assert identity["model_args"] == {"custom_outputs": None}


def test_custom_layer_identity_records_frame_source():
    """A data-only custom layer still records the code that mounted it."""

    def mount_view(raw):
        return raw

    identity = custom_layer_identities([Layer(name="custom_view", frame=mount_view)])[0]
    assert identity["scanner"] is None
    assert identity["frame_source"]["path"].endswith("test_provenance.py")
    assert len(identity["frame_source"]["sha256"]) == 64


def test_a_scan_created_before_provenance_still_loads():
    """Legacy stores remain readable and state that provenance was not
    recorded."""
    fixture = Path(__file__).parent / "fixtures" / "demo_scan"
    assert load(str(fixture)).provenance is None
