"""Added run identities survive reload without duplicating Scout's facts."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest
from conftest import turn_counter
from helpers import demo_judge
from inspect_scout import ScanSpec

import transect.provenance as provenance_module
from transect import Layer, Spec, __version__, load, transect, turns_frame
from transect.provenance import canonical_spec_json
from transect.spec import load_spec


@pytest.mark.parametrize("source", ["eval_log", "openclaw"])
def test_scan_and_reload_provenance(
    source, demo_log, openclaw_log, tmp_path, monkeypatch
):
    """Novel identities live in metadata; reload reads stored facts without inputs."""
    original = demo_log if source == "eval_log" else openclaw_log
    log = tmp_path / original.name
    shutil.copyfile(original, log)
    spec_path = tmp_path / "spec.yaml"
    shutil.copyfile(demo_log.parents[2] / "examples" / "spec.yaml", spec_path)
    spec = load_spec(spec_path)
    hashed_paths = []
    file_sha256 = provenance_module.file_sha256

    def record_hash(path):
        hashed_paths.append(path)
        return file_sha256(path)

    monkeypatch.setattr(provenance_module, "file_sha256", record_hash)
    layers = [Layer(name="custom_turns", scanner=turn_counter, frame=turns_frame)]
    sample = "house-price" if source == "eval_log" else None
    results = transect(
        logs=str(log),
        spec=spec_path,
        sample=sample,
        epochs="all",
        judge_models=demo_judge() if source == "eval_log" else None,
        narrate=False,
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=layers,
    )

    provenance = results.provenance
    assert provenance is not None
    stored = json.loads((Path(results.scan_location) / "_scan.json").read_text())
    assert provenance.scan == ScanSpec.model_validate(stored)
    assert stored["metadata"] == {
        "transect": provenance.model_dump(mode="json", exclude={"scan"})
    }
    assert set(stored["metadata"]["transect"]) == {
        "transect_version",
        "spec_id",
        "spec_source",
        "spec_source_sha256",
        "sample",
        "epochs",
        "inputs",
        "custom_layers",
    }
    assert stored["scanners"]["eval_setup"]["params"] == {}
    assert all("provenance" not in s["params"] for s in stored["scanners"].values())
    assert provenance.spec_source == str(spec_path.resolve())
    assert provenance.spec_source_sha256 == sha256(spec_path.read_bytes()).hexdigest()
    assert (
        provenance.spec_id
        == f"sha256:{sha256(canonical_spec_json(spec).encode()).hexdigest()}"
    )
    assert provenance.transect_version == __version__
    assert (provenance.sample, provenance.epochs) == (sample, "all")
    identity = provenance.custom_layers[0]
    assert set(identity) == {"name", "scanner", "scanner_source", "frame_source"}
    assert Path(identity["scanner_source"]["path"]).name == "conftest.py"
    for key in ("scanner_source", "frame_source"):
        path = Path(identity[key]["path"])
        assert identity[key]["sha256"] == sha256(path.read_bytes()).hexdigest()
    allowed_hashes = {
        Path(identity[key]["path"]) for key in ("scanner_source", "frame_source")
    }
    if source == "eval_log":
        allowed_hashes.add(log.resolve())
        assert provenance.inputs == {
            str(log.resolve()): sha256(log.read_bytes()).hexdigest()
        }
        for key in ("decision_phases", "subagent_classification"):
            params = provenance.scan.scanners[key].params
            assert params == stored["scanners"][key]["params"]
            assert params["judge_models"]["model"] == "mockllm/model"
    else:
        assert provenance.inputs == {}
        assert provenance.scan.transcripts is not None
        snapshot = Path(provenance.scan.transcripts.location)
        assert snapshot.is_dir()
        assert str(snapshot) == results.transcripts_location
        assert "transcript_snapshots" in snapshot.parts
    assert set(hashed_paths) == allowed_hashes

    def no_hash(path):
        pytest.fail(f"load() tried to hash {path}")

    monkeypatch.setattr(provenance_module, "file_sha256", no_hash)
    log.unlink()
    spec_path.unlink()
    assert load(results.scan_location, extra_layers=layers).provenance == provenance
    assert (
        load(str(Path(__file__).parent / "fixtures" / "demo_scan")).provenance is None
    )


def test_spec_identity_tracks_content_not_mapping_order():
    """Equivalent mappings share an identity; changed behavioural definitions differ."""
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
