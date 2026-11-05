"""Transect run identities and a read-time view of Scout's scan provenance."""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

from inspect_ai._util.registry import registry_info
from inspect_scout import ScanSpec, TranscriptInfo
from pydantic import BaseModel, ConfigDict

from transect.layers import Layer
from transect.spec import Spec


class RunMetadata(BaseModel):
    """Identities no scanner owns, stored under Scout metadata's transect key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transect_version: str
    """Installed Transect package version, separate from Scout's cwd revision."""

    spec_id: str
    """SHA-256 identity of the canonical validated Spec JSON."""

    spec_source: str | None
    """Resolved source path when the caller supplied a Spec file."""

    spec_source_sha256: str | None
    """SHA-256 of the source Spec file bytes."""

    sample: str | None
    """Sample selection argument, before automatic selection."""

    epochs: int | list[int] | Literal["all"] | None
    """Epoch selection argument, before automatic selection."""

    inputs: dict[str, str]
    """Resolved eval-log source paths and file hashes; databases are not hashed."""

    custom_layers: list[dict[str, Any]]
    """Custom layer scanner and frame source identities."""


class RunProvenance(RunMetadata):
    """Read-time view combining Transect metadata with Scout's stored facts.

    Judge facts remain scoped to each entry in ``scan.scanners`` and its
    params. The scan header owns timestamps, revision, options, package
    versions and transcript locations. This combined view is never stored.
    """

    scan: ScanSpec
    """Scout's recorded header and scanner specifications, without re-derivation."""


def build_run_metadata(
    spec: Spec,
    *,
    spec_source: str | Path | None,
    spec_bytes: bytes | None,
    sample: str | None,
    epochs: int | list[int] | Literal["all"] | None,
    inputs: dict[str, str],
    custom_layers: list[dict[str, Any]],
) -> RunMetadata:
    """Build only Transect's additions to Scout's run-level metadata."""
    spec_hash = hashlib.sha256(canonical_spec_json(spec).encode()).hexdigest()
    return RunMetadata(
        transect_version=package_version("transect"),
        spec_id=f"sha256:{spec_hash}",
        spec_source=str(Path(spec_source).resolve())
        if spec_source is not None
        else None,
        spec_source_sha256=(
            hashlib.sha256(spec_bytes).hexdigest() if spec_bytes is not None else None
        ),
        sample=sample,
        epochs=epochs,
        inputs=inputs,
        custom_layers=custom_layers,
    )


def provenance_from_scan(scan: ScanSpec) -> RunProvenance | None:
    """Read Transect metadata and existing Scout facts without reading inputs."""
    raw = (scan.metadata or {}).get("transect")
    if raw is None:
        return None
    metadata = RunMetadata.model_validate(raw)
    return RunProvenance(**metadata.model_dump(), scan=scan)


def canonical_spec_json(spec: Spec) -> str:
    """A stable JSON encoding for content-addressing a validated Spec."""
    return json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def input_hashes(transcripts: list[TranscriptInfo]) -> dict[str, str]:
    """Hash selected eval-log files, once per source; never hash database files."""
    paths = sorted(
        {
            Path(info.source_uri).resolve()
            for info in transcripts
            if info.source_type == "eval_log" and info.source_uri is not None
        }
    )
    return {str(path): file_sha256(path) for path in paths}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def custom_layer_identities(layers: list[Layer]) -> list[dict[str, Any]]:
    """Scanner and frame source identities, separate from scanner parameters."""
    return [
        {
            "name": layer.name,
            "scanner": (
                str(registry_info(layer.scanner).name)
                if layer.scanner is not None
                else None
            ),
            "scanner_source": (
                _callable_source(layer.scanner) if layer.scanner is not None else None
            ),
            "frame_source": (
                _callable_source(layer.frame) if callable(layer.frame) else None
            ),
        }
        for layer in layers
    ]


def _callable_source(value: Any) -> dict[str, str] | None:
    callable_value = value if inspect.isfunction(value) else type(value).__call__
    try:
        path = inspect.getsourcefile(inspect.unwrap(callable_value))
    except (OSError, TypeError):
        return None
    if path is None:
        return None
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}
