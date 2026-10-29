"""Provenance retained with every new scan store."""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, cast

from inspect_ai._util.registry import registry_info
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from transect.spec import Spec

PROVENANCE_SCHEMA_VERSION = 1

_PACKAGE_NAMES = (
    "transect",
    "inspect-ai",
    "inspect-scout",
    "inspect-viz",
    "pandas",
    "pydantic",
    "pyyaml",
    "jinja2",
    "markdown-it-py",
)


class RunProvenance(BaseModel):
    """The inputs and effective settings that produced a scan.

    A new scan records this object. Older scans load with
    ``TransectResults.provenance`` set to None. Expert evidence and
    validation status are intentionally absent: those remain user-owned
    research records.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    """Version of this provenance record."""

    spec_id: str
    """Content-addressed identity for the validated Spec."""

    spec_sha256: str
    """SHA-256 of the canonical JSON serialization of the validated Spec."""

    spec: dict[str, Any]
    """The validated Spec as stored and used by the run."""

    spec_source: str | None
    """Resolved source path when the caller supplied a Spec file."""

    spec_source_sha256: str | None
    """SHA-256 of the source Spec file bytes."""

    spec_text: str | None
    """The source Spec text when the caller supplied a file."""

    packages: dict[str, str | None]
    """Installed versions of Transect and its key runtime dependencies."""

    settings: dict[str, Any]
    """Effective analysis and execution settings for the run."""

    inputs: dict[str, str]
    """Resolved source path to SHA-256 for each scanned input file."""

    custom_layers: list[dict[str, Any]]
    """Registered scanners and source identity for custom analysis layers."""


def build_run_provenance(
    spec: Spec,
    *,
    spec_source: str | Path | None,
    spec_bytes: bytes | None,
    settings: dict[str, Any],
    inputs: dict[str, str],
    custom_layers: list[dict[str, Any]],
) -> RunProvenance:
    """Build the provenance record stamped into a new scan."""
    canonical = canonical_spec_json(spec)
    spec_hash = hashlib.sha256(canonical.encode()).hexdigest()
    source_path = (
        str(Path(spec_source).expanduser().resolve())
        if spec_source is not None
        else None
    )
    return RunProvenance(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        spec_id=f"sha256:{spec_hash}",
        spec_sha256=spec_hash,
        spec=spec.model_dump(mode="json"),
        spec_source=source_path,
        spec_source_sha256=(
            hashlib.sha256(spec_bytes).hexdigest() if spec_bytes is not None else None
        ),
        spec_text=spec_bytes.decode("utf-8") if spec_bytes is not None else None,
        packages=_package_versions(),
        settings=settings,
        inputs=inputs,
        custom_layers=custom_layers,
    )


def provenance_from_scanner_params(
    params: dict[str, Any] | None,
) -> RunProvenance | None:
    """Read a provenance model from a scanner's recorded parameters.

    Scans created before provenance support return None.
    """
    raw = (params or {}).get("provenance")
    return RunProvenance.model_validate(raw) if raw is not None else None


def canonical_spec_json(spec: Spec) -> str:
    """A stable JSON encoding for content-addressing a validated Spec."""
    return json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def input_hashes(transcripts: Any) -> dict[str, str]:
    """Resolved scan input paths and their file hashes."""
    ids = getattr(transcripts, "transcript_ids", None) or {}
    uris = (
        {
            info.source_uri
            for info in transcripts
            if getattr(info, "source_uri", None) is not None
        }
        if isinstance(transcripts, list)
        else set()
    )
    paths = sorted(
        {
            Path(str(uri)).expanduser().resolve()
            for uri in [*ids.values(), *uris]
            if uri is not None
        }
    )
    return {str(path): file_sha256(path) for path in paths if path.is_file()}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(value: Any) -> Any:
    """A JSON-safe identity for a judge model argument."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [model_identity(item) for item in value]
    config = getattr(value, "config", None)
    config_value = (
        cast(Any, config).model_dump(mode="json", exclude_none=True)
        if config is not None and hasattr(config, "model_dump")
        else config
    )
    api = getattr(value, "api", None)
    model_args = getattr(value, "model_args", None)
    return {
        "model": str(value),
        "config": config_value,
        "base_url": getattr(api, "base_url", None),
        "model_args": (
            to_jsonable_python(model_args, fallback=lambda _: None)
            if model_args is not None
            else None
        ),
    }


def custom_layer_identities(layers: list[Any]) -> list[dict[str, Any]]:
    """Scanner and source identity for user-defined analysis layers."""
    records = []
    for layer in layers:
        scanner = layer.scanner
        record: dict[str, Any] = {
            "name": layer.name,
            "scanner": (
                str(registry_info(scanner).name) if scanner is not None else None
            ),
            "scanner_source": (
                _callable_source(scanner) if scanner is not None else None
            ),
            "scanner_args": to_jsonable_python(
                layer.scanner_args, fallback=lambda _: None
            ),
            "frame_source": (
                _callable_source(layer.frame) if callable(layer.frame) else None
            ),
        }
        records.append(record)
    return records


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _callable_source(value: Any) -> dict[str, str] | None:
    callable_value = value if inspect.isfunction(value) else type(value).__call__
    try:
        path = inspect.getsourcefile(callable_value)
    except (OSError, TypeError):
        return None
    if path is None:
        return None
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved) if resolved.is_file() else "",
    }
