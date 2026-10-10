"""Developer-authored specs: one eval's phase/subagent vocab."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class Phase(BaseModel):
    """One phase of the eval's expected trajectory."""

    model_config = ConfigDict(extra="forbid")

    label: str
    """The phase name the judge answers with (and the report shows)."""

    description: str = ""
    """What work belongs to this phase, in the author's own words -
    rendered into the judge's rubric."""

    ops: bool = False
    """Marks this phase as the operational bucket (setup/coordination/
    plumbing); without one the reserved "ops" phase is appended."""


class SubagentLabel(BaseModel):
    """One sub-agent role the eval is expected to spawn."""

    model_config = ConfigDict(extra="forbid")

    label: str
    """The declared role name, retained verbatim in the spec.

    Scanners lowercase it and replace whitespace with underscores for
    their closed answer vocabulary and result keys.
    """

    description: str = ""
    """What this role does, in the author's own words - rendered into
    the judge's rubric."""


class Spec(BaseModel):
    """One eval's description: the vocabularies the judges hold
    against the transcript. A bare string in either list means
    ``{"label": <str>}``."""

    model_config = ConfigDict(extra="forbid")

    phases: list[Phase] = []
    """The phase vocabulary; empty disables phase segmentation."""

    subagent_labels: list[SubagentLabel] = []
    """The sub-agent role vocabulary; empty disables classification."""

    context: str = ""
    """Free-text task context for phase prompts; not passed to subagent judges."""

    extra: dict[str, Any] = {}
    """The user's own namespace (e.g. custom Layer content, keyed by
    layer name); never validated, never read by Transect itself."""

    @field_validator("phases", "subagent_labels", mode="before")
    @classmethod
    def _plain_labels_ok(cls, value):  # "recon" == {"label": "recon"}
        return [{"label": p} if isinstance(p, str) else p for p in (value or [])]


def load_spec(path: Path | str) -> Spec:
    """Load a spec from JSON or YAML (same schema either way)."""
    return Spec.model_validate(yaml.safe_load(Path(path).read_text()))
