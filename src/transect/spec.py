"""Developer-authored specs: one eval's phase/subagent vocab."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DEFAULT_OPS_LABEL = "ops"
"""The label of the reserved operational bucket, appended when a spec
keeps the default ``phase_ops_label`` and declares no phase so named."""

NONE_PHASE_LABEL = "none_of_the_above"
"""The reserved escape hatch, offered on every judged surface."""


class Phase(BaseModel):
    """One phase of the eval's expected trajectory."""

    model_config = ConfigDict(extra="forbid")

    label: str
    """The phase name the judge answers with (and the report shows)."""

    description: str = ""
    """What work belongs to this phase, in the author's own words -
    rendered into the judge's rubric."""

    @model_validator(mode="before")
    @classmethod
    def _ops_flag_moved_to_the_spec(cls, value):
        if isinstance(value, dict) and "ops" in value:
            raise ValueError(
                "the per-phase ops flag is gone: "
                + (
                    "name the operational bucket with "
                    f"phase_ops_label: {value.get('label')!r} on the spec"
                    if value["ops"]
                    else "ops: false never did anything, drop it"
                )
            )
        return value


class SubagentLabel(BaseModel):
    """One sub-agent role the eval is expected to spawn."""

    model_config = ConfigDict(extra="forbid")

    label: str
    """The role name the judge answers with: its own key (see
    `label_key`), since answers are stored under that key."""

    description: str = ""
    """What this role does, in the author's own words - rendered into
    the judge's rubric."""

    @field_validator("label")
    @classmethod
    def _label_is_its_own_key(cls, value: str) -> str:
        key = label_key(value)
        if not is_label_key(value):
            raise ValueError(
                f"sub-agent label {value!r} must be its own key"
                + (f", {key!r}" if key else "")
                + ": judge answers are stored lowercased with whitespace as "
                "underscores, and a label spelled otherwise never joins them"
            )
        return value


class Spec(BaseModel):
    """One eval's description: the vocabularies the judges hold
    against the transcript. A bare string in either list means
    ``{"label": <str>}``."""

    model_config = ConfigDict(extra="forbid")

    phases: list[Phase] = []
    """The phase vocabulary; empty disables phase segmentation."""

    subagent_labels: list[SubagentLabel] = []
    """The sub-agent role vocabulary; empty disables classification."""

    phase_ops_label: str | None = "ops"
    """The phase label that is the operational bucket (setup /
    coordination / plumbing). A declared phase with this label is the
    bucket, defined by its description; with the default and no such
    phase the reserved bucket is appended. Any other label must name a
    declared phase. ``None`` offers no bucket, so operational turns take
    a declared phase or ``none_of_the_above``. Phase-only: the sub-agent
    surface has no operational role."""

    context: str = ""
    """Free-text task context for phase prompts; not passed to subagent judges."""

    extra: dict[str, Any] = {}
    """The user's own namespace (e.g. custom Layer content, keyed by
    layer name); never validated, never read by Transect itself."""

    @field_validator("phases", "subagent_labels", mode="before")
    @classmethod
    def _plain_labels_ok(cls, value):  # "recon" == {"label": "recon"}
        return [{"label": p} if isinstance(p, str) else p for p in (value or [])]

    @model_validator(mode="after")
    def _phase_ops_label_names_a_phase(self):
        label = self.phase_ops_label
        if label is None:
            return self
        if label == NONE_PHASE_LABEL:
            raise ValueError(
                f"phase_ops_label {label!r} is the escape hatch, not a bucket"
            )
        if label != DEFAULT_OPS_LABEL and all(p.label != label for p in self.phases):
            raise ValueError(
                f"phase_ops_label {label!r} names no declared phase: "
                f"{[p.label for p in self.phases]}"
            )
        return self


def label_key(label: str) -> str:
    """The key a closed label is stored and matched under: lowercased,
    whitespace runs collapsed to one underscore. The judged scanners
    normalize every answer with it, so a declared closed label must
    already equal its key; `SubagentLabel` and `cohort_llm_scanner`
    refuse one that does not."""
    return "_".join(label.strip().lower().split())


def is_label_key(label: object) -> bool:
    """Whether ``label`` is a non-empty string equal to its own key."""
    return isinstance(label, str) and bool(label) and label_key(label) == label


def load_spec(path: Path | str) -> Spec:
    """Load a spec from JSON or YAML (same schema either way)."""
    return Spec.model_validate(yaml.safe_load(Path(path).read_text()))
