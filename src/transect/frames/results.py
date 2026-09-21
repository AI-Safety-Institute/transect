"""TransectResults: the return value of transect() and load()."""

from dataclasses import dataclass, field, fields

import pandas as pd

from transect.scan_status import ScanStatus


@dataclass
class TransectResults:
    """One Transect run: the frames + locations + report/viewer state.

    Per-column contracts live in the corresponding ``transect/frames/*.py``.
    """

    scan_location: str
    """This run's scan store: written by transect(), resolved by load()."""

    transcript_info: pd.DataFrame
    """One row per transcript: task, model, outcome, wall clock, and
    the run's setup facts (prompts, scaffold arguments, eval config)
    for the report's intro."""

    token_timeline: pd.DataFrame
    """One row per model turn: token usage, new work, context size."""

    flushes: pd.DataFrame
    """One row per context compaction: turn, kind, tokens before/after."""

    interventions: pd.DataFrame
    """One row per mid-run human intervention: turn, channel, content."""

    lane_activity: pd.DataFrame
    """One row per (turn, sub-agent span) of tool activity."""

    phases: pd.DataFrame
    """One row per phase: label, range, narration, reliability."""

    phase_turns: pd.DataFrame
    """One row per turn: the phase label and how it was decided."""

    turn_groups: pd.DataFrame
    """One row per narrated turn group within a phase."""

    phase_turn_votes: pd.DataFrame
    """One row per (judge member, turn): the member's own vote."""

    subagents: pd.DataFrame
    """One row per spawned sub-agent: classified role + span facts."""

    subagent_votes: pd.DataFrame
    """One row per (judge member, span): the member's own vote."""

    label_definitions: pd.DataFrame
    """One row per (surface, label): the rubric the judges classified
    against."""

    transcripts_location: str | None = None
    """Where the scan's transcripts live, the Scout viewer target."""

    scan_status: ScanStatus = field(default_factory=ScanStatus, kw_only=True)
    """Stored execution state and recorded coverage for the whole scan.

    This remains run-wide when rendering a selected epoch. Unknown custom
    content contracts do not imply either complete or failed semantic coverage.
    """

    layer_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    """User layers' pandas frames, keyed by layer name."""

    turn_tags: pd.DataFrame | None = None
    """The user layers' tag families: one row per (transcript_id, turn).
    Fed by ``Layer.tags`` over each layer's per-turn frame; the phase
    cards' chips, their filters, and the Token spend grouping read it."""

    extra_layers: list = field(default_factory=list)
    """The user layer declarations, as passed to ``extra_layers``."""

    report_paths: list[str] = field(default_factory=list)
    """Rendered report file(s); one per epoch when several rendered."""

    viewer_url: str | None = None
    """The spawned Scout viewer's URL; None when viewer=False."""

    def frames(self) -> dict[str, pd.DataFrame]:
        """The built-in DataFrame fields keyed by name (user layers
        live at ``layer_frames``, tag families at ``turn_tags``)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.type in (pd.DataFrame, "pd.DataFrame")
            and isinstance(getattr(self, f.name), pd.DataFrame)
        }


def builtin_frame_names() -> tuple[str, ...]:
    """The built-in frame field names."""
    frame_fields = tuple(
        f.name
        for f in fields(TransectResults)
        if f.type in (pd.DataFrame, "pd.DataFrame")
    )
    return (*frame_fields, "turn_tags", "extra_layers", "layer_frames")
