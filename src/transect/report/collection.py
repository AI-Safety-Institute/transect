"""Collection reliability from retained frames and original judge ballots.

Each scanner is partitioned by recorded judge setup and label vocabulary.
Missing setup evidence never becomes a pooled comparison group.
"""

from dataclasses import dataclass

import pandas as pd

from transect import reliability
from transect.frames import TransectResults
from transect.frames.common import JUDGE_COLS
from transect.frames.user import member_ballots
from transect.report.colors import cell_text_color, health_color
from transect.report.reliability import describe_regime


@dataclass
class _Surface:
    name: str
    vocabulary: str
    entity: pd.DataFrame
    units: pd.DataFrame
    votes: pd.DataFrame
    unit_col: str
    label_col: str
    phase: bool = False


def collection_reliability(results: TransectResults) -> list[dict]:
    """Summarize built-in and audit-declaring custom scanners without model calls.

    Counts of available runs include recorded units with no decided label.
    Setup scope counts include every run assigned to that setup, including
    records without retained judging units. Unknown setups are explicit.
    """
    rows = []
    requested = {s.scanner for s in results.scan_status.scanners}
    surfaces = []
    if len(results.phase_turns) or any("decision_phases" in s for s in requested):
        surfaces.append(
            _Surface(
                "Phases",
                "phases",
                results.phases,
                results.phase_turns,
                results.phase_turn_votes,
                "turn",
                "phase",
                phase=True,
            )
        )
    if results.subagents.judge_regime.notna().any() or any(
        "subagent_classification" in s for s in requested
    ):
        surfaces.append(
            _Surface(
                "Sub-agents",
                "subagents",
                results.subagents,
                results.subagents,
                results.subagent_votes,
                "agent_span_id",
                "label",
            )
        )
    for surface in surfaces:
        rows.extend(_groups(surface, results))
    for layer in results.extra_layers:
        if layer.audit is None:
            continue
        frame = results.layer_frames.get(layer.name)
        if frame is None:
            rows.append(_unavailable(layer.name, "No retained frame"))
            continue
        unit_col, label_col = layer.audit
        try:
            surface = _Surface(
                layer.name,
                layer.name,
                frame,
                frame,
                member_ballots(frame, unit_col, label_col),
                unit_col,
                label_col,
            )
            rows.extend(_groups(surface, results))
        except (AttributeError, KeyError, ValueError, TypeError) as error:
            rows.append(_unavailable(layer.name, f"Not audited: {error}"))
    return rows


def _unavailable(name: str, reason: str) -> dict:
    """Unavailable metrics have no invented denominators or scores."""
    return {"name": name, "available": False, "reason": reason}


def _groups(surface: _Surface, results: TransectResults) -> list[dict]:
    entity, units = surface.entity, surface.units
    columns = [*JUDGE_COLS, "judge_models", "verifier_model"]
    # Custom layers must identify their own units before collection pooling.
    for frame, required in (
        (entity, ["transcript_id", *columns]),
        (
            units,
            [
                "transcript_id",
                surface.unit_col,
                "sample_id",
                "task_set",
                "judge_agreement",
            ],
        ),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise KeyError(f"{surface.name}: missing columns {sorted(missing)}")
    tids = list(
        dict.fromkeys(
            [
                *results.transcript_info.transcript_id,
                *results.token_timeline.transcript_id,
                *units.transcript_id,
            ]
        )
    )
    groups: dict[tuple, list] = {}
    setups = {}
    for tid in tids:
        own = entity[(entity.transcript_id == tid) & entity.judge_regime.notna()]
        if len(own[[*JUDGE_COLS, "judge_models"]].drop_duplicates()) > 1:
            raise ValueError(f"Inconsistent judge setup within transcript {tid}")
        regime = reliability.detect_regime(own)
        definitions = results.label_definitions
        vocabulary = tuple(
            definitions.loc[
                (definitions.transcript_id == tid)
                & (definitions.surface == surface.vocabulary),
                "label",
            ].tolist()
        )
        # Missing setup evidence is kept per run if units exist, never combined.
        key = (
            (regime, vocabulary)
            if regime.kind != "none"
            else (None, tid if (units.transcript_id == tid).any() else None)
        )
        groups.setdefault(key, []).append(tid)
        setups[key] = regime
    rows = []
    for key, group in groups.items():
        scoped = units[units.transcript_id.isin(group)].reset_index(drop=True)
        if not len(scoped):
            rows.append(
                {
                    **_unavailable(surface.name, "No retained judging units"),
                    "scope": len(group),
                }
            )
            continue
        e = entity[entity.transcript_id.isin(group)]
        v = surface.votes[surface.votes.transcript_id.isin(group)]
        rows.append(_summarize(surface, e, scoped, v, setups[key], len(group)))
    return rows


def _summarize(
    surface: _Surface,
    entity: pd.DataFrame,
    units: pd.DataFrame,
    votes: pd.DataFrame,
    regime: reliability.Regime,
    scope: int,
) -> dict:
    keys = ["transcript_id", surface.unit_col]
    if units.duplicated(keys).any():
        raise ValueError(f"Duplicate {surface.name} judgment units")
    if votes.duplicated([*keys, "model", "roll"]).any():
        raise ValueError(f"Duplicate {surface.name} original ballots")
    eligible = (
        units[~units.basis.isin(["attributed", "unattributed"])].copy()
        if surface.phase
        else units.copy()
    )
    recorded = votes.merge(eligible[keys], on=keys, how="inner", validate="many_to_one")
    valid = recorded[recorded[surface.label_col].notna()].copy()
    if not surface.phase:
        valid = valid[valid.status == "ok"]
    counts = valid.groupby(keys, dropna=False).size()
    index = pd.MultiIndex.from_frame(eligible[keys])
    eligible.loc[
        counts.reindex(index, fill_value=0).to_numpy() < 2, "judge_agreement"
    ] = None
    mean = reliability.mean_stat(eligible.judge_agreement)
    balanced = reliability.equal_sample_agreement(eligible)
    cohort = reliability.cohort_agreement(valid, surface.unit_col, surface.label_col)
    reviews = reliability.review_units(entity)
    completed = reviews[reviews.verifier_completed] if len(reviews) else reviews
    relabelled = int(completed.overturned.fillna(False).sum()) if len(completed) else 0
    random = reliability.spot_check_overturns(entity)
    verifier = (
        None
        if regime.kind == "none"
        else {
            "on": regime.verifier_on,
            "selected": len(reviews),
            "completed": len(completed),
            "failed": len(reviews) - len(completed),
            "relabelled": relabelled,
            "random": random,
        }
    )
    status_col = "basis" if surface.phase else "status"
    has_ballots = len(recorded) > 0 or regime.kind in ("k_roll", "cohort")
    return {
        "name": surface.name,
        "available": True,
        "setup": describe_regime(regime)
        if regime.kind != "none"
        else "Setup unavailable",
        "records": int(units.transcript_id.nunique()),
        "scope": scope,
        "units": len(units),
        "eligible": len(eligible),
        "statuses": units[status_col].value_counts(dropna=False).to_dict(),
        "mean": mean,
        "balanced": balanced,
        "cohort": cohort,
        "paired_units": int((counts >= 2).sum()),
        "recorded": len(recorded) if has_ballots else None,
        "valid": len(valid) if has_ballots else None,
        "expected": len(eligible) * regime.n_models * regime.k_rolls
        if regime.kind != "none"
        else None,
        "verifier": verifier,
        "styles": {
            name: _cell_style(value)
            for name, value in {
                "mean": mean.mean,
                "balanced": balanced.mean,
                "pairwise": cohort.percent_agreement,
                "alpha": cohort.alpha,
                "ac1": cohort.ac1,
            }.items()
        },
    }


def _cell_style(value: float | None) -> str:
    """Use the detail audit's continuous ramp and contrast-aware text colour."""
    background = health_color(value)
    return f"background:{background};color:{cell_text_color(background)}"
