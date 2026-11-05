"""Collection reliability for the run index: each judged surface of a
multi-epoch scan, pooled per recorded judge setup and label vocabulary.

The per-unit agreement is the frames' own ``judge_agreement`` (the
scanner's decision), pooled across transcripts; the pairwise, alpha and
AC1 coefficients come from the original ballots with transcript-scoped
unit keys; verifier counts come from the pooled review units. A surface
whose frames fail their contract gets a row saying why instead of the
index failing.
"""

from dataclasses import dataclass

import pandas as pd

from transect import reliability
from transect.frames import TransectResults
from transect.frames.common import JUDGE_COLS
from transect.frames.user import member_ballots
from transect.report.colors import cell_text_color, health_color
from transect.report.reliability import UNSEEN_BASES, describe_regime


@dataclass(frozen=True)
class _Surface:
    """One judged surface: its entity frame (judge identity, reviews),
    its unit frame (per-unit agreement) and its ballots."""

    name: str
    vocabulary: str
    entity: pd.DataFrame
    units: pd.DataFrame
    votes: pd.DataFrame
    unit_col: str
    label_col: str


def collection_reliability(results: TransectResults) -> list[dict]:
    """One row per judged surface and judge setup of the scan.

    The built-in surfaces are those whose scanner the scan ran; a custom
    layer joins when it declares an audit. Transcripts group by their
    recorded judge setup and label vocabulary, and each group pools its
    units and ballots; the transcripts without a recorded setup form one
    group, which says so when it holds no judged units.
    """
    ran = {status.scanner for status in results.scan_status.scanners}
    rows: list[dict] = []
    surfaces: list[_Surface] = []
    if "decision_phases" in ran:
        surfaces.append(
            _Surface(
                "Phases",
                "phases",
                results.phases,
                results.phase_turns,
                results.phase_turn_votes,
                "turn",
                "phase",
            )
        )
    if "subagent_classification" in ran:
        # only sub-agent lanes are classified; role lanes carry no label
        spans = results.subagents[results.subagents.lane_kind == "subagent"]
        surfaces.append(
            _Surface(
                "Sub-agents",
                "subagents",
                spans,
                spans,
                results.subagent_votes,
                "agent_span_id",
                "label",
            )
        )
    for layer in results.extra_layers:
        if layer.audit is None:
            continue
        frame = results.layer_frames[layer.name]
        unit_col, label_col = layer.audit
        try:
            votes = member_ballots(frame, unit_col, label_col)
        except (AttributeError, KeyError, ValueError, TypeError) as error:
            rows.append(_unavailable(layer.name, f"not audited: {error}"))
            continue
        surfaces.append(
            _Surface(layer.name, layer.name, frame, frame, votes, unit_col, label_col)
        )
    for surface in surfaces:
        try:
            rows.extend(_groups(surface, results))
        except ValueError as error:
            rows.append(_unavailable(surface.name, f"not audited: {error}"))
    return rows


def _unavailable(name: str, reason: str) -> dict:
    """A row with no numbers: the reason, never invented denominators."""
    return {"name": name, "available": False, "reason": reason}


def _groups(surface: _Surface, results: TransectResults) -> list[dict]:
    """The surface's rows: its transcripts grouped by judge setup and
    vocabulary, each group summarised over its pooled frames; a group
    without a recorded setup, or without judged units, states that."""
    for frame, required in (
        (surface.entity, ("transcript_id", "judge_regime")),
        (surface.units, ("transcript_id", surface.unit_col, "judge_agreement")),
    ):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"missing columns {missing}")
    definitions = results.label_definitions
    groups: dict[tuple, list[str]] = {}
    regimes: dict[tuple, reliability.Regime] = {}
    for transcript_id in results.transcript_info.transcript_id:
        entity = surface.entity
        own = entity[
            (entity.transcript_id == transcript_id) & entity.judge_regime.notna()
        ]
        if len(own[[*JUDGE_COLS, "judge_models"]].drop_duplicates()) > 1:
            raise ValueError(f"several judge setups within transcript {transcript_id}")
        regime = reliability.detect_regime(own)
        vocabulary = frozenset(
            definitions.loc[
                (definitions.transcript_id == transcript_id)
                & (definitions.surface == surface.vocabulary),
                "label",
            ]
        )
        key = (regime, vocabulary) if regime.kind != "none" else (None, None)
        groups.setdefault(key, []).append(transcript_id)
        regimes[key] = regime
    rows = []
    for key, members in groups.items():
        units = surface.units[surface.units.transcript_id.isin(members)]
        if not len(units) or regimes[key].kind == "none":
            reason = "no judged units" if not len(units) else "no judge setup recorded"
            rows.append({**_unavailable(surface.name, reason), "scope": len(members)})
            continue
        entity = surface.entity[surface.entity.transcript_id.isin(members)]
        votes = surface.votes[surface.votes.transcript_id.isin(members)]
        rows.append(
            _summarize(surface, entity, units, votes, regimes[key], len(members))
        )
    return rows


def _summarize(
    surface: _Surface,
    entity: pd.DataFrame,
    units: pd.DataFrame,
    votes: pd.DataFrame,
    regime: reliability.Regime,
    scope: int,
) -> dict:
    """One group's row from its pooled frames; the agreement numbers
    exist only for a voting setup (k-rolls or a cohort)."""
    keys = ["transcript_id", surface.unit_col]
    if units.duplicated(keys).any():
        raise ValueError(f"duplicate {surface.name} judgement units")
    if votes.duplicated([*keys, "model", "roll"]).any():
        raise ValueError(f"duplicate {surface.name} original ballots")
    # the units judging attempted: phase turns the judge saw, every span
    seen = units[~units.basis.isin(UNSEEN_BASES)] if "basis" in units.columns else units
    reviews = reliability.review_units(entity)
    relabelled = reliability.relabel_rate(entity).overall
    row: dict = {
        "name": surface.name,
        "available": True,
        "setup": describe_regime(regime),
        "voting": regime.kind in ("k_roll", "cohort"),
        "scope": scope,
        "seen": len(seen),
        "verifier": {
            "on": regime.verifier_on,
            "selected": len(reviews),
            "completed": relabelled.of,
            "relabelled": relabelled.count,
            "random": reliability.spot_check_overturns(entity),
        },
    }
    if not row["voting"]:
        return row
    ballots = votes.merge(seen[keys], on=keys, validate="many_to_one")
    ballots = ballots[ballots[surface.label_col].notna()]
    if "status" in ballots.columns:
        ballots = ballots[ballots.status == "ok"]
    mean = reliability.mean_stat(seen.judge_agreement)
    cohort = reliability.cohort_agreement(ballots, surface.unit_col, surface.label_col)
    row.update(
        ballots=len(ballots),
        expected=len(seen) * regime.n_models * regime.k_rolls,
        mean=mean,
        cohort=cohort,
        styles={
            name: _cell_style(value)
            for name, value in {
                "mean": mean.mean,
                "pairwise": cohort.percent_agreement,
                "alpha": cohort.alpha,
                "ac1": cohort.ac1,
            }.items()
        },
    )
    return row


def _cell_style(value: float | None) -> str:
    """The per-run audit's colour ramp with contrast-aware text."""
    background = health_color(value)
    return f"background:{background};color:{cell_text_color(background)}"
