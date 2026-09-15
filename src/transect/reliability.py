"""Reliability statistics over the frames.

Everything here takes already-loaded frames (``transect.load(...).frames()``,
or plain values derived from them) and returns plain
dataclasses/numbers:

    import transect
    from transect.reliability import cohort_agreement

    frames = transect.load("scans/").frames()
    cohort_agreement(frames["phase_turn_votes"], "turn", "phase")
"""

from collections import Counter
from dataclasses import dataclass
from typing import get_args

import numpy as np
import pandas as pd

from transect.frames.common import JUDGE_COLS
from transect.scanners.cohort import LabelSource

__all__ = [
    "CohortAgreement",
    "LabelStats",
    "Mean",
    "MemberCoverage",
    "Rate",
    "Regime",
    "RelabelRate",
    "cohort_agreement",
    "detect_regime",
    "label_stats",
    "member_coverage",
    "relabel_rate",
    "spot_check_overturns",
    "wilson_interval",
]


@dataclass(frozen=True)
class Regime:
    """One judged surface's inter-rater setup, read off its frame's
    judge-identity columns by `detect_regime`.
    """

    kind: str
    """"none" (the judge scanner never ran, or ran with no judged
    rows) / "solo" (one model, one roll) / "k_roll" (one model, > 1
    roll) / "cohort" (> 1 distinct model)."""

    n_models: int
    """Distinct judge models that judged this surface."""

    k_rolls: int
    """Repeated rolls per model (1 outside the k-roll regime)."""

    models: tuple[str, ...]
    """The judge model names, as the frames record them."""

    verifier_on: bool
    """A second-round verifier was armed for this surface (whether or
    not it reviewed anything on this transcript)."""

    verifier_model: str | None
    """The verifier's model name; None when off or unrecorded."""

    verifier_same_model: bool
    """The verifier is the same model as the (lone) judge - its
    reviews are a self-consistency check, not an independent second
    opinion."""


NO_REGIME = Regime("none", 0, 1, (), False, None, False)


@dataclass(frozen=True)
class Rate:
    """A k-of-n proportion with its interval - the shape every
    counted reliability fact shares (re-labels, spot-checks, minority
    votes, member coverage)."""

    count: int
    """k - how many of the denominator hit."""

    of: int
    """n - the denominator."""

    rate: float | None
    """count / of; None when the denominator is zero."""

    ci: tuple[float, float] | None
    """Wilson 95% interval on the rate; None when the denominator is
    zero."""


NO_RATE = Rate(0, 0, None, None)


@dataclass(frozen=True)
class CohortAgreement:
    """Cohort inter-judge agreement over one judged surface's votes
    (`phase_turn_votes`/`subagent_votes`, one row per (unit, member))."""

    percent_agreement: float | None
    """Pooled pairwise agreement, chance-uncorrected; None when no
    unit has two or more raters."""

    alpha: float | None
    """Krippendorff's alpha (nominal) - reads low when one label
    dominates; None when expected disagreement is zero."""

    ac1: float | None
    """Gwet's AC1: chance-corrected agreement like alpha, but biased
    the opposite way when one label dominates (alpha reads too low
    there, AC1 too high) - read the two together as a bracket around
    the true agreement. None when only one category was ever used."""

    n: int
    """Ratings entering the computation (units with >= 2 raters)."""


@dataclass(frozen=True)
class Mean:
    """A mean with its interval - the "Confidence = X ± Y (min Z),
    N = A" display shape: X = mean, Y = 95% CI half-width, Z = min,
    A = N. ``ci_half_width`` is None whenever N < `_CI_MIN_N` makes
    the normal-approximation interval unreliable."""

    mean: float | None
    """The mean; None with no values."""

    ci_half_width: float | None
    """95% CI half-width (normal approximation); None below N=8,
    where it would overstate precision."""

    minimum: float | None
    """The smallest value observed."""

    n: int
    """Values entering the mean."""


NO_MEAN = Mean(None, None, None, 0)

# The ± is computed with the usual large-sample rule (z=1.96). On a
# handful of values that rule badly understates the real uncertainty,
# so showing a ± there would look more precise than the data is -
# worse than showing no interval at all.
_CI_MIN_N = 8


@dataclass(frozen=True)
class LabelStats:
    """One classification's reliability facts over the units it ended
    up deciding, plus the verifier's view."""

    label: str
    """The classification."""

    n: int
    """Decided units (e.g. turns/spans) carrying this label."""

    agreement: Mean
    """Per-unit vote agreement over its decided units."""

    confidence: Mean
    """Decided confidence over its decided units."""

    relabelled: Rate
    """Of the verifier-reviewed units whose original label was this
    one, how many the verifier re-labelled."""

    spot_checked: Rate
    """Of the random-sample verifier reviews with this original label,
    how many were re-labelled."""

    minority: Rate
    """Of the votes naming it on vote-decided units, how many lost
    (verifier-decided units are excluded)."""


@dataclass(frozen=True)
class RelabelRate:
    """Verifier re-label rates: the overall figure plus the
    per-original-label breakdown."""

    overall: Rate
    by_label: dict[str, Rate]
    """Keyed by the original (pre-verifier) label."""


@dataclass(frozen=True)
class MemberCoverage:
    """One cohort member's coverage facts."""

    model: str
    """The member's judge model name."""

    roll: int
    """The member's 0-based roll index."""

    coverage: Rate
    """Units where it produced a vote, of the units it was asked to
    judge."""

    misses: dict[str, int]
    """Miss reason -> count over the remainder."""


def rate(count: int, of: int) -> Rate:
    """A `Rate` from its counts: rate + Wilson 95% interval, both None
    when the denominator is zero."""
    if of <= 0:
        return Rate(count, of, None, None)
    return Rate(count, of, count / of, wilson_interval(count, of))


def detect_regime(frame: pd.DataFrame) -> Regime:
    """Read one judged surface's regime off its frame's judge-identity
    columns.

    The judged scanners stamp the regime facts at factory time
    (``value["judge"]``, stamped by `cohort_llm_scanner`) and the frames
    project them as the judge-identity columns (`judge_regime`,
    `n_models`, `k_rolls`, `verifier_armed`, `verifier_same_model`), so
    this is a read, not a heuristic.

    Args:
        frame: Any frame whose judged rows carry those columns plus
            ``judge_models`` and ``verifier_model`` -
            ``frames["phases"]`` and ``frames["subagents"]`` do; a
            custom frame conforms by projecting
            `transect.frames.common.judge_identity`.

    Returns:
        The `Regime`; NO_REGIME when the frame is empty or no row was
        judged. Raises KeyError on a frame without the columns,
        naming the missing ones and how to project them.
    """
    _require(
        frame,
        (*JUDGE_COLS, "judge_models", "verifier_model"),
        "project the five judge-identity columns with transect.judge_identity "
        "and record judge_models / verifier_model alongside them (a "
        "scanner built on transect.cohort_llm_scanner stamps the source "
        "block automatically)",
    )
    if not len(frame):
        return NO_REGIME
    judged = frame[frame["judge_regime"].notna()]
    if not len(judged):
        return NO_REGIME
    row = judged.iloc[0]
    joined = row.get("judge_models")
    models = tuple(str(joined).split("+")) if pd.notna(joined) else ()
    named = judged["verifier_model"].dropna()
    verifier_model = str(named.iloc[0]) if len(named) else None
    return Regime(
        str(row["judge_regime"]),
        int(row["n_models"]),
        int(row["k_rolls"]),
        models,
        bool(row["verifier_armed"]),
        verifier_model,
        bool(row["verifier_same_model"]),
    )


def cohort_agreement(
    votes: pd.DataFrame, unit_col: str, label_col: str
) -> CohortAgreement:
    """Cohort agreement coefficients over the per-member ballots.

    Args:
        votes: The per-member ballots, one row per (unit, member):
            ``frames["phase_turn_votes"]`` or
            ``frames["subagent_votes"]`` (any long-format frame with
            the two named columns works). Units are keyed per
            transcript when a ``transcript_id`` column is present
            (the frames carry one), so multi-transcript frames pool
            safely.
        unit_col: The judged-unit key: "turn" for phases,
            "agent_span_id" for sub-agents.
        label_col: The member's voted label: "phase" for phases,
            "label" for sub-agents.

    Returns:
        A `CohortAgreement` (percent agreement, alpha, AC1, n) over
        the units with two or more raters.
    """
    _require(
        votes,
        (unit_col, label_col),
        "cohort_agreement reads long-format ballots: one row per "
        "(unit, member) with the voted value",
    )
    units = _ratings_by_unit(votes, unit_col, label_col)
    percent, _ = percent_agreement_nominal(units)
    alpha, n = krippendorff_alpha_nominal(units)
    ac1, _ = gwet_ac1_nominal(units)
    return CohortAgreement(percent, alpha, ac1, n)


def label_stats(
    decided: pd.DataFrame,
    votes: pd.DataFrame,
    frame: pd.DataFrame,
    unit_col: str,
    label_col: str,
    vocabulary: list[str] | None = None,
) -> list[LabelStats]:
    """Per-classification stats for one judged surface.

    Args:
        decided: One row per decided unit, carrying ``label_col`` (the
            decided label), ``judge_agreement``, ``confidence`` and
            ``label_source``. Phases: ``frames["phase_turns"]``
            filtered to ``basis == "judged"``; sub-agents:
            ``frames["subagents"]`` filtered to rows with a non-null
            ``label``.
        votes: The per-member ballots, one row per (unit, member);
            ``label_col`` is the member's own label. Phases:
            ``frames["phase_turn_votes"]``; sub-agents:
            ``frames["subagent_votes"]``.
        frame: The verifier-bearing frame, unfiltered - read for
            ``original_label`` / ``verifier_reviewed`` / ``overturned``
            / ``verifier_trigger``. Phases: ``frames["phases"]``;
            sub-agents: ``frames["subagents"]``.
        unit_col: The unit key joining ``decided`` and ``votes``:
            "turn" for phases, "agent_span_id" for sub-agents. The
            join is scoped per transcript when both frames carry
            ``transcript_id`` (the frames do).
        label_col: The label column in ``decided`` and ``votes``:
            "phase" for phases, "label" for sub-agents.
        vocabulary: The declared label roster, in declared order.

    Returns:
        One `LabelStats` per label: the declared ``vocabulary`` in its
        declared order - a label no judgement ever used still gets a
        row (an unused label is information).
    """
    _require(
        decided,
        (unit_col, label_col, "judge_agreement", "confidence", "label_source"),
        "label_stats's decided frame carries the decided label per unit "
        "with judge_agreement, confidence and label_source",
    )
    _require(
        votes,
        (unit_col, label_col),
        "label_stats's votes frame carries one row per (unit, member) "
        "with the member's own label",
    )
    _require(
        frame,
        ("original_label", "verifier_reviewed", "overturned", "verifier_trigger"),
        "label_stats's frame argument carries the flattened verifier "
        "review columns a judged frame projects",
    )
    decided_of = decided[decided[label_col].notna()] if len(decided) else decided
    voted = votes[votes[label_col].notna()] if len(votes) else votes
    examined = frame[frame.verifier_reviewed.fillna(False)] if len(frame) else frame
    observed = sorted(
        {str(v) for v in (decided_of[label_col] if len(decided_of) else [])}
        | {str(v) for v in (voted[label_col] if len(voted) else [])}
        | {str(v) for v in (examined.original_label.dropna() if len(examined) else [])}
    )
    labels = list(vocabulary or [])
    labels += [label for label in observed if label not in labels]
    dissent = pd.DataFrame()
    if len(voted) and len(decided_of):
        # verifier-decided units are excluded from the minority join
        vote_decided = decided_of[decided_of.label_source != "verifier"]
        key = [c for c in _unit_key(voted, unit_col) if c in vote_decided.columns]
        dissent = voted.merge(
            vote_decided[[*key, label_col]].rename(columns={label_col: "_decided"}),
            on=key,
        )
    out = []
    for label in labels:
        mine = decided_of[decided_of[label_col] == label]
        examined_mine = (
            examined[examined.original_label == label] if len(examined) else examined
        )
        sampled = (
            examined_mine[examined_mine.verifier_trigger == "random_sample"]
            if len(examined_mine)
            else examined_mine
        )
        named = dissent[dissent[label_col] == label] if len(dissent) else dissent
        out.append(
            LabelStats(
                label=label,
                n=len(mine),
                agreement=mean_stat(mine.judge_agreement if len(mine) else []),
                confidence=mean_stat(mine.confidence if len(mine) else []),
                relabelled=rate(
                    int(examined_mine.overturned.fillna(False).sum())
                    if len(examined_mine)
                    else 0,
                    len(examined_mine),
                ),
                spot_checked=rate(
                    int(sampled.overturned.fillna(False).sum()) if len(sampled) else 0,
                    len(sampled),
                ),
                minority=rate(
                    int((named[label_col] != named._decided).sum())
                    if len(named)
                    else 0,
                    len(named),
                ),
            )
        )
    return out


def relabel_rate(frame: pd.DataFrame, label_col: str = "original_label") -> RelabelRate:
    """Verifier re-label rate = overturned / examined.

    Args:
        frame: The verifier-bearing frame: ``frames["phases"]`` or
            ``frames["subagents"]`` (columns ``verifier_reviewed``,
            ``overturned``, ``label_col``).
        label_col: The pre-verifier label column for the by-label
            breakdown; the default is the frames' own
            ``original_label``.

    Returns:
        A `RelabelRate`: the overall `Rate` plus one per original
        label.
    """
    _require(
        frame,
        ("verifier_reviewed", "overturned", label_col),
        "relabel_rate reads the flattened verifier review columns a "
        "judged frame projects",
    )
    if not len(frame):
        return RelabelRate(NO_RATE, {})
    examined = frame[frame.verifier_reviewed.fillna(False)]
    if not len(examined):
        return RelabelRate(NO_RATE, {})
    overall = rate(int(examined.overturned.fillna(False).sum()), len(examined))
    by_label = {
        str(label): rate(int(group.overturned.fillna(False).sum()), len(group))
        for label, group in examined.groupby(label_col, sort=False)
    }
    return RelabelRate(overall, dict(sorted(by_label.items())))


def spot_check_overturns(frame: pd.DataFrame) -> Rate:
    """Count of verifier reviews triggered by a random sample (not the
    judge's own stated doubt) that overturned the label - a
    confidently-wrong judge caught by chance rather than by its own
    uncertainty.

    Args:
        frame: The verifier-bearing frame: ``frames["phases"]`` or
            ``frames["subagents"]`` (columns ``overturned``,
            ``verifier_trigger``).

    Returns:
        A `Rate`: overturned of randomly-sampled reviews.
    """
    _require(
        frame,
        ("overturned", "verifier_trigger"),
        "spot_check_overturns reads the flattened verifier review "
        "columns a judged frame projects",
    )
    if not len(frame):
        return NO_RATE
    overturned = frame.overturned.fillna(False).astype(bool)
    sampled = frame.verifier_trigger == "random_sample"
    return rate(int((overturned & sampled).sum()), int(sampled.sum()))


def member_coverage(
    votes: pd.DataFrame, ok_col: str, ok_value: str, reasons: tuple[str, ...]
) -> list[MemberCoverage]:
    """Per-member coverage: rows produced of asked, with per-reason
    miss counts.

    Args:
        votes: The per-member ballots:
            ``frames["phase_turn_votes"]`` or
            ``frames["subagent_votes"]``.
        ok_col: The column holding each row's produced/missed state:
            "basis" for phases, "status" for sub-agents.
        ok_value: The ``ok_col`` value that counts as produced:
            "judged" for phases, "ok" for sub-agents.
        reasons: The miss reasons to count, fixed so every member
            carries the same set (phases: refusal / no_answer /
            missing_turn / filled; sub-agents: refusal / error).

    Returns:
        One `MemberCoverage` per (model, roll).
    """
    _require(
        votes,
        ("model", "roll", ok_col),
        "member_coverage reads a per-member frame with one row per (model, roll) unit",
    )
    if not len(votes):
        return []
    out = []
    for (model, roll), sub in votes.groupby(["model", "roll"]):
        produced = int((sub[ok_col] == ok_value).sum())
        missed = sub[sub[ok_col] != ok_value]
        counts = missed[ok_col].value_counts()
        out.append(
            MemberCoverage(
                model=str(model),
                roll=int(str(roll)),
                coverage=rate(produced, len(sub)),
                misses={reason: int(counts.get(reason, 0)) for reason in reasons},
            )
        )
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion k/n.

    Args:
        k: Successes.
        n: Trials.
        z: The normal quantile; the default 1.96 is the 95% interval.

    Returns:
        ``(low, high)``, or ``None`` when n == 0 (no interval without
        an observation). Bounds are clamped to [0, 1]."""
    if n <= 0:
        return None
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return lower, upper


def mean_stat(values) -> Mean:
    """Mean / 95% CI half-width / min / N over a confidence-like
    column. Works equally for a phase's per-phase `confidence` column
    and a judged surface's per-unit `judge_agreement` column.

    The CI is omitted (not computed at all) below N = `_CI_MIN_N`."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = len(vals)
    if n == 0:
        return NO_MEAN
    mean = float(vals.mean())
    minimum = float(vals.min())
    ci = None
    if n >= _CI_MIN_N:
        std = float(vals.std(ddof=1))
        ci = 1.96 * std / (n**0.5)
    return Mean(mean, ci, minimum, n)


def provenance_shares(
    decided: pd.DataFrame, label_col: str, labels: list[str]
) -> dict[str, dict[str, int]]:
    """Per deciding label source, the count of each classification's
    decided units it decided: {source: {label: count}} over the
    LabelSource vocabulary, in ``labels`` order."""
    _require(
        decided,
        (label_col, "label_source"),
        "provenance_shares reads the decided rows' label and their "
        "deciding label_source",
    )
    sources = list(get_args(LabelSource))
    out: dict[str, dict[str, int]] = {
        source: dict.fromkeys(labels, 0) for source in sources
    }
    if not len(decided):
        return out
    rows = decided[decided[label_col].notna() & decided.label_source.notna()]
    for label, source in zip(rows[label_col], rows.label_source, strict=True):
        if str(source) in out and str(label) in out[str(source)]:
            out[str(source)][str(label)] += 1
    return out


def percent_agreement_nominal(units: list[list]) -> tuple[float | None, int]:
    """Average pairwise agreement over units with >= 2 raters: agreeing
    rater-pairs / total rater-pairs, pooled globally (not per-unit
    averaged) across every pairable unit. A plain, chance-uncorrected
    companion to `krippendorff_alpha_nominal`. Returns ``(None, 0)``
    when no unit has >= 2 raters."""
    pairable = [u for u in units if len(u) >= 2]
    if not pairable:
        return None, 0
    agree = total = 0
    n = 0
    for u in pairable:
        counts = Counter(u)
        m = len(u)
        n += m
        total += m * (m - 1) // 2
        agree += sum(c * (c - 1) // 2 for c in counts.values())
    return (agree / total if total else None), n


def krippendorff_alpha_nominal(units: list[list]) -> tuple[float | None, int]:
    """Krippendorff's alpha (nominal metric), via the standard
    coincidence-matrix construction (Krippendorff, 2004).

    Returns ``(None, n)`` when the expected disagreement is zero."""
    pairable = [u for u in units if len(u) >= 2]
    if not pairable:
        return None, 0
    values = sorted({v for u in pairable for v in u}, key=str)
    index = {v: i for i, v in enumerate(values)}
    v_n = len(values)
    o = np.zeros((v_n, v_n))
    n = 0
    for u in pairable:
        m = len(u)
        n += m
        counts = Counter(u)
        for v, nv in counts.items():
            for vp, nvp in counts.items():
                weight = nv * (nv - 1) if v == vp else nv * nvp
                o[index[v], index[vp]] += weight / (m - 1)
    n_v = o.sum(axis=1)
    off_diag = ~np.eye(v_n, dtype=bool)
    do = o[off_diag].sum() / n
    de_numerator = (n_v[:, None] * n_v[None, :])[off_diag].sum()
    de = de_numerator / (n * (n - 1))
    if de == 0:
        return None, n
    return 1.0 - do / de, n


def gwet_ac1_nominal(units: list[list]) -> tuple[float | None, int]:
    """Gwet's AC1 (nominal): pa the pooled pairwise agreement over
    units with >= 2 raters (`percent_agreement_nominal`), pe the
    chance agreement from the pooled category prevalences,
    pe = sum_q pi_q (1 - pi_q) / (Q - 1); AC1 = (pa - pe) / (1 - pe).

    Returns ``(None, n)`` with a single observed category (agreement
    is undefined, not perfect) or with no pairable unit."""
    pairable = [u for u in units if len(u) >= 2]
    if not pairable:
        return None, 0
    pa, n = percent_agreement_nominal(units)
    ratings = [r for u in pairable for r in u]
    counts = Counter(ratings)
    total = len(ratings)
    q = len(counts)
    if q < 2 or pa is None:
        return None, n
    # pe <= 1/q < 1 for q >= 2, so no division-by-zero guard is needed
    pe = sum((c / total) * (1 - c / total) for c in counts.values()) / (q - 1)
    return (pa - pe) / (1 - pe), n


def _require(frame: pd.DataFrame, columns: tuple, hint: str) -> None:
    """Fail when a frame lacks the columns a statistic reads."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing the columns {missing}; {hint}")


def _ratings_by_unit(votes: pd.DataFrame, unit_col: str, label_col: str) -> list[list]:
    """Group the ballots by unit: one list per unit, each
    holding that unit's member ratings. Shared input shape
    for `percent_agreement_nominal` / `krippendorff_alpha_nominal`.
    Units are keyed per transcript when the frame carries a
    ``transcript_id`` column."""
    if not len(votes):
        return []
    return [
        list(group[label_col].dropna())
        for _, group in votes.groupby(_unit_key(votes, unit_col), sort=False)
    ]


def _unit_key(frame: pd.DataFrame, unit_col: str) -> list[str]:
    """The judged-unit grouping/join key: ``unit_col``, scoped per
    transcript when the frame carries ``transcript_id``."""
    return [c for c in ("transcript_id", unit_col) if c in frame.columns]
