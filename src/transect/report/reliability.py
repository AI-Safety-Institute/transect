"""The report-specific reliability copy layer for the "Reliability &
provenance audit" section: flags with their heuristic thresholds and
remediation prose, confidence tiers, display formatting, and the
abstention counts. The statistics themselves live in
`transect.reliability` (the public analysis surface) and are re-exported
here with redundant aliases so `sections.py` keeps one `reliability.`
namespace for both layers.
"""

from dataclasses import dataclass

import pandas as pd

from transect.reliability import (
    NO_MEAN as NO_MEAN,
    CohortAgreement as CohortAgreement,
    LabelStats as LabelStats,
    Mean as Mean,
    MemberCoverage as MemberCoverage,
    Rate as Rate,
    Regime as Regime,
    RelabelRate as RelabelRate,
    cohort_agreement as cohort_agreement,
    detect_regime as detect_regime,
    label_stats as label_stats,
    mean_stat as mean_stat,
    member_coverage as member_coverage,
    provenance_shares as provenance_shares,
    relabel_rate as relabel_rate,
    spot_check_overturns as spot_check_overturns,
    wilson_interval as wilson_interval,
)
from transect.report.colors import _UNJUDGED_BASES

# The flagging policy: single source for the flag thresholds and the
# confidence-tier bounds.
AGREEMENT_RELIABLE = 0.80  # Krippendorff's conventional "reliable" band
AGREEMENT_TENTATIVE = 0.66  # below: unreliable; between the two: tentative
KROLL_SUSPICIOUSLY_HIGH = 0.95  # self-agreement so high it suggests anchoring
RELABEL_RATE_HIGH = 0.20  # verifier re-labels more than 1 in 5 examined


def describe_regime(regime: Regime) -> str:
    """One-line regime label for the setup block, e.g. "solo
    (claude-sonnet-4-6)", "solo + k-rolls (k=5, claude-opus-4-7)",
    "cohort of 3 models (a, b, c)", "no judge ran"."""
    if regime.kind == "none":
        return "no judge ran"
    if regime.kind == "cohort":
        return f"cohort of {regime.n_models} models ({', '.join(regime.models)})"
    name = f" ({regime.models[0]})" if regime.models else ""
    if regime.kind == "k_roll":
        return f"solo + k-rolls (k={regime.k_rolls}){name}"
    return f"solo{name}"


def format_confidence(stat: Mean) -> str:
    """Render a `Mean`; "no data" is honest, not an empty string,
    when N = 0."""
    if stat.n == 0 or stat.mean is None:
        return "no data"
    if stat.ci_half_width is None:
        return f"{stat.mean:.2f} (min {stat.minimum:.2f}), N = {stat.n}"
    return (
        f"{stat.mean:.2f} ± {stat.ci_half_width:.2f} "
        f"(min {stat.minimum:.2f}), N = {stat.n}"
    )


@dataclass(frozen=True)
class TierCounts:
    high: int
    medium: int
    low: int
    n: int


def confidence_tiers(
    values,
    high: float = AGREEMENT_RELIABLE,
    low: float = AGREEMENT_TENTATIVE,
) -> TierCounts:
    """Confidence-tier counts: high >= ``high``, low < ``low``, medium
    the remainder. Default bounds reuse the cohort-agreement bands."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = len(vals)
    high_n = int((vals >= high).sum())
    low_n = int((vals < low).sum())
    return TierCounts(high_n, n - high_n - low_n, low_n, n)


def abstention_counts(phase_turns: pd.DataFrame) -> dict[str, int]:
    """Per-basis counts over the abstention surface (refusal /
    no_answer / missing_turn - `colors._UNJUDGED_BASES`, the same set
    the phase band greys out). These three bases are pipeline-level
    fallbacks - the judge never delivered a verdict. The judge-stated
    escape is separate and already in every rubric (the reserved
    "none_of_the_above" label, with the judge's explanation); its
    concentration shows per label in the audit maps."""
    if not len(phase_turns):
        return dict.fromkeys(_UNJUDGED_BASES, 0)
    counts = phase_turns.basis.value_counts()
    return {basis: int(counts.get(basis, 0)) for basis in _UNJUDGED_BASES}


@dataclass(frozen=True)
class Flag:
    """One flagged metric, self-contained: a flag must explain what its
    threshold means and why crossing it matters."""

    metric: str
    value: str
    level: str  # "amber" | "red"
    explanation: str  # what crossing the threshold means, in plain words
    remediation: str  # what to do about it


def build_flags(
    regime: Regime,
    k_roll_stat: Mean,
    cohort: CohortAgreement,
    relabel: RelabelRate,
    spot_check: Rate,
) -> list[Flag]:
    """Flags for one judged entity, each self-contained: metric, value,
    a plain-language explanation of what its threshold means, and a
    remediation. A flag only renders when a metric actually crosses
    into amber/red.

    Thresholds are the module-level flagging-policy constants above,
    plus the thresholdless rule: any random-sample spot-check overturn
    flags red.
    """
    flags: list[Flag] = []
    if regime.kind == "k_roll" and k_roll_stat.n and k_roll_stat.mean is not None:
        if k_roll_stat.mean < AGREEMENT_RELIABLE:
            flags.append(
                Flag(
                    "k-roll self-consistency (mean per-turn agreement)",
                    f"{k_roll_stat.mean:.2f}",
                    "amber",
                    f"Flagged below {AGREEMENT_RELIABLE:.2f}, i.e. the same judge "
                    "model, asked to repeat "
                    "its own labelling several times over, agrees with itself on "
                    "fewer than 4 in 5 turns: unusually noisy for what should be a "
                    "repeatable process.",
                    "Check the judge's sampling settings first: lower temperature "
                    "or top_p where the model exposes them; reasoning models often "
                    "fix or ignore temperature and tune with reasoning effort "
                    "instead. If inconsistency remains, tighten the rubric's "
                    "definitions.",
                )
            )
        elif k_roll_stat.mean > KROLL_SUSPICIOUSLY_HIGH:
            flags.append(
                Flag(
                    "k-roll self-consistency (mean per-turn agreement)",
                    f"{k_roll_stat.mean:.2f}",
                    "amber",
                    f"Flagged above {KROLL_SUSPICIOUSLY_HIGH:.2f}: near-perfect "
                    "self-agreement. At "
                    "temperature 0 this is expected - repeat rolls near-replay "
                    "each other, so it says nothing about quality. With sampling "
                    "enabled, self-agreement this extreme can indicate a "
                    "rubric/vocabulary artifact rather than a genuinely "
                    "unambiguous transcript.",
                    "Run a scrambled-vocabulary check to rule out label-name "
                    "anchoring before leaning on this number.",
                )
            )
    if regime.kind == "cohort" and cohort.alpha is not None:
        if cohort.alpha < AGREEMENT_TENTATIVE:
            level = "red"
            explanation = (
                f"Flagged below {AGREEMENT_TENTATIVE:.2f}: judges disagree too "
                "often to treat these labels as consistent."
            )
        elif cohort.alpha < AGREEMENT_RELIABLE:
            level = "amber"
            explanation = (
                f"Flagged between {AGREEMENT_TENTATIVE:.2f} and "
                f"{AGREEMENT_RELIABLE:.2f}: borderline agreement, worth a "
                "look but not alarming on its own."
            )
        else:
            level = None
            explanation = ""
        if level:
            flags.append(
                Flag(
                    "cohort inter-judge agreement (Krippendorff's α)",
                    f"{cohort.alpha:.2f}",
                    level,
                    explanation,
                    "A larger cohort narrows the estimate's uncertainty but "
                    "rarely helps beyond ~5 members.",
                )
            )
    if relabel.overall.of > 0:
        if (
            relabel.overall.rate is not None
            and relabel.overall.rate >= RELABEL_RATE_HIGH
        ):
            remediation = (
                "The rubric may need tightening, starting with the most-relabelled "
                "classification."
            )
            if regime.verifier_same_model:
                remediation += (
                    " The verifier here is the same model as the judge, so this "
                    "rate reflects self-consistency under a second look rather "
                    "than an independent reviewer's disagreement: treat it as a "
                    "lower bound on what an independent check might find."
                )
            flags.append(
                Flag(
                    "verifier re-label rate (overall)",
                    f"{relabel.overall.rate:.0%} "
                    f"({relabel.overall.count} of {relabel.overall.of})",
                    "amber",
                    f"Flagged above {RELABEL_RATE_HIGH:.2f}, i.e. more than 1 "
                    "in 5 examined labels "
                    "overturned.",
                    remediation,
                )
            )
        for label, row in relabel.by_label.items():
            if row.rate is not None and row.rate >= RELABEL_RATE_HIGH:
                remediation = (
                    "This classification's definition may need tightening: it is "
                    "one of the ones most often relabelled by the verifier."
                )
                if regime.verifier_same_model:
                    remediation += (
                        " The verifier here is the same model as the judge, so "
                        "treat this as a self-consistency signal, not an "
                        "independent reviewer's disagreement."
                    )
                flags.append(
                    Flag(
                        f"verifier re-label rate ({label})",
                        f"{row.rate:.0%} ({row.count} of {row.of})",
                        "amber",
                        f"Flagged above {RELABEL_RATE_HIGH:.2f}, i.e. more than 1 "
                        "in 5 examined labels "
                        "originally classified this way were overturned.",
                        remediation,
                    )
                )
    if spot_check.count > 0:
        flags.append(
            Flag(
                "verifier spot-check overturns",
                f"{spot_check.count} of {spot_check.of} sampled",
                "red",
                "Flagged whenever this count is above zero: a random, not "
                "doubt-triggered, sample was overturned by the verifier, i.e. the "
                "judge got a randomly-picked case wrong without flagging its own "
                "doubt about it first. That is a sign of confident mislabelling, "
                "not just noisy uncertainty.",
                "Treat labels from this surface with extra caution; a "
                "planted-incorrect-label experiment can help confirm the "
                "verifier's catch rate.",
            )
        )
    return flags
