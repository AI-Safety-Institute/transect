"""phases_df: one row per stitched decision phase.

Columns (identity prefix explained in common.py):

- phase_index: the phase's position within its transcript - the
  join key for phase_turns_df / turn_groups_df / phase_turn_votes_df.
- phase: the label.
- turn_start / turn_end: inclusive turn range (0-based).
- n_turns: reasoning-bearing turns inside the phase - the turns
  the judge actually saw (tool-call-only, failed, and sub-agent
  turns in range are not counted).
- confidence: mean of the phase's per-turn confidences (filled
  turns included, at 0.3).
- min_confidence: minimum per-turn confidence in the phase.
- min_agreement: minimum per-turn vote agreement (voting regimes).
- explanation: the first contributing segment's explanation.
- verifier_reviewed: legacy representative review-record flag for this phase.
- verifier_completed: that representative review returned a usable verdict.
  Original-unit counts must use verifier_reviews, not these display flags.
- verifier_reviews: list of original phase review units, with their original
  phase index, turn range and nested review fields; [] means none selected.
  Use these for review counts; the flattened verifier fields describe
  only a representative review.
- overturned: the verifier relabelled it.
- verifier_trigger / verifier_label / verifier_confidence /
  verifier_explanation / verifier_status / original_label /
  original_confidence / original_explanation: the flattened
  VerifierReview record (common.VERIFIER_COLS; NaN when never
  reviewed).
- headline / summary: complete narrator output (blank headline uses a template).
- narration_group_status: accepted (complete partition, not factual validation),
  invalid_partition / empty_groups / no_narrative (neutral grouping), or not_run.
  Missing historical status is unknown; it is not inferred from the prose.
- anchor_event_id: Scout-viewer deep-link anchor.
- judge_models: the judge model(s) that segmented the run, as one
  "+"-joined string (one name solo; the cohort's distinct models
  joined; a k-roll judge appears once).
- verifier_model: the verifier's resolved model name.
- n_members: voting members (models x rolls; voting regimes).
- verifier_n_low_confidence / _n_low_agreement / _n_wedge /
  _n_random_sample / _n_no_verdict / _n_relabelled /
  _n_weak_relabel / _n_random_sample_relabelled: the verifier's own
  exact selection/outcome counts (the scanner's audit block),
  run-constant per transcript; NaN when the verifier never ran.
  Unlike row aggregation these include reviews with no verdict and
  relabels merged away by re-stitching.
- judge_regime / n_models / k_rolls / verifier_armed /
  verifier_same_model: the scanner-stamped judge-identity columns
  (common.JUDGE_COLS), constant across the run.
- judge_agreement: mean per-turn vote agreement over the range.
- confidence_spread: 95%-CI half-width of the phase's per-turn
  confidences.
- confidence_source: who the confidence describes - single_judge /
  majority_vote / verifier.
- new_work_tokens: per-phase spend - token_timeline ``new_work``
  summed over the turns assigned to the phase.
- schema_version: the frames contract version.
"""

from typing import get_args

import pandas as pd

from transect.frames.common import (
    CALL_STATUSES,
    IDENTITY_COLS,
    JUDGE_COLS,
    VERIFIER_COLS,
    categorical,
    identity,
    joined_judge_models,
    judge_identity,
    result_value,
    verifier_review,
    with_schema,
)
from transect.scanners.phases_common import NarrationGroupStatus

# VerifierAudit count fields (scanners/phases_verify.py)
_AUDIT_COUNTS = (
    "n_low_confidence",
    "n_low_agreement",
    "n_wedge",
    "n_random_sample",
    "n_no_verdict",
    "n_relabelled",
    "n_weak_relabel",
    "n_random_sample_relabelled",
)


def phases_df(
    results: pd.DataFrame,
    phase_turns: pd.DataFrame | None = None,
    token_timeline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """decision_phases results -> one row per stitched phase.

    ``phase_index`` is the phase's position within its transcript.
    ``phase_turns`` + ``token_timeline`` (both optional) feed the
    ``new_work_tokens`` rollup; without them the column is NaN."""
    rows = []
    vocabulary: list[str] = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        audit = value.get("verifier") or {}
        verifier_model = audit.get("verifier_model")
        # the verifier's own exact selection/outcome counts, run-constant
        # per transcript; None (not 0) when the verifier pass never ran
        audit_counts = {
            f"verifier_{k}": (audit.get(k) if audit.get("ran") else None)
            for k in _AUDIT_COUNTS
        }
        joined = joined_judge_models(value)
        agreement_block = (value.get("cohort") or {}).get("agreement") or {}
        narrator_model = (value.get("narrator") or {}).get("narrator_model")
        for name in value.get("phase_names") or []:
            if name not in vocabulary:
                vocabulary.append(name)
        for index, phase in enumerate(value.get("phases") or []):
            # flatten the nested VerifierReview into the frame columns;
            # every judged statistic arrives precomputed on the phase
            # (computed where the judging happened - frames project)
            phase = dict(phase)
            review = phase.pop("verifier", None) or {}
            rows.append(
                {
                    **identity_cols,
                    "phase_index": index,
                    **phase,
                    "verifier_reviews": phase.get("verifier_reviews"),
                    **verifier_review(review),
                    "judge_models": joined,
                    "verifier_model": verifier_model,
                    "narrator_model": narrator_model,
                    "n_members": agreement_block.get("n_members"),
                    **judge_identity(value),
                    **audit_counts,
                }
            )
    columns = [
        *IDENTITY_COLS,
        "phase_index",
        "phase",
        "turn_start",
        "turn_end",
        "n_turns",
        "confidence",
        "min_confidence",
        "min_agreement",
        "explanation",
        "verifier_reviews",
        *VERIFIER_COLS,
        "headline",
        "summary",
        "narration_group_status",
        "anchor_event_id",
        "judge_models",
        "verifier_model",
        "narrator_model",
        "n_members",
        *JUDGE_COLS,
        *(f"verifier_{k}" for k in _AUDIT_COUNTS),
        "judge_agreement",
        "confidence_spread",
        "confidence_source",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "min_agreement": "float64",
            "judge_agreement": "float64",
            "verifier_confidence": "float64",
            "original_confidence": "float64",
            "confidence_spread": "float64",
            "n_members": "Int64",
            "n_models": "Int64",
            "k_rolls": "Int64",
            "verifier_armed": "boolean",
            "verifier_same_model": "boolean",
            **{f"verifier_{k}": "Int64" for k in _AUDIT_COUNTS},
        }
    )
    df["phase"] = categorical(df.phase, vocabulary)
    df["narration_group_status"] = categorical(
        df.narration_group_status, list(get_args(NarrationGroupStatus))
    )
    df["verifier_status"] = categorical(df.verifier_status, CALL_STATUSES)
    df["new_work_tokens"] = _new_work_rollup(df, phase_turns, token_timeline)
    return with_schema(df)


def _new_work_rollup(
    phases: pd.DataFrame,
    phase_turns: pd.DataFrame | None,
    token_timeline: pd.DataFrame | None,
) -> pd.Series:
    """Dense-attributed per-phase spend: token_timeline ``new_work``
    summed over the turns the dense map assigns to each phase (all
    lanes; tool-only turns included). NaN without the inputs."""
    if (
        phase_turns is None
        or token_timeline is None
        or not len(phase_turns)
        or "new_work" not in token_timeline.columns
    ):
        return pd.Series(pd.NA, index=phases.index, dtype="Float64")
    per_turn = token_timeline.groupby(
        ["transcript_id", "turn"], as_index=False
    ).new_work.sum()
    rollup = (
        phase_turns[["transcript_id", "turn", "phase_index"]]
        .merge(per_turn, on=["transcript_id", "turn"], how="left")
        .dropna(subset=["phase_index"])
        .groupby(["transcript_id", "phase_index"], as_index=False)
        .new_work.sum(min_count=1)
    )
    out = phases[["transcript_id", "phase_index"]].merge(
        rollup, on=["transcript_id", "phase_index"], how="left"
    )
    return out.new_work.set_axis(phases.index).astype("Float64")
