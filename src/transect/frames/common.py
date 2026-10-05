"""Shared resources for the frame builders.

Bump SCHEMA_VERSION on any breaking column change.

Identity: every frame carries the same five-column prefix, mapped
from Scout's result columns by IDENTITY_COLS -

- sample_id: the logical dataset item (Scout task_id); shared across
  epochs and task_sets of the same sample.
- task_set: the eval run/task the transcript came from (Scout
  task_set; one value per eval log.
- epoch: 1-based repeat index (Scout task_repeat).
- transcript_id: the physical run - unique per (sample, epoch,
  task_set); the join key across frames and the viewer deep-link id.
- agent: the harness/agent that produced the run.
"""

import json
import sys
from typing import get_args

import pandas as pd

from transect.scanners.cohort import CallStatus, LabelSource, VerifierReview
from transect.scanners.phases_common import TurnBasis

SCHEMA_VERSION = "0.2"

IDENTITY_COLS = {
    "sample_id": "transcript_task_id",
    "task_set": "transcript_task_set",
    "epoch": "transcript_task_repeat",
    "transcript_id": "transcript_id",
    "agent": "transcript_agent",
}

# the standard judged-scanner identity columns
JUDGE_COLS = (
    "judge_regime",
    "n_models",
    "k_rolls",
    "verifier_armed",
    "verifier_same_model",
)

# enumerated-column vocabularies, derived from the scanners' own Literals
LABEL_SOURCES = list(get_args(LabelSource))
BASES = list(get_args(TurnBasis))
CALL_STATUSES = list(get_args(CallStatus))

# The flattened VerifierReview columns (verifier_model is excluded)
_REVIEW_RENAMES = {"trigger": "verifier_trigger", "status": "verifier_status"}
VERIFIER_COLS = (
    "verifier_selected",
    "verifier_completed",
    *(
        _REVIEW_RENAMES.get(name, name)
        for name in VerifierReview.model_fields
        if name != "verifier_model"
    ),
)


def identity(row) -> dict:
    """The four identity columns extracted from a Scout results row."""
    return {ours: row.get(theirs) for ours, theirs in IDENTITY_COLS.items()}


def with_schema(df: pd.DataFrame) -> pd.DataFrame:
    df["schema_version"] = SCHEMA_VERSION
    return df


def result_value(cell) -> dict:
    """Decode a scanner Result value cell (dict, JSON string, or junk) to a dict."""
    if isinstance(cell, str):
        cell = json.loads(cell)
    return cell if isinstance(cell, dict) else {}


def item_metadata(row) -> dict:
    """The loader item's ``Result.metadata`` for a results row."""
    return result_value(row.get("metadata"))


def input_id(row) -> str | None:
    """The row's loader-item id (the store's ``input_ids`` column)."""
    ids = row.get("input_ids")
    if isinstance(ids, str):
        ids = json.loads(ids)
    if isinstance(ids, (list, tuple)) and len(ids):
        return str(ids[0])
    return None


def lane_series(per_turn: pd.DataFrame) -> pd.Series:
    """Lane key per row: agent_span_id; a span-less orchestrator turn is
    the main lane; a span-less off-axis call (turn NA: an init or scorer
    call) is its own one-row lane, so its window never seeds another
    conversation's derivations. The shared lane-identity definition."""
    if "agent_span_id" not in per_turn.columns:
        return pd.Series("__main__", index=per_turn.index)
    key = per_turn.agent_span_id.astype(object).copy()
    spanless = key.isna()
    if "turn" in per_turn.columns:
        off_axis = spanless & per_turn.turn.isna()
        key[off_axis] = [f"__offaxis_{index}" for index in per_turn.index[off_axis]]
        spanless = spanless & ~off_axis
    key[spanless] = "__main__"
    return key


def judge_identity(value: dict) -> dict:
    """Project ``value["judge"]`` to the `JUDGE_COLS` dict."""
    judge = value.get("judge")
    if not isinstance(judge, dict):
        raise KeyError(
            "scan value carries no 'judge' block, so the judge-identity "
            "columns cannot be built; re-scan the transcripts, or build "
            "the custom scanner on transect.cohort_llm_scanner (which stamps "
            "the block automatically)"
        )
    return {
        "judge_regime": judge.get("regime"),
        "n_models": judge.get("n_models"),
        "k_rolls": judge.get("k_rolls"),
        "verifier_armed": judge.get("verifier_armed"),
        "verifier_same_model": judge.get("verifier_same_model"),
    }


def verifier_review(review: dict) -> dict:
    """A review record ({} when no review ran) flattened to the
    `VERIFIER_COLS` cells. ``verifier_selected`` means a review record
    exists (failed attempts included); ``verifier_completed`` means it
    carries a usable verdict."""
    label = review.get("verifier_label")
    completed = (
        bool(review)
        and review.get("status") in (None, "ok")
        and isinstance(label, str)
        and bool(label.strip())
    )
    row: dict = {
        "verifier_selected": bool(review),
        "verifier_completed": completed,
    }
    for name in VerifierReview.model_fields:
        if name != "verifier_model":
            row[_REVIEW_RENAMES.get(name, name)] = review.get(name)
    row["overturned"] = bool(review.get("overturned"))
    return row


def joined_judge_models(value: dict) -> str | None:
    """The judge models as a "+"-joined string (hashable for groupby)."""
    judge = value.get("judge")
    stamped = judge.get("models") if isinstance(judge, dict) else None
    models = value.get("judge_models") or stamped or []
    return "+".join(models) if models else None


def mean(values: list) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def categorical(series: pd.Series, categories: list[str]) -> pd.Series:
    """An enumerated column as orderless Categorical with the declared
    vocabulary."""
    unknown = sorted(set(series.dropna()) - set(categories))
    if unknown:
        print(
            f"WARNING: values outside the declared vocabulary {unknown} "
            f"(declared: {categories}); coerced to NaN",
            file=sys.stderr,
        )
    return pd.Series(pd.Categorical(series, categories=categories), index=series.index)
