"""Frame builders for user layers.

User frames are plain pandas with the identity prefix.

- `turns_frame` covers both per-turn layer shapes (exploded value
  lists, and one-result-per-item loader scans, judged or not).
- `generic_flatten` is the default when a layer gives no frame fn.
"""

import pandas as pd

from transect.frames.common import (
    identity,
    input_id,
    item_metadata,
    joined_judge_models,
    judge_identity,
    result_value,
    verifier_review,
)


def turns_frame(results: pd.DataFrame, key: str = "turns") -> pd.DataFrame:
    """Identity + one row per turn, for per-turn layer shapes.

    A result whose value carries a ``key`` list explodes into one row
    per entry (the mechanical shape: one result per transcript packing
    its turns as a list, `token_timeline`'s convention; a judged batch
    result - ``cohort_llm_scanner(batch=True)`` - explodes its entries
    through the same judged projection as loader items). Any other
    result is a loader item (e.g. one `transect.reasoning_turns` turn) and
    projects one row: the item id and its ``Result.metadata`` (the
    ``turn`` key for a per-turn loader), the decided ``label``, and -
    whenever the value carries a judge block - the judged columns.

    Args:
        results: The layer scanner's raw results table.
        key: The value key holding the per-turn entry list; a batch
            result's entries (under the unit-neutral ``items`` key)
            are found regardless.
    """
    rows: list[dict] = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r.get("value"))
        entries = value.get(key)
        if not isinstance(entries, list):
            # the batch scanner's unit-neutral key
            entries = value.get("items")
        if isinstance(entries, list):
            # a judged batch result (cohort_llm_scanner(batch=True))
            if "judge" in value:
                rows.extend(
                    {**identity_cols, **_batch_entry_row(entry, value)}
                    for entry in entries
                )
            else:
                rows.extend({**identity_cols, **entry} for entry in entries)
        else:
            rows.append({**identity_cols, **_item_row(r, value)})
    frame = pd.DataFrame(rows)
    if "turn" in frame.columns:
        frame = frame.sort_values(["transcript_id", "turn"]).reset_index(drop=True)
    return frame


def generic_flatten(results: pd.DataFrame) -> pd.DataFrame:
    """The default layer frame: identity columns + the value's
    top-level keys, one row per result. Scalar values project as
    columns; nested values stay raw objects in their cells."""
    rows = []
    for _, r in results.iterrows():
        rows.append({**identity(r), **result_value(r.get("value"))})
    return pd.DataFrame(rows)


def member_ballots(
    frame: pd.DataFrame, unit_col: str, label_col: str = "label"
) -> pd.DataFrame:
    """Long-format per-member ballots off a judged layer frame: one row
    per (unit, cohort member) - the votes shape `transect.reliability`'s
    cohort_agreement / label_stats / member_coverage read.

    Reads the frame's raw ``members`` column (`turns_frame` carries it
    for cohort-judged loader items and batch entries alike) and stamps
    each member row with the
    frame row's ``unit_col`` (plus ``transcript_id`` when present); the
    member's own label lands in ``label_col``. Empty outside the voting
    regimes.

    Args:
        frame: The layer's judged frame.
        unit_col: The frame's judged-unit key column.
        label_col: The ballots' label column name, matching the frame's
            decided-label column so the two can join.
    """
    carry = [c for c in ("transcript_id", unit_col) if c in frame.columns]
    rows: list[dict] = []
    if "members" in frame.columns:
        for _, r in frame.iterrows():
            members = r["members"]
            if members is None or isinstance(members, float):
                continue  # no cohort block on this row (solo, or mechanical)
            rows.extend({**{c: r[c] for c in carry}, **m} for m in list(members))
    columns = [
        *carry,
        "model",
        "roll",
        "label",
        "confidence",
        "explanation",
        "status",
        "error",
    ]
    out = pd.DataFrame(rows, columns=columns)
    if label_col != "label" and label_col not in out.columns:
        out = out.rename(columns={"label": label_col})
    return out


def layer_frame(layer, raw: pd.DataFrame) -> pd.DataFrame:
    """One layer's mounted frame: the ready DataFrame as handed over,
    the frame fn over the scanner's raw results, or the generic flatten.
    """
    if isinstance(layer.frame, pd.DataFrame):
        return layer.frame
    if layer.frame is not None:
        return layer.frame(raw)
    return generic_flatten(raw)


def _judged_cells(unit: dict, identity_source: dict) -> dict:
    """The judged overlay shared by loader items and batch entries:

    - vote stats
    - member records
    - the flattened verifier review
    - the run-constant judge identity

    ``unit`` holds one judged unit's cohort/verifier blocks;
    ``identity_source`` is where the judge block was stamped (the
    value itself for items, the batch result's value for entries)."""
    armed = unit.get("verifier") or {}
    review = armed if armed.get("ran") else {}
    cohort = unit.get("cohort") or {}
    vote = cohort.get("vote") or {}
    return {
        "confidence_pm": vote.get("confidence_pm"),
        "judge_agreement": cohort.get("agreement"),
        "members": cohort.get("members"),
        "judge_models": joined_judge_models(identity_source),
        **verifier_review(review),
        "verifier_model": review.get("verifier_model") or armed.get("model"),
        **judge_identity(identity_source),
    }


def _batch_entry_row(entry: dict, parent: dict) -> dict:
    """One batch entry's judged projection."""
    return {
        **{k: v for k, v in entry.items() if k not in ("cohort", "verifier")},
        **_judged_cells(entry, parent),
    }


def _item_row(r, value: dict) -> dict:
    """One loader item's row: metadata + label, plus the judged
    projection when the scanner stamped a judge block."""
    row = {"item": input_id(r), **item_metadata(r)}
    raw = r.get("answer")
    label = (
        "_".join(raw.strip().lower().split())
        if isinstance(raw, str) and raw.strip()
        else None
    )
    row["label"] = label
    if "judge" not in value:
        row.update(value)
        return row
    explanation = r.get("explanation")
    row.update(
        {
            "confidence": value.get("confidence"),
            "label_source": value.get("label_source"),
            "explanation": explanation if isinstance(explanation, str) else None,
            "status": value.get("status") or ("ok" if label else None),
            **_judged_cells(value, value),
        }
    )
    return row
