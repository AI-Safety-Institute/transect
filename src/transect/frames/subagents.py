"""subagents_df: one row per agent span - structural facts always,
judge columns when the classification scanner ran.

The structural half is built from the token_timeline scanner's span
record (identity, axis coordinates) plus `lane_activity_df` and
`token_timeline_df` (activity rollups), so the frame exists even on
scans with no judge; the classification half left-joins onto it and is
NaN otherwise.

Columns (identity prefix explained in common.py):

- agent_span_id / agent_lane: the sub-agent span (id / name).
- spawn_turn: the orchestrator turn preceding the span's begin event
  (the turn that issued the spawn when the source links them, the
  importer's file-order anchor otherwise). Phase membership for the
  sub-agent count and the delegated spend uses this.
- anchor_turn / end_turn: the orchestrator turns active at the span's
  first activity and at its end (frames/spine.py: the turn whose
  wall-clock cell holds each timestamp).
- turn_source: "timestamp" (the two turns above follow wall clock) or
  "event_order" (no usable clock: anchor_turn is spawn_turn and
  end_turn the orchestrator turn preceding the span's end event).
- started_at / ended_at: ISO timestamps of the span's first activity
  and of its end - the recorded end, else its last observed activity,
  end_recorded telling which. None where the source recorded nothing.
- end_recorded: the source recorded the span's end (False on OpenClaw,
  whose ends are synthesised at last activity).
- tool_calls / busy_seconds: summed tool activity (lane_activity).
- output_tokens / new_work / billable: summed model-turn token
  rollups (token_timeline agent rows; NaN for tool-events-only spans).
- label: the assigned classification.
- span_task / span_task_truncated: the exact task text the judge saw.
- confidence_pm: cross-member confidence spread on the vote.
- original_explanation: the pre-overturn judge explanation.
- label_source: who decided - single_judge / majority_vote /
  verifier.
- explanation: the decider's explanation.
- span_task_source: where the loader found the span's task text
  (e.g. spawn_prompt / handoff_input).
- status: ok / refusal / error.
- judge_models: the judge model(s) that classified the span.
- confidence: the decider's stated confidence (vote = modal-side
  mean; overturn = the verifier's).
- judge_agreement: vote agreement (NaN unless >= 2 voted).
- n_voting / n_members: vote participation.
- verifier_selected: a review record exists, failed attempts included.
- verifier_completed: a usable verifier verdict was returned.
- overturned: the verifier replaced the label.
- verifier_trigger / verifier_label / verifier_confidence /
  verifier_explanation / verifier_status / original_label /
  original_confidence / original_explanation: the flattened
  VerifierReview record (common.VERIFIER_COLS; NaN when no review
  ran; verifier_status is the review call's ok / refusal / error).
- verifier_model: the verifier judge whenever the verifier was armed
  for the run, reviews or not (the armed stamp); None when the
  verifier was off. "A review happened" is verifier_selected, never
  this column's presence.
- judge_regime / n_models / k_rolls / verifier_armed /
  verifier_same_model: the scanner-stamped judge-identity columns
  (common.JUDGE_COLS), constant across a span's judged rows; None on
  unjudged spans and on errored solo scans (no value to stamp).
- schema_version: the frames contract version.
"""

from typing import Literal

import pandas as pd

from transect.frames import spine
from transect.frames.common import (
    CALL_STATUSES,
    IDENTITY_COLS,
    JUDGE_COLS,
    LABEL_SOURCES,
    VERIFIER_COLS,
    categorical,
    identity,
    input_id,
    item_metadata,
    joined_judge_models,
    judge_identity,
    result_value,
    verifier_review,
    with_schema,
)


def subagents_df(
    results: pd.DataFrame,
    timeline_results: pd.DataFrame | None = None,
    lane_activity: pd.DataFrame | None = None,
    token_timeline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per agent span: structural spine from the token_timeline
    scanner's raw results (``timeline_results``: the span record and the
    orchestrator clock) plus ``lane_activity`` and ``token_timeline``
    rollups; classification columns left-joined from the
    subagent_classification results (NaN when the scanner never ran)."""
    judge = _judge_rows(results)
    spine = _structural_spine(timeline_results, lane_activity, token_timeline)
    if spine is None:
        merged = judge
        for col in _SPINE_FLAGS:
            merged[col] = False
        for col in _SPINE_TEXT:
            merged[col] = pd.Series(pd.NA, index=merged.index, dtype="string")
        for col, dtype in _SPINE_DTYPES:
            merged[col] = None
            merged[col] = merged[col].astype(dtype)
    else:
        drop = ["agent_lane", "schema_version"] + [
            c for c in IDENTITY_COLS if c != "transcript_id"
        ]
        merged = spine.merge(
            judge.drop(columns=[c for c in drop if c in judge.columns]),
            on=["transcript_id", "agent_span_id"],
            how="outer",
        )
    merged["label"] = categorical(merged.label, _vocabulary(results))
    merged["label_source"] = categorical(merged.label_source, LABEL_SOURCES)
    merged["status"] = categorical(merged.status, CALL_STATUSES)
    merged["verifier_status"] = categorical(merged.verifier_status, CALL_STATUSES)
    ordered = [
        *IDENTITY_COLS,
        "agent_span_id",
        "agent_lane",
        "spawn_turn",
        "anchor_turn",
        "end_turn",
        "turn_source",
        "started_at",
        "ended_at",
        "end_recorded",
        "tool_calls",
        "busy_seconds",
        "output_tokens",
        "new_work",
        "billable",
        "label",
        "label_source",
        "explanation",
        "span_task",
        "span_task_truncated",
        "span_task_source",
        "status",
        "judge_models",
        "confidence",
        "judge_agreement",
        "confidence_pm",
        "n_voting",
        "n_members",
        *JUDGE_COLS,
        *VERIFIER_COLS,
        "verifier_model",
    ]
    merged = merged[[c for c in ordered if c in merged.columns]]
    merged["verifier_selected"] = merged.verifier_selected.fillna(False).astype(bool)
    merged["overturned"] = merged.overturned.fillna(False).astype(bool)
    return with_schema(
        merged.sort_values(
            ["transcript_id", "anchor_turn", "started_at", "agent_span_id"],
            na_position="last",
        ).reset_index(drop=True)
    )


def _vocabulary(results: pd.DataFrame) -> list[str]:
    """The scan-recorded label vocabulary."""
    out: list[str] = []
    for _, r in results.iterrows():
        value = result_value(r.get("value"))
        for entry in value.get("label_vocab") or []:
            name = entry.get("label")
            if name and name not in out:
                out.append(name)
    return out


# the structural columns
_SPINE_DTYPES: tuple[tuple[str, Literal["Int64", "Float64"]], ...] = (
    ("spawn_turn", "Int64"),
    ("anchor_turn", "Int64"),
    ("end_turn", "Int64"),
    ("tool_calls", "Int64"),
    ("busy_seconds", "Float64"),
    ("output_tokens", "Float64"),
    ("new_work", "Float64"),
    ("billable", "Float64"),
)
_SPINE_FLAGS = ("end_recorded",)
_SPINE_TEXT = ("turn_source", "started_at", "ended_at")


def _structural_spine(
    timeline_results: pd.DataFrame | None,
    lane_activity: pd.DataFrame | None,
    token_timeline: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Per-span structural facts: turns and timestamps from the span
    record, rollups from the two activity frames."""
    parts = []
    records = _span_turns(timeline_results)
    if records is not None:
        parts.append(records.set_index(["transcript_id", "agent_span_id"]))
    if lane_activity is not None and len(lane_activity):
        tools = lane_activity.groupby(
            ["transcript_id", "agent_span_id"], sort=False
        ).agg(
            sample_id=("sample_id", "first"),
            task_set=("task_set", "first"),
            epoch=("epoch", "first"),
            agent=("agent", "first"),
            agent_lane=("agent_lane", "first"),
            tool_calls=("tool_calls", "sum"),
            busy_seconds=("busy_seconds", lambda s: s.sum(min_count=1)),
        )
        parts.append(tools)
    if token_timeline is not None and len(token_timeline):
        agent_rows = token_timeline[token_timeline.agent_span_id.notna()]
        if len(agent_rows):
            model = agent_rows.groupby(
                ["transcript_id", "agent_span_id"], sort=False
            ).agg(
                sample_id=("sample_id", "first"),
                task_set=("task_set", "first"),
                epoch=("epoch", "first"),
                agent=("agent", "first"),
                agent_lane=("agent_lane", "first"),
                output_tokens=("output_tokens", lambda s: s.sum(min_count=1)),
                new_work=("new_work", lambda s: s.sum(min_count=1)),
                billable=("billable", lambda s: s.sum(min_count=1)),
            )
            parts.append(model)
    if not parts:
        return None
    spine = parts[0]
    for part in parts[1:]:
        spine = spine.join(part, how="outer", rsuffix="_m")
        # coalesce the shared identity/lane columns from either source
        for col in ("sample_id", "task_set", "epoch", "agent", "agent_lane"):
            twin = f"{col}_m"
            if twin in spine.columns:
                spine[col] = spine[col].combine_first(spine[twin])
                spine = spine.drop(columns=[twin])
    for col in (*_SPINE_FLAGS, *_SPINE_TEXT, *(c for c, _ in _SPINE_DTYPES)):
        if col not in spine.columns:
            spine[col] = None
    for col in _SPINE_FLAGS:
        spine[col] = spine[col].fillna(False).astype(bool)
    for col in _SPINE_TEXT:
        spine[col] = spine[col].astype("string")
    for col, dtype in _SPINE_DTYPES:
        spine[col] = spine[col].astype(dtype)
    return spine.reset_index()


def _span_turns(timeline_results: pd.DataFrame | None) -> pd.DataFrame | None:
    """The span record with its orchestrator turns: one row per
    sub-agent span (`frames.spine.span_turns`)."""
    if timeline_results is None or not len(timeline_results):
        return None
    rows = []
    for _, r in timeline_results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        orchestrator = sorted(
            (e for e in value.get("timeline") or [] if e.get("turn") is not None),
            key=lambda e: e["turn"],
        )
        cells = spine.clock(
            [e.get("timestamp") for e in orchestrator],
            orchestrator[-1].get("completed") if orchestrator else None,
        )
        for record in value.get("spans") or []:
            end_recorded = bool(record.get("end_recorded"))
            span = {
                "spawn_turn": record["spawn_turn"],
                "started_at": record.get("first_at"),
                "ended_at": record.get("end_at")
                if end_recorded
                else record.get("last_at"),
                "end_recorded": end_recorded,
                "event_order_end_turn": record.get("event_order_end_turn"),
            }
            rows.append(
                {
                    **identity_cols,
                    "agent_span_id": record["agent_span_id"],
                    "agent_lane": record["agent_lane"],
                    "spawn_turn": span["spawn_turn"],
                    **spine.span_turns(span, cells),
                    "started_at": span["started_at"],
                    "ended_at": span["ended_at"],
                    "end_recorded": end_recorded,
                }
            )
    return pd.DataFrame(rows) if rows else None


def _judge_rows(results: pd.DataFrame) -> pd.DataFrame:
    """subagent_classification results -> one judge row per span."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        # identity from the store's own input_ids
        span_id = input_id(r)
        if not span_id:
            continue  # not a span item (defensive)
        meta = item_metadata(r)
        error = r.get("scan_error")
        has_error = isinstance(error, str) and bool(error)
        value = result_value(r.get("value"))
        armed = value.get("verifier") or {}
        review = armed if armed.get("ran") else {}
        raw = r.get("label")
        label = None
        if not has_error and isinstance(raw, str) and raw.strip():
            label = "_".join(raw.strip().lower().split())
        confidence = value.get("confidence")
        label_source = value.get("label_source")
        agreement = n_voting = n_members = confidence_pm = None
        if "cohort" in value:  # voting regimes carry the cohort block
            block = value.get("cohort") or {}
            vote = block.get("vote") or {}
            agreement = block.get("agreement")
            n_voting = vote.get("n_voting")
            confidence_pm = vote.get("confidence_pm")
            n_members = len(block.get("members") or []) or None
            status = value.get("status") or "error"
            judge_model = joined_judge_models(value)
        else:
            # solo errors surface as Scout scan errors, not value facts
            if has_error:
                category = r.get("scan_error_type")
                status = (
                    "refusal"
                    if isinstance(category, str) and category == "refusal"
                    else "error"
                )
            else:
                status = "ok" if label else "error"
            stamped = value.get("judge")
            roster = stamped.get("models") if isinstance(stamped, dict) else None
            judge_model = "+".join(roster) if roster else _judge_model_of(r)
        # an errored solo scan records no value, so there is no judge
        # block to read - those rows project None for the judge columns
        judge_facts = (
            dict.fromkeys(JUDGE_COLS)
            if has_error and "judge" not in value
            else judge_identity(value)
        )
        explanation = r.get("explanation")
        rows.append(
            {
                **identity_cols,
                "agent_span_id": span_id,
                "agent_lane": meta.get("agent_lane"),
                "label": label,
                "label_source": label_source,
                "explanation": explanation if isinstance(explanation, str) else None,
                "span_task": meta.get("span_task"),
                "span_task_truncated": bool(meta.get("span_task_truncated")),
                "span_task_source": meta.get("span_task_source"),
                "status": status,
                "judge_models": judge_model,
                "confidence": confidence,
                "judge_agreement": agreement,
                "confidence_pm": confidence_pm,
                "n_voting": n_voting,
                "n_members": n_members,
                **verifier_review(review),
                "verifier_model": review.get("verifier_model") or armed.get("model"),
                **judge_facts,
            }
        )
    columns = [
        *IDENTITY_COLS,
        "agent_span_id",
        "agent_lane",
        "label",
        "label_source",
        "explanation",
        "span_task",
        "span_task_truncated",
        "span_task_source",
        "status",
        "judge_models",
        "confidence",
        "judge_agreement",
        "confidence_pm",
        "n_voting",
        "n_members",
        *JUDGE_COLS,
        *VERIFIER_COLS,
        "verifier_model",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # pin nullable numerics: ratios as float64, counts as Int64
    df = df.astype(
        {
            "confidence": "float64",
            "judge_agreement": "float64",
            "n_voting": "Int64",
            "n_members": "Int64",
            "n_models": "Int64",
            "k_rolls": "Int64",
            "verifier_armed": "boolean",
            "verifier_same_model": "boolean",
            "verifier_confidence": "float64",
            "original_confidence": "float64",
        }
    )
    return with_schema(df)


def _judge_model_of(r) -> str | None:
    """Judge model name from the row's model-usage record (keyed by
    model) - for solo rows without a stamped roster: errored rows (no
    value at all) and the models=None default path (empty roster)."""
    usage = result_value(r.get("scan_model_usage"))
    return next(iter(usage)) if usage else None
