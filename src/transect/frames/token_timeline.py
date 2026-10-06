"""token_timeline_df: one row per model turn in any lane (orchestrator
turns on the axis, sub-agent and init/scorer calls off it) + token views.

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn; NA on a sub-agent's own turns (they
  are off the axis) and on init/scorer calls.
- lane_turn: 0-based ordinal within the turn's own lane; NA on an
  off-axis call, which is in no lane.
- n_tool_calls: tool calls requested by this turn's assistant message.
- agent_lane / agent_span_id: the turn's sub-agent span (name / id);
  None for orchestrator turns.
- timestamp / completed: the model call's start and completion, ISO
  8601; None when unrecorded.
- output_tokens / input_tokens / total_tokens /
  input_tokens_cache_read / input_tokens_cache_write /
  reasoning_tokens: inspect-ai ModelUsage fields verbatim; None =
  the provider did not report.
- context: context-window size at this turn.
- new_work: new content processed this turn.
- billable: tokens excluding cache reads (uncached input + output +
  full cache writes), not monetary cost.
- turn_total: input + cache writes/reads + output (includes reasoning).
- schema_version: the frames contract version.

Full derivation definitions on _derive_token_views."""

import pandas as pd

from transect.frames.common import (
    IDENTITY_COLS,
    identity,
    lane_series,
    result_value,
    with_schema,
)

_TOKEN_FIELDS = (
    "output_tokens",
    "input_tokens",
    "total_tokens",
    "input_tokens_cache_read",
    "input_tokens_cache_write",
    "reasoning_tokens",
)

_DERIVED_FIELDS = ("context", "new_work", "billable", "turn_total")


def token_timeline_df(results: pd.DataFrame) -> pd.DataFrame:
    """token_timeline scanner results -> one row per model turn in any lane,
    + derived token-view columns (see _derive_token_views)."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        timeline = result_value(r["value"]).get("timeline") or []
        for entry in timeline:
            row = {
                **identity_cols,
                "turn": entry["turn"],
                "lane_turn": entry["lane_turn"],
                "n_tool_calls": entry.get("n_tool_calls"),
                "agent_lane": entry.get("agent_lane"),
                "agent_span_id": entry.get("agent_span_id"),
                "timestamp": entry.get("timestamp"),
                "completed": entry.get("completed"),
            }
            row.update({field: entry.get(field) for field in _TOKEN_FIELDS})
            rows.append(row)
    columns = [
        *IDENTITY_COLS,
        "turn",
        "lane_turn",
        "n_tool_calls",
        "agent_lane",
        "agent_span_id",
        "timestamp",
        "completed",
        *_TOKEN_FIELDS,
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "turn": "Int64",
            "lane_turn": "Int64",
            "n_tool_calls": "Int64",
            "timestamp": "string",
            "completed": "string",
        }
    )
    if len(df):
        df = _derive_token_views(df)
    else:
        # an empty run still carries the full documented schema
        df = df.reindex(columns=[*columns, *_DERIVED_FIELDS])
    # pin nullable numerics: token counts as Int64
    df = df.astype(dict.fromkeys([*_TOKEN_FIELDS, *_DERIVED_FIELDS], "Int64"))
    return with_schema(df)


def _derive_token_views(timeline: pd.DataFrame) -> pd.DataFrame:
    """Add token views under the supported Inspect ModelUsage contract.

    input_tokens excludes cache reads/writes; output_tokens includes
    reasoning_tokens. Preserve those raw counters and count output once.
    Importers with other conventions must normalize before this frame.

    - context: input + cache writes + cache reads.
    - new_work: input + output + cache writes capped at context growth
      since the previous non-gap turn in this lane. This is a heuristic
      for new content, not a measurement of cognitive work or dollar cost.
    - billable: input + output + full cache writes; excludes cache
      reads and has no price weighting.
    - turn_total: context + output, independent of raw total_tokens.

    Gap turns (usage-less or all-zero usage) keep NA derived fields; the
    previous context carries forward, never reset. Unknown optional
    breakdowns are treated as zero in these derived views.
    """
    derived_rows = []
    # rows keep the store's event order; a sub-agent row's turn is NA, so
    # the join back is positional (the frame's own index), never by turn
    for _transcript_id, group in timeline.groupby("transcript_id", sort=False):
        # Each agent span is its own conversation with its own context window
        for _, turns in group.groupby(lane_series(group), sort=False):
            prev_ctx = 0
            for index, t in turns.iterrows():
                row = {
                    "_row": index,
                    "context": None,
                    "new_work": None,
                    "billable": None,
                    "turn_total": None,
                }
                if not _is_gap(t):
                    inp = _int0(t["input_tokens"])
                    cw = _int0(t["input_tokens_cache_write"])
                    cr = _int0(t["input_tokens_cache_read"])
                    out = _int0(t["output_tokens"])
                    new = inp
                    ctx = new + cw + cr
                    base = new + out
                    row.update(
                        context=ctx,
                        new_work=base + min(cw, max(0, ctx - prev_ctx)),
                        billable=base + cw,
                        turn_total=new + cw + cr + out,
                    )
                    prev_ctx = ctx
                derived_rows.append(row)
    derived = pd.DataFrame(derived_rows).set_index("_row")[list(_DERIVED_FIELDS)]
    return timeline.join(derived)


def _is_gap(turn_row) -> bool:
    """Usage-less turn, or an all-zero placeholder."""
    if pd.isna(turn_row["input_tokens"]):
        return True
    fields = (
        "input_tokens",
        "output_tokens",
        "input_tokens_cache_read",
        "input_tokens_cache_write",
    )
    return all(_int0(turn_row[f]) == 0 for f in fields)


def _int0(value) -> int:
    """NaN/None-safe int: missing counts as 0."""
    return 0 if pd.isna(value) else int(value)
