"""token_timeline_df: one row per model turn + token views.

Columns (identity prefix explained in common.py):

- turn: 0-based model-turn index.
- n_tool_calls: tool calls requested by this turn's assistant message.
- agent_lane / agent_span_id: the turn's sub-agent span (name / id);
  None for main-lane turns.
- output_tokens / input_tokens / total_tokens /
  input_tokens_cache_read / input_tokens_cache_write /
  reasoning_tokens: inspect-ai ModelUsage fields verbatim; None =
  the provider did not report.
- cache_semantics: "exclusive" or "inclusive" - what input_tokens
  means for this provider (detected per transcript).
- context: context-window size at this turn.
- new_work: new content processed this turn.
- billable: what the turn costs (full cache writes included).
- turn_total: input_new + cache writes/reads + reasoning + output.
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

_DERIVED_NUMERIC = ("context", "new_work", "billable", "turn_total")
_DERIVED_FIELDS = ("cache_semantics", *_DERIVED_NUMERIC)


def token_timeline_df(results: pd.DataFrame) -> pd.DataFrame:
    """token_timeline scanner results -> one row per model turn,
    + derived token-view columns (see _derive_token_views)."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        timeline = result_value(r["value"]).get("timeline") or []
        for entry in timeline:
            row = {
                **identity_cols,
                "turn": entry["turn"],
                "n_tool_calls": entry.get("n_tool_calls"),
                "agent_lane": entry.get("agent_lane"),
                "agent_span_id": entry.get("agent_span_id"),
            }
            row.update({field: entry.get(field) for field in _TOKEN_FIELDS})
            rows.append(row)
    columns = [
        *IDENTITY_COLS,
        "turn",
        "n_tool_calls",
        "agent_lane",
        "agent_span_id",
        *_TOKEN_FIELDS,
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["n_tool_calls"] = df.n_tool_calls.astype("Int64")
    if len(df):
        df = _derive_token_views(df)
    else:
        # an empty run still carries the full documented schema
        df = df.reindex(columns=[*columns, *_DERIVED_FIELDS])
    # pin nullable numerics: token counts as Int64
    df = df.astype(dict.fromkeys([*_TOKEN_FIELDS, *_DERIVED_NUMERIC], "Int64"))
    return with_schema(df)


def _derive_token_views(timeline: pd.DataFrame) -> pd.DataFrame:
    """Add provider-aware derived token columns to a timeline frame.

    Provider-aware: providers disagree on what input_tokens means, so the
    reporting convention is detected per transcript and normalised.

    New columns:

    - cache_semantics: "exclusive" = input_tokens is uncached input only
      (cache reads/writes reported separately); "inclusive" = the full
      prompt, cached context included.
    - context: context-window size = input_new + cache_write + cache_read,
      where input_new (an intermediate, not a column) is the genuinely
      new input: input_tokens as-is when exclusive; input_tokens -
      cache_read - cache_write (floored at 0) when inclusive.
    - new_work: input_new + output + reasoning + cache_write capped at the
      context growth since the previous non-gap turn, so re-caching an
      unchanged context (e.g. after cache TTL expiry) is not counted.
    - billable: input_new + output + reasoning + full cache_write
      (re-caching is paid for even when it is not new work).
    - turn_total: input_new + cache_write + cache_read + reasoning +
      output. Distinct from the raw provider total_tokens column.

    Gap turns (usage-less or all-zero usage) keep NA derived fields; the
    previous context carries forward, never reset.
    """
    derived_rows = []
    for transcript_id, group in timeline.groupby("transcript_id", sort=False):
        all_turns = group.sort_values("turn")
        inclusive = _detect_cache_inclusive(all_turns)
        # Each agent span is its own conversation with its own context window
        for _, turns in all_turns.groupby(lane_series(all_turns), sort=False):
            prev_ctx = 0
            for _, t in turns.iterrows():
                row = {
                    "transcript_id": transcript_id,
                    "turn": t["turn"],
                    "cache_semantics": "inclusive" if inclusive else "exclusive",
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
                    reasoning = _int0(t["reasoning_tokens"])
                    new = inp
                    if inclusive:
                        # floor at 0: a provider mis-report cannot go negative
                        new = max(0, inp - cr - cw)
                    ctx = new + cw + cr
                    base = new + out + reasoning
                    row.update(
                        context=ctx,
                        new_work=base + min(cw, max(0, ctx - prev_ctx)),
                        billable=base + cw,
                        turn_total=new + cw + cr + reasoning + out,
                    )
                    prev_ctx = ctx
                derived_rows.append(row)
    derived = pd.DataFrame(derived_rows)
    derived = derived[["transcript_id", "turn", *_DERIVED_FIELDS]]
    return timeline.merge(derived, on=["transcript_id", "turn"], how="left")


def _detect_cache_inclusive(turns: pd.DataFrame) -> bool:
    """Decide whether input_tokens includes the re-read cached context."""
    any_cache_read = False
    for _, t in turns.iterrows():
        if _is_gap(t):
            continue
        cache_read = _int0(t["input_tokens_cache_read"])
        if cache_read > 0:
            any_cache_read = True
        if cache_read > _int0(t["input_tokens"]):
            return False
    return any_cache_read


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
