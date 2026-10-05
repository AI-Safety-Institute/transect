"""lane_activity_df: tool-events-only sub-agent activity.

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn preceding the tool event.
- agent_lane / agent_span_id: the sub-agent span (name / id).
- tool_calls: tool events in that span at that turn.
- busy_seconds: summed tool wall-time; None = not reported.
- started_at: the span's first activity timestamp.
- span_end_turn: the orchestrator turn preceding the span's end event;
  None when the span never closed.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def lane_activity_df(results: pd.DataFrame) -> pd.DataFrame:
    """token_timeline lane_activity -> one row per (turn, agent span) of
    tool activity, for every sub-agent span."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        end_of = {
            s["agent_span_id"]: s["event_order_end_turn"]
            for s in value.get("spans") or []
        }
        for entry in value.get("lane_activity") or []:
            rows.append(
                {
                    **identity_cols,
                    **entry,
                    "span_end_turn": end_of.get(entry.get("agent_span_id")),
                }
            )
    columns = [
        *IDENTITY_COLS,
        "turn",
        "agent_lane",
        "agent_span_id",
        "tool_calls",
        "busy_seconds",
        "started_at",
        "span_end_turn",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # pin nullable numerics (see phases_df)
    df = df.astype({"busy_seconds": "float64", "span_end_turn": "Int64"})
    return with_schema(df)
