"""subagent_votes_df: per-member ballots for classified spans.

One row per (transcript, span, member); a child frame of
subagents_df.

Columns (identity prefix explained in common.py):

- agent_span_id: joins subagents_df within the transcript.
- model / roll: the cohort member identity (roll 0-based).
- label: the member's label for the span.
- confidence: the member's stated confidence.
- explanation: the member's explanation.
- status: ok / refusal / error for that member's call.
- error: the reason behind an error status; None otherwise.
- schema_version: the frames contract version.

A pure projection of the member records; empty outside the
voting regimes."""

import pandas as pd

from transect.frames.common import (
    IDENTITY_COLS,
    identity,
    input_id,
    result_value,
    with_schema,
)


def subagent_votes_df(results: pd.DataFrame) -> pd.DataFrame:
    """subagent_classification results -> one row per (span, member)."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        span_id = input_id(r)
        if not span_id:
            continue
        cohort = result_value(r.get("value")).get("cohort") or {}
        for member in cohort.get("members") or []:
            rows.append(
                {
                    **identity_cols,
                    "agent_span_id": span_id,
                    **member,
                }
            )
    columns = [
        *IDENTITY_COLS,
        "agent_span_id",
        "model",
        "roll",
        "label",
        "confidence",
        "explanation",
        "status",
        "error",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype({"roll": "Int64", "confidence": "float64"})
    return with_schema(df)
