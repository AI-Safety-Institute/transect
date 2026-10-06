"""phase_turn_votes_df: per-member per-turn judgements (voting regimes).

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn. One row per (member, digest turn).
- model / roll: the cohort member identity (roll 0-based).
- basis: the member's own per-turn basis - judged / filled /
  no_answer / refusal / missing_turn (the scanner's vocabulary).
- phase: the member's own per-turn label; NA unless basis is
  judged - a filled turn's propagated label is not a vote.
- confidence: the member's stated per-turn confidence; NA unless
  basis is judged (fill confidences are scanner plumbing).
- schema_version: the frames contract version.

Empty outside the voting regimes (``cohort.ran`` false, solo judge).
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def phase_turn_votes_df(results: pd.DataFrame) -> pd.DataFrame:
    """decision_phases results -> one row per (cohort member, digest turn)."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        cohort = value.get("cohort") or {}
        for member in cohort.get("members") or []:
            for t in member.get("turns") or []:
                phase = t.get("phase")
                judged = t.get("basis") == "judged" and phase is not None
                rows.append(
                    {
                        **identity_cols,
                        "turn": t["turn"],
                        "model": member.get("model"),
                        "roll": member.get("roll"),
                        "basis": t.get("basis"),
                        "phase": phase if judged else None,
                        "confidence": t.get("confidence") if judged else None,
                    }
                )
    columns = [*IDENTITY_COLS, "turn", "model", "roll", "basis", "phase", "confidence"]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype({"turn": "Int64", "roll": "Int64", "confidence": "float64"})
    return with_schema(df)
