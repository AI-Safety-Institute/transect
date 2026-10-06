"""phase_turns_df: the dense per-turn phase surface.

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn (one row each).
- phase_index: joins phases_df within the transcript.
- phase: the turn's label.
- basis: how the turn got its label - judged / filled / attributed /
  refusal / no_answer / missing_turn.
- label_source: which regime decided - single_judge / majority_vote
  / verifier; doubles as the confidence's source wherever
  confidence is non-null.
- confidence: the deciding regime's per-turn confidence.
- confidence_pm: spread across cohort members on this turn.
- judge_agreement: the turn's vote agreement (voting regimes; None
  when fewer than two members voted on the turn).
- n_voting: how many members voted on the turn (voting regimes).
- schema_version: the frames contract version.

The judge roster lives on ``phases.judge_models`` (one row per phase),
joined by ``phase_index``; per-turn spend lives on ``token_timeline``,
joined by ``(transcript_id, turn)``.
"""

import pandas as pd

from transect.frames.common import (
    BASES,
    IDENTITY_COLS,
    LABEL_SOURCES,
    categorical,
    identity,
    result_value,
    with_schema,
)


def phase_turns_df(results: pd.DataFrame) -> pd.DataFrame:
    """decision_phases results -> one row per orchestrator turn."""
    rows = []
    vocabulary: list[str] = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        phases = value.get("phases") or []
        # per-turn voter count, from the cohort vote block
        vote_rows = (value.get("cohort") or {}).get("vote") or []
        n_voting_of = {v["turn"]: v.get("n_voting") for v in vote_rows}
        for name in value.get("phase_names") or []:
            if name not in vocabulary:
                vocabulary.append(name)
        for entry in value.get("turns") or []:
            index = entry.get("phase_index")
            rows.append(
                {
                    **identity_cols,
                    "turn": entry.get("turn"),
                    "phase_index": index,
                    "phase": (
                        phases[index]["phase"]
                        if index is not None and index < len(phases)
                        else None
                    ),
                    "basis": entry.get("basis"),
                    "label_source": entry.get("label_source"),
                    "confidence": entry.get("confidence"),
                    "confidence_pm": entry.get("confidence_pm"),
                    "judge_agreement": entry.get("agreement"),
                    "n_voting": n_voting_of.get(entry.get("turn")),
                }
            )
    columns = [
        *IDENTITY_COLS,
        "turn",
        "phase_index",
        "phase",
        "basis",
        "label_source",
        "confidence",
        "confidence_pm",
        "judge_agreement",
        "n_voting",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "judge_agreement": "float64",
            "phase_index": "Int64",
            "confidence": "float64",
            "confidence_pm": "float64",
            "n_voting": "Int64",
        }
    )
    df["phase"] = categorical(df.phase, vocabulary)
    df["basis"] = categorical(df.basis, BASES)
    df["label_source"] = categorical(df.label_source, LABEL_SOURCES)
    return with_schema(df)
