"""phase_turns_df: the dense per-turn phase surface.

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn (one row each).
- phase_index: joins phases_df within the transcript; None on
  unjudged and unattributed turns.
- phase: the turn's label; None with phase_index.
- basis: how the turn got its label - judged / filled / attributed /
  unattributed / refusal / no_answer / missing_turn.
- label_source: which regime decided - single_judge / majority_vote
  / verifier; doubles as the confidence's source wherever
  confidence is non-null.
- confidence: the deciding regime's per-turn confidence.
- confidence_pm: spread across cohort members on this turn.
- judge_agreement: the turn's vote agreement (voting regimes; None
  when fewer than two members voted on the turn).
- n_voting: how many members voted on the turn (voting regimes).
- judge_models / verifier_model: the configured judge roster ("+"-joined)
  and verifier model, retained even when no phase was produced.
- judge_regime / n_models / k_rolls / verifier_armed /
  verifier_same_model: the scanner-stamped judge setup (common.JUDGE_COLS),
  constant across the transcript, including unjudged and unseen turns.
- schema_version: the frames contract version.

The judge setup also lives on ``phases`` (one row per phase), joined by
``phase_index``; per-turn spend lives on ``token_timeline``,
joined by ``(transcript_id, turn)``.
"""

import pandas as pd

from transect.frames.common import (
    BASES,
    IDENTITY_COLS,
    JUDGE_COLS,
    LABEL_SOURCES,
    categorical,
    identity,
    joined_judge_models,
    judge_identity,
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
        for name in value.get("phase_names") or []:
            if name not in vocabulary:
                vocabulary.append(name)
        entries = value.get("turns") or []
        if not entries:
            continue
        judge_facts = judge_identity(value)
        judge_models = joined_judge_models(value)
        verifier_model = value["judge"].get("verifier_model")
        # per-turn voter count, from the cohort vote block
        vote_rows = (value.get("cohort") or {}).get("vote") or []
        n_voting_of = {v["turn"]: v.get("n_voting") for v in vote_rows}
        for entry in entries:
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
                    "judge_models": judge_models,
                    "verifier_model": verifier_model,
                    **judge_facts,
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
        "judge_models",
        "verifier_model",
        *JUDGE_COLS,
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "judge_agreement": "float64",
            "phase_index": "Int64",
            "confidence": "float64",
            "confidence_pm": "float64",
            "n_voting": "Int64",
            "n_models": "Int64",
            "k_rolls": "Int64",
            "verifier_armed": "boolean",
            "verifier_same_model": "boolean",
        }
    )
    df["phase"] = categorical(df.phase, vocabulary)
    df["basis"] = categorical(df.basis, BASES)
    df["label_source"] = categorical(df.label_source, LABEL_SOURCES)
    return with_schema(df)
