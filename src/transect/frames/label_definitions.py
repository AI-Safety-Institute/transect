"""label_definitions_df: the judged surfaces' label vocabularies with
their definitions. One row per (surface, label).

Columns (identity prefix explained in common.py):

- surface: "phases" / "subagents", or a custom layer's name.
- label: the resolved label. Sub-agent keys are lowercased with whitespace
  replaced by underscores, including on older stores. Phase and custom-layer
  keys stay as recorded; new cohort scanners record normalized custom keys.
- description: the rubric text the judge saw.
- ops: the operational-bucket flag (phases only).
- reserved: scanner-added rather than spec-declared.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema
from transect.scanners._labels import normalize_vocabulary


def label_definitions_df(
    raw_phases: pd.DataFrame,
    raw_subagents: pd.DataFrame,
    layer_results: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """decision_phases + subagent_classification results -> one row per
    (surface, label). ``layer_results`` adds one surface per custom
    layer (name -> its scanner's raw results), read at the same
    ``label_vocab`` key `transect.cohort_llm_scanner` stamps."""
    rows = _surface_rows(raw_phases, "phases", "phase_vocab")
    rows += _surface_rows(raw_subagents, "subagents", "label_vocab")
    for surface, raw in (layer_results or {}).items():
        rows += _surface_rows(raw, surface, "label_vocab")
    columns = [*IDENTITY_COLS, "surface", "label", "description", "ops", "reserved"]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype({"ops": "boolean", "reserved": "boolean"})
    return with_schema(df)


def _surface_rows(raw: pd.DataFrame, surface: str, key: str) -> list[dict]:
    """One surface's vocabulary rows: the first result per transcript
    that carries the vocab wins; duplicate labels within it collapse."""
    rows: list[dict] = []
    done: set[str] = set()
    for _, r in raw.iterrows():
        identity_cols = identity(r)
        if identity_cols["transcript_id"] in done:
            continue
        value = result_value(r.get("value"))
        vocab = value.get(key) or []
        if not vocab:
            continue
        if surface == "subagents":
            vocab = normalize_vocabulary(vocab, allow_duplicates=True)
        done.add(identity_cols["transcript_id"])
        seen: set[str] = set()
        for entry in vocab:
            label = entry.get("label")
            if not label or label in seen:
                continue
            seen.add(label)
            rows.append(
                {
                    **identity_cols,
                    "surface": surface,
                    "label": label,
                    "description": entry.get("description"),
                    "ops": entry.get("ops"),
                    "reserved": entry.get("reserved"),
                }
            )
    return rows
