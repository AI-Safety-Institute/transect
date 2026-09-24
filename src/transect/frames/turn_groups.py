"""turn_groups_df: narrator turn groups.

Columns (identity prefix explained in common.py):

- phase_index: joins phases_df within the transcript.
- group_index: the group's position within its phase.
- turn_start / turn_end: inclusive range; groups partition their
  phase's range gaplessly.
- title / gist: the narrator's short name and one-line summary. Invalid
  partitions use a neutral phase title and empty gist; factual prose is
  never stretched onto a repaired range.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def turn_groups_df(results: pd.DataFrame) -> pd.DataFrame:
    """decision_phases results -> one row per narrative turn group."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        value = result_value(r["value"])
        for phase_index, phase in enumerate(value.get("phases") or []):
            for group_index, group in enumerate(phase.get("turn_groups") or []):
                rows.append(
                    {
                        **identity_cols,
                        "phase_index": phase_index,
                        "group_index": group_index,
                        **group,
                    }
                )
    columns = [
        *IDENTITY_COLS,
        "phase_index",
        "group_index",
        "turn_start",
        "turn_end",
        "title",
        "gist",
    ]
    return with_schema(pd.DataFrame(rows, columns=columns))
