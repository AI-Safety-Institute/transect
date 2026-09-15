"""interventions_df: mid-run human interventions.

Columns (identity prefix explained in common.py):

- turn: 0-based model turn the intervention precedes.
- channel: how the human reached the run -
  - operator: user message with source="operator" - unsolicited
    steering injected by a human operator (e.g. ACP).
  - input: user message with source="input".
  - input_event: an inspect InputEvent - the human's answer to an
    agent-initiated ask_user / request_input prompt.

  The transcript's first user message is the task prompt, never an
  intervention, whichever channel delivered it.
- content: the full intervention text.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def interventions_df(results: pd.DataFrame) -> pd.DataFrame:
    """human_intervention results -> one row per mid-run human input."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        for i in result_value(r["value"]).get("interventions") or []:
            rows.append({**identity_cols, **i})
    columns = [*IDENTITY_COLS, "turn", "channel", "content"]
    return with_schema(pd.DataFrame(rows, columns=columns))
