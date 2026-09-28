"""interventions_df: mid-run human interventions.

Columns (identity prefix explained in common.py):

- turn: 0-based model turn the intervention precedes.
- channel: how the human reached the run -
  - operator: user message with source="operator" - unsolicited
    steering injected by a human operator (ACP, OpenClaw inbound
    messages).
  - input: user message with source="input".
  - input_event: an inspect InputEvent - the human's answer to an
    agent-initiated ask_user / request_input prompt, or a console
    input_screen recording.
  - approval: an inspect ApprovalEvent decided by the human approver.

  The transcript's first user message is the task prompt, never an
  intervention, whichever channel delivered it.
- content: the human's own text - the message, the answer, or the
  approval explanation.
- prompt: what the run put to the human, when the source records it
  (the ask_user question, the tool call awaiting approval); None on
  human-initiated channels and console recordings.
- outcome: how an agent-initiated exchange concluded - the InputEvent
  outcome (accepted / declined / cancelled) or the approval decision
  (approve / modify / reject / escalate / terminate); None otherwise,
  including older logs that never recorded one.
- initiator: "human" (operator, input) or "agent" (input_event,
  approval) - who opened the exchange.
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
    columns = [
        *IDENTITY_COLS,
        "turn",
        "channel",
        "content",
        "prompt",
        "outcome",
        "initiator",
    ]
    return with_schema(pd.DataFrame(rows, columns=columns))
