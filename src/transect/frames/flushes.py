"""flushes_df: context-window compaction events.

Columns (identity prefix explained in common.py):

- turn: 0-based orchestrator turn the flush precedes (the first
  post-flush orchestrator turn).
- agent_span_id: the sub-agent lane the compaction happened in; None on
  the orchestrator. Charts, card tags and the inferences below read
  orchestrator rows only; a sub-agent's compaction is kept, not drawn.
- type: the compaction kind as recorded (e.g. summary), or
  "token_drop" for detected-not-recorded drops.
- source: who recorded the event (e.g. inspect / openclaw), or
  "synthesized" for detected-not-recorded drops.
- tokens_before / tokens_after: window size around the flush; None =
  not reported and not inferrable.
- tokens_after_inferred: tokens_after came from the first non-gap
  main-lane turn after the flush, not the event itself.
- role: the model role whose conversation was compacted, when recorded.
- metadata: the complete recorded compaction event metadata (object), or
  None.
- strategy: the recorded strategy name (e.g. CompactionSummary).
- messages_before / messages_after: recorded message counts, nullable integers.
- trigger: the recorded trigger (e.g. forced / threshold).
- compaction_prompt: the summarization call's formatted prompt, verbatim
  as the model saw it.
- compaction_nudge: the pre-compaction save-to-memory warning, verbatim.
  Both text columns are Inspect eval-only (scanners/compaction.py says
  how they are located). role, strategy, trigger and the two text
  columns are nullable strings; missing means the source recorded no
  such fact. Synthesized drops carry none of them.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import (
    IDENTITY_COLS,
    identity,
    result_value,
    with_schema,
)

_METADATA_COLUMNS = ("strategy", "messages_before", "messages_after", "trigger")
_RECORDED_COLUMNS = ("role", "metadata", "compaction_prompt", "compaction_nudge")
_TEXT_COLUMNS = ("role", "strategy", "trigger", "compaction_prompt", "compaction_nudge")


def flushes_df(results: pd.DataFrame, token_timeline: pd.DataFrame) -> pd.DataFrame:
    """context_flush scanner results -> one row per compaction event.

    Takes ``token_timeline`` because recorded compaction events alone
    do not say what the window looked like: the per-turn ``context``
    series supplies (a) the ``tokens_after`` inference when the event
    omits it, and (b) the detection of unrecorded compactions - the
    0.6x sustained-drop scan over the orchestrator's context series
    synthesizes flush rows the source never emitted. Both read the
    orchestrator's rows only (``turn`` non-null): a sub-agent lane is
    its own conversation and never supplies an orchestrator reading.

    A flush never empties the window to zero: when the event omits
    tokens_after, it is inferred from the first non-gap orchestrator
    turn at/after the flush.
    """
    per_turn_of: dict[str, pd.DataFrame] = {}
    for group_key, group in token_timeline.groupby("transcript_id", sort=False):
        per_turn_of[str(group_key)] = group[group.turn.notna()].sort_values("turn")

    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        turns = per_turn_of.get(identity_cols["transcript_id"])
        for f in result_value(r["value"]).get("flushes") or []:
            metadata = f["metadata"] or {}
            # indexed, not .get: a store scanned before these fields existed
            # must fail loudly here (it needs a re-scan, not a fallback)
            details = {name: f[name] for name in _RECORDED_COLUMNS}
            details.update({name: metadata.get(name) for name in _METADATA_COLUMNS})
            f = {**f, "tokens_after_inferred": False}
            if (
                not f.get("tokens_after")
                and f.get("tokens_before")
                and turns is not None
                and f.get("agent_span_id") is None
            ):
                observed = turns[(turns.turn >= f["turn"]) & turns.context.notna()]
                if len(observed):
                    f["tokens_after"] = int(observed.context.iloc[0])
                    f["tokens_after_inferred"] = True
            rows.append({**identity_cols, **f, **details})
    # Synthesize flushes from context drops: a drop counts when context
    # falls below 0.6x the previous non-gap turn and stays below.
    recorded_turns: dict[str, set[int]] = {}
    for row in rows:
        if row.get("agent_span_id") is None:
            recorded_turns.setdefault(row["transcript_id"], set()).add(int(row["turn"]))
    for transcript_id, per_turn in per_turn_of.items():
        if not len(per_turn):
            continue
        identity_row = per_turn.iloc[0]
        identity_cols = {ours: identity_row.get(ours) for ours in IDENTITY_COLS}
        nearby = recorded_turns.get(transcript_id, set())
        rows.extend(
            {**identity_cols, "agent_span_id": None, **drop}
            for drop in _synthesized_drops(per_turn, nearby)
        )
    columns = [
        *IDENTITY_COLS,
        "turn",
        "agent_span_id",
        "type",
        "source",
        "tokens_before",
        "tokens_after",
        "tokens_after_inferred",
        *_METADATA_COLUMNS,
        *_RECORDED_COLUMNS,
    ]
    df = pd.DataFrame(rows, columns=columns)
    # pin nullable numerics and strings, so an all-missing column is not
    # inferred as float NaN
    df = df.astype(
        {
            **dict.fromkeys(
                ("tokens_before", "tokens_after", "messages_before", "messages_after"),
                "Int64",
            ),
            **dict.fromkeys(_TEXT_COLUMNS, "string"),
            "agent_span_id": "string",
        }
    )
    return with_schema(df)


def _synthesized_drops(orchestrator_turns: pd.DataFrame, nearby: set[int]):
    """The 0.6x sustained-drop scan over the orchestrator's context series."""
    ctx = orchestrator_turns[["turn", "context"]].dropna()
    turns = ctx.turn.tolist()
    values = ctx.context.tolist()
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if cur >= 0.6 * prev:
            continue
        nxt = values[i + 1] if i + 1 < len(values) else None
        if nxt is not None and nxt >= 0.6 * prev:
            continue  # transient dip, not a reset
        turn = int(turns[i])
        if any(abs(turn - r) <= 1 for r in nearby):
            continue
        yield {
            "turn": turn,
            "type": "token_drop",
            "source": "synthesized",
            "tokens_before": int(prev),
            "tokens_after": int(cur),
            "tokens_after_inferred": False,
        }
