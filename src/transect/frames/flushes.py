"""flushes_df: context-window compaction events.

Columns (identity prefix explained in common.py):

- turn: 0-based main-lane turn the flush precedes (the first
  post-flush main-lane turn), for a flush in any lane.
- agent_span_id: the sub-agent lane the compaction happened in; None on
  the orchestrator.
- lane_turn: the first post-flush turn of the compacted lane itself
  (token_timeline's lane_turn): equal to turn on the orchestrator, the
  sub-agent's own ordinal otherwise.
- type: the compaction kind as recorded (e.g. summary), or
  "token_drop" for detected-not-recorded drops.
- source: who recorded the event (e.g. inspect / openclaw), or
  "synthesized" for detected-not-recorded drops; detection runs over
  every lane's own context series, the orchestrator's and each
  sub-agent's.
- tokens_before / tokens_after: window size around the flush; None =
  not reported and not inferrable.
- tokens_after_inferred: tokens_after came from the first non-gap turn
  of the compacted lane after the flush, not the event itself.
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
    lane_series,
    result_value,
    with_schema,
)

_METADATA_COLUMNS = ("strategy", "messages_before", "messages_after", "trigger")
_RECORDED_COLUMNS = ("role", "metadata", "compaction_prompt", "compaction_nudge")
_INT_COLUMNS = (
    "lane_turn",
    "tokens_before",
    "tokens_after",
    "messages_before",
    "messages_after",
)
_TEXT_COLUMNS = (
    "agent_span_id",
    "role",
    "strategy",
    "trigger",
    "compaction_prompt",
    "compaction_nudge",
)


def flushes_df(results: pd.DataFrame, token_timeline: pd.DataFrame) -> pd.DataFrame:
    """context_flush scanner results -> one row per compaction event.

    Takes ``token_timeline`` because recorded compaction events alone
    do not say what the window looked like: the per-turn ``context``
    series supplies (a) the ``tokens_after`` inference when the event
    omits it, and (b) the detection of unrecorded compactions - the
    0.6x sustained-drop scan over each lane's context series
    synthesizes flush rows the source never emitted. Each lane is its
    own conversation, so both read the compacted lane's rows only, and
    a switch between lanes is never mistaken for a flush.

    A flush never empties the window to zero: when the event omits
    tokens_after, it is inferred from the first non-gap turn of its own
    lane at/after the flush.
    """
    lanes_of: dict[str, dict[str, pd.DataFrame]] = {}
    for group_key, group in token_timeline.groupby("transcript_id", sort=False):
        lanes_of[str(group_key)] = _lane_frames(group)

    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        lanes = lanes_of.get(identity_cols["transcript_id"], {})
        for f in result_value(r["value"]).get("flushes") or []:
            metadata = f["metadata"] or {}
            # indexed, not .get: a store scanned before these fields existed
            # must fail loudly here (it needs a re-scan, not a fallback)
            details = {name: f[name] for name in _RECORDED_COLUMNS}
            details.update({name: metadata.get(name) for name in _METADATA_COLUMNS})
            f = {**f, "lane_turn": f["lane_turn"], "tokens_after_inferred": False}
            lane = lanes.get(_lane_key(f.get("agent_span_id")))
            if (
                not f.get("tokens_after")
                and f.get("tokens_before")
                and lane is not None
            ):
                observed = lane[
                    (lane.lane_turn >= f["lane_turn"]) & lane.context.notna()
                ]
                if len(observed):
                    f["tokens_after"] = int(observed.context.iloc[0])
                    f["tokens_after_inferred"] = True
            rows.append({**identity_cols, **f, **details})
    # Synthesize flushes from context drops, lane by lane: a drop counts
    # when context falls below 0.6x the previous non-gap turn of the
    # same lane and stays below.
    recorded: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (row["transcript_id"], _lane_key(row.get("agent_span_id")))
        recorded.setdefault(key, set()).add(int(row["lane_turn"]))
    for transcript_id, lanes in lanes_of.items():
        for lane_key, lane in lanes.items():
            identity_row = lane.iloc[0]
            identity_cols = {ours: identity_row.get(ours) for ours in IDENTITY_COLS}
            nearby = recorded.get((transcript_id, lane_key), set())
            rows.extend(
                {**identity_cols, **drop} for drop in _synthesized_drops(lane, nearby)
            )
    columns = [
        *IDENTITY_COLS,
        "turn",
        "agent_span_id",
        "lane_turn",
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
            **dict.fromkeys(_INT_COLUMNS, "Int64"),
            **dict.fromkeys(_TEXT_COLUMNS, "string"),
        }
    )
    return with_schema(df)


def _lane_key(agent_span_id) -> str:
    return (
        "__main__"
        if agent_span_id is None or pd.isna(agent_span_id)
        else str(agent_span_id)
    )


def _lane_frames(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One transcript's token rows split by lane (`lane_series`), each in
    event order with ``axis_turn``: the main-lane turn the axis was on
    when the row's call happened (the row's own turn on the main lane,
    the count of main-lane turns before it otherwise). Off-axis calls
    (init/scorer: one-row lanes) are left out."""
    ordered = group.copy()
    # main-lane turns are contiguous from 0 in event order, so "main
    # turns before this row" is the row's own turn on the main lane and
    # the axis position of any other row
    on_axis = ordered.turn.notna().astype(int)
    ordered["axis_turn"] = on_axis.cumsum() - on_axis
    lanes: dict[str, pd.DataFrame] = {}
    for key, lane in ordered.groupby(lane_series(ordered), sort=False):
        name = str(key)
        if name.startswith("__offaxis_"):
            continue
        lanes[name] = lane
    return lanes


def _synthesized_drops(lane: pd.DataFrame, nearby: set[int]):
    """The 0.6x sustained-drop scan over one lane's context series,
    ``nearby`` being the lane turns of that lane's recorded flushes."""
    ctx = lane[["axis_turn", "lane_turn", "agent_span_id", "context"]].dropna(
        subset=["lane_turn", "context"]
    )
    axis_turns = ctx.axis_turn.tolist()
    lane_turns = ctx.lane_turn.tolist()
    span_ids = ctx.agent_span_id.tolist()
    values = ctx.context.tolist()
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if cur >= 0.6 * prev:
            continue
        nxt = values[i + 1] if i + 1 < len(values) else None
        if nxt is not None and nxt >= 0.6 * prev:
            continue  # transient dip, not a reset
        lane_turn = int(lane_turns[i])
        if any(abs(lane_turn - r) <= 1 for r in nearby):
            continue
        span_id = span_ids[i]
        yield {
            "turn": int(axis_turns[i]),
            "agent_span_id": None if pd.isna(span_id) else str(span_id),
            "lane_turn": lane_turn,
            "type": "token_drop",
            "source": "synthesized",
            "tokens_before": int(prev),
            "tokens_after": int(cur),
            "tokens_after_inferred": False,
        }
