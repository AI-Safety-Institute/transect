"""Tag-family selection, validation, and the results.turn_tags mount."""

import pandas as pd
import pytest

from transect import Layer
from transect.frames.subagent_votes import subagent_votes_df
from transect.frames.subagents import subagents_df
from transect.tags import RESERVED_TAG_COLUMNS, select_tags, turn_tags_frame

N_TURNS = {"tr1": 4, "tr2": 3}


def cohort_shaped(**overrides) -> pd.DataFrame:
    """A per-turn frame the way a cohort_llm_scanner-built layer emits
    it: the label payload plus our bookkeeping columns."""
    base = {
        "transcript_id": ["tr1", "tr1"],
        "turn": [0, 1],
        "label": ["reading", "coding"],
        "confidence": [0.9, 0.93],
        "label_source": ["majority_vote", "majority_vote"],
        "status": ["ok", "ok"],
        "judge_models": ["m", "m"],
        "judge_regime": ["k_roll", "k_roll"],
        "error": [None, None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_true_on_a_cohort_frame_selects_only_the_label():
    """A layer built on cohort_llm_scanner is safe to tag blind:
    tags=True turns just the judged label into chips. A dict
    renames the family, an explicit list can opt bookkeeping in, and
    numeric columns never become chips."""
    frame = cohort_shaped()
    assert select_tags(Layer(name="a", tags=True), frame) == {"label": "label"}
    assert select_tags(Layer(name="a", tags={"activity": "label"}), frame) == {
        "activity": "label"
    }
    assert select_tags(Layer(name="a", tags=["label", "label_source"]), frame) == {
        "label": "label",
        "label_source": "label_source",
    }
    with pytest.raises(ValueError, match="confidence"):
        select_tags(Layer(name="a", tags=["confidence"]), frame)


def test_reserved_vocabulary_covers_the_cohort_bookkeeping():
    """Drift pin: every non-payload string column our cohort-fed
    frames emit is reserved."""
    votes_cols = set(subagent_votes_df(pd.DataFrame()).columns)
    span_cols = set(subagents_df(pd.DataFrame()).columns)
    payload_or_shape = {
        "label",
        "turn",
        "transcript_id",
        "sample_id",
        "task_set",
        "epoch",
        "agent",
        "agent_span_id",
        "agent_lane",
        "schema_version",
        "spawn_turn",
        "anchor_turn",
        "end_turn",
        "turn_source",
        "end_recorded",
        "started_at",
        "ended_at",
        "tool_calls",
        "busy_seconds",
        "output_tokens",
        "new_work",
        "billable",
    }
    bookkeeping = (votes_cols | span_cols) - payload_or_shape
    missing = {
        c
        for c in bookkeeping
        if c not in RESERVED_TAG_COLUMNS
        and not any(c.endswith(suffix) for suffix in ("_pm", "_truncated"))
        and c
        not in ("confidence", "n_voting", "n_members", "n_models", "k_rolls", "roll")
    }
    assert missing <= RESERVED_TAG_COLUMNS, f"unreserved bookkeeping: {missing}"


def test_mount_is_wide_composes_layers_and_broadcasts():
    """Two tagging layers mount into one wide frame keyed by
    (transcript_id, turn); a source without transcript_id applies to
    every transcript."""
    activity_frame = cohort_shaped()
    shell_frame = pd.DataFrame({"turn": [1], "shell": ["pytest"]})
    activity = Layer(name="act", frame=activity_frame, tags={"activity": "label"})
    shell = Layer(name="sh", frame=shell_frame, tags=True)
    # no tagging layers -> None, not an empty frame (render branches on it)
    assert turn_tags_frame([Layer(name="a", frame=activity_frame)], {}, N_TURNS) is None
    tags = turn_tags_frame(
        [activity, shell],
        {"act": activity_frame, "sh": shell_frame},
        N_TURNS,
    )
    assert tags is not None
    assert set(tags.columns) == {"transcript_id", "turn", "activity", "shell"}
    tr1 = tags[tags.transcript_id == "tr1"].set_index("turn")
    assert tr1.loc[0, "activity"] == "reading"
    assert tr1.loc[1, "shell"] == "pytest"  # broadcast reached tr1
    tr2 = tags[tags.transcript_id == "tr2"].set_index("turn")
    assert tr2.loc[1, "shell"] == "pytest"  # ...and tr2
    assert pd.isna(tr2.get("activity", pd.Series(index=tr2.index)).get(1))


@pytest.mark.parametrize(
    ("layers_frames", "match"),
    [
        (
            lambda: (
                [Layer(name="spans", tags=True)],
                {
                    "spans": pd.DataFrame(
                        {"start_turn": [1], "end_turn": [3], "label": ["burst"]}
                    )
                },
            ),
            "per-turn",
        ),
        (
            lambda: (
                [Layer(name="a", tags=True)],
                {"a": cohort_shaped(turn=[0, 99])},
            ),
            "out of range",
        ),
        (
            lambda: (
                [Layer(name="a", tags=True)],
                {"a": cohort_shaped(turn=[1, 1])},
            ),
            "duplicate",
        ),
        (
            lambda: (
                [
                    Layer(name="a", frame=cohort_shaped(), tags={"x": "label"}),
                    Layer(name="b", frame=cohort_shaped(), tags={"x": "label"}),
                ],
                {"a": cohort_shaped(), "b": cohort_shaped()},
            ),
            "'a'.*'b'|'b'.*'a'",
        ),
        (
            lambda: (
                [Layer(name="a", tags=True)],
                {"a": cohort_shaped(turn=[0.5, 1.5])},
            ),
            "integer",
        ),
        (
            lambda: (
                [Layer(name="a", tags=True)],
                {"a": pd.DataFrame({"turn": [0, 1], "transcript_id": ["tr1", "tr1"]})},
            ),
            "no eligible",
        ),
        (
            lambda: (
                [Layer(name="a", tags=["missing_col"])],
                {"a": cohort_shaped()},
            ),
            "missing_col",
        ),
    ],
    ids=[
        "span-shape-refused",
        "turn-out-of-range",
        "duplicate-turn",
        "family-collision-names-both-layers",
        "non-integer-turn",
        "true-selecting-nothing",
        "explicit-column-absent",
    ],
)
def test_tag_validation_rejects_each_malformed_shape(layers_frames, match):
    layers, frames = layers_frames()
    with pytest.raises(ValueError, match=match):
        turn_tags_frame(layers, frames, N_TURNS)


def test_end_to_end_tags_ride_the_run(layered_run):
    """The shared $0 run: the per-turn layer's string column and the
    data-only carrier's column mount as wide families."""
    results, _ = layered_run
    tags = results.turn_tags
    assert set(tags.columns) == {"transcript_id", "turn", "band", "shell"}
    row0 = tags[tags.turn == 0].iloc[0]
    assert row0.band == "early"
    assert list(tags[tags.turn == 1].shell) == ["pip"]
