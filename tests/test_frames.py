"""The frames contract: every scanner result projects into a pandas
dataframe with the shared identity prefix, a schema_version, and the
column semantics its module docstring declares.

Mechanical frames are checked end to end from a real $0 run over the
demo log; judged frames from real scanner Results assembled into raw
results rows (Scout's results tables).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from helpers import (
    CLOSED,
    PHASES_SPEC,
    StubTranscript,
    group,
    model_turn,
    narrate_answer,
    narrative,
    run_item,
    run_scan,
    scripted_judge,
    seg,
    seg_answer,
    verify_answer,
    voter,
)
from inspect_ai.log import read_eval_log
from test_subagent_classification import span_item

from transect.api import _run
from transect.frames import (
    label_definitions_df,
    phase_turn_votes_df,
    phase_turns_df,
    phases_df,
    subagent_votes_df,
    subagents_df,
    token_timeline_df,
    turn_groups_df,
)
from transect.frames.common import SCHEMA_VERSION
from transect.frames.flushes import flushes_df
from transect.scanners.cohort import judge_setup
from transect.scanners.helpers import orchestrator_turns
from transect.scanners.phases import decision_phases
from transect.scanners.subagents import subagent_classification
from transect.spec import Spec

IDENTITY = ["sample_id", "task_set", "epoch", "transcript_id", "agent"]
MECHANICAL = [
    "token_timeline",
    "flushes",
    "interventions",
    "lane_activity",
    "transcript_info",
]
JUDGED = [
    "phases",
    "phase_turns",
    "turn_groups",
    "phase_turn_votes",
    "subagent_votes",
    "label_definitions",
]
DEMO_LOG = Path(__file__).parents[1] / "examples" / "logs" / "house_price_demo.eval"


def raw_row(result, metadata=None, transcript_id="tr1", input_ids=None):
    """One Scout-results-table row from a scanner Result: item facts
    ride the Result's metadata column, item identity the input_ids
    column (present even when a scan errors and no Result exists)."""
    if input_ids is None:
        span = (metadata or {}).get("agent_span_id")
        input_ids = [span or transcript_id]
    return {
        "transcript_task_id": "s1",
        "transcript_task_set": "ts",
        "transcript_task_repeat": 1,
        "transcript_id": transcript_id,
        "transcript_agent": "react",
        "input_ids": json.dumps(input_ids),
        "metadata": json.dumps(metadata) if metadata is not None else None,
        "value": json.dumps(result.value),
        "label": result.label,
        "answer": result.answer,
        "explanation": result.explanation,
        "error": None,
        "error_type": None,
    }


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    """One $0 run over the committed demo log, shared by the module."""
    if not DEMO_LOG.exists():
        pytest.skip(
            "demo log missing - run: python tests/fixtures/generate_demo_eval.py"
        )
    scans = tmp_path_factory.mktemp("scans")
    return _run(logs=str(DEMO_LOG.parent), spec=Spec(), scans_dir=str(scans))


@pytest.fixture(scope="module")
def raw_phases():
    """A real decision_phases Result (k-roll + verifier relabel) as a
    raw results row."""
    judge = scripted_judge(
        seg_answer(seg(0, 1, "setup", 0.4), seg(2, 3, "experiment", 0.9)),
        seg_answer(seg(0, 1, "setup", 0.5), seg(2, 3, "experiment", 0.9)),
        verify_answer(
            {
                "phase_index": 0,
                "phase": "experiment",
                "confidence": 0.9,
                "explanation": "actually experiments",
            }
        ),
    )
    result = run_scan(
        decision_phases(PHASES_SPEC, judge, k_rolls=2, verify=True, narrate=False),
        [model_turn(f"t{i}") for i in range(4)],
    )
    return pd.DataFrame([raw_row(result)])


@pytest.mark.parametrize("name", MECHANICAL)
def test_mechanical_frames_carry_identity_and_schema(demo_results, name):
    """Every mechanical frame from a real run has the identity prefix,
    the schema version, and at least one row for the demo log."""
    frame = demo_results.frames()[name]
    assert set(IDENTITY) <= set(frame.columns)
    assert (frame.schema_version == SCHEMA_VERSION).all()
    assert len(frame) > 0
    assert set(frame.transcript_id) == set(demo_results.token_timeline.transcript_id)


@pytest.mark.parametrize("name", JUDGED)
def test_judged_frames_are_empty_but_typed_on_a_mechanical_run(demo_results, name):
    """Without judge_models the judged frames exist with their full
    column set and zero rows - never missing, never half-shaped."""
    frame = demo_results.frames()[name]
    assert len(frame) == 0
    assert set(IDENTITY) <= set(frame.columns)


def test_token_timeline_numbers_orchestrator_turns_only(demo_results):
    """Ten orchestrator turns take 0..9; the seven sub-agent turns carry
    no turn but keep their lane and token views; totals are unchanged."""
    timeline = demo_results.token_timeline
    main = timeline[timeline.turn.notna()].sort_values("turn")
    assert main.turn.tolist() == list(range(10))
    assert int(timeline.turn.isna().sum()) == 7
    assert set(timeline.agent_lane.dropna()) == {"eda", "alt_model", "reviewer"}
    spots = main.set_index("turn")
    assert (spots.loc[0, "new_work"], spots.loc[0, "context"]) == (1020, 926)
    # turn 6 is the summarization call; turn 7 runs on the compacted window
    assert (spots.loc[6, "new_work"], spots.loc[6, "context"]) == (3588, 2667)
    assert (spots.loc[7, "new_work"], spots.loc[7, "context"]) == (1491, 1228)
    assert (spots.loc[9, "new_work"], spots.loc[9, "context"]) == (2005, 1761)
    assert timeline.new_work.sum() == 27221
    assert timeline[timeline.agent_lane == "eda"].lane_turn.tolist() == [0, 1]
    assert timeline.timestamp.str.startswith("20").all()


def test_subagent_spans_are_placed_on_the_orchestrator_axis(demo_results):
    """Each handoff span's anchor and end turns are its transfer turn
    (the orchestrator waits), it spawns at that turn, and it keeps its
    per-span spend."""
    subagents = demo_results.frames()["subagents"]
    assert subagents.label.isna().all()
    expected = pd.DataFrame(
        [
            ("eda", 3, 3, 3, "timestamp", True, 2, 1750.0),
            ("alt_model", 5, 5, 5, "timestamp", True, 2, 2701.0),
            ("reviewer", 8, 8, 8, "timestamp", True, 3, 6242.0),
        ],
        columns=[
            "agent_lane",
            "spawn_turn",
            "anchor_turn",
            "end_turn",
            "turn_source",
            "end_recorded",
            "tool_calls",
            "new_work",
        ],
    )
    pd.testing.assert_frame_equal(
        subagents[expected.columns].reset_index(drop=True),
        expected,
        check_dtype=False,
    )
    # each span ran inside the wall-clock interval of its transfer turn
    assert (subagents.started_at < subagents.ended_at).all()
    assert subagents.started_at.str.startswith("20").all()
    timeline = demo_results.token_timeline
    lane_sums = timeline.groupby("agent_lane").new_work.sum()
    assert dict(zip(subagents.agent_lane, subagents.new_work, strict=True)) == dict(
        lane_sums
    )


def test_flushes_frame_carries_the_demo_compaction(demo_results):
    """The demo's one compaction lands with its exact recorded facts."""
    flushes = demo_results.flushes
    expected = pd.DataFrame(
        [
            {
                "turn": 7,
                "agent_span_id": pd.NA,
                "lane_turn": 7,
                "type": "summary",
                "source": "inspect",
                "tokens_before": 1906,
                "tokens_after": 658,
                "strategy": "CompactionSummary",
                "trigger": "threshold",
            }
        ]
    )
    pd.testing.assert_frame_equal(
        flushes[expected.columns].reset_index(drop=True),
        expected,
        check_dtype=False,
    )
    # the configured instructions are substituted into the recorded prompt
    assert (
        "Keep the fold protocol and every CV RMSLE figure."
        in (flushes.compaction_prompt.iloc[0])
    )
    assert flushes.compaction_nudge.iloc[0].startswith("Context compaction approaching")


def test_interventions_frame_carries_the_planted_interventions(demo_results):
    """Both planted interventions land with their turn, channel, and
    full content."""
    interventions = demo_results.interventions.sort_values("turn")
    assert list(zip(interventions.turn, interventions.channel, strict=True)) == [
        (7, "operator"),
        (9, "input_event"),
    ]
    assert interventions.content.iloc[0].startswith("Operator note: we are time-boxed")
    assert interventions.initiator.tolist() == ["human", "agent"]
    assert interventions.prompt.iloc[1].startswith("Submit the blended predictions")
    assert interventions.content.iloc[1] == "y"


def test_flush_and_intervention_scanners_share_one_turn_axis(demo_results):
    """Three independent readings of where the demo's compaction sits agree:
    the flush turn from the event walk, the first model input that carried
    the compaction summary message, and the operator note placed after it."""
    log = read_eval_log(str(DEMO_LOG), resolve_attachments=True)
    sample = log.samples[0]
    turns = orchestrator_turns(StubTranscript(sample.events))
    (summary,) = [m for m in sample.messages if (m.metadata or {}).get("summary")]
    first_seen = next(
        turn for turn, e, _ in turns if summary.id in {m.id for m in e.input}
    )
    (flush_turn,) = demo_results.flushes.turn
    interventions = demo_results.interventions
    (note_turn,) = interventions[interventions.channel == "operator"].turn
    assert flush_turn == first_seen == note_turn == 7


def test_lane_activity_frame_anchors_tool_activity_to_orchestrator_turns(
    demo_results,
):
    """Each sub-agent's tool events collapse onto the orchestrator turn
    that spawned it (the handoff turn); the main lane never appears."""
    expected = pd.DataFrame(
        [(3, "eda", 2, 3), (5, "alt_model", 2, 5), (8, "reviewer", 3, 8)],
        columns=["turn", "agent_lane", "tool_calls", "span_end_turn"],
    )
    got = demo_results.lane_activity.sort_values("turn")
    pd.testing.assert_frame_equal(
        got[expected.columns].reset_index(drop=True), expected, check_dtype=False
    )


def test_transcript_info_describes_the_run(demo_results):
    row = demo_results.transcript_info.iloc[0]
    assert row.sample_id == "house-price"
    assert row.model == "mockllm/model"
    assert row.epoch == 1
    assert row.task_name == "house_price_demo"
    assert row.wallclock_seconds == 1886.0
    assert row.success is None
    assert row.message_count == 35
    assert row.total_tokens > 0
    assert row.error is None and row.limit is None
    assert str(row.source_file).endswith(".eval")


def test_eval_setup_captures_prompts_scaffold_and_header(demo_results):
    """The intro's setup facts ride transcript_info."""
    row = demo_results.transcript_info.iloc[0]
    assert bool(row.header_available)
    assert row.scaffold_prompt.startswith("You are the lead engineer")
    assert "bash" in row.tools
    assert row.task_message
    assert row.truncation == "disabled"
    assert row.message_limit is None
    assert row.dataset_samples == 1
    assert row.epochs == 1


def test_phases_frame_records_labels_reliability_and_the_verifier(raw_phases):
    """The stitched phases project with confidence, agreement, the
    deciding regime, and the verifier's overturn trail."""
    frame = phases_df(raw_phases, phase_turns=phase_turns_df(raw_phases))
    overturned = frame[frame.overturned]
    assert len(overturned) == 1
    row = overturned.iloc[0]
    assert row.phase == "experiment"
    assert row.original_label == "setup"
    assert row.verifier_trigger == "low_confidence"
    assert row.confidence_source == "verifier"
    assert frame.judge_agreement.notna().all()
    # the scanner's own audit counts project run-constant, including
    # the no-verdict review that row aggregation could never represent
    audit = frame.iloc[0]
    assert audit.verifier_n_low_confidence == 1
    assert audit.verifier_n_relabelled == 1
    assert audit.verifier_n_random_sample == 1
    assert audit.verifier_n_no_verdict == 1
    assert audit.verifier_n_weak_relabel == 0


def test_phase_turns_and_votes_expose_the_per_member_ballots(raw_phases):
    """phase_turns carries the decided per-turn labels; phase_turn_votes
    keeps each roll's own vote behind them."""
    turns = phase_turns_df(raw_phases)
    assert sorted(turns.turn) == [0, 1, 2, 3]
    votes = phase_turn_votes_df(raw_phases)
    assert set(votes.roll) == {0, 1}
    assert len(votes) == 8


def test_label_definitions_project_the_judged_rubric(raw_phases):
    """The vocabulary the judge classified against lands one row per
    label, with the reserved escape labels flagged."""
    defs = label_definitions_df(raw_phases, pd.DataFrame())
    assert set(defs.label) >= {"setup", "experiment", "none_of_the_above"}
    assert defs[defs.label == "none_of_the_above"].reserved.iloc[0]


def test_subagents_frame_maps_ballots_and_the_vote():
    """A cohort-classified span projects the voted label, per-ballot
    rows, and who decided."""
    members = [
        voter("mockllm/model", "literature_survey", 0.9),
        voter("mockllm/model2", "literature_survey", 0.7),
        voter("mockllm/model3", "experiment_run", 0.95),
    ]
    result = run_item(subagent_classification(CLOSED, members), span_item())
    metadata = {
        "agent_span_id": "A",
        "agent_lane": "lit",
        "span_task_source": "spawn_prompt",
        "span_task": "Survey related work",
        "span_task_truncated": False,
    }
    raw = pd.DataFrame([raw_row(result, metadata=metadata)])
    subagents = subagents_df(raw)
    row = subagents.iloc[0]
    assert row.label == "literature_survey"
    assert row.label_source == "majority_vote"
    assert row.agent_span_id == "A"
    votes = subagent_votes_df(raw)
    assert len(votes) == 3
    assert set(votes.label) == {"literature_survey", "experiment_run"}


def test_turn_groups_project_the_narrated_partition():
    """Narrator groups land one row each, partitioning their phase."""
    judge = scripted_judge(
        seg_answer(seg(0, 3, "setup", 0.9)),
        narrate_answer(
            narrative(0, groups=[group(0, 1, title="install"), group(2, 3)])
        ),
    )
    result = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False),
        [model_turn(f"t{i}") for i in range(4)],
    )
    groups = turn_groups_df(pd.DataFrame([raw_row(result)]))
    assert [(g.turn_start, g.turn_end) for g in groups.itertuples()] == [(0, 1), (2, 3)]
    assert groups.title.iloc[0] == "install"


def test_frame_builders_tolerate_empty_results():
    """An empty raw results table produces an empty, fully-columned
    frame (the no-judge run path)."""
    empty = pd.DataFrame()
    for builder in (phases_df, phase_turns_df, subagents_df, token_timeline_df):
        frame = builder(empty)
        assert len(frame) == 0
        assert "transcript_id" in frame.columns


def test_phases_frame_marks_audit_counts_absent_when_verifier_off():
    """verifier_n_* are None (not 0) when the verifier never ran, so
    "off" can never read as "ran, zero selected"."""
    judge = scripted_judge(seg_answer(seg(0, 3, "setup", 0.9)))
    result = run_scan(
        decision_phases(PHASES_SPEC, judge, verify=False, narrate=False),
        [model_turn(f"t{i}") for i in range(4)],
    )
    frame = phases_df(pd.DataFrame([raw_row(result)]))
    cols = [c for c in frame.columns if c.startswith("verifier_n_")]
    assert len(cols) == 8
    assert frame[cols].isna().all().all()


def test_solo_default_model_rows_read_the_judge_from_usage():
    """The models=None path stamps an empty roster; the frame falls
    back to the recorded model usage for the judge name."""
    block = judge_setup([], 1, verifier_armed=True, verifier_model=None)
    value = {
        "confidence": 0.9,
        "label_source": "single_judge",
        "judge": block,
        "verifier": {"ran": False, "model": None},
    }
    result = SimpleNamespace(
        value=value, label="explorer", answer=None, explanation="e"
    )
    row = raw_row(result, metadata={"agent_span_id": "sp1"})
    row["scan_model_usage"] = json.dumps({"mockllm/model": {"total_tokens": 1}})
    frame = subagents_df(pd.DataFrame([row]))
    r = frame.iloc[0]
    assert r.judge_models == "mockllm/model"
    assert r.judge_regime == "solo"
    assert bool(r.verifier_armed) is True


@pytest.mark.parametrize(
    "storage,category,status",
    [
        ("absent", None, "error"),
        ("object", None, "error"),
        ("parquet", None, "error"),
        ("parquet", "refusal", "refusal"),
        ("parquet", "other", "error"),
    ],
)
def test_errored_solo_rows_project_absent_judge_columns(
    tmp_path, storage, category, status
):
    """An errored solo scan records no value, so there is no judge
    block: the judge columns project as absent, dtypes intact."""
    result = SimpleNamespace(value={}, label=None, answer=None, explanation=None)
    # an errored scan records no Result, so no Result.metadata - the
    # span id must still arrive via the store's input_ids column
    row = raw_row(result, metadata=None, input_ids=["sp1"])
    row["scan_error"] = "boom"
    if storage != "absent":
        row["scan_error_type"] = category
    raw = pd.DataFrame([row])
    if storage == "parquet":
        raw["scan_error_type"] = raw["scan_error_type"].astype("string[pyarrow]")
        path = tmp_path / "errors.parquet"
        raw.to_parquet(path)
        raw = pd.read_parquet(path, dtype_backend="pyarrow")
    frame = subagents_df(raw)
    r = frame.iloc[0]
    assert r.agent_span_id == "sp1"
    assert r.status == status
    assert pd.isna(r.judge_regime)
    assert pd.isna(r.n_models)
    assert pd.isna(r.verifier_armed)


def test_context_drops_synthesize_flushes_with_the_documented_fences():
    """The 0.6x sustained-drop scan invents reviewer-facing rows, so
    its three fences are pinned: a sustained drop yields a row, a
    transient dip does not, and a drop adjacent to a recorded flush is
    suppressed rather than double-reported."""
    timeline = pd.DataFrame(
        {
            "transcript_id": ["tr1"] * 10,
            "turn": range(10),
            "lane_turn": range(10),
            "context": [1000, 950, 300, 280, 900, 400, 950, 900, 350, 300],
            "agent_span_id": [None] * 10,
            "agent_lane": [None] * 10,
        }
    )
    recorded = SimpleNamespace(
        value={
            "flushes": [
                {
                    "turn": 8,
                    "agent_span_id": None,
                    "lane_turn": 8,
                    "type": "context",
                    "source": "inspect",
                    "tokens_before": 900,
                    "tokens_after": 350,
                    "role": None,
                    "metadata": None,
                    "compaction_prompt": None,
                    "compaction_nudge": None,
                }
            ]
        },
        label=None,
        answer=None,
        explanation=None,
    )
    frame = flushes_df(pd.DataFrame([raw_row(recorded)]), timeline)
    assert sorted(frame.turn) == [2, 8]
    synthesized = frame[frame.source == "synthesized"].iloc[0]
    assert synthesized.turn == 2
    assert (synthesized.tokens_before, synthesized.tokens_after) == (950, 300)
    # turn 5 dipped to 400 but recovered to 950: transient, no row;
    # turn 8's drop is the recorded flush itself: suppressed


def test_sub_agent_lanes_synthesize_their_own_drops():
    """A context reset inside a sub-agent lane is detected on that lane's
    own series and lands tagged with the lane, the lane turn it precedes,
    and the orchestrator turn the axis was on at the time."""
    timeline = pd.DataFrame(
        {
            "transcript_id": ["tr1"] * 5,
            "turn": [0, 1, None, None, 2],
            "lane_turn": [0, 1, 0, 1, 2],
            "context": [1000, 1100, 900, 100, 1200],
            "agent_span_id": [None, None, "A", "A", None],
            "agent_lane": [None, None, "a", "a", None],
        }
    ).astype({"turn": "Int64", "lane_turn": "Int64"})
    frame = flushes_df(pd.DataFrame(), timeline)
    (drop,) = frame.itertuples()
    assert (drop.agent_span_id, drop.turn, drop.lane_turn) == ("A", 2, 1)
    assert (drop.source, drop.tokens_before, drop.tokens_after) == (
        "synthesized",
        900,
        100,
    )


def test_a_sub_agent_flush_infers_tokens_after_from_its_own_lane():
    """A recorded sub-agent compaction without tokens_after reads the
    next window of its own lane, and suppresses the drop it explains."""
    timeline = pd.DataFrame(
        {
            "transcript_id": ["tr1"] * 3,
            "turn": [0, None, None],
            "lane_turn": [0, 0, 1],
            "context": [1000, 800, 150],
            "agent_span_id": [None, "A", "A"],
            "agent_lane": [None, "a", "a"],
        }
    ).astype({"turn": "Int64", "lane_turn": "Int64"})
    recorded = SimpleNamespace(
        value={
            "flushes": [
                {
                    "turn": 1,
                    "agent_span_id": "A",
                    "lane_turn": 1,
                    "type": "summary",
                    "source": "inspect",
                    "tokens_before": 800,
                    "tokens_after": 0,
                    "role": None,
                    "metadata": None,
                    "compaction_prompt": None,
                    "compaction_nudge": None,
                }
            ]
        },
        label=None,
        answer=None,
        explanation=None,
    )
    frame = flushes_df(pd.DataFrame([raw_row(recorded)]), timeline)
    (flush,) = frame.itertuples()
    assert (flush.tokens_after, flush.tokens_after_inferred) == (150, True)
    assert flush.source == "inspect"


def test_a_recorded_flush_without_tokens_after_infers_it():
    """An export that omits tokens_after gets it from the next real
    context reading, flagged as inferred rather than passed off as
    recorded."""
    timeline = pd.DataFrame(
        {
            "transcript_id": ["tr1"] * 5,
            "turn": range(5),
            "lane_turn": range(5),
            "context": [1000, 900, 850, 200, 190],
            "agent_span_id": [None] * 5,
            "agent_lane": [None] * 5,
        }
    )
    recorded = SimpleNamespace(
        value={
            "flushes": [
                {
                    "turn": 3,
                    "agent_span_id": None,
                    "lane_turn": 3,
                    "type": "context",
                    "source": "inspect",
                    "tokens_before": 850,
                    "tokens_after": 0,
                    "role": None,
                    "metadata": None,
                    "compaction_prompt": None,
                    "compaction_nudge": None,
                }
            ]
        },
        label=None,
        answer=None,
        explanation=None,
    )
    frame = flushes_df(pd.DataFrame([raw_row(recorded)]), timeline)
    row = frame[frame.source == "inspect"].iloc[0]
    assert (int(row.tokens_after), bool(row.tokens_after_inferred)) == (200, True)
    # and the same drop is not also synthesized (suppressed as nearby)
    assert (frame.source == "synthesized").sum() == 0


def test_phase_rollups_split_orchestrator_and_delegated_spend(raw_phases):
    """Orchestrator spend sums the phase's own turns; delegated spend
    sums the spans spawned inside the phase; the count follows spawns."""
    phase_turns = pd.DataFrame(
        {"transcript_id": ["tr1"] * 4, "turn": [0, 1, 2, 3], "phase_index": [0] * 4}
    )
    timeline = pd.DataFrame(
        {
            "transcript_id": ["tr1"] * 6,
            "turn": [0, 1, None, None, 2, 3],
            "new_work": [10, 20, 500, 600, 30, 40],
        }
    ).astype({"turn": "Int64"})
    subagents = pd.DataFrame(
        {
            "transcript_id": ["tr1", "tr1", "tr1"],
            "agent_span_id": ["A", "B", "C"],
            "spawn_turn": [1, 3, 9],
            "new_work": [1100.0, None, 5.0],
        }
    )
    phases = phases_df(
        raw_phases,
        phase_turns=phase_turns,
        token_timeline=timeline,
        subagents=subagents,
    )
    (phase,) = phases.itertuples()
    assert phase.new_work_tokens == 100
    assert phase.delegated_new_work_tokens == 1100
    assert phase.n_subagents == 2


def test_phase_delegated_spend_is_absent_not_zero_without_usage(raw_phases):
    """Tool-only lanes spawned in a phase count, but contribute NA spend."""
    subagents = pd.DataFrame(
        {
            "transcript_id": ["tr1"],
            "agent_span_id": ["A"],
            "spawn_turn": [1],
            "new_work": [None],
        }
    ).astype({"new_work": "Float64"})
    phases = phases_df(raw_phases, subagents=subagents)
    (phase,) = phases.itertuples()
    assert pd.isna(phase.delegated_new_work_tokens)
    assert phase.n_subagents == 1


def test_a_skewed_orchestrator_clock_falls_back_to_event_order():
    """Timestamps that run backwards between turns are unusable: spans
    sit at their spawn turn rather than at a wrong wall-clock position."""
    value = {
        "timeline": [
            {"turn": 0, "timestamp": "2026-01-01T10:00:00", "completed": None},
            {"turn": 1, "timestamp": "2026-01-01T09:00:00", "completed": None},
        ],
        "spans": [
            {
                "agent_span_id": "A",
                "agent_lane": "a",
                "spawn_turn": 1,
                "first_at": "2026-01-01T10:00:10",
                "last_at": "2026-01-01T10:00:20",
                "end_at": None,
                "end_recorded": False,
                "event_order_end_turn": 1,
            }
        ],
        "lane_activity": [],
    }
    raw = pd.DataFrame(
        [
            raw_row(
                SimpleNamespace(value=value, label=None, answer=None, explanation=None)
            )
        ]
    )
    frame = subagents_df(pd.DataFrame(), timeline_results=raw)
    assert frame.turn_source.tolist() == ["event_order"]
    assert frame.anchor_turn.tolist() == [1]


def test_an_off_axis_call_never_seeds_the_orchestrator_context():
    """An init-phase model call is its own conversation: the first
    orchestrator turn's new-work cap starts from an empty window."""
    value = {
        "timeline": [
            {
                "turn": None,
                "lane_turn": None,
                "agent_lane": None,
                "agent_span_id": None,
                "n_tool_calls": 0,
                "timestamp": None,
                "completed": None,
                "input_tokens": 5000,
                "output_tokens": 10,
                "total_tokens": 5010,
                "input_tokens_cache_read": 0,
                "input_tokens_cache_write": 0,
                "reasoning_tokens": 0,
            },
            {
                "turn": 0,
                "lane_turn": 0,
                "agent_lane": None,
                "agent_span_id": None,
                "n_tool_calls": 0,
                "timestamp": None,
                "completed": None,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 170,
                "input_tokens_cache_read": 0,
                "input_tokens_cache_write": 50,
                "reasoning_tokens": 0,
            },
        ],
        "spans": [],
        "lane_activity": [],
    }
    row = raw_row(
        SimpleNamespace(value=value, label=None, answer=None, explanation=None)
    )
    frame = token_timeline_df(pd.DataFrame([row]))
    main = frame[frame.turn.notna()]
    # input + output + the full cache write (window grew from nothing)
    assert main.new_work.tolist() == [170]


def test_an_empty_subagents_frame_means_zero_spawns_not_unknown(raw_phases):
    """A complete structural scan with no spans is a known zero count;
    only a missing input is NA."""
    phases = phases_df(raw_phases, subagents=subagents_df(pd.DataFrame()))
    assert phases.n_subagents.tolist() == [0]
    assert pd.isna(phases.delegated_new_work_tokens.iloc[0])
    absent = phases_df(raw_phases, subagents=None)
    assert pd.isna(absent.n_subagents.iloc[0])
