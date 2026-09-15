"""Batch judging: N units per cohort_llm_scanner call."""

import re
from collections.abc import AsyncIterator

import pandas as pd
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_scout import Loader, Transcript, loader, scanner as scout_scanner

import transect
from transect.reliability import detect_regime, label_stats
from transect.scanners.cohort import _split_batch_content, batch_item_content

RISK = ["risky", "safe"]


def batch_output(model, *entries):
    return ModelOutput.for_tool_call(
        model, "answer", {"items": list(entries), "explanation": "batch"}
    )


def marked_units(input) -> dict[int, int]:
    """ordinal -> turn number, read off the [ITEM i] blocks (each
    digest line starts with its turn number)."""
    text = "\n".join(getattr(m, "text", "") or "" for m in input)
    return {
        int(i): int(turn)
        for i, turn in re.findall(r"^\[ITEM (\d+)\]\n(\d+):", text, re.MULTILINE)
    }


def test_batch_content_round_trips():
    """The loader's marker format splits back per unit."""
    content = batch_item_content(["4: text a", "7: text b [DELEGATES] x"])
    assert _split_batch_content(content) == {
        1: "4: text a",
        2: "7: text b [DELEGATES] x",
    }


def test_batched_cohort_votes_per_turn(demo_log, tmp_path):
    """Two members, batch=8: one call per member per window, votes
    per turn, a skipped unit as no_answer, bad entries dropped."""
    calls = {"a": 0, "b": 0}

    def member_a(input, tools, tool_choice, config):
        calls["a"] += 1
        units = marked_units(input)
        entries = [
            {"item": i, "label": "risky", "confidence": 0.9, "explanation": "a"}
            for i, turn in units.items()
            if turn != 7  # planted skip: no_answer for this member
        ]
        # dropped by the parser: an undeclared ordinal, and a duplicate
        entries.append(
            {"item": 99, "label": "safe", "confidence": 0.9, "explanation": "x"}
        )
        first = next(iter(units))
        entries.append(
            {"item": first, "label": "safe", "confidence": 0.1, "explanation": "d"}
        )
        return batch_output("mockllm/model", *entries)

    def member_b(input, tools, tool_choice, config):
        calls["b"] += 1
        return batch_output(
            "mockllm/model2",
            *(
                {
                    "item": i,
                    "label": "safe" if turn in (3, 7) else "risky",
                    "confidence": 0.4,
                    "explanation": "b",
                }
                for i, turn in marked_units(input).items()
            ),
        )

    @scout_scanner(loader=transect.reasoning_turns(batch=8))
    def batch_risk(judge_models=None):
        return transect.cohort_llm_scanner(
            question="Is this turn risky?",
            answer=RISK,
            models=judge_models,
            batch=True,
        )

    results = transect.transect(
        logs=str(demo_log.parent),
        spec=transect.Spec(),
        judge_models=[
            get_model("mockllm/model", custom_outputs=member_a, memoize=False),
            get_model("mockllm/model2", custom_outputs=member_b, memoize=False),
        ],
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="batch_risk",
                scanner=batch_risk,
                frame=transect.turns_frame,
                tags={"risk": "label"},
                audit=("turn", "label"),
            )
        ],
    )
    # 9 reasoning turns, batch=8 -> 2 windows -> 2 calls per member
    assert calls == {"a": 2, "b": 2}

    frame = results.layer_frames["batch_risk"]
    assert len(frame) == 9
    by_turn = frame.set_index("turn")
    # turn 3 split 1-1: the tie-break goes to the higher confidence
    assert by_turn.loc[3, "label"] == "risky"
    assert by_turn.loc[3, "judge_agreement"] == 0.5
    # turn 7: member a skipped, so b decides alone (no agreement fact)
    assert by_turn.loc[7, "label"] == "safe"
    assert pd.isna(by_turn.loc[7, "judge_agreement"])
    turn7_members = frame[frame.turn == 7].members.iloc[0]
    a_record = next(m for m in turn7_members if m["model"] == "mockllm/model")
    assert a_record["status"] == "no_answer"
    assert (frame.status == "ok").all()
    assert set(frame.label_source) == {"majority_vote"}

    regime = detect_regime(frame)
    assert regime.kind == "cohort" and regime.n_models == 2

    # every (turn, member) slot gets a ballot row; the skipped slot
    # carries status no_answer with no label, like the built-in votes
    ballots = transect.member_ballots(frame, "turn")
    assert len(ballots) == 18
    assert int(ballots.label.notna().sum()) == 17
    assert (ballots[ballots.label.isna()].status == "no_answer").all()
    stats = {
        st.label: st
        for st in label_stats(
            frame[frame.label.notna()], ballots, frame, "turn", "label", vocabulary=RISK
        )
    }
    assert stats["risky"].n == 8 and stats["safe"].n == 1

    # the report renders the layer's band and audit block from the
    # exploded frame
    html = open(results.report_paths[0]).read()
    assert "<h4>batch_risk" in html


def test_batched_solo_verifier_reviews_only_the_doubtful_turn(demo_log, tmp_path):
    """Solo batch: only the doubtful turn gets a verifier review,
    seen as a single-unit item."""
    reviews = {"n": 0}

    def judge(input, tools, tool_choice, config):
        text = "\n".join(getattr(m, "text", "") or "" for m in input)
        if "second-round reviewer" in text:
            reviews["n"] += 1
            assert "[ITEM" not in text  # the verifier sees one unit only
            return ModelOutput.for_tool_call(
                "mockllm/model",
                "answer",
                {"label": "safe", "confidence": 0.9, "explanation": "review"},
            )
        return batch_output(
            "mockllm/model",
            *(
                {
                    "item": i,
                    "label": "risky",
                    "confidence": 0.3 if turn == 6 else 0.9,
                    "explanation": "j",
                }
                for i, turn in marked_units(input).items()
            ),
        )

    @scout_scanner(loader=transect.reasoning_turns(batch=8))
    def batch_risk_solo(judge_models=None):
        return transect.cohort_llm_scanner(
            question="Is this turn risky?",
            answer=RISK,
            models=judge_models,
            batch=True,
            verify_sample=0.0,
        )

    results = transect.transect(
        logs=str(demo_log.parent),
        spec=transect.Spec(),
        judge_models=get_model("mockllm/model", custom_outputs=judge, memoize=False),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="risk_solo", scanner=batch_risk_solo, frame=transect.turns_frame
            )
        ],
    )
    frame = results.layer_frames["risk_solo"]
    assert reviews["n"] == 1
    by_turn = frame.set_index("turn")
    assert by_turn.loc[6, "label"] == "safe"
    assert by_turn.loc[6, "label_source"] == "verifier"
    assert bool(by_turn.loc[6, "overturned"])
    others = frame[frame.turn != 6]
    assert not others.verifier_reviewed.any()
    assert set(others.label) == {"risky"}
    assert set(others.label_source) == {"single_judge"}


def test_batching_is_unit_neutral(demo_log, tmp_path):
    """Non-turn units: loader facts become frame columns and the
    audit contract holds - nothing assumes turns."""

    @loader(messages="all")
    def halves() -> Loader[Transcript]:
        async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
            yield Transcript(
                transcript_id=f"{transcript.transcript_id}:halves",
                messages=[
                    ChatMessageUser(
                        content=batch_item_content(
                            ["first half of the run", "second half of the run"]
                        )
                    )
                ],
                metadata={"items": [{"half": 0}, {"half": 1}]},
            )

        return load

    def judge(input, tools, tool_choice, config):
        return batch_output(
            "mockllm/model",
            {"item": 1, "label": "safe", "confidence": 0.9, "explanation": "s"},
            {"item": 2, "label": "risky", "confidence": 0.8, "explanation": "r"},
        )

    @scout_scanner(loader=halves())
    def half_risk(judge_models=None):
        return transect.cohort_llm_scanner(
            question="Is this half risky?",
            answer=RISK,
            models=judge_models,
            batch=True,
            verify=False,
        )

    results = transect.transect(
        logs=str(demo_log.parent),
        spec=transect.Spec(),
        judge_models=get_model("mockllm/model", custom_outputs=judge, memoize=False),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        extra_layers=[
            transect.Layer(
                name="half_risk",
                scanner=half_risk,
                frame=transect.turns_frame,
                audit=("half", "label"),
            )
        ],
    )
    frame = results.layer_frames["half_risk"]
    assert list(frame.half) == [0, 1]
    assert list(frame.label) == ["safe", "risky"]
    assert "turn" not in frame.columns
    assert detect_regime(frame).kind == "solo"
    html = open(results.report_paths[0]).read()
    assert "<h4>half_risk" in html
