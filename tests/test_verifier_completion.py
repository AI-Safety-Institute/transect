"""Failed verifier attempts remain visible without diluting completed-review rates."""

import pandas as pd
import pytest
from inspect_scout import Result
from test_frames import raw_row

from transect.frames import subagents_df
from transect.frames.common import verifier_review
from transect.reliability import relabel_rate, spot_check_overturns
from transect.scanners.cohort import judge_setup


@pytest.mark.parametrize("status", ["error", "refusal", "no_answer", "ok", None])
def test_missing_verdict_preserves_review_presence_without_completion(status):
    """Failed, answerless and absent reviews cannot become completed verdicts."""
    review = {"status": status, "verifier_label": None} if status else {}
    if status not in ("ok", None):
        review.update(original_label="a", trigger="random_sample", overturned=False)
    row = verifier_review(review)
    assert row["verifier_selected"] is (status is not None)
    assert row["verifier_completed"] is False
    if status not in ("ok", None):
        assert row["overturned"] is False


def test_persisted_subagent_failures_do_not_dilute_overturn_rate(tmp_path):
    rows = []
    for i in range(10):
        ok = i == 0
        review = {
            "ran": True,
            "trigger": "random_sample",
            "original_label": "a",
            "verifier_label": "b" if ok else None,
            "status": "ok" if ok else "error",
            "overturned": ok,
        }
        result = Result(
            label="b" if ok else "a",
            value={
                "confidence": 0.9,
                "label_source": "verifier" if ok else "single_judge",
                "label_vocab": [{"label": "a"}, {"label": "b"}],
                "verifier": review,
                "judge": judge_setup(
                    ["mockllm/judge"],
                    1,
                    verifier_armed=True,
                    verifier_model="mockllm/judge",
                ),
            },
        )
        rows.append(raw_row(result, metadata={"agent_span_id": f"s{i}"}))
    path = tmp_path / "results.parquet"
    pd.DataFrame(rows).to_parquet(path)
    frame = subagents_df(pd.read_parquet(path))
    assert frame.verifier_selected.sum() == 10
    assert frame.verifier_completed.sum() == 1
    assert relabel_rate(frame).overall.rate == 1.0
    assert spot_check_overturns(frame).of == 1
