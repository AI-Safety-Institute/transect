"""One-shot generator for the committed judged scan stores.

Two $0 stores over the committed demo eval
(examples/logs/house_price_demo.eval), covering the two regime
families and their report surfaces:

- tests/fixtures/demo_scan: solo k-roll (k=3) + same-model verifier,
  with planted signals - a dissenting roll (amber self-consistency)
  and a random-sample overturn (red spot-check, amber re-label rate) -
  so verifier selection/outcomes, tooltip verifier lines, and the
  k-roll flag family all render.
- tests/fixtures/demo_scan_cohort: 3-model cohort with one systematic
  dissenter on both surfaces (phases and one sub-agent span) - cohort
  agreement (alpha + AC1) in the audit, inline flags in both sections
  (red for Phases, amber for Sub-agents), split member ballots in the
  tooltips. Verifier off (cohort regime).

Regenerate after a scanner value-schema or store-format change; paths
inside the stores are relative to the repo root, so run from there.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import re

import pyarrow as pa
import pyarrow.parquet as pq
from helpers import (
    demo_judge,
    narrate_answer,
    narrative,
    scripted_groups,
    seg,
    seg_answer,
    verify_answer,
    vote_output,
)
from inspect_ai.model import get_model
from pytest import MonkeyPatch

from transect import load, transect

# neuter inspect's on-disk model cache (the same patch the conftest
# applies for tests): scripted answers must come from the script, never
# replay a previous run's cached output for an identical prompt
# signature-agnostic: inspect calls these with keyword arguments
# (cache_store(entry=..., output=...)), so named params must not drift
_patch = MonkeyPatch()
_patch.setattr("inspect_ai.model._model.cache_fetch", lambda *a, **kw: None)
_patch.setattr("inspect_ai.model._model.cache_store", lambda *a, **kw: None)

ROOT = Path(__file__).parents[2]

_SUBAGENT_LABELS = {
    "eda": "data_analysis",
    "alt_model": "model_experimentation",
    "reviewer": "result_review",
}


def kroll_flagged_judge():
    """demo_judge variant that plants reliability signals for the
    k-roll store: the third segmentation roll dissents (per-turn
    self-consistency ~0.67 - amber) and the verifier overturns phase 0
    of its random sample (spot-check overturn - red; re-label rate 50%
    overall, 100% for model_development - amber). Sub-agent answers
    stay unanimous. The dissent is planted by call count, not roll
    identity - a judge-call retry would shift which roll dissents;
    `check_planted_signals` catches that on regeneration."""
    seg_calls = {"n": 0}

    def route(input, tools, tool_choice, config):
        text = "\n".join(getattr(m, "text", "") or "" for m in input)
        if "Review these phases:" in text:
            # overturn only phase 0; confirm the rest, so the other
            # phase keeps its vote-agreement rows (the self-consistency
            # indicator needs surviving judged turns)
            return verify_answer(
                *(
                    {
                        "phase_index": int(k),
                        "phase": "ensembling" if int(k) == 0 else current,
                        "confidence": 0.9,
                        "explanation": "planted overturn"
                        if int(k) == 0
                        else "label holds",
                    }
                    for k, current in re.findall(r"PHASE (\d+) \[current=(\w+)", text)
                )
            )
        if "Narrate these phases:" in text:
            return narrate_answer(
                *(
                    narrative(
                        int(k),
                        headline=f"Scripted headline for {label}",
                        groups=scripted_groups(text, int(k)),
                    )
                    for k, label in re.findall(r"PHASE (\d+) \[(\w+),", text)
                )
            )
        segment_range = re.search(r"Segment turns (\d+)\.\.(\d+)", text)
        if segment_range:
            first, last = map(int, segment_range.groups())
            seg_calls["n"] += 1
            mid = (first + last) // 2
            if seg_calls["n"] >= 3:  # the dissenting roll
                return seg_answer(seg(first, last, "baseline_modeling", 0.9))
            return seg_answer(
                seg(first, mid, "model_development", 0.9),
                seg(mid + 1, last, "validation_and_submission", 0.9),
            )
        lane = re.search(r"name: (\w+)", text)
        if lane and "spawn task" in text:
            return vote_output(_SUBAGENT_LABELS[lane.group(1)], 0.9, "scripted")
        raise AssertionError(f"unroutable judge request: {text[:200]}")

    return get_model("mockllm/model", custom_outputs=route, memoize=False)


def run(store: Path, **judge_setup) -> None:
    if store.exists():
        shutil.rmtree(store)
    transect(
        logs="examples/logs/house_price_demo.eval",
        spec="examples/spec.yaml",
        scans_dir=str(store.relative_to(ROOT)),
        viewer=False,
        open_report=False,
        report_path=None,
        **judge_setup,
    )
    portable_sources(store)
    for report in store.rglob("report*.html"):
        report.unlink()
    print(f"wrote {store}")


def portable_sources(store: Path) -> None:
    """Keep synthetic fixture source references relative to the checkout root.

    Scout resolves input filenames to absolute paths during scanning. Normalize
    only this checkout's prefix, including copies of the same source reference
    embedded in metadata/input JSON. Raw synthetic transcript content is unchanged.
    """
    prefix = str(ROOT.resolve()) + "/"

    def relative(value):
        if isinstance(value, str):
            return value.replace(prefix, "")
        if isinstance(value, dict):
            return {key: relative(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relative(item) for item in value]
        return value

    for path in store.rglob("*.json"):
        value = relative(json.loads(path.read_text()))
        if path.name == "_scan.json":
            # Scout names a scan after the working directory; pin the
            # fixture's name so a checkout's directory does not leak in
            value["scan_name"] = "transect"
        path.write_text(
            json.dumps(value, indent=None if path.name == "_summary.json" else 2)
        )
    for path in store.rglob("*.parquet"):
        table = pq.ParquetFile(path).read()
        for index, column in enumerate(table.columns):
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                table = table.set_column(
                    index,
                    table.schema.field(index),
                    pa.array(
                        [relative(item) for item in column.to_pylist()],
                        type=column.type,
                    ),
                )
        pq.write_table(table, path)


def check_planted_signals(store: Path, regime: str) -> None:
    """Fail generation loudly if a store lacks the signals it exists
    to light up - a mockllm routing failure is swallowed into scout's
    per-transcript error capture and would otherwise commit a
    silently signal-less store."""
    for errors_file in store.rglob("_errors.jsonl"):
        lines = errors_file.read_text().strip()
        assert not lines, f"{errors_file} records scan errors:\n{lines}"
    frames = load(str(store.relative_to(ROOT))).frames()
    phases, turns = frames["phases"], frames["phase_turns"]
    assert (phases.narration_group_status == "complete").all(), (
        "scripted narrator groups must partition every phase"
    )
    if regime == "kroll":
        reviewed = phases[phases.verifier_selected.fillna(False)]
        assert len(reviewed) == 2, f"expected 2 reviewed phases: {len(reviewed)}"
        assert (reviewed.verifier_trigger == "random_sample").all()
        assert int(phases.overturned.fillna(False).sum()) == 1
        assert "ensembling" in set(phases.phase)
        agreement = turns.judge_agreement.dropna()
        assert len(agreement) and 0.6 < agreement.mean() < 0.7, (
            "planted k-roll dissent missing (agreement "
            f"{agreement.mean() if len(agreement) else 'n/a'})"
        )
        span_votes = frames["subagent_votes"]
        assert (span_votes.groupby("agent_span_id").label.nunique() == 1).all()
    else:
        split = turns.judge_agreement.dropna()
        assert len(split) and (split < 1.0).all(), "phase dissenter missing"
        span_labels = frames["subagent_votes"].groupby("agent_span_id").label.nunique()
        assert (span_labels > 1).any(), "sub-agent dissenter missing"
    print(f"checked {store}")


def main() -> None:
    kroll_store = ROOT / "tests" / "fixtures" / "demo_scan"
    run(
        kroll_store,
        judge_models=kroll_flagged_judge(),
        k_rolls=3,
        verify=True,
        verify_sample=1.0,
    )
    check_planted_signals(kroll_store, "kroll")
    dissenting_labels = {
        "eda": "model_experimentation",  # dissents from data_analysis
        "alt_model": "model_experimentation",
        "reviewer": "result_review",
    }
    cohort_store = ROOT / "tests" / "fixtures" / "demo_scan_cohort"
    run(
        cohort_store,
        judge_models=[
            demo_judge(),
            demo_judge(
                model="mockllm/model2",
                phase_label="baseline_modeling",
                subagent_labels=dissenting_labels,
            ),
            demo_judge(model="mockllm/model3"),
        ],
    )
    check_planted_signals(cohort_store, "cohort")


if __name__ == "__main__":
    main()
