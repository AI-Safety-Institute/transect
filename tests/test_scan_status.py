"""Run-wide execution facts are projected from the persisted scan store."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from inspect_scout import (
    ScannerSpec,
    ScanResultsDF,
    ScanSpec,
    ScanTranscripts,
    Summary,
    Worklist,
    scan_results_df,
)

from transect.report.sections import scan_status_view
from transect.scan_status import ScanStatus, build_scan_status


def raw_scan(name, rows, *, complete=True, scanned=1, errors=0, snapshot=True):
    """Build Scout's public read model without a provider or scan invocation."""
    return ScanResultsDF(
        complete=complete,
        spec=ScanSpec(
            scan_name="coverage",
            scanners={"test": ScannerSpec(name=name)},
            transcripts=(
                ScanTranscripts(type="eval_log", transcript_ids={"t1": None})
                if snapshot
                else None
            ),
        ),
        location="unused",
        summary=Summary(scanners={"test": {"scans": scanned, "errors": errors}}),
        errors=[],
        scanners={"test": pd.DataFrame(rows)},
    )


def test_unknown_default_does_not_assert_success():
    """Manually constructed results have unknown execution status."""
    status = ScanStatus()
    assert status.outer_complete is None
    assert status.scanners == [] and not status.has_failures


def test_requested_absent_scanner_and_interruption_are_visible():
    """The stored requested roster survives absence of a results table."""
    raw = raw_scan("transect/decision_phases", [], complete=False, scanned=0)
    raw.scanners = {}
    status = build_scan_status(raw, set())
    coverage = status.scanners[0]
    assert status.has_failures and status.outer_complete is False
    assert coverage.scanner == "test" and coverage.mounted
    assert coverage.total_transcripts == coverage.missing_scans == 1


@pytest.mark.parametrize("error_type", [None, pd.NA, "refusal"])
def test_nullable_scout_errors_preserve_identity(error_type):
    """Recorded errors are visible without treating nullable categories as bools."""
    raw = raw_scan(
        "custom/scanner",
        [
            {
                "transcript_id": "t1",
                "value": None,
                "scan_error": "provider unavailable",
                "scan_error_type": error_type,
            }
        ],
        errors=1,
    )
    status = build_scan_status(raw, {"test"})
    assert status.has_failures
    assert status.scanners[0].errors == 1
    assert status.errors[0].transcript_id == "t1"
    assert status.errors[0].message == "provider unavailable"
    assert status.errors[0].refusal == (isinstance(error_type, str))


@pytest.mark.parametrize(
    "value", [False, 0, [], {}, {"status": "error"}, {"items": [{"status": "error"}]}]
)
def test_custom_result_payloads_are_not_execution_failures(value):
    """Application-specific payloads are opaque here; execution alone decides."""
    raw = raw_scan("custom/scanner", [{"value": value}])
    status = build_scan_status(raw, set())
    assert not status.has_failures
    assert not status.scanners[0].mounted


def test_missing_snapshot_and_worklist_preserve_scan_scope():
    """Unknown selection stays nullable while a worklist supplies exact scope."""
    raw = raw_scan("custom/scanner", [], scanned=0, snapshot=False)
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.total_transcripts is coverage.missing_scans is None
    raw.spec.worklist = [Worklist(scanner="test", transcripts=["t1", "t2"])]
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.total_transcripts == coverage.missing_scans == 2


def test_worklist_narrows_transcript_snapshot():
    """A scanner's worklist takes precedence over the wider scan snapshot."""
    raw = raw_scan("custom/scanner", [{"value": False}])
    assert raw.spec.transcripts is not None
    raw.spec.transcripts.transcript_ids["t2"] = None
    raw.spec.worklist = [Worklist(scanner="test", transcripts=["t1"])]
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.total_transcripts == 1 and coverage.missing_scans == 0


def test_missing_table_despite_recorded_scans_is_an_error():
    """The execution summary and the stored tables must agree."""
    raw = raw_scan("transect/subagent_classification", [])
    raw.scanners = {}
    status = build_scan_status(raw, set())
    coverage = status.scanners[0]
    assert coverage.has_failures and coverage.errors == 1
    assert coverage.completed_transcripts == 0
    assert any("table is missing" in error.message for error in status.errors)
    assert scan_status_view(status)["execution"] == "finished; results incomplete"


def test_completed_subtracts_errored_transcripts():
    """A transcript with a recorded scan error is attempted, not completed."""
    raw = raw_scan(
        "custom/scanner",
        [{"transcript_id": "t1", "scan_error": "boom"}],
        scanned=2,
        errors=1,
    )
    raw.spec.worklist = [Worklist(scanner="test", transcripts=["t1", "t2"])]
    coverage = build_scan_status(raw, {"test"}).scanners[0]
    assert (coverage.scanned_transcripts, coverage.completed_transcripts) == (2, 1)
    assert coverage.errors == 1 and coverage.has_failures


def test_never_attempted_scanner_with_unknown_scope_is_a_failure():
    """0 of 0 is a red cell with its own explanation, not a healthy row."""
    raw = raw_scan("custom/scanner", [], scanned=0, snapshot=False)
    status = build_scan_status(raw, {"test"})
    assert status.has_failures
    view = scan_status_view(status)
    (row,) = view["rows"]
    assert row["completed"]["text"] == "0 of 0"
    assert row["completed"]["bg"] is not None
    assert view["unattempted"] == ["test"]


def test_resumed_scan_counts_stay_within_scope():
    """A retry-resume can inflate summary counters; the record caps the
    scan count at the known scope and counts recorded errors only."""
    raw = raw_scan(
        "transect/decision_phases", [{"transcript_id": "t1"}], scanned=2, errors=1
    )
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.scanned_transcripts == coverage.total_transcripts == 1
    assert coverage.errors == 0 and not coverage.has_failures


def _usage_json(**models):
    import json

    return json.dumps(models)


def test_model_usage_sums_rows_per_model_and_keeps_unreported_fields_absent():
    """Billed usage aggregates the stored per-row model usage by model name;
    a field no row reported stays None rather than becoming a zero."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "transcript_id": "t1",
                "scan_model_usage": _usage_json(
                    **{
                        "anthropic/judge": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                            "input_tokens_cache_read": 40,
                            "total_cost": 0.5,
                        },
                        "openai/verifier": {
                            "input_tokens": 20,
                            "output_tokens": 2,
                            "total_tokens": 22,
                        },
                    }
                ),
            },
            {
                "transcript_id": "t2",
                "scan_model_usage": _usage_json(
                    **{
                        "anthropic/judge": {
                            "input_tokens": 50,
                            "output_tokens": 5,
                            "total_tokens": 55,
                            "input_tokens_cache_read": 10,
                            "total_cost": 0.25,
                        }
                    }
                ),
            },
            {"transcript_id": "t3", "scan_model_usage": None},
        ],
        scanned=3,
    )
    coverage = build_scan_status(raw, set()).scanners[0]
    judge, verifier = coverage.model_usage
    assert (judge.model, judge.input_tokens, judge.output_tokens) == (
        "anthropic/judge",
        150,
        15,
    )
    assert judge.total_tokens == 165 and judge.input_tokens_cache_read == 50
    assert judge.input_tokens_cache_write is None and judge.reasoning_tokens is None
    assert judge.total_cost == 0.75
    assert (verifier.model, verifier.total_tokens) == ("openai/verifier", 22)
    assert verifier.total_cost is None and verifier.input_tokens_cache_read is None


def test_structural_scanner_records_no_model_usage():
    """A scanner that made no model calls has an empty usage list, not zeros."""
    raw = raw_scan("transect/token_timeline", [{"scan_model_usage": "{}"}])
    assert build_scan_status(raw, set()).scanners[0].model_usage == []


def test_run_wide_model_usage_totals_across_scanners():
    """The run-wide total sums each model across every scanner."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "scan_model_usage": _usage_json(
                    **{"m": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
                )
            }
        ],
    )
    status = build_scan_status(raw, set())
    status.scanners.append(replace(status.scanners[0], scanner="other"))
    (total,) = status.model_usage
    assert (total.model, total.total_tokens, total.total_cost) == ("m", 4, None)


def test_stored_cohort_scan_usage_matches_scout_summary():
    """Row-aggregated usage agrees with Scout's own per-scanner summary."""
    from transect import load

    store = Path(__file__).parent / "fixtures" / "demo_scan_cohort"
    results = load(str(store))
    status = results.scan_status
    raw = scan_results_df(results.scan_location)
    phases = next(s for s in status.scanners if s.scanner == "decision_phases")
    summary = raw.summary.scanners["decision_phases"].model_usage
    assert {u.model: u.total_tokens for u in phases.model_usage} == {
        model: usage.total_tokens for model, usage in summary.items()
    }
    assert len(phases.model_usage) == 3
    assert all(u.total_cost is None for u in phases.model_usage)


def test_scan_status_view_lists_billed_usage_by_scanner_and_model():
    """The report block lists usage per scanner and model, adds a run total
    only once a second scanner billed, and leaves an unreported field None
    for the template's faded dash."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "scan_model_usage": _usage_json(
                    **{
                        "a/judge": {
                            "input_tokens": 1200,
                            "output_tokens": 34,
                            "total_tokens": 1234,
                            "total_cost": 0.0125,
                        },
                        "b/judge": {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "total_tokens": 11,
                        },
                    }
                )
            }
        ],
    )
    status = build_scan_status(raw, set())
    rows = scan_status_view(status)["usage"]
    assert [(r["scanner"], r["model"]) for r in rows] == [
        ("test", "a/judge"),
        ("test", "b/judge"),
    ]
    assert rows[0]["total"] == "1,234" and rows[0]["cost"] == "$0.0125"
    assert rows[1]["cost"] is None and rows[1]["cache_read"] is None
    status.scanners.append(replace(status.scanners[0], scanner="other"))
    rows = scan_status_view(status)["usage"]
    assert [(r["scanner"], r["model"]) for r in rows[-2:]] == [
        ("all scanners", "a/judge"),
        ("all scanners", "b/judge"),
    ]
    assert rows[-2]["total"] == "2,468" and rows[-2]["cost"] == "$0.0250"


def test_scan_status_view_without_billed_usage_says_so():
    """A structural-only or fully cached scan renders an explicit note."""
    raw = raw_scan("transect/token_timeline", [{"scan_model_usage": "{}"}])
    view = scan_status_view(build_scan_status(raw, set()))
    assert view["usage"] == []
