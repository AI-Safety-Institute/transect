"""Run-wide execution facts are projected from the persisted scan store."""

import pandas as pd
import pytest
from inspect_scout import (
    ScannerSpec,
    ScanResultsDF,
    ScanSpec,
    ScanTranscripts,
    Summary,
    Worklist,
)

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
    assert status.scanners[0].execution_errors == 1
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


def test_missing_table_despite_recorded_scans_is_an_integrity_error():
    """The execution summary and the stored tables must agree."""
    raw = raw_scan("transect/subagent_classification", [])
    raw.scanners = {}
    status = build_scan_status(raw, set())
    coverage = status.scanners[0]
    assert coverage.has_failures and coverage.integrity_errors == 1
    assert coverage.execution_errors == 0
    assert any("table is missing" in error.message for error in status.errors)
