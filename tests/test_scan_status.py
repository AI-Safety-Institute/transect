"""Execution and usable coverage are separate persisted facts."""

import json

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


def cohort_value(**extra):
    """Use the helper's documented identity and vocabulary contract."""
    return {"judge": {"regime": "cohort"}, "label_vocab": [], **extra}


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
    assert coverage.expected_transcripts == coverage.missing_transcripts == 1


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


def test_nested_cohort_failure_is_not_hidden_by_parent_success():
    """A batch's usable units are counted individually even when its parent is ok."""
    value = cohort_value(
        status="ok",
        items=[
            {"label": "a", "status": "ok"},
            {"label": None, "status": "error"},
            {"label": None, "status": "refusal"},
            {"label": None, "status": "no_answer"},
        ],
    )
    raw = raw_scan("custom/scanner", [{"value": json.dumps(value)}])
    status = build_scan_status(raw, {"test"})
    coverage = status.scanners[0]
    assert status.outer_complete and status.has_failures
    assert coverage.contract == "cohort"
    assert (coverage.observed_units, coverage.usable_units) == (4, 1)
    assert (coverage.failed_units, coverage.refused_units, coverage.missing_units) == (
        1,
        1,
        1,
    )


def test_usable_label_keeps_member_and_verifier_degradation_separate():
    """A surviving label remains usable when a member or verifier fails."""
    value = cohort_value(
        status="ok",
        label_source="majority_vote",
        cohort={"members": [{"status": "ok"}, {"status": "error"}]},
        verifier={"ran": True, "status": "refusal"},
    )
    raw = raw_scan("custom/scanner", [{"value": value, "label": "a"}])
    coverage = build_scan_status(raw, {"test"}).scanners[0]
    assert coverage.usable_units == 1 and coverage.failed_units == 0
    assert coverage.degraded_judgements == 2


def test_phases_distinguish_judged_filled_and_attributed_turns():
    """Projected or filled labels never count as direct judge coverage."""
    bases = ["judged", "filled", "attributed", "refusal", "no_answer", "missing_turn"]
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "value": {
                    "turns": [{"basis": basis, "phase_index": 0} for basis in bases],
                    "phases": [{"phase": "a"}],
                    "verifier": {"ran": True, "n_no_verdict": 2},
                    "narrator": {"ran": True, "n_fallback": 1},
                }
            }
        ],
    )
    coverage = build_scan_status(raw, {"test"}).scanners[0]
    assert (coverage.observed_units, coverage.usable_units) == (5, 1)
    assert coverage.filled_units == coverage.attributed_units == 1
    assert (coverage.failed_units, coverage.refused_units, coverage.missing_units) == (
        1,
        1,
        1,
    )
    assert coverage.degraded_judgements == 2
    assert any("narrator" in note for note in coverage.notes)


@pytest.mark.parametrize(
    "value", [False, 0, [], {}, {"status": "error"}, {"items": [{"status": "error"}]}]
)
def test_arbitrary_custom_values_have_unknown_semantic_coverage(value):
    """Application-specific status keys and falsy results are not failures."""
    raw = raw_scan("custom/scanner", [{"value": value}])
    status = build_scan_status(raw, set())
    coverage = status.scanners[0]
    assert not status.has_failures
    assert coverage.contract == "unknown" and coverage.usable_units is None
    assert coverage.failed_units is None and not coverage.mounted


def test_completed_zero_item_loader_is_not_failed():
    """Completed execution can legitimately yield no classifier inputs."""
    raw = raw_scan("transect/subagent_classification", [])
    status = build_scan_status(raw, {"test"})
    coverage = status.scanners[0]
    assert not status.has_failures
    assert coverage.observed_units == coverage.usable_units == 0
    assert any("zero" in note for note in coverage.notes)


def test_missing_snapshot_and_worklist_preserve_expected_scope():
    """Unknown selection stays nullable while a worklist supplies exact scope."""
    raw = raw_scan("custom/scanner", [], scanned=0, snapshot=False)
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.expected_transcripts is coverage.missing_transcripts is None
    raw.spec.worklist = [Worklist(scanner="test", transcripts=["t1", "t2"])]
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.expected_transcripts == coverage.missing_transcripts == 2


def test_known_contract_without_payload_is_not_successful_zero_coverage():
    """Missing required known payload fields remain visible."""
    raw = raw_scan("transect/decision_phases", [{"value": {}}])
    coverage = build_scan_status(raw, {"test"}).scanners[0]
    assert coverage.has_failures and coverage.usable_units is None


def test_judged_phase_without_resolvable_label_is_missing():
    """A basis flag cannot establish usable coverage without its phase label."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "value": {
                    "turns": [{"basis": "judged", "phase_index": None}],
                    "phases": [],
                }
            }
        ],
    )
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.usable_units == 0 and coverage.missing_units == 1
    assert coverage.has_failures


def test_zero_reasoning_turns_is_valid_phase_coverage():
    """A completed phase scanner can legitimately have no eligible turns."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "value": {
                    "turns": [{"basis": "attributed", "phase_index": None}],
                    "phases": [],
                }
            }
        ],
    )
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.observed_units == coverage.usable_units == 0
    assert coverage.attributed_units == 1 and not coverage.has_failures


def test_worklist_narrows_transcript_snapshot():
    """A scanner's worklist takes precedence over the wider scan snapshot."""
    raw = raw_scan("custom/scanner", [{"value": False}])
    assert raw.spec.transcripts is not None
    raw.spec.transcripts.transcript_ids["t2"] = None
    raw.spec.worklist = [Worklist(scanner="test", transcripts=["t1"])]
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.expected_transcripts == 1 and coverage.missing_transcripts == 0


def test_phase_member_failure_is_separate_from_consensus_coverage():
    """A phase member's refusal remains visible beside a usable consensus label."""
    raw = raw_scan(
        "transect/decision_phases",
        [
            {
                "value": {
                    "turns": [{"basis": "judged", "phase_index": 0}],
                    "phases": [{"phase": "a"}],
                    "cohort": {"members": [{"turns": [{"basis": "refusal"}]}]},
                }
            }
        ],
    )
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.usable_units == 1 and coverage.degraded_judgements == 1


@pytest.mark.parametrize(
    "outcome,label",
    [("error", None), ("error", "a"), ("refusal", "a"), ("no_answer", "a")],
)
def test_unmounted_helper_failure_remains_visible_even_with_a_label(outcome, label):
    """Neither a stray label nor an absent custom frame conceals failed outcomes."""
    value = cohort_value(status=outcome, label_source="single_judge" if label else None)
    raw = raw_scan("custom/scanner", [{"value": value, "label": label}])
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.observed_units == 1 and not coverage.mounted
    assert coverage.usable_units == 0 and coverage.has_failures
    if outcome == "error":
        assert coverage.failed_units == 1


def test_missing_table_is_not_confused_with_valid_empty_loader():
    """A missing stored table cannot support a zero-item loader explanation."""
    raw = raw_scan("transect/subagent_classification", [])
    raw.scanners = {}
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.has_failures and coverage.usable_units is None


@pytest.mark.parametrize(
    "name",
    [
        "transect/decision_phases",
        "transect/token_timeline",
        "transect/context_flush",
        "transect/human_intervention",
        "transect/eval_setup",
    ],
)
@pytest.mark.parametrize("recorded_scans, stored_rows", [(1, 0), (2, 1)])
def test_singleton_scanners_expose_missing_stored_results(
    name, recorded_scans, stored_rows
):
    """Known singleton scanners must retain one result per executed transcript."""
    rows = [{"transcript_id": "t1", "value": {"turns": [], "phases": []}}] * stored_rows
    raw = raw_scan(name, rows, scanned=recorded_scans)
    coverage = build_scan_status(raw, set()).scanners[0]
    assert coverage.has_failures and coverage.coverage_errors == 1
    assert any("one result" in note for note in coverage.notes)
    assert not any("valid empty loader" in note for note in coverage.notes)


@pytest.mark.parametrize("helper_first", [False, True])
def test_mixed_custom_contract_coverage_is_unknown_in_either_order(helper_first):
    """Mixed application and helper rows have explicit unassessed coverage."""
    rows = [
        {"value": {"status": "error"}, "label": None},
        {"value": cohort_value(status="ok", label_source="single_judge"), "label": "a"},
    ]
    if helper_first:
        rows.reverse()
    coverage = build_scan_status(raw_scan("custom/scanner", rows), set()).scanners[0]
    assert coverage.contract == "unknown"
    assert coverage.usable_units is coverage.observed_units is None
    assert coverage.coverage_errors == 0 and not coverage.has_failures
    assert any("1 helper" in note and "1 unassessed" in note for note in coverage.notes)


def test_empty_custom_loader_retains_unknown_semantics_without_failure():
    """Zero custom result rows cannot establish either failed or usable items."""
    coverage = build_scan_status(raw_scan("custom/scanner", []), set()).scanners[0]
    assert coverage.contract == "unknown" and coverage.usable_units is None
    assert not coverage.has_failures
    assert any("valid empty loader" in note for note in coverage.notes)
