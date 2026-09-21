"""Stored execution facts and bounded checks of usable judged coverage.

Scout completion describes execution, not usable labels. Coverage counts
describe recorded units under Transect's phase and cohort contracts. They
cannot establish that an arbitrary custom loader emitted every intended item.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from inspect_scout import ScanResultsDF


@dataclass
class ScanError:
    """A stored execution error, without the potentially large traceback."""

    scanner: str
    transcript_id: str | None
    message: str
    refusal: bool = False


@dataclass
class ScannerCoverage:
    """One requested scanner's execution and recorded-unit coverage.

    Nullable unit counts mean the content contract could not be assessed.
    For phases, units are reasoning-bearing turns: attributed turns are
    excluded, and filled turns are counted separately from directly judged
    usable turns. For cohorts, units are loader items or declared batch items.
    ``degraded_judgements`` counts failed member/unit and verifier outcomes,
    not distinct API calls. ``coverage_errors`` counts malformed payload rows
    and discrepancies in the number of stored singleton results.
    """

    scanner: str
    name: str
    mounted: bool
    expected_transcripts: int | None = None
    scanned_transcripts: int = 0
    missing_transcripts: int | None = None
    result_rows: int = 0
    execution_errors: int = 0
    contract: str = "unknown"
    unit: str | None = None
    observed_units: int | None = None
    usable_units: int | None = None
    failed_units: int | None = None
    refused_units: int | None = None
    missing_units: int | None = None
    filled_units: int = 0
    attributed_units: int = 0
    degraded_judgements: int = 0
    coverage_errors: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """Whether stored evidence identifies missing or unusable output."""
        return any(
            count is not None and count > 0
            for count in (
                self.missing_transcripts,
                self.execution_errors,
                self.coverage_errors,
                self.failed_units,
                self.refused_units,
                self.missing_units,
            )
        )


@dataclass
class ScanStatus:
    """Run-wide status; the default explicitly leaves execution unknown."""

    outer_complete: bool | None = None
    scanners: list[ScannerCoverage] = field(default_factory=list)
    errors: list[ScanError] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """Whether execution or assessed usable coverage is incomplete."""
        return (
            self.outer_complete is False
            or bool(self.errors)
            or any(scanner.has_failures for scanner in self.scanners)
        )


_STRUCTURAL = {
    "transect/token_timeline",
    "transect/context_flush",
    "transect/human_intervention",
    "transect/eval_setup",
}


def build_scan_status(raw: ScanResultsDF, mounted_scanners: set[str]) -> ScanStatus:
    """Project persisted Scout metadata and supported value contracts.

    The caller can exclude Scout's heavy columns. No transcript reads,
    model calls or custom frame functions are needed here. Expected execution
    counts remain unknown when the stored transcript snapshot is absent.
    """
    status = ScanStatus(outer_complete=raw.complete)
    for error in raw.errors:
        status.errors.append(
            ScanError(error.scanner, error.transcript_id, error.error, error.refusal)
        )
    for key, spec in raw.spec.scanners.items():
        expected: int | None = None
        if raw.spec.worklist is not None:
            expected = len(
                {
                    tid
                    for work in raw.spec.worklist
                    if work.scanner == key
                    for tid in work.transcripts
                }
            )
        elif raw.spec.transcripts and raw.spec.transcripts.transcript_ids:
            expected = len(raw.spec.transcripts.transcript_ids)
        summary = raw.summary.scanners.get(key)
        coverage = ScannerCoverage(
            scanner=key,
            name=spec.name,
            mounted=key in mounted_scanners
            or spec.name in _STRUCTURAL
            or spec.name
            in ("transect/decision_phases", "transect/subagent_classification"),
            expected_transcripts=expected,
            scanned_transcripts=summary.scans if summary else 0,
            execution_errors=summary.errors if summary else 0,
        )
        if expected is not None:
            coverage.missing_transcripts = max(
                0, expected - coverage.scanned_transcripts
            )
        if spec.name == "transect/decision_phases":
            coverage.contract, coverage.unit = "phases", "reasoning turns"
        elif spec.name == "transect/subagent_classification":
            coverage.contract, coverage.unit = "cohort", "items"
        elif spec.name in _STRUCTURAL:
            coverage.contract = "structural"
        table = raw.scanners.get(key)
        if table is None and coverage.scanned_transcripts:
            coverage.coverage_errors += 1
            coverage.notes.append(
                "The execution summary records scans but their results "
                "table is missing."
            )
        if table is not None:
            coverage.result_rows = len(table)
            if (
                coverage.contract in ("phases", "structural")
                and summary is not None
                and len(table) != summary.scans
            ):
                coverage.coverage_errors += abs(len(table) - summary.scans)
                coverage.notes.append(
                    "This scanner returns one result per executed transcript; "
                    f"the summary records {summary.scans} executions but "
                    f"the table contains {len(table)} rows."
                )
            if coverage.contract == "unknown":
                helper_rows = unassessed_rows = 0
                for _, row in table.iterrows():
                    if _text(row.get("scan_error")) is not None:
                        continue
                    if _cohort_contract(_object(row.get("value"))):
                        helper_rows += 1
                    else:
                        unassessed_rows += 1
                if helper_rows and not unassessed_rows:
                    coverage.contract, coverage.unit = "cohort", "items"
                elif helper_rows:
                    coverage.notes.append(
                        f"Mixed custom contracts: {helper_rows} helper-shaped "
                        f"rows and {unassessed_rows} unassessed rows. "
                        "Semantic coverage for this scanner is unknown."
                    )
            for _, row in table.iterrows():
                error_message = _text(row.get("scan_error"))
                if error_message is not None:
                    record = ScanError(
                        key,
                        _text(row.get("transcript_id")),
                        error_message,
                        _text(row.get("scan_error_type")) == "refusal",
                    )
                    if record not in status.errors:
                        status.errors.append(record)
                    continue
                value = _object(row.get("value"))
                if coverage.contract == "phases":
                    _phases(coverage, value)
                elif coverage.contract == "cohort":
                    _cohort(coverage, value, row.get("label"))
        recorded_errors = sum(error.scanner == key for error in status.errors)
        coverage.execution_errors = max(coverage.execution_errors, recorded_errors)
        if (
            coverage.result_rows == 0
            and table is not None
            and coverage.scanned_transcripts > 0
            and not coverage.execution_errors
            and coverage.contract not in ("phases", "structural")
        ):
            coverage.notes.append(
                "Recorded scanner execution returned zero result items; "
                "this can be a valid empty loader result."
            )
            if coverage.contract == "cohort":
                _start_counts(coverage)
        if coverage.contract == "unknown":
            coverage.notes.append(
                "Usable coverage is not assessed for this custom scanner contract."
            )
        if not coverage.mounted:
            coverage.notes.append("This scanner is not mounted into report frames.")
        status.scanners.append(coverage)
    return status


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _cohort_contract(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("judge"), dict)
        and isinstance(value.get("label_vocab"), list)
        and ("label_source" in value or isinstance(value.get("items"), list))
    )


def _start_counts(coverage: ScannerCoverage) -> None:
    if coverage.observed_units is None:
        coverage.observed_units = coverage.usable_units = 0
        coverage.failed_units = coverage.refused_units = coverage.missing_units = 0


def _count(coverage: ScannerCoverage, field_name: str, n: int = 1) -> None:
    setattr(coverage, field_name, (getattr(coverage, field_name) or 0) + n)


def _bad_payload(coverage: ScannerCoverage) -> None:
    coverage.coverage_errors += 1
    note = "A result is missing or violates the documented coverage payload."
    if note not in coverage.notes:
        coverage.notes.append(note)


def _phases(coverage: ScannerCoverage, value: dict[str, Any]) -> None:
    turns = value.get("turns")
    phases = value.get("phases")
    if not isinstance(turns, list) or not isinstance(phases, list):
        _bad_payload(coverage)
        return
    _start_counts(coverage)
    for turn in turns:
        basis = _text(turn.get("basis")) if isinstance(turn, dict) else None
        if basis == "attributed":
            coverage.attributed_units += 1
            continue
        _count(coverage, "observed_units")
        field_name = {
            "judged": "usable_units",
            "filled": "filled_units",
            "refusal": "refused_units",
            "no_answer": "failed_units",
            "missing_turn": "missing_units",
        }.get(basis or "")
        if basis in ("judged", "filled"):
            phase_index = turn.get("phase_index")
            if (
                not isinstance(phase_index, int)
                or not 0 <= phase_index < len(phases)
                or not isinstance(phases[phase_index], dict)
                or not _text(phases[phase_index].get("phase"))
            ):
                _count(coverage, "missing_units")
                continue
        if field_name:
            _count(coverage, field_name)
        else:
            _bad_payload(coverage)
    verifier = _object(value.get("verifier"))
    coverage.degraded_judgements += _positive_int(verifier.get("n_no_verdict"))
    members = _object(value.get("cohort")).get("members", [])
    if isinstance(members, list):
        for member in members:
            turns = member.get("turns", []) if isinstance(member, dict) else []
            if isinstance(turns, list):
                coverage.degraded_judgements += sum(
                    isinstance(turn, dict)
                    and turn.get("basis") in ("refusal", "no_answer", "missing_turn")
                    for turn in turns
                )
    narrator = _object(value.get("narrator"))
    fallbacks = _positive_int(narrator.get("n_fallback"))
    if fallbacks:
        coverage.notes.append(
            f"Phase narrator used fallback output for {fallbacks} phases."
        )


def _cohort(coverage: ScannerCoverage, value: dict[str, Any], label: Any) -> None:
    if not _cohort_contract(value):
        _bad_payload(coverage)
        return
    entries = value.get("items")
    if entries is None:
        entries = [{**value, "label": label}]
    if not isinstance(entries, list):
        _bad_payload(coverage)
        return
    _start_counts(coverage)
    for entry in entries:
        _count(coverage, "observed_units")
        if not isinstance(entry, dict):
            _bad_payload(coverage)
            continue
        outcome = entry.get("status")
        if outcome == "refusal":
            _count(coverage, "refused_units")
        elif outcome == "error":
            _count(coverage, "failed_units")
        elif outcome == "no_answer":
            _count(coverage, "missing_units")
        elif _text(entry.get("label")):
            _count(coverage, "usable_units")
        else:
            _count(coverage, "missing_units")
        members = _object(entry.get("cohort")).get("members", [])
        if isinstance(members, list):
            coverage.degraded_judgements += sum(
                isinstance(member, dict)
                and member.get("status") in ("error", "refusal", "no_answer")
                for member in members
            )
        verifier = _object(entry.get("verifier"))
        if verifier.get("status") in ("error", "refusal", "no_answer"):
            coverage.degraded_judgements += 1


def _positive_int(value: Any) -> int:
    return max(0, value) if isinstance(value, int) else 0
