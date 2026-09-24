"""Run-wide execution record for the stored scan.

Scout completion describes execution, not usable labels. This module
projects only execution-level facts: which requested scanners ran, on
how many transcripts, and what errors the store recorded. Judgement
quality (abstentions, filled labels, member and verifier degradation)
is the reliability audit's territory, computed per transcript from the
mounted frames.
"""

from dataclasses import dataclass, field

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
    """One requested scanner's run-wide execution record.

    ``scanned_transcripts`` counts scan attempts (the scanner reached
    the transcript, successfully or not); ``completed_transcripts``
    subtracts the transcripts with a recorded scan error. ``errors``
    counts this scanner's entries in the run's error list: recorded
    scan errors plus store anomalies such as a results table missing
    despite recorded scans (e.g. a partially copied store).
    """

    scanner: str
    mounted: bool
    total_transcripts: int | None = None
    scanned_transcripts: int = 0
    completed_transcripts: int = 0
    errors: int = 0

    @property
    def missing_scans(self) -> int | None:
        """Transcripts in scope that no recorded scan reached."""
        if self.total_transcripts is None:
            return None
        return max(0, self.total_transcripts - self.scanned_transcripts)

    @property
    def never_attempted(self) -> bool:
        """Requested, but no scan ran and the store records no scope."""
        return self.total_transcripts is None and self.scanned_transcripts == 0

    @property
    def has_failures(self) -> bool:
        """Whether stored evidence identifies incomplete or errored execution."""
        return bool(self.missing_scans or self.errors or self.never_attempted)


@dataclass
class ScanStatus:
    """Run-wide status; the default explicitly leaves execution unknown."""

    outer_complete: bool | None = None
    scanners: list[ScannerCoverage] = field(default_factory=list)
    errors: list[ScanError] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """Whether recorded execution is incomplete or errored anywhere."""
        return (
            self.outer_complete is False
            or bool(self.errors)
            or any(scanner.has_failures for scanner in self.scanners)
        )


def build_scan_status(raw: ScanResultsDF, mounted_scanners: set[str]) -> ScanStatus:
    """Project persisted Scout execution metadata.

    The caller can exclude Scout's heavy columns. No transcript reads,
    model calls or custom frame functions are needed here. Total
    transcript counts remain unknown when neither a worklist nor the
    stored transcript snapshot names the scan's scope.
    """
    status = ScanStatus(outer_complete=raw.complete)
    for error in raw.errors:
        status.errors.append(
            ScanError(error.scanner, error.transcript_id, error.error, error.refusal)
        )
    for key, spec in raw.spec.scanners.items():
        total: int | None = None
        if raw.spec.worklist is not None:
            total = len(
                {
                    tid
                    for work in raw.spec.worklist
                    if work.scanner == key
                    for tid in work.transcripts
                }
            )
        elif raw.spec.transcripts and raw.spec.transcripts.transcript_ids:
            total = len(raw.spec.transcripts.transcript_ids)
        summary = raw.summary.scanners.get(key)
        # summary.scans counts scan events, which a retry-resume can
        # inflate past the transcript count; cap at the known scope
        scanned = summary.scans if summary else 0
        if total is not None:
            scanned = min(scanned, total)
        coverage = ScannerCoverage(
            scanner=key,
            mounted=key in mounted_scanners or spec.name.startswith("transect/"),
            total_transcripts=total,
            scanned_transcripts=scanned,
        )
        table = raw.scanners.get(key)
        table_missing = table is None and coverage.scanned_transcripts > 0
        if table is not None:
            for _, row in table.iterrows():
                error_message = _text(row.get("scan_error"))
                if error_message is None:
                    continue
                record = ScanError(
                    key,
                    _text(row.get("transcript_id")),
                    error_message,
                    _text(row.get("scan_error_type")) == "refusal",
                )
                if record not in status.errors:
                    status.errors.append(record)
        elif table_missing:
            status.errors.append(
                ScanError(
                    key,
                    None,
                    "The execution summary records scans but their results "
                    "table is missing - typically an incompletely copied "
                    "scan directory; re-copy the full store or re-run the "
                    "scan.",
                )
            )
        # count recorded entries, not the summary counter: a resume can
        # leave a stale summary errors count behind resolved retries
        coverage.errors = sum(error.scanner == key for error in status.errors)
        errored = {
            error.transcript_id
            for error in status.errors
            if error.scanner == key and error.transcript_id is not None
        }
        # a missing results table leaves no usable output at all
        coverage.completed_transcripts = (
            0 if table_missing else max(0, coverage.scanned_transcripts - len(errored))
        )
        status.scanners.append(coverage)
    return status


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None
