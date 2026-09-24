"""The public Transect API.

``transect()`` runs the default layers over the logs and produces the
report; ``load()`` re-reads an existing scan; ``render()`` renders
the HTML report (and brings up the Scout viewer) from either.
"""

import asyncio
import atexit
import dataclasses
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from inspect_ai.model import Model
from inspect_scout import (
    columns,
    scan as scout_scan,
    scan_results_df,
    transcripts_db,
    transcripts_from,
)

from transect.frames import (
    TransectResults,
    flushes_df,
    interventions_df,
    label_definitions_df,
    lane_activity_df,
    phase_turn_votes_df,
    phase_turns_df,
    phases_df,
    subagent_votes_df,
    subagents_df,
    token_timeline_df,
    transcript_info_df,
    turn_groups_df,
)
from transect.frames.results import builtin_frame_names
from transect.frames.user import layer_frame
from transect.layers import (
    Layer,
    resolve_scanner_factories,
    scanner_key,
    validate_audit,
    validate_layers,
)
from transect.report import render_report
from transect.report.blocks import validate_section
from transect.report.render import validate_section_order
from transect.scan_status import build_scan_status
from transect.scanners.base import (
    context_flush,
    eval_setup,
    human_intervention,
    token_timeline,
)
from transect.scanners.phases import decision_phases
from transect.scanners.subagents import subagent_classification
from transect.selection import read_index, select_transcripts
from transect.spec import Spec, load_spec
from transect.tags import turn_tags_frame


def transect(
    logs: str | list[str],
    spec: str | Path | Spec,
    *,
    sample: str | None = None,
    epochs: int | list[int] | Literal["all"] | None = None,
    judge_models: str | Model | list[str | Model] | None = None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verifier_model: str | Model | None = None,
    verify_sample: float | None = None,
    scans_dir: str = "scans",
    report_path: str | None = None,
    title: str | None = None,
    viewer: bool = True,
    open_report: bool = True,
    max_processes: int = 1,
    extra_layers: list[Layer] | None = None,
    section_order: list[str] | None = None,
    sensitivity: str | None = None,
) -> TransectResults:
    """The default path: scan the logs, build the frames, render the
    report, bring up the Scout viewer.

    Every run extracts the structural ($0) surfaces: per-turn token
    usage, context-window compactions, mid-run human interventions,
    and sub-agent activity lanes. With ``judge_models`` set, the
    judged surfaces are added: the run is segmented into labelled,
    narrated phases, and each spawned sub-agent is classified
    against the spec's subagent labels.

    The judge setup: one model judges solo, with a second-round
    verifier re-checking its doubtful judgements; several models (or
    ``k_rolls`` repeats of one) form a cohort whose majority vote
    decides the labels.

    Args:
        logs: Inspect ``.eval`` log file(s)/dir, or OpenClaw ``.jsonl``.
        spec: The eval description (phase + sub-agent label
            vocabularies, task context) - a ``.json``/``.yaml`` path
            or a loaded ``Spec``.
        sample: The sample id to scan, a run scans one sample.
            Optional for a single-sample log; required (the error
            lists the available ids) when the log carries several.
        epochs: Epoch selection for a multi-epoch ``.eval`` log
            (1-based); ``"all"`` scans every epoch (the report
            splits into one file per epoch); ``None`` = auto - the
            sample's earliest successful epoch (the earliest epoch
            when none succeeded).
        judge_models: One model (solo), or several (majority-vote
            cohort). ``None`` disables built-in judged scanners; custom
            ``extra_layers`` may still call models and incur charges.
        k_rolls: Repeated rolls of one model (mutually exclusive with
            a multi-model list).
        verify: The second-round verifier, a re-review of doubtful
            phase labels and sub-agent classifications. ``None`` =
            auto: on for a single judge model (solo and k-roll), off
            for a multi-model cohort. ``True`` forces on (raises
            with a cohort); ``False`` = off.
        verifier_model: Verifier judge; ``None`` uses the (first)
            model from ``judge_models``.
        verify_sample: The share of judged units (phases and
            sub-agent spans) the verifier additionally spot-checks at
            random. ``None`` = the default 5% (phases floored at 3);
            ``0.0`` disables; ``1.0`` reviews everything. Phases
            realize the share as an exact stratified draw, spans
            per-span at that probability. No effect when the
            verifier is off (cohort regime, or ``verify=False``);
            values outside [0, 1] are rejected.
        scans_dir: Where the scan store (and report) live.
        report_path: Where the report lands. A ``.html`` file path;
            an existing directory gets ``report.html`` placed inside
            it; ``None`` = ``<scans_dir>/report.html``.
        title: Report title; ``None`` is passed through to ``render()``,
            which derives it from the recovered task name when possible.
        viewer: Spawn a Scout viewer and wire the report's deep
            links. On a TTY the viewer lives and dies with this
            Python process (Ctrl+C to exit); without one (a coding
            agent, CI) it is left serving at the printed URL.
        open_report: Open the rendered HTML in the browser.
        max_processes: Parallel scan worker processes.
        extra_layers: User-injected `Layer` additions: each layer's
            scanner joins the scan batch and its frame mounts at
            ``results.layer_frames[name]``.
        section_order: Report section order: the named sections render
            first, in this order; unnamed ones follow in the default
            reading order. Keys are ``transect.report.SECTION_KEYS`` plus
            each custom layer's name; the reliability audit always
            renders last, and the run-wide Scan execution & coverage
            block renders once after all transcripts.
        sensitivity: A protective marking (e.g. "OFFICIAL SENSITIVE")
            rendered verbatim as fixed banners at the top and bottom
            of the report and appended to the report title. ``None``
            (the default) renders no marking.

    Returns:
        ``TransectResults`` - the frames (plain pandas), plus
        ``report_paths``/``viewer_url``.
    """
    # a typo'd section key must fail here, not after the judge spend
    validate_section_order(section_order, [layer.name for layer in extra_layers or []])
    loaded_spec = spec if isinstance(spec, Spec) else load_spec(spec)
    if judge_models is None:
        print(
            "WARNING: no judge_models - the judged surfaces (decision "
            "phases, sub-agent classification) will be absent from the "
            "report"
        )
    else:
        if not loaded_spec.phases:
            print(
                "WARNING: the spec declares no phases - the report "
                "will have no phase segmentation"
            )
        if not loaded_spec.subagent_labels:
            print(
                "WARNING: the spec declares no subagent_labels - "
                "the report will have no sub-agent classification"
            )
    results = _run(
        logs=logs,
        spec=loaded_spec,
        sample=sample,
        epochs=epochs,
        judge_models=judge_models,
        k_rolls=k_rolls,
        verify=verify,
        verifier_model=verifier_model,
        verify_sample=verify_sample,
        scans_dir=scans_dir,
        max_processes=max_processes,
        extra_layers=extra_layers,
    )
    return render(
        results,
        report_path=report_path,
        title=title,
        viewer=viewer,
        open_report=open_report,
        section_order=section_order,
        sensitivity=sensitivity,
    )


def load(scans_dir: str, extra_layers: list[Layer] | None = None) -> TransectResults:
    """Re-read an existing scan into frames and run-wide scan_status.

    No scanning, API calls or report render. Available partial results keep
    their recorded errors and coverage, provided required structural tables and
    any mounted custom-frame contracts can be loaded. Otherwise loading raises.

    Args:
        scans_dir: A transect() ``scans_dir`` (the latest scan in it is
            loaded) or a specific scan directory.
            ``results.scan_location`` records the exact scan resolved.
        extra_layers: The run's `Layer` list, to remount their frames
            from the stored scan (the frame fns re-run; a layer whose
            scanner is absent from the store fails loudly).
    """
    scan_location = str(_resolve_scan(scans_dir))
    claimed = {
        scanner_key(layer.scanner)
        for layer in extra_layers or []
        if layer.scanner is not None
    }
    results = scan_results_df(scan_location, exclude_columns=["input"])
    unclaimed = sorted(
        key
        for key, scanner in results.spec.scanners.items()
        if not scanner.name.startswith("transect/") and key not in claimed
    )
    if unclaimed:
        print(
            "WARNING: the scan store carries results for custom "
            f"scanner(s) {', '.join(unclaimed)} that no layer claims - "
            "they will not be read into frames or the report; pass the "
            "matching extra_layers to include them"
        )
    raw = results.scanners["token_timeline"]
    token_timeline = token_timeline_df(raw)
    lane_activity = lane_activity_df(raw)
    # judge scanners are opt-in: absent from the scan -> empty frames
    raw_phases = results.scanners.get("decision_phases", pd.DataFrame())
    raw_subagents = results.scanners.get("subagent_classification", pd.DataFrame())
    transcripts = getattr(results.spec, "transcripts", None)
    transect_results = TransectResults(
        scan_location=scan_location,
        transcripts_location=_transcripts_location(transcripts),
        scan_status=build_scan_status(results, mounted_scanners=claimed),
        token_timeline=token_timeline,
        flushes=flushes_df(results.scanners["context_flush"], token_timeline),
        interventions=interventions_df(results.scanners["human_intervention"]),
        lane_activity=lane_activity,
        transcript_info=transcript_info_df(results.scanners["eval_setup"]),
        phases=phases_df(
            raw_phases,
            phase_turns=(phase_turns := phase_turns_df(raw_phases)),
            token_timeline=token_timeline,
        ),
        phase_turns=phase_turns,
        turn_groups=turn_groups_df(raw_phases),
        phase_turn_votes=phase_turn_votes_df(raw_phases),
        subagents=subagents_df(
            raw_subagents, lane_activity=lane_activity, token_timeline=token_timeline
        ),
        subagent_votes=subagent_votes_df(raw_subagents),
        label_definitions=label_definitions_df(
            raw_phases,
            raw_subagents,
            layer_results={
                layer.name: results.scanners[key]
                for layer in extra_layers or []
                if layer.scanner is not None
                and (key := scanner_key(layer.scanner)) in results.scanners
            },
        ),
        layer_frames=(mounted := _layer_frames(results, extra_layers or [])),
        turn_tags=turn_tags_frame(
            extra_layers or [], mounted, _n_turns(token_timeline)
        ),
        extra_layers=list(extra_layers or []),
    )
    run_tids = (
        set(token_timeline.transcript_id.astype(str)) if len(token_timeline) else set()
    )
    for layer in transect_results.extra_layers:
        frame = transect_results.layer_frames.get(layer.name)
        if (
            run_tids
            and frame is not None
            and len(frame)
            and "transcript_id" in frame.columns
            and not set(frame.transcript_id.dropna().astype(str)) & run_tids
        ):
            raise ValueError(
                f"layer {layer.name!r}: frame shares no transcript_id "
                "with this scan, so its rows can never join - check them "
                "against results.token_timeline.transcript_id"
            )
        if layer.section:
            validate_section(layer, frame)
        if layer.audit is not None:
            validate_audit(layer, frame)
    return transect_results


def render(
    results: TransectResults,
    *,
    report_path: str | None = None,
    title: str | None = None,
    viewer: bool = True,
    open_report: bool = True,
    section_order: list[str] | None = None,
    sensitivity: str | None = None,
) -> TransectResults:
    """Render the HTML report from TransectResults produced by
    transect() or load().

    Fills ``report_paths``/``viewer_url`` on the given results and
    returns that same object. Normally one report file renders;
    when the frames carry more than one epoch, one file renders per
    epoch (``report_epoch2.html``, ...).

    Args:
        results: The frames to render (from transect() or load()).
        report_path: Where the report lands. A ``.html`` file path;
            an existing directory gets ``report.html`` placed inside
            it; ``None`` = a ``report.html`` next to the scan store.
        title: Report title; ``None`` derives from the recovered task
            name (``results.transcript_info.task_name``), falling back
            to the scan's transcripts location.
        viewer: Spawn a Scout viewer and wire the report's deep
            links. On a TTY the viewer lives and dies with this
            Python process (Ctrl+C to exit); without one (a coding
            agent, CI) it is left serving at the printed URL.
        open_report: Open the rendered HTML in the browser.
        section_order: Report section order: the named sections render
            first, in this order; unnamed ones follow in the default
            reading order. Keys are ``transect.report.SECTION_KEYS`` plus
            each custom layer's name; the reliability audit always
            renders last, and the run-wide Scan execution & coverage
            block renders once after all transcripts.
        sensitivity: A protective marking (e.g. "OFFICIAL SENSITIVE")
            rendered verbatim as fixed banners at the top and bottom
            of the report and appended to the report title. ``None``
            (the default) renders no marking.
    """
    scan_path = Path(results.scan_location)
    base = scan_path.parent if scan_path.name.startswith("scan_id=") else scan_path
    path = Path(report_path) if report_path else base / "report.html"
    if path.is_dir():  # a directory: place the report inside it
        path = path / "report.html"
    if title is None:
        stem = _task_name(results) or (
            Path(results.transcripts_location).stem
            if results.transcripts_location
            else base.name
        )
        title = f"Transect report · {stem}"
    if sensitivity:
        title = f"{title} · {sensitivity}"
    viewer_url = None
    if viewer and results.transcripts_location:
        viewer_url = _spawn_viewer(results.transcripts_location)
    path.parent.mkdir(parents=True, exist_ok=True)
    # split per-epoch only when some sample carries several epochs
    # (epochs="all"/list); auto-selection picks one epoch per sample
    epochs = (
        sorted(int(e) for e in results.token_timeline.epoch.dropna().unique())
        if len(results.token_timeline)
        and results.token_timeline.groupby("sample_id").epoch.nunique().max() > 1
        else []
    )
    views = (
        [
            (
                path.with_name(f"{path.stem}_epoch{epoch}{path.suffix}"),
                f"{title} · epoch {epoch}",
                _epoch_view(results, epoch),
            )
            for epoch in epochs
        ]
        if epochs
        else [(path, title, results)]
    )
    paths: list[str] = []
    for out, view_title, view in views:
        out.write_text(
            render_report(
                view,
                title=view_title,
                viewer_base_url=viewer_url,
                section_order=section_order,
                sensitivity=sensitivity,
            )
        )
        paths.append(str(out))
    results.report_paths = paths
    results.viewer_url = viewer_url
    for p in paths:
        print(f"report  -> {p}")
    if viewer_url is not None:
        print(f"viewer  -> {viewer_url}  (deep links active)")
    if open_report:
        webbrowser.open(Path(paths[0]).resolve().as_uri())
    return results


def _run(
    logs: str | list[str],
    spec: Spec,
    *,
    sample: str | None = None,
    epochs: int | list[int] | Literal["all"] | None = None,
    judge_models: str | Model | list[str | Model] | None = None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verifier_model: str | Model | None = None,
    verify_sample: float | None = None,
    scans_dir: str = "scans",
    max_processes: int = 1,
    extra_layers: list[Layer] | None = None,
) -> TransectResults:
    """Scan the logs with the Transect scanner set and return the frames.

    ``logs`` may be Inspect ``.eval`` logs (scanned directly) or OpenClaw
    telemetry ``.jsonl`` files, which are first imported into a transcript
    retained snapshot database under ``scans_dir`` and scanned from there.
    """
    extra_layers = resolve_scanner_factories(
        list(extra_layers or []),
        {
            "judge_models": judge_models,
            "k_rolls": k_rolls,
            "verify": verify,
            "verifier_model": verifier_model,
            "verify_sample": verify_sample,
        },
    )
    validate_layers(extra_layers, builtin_frame_names())
    jsonl = _openclaw_files(logs)
    scanners = _scanners(
        spec,
        judge_models,
        k_rolls=k_rolls,
        verify=verify,
        verifier_model=verifier_model,
        verify_sample=verify_sample,
    )
    if jsonl:
        # Keep new snapshots outside an existing database's recursive search root.
        logs = os.path.join(scans_dir, "transcript_snapshots", uuid4().hex)
        asyncio.run(_import_openclaw(jsonl, logs))
    transcripts = transcripts_from(logs if isinstance(logs, str) else list(logs))
    # one-sample rule + epoch selection
    selection = select_transcripts(
        asyncio.run(read_index(transcripts)), sample=sample, epochs=epochs
    )
    for note in selection.notes:
        print(f"WARNING: {note}")
    if selection.transcript_ids is not None:
        transcripts = transcripts.where(
            columns.transcript_id.in_(selection.transcript_ids)
        )
    # no scan-level reuse: every call performs a fresh scan (judge
    # calls are deduplicated by inspect-ai's generate cache)
    scanners += [
        layer.scanner for layer in extra_layers or [] if layer.scanner is not None
    ]
    status = scout_scan(
        scanners=scanners,
        transcripts=transcripts,
        scans=scans_dir,
        max_processes=max_processes,
    )
    return load(status.location, extra_layers=extra_layers)


def _scanners(
    spec,
    judge_models: str | Model | list[str | Model] | None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verifier_model: str | Model | None = None,
    verify_sample: float | None = None,
) -> list:
    """The scanner set for a run: structural always; judge scanners when a
    model is given - decision_phases additionally needs a phase vocab.
    subagent_classification additionally needs subagent_labels."""
    scanners = [
        token_timeline(),
        context_flush(),
        human_intervention(),
        eval_setup(),
    ]
    if judge_models is not None:
        if isinstance(judge_models, list) and not judge_models:
            raise ValueError(
                "judge_models is an empty list - pass at least one "
                "judge model (or None for the judge-less scanner set)"
            )
        if spec.phases:
            scanners.append(
                decision_phases(
                    spec,
                    judge_models=judge_models,
                    k_rolls=k_rolls,
                    verify=verify,
                    verifier_model=verifier_model,
                    verify_sample=verify_sample,
                )
            )
        if spec.subagent_labels:
            scanners.append(
                subagent_classification(
                    spec,
                    judge_models=judge_models,
                    k_rolls=k_rolls,
                    verify=verify,
                    verifier_model=verifier_model,
                    verify_sample=verify_sample,
                )
            )
    return scanners


def _openclaw_files(logs: str | list[str]) -> list[Path]:
    """OpenClaw telemetry .jsonl paths in ``logs``, or [] for eval logs."""
    paths = [Path(p) for p in ([logs] if isinstance(logs, str) else logs)]
    jsonl: list[Path] = []
    eval_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            jsonl.extend(sorted(p.glob("*.jsonl")))
            if any(p.glob("*.eval")) or any(
                _is_inspect_json_log(j) for j in p.glob("*.json")
            ):
                eval_paths.append(p)
        elif p.suffix == ".jsonl":
            jsonl.append(p)
        else:
            eval_paths.append(p)
    if jsonl and eval_paths:
        raise ValueError(
            f"cannot mix OpenClaw .jsonl and eval logs in one run: {eval_paths[0]}"
        )
    return jsonl


def _is_inspect_json_log(path: Path) -> bool:
    """Whether a .json file looks like an Inspect eval log (vs a
    stray settings/manifest file)."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False
    return b'"version"' in head and b'"eval"' in head


async def _import_openclaw(files: list[Path], db_dir: str) -> None:
    """Import the supplied telemetry into a new DB, rejecting ambiguous identities."""
    from transect.ingestion.openclaw_telemetry_hal import openclaw_telemetry_hal

    fresh = []
    sources: dict[str, Path] = {}
    for file in files:
        async for transcript in openclaw_telemetry_hal(file):
            previous = sources.get(transcript.transcript_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate transcript id {transcript.transcript_id!r} in "
                    f"{previous} and {file}; supply only one file for this identity"
                )
            sources[transcript.transcript_id] = file
            fresh.append(transcript)
    if not fresh:
        raise ValueError("no transcripts found in the supplied OpenClaw files")
    db = transcripts_db(db_dir)
    await db.connect()
    try:
        await db.insert(fresh)
    finally:
        await db.disconnect()


def _epoch_view(results: TransectResults, epoch: int) -> TransectResults:
    """Per-epoch view: every frame filtered to the epoch's transcripts
    (via the token timeline's epoch column).

    The original results object is untouched, views only feed the report."""
    tids = set(
        results.token_timeline[results.token_timeline.epoch == epoch].transcript_id
    )

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        if "transcript_id" in frame.columns:
            return frame[frame.transcript_id.isin(tids)].reset_index(drop=True)
        return frame

    cuts: dict[str, Any] = {
        name: cut(frame) for name, frame in results.frames().items()
    }
    return dataclasses.replace(
        results,
        layer_frames={name: cut(df) for name, df in results.layer_frames.items()},
        **cuts,
    )


def _n_turns(token_timeline: pd.DataFrame) -> dict[str, int]:
    """Per-transcript turn counts off the timeline's turn indices."""
    if not len(token_timeline):
        return {}
    counts = token_timeline.groupby("transcript_id").turn.max().add(1)
    return {str(tid): int(n) for tid, n in counts.items()}


def _layer_frames(results, layers: list[Layer]) -> dict[str, pd.DataFrame]:
    """Mount each layer's frame from the scan's results tables."""
    validate_layers(layers, builtin_frame_names())
    mounted: dict[str, pd.DataFrame] = {}
    for layer in layers:
        if layer.scanner is not None:
            key = scanner_key(layer.scanner)
            if key not in results.scanners:
                raise KeyError(
                    f"scan store carries no results for layer scanner "
                    f"{key!r} - this store was scanned without it; re-run "
                    "transect() with the layer"
                )
            raw = results.scanners[key]
            if not len(raw) and (layer.section or layer.tags or layer.audit):
                raise ValueError(
                    f"layer {layer.name!r}: scanner {key!r} produced no "
                    "results for this scan, so there is nothing to "
                    "render or tag - check the layer's loader against "
                    "these transcripts"
                )
            mounted[layer.name] = layer_frame(layer, raw)
        else:
            mounted[layer.name] = layer_frame(layer, pd.DataFrame())
    return mounted


def _resolve_scan(scans_dir: str) -> Path:
    """Resolve which scan to load.

    Each transect() run adds a new scan_id=* subdirectory under its
    scans_dir. Given that parent, pick the newest by mtime; given a
    specific scan directory (no scan_id=* children), use it as-is."""
    root = Path(scans_dir)
    scans = sorted(
        (p for p in root.glob("scan_id=*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if scans:
        return scans[-1]
    if (root / "_scan.json").exists():
        return root
    raise FileNotFoundError(
        f"no scan found under {scans_dir} - expected a scans_dir with "
        "scan_id=* subdirectories, or one specific scan directory"
    )


def _task_name(results: TransectResults) -> str | None:
    """The recovered task name for the report title, or None."""
    info = results.transcript_info
    if not len(info):
        return None
    names = info.task_name.dropna()
    return str(names.iloc[0]) if len(names) else None


def _transcripts_location(transcripts) -> str | None:
    """The viewer target: transcript-DB scans record it directly;
    eval-log scans leave location empty but map transcript ids to
    their log paths.
    """
    location = getattr(transcripts, "location", None)
    if location is not None:
        return str(location)
    ids = getattr(transcripts, "transcript_ids", None) or {}
    paths = {str(p) for p in ids.values()}
    return os.path.commonpath(paths) if paths else None


def _viewer_exit_mode(*, interactive: bool, tty: bool) -> str:
    """How the spawned viewer relates to this process's lifetime.

    - "terminate": an interactive session (REPL, notebook) kills the
      viewer at interpreter exit.
    - "hold": a plain script on a TTY blocks at exit until Ctrl+C,
      then terminates it.
    - "detach": a non-TTY script (a coding agent, CI) leaves it
      serving at the printed URL.
    """
    if interactive:
        return "terminate"
    return "hold" if tty else "detach"


def _spawn_viewer(transcripts_location: str) -> str:
    """Start a ``scout view`` child on a free port.

    Process-bound in interactive/TTY contexts; detached and left
    serving when stdout is not a TTY.
    """
    mode = _viewer_exit_mode(
        interactive=hasattr(sys, "ps1") or "ipykernel" in sys.modules,
        tty=sys.stdout.isatty(),
    )
    port = _free_port()
    process = subprocess.Popen(
        [
            "scout",
            "view",
            "--transcripts",
            str(transcripts_location),
            "--port",
            str(port),
            "--no-browser",
            "--display",
            "none",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(mode == "detach"),
    )
    url = f"http://127.0.0.1:{port}"
    if mode == "hold":
        atexit.register(_hold_viewer, process, url)
    elif mode == "terminate":
        atexit.register(_terminate_viewer, process)
    return url


def _hold_viewer(process, url: str) -> None:
    """At a TTY script's exit: keep the viewer serving until Ctrl+C."""
    if process.poll() is not None:
        return  # viewer already gone
    print(f"viewer still serving at {url} - press Ctrl+C to exit")
    try:
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()


def _terminate_viewer(process) -> None:
    if process.poll() is None:
        process.terminate()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
