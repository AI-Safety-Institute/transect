"""The public transect/load/render functions, end to end on real logs."""

import pandas as pd
import pytest
from helpers import JSON_SPEC, demo_judge
from inspect_ai._util.registry import registry_info
from inspect_ai.event import timeline_build
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig
from inspect_ai.model._model import Model

import transect.api as api
from transect import load, reliability, transect
from transect.api import _run, _scanners, _viewer_exit_mode
from transect.spec import Spec


def test_triage_runs_the_default_path_end_to_end(demo_log, tmp_path, capsys):
    """One $0 transect() call scans the log, mounts the frames, renders the
    report at the default path, and warns that judged surfaces are off."""
    results = transect(
        logs=str(demo_log),
        spec=str(demo_log.parents[2] / "examples" / "spec.yaml"),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    (report,) = results.report_paths
    html = open(report).read()
    assert "Token telemetry" in html and "Sub-agent activity" in html
    assert len(results.token_timeline) == 17
    assert "no judge_models" in capsys.readouterr().out
    assert set(results.frames()) == {
        "token_timeline",
        "flushes",
        "interventions",
        "lane_activity",
        "transcript_info",
        "phases",
        "phase_turns",
        "turn_groups",
        "phase_turn_votes",
        "subagents",
        "subagent_votes",
        "label_definitions",
    }


def test_structural_scanners_read_logs_with_stored_timelines(demo_log, tmp_path):
    """A log whose samples store their timeline scans without a recorded
    error on any structural scanner and numbers the same turns as the
    same log without one (inspect_scout < 0.6.0 failed every scanner
    that filters events on such a log)."""
    log = read_eval_log(str(demo_log))
    for sample in log.samples or []:
        sample.timelines = [timeline_build(sample.events)]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    write_eval_log(log, str(logs_dir / "timelines.eval"))
    stored, plain = (
        transect(
            logs=str(source),
            spec=Spec(),
            scans_dir=str(tmp_path / f"scans_{name}"),
            viewer=False,
            open_report=False,
        )
        for name, source in [("stored", logs_dir), ("plain", demo_log)]
    )
    assert not stored.scan_status.has_failures, stored.scan_status.errors
    assert [s.errors for s in stored.scan_status.scanners] == [0] * 4
    assert (
        stored.token_timeline["turn"].tolist() == plain.token_timeline["turn"].tolist()
    )
    assert len(stored.transcript_info) == 1
    assert stored.transcript_info["scan_error"].isna().all()


def test_scan_knobs_reach_scout(demo_log, tmp_path, monkeypatch):
    """The Scout scheduling knobs transect() exposes are passed to the scan
    call as given, and the scan still completes at $0."""
    real, seen = api.scout_scan, {}

    def capture(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(api, "scout_scan", capture)
    config = GenerateConfig(max_connections=2, timeout=30)
    results = transect(
        logs=str(demo_log),
        spec=Spec(),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        max_transcripts=3,
        model_config=config,
        results_buffer=5,
        fail_on_error=True,
    )
    assert not results.scan_status.has_failures
    assert seen["max_transcripts"] == 3 and seen["model_config"] == config
    assert seen["results_buffer"] == 5 and seen["fail_on_error"] is True
    assert seen["max_processes"] == 1


def test_the_callers_model_config_is_not_written_to(demo_log, tmp_path):
    """Scout fills max_connections on the scan-level config it is handed;
    the object the caller passed stays as they built it, so reusing it
    across runs cannot carry one run's limit into the next."""
    config = GenerateConfig(timeout=30)
    transect(
        logs=str(demo_log),
        spec=Spec(),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        max_transcripts=4,
        model_config=config,
    )
    assert config.max_connections is None


@pytest.mark.parametrize(
    "field, value", [("batch", True), ("temperature", 0.5), ("max_tokens", 100)]
)
def test_a_model_config_field_the_judges_never_see_is_refused(
    demo_log, tmp_path, field, value
):
    """inspect_ai passes only the connection, retry, timeout and cache
    settings of the scan-level config on to the judge models, so a
    sampling or batching field there would silently do nothing: it is
    refused up front, naming the field and where it belongs."""
    with pytest.raises(ValueError, match=f"{field}.*on the judge"):
        transect(
            logs=str(demo_log),
            spec=Spec(),
            scans_dir=str(tmp_path / "scans"),
            viewer=False,
            open_report=False,
            model_config=GenerateConfig(**{field: value}),
        )


def test_the_scan_level_connection_limit_governs_every_judge_call(
    demo_log, tmp_path, monkeypatch
):
    """The scan-level max_connections (Scout fills it from
    max_transcripts) overrides a judge's own on every judged path, phase
    judge, verifier, narrator and sub-agent judge alike, while the
    judge's own sampling settings survive: the contract the docs state."""
    resolved = []
    real = Model._resolve_config

    def record(self, config):
        resolved.append(real(self, config))
        return resolved[-1]

    monkeypatch.setattr(Model, "_resolve_config", record)
    judge = demo_judge()
    judge.config = GenerateConfig(temperature=0.25, max_connections=1)
    transect(
        logs=str(demo_log),
        spec=str(demo_log.parents[2] / "examples" / "spec.yaml"),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        judge_models=judge,
        verify=True,
        verify_sample=1.0,
        max_transcripts=7,
    )
    assert len(resolved) > 10
    assert {(c.max_connections, c.temperature) for c in resolved} == {(7, 0.25)}


def test_judge_call_shaping_reaches_the_scanner_factories(
    demo_log, tmp_path, monkeypatch
):
    """chunk, verify_chunk, narrate and subagent_activity travel from
    transect() to decision_phases and subagent_classification by name,
    and the judged scan still completes at $0 with them applied."""
    seen = {}
    for name in ["decision_phases", "subagent_classification"]:
        real = getattr(api, name)

        def capture(spec, _real=real, _name=name, **kw):
            seen[_name] = kw
            return _real(spec, **kw)

        monkeypatch.setattr(api, name, capture)
    results = transect(
        logs=str(demo_log),
        spec=str(demo_log.parents[2] / "examples" / "spec.yaml"),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        judge_models=demo_judge(),
        verify=False,
        chunk=10,
        verify_chunk=4,
        narrate=False,
        subagent_activity=True,
    )
    assert not results.scan_status.has_failures
    phases, subagents = seen["decision_phases"], seen["subagent_classification"]
    assert (phases["chunk"], phases["verify_chunk"]) == (10, 4)
    assert phases["narrate"] is False
    assert subagents["activity"] is True
    assert (results.phases["headline"] == "").all()


def test_load_rereads_a_scan_identically_without_rescanning(demo_log, tmp_path):
    """load() on a finished scans_dir reproduces the same frames from
    disk, with no scanning and no report render."""
    ran = _run(logs=str(demo_log), spec=Spec(), scans_dir=str(tmp_path / "scans"))
    loaded = load(str(tmp_path / "scans"))
    assert loaded.scan_location == ran.scan_location
    pd.testing.assert_frame_equal(
        loaded.token_timeline.sort_values("turn").reset_index(drop=True),
        ran.token_timeline.sort_values("turn").reset_index(drop=True),
    )


def test_epochs_all_renders_one_report_per_epoch(epochs_logs, tmp_path):
    """A multi-epoch sample under epochs="all" produces a report file
    per epoch, each named by its epoch."""
    results = transect(
        logs=str(epochs_logs),
        spec=JSON_SPEC,
        sample="epoch-sample-1",
        epochs="all",
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
    )
    assert len(results.report_paths) == 3
    assert sorted(results.token_timeline.epoch.unique()) == [1, 2, 3]
    assert results.token_timeline.transcript_id.nunique() == 3
    assert all(s.total_transcripts == 3 for s in results.scan_status.scanners)
    for path in results.report_paths:
        assert 'id="scan-status"' in open(path).read()


def test_viewer_lifetime_follows_the_execution_context(monkeypatch):
    """Interactive sessions terminate the viewer at exit, a TTY script
    holds it for Ctrl+C, and a non-TTY script detaches it so the printed
    URL outlives the process."""
    assert _viewer_exit_mode(interactive=True, tty=True) == "terminate"
    assert _viewer_exit_mode(interactive=False, tty=True) == "hold"
    assert _viewer_exit_mode(interactive=False, tty=False) == "detach"

    calls, registered = {}, []

    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(
        "transect.api.subprocess.Popen", lambda cmd, **kw: calls.update(kw) or Proc()
    )
    monkeypatch.setattr("transect.api.atexit.register", lambda *a: registered.append(a))
    monkeypatch.setattr(api, "_viewer_exit_mode", lambda **_: "detach")
    url = api._spawn_viewer("some/transcripts")
    assert url.startswith("http://127.0.0.1:")
    assert calls.get("start_new_session") is True
    assert registered == []


def test_scanner_set_matches_the_spec_vocabularies():
    """decision_phases only joins the batch when the spec declares
    phases; subagent_classification when it declares labels."""

    def names(spec):
        return [registry_info(s).name for s in _scanners(spec, "mockllm/model")]

    with_labels = names(Spec.model_validate({"subagent_labels": ["helper"]}))
    assert not any("decision_phases" in n for n in with_labels)
    assert any("subagent_classification" in n for n in with_labels)
    with_phases = names(Spec.model_validate({"phases": ["recon"]}))
    assert any("decision_phases" in n for n in with_phases)


@pytest.mark.parametrize(
    "regime",
    ["solo", "solo-verified", "k-roll", "cohort"],
)
def test_triage_judged_regimes_end_to_end(regime, demo_log, tmp_path):
    """The full judged pipeline on the demo log, per judge regime: the
    frames carry the scripted labels and the regime's own reliability
    markers, and the report renders."""
    setups = {
        "solo": dict(judge_models=demo_judge(), verify=False),
        # verify_sample=1.0: the spot check is a seeded draw per item id,
        # so a regenerated log would otherwise decide how many get picked
        "solo-verified": dict(
            judge_models=demo_judge(), verify=True, verify_sample=1.0
        ),
        "k-roll": dict(judge_models=demo_judge(), k_rolls=3),
        "cohort": dict(judge_models=[demo_judge(), demo_judge(model="mockllm/model2")]),
    }
    results = transect(
        logs=str(demo_log),
        spec=str(demo_log.parents[2] / "examples" / "spec.yaml"),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        **setups[regime],
    )

    phases = results.phases
    assert set(phases.phase) == {"model_development"}
    assert (phases.confidence == 0.9).all()
    subagents = results.subagents
    assert dict(zip(subagents.agent_lane, subagents.label, strict=True)) == {
        "eda": "data_analysis",
        "alt_model": "model_experimentation",
        "reviewer": "result_review",
    }
    assert "Traceback" not in open(results.report_paths[0]).read()

    votes = results.phase_turn_votes
    if regime == "solo":
        assert len(votes) == 0
        assert (phases.confidence_source == "single_judge").all()
        assert not phases.verifier_selected.any()
    if regime == "solo-verified":
        reviewed = phases[phases.verifier_selected]
        assert len(reviewed) == 1
        assert (reviewed.verifier_trigger == "random_sample").all()
        assert not reviewed.overturned.any()
        spans = results.subagents
        assert spans.verifier_model.notna().all()
        spot = spans[spans.verifier_selected.fillna(False)]
        assert len(spot) == 3
        assert (spot.verifier_trigger == "random_sample").all()
        assert not spot.overturned.any()
    if regime == "k-roll":
        assert set(votes.roll) == {0, 1, 2}
        assert (votes[votes.basis == "judged"].phase == "model_development").all()
        assert (results.subagents.label_source == "majority_vote").all()
        assert (results.subagents.n_members == 3).all()
    if regime == "cohort":
        assert votes.model.nunique() == 2
        assert (phases.confidence_source == "majority_vote").all()
        assert (results.subagents.judge_models == "mockllm/model+mockllm/model2").all()


def test_verify_sample_zero_renders_an_armed_idle_verifier(demo_log, tmp_path):
    """verify_sample=0.0 with confident answers arms the verifier but
    examines nothing; the audit renders without the re-label rows."""
    results = transect(
        logs=str(demo_log),
        spec=str(demo_log.parents[2] / "examples" / "spec.yaml"),
        scans_dir=str(tmp_path / "scans"),
        viewer=False,
        open_report=False,
        judge_models=demo_judge(),
        verify=True,
        verify_sample=0.0,
    )
    assert not results.phases.verifier_selected.any()
    assert results.phases.verifier_model.notna().all()
    html = open(results.report_paths[0]).read()
    assert "Traceback" not in html
    assert "Verifier re-label rate" not in html


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        (dict(judge_models=["mockllm/model", "mockllm/model2"], verify=True), "XOR"),
        (
            dict(judge_models=["mockllm/model", "mockllm/model2"], k_rolls=3),
            "mutually exclusive",
        ),
    ],
    ids=["verifier-with-cohort", "cohort-with-k-rolls"],
)
def test_triage_rejects_contradictory_judge_setups(setup, match, tmp_path):
    """The regime rules hold at the public entry point, before any
    scanning: verifier XOR cohort, and cohort XOR k-rolls."""
    spec = Spec.model_validate({"phases": ["recon"], "subagent_labels": ["helper"]})
    with pytest.raises(ValueError, match=match):
        transect(
            logs=str(tmp_path / "never-touched"),
            spec=spec,
            scans_dir=str(tmp_path / "scans"),
            viewer=False,
            open_report=False,
            **setup,
        )


def test_public_reliability_surface_is_complete():
    """Every name transect.reliability declares in __all__ resolves - a
    rename that leaves a stale export entry fails here."""
    missing = [name for name in reliability.__all__ if not hasattr(reliability, name)]
    assert missing == []


def test_empty_judge_models_list_is_rejected(tmp_path):
    """judge_models=[] is a contradiction, not a judge-less run: it
    fails loudly at the entry point instead of silently degrading to
    the mechanical scanner set (that is judge_models=None's job)."""
    spec = Spec.model_validate({"phases": ["recon"], "subagent_labels": ["helper"]})
    with pytest.raises(ValueError, match="empty list"):
        transect(
            logs=str(tmp_path / "never-touched"),
            spec=spec,
            scans_dir=str(tmp_path / "scans"),
            viewer=False,
            open_report=False,
            judge_models=[],
        )


def test_load_without_a_scan_fails_loudly(tmp_path):
    """load() on a directory holding no scan names the problem itself
    rather than handing an unreadable path to the store reader."""
    with pytest.raises(FileNotFoundError, match="no scan found"):
        load(str(tmp_path))
