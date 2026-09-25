"""Demo arguments are checked before any potentially paid pipeline call."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "args,exit_code,judged",
    [
        (["--help"], 0, None),
        (["--structrual"], 2, None),
        (["--structural"], None, False),
        ([], None, True),
    ],
)
def test_demo_arguments_before_pipeline(monkeypatch, args, exit_code, judged):
    """Help and invalid flags stop; documented modes retain their judge settings."""
    calls = []

    def pipeline(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(report_paths=["report.html"], viewer_url=None)

    monkeypatch.setitem(sys.modules, "transect", SimpleNamespace(transect=pipeline))
    script = Path(__file__).parents[1] / "examples/transect_kroll.py"
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    if exit_code is not None:
        with pytest.raises(SystemExit) as raised:
            runpy.run_path(str(script), run_name="__main__")
        assert raised.value.code == exit_code
        assert calls == []
    else:
        runpy.run_path(str(script), run_name="__main__")
        assert len(calls) == 1
        assert ("judge_models" in calls[0]) is judged
