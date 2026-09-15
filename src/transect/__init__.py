"""Transect: transcript analysis for agent evals."""

from importlib.metadata import version as _version

from transect import reliability
from transect.api import load, render, transect
from transect.frames import TransectResults
from transect.frames.common import judge_identity
from transect.frames.user import member_ballots, turns_frame
from transect.layers import Layer
from transect.scanners.cohort import cohort_llm_scanner
from transect.scanners.loaders import reasoning_turns
from transect.spec import Spec, load_spec

__version__ = _version("transect")

__all__ = [
    "Layer",
    "Spec",
    "TransectResults",
    "cohort_llm_scanner",
    "judge_identity",
    "load",
    "load_spec",
    "member_ballots",
    "reasoning_turns",
    "reliability",
    "render",
    "transect",
    "turns_frame",
]
