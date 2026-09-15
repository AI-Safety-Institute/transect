"""OpenClaw ``telemetry-hal`` importer.

Imports the JSONL telemetry written by the ``openclaw-telemetry-hal`` plugin
(https://github.com/sage-princeton/openclaw-telemetry-hal). These are **not**
native OpenClaw session files, which are an entirely different, richer schema.
See the upstream example's ``README.md`` for the schema analysis and known
limitations.

Vendored from inspect_scout's worked example (``examples/sources/
openclaw_telemetry_hal``, meridianlabs-ai/inspect_scout @ 817a8f07 / main,
2026-08-24). Upstream #565 + #566 upstreamed every previous transect adaptation
(cron/explicit session kinds, operator provenance, spawn-less anchor
placement, roll-up aggregate drops, durationMs-derived tool widths), so the
code modules now mirror upstream byte-for-byte. The only local deltas are
this docstring and ``populate_db.py``'s usage docstring (transect module path +
pointer to ``transect.api.run``). Upstream's ``README.md`` and ``tests/`` are not
vendored; transect keeps its own importer tests at
``tests/test_ingestion_selection.py``.
"""

from .client import OPENCLAW_TELEMETRY_HAL_SOURCE_TYPE
from .transcripts import openclaw_telemetry_hal

__all__ = ["openclaw_telemetry_hal", "OPENCLAW_TELEMETRY_HAL_SOURCE_TYPE"]
