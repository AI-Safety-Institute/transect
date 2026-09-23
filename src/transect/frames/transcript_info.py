"""transcript_info_df: one row per transcript - run identity, metadata,
and the run's setup facts.

Projected from the `eval_setup` structural scanner's results alone:
Scout stamps ``TranscriptInfo`` (the ``transcript_*`` columns) onto
every scanner's result rows, so identity + run metadata read straight
off that scanner's own rows and the setup facts come from its value -
no cross-scanner join.

Columns (identity prefix explained in common.py):

- model: the transcript's main LLM (``TranscriptInfo.model``).
- success: ``TranscriptInfo.success`` (bool). Note: OpenClaw import
  leaves this None - the importer never computes a score.
- score: the source's own recorded score value (JsonValue as text).
- wallclock_seconds: ``TranscriptInfo.total_time`` - the source's own
  wall-clock duration for the run.
- date: the run date as the source recorded it.
- message_count / total_tokens: run-size facts as recorded.
- error: the run's error text when it errored (None otherwise).
- limit: which limit terminated the run, when one did.
- source_file: the source log's path as recorded.
- source_type: Scout's source kind (eval_log for Inspect .eval files).
- task_name: the eval task/benchmark name when the source recorded one
  (``TranscriptInfo.task_set``); falls back to the source file's stem
  (``TranscriptInfo.source_uri``).
- scaffold_prompt: the agent scaffold's own prompt argument (the
  scaffold's system-prompt contribution, as configured).
- tools: the scaffold's tool roster, comma-joined tool names.
- attempts / submit / compaction / truncation / approval: the
  scaffold's arguments as run, compact display strings; None means
  the scaffold's own default applied (when the args were recorded at
  all - see header_available for the OpenClaw case).
- task_args / generate_config / model_roles: the importer's
  per-sample metadata, compact JSON strings (display cells for the
  intro, not analysis columns - parse the source log for analysis).
- header_available: whether the source log's header was read at scan
  time (False on OpenClaw imports - no header exists there).
- task_file / task_version: the eval code's task provenance.
- message_limit / token_limit / time_limit / working_limit / epochs /
  epochs_reducer / fail_on_error / sandbox: the eval-level config as
  configured; None with header_available=True means "not set", with
  False it means "data not found" (the renderer words the two apart).
- dataset_samples / dataset_shuffled / scorers: dataset shape and
  scorer names from the header.
- system_prompt / task_message: the verbatim initial prompts - the
  first system message and the first user (task) message the model
  actually saw. Deliberately the last columns: they can be multi-KB
  text blobs, and everything else should stay readable when the
  frame is printed.
- compaction_prompt: the recorded Inspect compaction prompt template,
  before substitution; None if unrecorded or not an Inspect eval log.
- schema_version: the frames contract version.

None means the source never recorded the fact; the empty strings
Scout stores for a no-error/no-limit run normalise to None too.
"""

import json
from pathlib import PurePosixPath

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema

_RAW_FIELDS = {
    "model": "transcript_model",
    "success": "transcript_success",
    "score": "transcript_score",
    "wallclock_seconds": "transcript_total_time",
    "date": "transcript_date",
    "message_count": "transcript_message_count",
    "total_tokens": "transcript_total_tokens",
    "error": "transcript_error",
    "limit": "transcript_limit",
    "source_file": "transcript_source_uri",
    "source_type": "transcript_source_type",
}

_HEADER_CONFIG_FIELDS = (
    "message_limit",
    "token_limit",
    "time_limit",
    "working_limit",
    "epochs",
    "epochs_reducer",
    "fail_on_error",
)

_SETUP_COLUMNS = [
    "scaffold_prompt",
    "tools",
    "attempts",
    "submit",
    "compaction",
    "truncation",
    "approval",
    "task_args",
    "generate_config",
    "model_roles",
    "header_available",
    "task_file",
    "task_version",
    *_HEADER_CONFIG_FIELDS,
    "sandbox",
    "dataset_samples",
    "dataset_shuffled",
    "scorers",
    "system_prompt",
    "task_message",
    "compaction_prompt",
]


def transcript_info_df(setup_results: pd.DataFrame) -> pd.DataFrame:
    """The eval_setup scanner's raw results frame -> one row per
    transcript of identity + run metadata + setup facts."""
    rows = []
    for _, r in setup_results.iterrows():
        row = identity(r)
        for ours, theirs in _RAW_FIELDS.items():
            row[ours] = _none_if_na(r.get(theirs))
        task_set = row["task_set"]
        row["task_name"] = (
            task_set
            if isinstance(task_set, str) and task_set
            else _stem(r.get("transcript_source_uri"))
        )
        row.update(_setup_columns(result_value(r.get("value"))))
        rows.append(row)
    columns = [*IDENTITY_COLS, *_RAW_FIELDS, "task_name", *_SETUP_COLUMNS]
    df = pd.DataFrame(rows, columns=columns)
    return with_schema(df)


def _setup_columns(value: dict) -> dict:
    """One eval_setup scanner value, flattened to the frame's setup
    columns."""
    row: dict = {}
    row["system_prompt"] = value.get("system_prompt")
    row["task_message"] = value.get("task_message")
    row["compaction_prompt"] = value["compaction_prompt"]
    agent_args = value.get("agent_args") or {}
    row["scaffold_prompt"] = agent_args.get("prompt")
    row["tools"] = _tool_names(agent_args.get("tools"))
    for field in ("attempts", "submit", "compaction", "truncation", "approval"):
        row[field] = _compact(agent_args.get(field))
    for field in ("task_args", "generate_config", "model_roles"):
        row[field] = _compact(value.get(field))
    header = value.get("header")
    row["header_available"] = header is not None
    header = header or {}
    config = header.get("config") or {}
    row["task_file"] = header.get("task_file")
    row["task_version"] = _compact(header.get("task_version"))
    for field in _HEADER_CONFIG_FIELDS:
        row[field] = _compact(config.get(field))
    row["sandbox"] = header.get("sandbox")
    dataset = header.get("dataset") or {}
    row["dataset_samples"] = dataset.get("samples")
    row["dataset_shuffled"] = dataset.get("shuffled")
    scorers = header.get("scorers")
    row["scorers"] = ", ".join(scorers) if scorers else None
    return row


def _tool_names(tools) -> str | None:
    """The tool roster as comma-joined names; None when unrecorded."""
    if not tools:
        return None
    names = [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]
    return ", ".join(names) if names else None


def _compact(value) -> str | int | float | bool | None:
    """A display cell: scalars pass through, containers become compact
    JSON, an empty container is None (nothing was configured)."""
    if value is None or value == {} or value == []:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, separators=(", ", ": "), default=str)


def _none_if_na(value):
    """Normalize a pandas missing-value cell (``pd.NA``/``NaN``/``None``)
    and Scout's empty-string cells (a no-error/no-limit run) to plain
    Python ``None``."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return None if value == "" else value


def _stem(source_uri) -> str | None:
    """The source file's stem, or None when there is no URI to read."""
    if source_uri is None or pd.isna(source_uri):
        return None
    return PurePosixPath(str(source_uri)).stem
