"""HTML section builders for the Transect report.

Each function takes the frame slice(s) it needs and returns one
`markupsafe.Markup` fragment rendered from a Jinja partial under
`templates/`; the chart surfaces live in `charts.py`, and `render.py`
composes both. Builders pass raw (unescaped) values into their
templates - autoescape escapes at interpolation time, so never
pre-escape a value here (it would double-escape).
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from markupsafe import Markup

from transect.frames import SCHEMA_VERSION
from transect.frames.transcript_info import _compact
from transect.frames.user import member_ballots
from transect.report import reliability
from transect.report._jinja import jinja_env
from transect.report.charts import _END_MARKER_GLYPH, has_judge_agreement
from transect.report.colors import (
    _UNJUDGED_BASES,
    _UNJUDGED_GREY,
    _label_colors,
    cell_text_color,
    chip_tint,
    health_color,
    ramp_anchors,
    share_color,
)
from transect.report.display import (
    CONFIDENCE_QUALIFIER,
    judge_label,
    member_display,
    multi_roll_models,
    roster,
)
from transect.report.excerpts import Excerpt, card_excerpts
from transect.report.lanes_layout import truncate_lane_name
from transect.scan_status import ModelTokenUsage
from transect.scanners.phases_common import humanise_phase

# Any: jinja TemplateModule macros are dynamic attributes
_notes: Any = jinja_env().get_template("notes.html.j2").module
_phase_cards_tpl: Any = jinja_env().get_template("phase_cards.html.j2").module
_subagent_tpl: Any = jinja_env().get_template("subagent_notes.html.j2").module
_reliability_tpl: Any = jinja_env().get_template("reliability.html.j2").module


def section(
    title: str, blocks: Sequence[Markup | None], anchor: str | None = None
) -> Markup:
    """One report section: an ``<h3>`` header followed by the given
    already-rendered fragments in order; empty/`None` entries are
    dropped, so callers can express optional blocks inline. ``anchor``
    sets an ``id`` for in-page links; ids must stay unique, so a
    per-transcript section takes one on its first occurrence only."""
    return _notes.section(title, [block for block in blocks if block], anchor)


_NOT_FOUND = "not recorded by source"


@dataclass(frozen=True)
class CompactionThreshold:
    """A recorded setting and its token count, when the unit is absolute."""

    label: str
    tokens: int | None


def compaction_threshold(info: pd.DataFrame) -> CompactionThreshold | None:
    """Read the saved Inspect configuration without resolving runtime defaults.

    Inspect distinguishes integer token counts from fractional floats.
    A recorded fraction alone cannot locate a line on a token axis: the
    runtime model capacity is not recorded alongside this setting.
    """
    if not len(info) or info.iloc[0].get("source_type") != "eval_log":
        return None
    raw = info.iloc[0].get("compaction")
    if not isinstance(raw, str):
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(config, dict):
        return None
    threshold = config.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        return None
    if isinstance(threshold, int) or threshold > 1:
        tokens = int(threshold)
        return CompactionThreshold(f"{tokens:,} tokens", tokens)
    return CompactionThreshold(
        f"{threshold * 100:g}% of context window (token count not recorded)", None
    )


def run_intro_line(info: pd.DataFrame) -> Markup | None:
    """The report's very-top introduction: one deterministic sentence
    composed from the run's own recorded facts."""
    if not len(info):
        return None
    row = info.iloc[0]

    def cell(name):
        value = row.get(name)
        return None if value is None or pd.isna(value) else value

    model = cell("model") or "an unrecorded model"
    task = cell("task_name") or "an unrecorded task"
    head = f"{model} ran {task}"
    if cell("sample_id") is not None:
        head += f" (sample {cell('sample_id')}, epoch {cell('epoch')})"
    agent = cell("agent")
    if agent:
        article = "an" if str(agent)[:1].lower() in "aeiou" else "a"
        head += f" under {article} {agent} scaffold"
    date = cell("date")
    if date:
        head += f" on {str(date)[:10]}"
    facts = []
    messages = cell("message_count")
    if messages is not None:
        facts.append(f"{int(messages):,} messages")
    wallclock = cell("wallclock_seconds")
    if wallclock is not None:
        facts.append(_fmt_busy(wallclock))
    tokens = cell("total_tokens")
    if tokens is not None:
        facts.append(f"{int(tokens):,} tokens")
    outcome = []
    if cell("score") is not None:
        outcome.append(f"scored {cell('score')}")
    elif cell("success") is not None:
        outcome.append(f"success: {'true' if cell('success') else 'false'}")
    if cell("error"):
        outcome.append("the run errored")
    if cell("limit"):
        outcome.append(f"ended at the {cell('limit')} limit")
    text = head
    if facts:
        text += "; " + ", ".join(facts)
    if outcome:
        # first letter only - .capitalize() would lowercase recorded
        # values (a score of "C" must not become "c")
        joined = "; ".join(outcome)
        text += ". " + joined[0].upper() + joined[1:]
    return _notes.intro_line(text + ".")


def eval_setup_blocks(info: pd.DataFrame, flushes: pd.DataFrame) -> Markup:
    """The intro's three default-collapsed expandables: Core Setup,
    Additional Config Details, Run Summary. Every row renders even
    when its fact is absent, wording "not recorded by source" (never
    recorded) apart from "not set" (header read, option not
    configured)."""
    irow = info.iloc[0] if len(info) else pd.Series(dtype=object)
    srow = irow
    header_raw = srow.get("header_available")
    header_read = bool(header_raw) if pd.notna(header_raw) else False

    def cell(row, name):
        value = row.get(name)
        if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
            return None
        if isinstance(value, (list, dict)):
            # same display rule as the frame's own _compact fields
            return _compact(value)
        return value

    def found(row, name):
        value = cell(row, name)
        return _NOT_FOUND if value is None else str(value)

    def configured(name):
        value = cell(srow, name)
        if value is not None:
            return str(value)
        return "not set" if header_read else _NOT_FOUND

    # any recorded scaffold argument means the args block exists, so an
    # absent field there is the scaffold's own default, not missing data
    args_recorded = any(
        cell(srow, field) is not None
        for field in (
            "scaffold_prompt",
            "tools",
            "attempts",
            "submit",
            "compaction",
            "truncation",
            "approval",
        )
    )

    def scaffold(name):
        value = cell(srow, name)
        if value is not None:
            return str(value)
        return "scaffold default" if args_recorded else _NOT_FOUND

    threshold = compaction_threshold(info)
    compaction_text = scaffold("compaction")
    if cell(srow, "compaction_prompt") is not None:
        # the template is its own card below; keep the row to the settings
        config = json.loads(compaction_text)
        config.pop("prompt", None)
        compaction_text = (
            f"{json.dumps(config)}; see the compaction prompt card below"
            if config
            else "see the compaction prompt card below"
        )
    core_rows = [
        ("model", found(irow, "model"), None),
        (
            "task",
            found(irow, "task_name")
            + (
                f" · sample {cell(irow, 'sample_id')} · epoch {cell(irow, 'epoch')}"
                if cell(irow, "sample_id") is not None
                else ""
            ),
            None,
        ),
        ("run date", found(irow, "date"), None),
        ("source log", found(irow, "source_file"), None),
        ("agent scaffold", found(irow, "agent"), None),
        (
            "scaffold prompt",
            scaffold("scaffold_prompt"),
            "The scaffold's own prompt argument - its contribution to the "
            "full system message; the verbatim system message the model "
            "actually saw is below.",
        ),
        ("tool roster", scaffold("tools"), None),
        (
            "attempts",
            scaffold("attempts"),
            "How many submission attempts the scaffold allows.",
        ),
        (
            "submit tool",
            scaffold("submit"),
            "The scaffold's answer-submission tool configuration. "
            '"scaffold default" means it ran with the scaffold\'s own: '
            "the log records only configured arguments, not the "
            "resolved default.",
        ),
        (
            "compaction",
            compaction_text,
            "The scaffold's context-compaction setting as configured; "
            "the log does not record the resolved scaffold default.",
        ),
        *(
            [
                (
                    "compaction threshold",
                    threshold.label,
                    "The configured trigger for compacting input context.",
                )
            ]
            if threshold is not None
            else []
        ),
        (
            "truncation",
            scaffold("truncation"),
            "How the scaffold truncates the conversation when the context "
            "window overflows.",
        ),
        ("approval policy", scaffold("approval"), None),
    ]
    prompts = [
        (label, value)
        for label, name in (
            ("System message (verbatim, as the model saw it)", "system_prompt"),
            ("Task message (verbatim, as the model saw it)", "task_message"),
        )
        if (value := cell(srow, name)) is not None
    ]
    if cell(srow, "source_type") == "eval_log":
        template = cell(srow, "compaction_prompt")
        # synthesized drops are detections, not recorded compactions
        recorded = flushes[flushes.source != "synthesized"]
        prompts.extend(
            _compaction_prompts(recorded, str(template) if template else None)
        )
    config_rows = [
        ("task args", found(srow, "task_args"), None),
        (
            "task source file",
            configured("task_file"),
            "The file that defined the eval task, as the source log recorded it.",
        ),
        ("task version", configured("task_version"), None),
        ("message limit", configured("message_limit"), None),
        ("token limit", configured("token_limit"), None),
        ("time limit", configured("time_limit"), None),
        ("working limit", configured("working_limit"), None),
        ("epochs", configured("epochs"), None),
        ("epochs reducer", configured("epochs_reducer"), None),
        ("fail on error", configured("fail_on_error"), None),
        ("sandbox", configured("sandbox"), None),
        ("generation config", found(srow, "generate_config"), None),
        ("model roles", found(srow, "model_roles"), None),
        (
            "dataset",
            _dataset_text(
                cell(srow, "dataset_samples"),
                cell(srow, "dataset_shuffled"),
                header_read,
            ),
            None,
        ),
        ("scorers", configured("scorers"), None),
    ]
    success = cell(irow, "success")
    wallclock = cell(irow, "wallclock_seconds")
    summary_rows = [
        ("messages", found(irow, "message_count"), None),
        (
            "wall-clock time",
            _fmt_busy(wallclock) if wallclock is not None else _NOT_FOUND,
            None,
        ),
        ("total tokens", found(irow, "total_tokens"), None),
        ("score", found(irow, "score"), None),
        (
            "success",
            ("true" if success else "false") if success is not None else _NOT_FOUND,
            None,
        ),
        ("error", cell(irow, "error") or "none recorded", None),
        ("terminating limit", cell(irow, "limit") or "none recorded", None),
    ]
    blocks = [
        {"title": "Core setup", "rows": core_rows, "prompts": prompts},
        {"title": "Additional config details", "rows": config_rows, "prompts": []},
        {"title": "Run summary", "rows": summary_rows, "prompts": []},
    ]
    return _notes.setup_blocks(blocks)


def _compaction_prompts(
    flushes: pd.DataFrame, configured_prompt: str | None
) -> list[tuple[str, str]]:
    """Distinct recorded texts, or explicit absence. A run that recorded no
    compaction gets no absence cards: only the configured template, if any,
    since nothing happened for the text to be missing from."""
    if not len(flushes):
        return (
            [("Compaction prompt (configured template)", configured_prompt)]
            if configured_prompt
            else []
        )
    prompts = []
    for field, label in (
        ("compaction_prompt", "Compaction prompt"),
        ("compaction_nudge", "Compaction nudge (before compaction)"),
    ):
        recorded = [
            text
            for text in flushes.sort_values("turn")[field].dropna().unique()
            if text
        ]
        for text in recorded:
            prompts.append((f"{label} (verbatim)", text))
        if recorded:
            continue
        if field == "compaction_prompt" and configured_prompt:
            prompts.append((f"{label} (configured template)", configured_prompt))
        else:
            prompts.append((label, _NOT_FOUND))
    return prompts


def phase_explanation() -> Markup:
    """The one-sentence description of how phase labels are produced -
    static prose, no data dependency."""
    return _notes.phase_explanation()


def phase_summary_line(
    phases: pd.DataFrame,
    phase_turns: pd.DataFrame,
) -> Markup:
    """The Phase timeline section's one-line judge summary: phase count,
    judge roster, cohort shape, mean per-turn agreement, and the
    verifier model when one was armed."""
    one = phases.sort_values("phase_index")
    label, judge, cohort = _phase_judge_facts(one, phase_turns)
    return _notes.phase_summary_line(
        len(one), label, judge, cohort, _verifier_name(one)
    )


def _verifier_name(entity: pd.DataFrame) -> str | None:
    """The armed verifier's model name (run-constant column), or None
    when the verifier was off."""
    if not len(entity) or not entity.verifier_model.notna().any():
        return None
    return str(entity.verifier_model.dropna().iloc[0])


def phase_chips(
    phases: pd.DataFrame,
    colors: dict[str, tuple[str, str | None]],
    definitions: pd.DataFrame | None = None,
) -> Markup:
    """The band's own legend: the hue-reuse caveat (if any) and the
    per-phase chips, rendered directly under the chart. The
    counts/judge/cohort/verifier detail lives on `phase_meta_line`, next
    to the phase cards.
    """
    one = phases.sort_values("phase_index")
    counts = one.groupby("phase").agg(n=("phase", "size"), turns=("n_turns", "sum"))
    chips = []
    n_reused = 0
    for label, (color, hatch) in colors.items():
        if label not in counts.index:
            continue
        if hatch is not None:
            n_reused += 1
        chips.append(
            {
                "swatch_style": _swatch_style(color, hatch),
                "text": f"{label} ({int(str(counts.at[label, 'n']))} phases / "
                f"{int(str(counts.at[label, 'turns']))} turns)",
            }
        )
    # the reserved buckets (ops + none_of_the_above) chip even at zero:
    # an unused escape hatch is information. Bucket identity comes from
    # the scan's own recorded vocabulary.
    vocab = (
        definitions[definitions.surface == "phases"]
        if definitions is not None and len(definitions)
        else None
    )
    buckets = (
        vocab[vocab.ops.fillna(False) | vocab.reserved.fillna(False)].label.tolist()
        if vocab is not None
        else []
    )
    for label in buckets:
        if label in counts.index:
            continue  # occurred -> already chipped with its own color
        chips.append(
            {
                "swatch_style": _swatch_style(_UNJUDGED_GREY, None),
                "text": f"{label} (0 phases / 0 turns)",
            }
        )
    # overflow labels past the palette slots reuse an earlier hue with a
    # hatch texture - called out so two same-hue hatched chips aren't
    # read as the same label.
    reuse_note = (
        f"{n_reused} label(s) reuse a band hue (the chip texture disambiguates)."
        if n_reused
        else None
    )
    return _notes.phase_chips(reuse_note, chips)


def layer_meta_line(frame: pd.DataFrame | None) -> Markup | None:
    """A custom layer's judge summary line, when it carries the judged columns."""
    if frame is None or not len(frame) or "judge_models" not in frame.columns:
        return None
    judged = frame[frame.judge_models.notna()]
    if not len(judged):
        return None
    raw = str(judged.judge_models.iloc[0])
    unit = "turn" if "turn" in judged.columns else "item"
    text = f"{len(judged)} {unit}(s) judged · {judge_label(raw)}: {roster(raw)}"
    if "verifier_model" in judged.columns:
        verifier = _verifier_name(judged)
        if verifier is not None:
            text += f" · verifier: {verifier}"
    return _notes.intro_line(text)


def label_chips(turn_counts: dict[str, int], colors: dict[str, str]) -> Markup:
    """Legend chips for a custom layer's band: label (n turns), in
    turn-count order, reusing the phase legend's chip macro."""
    chips = [
        {
            "swatch_style": _swatch_style(colors.get(label, _UNJUDGED_GREY), None),
            "text": f"{label} ({count} turns)",
        }
        for label, count in sorted(
            turn_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return _notes.phase_chips(None, chips)


def layer_definitions(
    definitions: pd.DataFrame | None, layer_name: str
) -> Markup | None:
    """A custom layer's recorded label vocabulary as an expandable."""
    if definitions is None or not len(definitions):
        return None
    rows = definitions[definitions.surface == layer_name]
    if not len(rows) or not rows.description.notna().any():
        return None
    entries = [
        {"label": row.label, "description": row.description}
        for row in rows.itertuples()
        if pd.notna(row.description)
    ]
    return _notes.label_definitions("Label definitions", entries)


def agreement_strip_caption(
    phase_turns: pd.DataFrame, phases: pd.DataFrame
) -> Markup | None:
    """The agreement strip's caption under the band chart; with no strip
    data it says why (solo judge, or no turn ended with a vote). `None`
    only when there are no phases."""
    if has_judge_agreement(phase_turns):
        return _notes.agreement_strip_caption()
    if not len(phases):
        return None
    regimes = phases.judge_regime.dropna().astype(str).tolist()
    return _notes.agreement_strip_absent(solo=bool(regimes) and regimes[0] == "solo")


def no_phases_note(status) -> Markup:
    """The Phase timeline section's body when the run has no phases: no
    phase judge was requested, or it ran and produced none."""
    requested = any(s.scanner == "decision_phases" for s in status.scanners)
    return _notes.no_phases_note(requested)


def phase_definitions(definitions: pd.DataFrame | None) -> Markup:
    """Expandable listing of the phase vocabulary (label + description)
    the judge classified against, from the scan's own recorded rubric."""
    return _definitions_expandable(definitions, "phases", "Phase definitions")


def token_intro(derived: bool, coincide: bool = False) -> Markup:
    """Text above the token telemetry chart(s): definitions of whichever
    measures + the linear/log scale the bars chart offers. ``coincide``
    (`charts.token_measures_coincide`) adds the note that the two
    per-turn measures are equal on this transcript.
    """
    if not derived:
        return _notes.token_intro_raw()
    return _notes.token_intro_derived(coincide)


def event_legend(
    has_context_chart: bool, flushes: bool = True, threshold: int | None = None
) -> Markup:
    """Legend for the flush-event glyph and, when drawn, the compaction
    threshold rule (``threshold`` in tokens)."""
    return _notes.event_legend(has_context_chart, flushes, threshold)


def flush_line(flushes: pd.DataFrame) -> Markup:
    """The context-flush list: a ``<details>`` whose summary is the
    count and whose body is one ``<li>`` per flush - position,
    type/source, tokens kept.
    """
    items = []
    for _, f in flushes.sort_values("turn").iterrows():
        amount = None
        if pd.notna(f.tokens_before) and pd.notna(f.tokens_after):
            amount = f"{int(f.tokens_before):,} → {int(f.tokens_after):,}"
            if f.tokens_after_inferred:
                amount += " (inferred)"
        items.append(
            {"turn": int(f.turn), "type": f.type, "source": f.source, "amount": amount}
        )
    return _notes.flush_line(items)


def intervention_legend() -> Markup:
    """The Human interventions chart's own marker legend."""
    return _notes.intervention_legend()


def intervention_line(interventions: pd.DataFrame) -> Markup:
    """The human-intervention list: a ``<details>`` whose summary is the
    count and whose body is one ``<li>`` per intervention with its full
    text - an agent-initiated one as the question asked and the answer
    given, with the recorded outcome.
    """

    def text(value) -> str | None:
        return None if value is None or pd.isna(value) else str(value)

    items = []
    for _, i in interventions.sort_values("turn", kind="stable").iterrows():
        initiator = text(i.initiator)
        items.append(
            {
                "turn": int(i.turn),
                "channel": i.channel,
                "content": text(i.content) or "",
                "prompt": text(i.prompt),
                "outcome": text(i.outcome),
                # a store written before initiator existed: read it off
                # the channel, so an old input_event row is not a "message"
                "agent": initiator == "agent"
                if initiator
                else i.channel in ("input_event", "approval"),
            }
        )
    return _notes.intervention_line(items)


def subagent_explanation() -> Markup:
    """The Sub-agent activity section's one-sentence description of how
    its labels are produced."""
    return _notes.subagent_explanation()


def subagent_notes(
    lanes: list[tuple],
    label_of: dict,
    subagents: pd.DataFrame,
    classification_ran: bool,
    has_boxes: bool,
    has_end_markers: bool,
) -> Markup:
    """The Sub-agent activity section's prose, rendered above its chart
    (the label legend renders below it - `subagent_legend`).

    ``lanes`` / ``label_of`` are the same span-grouping the orchestrator
    built for the chart; ``has_boxes`` says whether any span draws as a
    wall-clock box (the how-to-read line), ``has_end_markers`` whether
    any span's recorded end draws the completion glyph.
    """
    # three states, honestly distinguished: no classification at all
    # (grey note, and NO label vocabulary anywhere; classification joined
    # (labels, no note); and classification present but zero lanes joined
    # - a silent-failure smell (identity/span-id mismatch).
    any_joined = any(entry[0] in label_of for entry in lanes)
    if not classification_ran:
        state = "not_run"
    elif not any_joined:
        state = "warn"
    else:
        state = "ok"

    # one summary line under the section explanation
    summary = None
    judged_rows = subagents[subagents.judge_models.notna()]
    if classification_ran and len(judged_rows):
        joined = str(judged_rows.judge_models.iloc[0])
        cohort_part = None
        voted_rows = subagents[subagents.judge_agreement.notna()]
        if len(voted_rows):
            # regime facts come off the scanner-stamped judge columns
            first = voted_rows.iloc[0]
            n_models = int(first.n_models)
            regime = (
                f"{n_models} model(s)"
                if n_models > 1
                else f"1 model(s) × {int(first.k_rolls)} roll(s)"
            )
            cohort_part = {
                "regime": regime,
                "mean": f"{voted_rows.judge_agreement.mean():.2f}",
                "n_spans": len(voted_rows),
            }
        summary = {
            "n_spans": len(lanes),
            "judge_label": judge_label(joined),
            "judges": roster(joined),
            "cohort": cohort_part,
            "verifier": _verifier_name(subagents),
        }

    return _subagent_tpl.subagent_notes(
        state, summary, has_boxes, has_end_markers, _END_MARKER_GLYPH
    )


def subagent_legend(
    by_label: dict[str, list],
    colors: dict[str, str],
    classification_ran: bool,
) -> Markup | None:
    """The swimlanes' label legend, below the chart. No classifier run
    means no label vocabulary to chip."""
    if not classification_ran or not by_label:
        return None
    chips = [
        {
            "swatch_style": _swatch_style(colors[label], None),
            "text": f"{label} ({len(group)})",
        }
        for label, group in sorted(by_label.items())
    ]
    return _notes.phase_chips(None, chips)


def span_titles(subagents: pd.DataFrame, label_of: dict) -> dict:
    """One hover-tooltip cell set per span for the swimlanes chart:
    ``{span_id: {field: cell}}`` keyed by `charts.SPAN_TIP_FIELDS`.
    "no data" marks a value the source never recorded. The ``turns``
    cell says what the box means: the orchestrator turns active while
    the span ran (wall-clock), or only its spawn turn when the source
    recorded no usable timestamps; a span that outlived the last
    orchestrator turn says so."""

    def fmt(value) -> str:
        return "no data" if value is None or pd.isna(value) else f"{int(value):,}"

    titles = {}
    for row in subagents.itertuples():
        if str(row.position_source) == "timestamp":
            turns = (
                f"{int(str(row.anchor_turn))}–{int(str(row.end_turn))} "
                "(orchestrator turns active while it ran)"
            )
            if bool(row.after_last):
                turns += ", continued after the last orchestrator turn"
        else:
            turns = f"{int(str(row.spawn_turn))} (spawn turn; no timestamps)"
        titles[row.agent_span_id] = {
            "lane": truncate_lane_name(str(row.agent_lane)),
            "turns": turns,
            # "classification", not "label" - a tooltip channel named
            # `label` blanks the chart (charts.SPAN_TIP_FIELDS)
            "classification": _resolved_label(label_of, row.agent_span_id),
            # reliability cells mirror the agreement strip's: None (no
            # row) where the fact does not apply to this span
            "confidence": _span_confidence(row),
            "agreement": _span_agreement(row),
            "label source": (
                str(row.label_source) if pd.notna(row.label_source) else None
            ),
            "verifier": _span_verifier_cell(row),
            "tool calls": fmt(row.tool_calls),
            "busy": _fmt_busy(row.busy_seconds),
        }
    return titles


def _span_agreement(row) -> str | None:
    """The span tooltip's agreement cell, denominators stated."""
    if pd.isna(row.judge_agreement):
        return None
    text = f"{float(str(row.judge_agreement)):.2f}"
    if pd.notna(row.n_voting) and pd.notna(row.n_members):
        text += f" ({int(str(row.n_voting))} of {int(str(row.n_members))} voted)"
    return text


def subagent_definitions(definitions: pd.DataFrame | None) -> Markup:
    """The sub-agent classifier's own rubric."""
    return _definitions_expandable(
        definitions, "subagents", "Sub-agent label definitions"
    )


def subagent_reliability_flags(
    subagents: pd.DataFrame, subagent_votes: pd.DataFrame
) -> list[reliability.Flag]:
    """Any red/amber reliability flag for the Sub-agents entity, for
    inline surfacing in the Sub-agent activity section.

    Empty on a scan whose verifier examined no sub-agent spans.
    """
    _, flags = _entity_audit(
        "Sub-agents", subagents, subagent_votes, "agent_span_id", "label", subagents
    )
    return flags


def reliability_warnings(flags: list[reliability.Flag]) -> Markup | None:
    """An entity's flags inline in its own section, via the audit's
    `flag_line` macro so the two surfaces cannot drift. ``None`` when
    there are none."""
    if not flags:
        return None
    return Markup("".join(_reliability_tpl.flag_line(flag) for flag in flags))


def phase_reliability_flags(
    phases: pd.DataFrame,
    phase_turn_votes: pd.DataFrame,
    phase_turns: pd.DataFrame,
) -> list[reliability.Flag]:
    """Any red/amber reliability flag for the Phases entity, for inline
    surfacing in the Phase timeline section - the same `_entity_audit`
    call the audit section's "Phases" block makes, mirroring
    `subagent_reliability_flags`."""
    _, flags = _entity_audit(
        "Phases", phases, phase_turn_votes, "turn", "phase", phase_turns
    )
    return flags


def spawn_prompts_block(rows: list[dict]) -> Markup | None:
    """The spawn-prompts expandable, turn-ordered. ``rows`` are plain
    dicts built by ``render._subagent_section``; this only renders."""
    if not rows:
        return None
    return _subagent_tpl.spawn_prompts(rows)


def spend_data(
    one: pd.DataFrame,
    phases: pd.DataFrame,
    phase_colors: dict[str, tuple[str, str | None]],
    subagents: pd.DataFrame,
) -> tuple[list[dict], list[dict], bool] | None:
    """Token-spend bars from the frames' own declared rollups:
    per-phase from ``phases.new_work_tokens`` (dense-attributed, plus an
    "unjudged" bucket for whatever the dense map assigns to no phase),
    per-sub-agent-label from the merged ``subagents`` frame. Returns
    None when there is nothing to show; the third element flags the
    tool-events-only case (spans exist, no usage recorded) so the
    caller can say so instead of drawing nothing."""
    if not len(phases):
        return None
    total = one.new_work.dropna().sum()
    by_phase = phases.dropna(subset=["new_work_tokens"])
    phase_bars = []
    if len(by_phase):
        spend = (
            by_phase.groupby("phase", observed=True)
            .new_work_tokens.sum()
            .sort_values(ascending=False)
        )
        spend = spend[spend > 0]
        unjudged = int(total - spend.sum())
        if unjudged > 0:
            spend = pd.concat([spend, pd.Series({"unjudged": unjudged})])
            spend = spend.sort_values(ascending=False)
        total_spend = float(spend.sum())
        for bucket, value in spend.items():
            color = (
                _UNJUDGED_GREY
                if bucket == "unjudged"
                else phase_colors.get(str(bucket), (_UNJUDGED_GREY, None))[0]
            )
            phase_bars.append(
                {
                    "label": str(bucket),
                    "value": int(value),
                    "color": color,
                    "share": f"{value / total_spend:.0%} of new work",
                }
            )
    subagent_bars = []
    subagent_message = False
    classified = subagents[subagents.label.notna()] if len(subagents) else subagents
    if len(classified) and classified.new_work.notna().any():
        rollups = classified.groupby("label", observed=True).agg(
            new_work=("new_work", lambda v: v.sum(min_count=1)),
            output=("output_tokens", lambda v: v.sum(min_count=1)),
            billable=("billable", lambda v: v.sum(min_count=1)),
        )
        rollups = rollups.dropna(subset=["new_work"])
        rollups = rollups[rollups.new_work > 0].sort_values("new_work", ascending=False)
        if len(rollups):
            colors = _label_colors(sorted(rollups.index))
            # output/billable ride this chart's tooltip (the span
            # tooltips carry no token rollups - one home per fact)
            subagent_bars = [
                {
                    "label": str(cat),
                    "value": int(r.new_work),
                    "color": colors[str(cat)],
                    "share": None,
                    "output": None if pd.isna(r.output) else f"{int(r.output):,}",
                    "billable": (
                        None if pd.isna(r.billable) else f"{int(r.billable):,}"
                    ),
                }
                for cat, r in rollups.iterrows()
            ]
    elif len(classified):
        subagent_message = True
    if not phase_bars and not subagent_bars and not subagent_message:
        return None
    return phase_bars, subagent_bars, subagent_message


def tag_spend_bars(
    one: pd.DataFrame, turn_tags: pd.DataFrame | None, family: str
) -> list[dict]:
    """Token-spend bars grouped by one turn-tag family: per-turn
    ``new_work`` joined to the family's labels, summed per label."""
    if turn_tags is None or family not in turn_tags.columns or not len(one):
        return []
    per_turn = one.dropna(subset=["new_work"])[["turn", "new_work"]]
    tagged = turn_tags.dropna(subset=[family])[["turn", family]]
    if not len(per_turn) or not len(tagged):
        return []
    joined = per_turn.merge(tagged, on="turn")
    spend = (
        joined.groupby(family, observed=True)
        .new_work.sum()
        .sort_values(ascending=False)
    )
    spend = spend[spend > 0]
    if not len(spend):
        return []
    total = float(one.new_work.dropna().sum())
    untagged = int(total - spend.sum())
    if untagged > 0:
        spend = pd.concat([spend, pd.Series({"untagged": untagged})])
        spend = spend.sort_values(ascending=False)
    total_spend = float(spend.sum())
    colors = _label_colors(sorted(str(b) for b in spend.index if b != "untagged"))
    return [
        {
            "label": str(bucket),
            "value": int(value),
            "color": (
                _UNJUDGED_GREY
                if bucket == "untagged"
                else colors.get(str(bucket), _UNJUDGED_GREY)
            ),
            "share": f"{value / total_spend:.0%} of new work",
        }
        for bucket, value in spend.items()
    ]


def spend_family_control(scope: str, families: list[tuple[str, str]]) -> Markup:
    """The custom grouping's heading + family selector."""
    options = [{"family": family, "layer": layer} for family, layer in families]
    return _notes.spend_family_control(scope, options)


def spend_family_group(scope: str, family: str, chart: Markup, hidden: bool) -> Markup:
    """One family's chart wrapped for the selector's show/hide."""
    return _notes.spend_family_group(scope, family, chart, hidden)


def spend_group_heading(text: str) -> Markup:
    """One spend group's own sub-heading ("By phase label" / "By
    sub-agent label") - outside the chart iframe, so the two groups'
    charts stay independently-sized documents."""
    return _notes.group_heading(text)


def spend_subagent_unavailable() -> Markup:
    """The sub-agent half's honesty note when delegated spend cannot be
    computed (tool-events-only lanes carry no usage)."""
    return _notes.spend_subagent_unavailable()


def phase_cards_explanation() -> Markup:
    """One-liner under the "Phase cards" section header, saying what the
    section is and how it connects back to the timeline. Static, no data
    dependency."""
    return _notes.phase_cards_explanation()


def phase_meta_line(
    phases: pd.DataFrame,
    phase_turns: pd.DataFrame,
) -> Markup:
    """Phase-cards meta line: counts, judge attribution, and the
    cohort/verifier/unjudged notes."""
    one = phases.sort_values("phase_index")
    unjudged = (
        phase_turns[phase_turns.basis.isin(_UNJUDGED_BASES)]
        if len(phase_turns)
        else phase_turns
    )
    n_unjudged = len(unjudged) if len(phase_turns) else 0
    label, judge, cohort = _phase_judge_facts(one, phase_turns)

    return _notes.phase_meta_line(
        len(one),
        label,
        judge,
        cohort,
        _verifier_notes(one),
        n_unjudged,
    )


def narrator_line(phases: pd.DataFrame) -> Markup | None:
    """Model attribution for the cards' LLM-generated headlines and
    summaries."""
    if not phases.narrator_model.notna().any():
        return None
    return _notes.narrator_line(str(phases.narrator_model.dropna().iloc[0]))


def phase_card_controls(
    container_id: str,
    phases: pd.DataFrame,
    colors: dict[str, tuple[str, str | None]],
    turn_tags: pd.DataFrame | None = None,
    tag_layer_of: dict[str, str] | None = None,
) -> Markup:
    """The sort/filter bar above one transcript's phase cards.
    ``container_id`` matches `phase_cards`' wrapper; the JS handlers in
    `templates/report.html.j2` scope every lookup to it, so several
    transcripts' controls on one page cannot cross-filter. Label
    options follow ``colors``' order, agreeing with the band's legend
    chips.
    """
    present = set(phases.phase.unique()) if len(phases) else set()
    labels = [
        {
            "label": label,
            "swatch_style": _swatch_style(colors[label][0], colors[label][1]),
        }
        for label in colors
        if label in present
    ]
    tag_filters = [
        {
            "family": family,
            "options": [str(v) for v in sorted(turn_tags[family].dropna().unique())],
            "layer": (tag_layer_of or {}).get(family, ""),
        }
        for family in _tag_families(turn_tags)
        if turn_tags is not None and turn_tags[family].notna().any()
    ]
    return _phase_cards_tpl.phase_card_controls(container_id, labels, tag_filters)


def phase_cards(
    phases: pd.DataFrame,
    turn_groups: pd.DataFrame,
    turn_votes: pd.DataFrame,
    one: pd.DataFrame,
    flush_turns: list | None,
    intervention_turns: list | None,
    colors: dict[str, tuple[str, str | None]],
    link,
    container_id: str,
    card_id_prefix: str,
    excerpts: dict[int, Excerpt] | None = None,
    tool_counts: dict[int, int] | None = None,
    turn_tags: pd.DataFrame | None = None,
    tag_layer_of: dict[str, str] | None = None,
) -> Markup:
    """Phase cards: one expandable card per phase, chronological (the
    drill-down under the Phase timeline band).

    The spend tag is the phase's orchestrator new-work
    (``phases.new_work_tokens``), and the label says so; delegated spend
    has its own frame column and chart. Each card's ``data-*``
    attributes (documented in `templates/phase_cards.html.j2`) are
    read by `phase_card_controls`' sort/filter JS and reuse the values
    the visible tag line computes, so the two cannot disagree.
    ``card_id_prefix`` namespaces the card ids to match this
    transcript's `charts.phase_band` anchors; ``container_id`` is the
    filter bar's separate scope. ``excerpts`` (this transcript's
    ``{turn: Excerpt}`` map) turns each turn group into a dropdown of
    message text; ``None`` leaves plain list items, with the Scout
    deep link as the full-text path either way.
    """

    sorted_phases = phases.sort_values("phase_index").reset_index(drop=True)
    after_flush_flags = _card_event_flags(sorted_phases, flush_turns)
    after_intervention_flags = _card_event_flags(sorted_phases, intervention_turns)
    cards = []
    for pos, (_, p) in enumerate(sorted_phases.iterrows()):
        color, _hatch = colors.get(p.phase, (_UNJUDGED_GREY, None))
        # the frames' dense-attributed orchestrator rollup - the same
        # number the band and the spend chart report (declared-range sums
        # drop tool-only turns outside the judged ranges)
        spend = p.new_work_tokens if pd.notna(p.new_work_tokens) else 0
        tags = [
            f"turns {int(p.turn_start)}–{int(p.turn_end)}",
            # n_turns counts the phase's reasoning-bearing (digest) turns,
            # which can be fewer than the range width (tool-only turns)
            f"{int(p.n_turns)} reasoning turn(s)",
            f"{int(spend):,} orchestrator new-work tokens",
        ]
        n_tools = (
            sum(
                tool_counts.get(turn, 0)
                for turn in range(int(p.turn_start), int(p.turn_end) + 1)
            )
            if tool_counts
            else None
        )
        if n_tools is not None:
            tags.append(f"{n_tools} tool call(s)")
        # spans whose spawn turn lies in the phase (frames.phases)
        n_subagents = int(p.n_subagents) if pd.notna(p.n_subagents) else 0
        if n_subagents:
            tags.append(f"{n_subagents} sub-agent(s) spawned")
        if any(p.turn_start <= t <= p.turn_end for t in flush_turns or []):
            tags.append("compaction during phase")
        if any(p.turn_start <= t <= p.turn_end for t in intervention_turns or []):
            tags.append("human intervention during phase")
        agreement = p.get("judge_agreement")
        if agreement is not None and pd.notna(agreement):
            tags.append(f"mean agreement {agreement}")
        # class-box flag conditions: verifier overturn, low mean
        # confidence, or confidently-split votes. Min per-turn
        # agreement deliberately not flagged - it fires on most
        # cohort cards.
        issues = []
        if bool(p.get("overturned")):
            original = p.get("original_label")
            original_conf = p.get("original_confidence")
            if original is not None and pd.notna(original):
                was = f"was {original}"
                if original_conf is not None and pd.notna(original_conf):
                    was += f" ({float(original_conf):.2f})"
                issues.append(f"verifier overturned ({was})")
            else:
                issues.append("verifier overturned")
        raw_confidence = p.get("confidence")
        conf = (
            float(raw_confidence)
            if raw_confidence is not None and pd.notna(raw_confidence)
            else None
        )
        if conf is not None and conf <= 0.6:
            issues.append(f"low mean confidence ({conf:.2f})")
        agree = (
            float(agreement) if agreement is not None and pd.notna(agreement) else None
        )
        if conf is not None and agree is not None and conf >= 0.8 and agree <= 0.5:
            issues.append(
                f"high confidence but split votes "
                f"(mean conf {conf:.2f}, mean agreement {agree:.2f})"
            )
        # narrate=False (or a failed narrative) leaves headline empty:
        # fall back to the humanised phase label, never a bare "Phase N ·"
        headline = str(p.get("headline") or "").strip() or humanise_phase(p.phase)
        summary = str(p.get("summary") or "")
        groups = (
            turn_groups[turn_groups.phase_index == p.phase_index].sort_values(
                "group_index"
            )
            if len(turn_groups)
            else turn_groups
        )
        group_items = [
            {
                "turn_start": int(g.turn_start),
                "turn_end": int(g.turn_end),
                "title": str(g.title),
                "gist": g.gist,
                "excerpts": [],
                "n_more": 0,
            }
            for _, g in groups.iterrows()
        ]
        # one excerpt budget per card, split across its groups; one
        # pair per group even with no excerpts, so the zip stays strict
        selected = card_excerpts(
            excerpts or {},
            [(item["turn_start"], item["turn_end"]) for item in group_items],
        )
        for item, (shown, n_more) in zip(group_items, selected, strict=True):
            item["excerpts"] = shown
            item["n_more"] = n_more
        open_link = link(p.get("anchor_event_id")) if link is not None else None
        member_items = _phase_member_items(turn_votes, p)
        user_chips, tag_key = _card_tags(
            turn_tags, int(p.turn_start), int(p.turn_end), tag_layer_of or {}
        )
        cards.append(
            {
                "phase_index": int(p.phase_index),
                "card_id": f"{card_id_prefix}-{int(p.phase_index)}",
                "color": color,
                "headline": headline,
                "group_note": {
                    "invalid_partition": (
                        "Shown as one turn group: the narrator's groups did "
                        "not line up with the phase's turns."
                    ),
                    "empty_groups": (
                        "Shown as one turn group: the narrator did not split "
                        "this phase into turn groups."
                    ),
                    "no_narrative": (
                        "Shown as one turn group: no narration was available "
                        "for this phase."
                    ),
                }.get(p.narration_group_status),
                "tags": tags,
                "summary": summary,
                "groups": group_items,
                "members": member_items,
                "open_link": open_link,
                "label": str(p.phase),
                "spend": int(spend),
                "tools": n_tools if n_tools is not None else "",
                "n_subagents": n_subagents,
                "confidence": (float(p.confidence) if pd.notna(p.confidence) else ""),
                "issues": issues,
                "after_flush": bool(after_flush_flags[pos]),
                "after_intervention": bool(after_intervention_flags[pos]),
                "user_chips": user_chips,
                "tag_key": tag_key,
            }
        )
    return _phase_cards_tpl.phase_cards(container_id, cards)


def reliability_audit(
    phases: pd.DataFrame,
    phase_turns: pd.DataFrame,
    subagents: pd.DataFrame,
    subagent_votes: pd.DataFrame,
    label_definitions: pd.DataFrame | None,
    phase_turn_votes: pd.DataFrame,
    layer_audits: list[dict] | None = None,
) -> Markup:
    """The "Reliability & provenance audit" section - the report's last,
    always rendered. One block per judged surface; ``layer_audits``
    adds one badge-marked block per audit-declaring custom layer
    (``{name, frame, unit_col, label_col}``, the frame already cut to
    this transcript).

    Scoped to one transcript.
    """

    def declared(surface: str) -> list[str] | None:
        if label_definitions is None or not len(label_definitions):
            return None
        rows = label_definitions[label_definitions.surface == surface]
        return list(rows.label) if len(rows) else None

    custom_blocks = []
    for audit in layer_audits or []:
        frame: pd.DataFrame | None = audit["frame"]
        if frame is None or not len(frame):
            continue
        if "judge_regime" not in frame.columns or not frame.judge_regime.notna().any():
            continue  # the layer's judge never ran: nothing to audit
        unit_col, label_col = audit["unit_col"], audit["label_col"]
        unit_word = unit_col.replace("_", " ")
        # an arbitrary user frame may not carry the columns the audit
        # reads; the report renders the failure instead of crashing
        try:
            decided = frame[frame[label_col].notna()]
            extra_rows = [
                {
                    "label": "Judged units",
                    "value": f"{len(decided)} {unit_word}(s) of {len(frame)}",
                    "definition": "This layer's frame rows carrying a decided "
                    "label, of all its rows for this transcript.",
                }
            ]
            extra_rows += _span_verifier_rows(frame, unit_word=unit_word)
            block, block_flags = _entity_audit(
                audit["name"],
                frame,
                member_ballots(frame, unit_col, label_col),
                unit_col,
                label_col,
                frame,
                extra_rows=extra_rows,
                vocabulary=declared(audit["name"]),
                unit_word=unit_word,
                badge=True,
            )
        except Exception as error:
            block, block_flags = _unauditable_layer_block(audit["name"], error), []
        custom_blocks.append((audit["name"], block, block_flags))

    phases_ran = bool(len(phases))
    subagents_ran = bool(len(subagents)) and bool(subagents.status.notna().any())
    if not phases_ran and not subagents_ran and not custom_blocks:
        return _notes.intro_line(
            "No judged surfaces in this scan: the mechanical frames carry "
            "no classifications to audit. Pass judge_models to classify "
            "phases and sub-agents."
        )

    blocks = []
    flag_groups = []
    if phases_ran:
        total_turns = len(phase_turns)
        judged_turns = int((phase_turns.basis == "judged").sum())
        unjudged_turns = sum(reliability.abstention_counts(phase_turns).values())
        other_turns = total_turns - judged_turns - unjudged_turns
        phase_extra_rows = [
            {
                "label": "Turns",
                "value": (
                    f"{total_turns} total · {judged_turns} judged · "
                    f"{other_turns} filled/attributed · {unjudged_turns} unjudged"
                ),
                "definition": "Model turns the decision-phases judge covered, "
                "split by how each turn's label was decided: judged (labelled "
                "directly); filled (a reasoning turn no judge answer covered, "
                "inheriting the previous label at low confidence); attributed "
                "(content-free tool-call-only or failed turns the judge never "
                "saw, taking the phase whose range contains them); or "
                "unjudged (refusal / no_answer / missing_turn).",
            },
            {
                "label": "Decision phases",
                "value": f"{len(phases)} phase(s)",
                "definition": "Stitched decision-phase count for this transcript.",
            },
        ]
        phase_extra_rows += _span_verifier_rows(phases, unit_word="phase")
        phases_block, phases_flags = _entity_audit(
            "Phases",
            phases,
            phase_turn_votes,
            "turn",
            "phase",
            phase_turns,
            extra_rows=phase_extra_rows,
            vocabulary=declared("phases"),
        )
        blocks.append(phases_block)
        flag_groups.append(
            {"title": "Phase segmentation and labelling", "flags": phases_flags}
        )
    if subagents_ran:
        subagents_extra_rows = [
            {
                "label": "Sub-agent spans",
                "value": f"{len(subagents)} span(s)",
                "definition": "Sub-agent span count for this transcript (every "
                "observed span; the classification columns are simply empty "
                "for spans no judge labelled).",
            },
        ]
        subagents_extra_rows += _span_verifier_rows(subagents)
        subagents_block, subagents_flags = _entity_audit(
            "Sub-agents",
            subagents,
            subagent_votes,
            "agent_span_id",
            "label",
            subagents,
            extra_rows=subagents_extra_rows,
            vocabulary=declared("subagents"),
        )
        blocks.append(subagents_block)
        flag_groups.append({"title": "Sub-agent labelling", "flags": subagents_flags})
    for name, block, block_flags in custom_blocks:
        blocks.append(block)
        flag_groups.append({"title": f"{name} labelling", "flags": block_flags})
    low, mid, high = ramp_anchors()
    return _reliability_tpl.reliability_audit(
        blocks, flag_groups, SCHEMA_VERSION, {"low": low, "mid": mid, "high": high}
    )


def scan_status_view(status) -> dict:
    """The run-wide execution block: per-scanner completed-of-scope
    counts (audit-red on failure states, plain otherwise), the
    recorded error list, the stored-but-unmounted and never-attempted
    scanner names, and the billed model usage per scanner and model
    followed by the run total (only when more than one scanner
    billed)."""
    rows = []
    unattempted = []
    usage = [
        _usage_row(scanner.scanner, u, i, len(scanner.model_usage))
        for scanner in status.scanners
        for i, u in enumerate(scanner.model_usage)
    ]
    if sum(bool(scanner.model_usage) for scanner in status.scanners) > 1:
        total = status.model_usage
        usage.extend(
            _usage_row("all scanners", u, i, len(total), total=True)
            for i, u in enumerate(total)
        )
    for scanner in status.scanners:
        completed = scanner.completed_transcripts
        # the denominator is the scan's scope; a store that records no
        # scope falls back to the attempts actually made
        total = scanner.total_transcripts
        of = total if total is not None else scanner.scanned_transcripts
        if scanner.never_attempted:
            unattempted.append(scanner.scanner)
        rows.append(
            {
                "name": scanner.scanner,
                "sub": None if scanner.mounted else "not mounted",
                "completed": _status_cell(
                    f"{completed} of {of}", bad=completed < of or of == 0
                ),
                "errors": _status_count(scanner.errors),
            }
        )
    execution = {
        None: "completion not recorded",
        True: "finished",
        False: "incomplete",
    }[status.outer_complete]
    # store-integrity entries are the only scanner-level (transcript-less)
    # errors: the execution happened, but its results are not all here
    if any(error.transcript_id is None for error in status.errors):
        execution += "; results incomplete"
    return {
        "failures": status.has_failures,
        "execution": execution,
        "rows": rows,
        "usage": usage,
        "unattempted": unattempted,
        "unmounted": [s.scanner for s in status.scanners if not s.mounted],
        "errors": [
            {
                "scanner": e.scanner,
                "transcript_id": e.transcript_id,
                # Scout stores the traceback separately; this is the
                # message alone, clipped against pathological payloads.
                "message": e.message
                if len(e.message) <= 300
                else e.message[:300] + "…",
            }
            for e in status.errors
        ],
    }


def _fmt_busy(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "no data"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def _swatch_style(color: str, hatch: str | None) -> str:
    if hatch is None:
        return f"background:{color};"
    return (
        f"background:repeating-linear-gradient(45deg,{color},"
        f"{color} 3px,white 3px,white 5px);"
    )


def _verifier_notes(phases: pd.DataFrame) -> dict | None:
    """Verifier tally for the phase meta line. "With a verdict", not
    "checked": a merged phase may span more turns than were reviewed."""
    units = reliability.review_units(phases)
    if not len(units):
        return None
    completed = units.verifier_completed.fillna(False).astype(bool)
    same = phases.verifier_same_model.dropna()
    return {
        "verified": int(completed.sum()),
        "selected": len(units),
        "overturned": int(units.loc[completed, "overturned"].fillna(False).sum()),
        "same_model": bool(same.iloc[0]) if len(same) else False,
    }


def _phase_judge_facts(
    one: pd.DataFrame,
    phase_turns: pd.DataFrame,
) -> tuple[str, str, dict | None]:
    """``(judge_label, judge_roster, cohort)`` shared by the timeline's
    summary line and the phase cards' fuller meta line, so the two can
    never disagree about the same judges."""
    raw = str(one.judge_models.iloc[0])
    agreements = phase_turns.dropna(subset=["judge_agreement"])
    cohort = None
    if len(agreements):
        n_models = int(one.n_models.iloc[0])
        k_rolls = int(one.k_rolls.iloc[0])
        detail = (
            f"{n_models} model(s)"
            if n_models > 1
            else f"1 model(s) × {k_rolls} roll(s)"
        )
        cohort = {
            "detail": detail,
            "mean": f"{agreements.judge_agreement.mean():.2f}",
            # turns with >= 2 voters: exactly the rows carrying a
            # per-turn agreement (lone-voter turns report None)
            "n_turns": len(agreements),
        }
    return judge_label(raw), roster(raw), cohort


def _definitions_expandable(
    definitions: pd.DataFrame | None, surface: str, title: str
) -> Markup:
    rows = (
        definitions[definitions.surface == surface]
        if definitions is not None and len(definitions)
        else None
    )
    if rows is None or not len(rows):
        return _notes.definitions_unavailable(title)
    if not rows.description.notna().any():
        return _notes.definitions_unavailable(title)
    entries = [
        {"label": row.label, "description": row.description}
        for row in rows.itertuples()
        if pd.notna(row.description)
    ]
    return _notes.label_definitions(title, entries)


def _tag_families(turn_tags: pd.DataFrame | None) -> list[str]:
    """The family columns of the wide turn_tags mount."""
    if turn_tags is None or not len(turn_tags):
        return []
    return [c for c in turn_tags.columns if c not in ("transcript_id", "turn")]


def _card_tags(
    turn_tags: pd.DataFrame | None,
    turn_start: int,
    turn_end: int,
    layer_of: dict[str, str],
) -> tuple[list[dict], str]:
    """One card's user tag chips: each family's distinct values inside
    the card's turn range as ``family=value`` texts, tinted per family."""
    chips = []
    for position, family in enumerate(_tag_families(turn_tags)):
        assert turn_tags is not None
        rows = turn_tags[(turn_tags.turn >= turn_start) & (turn_tags.turn <= turn_end)]
        for value in sorted(rows[family].dropna().unique()):
            chips.append(
                {
                    "text": f"{family}={value}",
                    "layer": layer_of.get(family, ""),
                    "style": chip_tint(position),
                }
            )
    if not chips:
        return [], ""
    return chips, "|" + "|".join(c["text"] for c in chips) + "|"


def _card_event_flags(
    phases_sorted: pd.DataFrame, event_turns: list | None
) -> list[bool]:
    """The "at/after a context flush/human intervention" filter flags,
    one entry per row of ``phases_sorted`` (phase_index order) - one
    implementation for both event kinds. Each event flags the phase
    containing it (or, in a gap, the nearest phase starting at/before
    it) plus the one immediately following.
    """
    starts = phases_sorted.turn_start.astype(int).tolist()
    ends = phases_sorted.turn_end.astype(int).tolist()
    n = len(starts)
    flagged = [False] * n
    for t in event_turns or []:
        t = int(t)
        j = next((k for k in range(n) if starts[k] <= t <= ends[k]), None)
        if j is None:
            candidates = [k for k in range(n) if starts[k] <= t]
            j = candidates[-1] if candidates else None
        if j is None:
            continue
        flagged[j] = True
        if j + 1 < n:
            flagged[j + 1] = True
    return flagged


def _phase_member_items(turn_votes: pd.DataFrame, phase) -> list[dict]:
    """One display row per cohort member for a phase card: modal label
    over the phase's range (count, then mean confidence, then earliest),
    support (share of the member's judged turns matching the consensus),
    and mean stated confidence - computed from the raw per-turn votes.
    Empty outside the voting regimes."""
    if not len(turn_votes):
        return []
    in_range = turn_votes[
        (turn_votes.turn >= phase.turn_start)
        & (turn_votes.turn <= phase.turn_end)
        & (turn_votes.basis == "judged")
        & turn_votes.phase.notna()
    ]
    if not len(in_range):
        return []
    multi = multi_roll_models(in_range)
    items = []
    for (model, roll), sub in in_range.groupby(["model", "roll"], sort=True):
        sub = sub.reset_index(drop=True)
        ranked: dict[str, tuple[int, float, int]] = {}
        for position, row in sub.iterrows():
            label = str(row.phase)
            count, conf_sum, first = ranked.get(label, (0, 0.0, position))
            ranked[label] = (
                count + 1,
                conf_sum + (row.confidence if pd.notna(row.confidence) else 0.0),
                min(first, position),
            )

        def rank(label: str, ranked=ranked) -> tuple[int, float, int]:
            count, conf_sum, first = ranked[label]
            return (count, conf_sum / count, -first)

        modal = max(ranked, key=rank)
        matching = int((sub.phase.astype(str) == str(phase.phase)).sum())
        confidences = sub.confidence.dropna()
        items.append(
            {
                "member": member_display(model, roll, multi, short=False),
                "phase": modal,
                "support": round(matching / len(sub), 3),
                "mean_confidence": (
                    round(float(confidences.mean()), 3) if len(confidences) else None
                ),
            }
        )
    return items


def _dataset_text(samples, shuffled, header_read: bool) -> str:
    """The dataset row: sample count plus the shuffled fact only when
    the header recorded one - an unrecorded flag is never asserted as
    "not shuffled"."""
    if samples is None:
        return "not set" if header_read else _NOT_FOUND
    text = f"{samples} sample(s)"
    if shuffled is not None:
        text += ", shuffled" if shuffled else ", not shuffled"
    return text


def _span_confidence(row) -> str | None:
    """The span's decided confidence, "0.90 ± 0.05" where a spread
    exists, suffixed with its provenance - "(mean)" when a vote
    decided, "(verifier)" after an overturn, bare for a single judge."""
    if pd.isna(row.confidence):
        return None
    conf = float(str(row.confidence))
    pm = float(str(row.confidence_pm)) if pd.notna(row.confidence_pm) else 0.0
    text = f"{conf:.2f} ± {pm:.2f}" if pm > 0 else f"{conf:.2f}"
    return text + CONFIDENCE_QUALIFIER.get(str(row.label_source), "")


def _span_verifier_cell(row) -> str | None:
    """One tooltip line for a reviewed span: what the verifier did -
    None (no row) for the unreviewed majority."""
    if not bool(row.verifier_selected):
        return None
    trigger = str(row.verifier_trigger).replace("_", " ")
    completed = getattr(row, "verifier_completed", None)
    status = getattr(row, "verifier_status", None)
    if (completed is not None and pd.notna(completed) and not bool(completed)) or (
        isinstance(status, str) and status in ("error", "refusal", "no_answer")
    ):
        return f"selected, no usable verdict ({trigger})"
    if bool(row.overturned):
        return f"overturned (was {row.original_label}; {trigger})"
    return f"reviewed, not overturned ({trigger})"


def _resolved_label(label_of: dict, span_id) -> str:
    """Shared "unclassified" fallback for a span with no judged (or a
    falsy/non-string) label - used by both the legend and the chart
    tooltips so the two cannot disagree."""
    label = label_of.get(span_id)
    return label if isinstance(label, str) and label else "unclassified"


def _verifier_text(regime: reliability.Regime) -> str:
    """The entity block's one-line verifier state, reusing the meta
    line's self-revision wording. "off" is exact - the
    scanner stamps ``verifier_armed`` at factory time - so an armed
    verifier that never triggered still reads on; a cohort never arms
    one."""
    if not regime.verifier_on:
        if regime.kind == "cohort":
            return "off (cohort regime: the majority vote is the correction mechanism)"
        return "off"
    model = regime.verifier_model or "(model not recorded)"
    if regime.verifier_same_model:
        return (
            f"on ({model}): self-revision under a different prompt; "
            "correctness unverified"
        )
    return f"on ({model})"


def _map_cell(
    top: str,
    bottom: str | None = None,
    health: float | None = None,
    share: float | None = None,
    ci: str | None = None,
) -> dict:
    """One audit-map cell: text (always), plus a background from the
    health ramp (red = low/unhealthy, steel = high/healthy) or the
    share ramp (intensity only - shares have no polarity). ``ci``
    renders after ``top`` in a smaller font."""
    bg = share_color(share) if share is not None else health_color(health)
    return {"top": top, "ci": ci, "bottom": bottom, "bg": bg, "fg": cell_text_color(bg)}


def _stat_cell(stat: reliability.Mean) -> dict:
    """A mean cell ("0.72 ± 0.09"), coloured by the mean; the ± is the
    mean's own CI half-width and is omitted where it is."""
    if stat.n == 0 or stat.mean is None:
        return _map_cell("n/a")
    top = f"{stat.mean:.2f}"
    if stat.ci_half_width is not None:
        top += f" ± {stat.ci_half_width:.2f}"
    return _map_cell(top, health=stat.mean)


def _rate_cell(
    value: reliability.Rate, unit: str, empty: str, invert: bool = True
) -> dict:
    """A `Rate` as a cell: "x% [a-b%]" over "count/of unit" (the Wilson
    interval is asymmetric, so brackets rather than a ±). ``invert``
    colours a high rate as unhealthy (re-label, spot-check, minority)."""
    if value.rate is None:
        return _map_cell(empty)
    return _map_cell(
        f"{value.rate:.0%}",
        ci=(
            f"[{value.ci[0]:.0%}\u2013{value.ci[1]:.0%}]"
            if value.ci is not None
            else None
        ),
        bottom=f"{value.count}/{value.of} {unit}",
        health=(1 - value.rate) if invert else value.rate,
    )


def _count_cell(count: int, asked: int) -> dict:
    """A miss-count cell, coloured by the surviving share."""
    health = 1 - (count / asked) if asked else None
    return _map_cell(str(count), health=health)


def _entity_maps(
    regime: reliability.Regime,
    stats: list[reliability.LabelStats],
    coverage: list[reliability.MemberCoverage],
    provenance: dict[str, dict[str, int]],
    unit_word: str,
    member_word: str,
) -> list[dict]:
    """The audit's per-entity maps, in reading order: Provenance,
    Reliability, Member coverage. Every metric row carries a
    full-numbers help tooltip; the same texts repeat in a collapsed
    legend under each table."""
    maps: list[dict] = []
    if not stats:
        return maps
    voting = regime.kind in ("k_roll", "cohort")
    columns = [{"top": st.label, "sub": f"N={st.n}"} for st in stats]
    group = f"Labels (N = decided {unit_word}s)"
    n_of = {st.label: st.n for st in stats}

    # which deciding mechanisms this run's judge setup makes possible -
    # an impossible source is not an empty row, it is no row
    source_help = {
        "single_judge": "decided directly by the lone judge.",
        "majority_vote": f"decided by the {member_word}' majority vote.",
        "verifier": "re-labelled by the second-round verifier.",
    }
    eligible = [
        source
        for source, possible in (
            ("single_judge", regime.kind == "solo"),
            ("majority_vote", voting),
            ("verifier", regime.verifier_on),
        )
        if possible
    ]
    # a source outside the regime's eligible set that still holds a
    # nonzero count (store drift, a mis-detected regime) renders anyway:
    # this table's shares must always sum over what actually decided
    for source in provenance:
        if source not in eligible and any(provenance[source].values()):
            eligible.append(source)
    eligible.sort(key=list(provenance).index)
    if any(count for by_label in provenance.values() for count in by_label.values()):
        maps.append(
            {
                "title": "Provenance map",
                "group": group,
                "columns": columns,
                "rows": [
                    {
                        "label": source,
                        "help": source_help[source],
                        "cells": [
                            _map_cell(
                                f"{provenance[source][st.label] / n_of[st.label]:.0%} "
                                f"({provenance[source][st.label]})",
                                share=provenance[source][st.label] / n_of[st.label],
                            )
                            if n_of[st.label] and provenance[source][st.label]
                            else _map_cell("-")
                            for st in stats
                        ],
                    }
                    for source in eligible
                ],
                "legend": [
                    (
                        "reading the cells",
                        f"share (count) of each classification's decided "
                        f"{unit_word}s, by the provenance that decided them; "
                        "rows show only the deciding mechanisms this run's "
                        "judge setup makes possible (plus any source the "
                        "store actually recorded, so the shares always "
                        "reconcile).",
                    )
                ]
                + [(source, source_help[source]) for source in eligible],
            }
        )

    rows = []
    legend = []
    if voting:
        help_text = (
            f"On the {unit_word}s that ended up with this label, how often "
            f"the {member_word} agreed with each other; the ± is a 95% CI "
            f"half-width on that mean (omitted below N=8, where a "
            f"normal-approximation ± would overstate precision). Each "
            f"{unit_word}'s own agreement divides the winning side by the "
            f"{member_word} that actually voted there, so the denominator "
            "shrinks when a member erred or refused - the member coverage "
            "map below counts those misses. Cells shade continuously on "
            "the mean itself: orange at 0, pale at 0.5, blue at 1 - no "
            "discrete tiers."
        )
        if regime.kind == "k_roll":
            help_text += (
                " The run-level self-consistency flag fires below 0.80 "
                "(heuristic, not a validated cutoff)."
            )
        rows.append(
            {
                "label": "mean agreement",
                "help": help_text,
                "cells": [_stat_cell(st.agreement) for st in stats],
            }
        )
        legend.append(("mean agreement", help_text))
    help_text = (
        f"On this label's decided {unit_word}s, how sure the deciding side "
        "was (dissenting votes not counted); the ± is a 95% CI half-width "
        "(omitted below N=8, where a normal-approximation ± would "
        "overstate precision). Heuristic tiers: high at 0.80 and above, low "
        "below 0.66."
    )
    rows.append(
        {
            "label": "mean confidence",
            "help": help_text,
            "cells": [_stat_cell(st.confidence) for st in stats],
        }
    )
    legend.append(("mean confidence", help_text))
    if regime.verifier_on:
        help_text = (
            "Of this label's judgements the verifier double-checked, the "
            "share it changed: the fraction is changed/examined, the "
            "brackets a Wilson 95% CI on that share. 0.20 and above flags "
            "amber (per label and overall; heuristic)."
        )
        rows.append(
            {
                "label": "verifier re-label rate",
                "help": help_text,
                "cells": [
                    _rate_cell(st.relabelled, "examined", "none examined")
                    for st in stats
                ],
            }
        )
        legend.append(("verifier re-label rate", help_text))
        help_text = (
            "Changes among the randomly picked double-checks only: the "
            "fraction is changed/sampled, the brackets a Wilson 95% CI. "
            "An applied relabel records disagreement, not established wrongness; "
            "any overturn here flags red for source review."
        )
        rows.append(
            {
                "label": "random spot-check overturns",
                "help": help_text,
                "cells": [
                    _rate_cell(st.spot_checked, "sampled", "none sampled")
                    for st in stats
                ],
            }
        )
        legend.append(("random spot-check overturns", help_text))
    if voting:
        help_text = (
            f"How often this label was a losing vote on a {unit_word} "
            "decided as something else: the fraction is losing votes/all "
            "votes naming it, the brackets a Wilson 95% CI. Votes on "
            f"{unit_word}s the verifier re-labelled are excluded on both "
            "sides - an overturn supersedes the ballot, so a vote there "
            "neither wins nor loses. A high share means the label keeps "
            "being proposed but rarely wins. Cells shade continuously on "
            "the inverse of the share (0% reads blue, 100% orange) - no "
            "flag threshold."
        )
        rows.append(
            {
                "label": "appears as minority vote",
                "help": help_text,
                "cells": [_rate_cell(st.minority, "votes", "no votes") for st in stats],
            }
        )
        legend.append(("appears as minority vote", help_text))
    legend.insert(
        0,
        (
            "reading the colours",
            "cells shade continuously (no discrete tiers): orange = "
            "low/unhealthy through pale to blue = high/healthy. Mean rows "
            "colour on the value itself; rate rows colour on the inverse, "
            "so a high re-label, spot-check, or minority rate reads "
            "orange. The number in the cell is always the fact.",
        ),
    )
    maps.append(
        {
            "title": "Reliability map",
            "group": group,
            "columns": columns,
            "rows": rows,
            "legend": legend,
        }
    )
    if coverage:
        reasons = list(coverage[0].misses)
        multi = {
            model
            for model in {member.model for member in coverage}
            if len({m.roll for m in coverage if m.model == model}) > 1
        }
        maps.append(
            {
                "title": "Member coverage",
                "group": None,
                "columns": [{"top": "produced of asked", "sub": None}]
                + [
                    {"top": reason.replace("_", " "), "sub": None} for reason in reasons
                ],
                "rows": [
                    {
                        "label": member_display(
                            member.model, member.roll, multi, short=False
                        ),
                        "help": None,
                        "cells": [
                            _rate_cell(
                                member.coverage,
                                unit_word + "s",
                                "none asked",
                                invert=False,
                            )
                        ]
                        + [
                            _count_cell(member.misses[reason], member.coverage.of)
                            for reason in reasons
                        ],
                    }
                    for member in coverage
                ],
                "legend": [
                    (
                        "produced of asked",
                        f"of the {unit_word}s this cohort member was asked to "
                        "judge, how many produced a vote: the fraction is "
                        "produced/asked, the brackets a Wilson 95% CI.",
                    ),
                    (
                        "miss reasons",
                        f"why the member's remaining {unit_word}s produced no "
                        "vote"
                        + (
                            ' ("filled" = the member answered its chunk but '
                            f"left these {unit_word}s uncovered)"
                            if any("filled" in c.misses for c in coverage)
                            else ""
                        )
                        + f". Misses on consecutive {unit_word}s matter more "
                        "than the same count scattered: that whole stretch is "
                        "decided by the remaining voters only, down to a "
                        "single voter with no vote agreement at all.",
                    ),
                ],
            }
        )
    return maps


def _unauditable_layer_block(name: str, error: Exception) -> dict:
    """The honest-absence fallback for a custom layer whose frame does
    not carry the columns the audit reads: the block names the failure
    instead of the report crashing on an arbitrary user frame."""
    return {
        "name": name,
        "rows": [
            {
                "label": "Not audited",
                "value": "This layer's frame could not be audited "
                f"({type(error).__name__}: {error}).",
                "definition": "The audit reads the judged columns "
                "transect.turns_frame emits (the decided label plus "
                "confidence, judge identity and agreement); a frame "
                "without them renders as data only, its coverage "
                "unassessed.",
            }
        ],
        "maps": [],
        "badge": True,
    }


def _entity_audit(
    name: str,
    entity: pd.DataFrame,
    members: pd.DataFrame,
    unit_col: str,
    value_col: str,
    agreement_source: pd.DataFrame,
    extra_rows: list[dict] | None = None,
    vocabulary: list[str] | None = None,
    unit_word: str | None = None,
    badge: bool = False,
) -> tuple[dict, list[reliability.Flag]]:
    """One judged surface's ("Phases" / "Sub-agents" / a custom layer's)
    reliability-audit rows + flags. ``agreement_source`` is whichever
    frame carries this surface's per-unit ``judge_agreement`` at the
    right grain - ``phase_turns`` (per-turn) for phases, ``subagents``
    itself (per-span) for sub-agents and custom layers; ``members`` is
    the matching per-member frame (``phase_turn_votes``/
    ``subagent_votes``/`member_ballots`) the cohort-agreement and
    k-roll-count computations read. ``extra_rows`` are this entity's
    own structural facts, built by the caller and prepended ahead of
    the judge-regime rows below. ``unit_word`` names the judged unit
    in map copy (defaults per surface); ``badge`` marks the block with
    the custom-layer badge.
    """
    regime = reliability.detect_regime(entity)
    k_roll_stat = reliability.mean_stat(agreement_source.judge_agreement)
    cohort = reliability.cohort_agreement(members, unit_col, value_col)
    relabel = reliability.relabel_rate(entity)
    spot_check = reliability.spot_check_overturns(entity)
    confidence_stat = reliability.mean_stat(entity.confidence)
    tiers = reliability.confidence_tiers(entity.confidence)

    # the per-classification maps: decided units, per-member coverage,
    # and deciding-provenance shares, each derived at this surface's
    # own grain (turns for Phases, spans for Sub-agents). The unjudged
    # rate excludes turns the judge never saw (attributed), so it reads
    # "of the units judging attempted, how many got no judgement".
    if name == "Phases":
        decided = (
            agreement_source[agreement_source.basis == "judged"]
            if len(agreement_source)
            else agreement_source
        )
        coverage = reliability.member_coverage(
            members,
            "basis",
            "judged",
            ("refusal", "no_answer", "missing_turn", "filled"),
        )
        unit_word = "turn"
        unjudged_count = sum(reliability.abstention_counts(agreement_source).values())
        attempted = len(agreement_source) - (
            int((agreement_source.basis == "attributed").sum())
            if len(agreement_source)
            else 0
        )
    else:
        decided = entity[entity[value_col].notna()] if len(entity) else entity
        coverage = reliability.member_coverage(
            members, "status", "ok", ("refusal", "error", "no_answer")
        )
        unit_word = unit_word or "span"
        unjudged_count = len(entity) - len(decided)
        attempted = len(entity)
    flags = reliability.build_flags(
        regime,
        k_roll_stat,
        cohort,
        relabel,
        spot_check,
        # a surface the judge never ran on has no unjudged share to flag
        unjudged=reliability.rate(unjudged_count, attempted)
        if regime.kind != "none"
        else reliability.NO_RATE,
    )
    stats = reliability.label_stats(
        decided, members, entity, unit_col, value_col, vocabulary=vocabulary
    )
    provenance = reliability.provenance_shares(
        decided, value_col, [st.label for st in stats]
    )
    maps = _entity_maps(
        regime,
        stats,
        coverage,
        provenance,
        unit_word,
        "rolls" if regime.kind == "k_roll" else "judges",
    )

    rows = list(extra_rows or [])
    rows += [
        {
            "label": "Judge regime",
            "value": reliability.describe_regime(regime),
            "definition": "The number of distinct judge models and/or repeated "
            "rolls that judged this surface, detected from its own frame.",
        },
        {
            "label": "Verifier",
            "value": _verifier_text(regime),
            "definition": "Whether a second-round verifier was armed for this "
            'surface. The scanner stamps the armed state, so "off" is exact: '
            "not armed. An armed verifier that never triggered still reads "
            "on, with zero counts in the Verifier selection row.",
        },
    ]
    if regime.kind == "k_roll":
        rows.append(
            {
                "label": "k-roll self-consistency (mean per-turn agreement)",
                "value": (
                    f"{k_roll_stat.mean:.2f}, N = {k_roll_stat.n}"
                    if k_roll_stat.n
                    else "no data"
                ),
                "definition": "Share of the k repeated rolls of the same judge "
                "model agreeing per turn, averaged across turns: self-"
                "consistency under repeat sampling, not agreement with any "
                "ground truth. Healthy heuristic: ≥ 0.80.",
            }
        )
    if regime.kind in ("cohort", "k_roll") and cohort.alpha is not None:
        rows.append(
            {
                "label": (
                    "Cohort inter-judge agreement"
                    if regime.kind == "cohort"
                    # for k-roll the same bracket is intra-rater
                    # (test-retest) self-consistency over the rolls,
                    # never "inter-judge"
                    else "Chance-corrected self-consistency (α / AC1)"
                ),
                "value": (
                    f"{cohort.percent_agreement:.0%} percent agreement · "
                    f"Krippendorff's α = {cohort.alpha:.2f}"
                    + (
                        f" · Gwet's AC1 = {cohort.ac1:.2f}"
                        if cohort.ac1 is not None
                        else ""
                    )
                    + f", N = {cohort.n}"
                    if cohort.alpha is not None
                    else "no data"
                ),
                "definition": "Percent agreement (chance-uncorrected) plus two "
                "chance-corrected coefficients with different chance models: "
                "Krippendorff's alpha and Gwet's AC1. They are not bounds on "
                "true agreement or correctness. "
                "Healthy heuristic on α: ≥ 0.80 green, 0.66–0.80 amber, "
                "< 0.66 red; flags threshold on α. These are reporting conventions.",
            }
        )
    if regime.verifier_on and relabel.overall.of:
        # an armed verifier can have examined nothing (zero doubt
        # triggers and an empty spot-check draw) - then there is no
        # rate to state and these rows are honestly absent; the
        # Verifier row above still says "on".
        overall = relabel.overall
        # "(95% CI x-y)", not the confidence mean's "X ± Y": a Wilson
        # proportion interval is asymmetric around the point estimate
        # (more so near 0/1, where a rate can't overshoot past the
        # boundary) - forcing a ± half-width here would misstate it as
        # symmetric. label_ci below is the same interval, per label.
        ci_text = (
            f" (95% CI {overall.ci[0]:.0%}–{overall.ci[1]:.0%})" if overall.ci else ""
        )
        rows.append(
            {
                "label": "Verifier re-label rate (overall)",
                "value": (
                    f"{overall.rate:.0%}{ci_text}, "
                    f"{overall.count}/{overall.of} examined"
                ),
                "definition": "Share of verifier-examined judgements the verifier "
                "overturned, with a Wilson 95% interval (v1 stand-in for "
                "Clopper-Pearson): a self-revision disclosure, not a "
                "correctness measure. Healthy heuristic: < 0.20.",
            }
        )
    rows.append(
        {
            "label": "Confidence",
            "value": reliability.format_confidence(confidence_stat),
            "definition": "Mean ± 95% CI (min), N of this surface's stated judge "
            'confidence: "Confidence = X ± Y (min Z), N = A" (± omitted below '
            "N=8, where a normal approximation would overstate precision).",
        }
    )
    rows.append(
        {
            "label": "Confidence tiers",
            "value": (
                f"high (≥0.80): {tiers.high} · medium: {tiers.medium} · "
                f"low (<0.66): {tiers.low} (N = {tiers.n})"
            ),
            "definition": "Count of units by stated-confidence tier: heuristic "
            "bands (reusing the cohort-agreement thresholds), not validated "
            "cutoffs.",
        }
    )
    if regime.verifier_on and (relabel.overall.of or spot_check.count):
        note = (
            ", including a random-sample spot-check overturn: treat labels "
            "with extra caution"
            if spot_check.count
            else ""
        )
        rows.append(
            {
                "label": "Verifier overturns",
                "value": f"{relabel.overall.count}/{relabel.overall.of} examined{note}",
                "definition": "Recap of the verifier re-label rate's raw counts, "
                "alongside the confidence numbers above.",
            }
        )
    return {"name": name, "rows": rows, "maps": maps, "badge": badge}, flags


def _span_verifier_rows(subagents: pd.DataFrame, unit_word: str = "span") -> list[dict]:
    """Verifier selection/outcome lines for the Sub-agents block and
    the custom-layer blocks, mirroring the phases rows."""
    if not len(subagents):
        return []
    selected = reliability.review_units(subagents)
    if not subagents.verifier_model.notna().any() and not len(selected):
        return []
    completed = selected[selected.verifier_completed] if len(selected) else selected
    triggers = (
        selected.verifier_trigger.value_counts()
        if len(selected)
        else pd.Series(dtype=int)
    )
    split = " · ".join(
        f"{int(count)} {str(trigger).replace('_', ' ')}"
        for trigger, count in triggers.items()
    )
    overturned = completed.overturned.fillna(False)
    weak = (
        ~overturned
        & completed.verifier_label.notna()
        & (completed.verifier_label != completed.original_label)
    )
    sampled = completed.verifier_trigger == "random_sample"
    return [
        {
            "label": "Verifier selection",
            "value": split or "none selected",
            "definition": f"Original {unit_word} units selected for review, "
            "split by trigger; not API calls.",
        },
        {
            "label": "Verifier outcomes",
            "value": f"{len(completed)} completed · "
            f"{len(selected) - len(completed)} without usable verdict · "
            f"{int(overturned.sum())} relabelled · {int(weak.sum())} "
            "weak relabel(s) recorded, not applied · "
            f"{int((overturned & sampled).sum())} of {int(sampled.sum())} "
            "completed random samples relabelled",
            "definition": "Relabel rates condition on usable completed verdicts; "
            "missing outcomes are reported separately.",
        },
    ]


def _status_cell(text: str, bad: bool) -> dict:
    """One execution-table cell: the audit ramp's red on a failure
    state, the table's default background otherwise."""
    if not bad:
        return {"text": text, "bg": None, "fg": None}
    bg = health_color(0.0)
    return {"text": text, "bg": bg, "fg": cell_text_color(bg)}


def _status_count(count: int) -> dict:
    """A failure-count cell: red when positive, plain otherwise."""
    return _status_cell(str(count), bad=count > 0)


def _usage_row(
    scanner: str,
    usage: ModelTokenUsage,
    index: int,
    group_size: int,
    *,
    total: bool = False,
) -> dict:
    """One billed-usage table row; a field no row reported is None and
    the template renders it as a faded dash. ``first``/``group_size``
    let the template merge a scanner's name cell across its models
    with a rowspan; ``total`` marks the run-wide rows."""

    def count(value) -> str | None:
        return None if value is None else f"{value:,}"

    if usage.total_cost is None:
        cost = None
    elif usage.total_cost >= 0.1:
        cost = f"${usage.total_cost:,.2f}"
    else:
        cost = f"${usage.total_cost:.4f}"
    return {
        "scanner": scanner,
        "is_total": total,
        "first": index == 0,
        "group_size": group_size,
        "model": usage.model,
        "input": count(usage.input_tokens),
        "output": count(usage.output_tokens),
        "total": count(usage.total_tokens),
        "cache_read": count(usage.input_tokens_cache_read),
        "cache_write": count(usage.input_tokens_cache_write),
        "reasoning": count(usage.reasoning_tokens),
        "cost": cost,
    }
