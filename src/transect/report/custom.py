"""Custom-layer sections: rendering a Layer's typed blocks.

Charts reuse the built-in turn-axis conventions (`charts.chart_width`
/ `turn_xlim`), so a custom per-turn chart lines up.
"""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from inspect_viz import Component as VizComponent, Data, Selection
from inspect_viz.input import select
from inspect_viz.interactor import highlight
from inspect_viz.mark import area_y, line, rect
from inspect_viz.plot import plot
from inspect_viz.table import table
from markdown_it import MarkdownIt
from markupsafe import Markup, escape

import transect.report.blocks as b
from transect.reliability import review_units
from transect.report import sections
from transect.report.charts import (
    _BAND_BODY,
    _BAND_MARGIN_BOTTOM,
    _BAND_MARGIN_LEFT,
    _BAND_MARGIN_RIGHT,
    _BAND_MARGIN_TOP,
    _IDENTITY_SCALE,
    _INTERVENTION_HEIGHT,
    _INTERVENTION_MARGIN_BOTTOM,
    _INTERVENTION_MARGIN_TOP,
    _event_hit_rect,
    _offset_rule_x,
    chart_width,
    turn_xlim,
)
from transect.report.colors import _UNJUDGED_GREY, _label_colors
from transect.report.embed import (
    _TIP_EXTRA_LINE_PX,
    _TIP_SHORT_ROW_PX,
    _WIDGET_ROW_HEIGHT,
    _section_height,
    _tip_floor,
    _tip_rows,
    embed_section,
)
from transect.report.style import _WIDGET_STYLE_CSS


@dataclass(frozen=True)
class SectionContext:
    """What a `Component` block's fn receives.

    ``frame`` is the layer's own frame, cut to this transcript.
    """

    transcript_id: str
    frame: pd.DataFrame | None
    n_turns: int


_CHART_HEIGHT = 150
_LANE_HEIGHT = 22
_TABLE_HEIGHT = 260
_MARGINS = 46

# a tooltip channel literally named "label" collides with the mark's
# own label channel and blanks the chart
_TIP_RENAMES = {"label": "classification"}

# bound to "" so re-selecting it clears the Selection (the same
# empty-string mechanism as charts._PHASE_FILTER_CLEAR_LABEL)
_FILTER_CLEAR_LABEL = "(all)"


def layer_section(
    layer,
    frame: pd.DataFrame | None,
    ctx: SectionContext,
    definitions: pd.DataFrame | None = None,
) -> Markup:
    """One layer's rendered section: badge-marked heading + its blocks,
    closed by the layer's recorded label definitions."""
    fragments: list[Markup] = []
    pending: list[VizComponent] = []
    pending_after: list[Markup] = []
    pending_height = 0

    def flush_pending() -> None:
        nonlocal pending, pending_after, pending_height
        if pending:
            fragments.append(
                Markup(
                    embed_section(pending, pending_height + _MARGINS, _WIDGET_STYLE_CSS)
                )
            )
            fragments.extend(pending_after)
            pending, pending_after, pending_height = [], [], 0

    for block in layer.section or []:
        if isinstance(block, b.Markdown):
            flush_pending()
            fragments.append(_markdown(block, ctx))
            continue
        components, height, after = _viz_block(block, frame, ctx)
        pending.extend(components)
        if after is not None:
            pending_after.append(after)
        pending_height += height
    flush_pending()
    heading = Markup(
        f'<h3>{escape(layer.name)} <span class="custom-layer-badge">custom</span></h3>'
    )
    meta = sections.layer_meta_line(frame) or Markup("")
    body = Markup("").join(fragments)
    tail = sections.layer_definitions(definitions, layer.name) or Markup("")
    return Markup(f'<div class="section">{heading}{meta}{body}{tail}</div>')


def _viz_block(block, own: pd.DataFrame | None, ctx: SectionContext):
    """One viz block -> (components, height, after-the-iframe HTML).

    A block-level ``frame`` override is cut to this transcript when it
    carries ``transcript_id`` (``own`` arrives pre-cut from render);
    without the column it broadcasts as-is."""
    if isinstance(block, b.Component):
        return [block.fn(ctx)], _CHART_HEIGHT, None
    frame = own
    if isinstance(block.frame, pd.DataFrame):
        frame = block.frame
        if "transcript_id" in frame.columns:
            frame = frame[frame.transcript_id == ctx.transcript_id]
    if frame is None:
        raise RuntimeError(
            f"{type(block).__name__} block reached render with no frame - "
            "load-time validation should have refused this layer"
        )
    if isinstance(block, b.Table):
        columns = block.columns or [c for c in frame.columns]
        return (
            [table(Data.from_dataframe(frame[columns]), height=_TABLE_HEIGHT)],
            _TABLE_HEIGHT,
            None,
        )
    if isinstance(block, b.TurnChart):
        return [_turn_chart(block, frame, ctx)], _CHART_HEIGHT, None
    if isinstance(block, b.TurnBand):
        return _turn_band(block, frame, ctx)
    if isinstance(block, b.SpanLanes):
        component, height = _span_lanes(block, frame, ctx)
        return [component], height, None
    if isinstance(block, b.EventMarks):
        component, height = _event_marks(block, frame, ctx)
        return [component], height, None
    raise ValueError(f"unrenderable block {type(block).__name__}")


def _tip_name(column: str) -> str:
    return _TIP_RENAMES.get(column, column)


def _turn_chart(block: b.TurnChart, frame: pd.DataFrame, ctx: SectionContext):
    """bar/line/step/area on the shared turn axis, with a per-turn
    tooltip column."""
    rows = frame.dropna(subset=[block.y]).sort_values("turn")
    values = pd.DataFrame(
        {
            "turn": rows.turn.astype(int),
            "y": rows[block.y].astype(float),
            "x1": rows.turn.astype(int) - 0.4,
            "x2": rows.turn.astype(int) + 0.4,
            "y0": 0.0,
        }
    )
    data = Data.from_dataframe(values)
    channels = {"turn": "turn", _tip_name(block.y): "y"}
    if block.kind == "bar":
        marks = [rect(data, x1="x1", x2="x2", y1="y0", y2="y", fill="#4c72b0")]
    elif block.kind == "area":
        marks = [area_y(data, x="turn", y="y", fill="#a8c4e0")]
    else:
        curve: Literal["step", "linear"] = "step" if block.kind == "step" else "linear"
        marks = [line(data, x="turn", y="y", stroke="#4c72b0", curve=curve)]
    # invisible per-turn hit column carrying the tooltip
    tips = pd.DataFrame(
        {
            "turn": values.turn,
            "tx1": values.turn - 0.5,
            "tx2": values.turn + 0.5,
            "y": values.y,
        }
    )
    marks.append(
        rect(
            Data.from_dataframe(tips),
            x1="tx1",
            x2="tx2",
            fill="#000",
            fill_opacity=0.0,
            channels=channels,
            tip=True,
        )
    )
    return plot(
        *marks,
        width=chart_width(ctx.n_turns),
        height=_CHART_HEIGHT,
        x_domain=turn_xlim(ctx.n_turns),
        x_label="turn",
        y_label=block.y,
    )


def _turn_band(block: b.TurnBand, frame: pd.DataFrame, ctx: SectionContext):
    """Contiguous same-label runs as coloured chunks at the phase
    band's measurements, with its filter select and legend chips;
    judge rows ride the tooltip when the frame carries them."""
    rows = frame.dropna(subset=[block.label]).sort_values("turn")
    runs: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        label = str(row[block.label])
        turn = int(row["turn"])
        if runs and runs[-1]["label"] == label and turn == int(runs[-1]["end"]) + 1:
            runs[-1]["end"] = turn
        else:
            runs.append({"label": label, "start": turn, "end": turn})
    colors = _label_colors(sorted({str(r["label"]) for r in runs}))
    bands = pd.DataFrame(
        {
            "x1": [int(r["start"]) - 0.5 for r in runs],
            "x2": [int(r["end"]) + 0.5 for r in runs],
            "y1": 0,
            "y2": 1,
            "fill": [colors.get(str(r["label"]), _UNJUDGED_GREY) for r in runs],
            "band_label": [r["label"] for r in runs],
            "turns": [f"{r['start']}–{r['end']}" for r in runs],
        }
    )
    channels = {_tip_name(block.label): "band_label", "turns": "turns"}
    for name, column, cells in _run_judge_cells(rows, runs):
        bands[column] = cells
        channels[name] = column
    # filler rows over turns the frame has no rows for
    bands["note"] = None
    channels["note"] = "note"
    covered = set(rows.turn.astype(int))
    gaps: list[tuple[int, int]] = []
    gap_start: int | None = None
    for turn in range(ctx.n_turns):
        if turn in covered:
            if gap_start is not None:
                gaps.append((gap_start, turn - 1))
                gap_start = None
        elif gap_start is None:
            gap_start = turn
    if gap_start is not None:
        gaps.append((gap_start, ctx.n_turns - 1))
    if gaps:
        filler = pd.DataFrame(
            {
                "x1": [start - 0.5 for start, _ in gaps],
                "x2": [end + 0.5 for _, end in gaps],
                "y1": 0,
                "y2": 1,
                "fill": "#ececec",
                "turns": [
                    str(start) if start == end else f"{start}–{end}"
                    for start, end in gaps
                ],
                "note": "missing turn",
            }
        )
        bands = pd.concat([bands, filler], ignore_index=True)
    turn_counts: dict[str, int] = {
        str(label): int(count)
        for label, count in rows[block.label].astype(str).value_counts().items()
    }
    label_options = sorted(turn_counts, key=lambda one: (-turn_counts[one], one))
    label_selection = Selection.single(cross=False)
    data = Data.from_dataframe(bands)
    filter_select = select(
        data=data,
        column="band_label",
        options={
            _FILTER_CLEAR_LABEL: "",
            **{label: label for label in label_options},
        },
        target=label_selection,
        label="Filter:",
    )
    band = plot(
        rect(
            data,
            x1="x1",
            x2="x2",
            y1="y1",
            y2="y2",
            fill="fill",
            channels=channels,
            tip=True,
        ),
        # filler rows share the mark, so an active filter dims them
        # with the unselected runs
        highlight(by=label_selection, opacity=0.15, fill=_UNJUDGED_GREY),
        width=chart_width(ctx.n_turns),
        height=_BAND_BODY + _BAND_MARGIN_TOP,
        margin_top=_BAND_MARGIN_TOP,
        margin_bottom=_BAND_MARGIN_BOTTOM,
        margin_left=_BAND_MARGIN_LEFT,
        margin_right=_BAND_MARGIN_RIGHT,
        x_domain=turn_xlim(ctx.n_turns),
        x_axis="top",
        x_label="turn",
        y_axis=False,
        color_scale=_IDENTITY_SCALE,
    )
    body = _BAND_BODY + _BAND_MARGIN_TOP + _BAND_MARGIN_BOTTOM
    # single-line tooltip rows, one two-line allowance for the label
    floor = _tip_floor(_tip_rows(channels), _TIP_SHORT_ROW_PX + 2 * _TIP_EXTRA_LINE_PX)
    height = max(body + _WIDGET_ROW_HEIGHT, floor)
    chips = sections.label_chips(turn_counts, colors)
    return [filter_select, band], height, chips


def _run_judge_cells(
    rows: pd.DataFrame, runs: list[dict]
) -> list[tuple[str, str, list]]:
    """Per-run judge tooltip cells."""
    pieces = [rows[(rows.turn >= r["start"]) & (rows.turn <= r["end"])] for r in runs]

    def carries(column: str) -> bool:
        return bool(column in rows.columns and rows[column].notna().any())

    cells: list[tuple[str, str, list]] = []
    if carries("confidence"):
        cells.append(
            (
                "confidence",
                "confidence_text",
                [
                    f"{p.confidence.mean():.2f} (mean)"
                    if p.confidence.notna().any()
                    else None
                    for p in pieces
                ],
            )
        )
    if carries("judge_agreement"):
        cells.append(
            (
                "judge agreement",
                "agreement_text",
                [
                    f"{p.judge_agreement.mean():.2f} (mean)"
                    if p.judge_agreement.notna().any()
                    else None
                    for p in pieces
                ],
            )
        )
    if carries("label_source"):
        cells.append(
            (
                "label source",
                "source_text",
                [
                    " · ".join(sorted(set(p.label_source.dropna().astype(str)))) or None
                    for p in pieces
                ],
            )
        )
    if carries("verifier_selected") and rows.verifier_selected.fillna(False).any():

        def reviewed(p: pd.DataFrame) -> str | None:
            units = review_units(p)
            if not len(units):
                return None
            completed = units[units.verifier_completed]
            overturned = (
                int(completed.overturned.fillna(False).sum())
                if "overturned" in completed.columns
                else 0
            )
            return (
                f"{len(units)} selected · {len(completed)} completed · "
                f"{len(units) - len(completed)} without usable verdict · "
                f"{overturned} overturned"
            )

        cells.append(("verifier", "verifier_text", [reviewed(p) for p in pieces]))
    return cells


def _span_lanes(block: b.SpanLanes, frame: pd.DataFrame, ctx: SectionContext):
    """One horizontal lane per row, rects spanning start..end."""
    rows = frame.dropna(subset=[block.start, block.end]).reset_index(drop=True)
    lanes = pd.DataFrame(
        {
            "x1": rows[block.start].astype(float) - 0.5,
            "x2": rows[block.end].astype(float) + 0.5,
            "y1": rows.index + 0.15,
            "y2": rows.index + 0.85,
            "lane_label": rows[block.label].astype(str),
        }
    )
    mark = rect(
        Data.from_dataframe(lanes),
        x1="x1",
        x2="x2",
        y1="y1",
        y2="y2",
        fill="#4c72b0",
        fill_opacity=0.55,
        channels={_tip_name(block.label): "lane_label"},
        tip=True,
    )
    height = max(len(rows), 1) * _LANE_HEIGHT + _MARGINS
    component = plot(
        mark,
        width=chart_width(ctx.n_turns),
        height=height,
        x_domain=turn_xlim(ctx.n_turns),
        x_label="turn",
        y_axis=False,
    )
    return component, height


def _event_marks(block: b.EventMarks, frame: pd.DataFrame, ctx: SectionContext):
    """Dashed rules at the frame's turns."""
    rows = frame.dropna(subset=["turn"])
    events = pd.DataFrame({"turn": rows.turn.astype(int), "event": str(block.label)})
    channels = {"event": "event", "turn": "turn"}
    component = plot(
        _event_hit_rect(events, -1, 1, channels),
        _offset_rule_x(
            Data.from_dataframe(events[["turn"]]),
            "#7b5cb8",
            "4,3",
            pointer_events="none",
            stroke_width=2,
        ),
        width=chart_width(ctx.n_turns),
        height=_INTERVENTION_HEIGHT,
        y_axis=False,
        y_domain=(-1, 1),
        margin_top=_INTERVENTION_MARGIN_TOP,
        margin_bottom=_INTERVENTION_MARGIN_BOTTOM,
        margin_left=_BAND_MARGIN_LEFT,
        margin_right=_BAND_MARGIN_RIGHT,
        x_domain=turn_xlim(ctx.n_turns),
        x_label="turn",
    )
    return component, _section_height(_INTERVENTION_HEIGHT, _tip_rows(channels))


def _markdown(block: b.Markdown, ctx: SectionContext) -> Markup:
    """Markdown -> HTML: raw HTML escaped (html=False - the text may
    interpolate transcript-derived strings), user headings demoted
    below the section title."""
    text = block.text(ctx) if callable(block.text) else block.text
    html = MarkdownIt("commonmark", {"html": False}).render(str(text))
    for level in (4, 3, 2, 1):  # h1..h4 -> h5..h6 (capped)
        demoted = min(level + 4, 6)
        html = html.replace(f"<h{level}>", f"<h{demoted}>").replace(
            f"</h{level}>", f"</h{demoted}>"
        )
    return Markup(f'<div class="custom-md">{html}</div>')
