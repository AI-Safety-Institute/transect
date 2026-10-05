"""Chart builders for the report's chart sections.

One chart section, one component tree, one iframe (`embed_section`).
Charts of one transcript share an x-domain (`turn_xlim`) and a width
(`chart_width`), so the same turn sits in the same pixel column in every
section; they share no Mosaic selection, and each chart carries its own
left margin (`_BAND_MARGIN_LEFT`, `_TOKEN_MARGIN_LEFT`,
`_SWIMLANE_MARGIN_LEFT`, `_INTERVENTION_MARGIN_LEFT`).

Constraints ledger (detail sits on each named function or constant):

- **Tooltips are tippy popovers, not Plot's own svg tip.** `inspect_viz`
  hides `g[aria-label="tip"]` and re-renders its content through tippy.
  Use `channels=`, never `title=`: a bound `title` channel renders
  label-less lines that the re-renderer parses into zero rows, so the
  popover shows an empty box.
- **One channel-key set per plot.** `readUserChannels` applies the first
  tip-bearing mark's channel keys as a plot-wide row filter, so a second
  tip mark with different keys renders nothing. One tip-bearing mark per
  plot.
- **A conditional tooltip row is a null cell**, not a second mark: an
  absent value emits no row at all.
- **Suppress an axis with `False`, never `None`.** `inspect_viz`'s
  attribute marshaller drops any `PlotAttributes` key whose value is
  `None` before serializing, so `y_axis=None` means "not passed".
  `plot()`'s own top-level `x_label`/`y_label` params use a `NOT_GIVEN`
  sentinel instead, so `None` does reach the JS side there.
- **Interactors bind to the mark immediately before them** in the
  `plot()` argument list, not to the plot. A `filter_by=` mark declared
  ahead of `nearest_x` hands the interactor empty data and blanks the
  chart with `undefined is not iterable`.
- **An interval mark (`rect`) sharing a plot with `nearest_x` throws
  in-browser** and blanks the chart, so `phase_band` and `swimlanes`
  carry no hover cursor of their own.
- **A literal-hex `fill=`/`opacity=` column is remapped through Plot's
  own scale** unless the plot carries `color_scale="identity"` /
  `opacity_scale="identity"`.
- **`href=` on the mark a `highlight` interactor targets kills that
  interactor's reactivity** - hence `phase_band`'s two-rect split: one
  visible/filterable rect, one transparent click-catcher declared after
  `highlight`.
- **A `Param` referenced only inside a mark's `sql()` channel text is
  not reactive** on its own; it needs a non-string `Param` reference on
  a real scale-defining attribute of the same mark (`token_stack`'s
  `y_scale=scale`). A mark with no scale to spare targets a `Selection`
  instead (`phase_band`'s label filter and fill-mode checkbox).
- **A checkbox-bound `Param` on a rule's `stroke_opacity` does not
  toggle the rule.** Filter its data with a `Selection` instead
  (`token_stack`'s threshold rule); the y-scale follows the visible
  marks, so hiding the rule rescales the curve - never pin the domain to
  the threshold, which squashes a curve far below it for good.
- **A `sql()` x-expression leaks into the x-scale's auto-inferred
  label.** Pin `x_label=` on any plot with a `sql()`-computed x.
- **An `x_axis="top"` plot's axis label is pinned near the svg's
  absolute top edge regardless of `margin_top`, while the tick-number
  row tracks `margin_top`.** A `margin_top` sized for the ticks alone
  renders them overlapping the label (`_BAND_MARGIN_TOP`).
- **A bare-string mark option that is not one of Plot's own recognized
  constants for that option resolves as a column name**, and the binder
  error blanks the whole plot. Check a symbol name against the `Symbol`
  literal on `inspect_viz.mark.dot`'s signature (`_END_MARKER_SYMBOL`).
- **A tooltip channel named `label` collides with Plot's own mark/scale
  option and blanks the chart** (`_RESERVED_TIP_CHANNELS`).

`inspect_viz` tracks every `Data` object in a module-global registry
that only grows, and `to_html` inlines every registry entry rather than
scoping to the component tree it was handed. Repeated `render_report`
calls in one process therefore accumulate: a later epoch's report file
carries earlier epochs' tables, and each section's iframe inlines every
other section's tables. Fixing it needs an upstream fragment-embed API;
not done here.
"""

from typing import Any

import pandas as pd
from inspect_viz import Component, Data, Param, Selection
from inspect_viz.input import checkbox, radio_group, select
from inspect_viz.interactor import highlight, nearest_x
from inspect_viz.layout import hconcat
from inspect_viz.mark import (
    Mark,
    TextStyles,
    TipOptions,
    dot,
    line,
    rect,
    rule_x,
    rule_y,
    text,
)
from inspect_viz.plot import plot
from inspect_viz.transform import Transform, sql

from transect.report.colors import _AGREEMENT_TEAL, _UNJUDGED_BASES, _UNJUDGED_GREY
from transect.report.display import (
    CONFIDENCE_QUALIFIER,
    MEMBER_NO_VOTE,
    member_display,
    multi_roll_models,
)
from transect.report.embed import (
    _LANE_WORST_CHARS_PER_LINE,
    _TIP_EXTRA_LINE_PX,
    _TIP_SHORT_ROW_PX,
    _section_height,
    _tip_floor,
    _tip_rows,
    tip_fit_height,
    wrap_row_px,
)


def chart_width(n_turns: int) -> int:
    """Shared per-transcript chart pixel width.

    Every chart builder for one transcript computes its width this way:
    a mismatch would put the same turn in a different pixel column in
    each section, which a reader comparing two sections reads as the
    charts disagreeing. 1.6px/turn keeps turn-to-turn spacing legible;
    floored at 1000px so a short transcript isn't cramped, capped at
    3000px so a very long one doesn't demand an unusable width.
    """
    return max(1000, min(3000, int(1.6 * n_turns)))


def turn_xlim(n_turns: int) -> tuple[float, float]:
    return (-0.5, n_turns - 0.5)


# "identity" is a real, browser-verified Plot scale value (see the
# literal-hex ledger entry in the module docstring) that inspect_viz's
# ColorScale literal does not list.
_IDENTITY_SCALE: Any = "identity"


def phase_band(
    phase_turns: pd.DataFrame,
    phases: pd.DataFrame,
    colors: dict[str, tuple[str, str | None]],
    card_id_prefix: str,
    turn_votes: pd.DataFrame,
) -> tuple[list[Component], int]:
    """The phase-timeline band: one continuous rect per contiguous run of
    the dense per-turn phase assignment (`phase_runs`), plus an optional
    per-turn agreement strip.

    Draws no event markers: flush markers live on the token telemetry
    chart, intervention markers on their own.

    Two kinds of row share one rect mark's data (see "One mark, two row
    kinds" below). Both come from the same pass over `phase_runs`, so
    every row spans ``run_start - 0.5`` to ``run_end + 0.5`` and the rows
    tile the axis: a turn present in ``phase_turns`` is inside exactly
    one rect, never two and never none.

    - **A judged run**, painted its phase's own colour. A phase whose
      turns are unbroken draws as a single chunk; one the dense map
      assigns in several stretches draws one rect per stretch, all
      sharing that phase's colour, tooltip and click target.
    - **An unjudged run**, coloured grey (`_UNJUDGED_GREY`) regardless of
      what phase the turns nominally belong to: turns whose ``basis`` is
      refusal/no_answer/missing_turn, or that own no resolvable
      ``phase_index``. The tooltip names whichever of the two is the
      actual reason - ``"unjudged (refusal)"`` off the turn's own basis,
      or ``"unjudged (no matching phase)"`` when the basis is an
      ordinarily-judged one and it is the phase index that dangles. An
      unjudged turn inside an otherwise-judged phase therefore splits
      that phase's chunk rather than being absorbed into it: a doubtful
      or absent judgement is never painted the neighbouring phase's hue.

    Both row kinds share one column set, so one ``rect`` mark and one
    tooltip serve both with no render-time branching. ``turns_range`` is
    the run's own rendered extent, not the owning phase's declared
    range: the dense map can assign more turns to a phase than its
    declared range covers, and a tooltip reading "turns 6-6" over a
    chunk visibly spanning 6-8 disagrees with what the reader is looking
    at. ``reasoning_turns_range`` carries the phase's declared range as a
    second, distinctly-labelled row, and only when it differs from the
    run's extent (`None` - and so no row at all - otherwise).

    A phase the dense map never assigns to any turn draws no rect: the
    dense map is the authority on which turn belongs to what, and
    inventing a chunk from a contradicted declared range is what
    produces white gaps. Such a phase is reachable through its card. The
    one case that does fall back to declared ranges is an empty
    ``phase_turns`` with a non-empty ``phases``.

    The owning phase's confidence drives fill opacity for both row kinds
    (0.85 at/above 0.6, else 0.5); a row with no owning phase falls to
    the dimmer value the same way a low-confidence one does. The
    checkbox-bound "confidence" fill mode replaces that two-level alpha
    with the judge's own per-run confidence value. Both modes need
    ``opacity_scale="identity"``: Plot otherwise auto-remaps the literal
    column through its own zero-based linear scale, which on a
    uniform-agreement scan collapses to a single value rendered at full
    opacity regardless of the number authored.

    Two plots, not one: the band and the agreement strip are visually
    distinct rows (110px vs 24px), `vconcat`-ed into this section's one
    iframe and sharing an x-domain and margins. The strip stays per-turn
    - ``judge_agreement`` varies turn-to-turn within one phase, which is
    the point of it - so it was never a candidate for the band's
    per-phase chunking. Each row is named in the left margin
    (`_margin_label`).

    No interactor and no hover line: an interval mark sharing a plot
    with `nearest_x` throws in-browser (module docstring), so this chart
    can never resolve a hover of its own. Values come from its tooltip.

    ``y1``/``y2`` are real per-row columns of 0/1, not the bare literals
    ``y1=0, y2=1``: a literal scalar channel option can leave a consumer
    with no real array to read, and a constant-valued column produces
    the identical unit-height rect with a real channel behind it.

    Both plots pass `y_axis=False` (not `None`): the y-axis would
    otherwise leak the internal ``"y1"``/``"y2"`` column names.

    The band's x-axis is `x_axis="top"` (the literal ``"top"`` is a real
    supported value, not just ``True``/``False``), so the band opens with
    a numeric turn reference and keeps the axis next to the section
    header, away from the legend chips below. `x_label="turn"` matches
    every other turn-axis chart here, and needs `_BAND_MARGIN_TOP`'s
    clearance to not collide with the tick row. The agreement strip keeps
    `x_axis=False`: a second tick row 9px below the first is clutter.

    Two inputs render ahead of the band. A phase-label `select` targets a
    `Selection` (not a `Param`: this mark has no scale to make a
    `sql()`-embedded Param reactive - module docstring) paired with
    `highlight(by=..., fill=_UNJUDGED_GREY)`, which greys every rect the
    selection doesn't match. ``label`` is the filter-matching column,
    distinct from the tooltip's ``phase_label`` cell: an unjudged row's
    ``label`` is always the literal ``"unjudged"``, matching no filter
    option, so any active filter greys it too. Options are this
    transcript's own phase labels in ``colors``' order, the same order
    `sections.phase_chips` uses for its legend chips.

    `_PHASE_FILTER_CLEAR_LABEL` is bound to the empty string, not a
    non-empty sentinel: selecting a sentinel adds a real
    ``label = '<sentinel>'`` predicate matching zero rows, which
    `highlight` reads as "nothing selected" and greys everything.
    Selecting an empty-string value clears the `Selection` on every
    click, which is what makes a mouse-only "show everything" work.

    A second rect, drawn last (after `highlight`, so it is not the mark
    `highlight` binds to), carries the click-to-card behaviour: ``href``
    (`_phase_anchor`) plus ``target="_parent"``, so a click anywhere in a
    phase's chunk opens its card. It is fully transparent and
    geometrically identical to the visible rect beneath it. Two marks,
    not one, because ``href`` on the mark `highlight` targets stops
    `highlight` reacting to the `Selection` entirely (module docstring).

    ``target="_parent"`` rather than a document-head ``<base>`` tag: a
    srcdoc iframe with no ``sandbox`` attribute is same-origin with its
    parent and free to navigate it, and the fragment resolves against
    the parent's URL, so the href needs no absolute prefix. A cell with
    no resolvable ``phase_index`` gets a ``None`` anchor, which Plot
    renders without the `<a>` wrapper rather than as a link to nowhere.
    The outer page's hash-navigation script (`templates/report.html.j2`)
    turns the resulting hash change into an opened, scrolled-to card.

    ``card_id_prefix`` is the id namespace this transcript's anchors
    resolve into, matching whatever prefix `sections.phase_cards` was
    given for the same transcript. `render.py` passes ``f"phase-{idx}"``:
    a bare ``phase-<phase_index>`` collides across transcripts on a
    multi-transcript page (two transcripts can each have a "phase 0"),
    producing duplicate ids and a click that always resolves to the
    earlier transcript's card.

    One mark, two row kinds: a second tip-bearing mark for the unjudged
    rows is ruled out by the "one channel-key set per plot" rule (module
    docstring), so both kinds share one `Data` table and one mark. The
    ``"phase"`` channel is the cell that maps 1:1 onto a card and is kept
    as its own row rather than folded into ``phase_label`` (which for an
    unjudged cell reads ``"unjudged (refusal)"`` and would bury it).
    ``confidence`` renders the owning phase's plain value - a single
    phase's confidence has no interval to attach.
    """
    per_turn = phase_turns.sort_values("turn").reset_index(drop=True)
    n_turns = (
        int(per_turn.turn.max()) + 1
        if len(per_turn)
        else int(phases.turn_end.max()) + 1
    )

    conf_of = dict(zip(phases.phase_index, phases.confidence, strict=True))
    span_of = {
        p.phase_index: (int(str(p.turn_start)), int(str(p.turn_end)))
        for p in phases.itertuples()
    }

    def alpha_of(confidence) -> float:
        # NaN confidence (no owning phase) compares False to >= 0.6 and
        # lands on the dimmer 0.5 - deliberate, see docstring
        return 0.85 if pd.notna(confidence) and confidence >= 0.6 else 0.5

    label_of = dict(zip(phases.phase_index, phases.phase, strict=True))

    def chunk_row(x1: float, x2: float, index, basis: str | None) -> dict:
        """One rect row. ``basis`` None marks a judged run (painted its
        phase's colour); anything else is the unjudged run's own basis,
        which the tooltip names."""
        phase = label_of.get(index)
        confidence = conf_of.get(index)
        span = span_of.get(index)
        grey = basis is not None or phase is None
        # A grey run is grey for one of two reasons and the tooltip has
        # to name the right one: where the per-turn record says the
        # judgement was refused/absent, its own basis is the reason;
        # where the basis is an ordinarily-judged one but the phase index
        # resolves to no row of `phases`, echoing the basis would read
        # "unjudged (judged)" and the honest reason is the missing phase.
        reason = basis if basis in _UNJUDGED_BASES else "no matching phase"
        # the run's own rendered extent (see docstring), recovered from
        # x1/x2 rather than threaded through both call sites - `x1 =
        # start - 0.5`/`x2 = end + 0.5` always, so the inverse is exact
        run_start, run_end = round(x1 + 0.5), round(x2 - 0.5)
        reasoning_range = (
            f"{span[0]}–{span[1]}"
            if span is not None and pd.notna(confidence)
            else None
        )
        return {
            "x1": x1,
            "x2": x2,
            "y1": 0,  # real columns, not literal y1=0/y2=1 - see docstring
            "y2": 1,
            "color": _UNJUDGED_GREY
            if grey
            else colors.get(str(phase), (_UNJUDGED_GREY, None))[0],
            "alpha": alpha_of(confidence),
            # the label filter's match column: a phase's own name, or
            # the literal "unjudged" (matching no filter option, so any
            # active filter greys the run too)
            "label": "unjudged" if grey else phase,
            "phase_index": index,
            "phase_label": (f"unjudged ({reason})" if grey else phase),
            "turns_range": f"{run_start}–{run_end}",
            # the phases frame's own declared range, shown as a second
            # tooltip row only when it differs from the run's extent
            # above - an absent cell emits no row at all
            "reasoning_turns_range": (
                reasoning_range
                if reasoning_range is not None
                and reasoning_range != f"{run_start}–{run_end}"
                else None
            ),
            "confidence": confidence,
            "anchor": _phase_anchor(index, span_of, card_id_prefix),
        }

    if len(per_turn):
        # one row per contiguous run of the dense per-turn assignment -
        # see `phase_runs` for why not one row per phase
        chunk_rows = [
            chunk_row(start - 0.5, end + 0.5, index, basis)
            for start, end, index, basis in phase_runs(per_turn, set(span_of))
        ]
    else:
        # no dense map to read: fall back to each phase's declared range.
        # Nothing real reaches this (a scan with phases always carries
        # per-turn rows), but it keeps a phases-only frame renderable
        # instead of raising on an empty chunk table.
        chunk_rows = [
            chunk_row(
                int(str(p.turn_start)) - 0.5,
                int(str(p.turn_end)) + 0.5,
                p.phase_index,
                None,
            )
            for p in phases.itertuples()
        ]

    chunks = pd.DataFrame(chunk_rows)
    width = chart_width(n_turns)
    # Confidence-as-opacity toggle: every chunk row exists twice, once
    # per fill mode. "solid" keeps the two-level alpha (`alpha_of`);
    # "confidence" sets fill opacity to the judge's own per-run
    # confidence (grey/unjudged runs keep their dim alpha). A
    # checkbox-bound Selection + `filter_by` picks which set renders -
    # data-level filtering, not a Param in sql() text, which this mark
    # cannot react to (module docstring).
    chunks = pd.concat(
        [
            chunks.assign(fill_mode="solid"),
            chunks.assign(
                fill_mode="confidence",
                alpha=chunks.confidence.astype(float).fillna(chunks.alpha),
            ),
        ],
        ignore_index=True,
    )
    data = Data.from_dataframe(
        chunks[
            [
                "x1",
                "x2",
                "y1",
                "y2",
                "color",
                "label",
                "alpha",
                "fill_mode",
                "phase_index",
                "phase_label",
                "turns_range",
                "reasoning_turns_range",
                "confidence",
                "anchor",
            ]
        ]
    )
    fill_mode_selection = Selection.single(cross=False)
    conf_checkbox = checkbox(
        data=data,
        target=fill_mode_selection,
        field="fill_mode",
        # checked -> the confidence rows render; unchecked -> the solid
        values=("confidence", "solid"),
        label="judge confidence as fill opacity",
    )

    # The phase-label filter, scoped to this chart only. Options are this
    # transcript's own labels in `colors`' order - the same
    # order/filtering `sections.phase_chips` uses. A Selection, not a
    # Param - see the docstring.
    label_options = [label for label in colors if label in set(phases.phase)]
    label_selection = Selection.single(cross=False)
    filter_select = select(
        data=data,
        column="label",
        # "" (empty string), never a non-empty sentinel - see docstring
        options={
            _PHASE_FILTER_CLEAR_LABEL: "",
            **{label: label for label in label_options},
        },
        target=label_selection,
        label="Filter phase:",
    )

    # Named rather than inline so this section's height floor can count
    # its rows (`_section_height`). "phase label", not "label" - see
    # `_RESERVED_TIP_CHANNELS`.
    band_channels = {
        "phase": "phase_index",
        "phase label": "phase_label",
        "turns": "turns_range",
        "reasoning turns": "reasoning_turns_range",
        "confidence": "confidence",
    }
    band = plot(
        # declared first so `highlight` still binds to the rect before it
        _margin_label("phases"),
        rect(
            data,
            x1="x1",
            x2="x2",
            y1="y1",
            y2="y2",
            fill="color",
            opacity="alpha",
            filter_by=fill_mode_selection,  # solid vs confidence rows -
            # see the mode-duplication comment above
            tip=TipOptions(pointer="x"),  # not tip="x" - see
            # _event_hit_rect
            channels=band_channels,
        ),
        highlight(by=label_selection, opacity=0.15, fill=_UNJUDGED_GREY),
        # the label filter's greying mechanism. Binds to the visible rect
        # mark immediately above it, never the click-catcher below.
        rect(
            data,
            x1="x1",
            x2="x2",
            y1="y1",
            y2="y2",
            filter_by=fill_mode_selection,  # same mode rows as the
            # visible mark - without this the twin doubles every hit cell
            fill_opacity=0,  # invisible click-catcher - href on the
            # visible mark would break highlight's reactivity
            href="anchor",  # -> that phase's card, None when the cell
            # owns no resolvable phase
            target="_parent",  # navigate the outer page, not this iframe
        ),
        width=width,
        height=_BAND_BODY + _BAND_MARGIN_TOP,
        margin_top=_BAND_MARGIN_TOP,  # room for the top axis's ticks
        # *and* its "turn" label - explicit, see _BAND_MARGIN_TOP
        margin_bottom=_BAND_MARGIN_BOTTOM,  # explicit - see _BAND_BODY
        y_axis=False,  # not None - see module docstring
        x_axis="top",  # this section's only turn reference
        x_label="turn",
        x_domain=turn_xlim(n_turns),
        margin_left=_BAND_MARGIN_LEFT,
        margin_right=_BAND_MARGIN_RIGHT,
        opacity_scale="identity",  # the alpha column carries authored
        # values (a confidence is its opacity); Plot would rescale them
        color_scale=_IDENTITY_SCALE,  # without this, Plot auto-binds the
        # per-row "color" column to its own categorical scale, silently
        # substituting Observable's palette for `colors.py`'s
    )
    components = [hconcat(filter_select, conf_checkbox), band]
    # height is computed after the strip block below: the strip's
    # per-member tooltip rows can make it the section's tallest, so the
    # floor takes the max of the two charts' row counts
    strip_tip_rows = 0
    member_assign: dict = {}

    # the agreement strip, per-turn (see docstring "Two plots, not one").
    # Its margins match the band's so its per-turn cells line up with the
    # band's turn columns directly above it.
    if has_judge_agreement(per_turn):
        # every turn gets a cell. A judged turn is teal
        # (`_AGREEMENT_TEAL`, darker = higher agreement); a turn the
        # judge never voted on is pale grey, its tooltip naming the basis
        # it got its label by instead.
        judged = per_turn.judge_agreement.notna()
        a = per_turn.judge_agreement.astype(float)
        basis = per_turn.basis.astype("string").fillna("")
        # a turn can be judged and still carry no agreement: fewer than
        # two members voted, or a verifier overturn re-stitched the
        # phase and superseded the votes. The cell stays grey (the
        # strip encodes vote agreement, and there is none) but names
        # the actual reason rather than claiming the turn was not
        # judged - or blaming a "single voter" for a verifier's doing.
        label_source = per_turn.label_source.astype("string")
        no_agreement_text = pd.Series(
            [
                "n/a (verifier re-label)"
                if src == "verifier"
                else ("n/a (single voter)" if b == "judged" else "not judged")
                for b, src in zip(basis, label_source, strict=True)
            ],
            index=per_turn.index,
        )
        # one "member votes" tooltip row joining every cohort member's
        # own per-turn vote + stated confidence (`phase_turn_votes_df`) -
        # a single wrapping row, not one row per member, so a bigger
        # cohort widens the row instead of raising every chart's
        # tooltip-height floor. Absent on solo scans.
        member_channels: dict[str, str] = {}
        if len(turn_votes):
            members = sorted(
                {
                    (str(m), int(r))
                    for m, r in zip(turn_votes.model, turn_votes.roll, strict=True)
                }
            )
            multi = multi_roll_models(turn_votes)
            # a k-roll run has one model: "roll-N" alone says everything
            # (the summary line above the chart names the model)
            one_model = len({m for m, _ in members}) == 1 and len(members) > 1
            votes_of: dict[int, list[str]] = {}
            for model, roll in members:
                key = (
                    f"roll-{roll}" if one_model else member_display(model, roll, multi)
                )
                sub = turn_votes[
                    (turn_votes.model == model) & (turn_votes.roll == roll)
                ]
                for turn, phase, conf, mb in zip(
                    sub.turn, sub.phase, sub.confidence, sub.basis, strict=True
                ):
                    if pd.notna(phase):
                        vote = f"{phase} ({conf:.2f})" if pd.notna(conf) else str(phase)
                    else:
                        # asked, produced no vote - name the reason
                        vote = MEMBER_NO_VOTE.get(
                            str(mb), str(mb) if pd.notna(mb) else "no judgement"
                        )
                    votes_of.setdefault(int(turn), []).append(f"{key}: {vote}")
            member_assign["member_votes"] = per_turn.turn.map(
                {t: " · ".join(v) for t, v in votes_of.items()}
            )
            member_channels["member votes"] = "member_votes"
        # the number carries its provenance: "(mean)" when a vote
        # decided (modal-side mean), "(verifier)" after an overturn,
        # bare for a single judge
        confidence_text = pd.Series(
            [
                (
                    (f"{c:.2f} ± {pm:.2f}" if pd.notna(pm) and pm > 0 else f"{c:.2f}")
                    + CONFIDENCE_QUALIFIER.get(str(src), "")
                )
                if pd.notna(c)
                else None
                for c, pm, src in zip(
                    per_turn.confidence.astype(float),
                    per_turn.confidence_pm.astype(float),
                    label_source,
                    strict=True,
                )
            ],
            index=per_turn.index,
        )
        total_members = (
            len(
                {
                    (str(m), int(r))
                    for m, r in zip(turn_votes.model, turn_votes.roll, strict=True)
                }
            )
            if len(turn_votes)
            else 0
        )
        agreement_display = pd.Series(
            [
                (
                    f"{v:.2f} ({int(nv)} of {total_members} voted)"
                    if total_members and pd.notna(nv)
                    else f"{v:.2f}"
                )
                if pd.notna(v)
                else None
                for v, nv in zip(a, per_turn.n_voting, strict=True)
            ],
            index=per_turn.index,
        )
        cells = per_turn.assign(
            x1=per_turn.turn - 0.5,
            x2=per_turn.turn + 0.5,
            y1=0,
            y2=1,
            fill=judged.map({True: _AGREEMENT_TEAL, False: _UNJUDGED_GREY}),
            opacity=(0.12 + 0.68 * a).where(judged, 0.5),
            phase_label_text=per_turn.phase,
            agreement_text=agreement_display.where(judged, no_agreement_text),
            confidence_text=confidence_text,
            # "(inherited)" on non-judged turns: the vote decided the
            # phase label the turn inherited, not this turn itself
            label_source_text=label_source.where(
                basis == "judged",
                label_source.map(lambda s: f"{s} (inherited)", na_action="ignore"),
            ),
            basis_text=basis.map(_STRIP_BASIS_WHY).fillna(basis),
            **member_assign,
        ).sort_values("turn")
        strip_channels = {
            "turn": "turn",
            # "phase label", never "label" - the module docstring's
            # Plot-option collision (blanks the whole chart)
            "phase label": "phase_label_text",
            "judge agreement": "agreement_text",
            "confidence": "confidence_text",
            **member_channels,
            # which regime decided the label - single_judge /
            # majority_vote / verifier
            "label source": "label_source_text",
            "label basis": "basis_text",
        }
        strip_tip_rows = _tip_rows(strip_channels)
        columns = [
            "turn",
            "x1",
            "x2",
            "y1",
            "y2",
            "fill",
            "opacity",
            "phase_label_text",
            "agreement_text",
            "confidence_text",
            "label_source_text",
            "basis_text",
            *member_assign,
        ]
        strip_data = Data.from_dataframe(cells[columns])
        strip = plot(
            _margin_label("agreement"),
            rect(
                strip_data,
                x1="x1",
                x2="x2",
                y1="y1",
                y2="y2",
                fill="fill",
                opacity="opacity",
                tip=TipOptions(pointer="x"),
                channels=strip_channels,
                # channels, not a bare tip=True: the automatic
                # per-channel dump would show this chart's internal
                # x1/x2/y1/y2/opacity columns
            ),
            width=width,
            height=_STRIP_HEIGHT,
            margin_top=_STRIP_MARGIN,  # explicit - on a strip this thin
            # Plot's default margins would eat the whole body
            margin_bottom=_STRIP_MARGIN,
            y_axis=False,  # not None - see module docstring
            x_axis=False,  # not None - see module docstring
            x_domain=turn_xlim(n_turns),
            margin_left=_BAND_MARGIN_LEFT,  # see the band's own margin_left
            margin_right=_BAND_MARGIN_RIGHT,
            color_scale=_IDENTITY_SCALE,  # the fill column carries literal
            # hexes; Plot would map them through a categorical scheme
            opacity_scale="identity",  # Plot otherwise fits its own
            # linear scale to the observed range - on a solo/k-roll scan
            # every agreement is uniformly 1.0, a degenerate single-point
            # domain Plot renders at full opacity (solid black) rather
            # than the authored 0.8
        )
        components.append(strip)

    # The taller of this section's two tooltips floors the iframe. The
    # strip's rows are single-line values, so they get the single-line
    # allowance rather than `_TIP_ROW_PX`'s two-line budget; its one
    # wrapping row is "label basis", ~3 lines at its longest phrase.
    strip_floor = 0
    if strip_tip_rows:
        votes_px = 0
        if "member_votes" in member_assign:
            longest = member_assign["member_votes"].astype("string").str.len().max()
            votes_px = wrap_row_px(int(longest)) if pd.notna(longest) else 0
        strip_floor = _tip_floor(
            strip_tip_rows,
            _TIP_SHORT_ROW_PX + 2 * _TIP_EXTRA_LINE_PX,
            second_tall_row_px=votes_px,
        )
    height = _section_height(
        _BAND_BODY + _BAND_MARGIN_TOP,
        tip_rows=_tip_rows(band_channels),
        widget_rows=1,
        floor=max(tip_fit_height(_tip_rows(band_channels)), strip_floor),
    )
    if strip_tip_rows:
        height += _STRIP_HEIGHT
    return components, height


def phase_runs(
    per_turn: pd.DataFrame, known_phases: set
) -> list[tuple[int, int, int | None, str | None]]:
    """Contiguous runs of one transcript's dense per-turn phase
    assignment: ``(start_turn, end_turn, phase_index, unjudged_basis)``
    per run, in turn order. ``unjudged_basis`` is ``None`` for a run of
    confidently-judged turns and the run's own ``basis`` string
    otherwise; ``phase_index`` is ``None`` when the turn owns no phase
    that ``known_phases`` resolves.

    This is `phase_band`'s chunk geometry, derived from ``phase_turns``
    rather than the ``phases`` frame's own ``turn_start``/``turn_end``
    because those declared ranges do not tile the turn axis: on a real
    scan, turns fall inside no phase's declared range while the dense map
    does assign them a phase. Drawing one rect per declared range leaves
    those turns painted by nothing - white gaps at turns the pipeline
    does have an answer for.

    A run breaks on a different phase index, a change in
    judged-vs-unjudged (or a different unjudged ``basis``, so a run's
    tooltip can name one), or a gap in the turn column - a hole there is
    a turn with no per-turn record at all, the one thing this band leaves
    blank rather than painting over from either side. Consecutive turns
    of one phase whose ``basis`` differs only among judged kinds merge
    into one run, since nothing in the tooltip distinguishes them.

    One deliberate exception, on malformed input only: a dangling phase
    index (naming no row of ``phases``) is reported with the turn's own
    raw basis, so two adjacent dangling turns whose bases differ come
    back as two runs even though `phase_band` renders them identically.
    Staying a faithful report of the per-turn record is worth an
    invisible seam in a case that only arises when the upstream frames
    disagree with each other.
    """
    runs: list[list] = []
    previous_key = None
    previous_turn: int | None = None
    for row in per_turn.itertuples():
        turn = int(str(row.turn))
        basis = str(row.basis)
        index = None if pd.isna(row.phase_index) else int(str(row.phase_index))
        # "unjudged" is the union of two facts, both of which forbid
        # painting the turn a phase's colour: the judgement was
        # refused/absent, or there is no phase to attribute it to
        unjudged = (
            basis in _UNJUDGED_BASES or index is None or index not in known_phases
        )
        key = (basis if unjudged else None, index)
        contiguous = previous_turn is not None and turn == previous_turn + 1
        if contiguous and key == previous_key:
            runs[-1][1] = turn
        else:
            runs.append([turn, turn, index, basis if unjudged else None])
        previous_key, previous_turn = key, turn
    return [(run[0], run[1], run[2], run[3]) for run in runs]


def has_judge_agreement(phase_turns: pd.DataFrame) -> bool:
    """Whether ``phase_turns`` carries any non-null per-turn
    ``judge_agreement`` value. `phase_band` gates its agreement-strip
    plot on this internally; exported so `sections.py` can gate the
    strip's caption on the same check rather than restating it.
    """
    return bool(phase_turns.judge_agreement.notna().any())


# The phase band's left/right margins, shared by the agreement strip
# and the interventions chart so their turn columns line up.
#
# 72 on the left fits the row labels (`_margin_label`): "agreement" at
# 11px, right-aligned 6px off the frame, leaves ~10px spare and clears
# the leftmost tick's text. 16 on the right, not less: Plot centres a tick's
# text on its tick position rather than right-aligning it, so the last
# tick can extend past the domain's right edge by half its own width -
# measured on a 6-turn/1000px fixture, 8px overflowed the svg by
# 0.46px. The risk is specific to short transcripts; at real scale the
# last "nice" tick lands well short of the padded domain edge.
_BAND_MARGIN_LEFT = 72
_BAND_MARGIN_RIGHT = 16

_MARGIN_LABEL_FILL = "#6c757d"  # the page's muted text colour


def _margin_label(name: str) -> Mark:
    """A row's name in the left margin `_BAND_MARGIN_LEFT` reserves:
    right-aligned 6px off the frame's left edge. No data source and a
    literal fill (the plot's colour scale is identity); no tip, so it
    stays outside the one-channel-key-set rule (module docstring)."""
    return text(
        text=[name],
        frame_anchor="left",
        styles=TextStyles(text_anchor="end", font_size=11),
        dx=-6,
        fill=_MARGIN_LABEL_FILL,
    )


# Room for the `x_axis="top"` tick numbers *and* the axis's own "turn"
# text label - two separate rows. Plot pins a top axis's label near the
# svg's absolute top edge regardless of `margin_top`, while the tick row
# tracks `margin_top`, so the two only clear each other once the ticks
# are pushed below the label's fixed bottom edge: at 24 they overlap by
# ~7px in a real browser, at 40 they clear by ~9px. Fixed property of
# the constant, unaffected by transcript length. The band's `height`
# adds this on top of its body, so the body's size is unaffected.
_BAND_MARGIN_TOP = 40


# The phase band's body, and the agreement strip's. Both margins are
# explicit because Plot reserves ~30px of bottom margin even with no
# bottom axis to label - a 70px band otherwise renders 40px of body.
# The strip is thinner than Plot's default margins put together, so
# without explicit ones its body would be clamped to nothing.
_BAND_BODY = 70
_BAND_MARGIN_BOTTOM = 4
_STRIP_HEIGHT = 24
_STRIP_MARGIN = 1


# The agreement strip's tooltip line per label ``basis``. Phrases follow
# the basis definitions in `transect.scanners.phases`' scan docstring; a new
# basis value falls through unmapped rather than lying.
_STRIP_BASIS_WHY = {
    "judged": "judged: reasoning turn scored by the judge(s)",
    "filled": "not judged: inherits the previous label",
    "attributed": (
        "not judged: content-free tool-only or failed turn; takes the surrounding label"
    ),
    "refusal": "not judged: the judge refused",
    "no_answer": "not judged: no valid judge answer",
    "missing_turn": "not judged: left uncovered by the judged chunks",
}


# The phase-filter select's "clear" menu item, bound to the empty-string
# value that actually clears the Selection (see `phase_band`). Named
# parenthetically so it reads as the "no filter" state alongside real
# phase labels rather than one more label choice.
_PHASE_FILTER_CLEAR_LABEL = "(all phases)"


def _phase_anchor(phase_index, span_of: dict, card_id_prefix: str) -> str | None:
    """One band row's phase-card link target:
    ``#<card_id_prefix>-<phase_index>`` (transcript-scoped - see
    `phase_band` for the collision it avoids), or ``None`` when
    ``phase_index`` doesn't resolve to a real phase. Reads the same
    ``span_of`` lookup `phase_band` uses for its
    ``reasoning_turns_range`` cell, so the tooltip and the link cannot
    disagree about which cells have a real phase to point at.
    ``phase_index in span_of`` is False for a NaN index (float NaN never
    equality-matches a dict key), so a row with no owning phase gets
    ``None`` here the same way that cell does.
    """
    if phase_index in span_of:
        return f"#{card_id_prefix}-{int(phase_index)}"
    return None


def has_derived_token_views(one: pd.DataFrame) -> bool:
    """Whether ``one`` carries the derived token-view columns the
    three-measure token chart needs. Absent them, `token_stack` falls
    back to a single raw ``output_tokens`` measure - exported so
    `sections.token_intro` picks the matching prose off the same check.
    """
    # value-based, not column-presence: the frames contract guarantees
    # the columns exist, so all-NA views are the real fallback condition
    return bool(one[list(_TOKEN_SOURCE_COLUMNS)].notna().any().any())


def token_measures_coincide(one: pd.DataFrame) -> bool:
    """Whether ``turn_total`` equals ``new_work`` on every turn carrying
    both, as on any source with no cache reads or writes (both reduce to
    input + output). False with no comparable turn. Exported for
    `sections.token_intro`."""
    both = one[["turn_total", "new_work"]].dropna()
    if not len(both):
        return False
    return bool(
        (both.turn_total.astype("int64") == both.new_work.astype("int64")).all()
    )


def draws_threshold(one: pd.DataFrame, compaction_threshold: int | None) -> bool:
    """Whether `token_stack` draws a threshold rule for this transcript:
    an absolute token count and a context series to draw it against.
    Exported so the section's legend reads the same decision."""
    return compaction_threshold is not None and bool(one.context.notna().any())


def token_stack(
    one: pd.DataFrame,
    flushes: pd.DataFrame | None,
    compaction_threshold: int | None = None,
) -> tuple[list[Component], int]:
    """Token telemetry for one transcript: a continuous-x bar chart (up
    to three selectable measures plus a linear/log scale toggle) and,
    when the derived columns support it, a second always-visible
    context-window step chart. Both carry real y-axes.

    With the derived token-view columns present (`has_derived_token_views`)
    this returns ``[measure_selector, scale_selector, bars_chart,
    context_chart]``; two `radio_group` inputs bind two `Param`s (measure
    default ``"turn_total"``, scale default ``"linear"``).
    ``cum_billable`` is a render-time cumsum over the derived
    ``billable`` column, not a stored frame column. Absent the derived
    columns, returns ``[scale_selector, bars_chart]`` over the single raw
    ``output_tokens`` measure - still scale-toggleable, since linear/log
    is orthogonal to which measure exists.

    Both charts carry `nearest_x` and their own hover line, each off its
    own `Selection` (`_hover_selection` once per plot, never shared), so
    hovering one never moves the other's line. The context chart's `line`
    mark takes a plain ``x="turn"`` channel, which is what makes it a
    legal `nearest_x` target; it resolves its own tooltip regardless of
    the interactor.

    Everything this function returns lands in one iframe: a `Param` bound
    to a radio cannot cross an iframe boundary, so the radios have to
    share a document with the bars chart, and the "one channel-key set
    per plot" rule is scoped to a `plot()` call rather than a document,
    so the context chart costs nothing extra there.

    Mark order on the bars chart is load-bearing (module docstring):
    ``rule_x`` (the bars), `nearest_x`, `highlight`, then the hover rule.
    The context chart follows the same rule minus `highlight` (one
    continuous line has no siblings to dim), with the flush rules last -
    they are plain unfiltered `rule_x` marks and would fail the same way
    ahead of `nearest_x`.

    The bars tooltip carries ``turn`` plus *every* measure, not the
    selected one: a channel value must be a plain column name, so a
    tooltip row cannot track the measure Param the way the ``y2`` channel
    does. Three honest rows beat one reactive row the library cannot
    express. Plot formats each numeric row with thousands separators. The
    context chart's tooltip is ``turn`` plus ``context window``.

    `y_scale` on the bars chart binds straight to the `scale` Param - a
    genuine axis-scale change. The context chart is always linear. Under
    log, the axis floor (`_bar_y1_floor_expr`) is the current measure's
    own smallest real value, so the domain spans the range the data
    occupies instead of stretching across empty decades below it.

    An explicit `x_label` is pinned on both charts: the flush rule's
    `sql()` x-expression would otherwise leak into the x-scale's
    auto-inferred label.

    Flush events draw on both charts as a dashed red rule at each flush
    turn, purely visual (``pointer_events="none"``), offset by the shared
    t-0.5 convention. The flush *detail* rides the bars chart's own tip
    (`_flush_column`) since a plot carries one tip-bearing mark; the
    context chart's rule is unlabelled. Interventions get their own chart
    (`interventions_chart`), never these.

    A recorded absolute compaction threshold adds a dotted horizontal
    rule to the context chart and a checkbox above it. The checkbox
    filters the rule's data with a Selection, and the y-scale follows:
    hiding a threshold far above the curve gives the curve the chart back.
    No threshold control is emitted without context data or an absolute
    token count; cumulative spend and output do not measure input context.
    """
    per_turn = one.sort_values("turn")
    if not len(per_turn):
        return [], 0
    n_turns = int(per_turn.turn.max()) + 1
    width = chart_width(n_turns)
    bar_width = _bar_stroke_width(width, n_turns)
    scale = Param("linear")
    scale_selector = radio_group(
        options=["linear", "log"], target=scale, label="Scale:"
    )
    hover = _hover_selection()  # this plot's own

    flush_data = None
    if flushes is not None and len(flushes):
        flush_data = Data.from_dataframe(flushes[["turn"]])

    def flush_rules() -> list[Mark]:
        # purely visual (pointer_events="none"); on the bars chart the
        # bar's own tooltip carries the flush detail (_flush_column)
        if flush_data is None:
            return []
        return [_offset_rule_x(flush_data, "#c44e52", "4,3", pointer_events="none")]

    def build_bars(data: Data, raw_expr: str, channels: dict[str, str]) -> Component:
        return plot(
            rule_x(
                data,
                x="turn",
                y1=_bar_y1_floor_expr(raw_expr),  # never reactive on
                # `scale` - see _bar_y1_floor_expr
                y2=_bar_value_expr(raw_expr, scale),
                stroke="#4c78a8",
                stroke_width=bar_width,
                tip=TipOptions(pointer="x"),  # not tip="x" - see docstring
                channels=channels,
            ),
            nearest_x(target=hover, channels=["x"]),  # must follow the
            # bars mark immediately - interactors bind to the previous mark
            highlight(by=hover, opacity=0.35),
            _hover_rule(
                Data.from_dataframe(pd.DataFrame({"turn": range(max(n_turns, 1))})),
                hover,
            ),  # after nearest_x, never before it - see docstring
            *flush_rules(),
            width=width,
            height=_TOKEN_HEIGHT,
            y_label="tokens",
            y_scale=scale,
            margin_left=_TOKEN_MARGIN_LEFT,  # shared with the context chart
            margin_top=_TOKEN_MARGIN_TOP,  # explicit, not auto
            margin_bottom=_TOKEN_MARGIN_BOTTOM,
            x_domain=turn_xlim(n_turns),
            x_label="turn",
        )

    if not has_derived_token_views(per_turn):
        raw = per_turn[["turn", "output_tokens"]].dropna().copy()
        raw["flush"] = _flush_column(raw.turn, flushes)
        raw_channels = {
            "turn": "turn",
            "output tokens": "output_tokens",
            "context flush": "flush",
        }
        chart = build_bars(Data.from_dataframe(raw), "output_tokens", raw_channels)
        # one widget row (this fallback path has no measure radio)
        raw_height = _section_height(
            _TOKEN_HEIGHT, tip_rows=_tip_rows(raw_channels), widget_rows=1
        )
        return [scale_selector, chart], raw_height

    wide = per_turn[["turn", "turn_total", "new_work", "billable", "context"]].copy()
    wide["cum_billable"] = wide.billable.cumsum()
    wide["flush"] = _flush_column(wide.turn, flushes)

    measure = Param("turn_total")
    measure_selector = radio_group(
        options={label: value for value, label in _TOKEN_MEASURES.items()},
        target=measure,
        label="Measure:",
    )
    cases = " ".join(f"when '{col}' then {col}" for col in _TOKEN_MEASURES)
    data = Data.from_dataframe(wide)
    # row order follows this dict: turn, the three measures in menu
    # order, then the flush row (present only on a flushing turn)
    bars_channels = (
        {"turn": "turn"}
        | {label: column for column, label in _TOKEN_MEASURES.items()}
        | {"context flush": "flush"}
    )
    bars_chart = build_bars(data, f"case {measure} {cases} end", bars_channels)

    context_data = Data.from_dataframe(wide[["turn", "context"]].dropna())
    context_hover = _hover_selection()  # this plot's own, never shared
    # with the bars chart's `hover` above
    threshold_controls: list[Component] = []
    threshold_marks: list[Mark] = []
    if compaction_threshold is not None and draws_threshold(one, compaction_threshold):
        threshold_data = Data.from_dataframe(
            pd.DataFrame({"threshold": [compaction_threshold], "visible": ["show"]})
        )
        threshold_selection = Selection.single(cross=False)
        threshold_controls.append(
            checkbox(
                data=threshold_data,
                label=f"Compaction threshold: {compaction_threshold:,} tokens (dotted)",
                target=threshold_selection,
                field="visible",
                checked=True,
                values=("show", "hide"),
            )
        )
        threshold_marks.append(
            rule_y(
                threshold_data,
                filter_by=threshold_selection,
                y="threshold",
                stroke="#9467bd",
                stroke_dasharray="2,3",
                stroke_width=1.5,
                pointer_events="none",
            )
        )
    context_chart = plot(
        line(
            context_data,
            x="turn",  # a plain column reference, not sql() - what makes
            # this a legal nearest_x target
            y="context",
            curve="step-after",
            stroke="#4c78a8",
            tip=TipOptions(pointer="x"),  # not tip="x" - see _event_hit_rect
            channels={"turn": "turn", "context window": "context"},
        ),
        nearest_x(target=context_hover, channels=["x"]),  # must follow
        # the line mark immediately - see the module docstring
        _hover_rule(
            Data.from_dataframe(pd.DataFrame({"turn": range(max(n_turns, 1))})),
            context_hover,
        ),  # after nearest_x, never before it - see docstring
        *flush_rules(),
        *threshold_marks,
        width=width,
        height=_CONTEXT_HEIGHT,
        y_label="context",
        margin_left=_TOKEN_MARGIN_LEFT,  # shared with the bars chart above
        margin_top=_CONTEXT_MARGIN_TOP,
        margin_bottom=_CONTEXT_MARGIN_BOTTOM,
        x_domain=turn_xlim(n_turns),
        x_label="turn",
    )
    # two widget rows (measure + scale); the height floor counts the bars
    # chart's tooltip, the taller of this section's two
    derived_height = _section_height(
        _TOKEN_HEIGHT + _CONTEXT_HEIGHT,
        tip_rows=_tip_rows(bars_channels),
        widget_rows=2 + len(threshold_controls),
    )
    return (
        [
            measure_selector,
            scale_selector,
            bars_chart,
            *threshold_controls,
            context_chart,
        ],
        derived_height,
    )


# measure column -> radio-button label, in the order the radio buttons
# render; turn_total first (the default measure). The context window is
# not a measure here - it has its own always-visible chart.
_TOKEN_MEASURES = {
    "turn_total": "per-turn total",
    "new_work": "per-turn new work",
    "cum_billable": "cumulative (excluding cache reads)",
}


# the source columns token_stack's three-measure chart needs: new_work/
# turn_total/context are read directly; billable is the source for the
# render-time cum_billable cumsum (not itself a menu measure).
_TOKEN_SOURCE_COLUMNS = {"new_work", "turn_total", "context", "billable"}


# The bars and context charts' shared left margin. Shared between the
# two deliberately: they stack in one section and read as one turn axis,
# so a left-edge misalignment would read as the charts disagreeing about
# where a turn sits. Sized in a real browser against the widest tick
# either axis ever prints - `cum_billable` on a *linear* scale, an
# 8-digit label (log uses Plot's shorter SI-suffix formatting, so it is
# never the binding constraint). 90px clears it by ~15px; 70px left only
# ~2.5px. `cum_billable`'s max grows with transcript length, so a much
# longer transcript than measured could still outgrow it.
_TOKEN_MARGIN_LEFT = 90


# Explicit margins, not Plot's auto-computation: a visible y-axis's own
# text label (`"↑ tokens"`) is drawn near the svg's absolute top edge -
# the same mechanism `_BAND_MARGIN_TOP` documents. Verified in a real
# browser; re-measure there if a label ever clips.
_TOKEN_HEIGHT = 210
_TOKEN_MARGIN_TOP = 30
_TOKEN_MARGIN_BOTTOM = 30


# The bars' y1 floor, and two constraints on how it may be computed
# (both browser-verified against inspect_viz 0.4.1):
#
# - A literal y1=0 renders under a linear y_scale but vanishes under log
#   with zero DOM nodes and zero console output - log(0) is undefined,
#   and Plot silently drops a mark whose bound channel cannot be
#   projected. No bar to literal zero can ever be log-honest.
# - y1 must never be reactive on the same `scale` Param that drives
#   `y_scale`. The two paths race: `y_scale` updates synchronously,
#   while a `sql()` channel is an async DuckDB requery - so one render
#   pass has the new (log) scale with the old (0) channel data, and that
#   transient log(0) blanks the mark permanently. Deterministic, not
#   timing-dependent.
#
# `_bar_y1_floor_expr` therefore reads the data's own minimum positive
# value for the selected measure: reactive on the measure Param only,
# never on `scale`, and always positive (this fallback covers the case
# where no positive value exists - token counts are integers >= 1).
_BAR_Y1_FALLBACK_FLOOR = "1"


# The context-window step chart's height budget: markedly shorter than
# the bars (secondary view, one step line, no radios above it - the
# space-efficiency alternative to overlaying it on the bars, which
# would need a second y-scale), margins explicit for the same reason
# as `_TOKEN_MARGIN_TOP`.
_CONTEXT_HEIGHT = 110
_CONTEXT_MARGIN_TOP = 30
_CONTEXT_MARGIN_BOTTOM = 30


def _hover_selection() -> Selection:
    """One chart's own hover selection. `empty=True`: before any hover
    there is no clause, so the hover rule filtered by it draws nothing.

    Local by construction: `token_stack` is the only builder that makes
    one, and calls this once per plot rather than sharing a Selection
    between its two.
    """
    return Selection.single(empty=True)


def _hover_rule(turns: Data, hover: Selection) -> Mark:
    """The red vertical line marking the hovered turn, drawn in the same
    plot whose `nearest_x` resolves it (`token_stack` is the one caller).
    """
    return rule_x(turns, x="turn", filter_by=hover, stroke="#c44e52", stroke_width=1.5)


def _offset_rule_x(data: Data, stroke: str, dasharray: str | None, **options) -> Mark:
    """One rule_x at t - 0.5, not t: events at turn t sit between turns
    t-1 and t on the shared axis. Factored out so both callers -
    `interventions_chart` and `token_stack` - draw the offset the same
    way.

    ``dasharray=None`` omits the option entirely rather than passing a
    literal ``None`` through (the marshaller drops None kwargs anyway,
    but a solid stroke is what the interventions chart wants);
    `token_stack`'s flush rules pass a real dasharray string.
    """
    if dasharray:
        options["stroke_dasharray"] = dasharray
    return rule_x(data, x=sql("turn - 0.5"), stroke=stroke, **options)


def _flush_detail(f) -> str:
    """One flush's tooltip cell: type/source plus the token delta when
    both endpoints are known - never fabricated (the same rule
    `sections.flush_line` follows: a missing endpoint means the amount
    is omitted, not guessed). ``(inferred)`` names the case where the
    export omitted ``tokens_after`` and the next real model event's
    reading was substituted. The turn number is its own tooltip row.
    """
    detail = f"{f.type}/{f.source}"
    if pd.notna(f.tokens_before) and pd.notna(f.tokens_after):
        detail += f" · {int(f.tokens_before):,} → {int(f.tokens_after):,}"
        if f.tokens_after_inferred:
            detail += " (inferred)"
    return detail


def _flush_column(turns: pd.Series, flushes: pd.DataFrame | None) -> list[str | None]:
    """A per-turn ``"context flush"`` tooltip column: `_flush_detail` on
    a turn that carries a flush, ``None`` everywhere else.

    The flush detail rides on the per-turn bar mark rather than a hit
    rect of its own because a plot carries one channel-key set (module
    docstring), and a ``None`` cell emits no tooltip row at all - so
    the flush row appears only on turns that actually flushed.
    """
    if flushes is None or not len(flushes):
        return [None] * len(turns)
    detail_of = {int(str(f.turn)): _flush_detail(f) for f in flushes.itertuples()}
    return [detail_of.get(int(str(turn))) for turn in turns]


def _bar_value_expr(raw_expr: str, scale: Param) -> Transform:
    """Wrap a raw SQL value expression with the log-scale non-positive
    guard: under log scale a zero or negative resolved value becomes
    null, so that turn's bar is absent rather than blanking the mark
    (see `_BAR_Y1_FALLBACK_FLOOR`). Under linear the guard is a no-op;
    a null source value stays null either way - a per-cell absence, not
    a per-row `dropna()`, so a turn missing one measure still renders
    under the others.

    This is the bar's ``y2``; `_bar_y1_floor_expr` is the other end.
    """
    return sql(
        f"case when '{scale}' = 'log' and ({raw_expr}) <= 0 then NULL "
        f"else ({raw_expr}) end",
        label="tokens",
    )


def _bar_y1_floor_expr(raw_expr: str) -> Transform:
    """The bar's ``y1`` - the data's own minimum positive value for
    whichever measure ``raw_expr`` resolves, broadcast to every row via
    a window aggregate (``OVER ()``, no ``GROUP BY`` needed).

    Reactive on whatever Param(s) ``raw_expr`` references (the measure
    Param, via `token_stack`'s per-measure ``CASE``) - never on
    `scale`, which is the constraint `_BAR_Y1_FALLBACK_FLOOR` records:
    this expression's SQL text never mentions ``scale``, so a scale
    click triggers no requery for it and there is nothing to race. One
    mechanism serves both scales; at real per-turn token magnitudes the
    gap between this floor and true 0 is imperceptible on a linear axis.

    A cosmetic consequence, not a bug: the turn carrying exactly the
    minimum gets ``y1 == y2`` and renders a zero-height bar. Its tooltip
    still reports the true value.
    """
    return sql(
        f"coalesce(min(case when ({raw_expr}) > 0 then ({raw_expr}) end) "
        f"over (), {_BAR_Y1_FALLBACK_FLOOR})"
    )


def _bar_stroke_width(width: int, n_turns: int) -> float:
    """One bar's pixel width: up to 6px when turns have room to spare,
    shrinking (never below 1.5px) as turns pack tighter. A fixed 6px
    smears into a solid block on a long transcript, where `chart_width`
    leaves only ~1.6px per turn. 0.6 of the per-turn pixel budget keeps
    a visible gap between adjacent bars.
    """
    return min(6.0, max(1.5, width / max(n_turns, 1) * 0.6))


def _event_hit_rect(
    frame: pd.DataFrame, y_lo: float, y_hi: float, channels: dict[str, str]
) -> Mark:
    """A transparent, one-turn-wide rect per event row, layered under a
    thin visual rule mark to give it a real hover target.

    A hairline `rule_x` has almost no area a pointer can land on -
    browser-verified: repeated exact-pixel-centre hovers on a dashed
    rule missed far more often than they hit. This rect spans one full
    turn (the half-turn placement convention every event marker in this
    module uses) and the caller's y-range, so a hover anywhere in that
    turn's column finds it. It carries the tooltip; the caller's visible
    rule mark stays tip-free - one mark draws, the other answers hover,
    and this stays the plot's only tip-bearing mark.
    """
    frame = frame.copy()
    frame["x1"] = frame.turn - 0.5
    frame["x2"] = frame.turn + 0.5
    frame["y1"] = y_lo
    frame["y2"] = y_hi
    columns = ["x1", "x2", "y1", "y2", *dict.fromkeys(channels.values())]
    data = Data.from_dataframe(frame[columns])
    return rect(
        data,
        x1="x1",
        x2="x2",
        y1="y1",
        y2="y2",
        opacity=0,
        tip=TipOptions(pointer="x"),  # resolve by turn, not 2-D distance.
        # Not the bare string `tip="x"` - Mosaic reads a string-valued
        # mark option as a column reference and the chart dies with
        # `Referenced column "x" not found`.
        channels=channels,
    )


def interventions_chart(act: pd.DataFrame, n_turns: int) -> tuple[Component, int]:
    """Human interventions: a dedicated timeline of their own.

    One navy solid `rule_x` per intervention turn, offset by the shared
    t-0.5 convention and purely visual (`pointer_events="none"`); an
    `_event_hit_rect` layered under it carries the hover tooltip (turn,
    channel, 80-char previews of the question asked and the content,
    the outcome - the last three null on rows without them), so
    hovering anywhere in that turn's column names its source.

    ``act`` is expected non-empty - the orchestrator guards, this
    builder does not re-guard.

    No `nearest_x` and no hover line, deliberately. The carrier this
    chart would need is sparse by construction (only its own event
    turns), so it was never a general hover source; and hosting
    `nearest_x` while drawing a rule filtered by the same selection
    breaks the tooltip outright - the plot becomes a client of the
    selection its own hover updates, so every hover re-renders it,
    `Nearest` re-attaches to the replaced svg, and the in-flight hover is
    lost.

    Axes: `y_axis=False` (never `None` - module docstring; row position
    carries no meaning, every rule spans the full body) and a visible
    bottom x-axis labelled "turn". Without it a lone strip of navy
    lines leaves a reader unable to place an intervention in the run
    without hovering it.

    Height: see `_INTERVENTION_HEIGHT` - the body (height minus the two
    explicit margins) is the number worth reading, not the total.
    """
    width = chart_width(n_turns)
    frame = act[["turn", "channel", "outcome"]].copy()
    # prompt/outcome are null cells on human-initiated rows (no row pops)
    frame["prompt_preview"] = [_preview(i.prompt) for i in act.itertuples()]
    frame["preview"] = [_preview(i.content) for i in act.itertuples()]
    data = Data.from_dataframe(frame[["turn"]])
    hit_channels = {
        "turn": "turn",
        "channel": "channel",
        "asked": "prompt_preview",
        "content": "preview",
        "outcome": "outcome",
    }
    component = plot(
        _event_hit_rect(
            frame,
            -1,
            1,
            hit_channels,
        ),  # the plot's only tip-bearing mark - see _event_hit_rect
        _offset_rule_x(data, "#0E2841", None, pointer_events="none", stroke_width=2),
        width=width,
        height=_INTERVENTION_HEIGHT,
        y_axis=False,  # not None - see module docstring
        y_domain=(-1, 1),
        margin_top=_INTERVENTION_MARGIN_TOP,  # explicit - see
        # _INTERVENTION_HEIGHT
        margin_bottom=_INTERVENTION_MARGIN_BOTTOM,
        x_domain=turn_xlim(n_turns),
        x_label="turn",
        margin_left=_INTERVENTION_MARGIN_LEFT,
        margin_right=_INTERVENTION_MARGIN_RIGHT,
    )
    # no widget row here; the content row is the one that wraps in
    # practice (an 80-character preview at tippy's 350px cap), which is
    # what `tip_fit_height`'s per-row allowance budgets for
    return component, _section_height(_INTERVENTION_HEIGHT, _tip_rows(hit_channels))


# The interventions chart's height budget. The body - height minus these
# two explicit margins - is what the navy event rules get, and is the
# number worth reading: Plot reserves ~20px top / ~30px bottom by default
# even with no axis needing tick-label room, so a 70px total left the
# rules a ~25px sliver. 94/4/30 gives a 60px body - the floor at which
# the navy rules stay legible even when Plot's `max-width: 100%`
# stamp scales a wide chart down.
_INTERVENTION_HEIGHT = 94
_INTERVENTION_MARGIN_TOP = 4
_INTERVENTION_MARGIN_BOTTOM = 30


# Aliases of the band's margins rather than two fresh numbers: this chart
# also suppresses its y-axis and draws nothing into either margin, so all
# either has to do is keep its x-axis's edge ticks clear of the svg's
# edges - exactly what `_BAND_MARGIN_LEFT`/`_BAND_MARGIN_RIGHT` were
# measured for. Sharing the values also lines this chart's inner plot
# rectangle up with the band's, so a turn sits in the same pixel column
# in both - they are turn strips a reader compares down the page.
_INTERVENTION_MARGIN_LEFT = _BAND_MARGIN_LEFT
_INTERVENTION_MARGIN_RIGHT = _BAND_MARGIN_RIGHT


def _preview(value) -> str | None:
    """A tooltip text cell trimmed to ~80 characters - long enough to
    identify the message, short enough to keep the tooltip legible (the
    list under the chart carries the full text). An ellipsis marks an
    actual truncation, never appended to text that already fit; None
    stays None (no tooltip row).
    """
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    text = str(value)
    return text if len(text) <= 80 else text[:80] + "…"


def swimlanes(
    rows: list[tuple],
    yticks: list[float],
    ylabels: list[str],
    end_markers: list[tuple[float, float]],
    n_turns: int,
    colors: dict[str, str],
    titles: list[dict[str, str]],
    tip_fields: tuple[str, ...],
) -> tuple[Component, int]:
    """Sub-agent activity swimlanes: one mark per placed span, packed
    into sub-lane rows by `lanes_layout.pack_lanes`.

    ``rows`` (``(row_y, x_start, width, label, lane_name, boxed)`` per
    span) and ``end_markers`` (``(end_pos, row_y)``, already filtered by
    the caller to spans with a harness-recorded end) come straight off a
    `PackedLanes`. ``titles`` is one dict per span in ``rows``' own
    order, keyed by ``tip_fields`` (the caller's subset of
    `SPAN_TIP_FIELDS`), one tooltip row per field -
    `render._subagent_section` builds it via `sections.span_titles` and
    joins it onto `PackedLanes.span_row`, which `pack_lanes` appends in
    the same per-span iteration as ``rows``, so the two line up
    positionally.

    ``yticks``/``ylabels`` are drawn here as one right-anchored `text`
    mark per label-block, positioned at the turn axis's domain start and
    nudged ``dx`` left into `_SWIMLANE_MARGIN_LEFT`, so they read outside
    the plotted turn range. The Jinja legend line beside the chart
    carries the same label/swatch pairs.

    **Box width, and what it is allowed to mean.** Each row's ``boxed``
    flag comes from `render._span_lanes` (the frame's ``turn_source``
    plus a usable orchestrator clock): a span placed by wall-clock has a
    real extent on the axis and draws as a box; a span the source gave
    no usable timestamps for sits at its spawn turn as a point, and
    draws as a tick.

    - boxed: a proportional box from ``x_start`` over ``width`` (the
      span's wall-clock activity mapped onto the orchestrator turns
      active at the time), floored at `span_min_box_width` so a short span
      is not sub-pixel on a long axis. What that width means is stated
      in the section's how-to-read line and in the box's own ``turns``
      tooltip cell.
    - tick: no box at all - a thin colored tick (`_TICK_STROKE_WIDTH`, a
      `rule_x` bounded to the row's own ``y1``/``y2``) marks only the
      spawn turn. A box of any fixed width would read as an extent the
      data does not have; a tick cannot imply duration.

    The two shapes share one plot, and the "one channel-key set per
    plot" rule (module docstring) allows one tip-bearing mark, so
    neither visible mark carries the tooltip: a single invisible `rect`
    over every row (the box's own extent for a boxed row, a hit
    footprint `span_min_box_width` wide around a tick) answers hover for
    both - the "one mark draws, another answers" idiom of
    `_event_hit_rect`, generalized. Hit-rects may overlap between
    adjacent rows (only the visible ticks must not) - opacity 0 either
    way. ``end_markers`` arrive already filtered to recorded ends.

    Y axis carries no ticks (``y_axis=False``, not ``None`` - module
    docstring): row_y values are arbitrary sub-lane placements with no
    natural tick text, and rendering the axis leaks raw row indices and a
    ``"↓ y1"`` internal-column-name label. The row-label `text` marks
    above are the readable surface. The x-axis stays visible - this
    section's own turn reference.

    Neither `nearest_x` nor a hover line: a plot mixing `rect` interval
    marks with a quantitative interactor crashes in-browser (module
    docstring). Span detail comes from this chart's own tooltip.

    `color_scale="identity"` is load-bearing: the span marks' colour is a
    per-row hex column from this project's own `_label_colors` palette,
    and without identity Plot auto-binds it to its own categorical scale,
    silently substituting Observable's palette - the span colours would
    then disagree with `sections.subagent_notes`' legend chips, which
    read the same `colors` dict directly.

    **Stacked-mark legibility.** Concurrent same-label spans pack into
    adjacent sub-lane rows at one row-pitch apiece and share a fill
    colour by construction, so two differently-timed spans can read as
    one fused shape. Two separate fixes, because they are two separate
    problems:

    - Proportional mode: a white `stroke` around every box
      (`_SWIMLANE_BOX_STROKE`) gives each its own visible edge however
      small the true gap, plus `_SWIMLANE_BOX_HALF_HEIGHT` headroom.
    - Uniform mode: a stroke could not fix it, because the geometry
      genuinely overlaps - every mark renders at the same fixed width
      regardless of its true (often zero-duration) extent, so two spans a
      handful of turns apart overlap rather than merely touching. The fix
      is upstream of anything drawn here: `lanes_layout.pack_lanes` is
      footprint-aware (`min_footprint`) and reserves a span's rendered
      footprint rather than its raw turn extent, and the mark itself is a
      2px tick rather than a box. Pinned by `pack_lanes`' unit tests and
      by a browser test reading real rendered geometry.
    """
    n_rows = int(max(row[0] for row in rows)) + 1 if rows else 1
    marks: list[Mark] = []
    max_lane_chars = 0
    max_votes_chars = 0
    if rows:
        frame = pd.DataFrame(
            rows, columns=["y", "x_start", "width", "label", "lane_name", "boxed"]
        )
        frame["y1"] = frame.y - _SWIMLANE_BOX_HALF_HEIGHT
        frame["y2"] = frame.y + _SWIMLANE_BOX_HALF_HEIGHT
        frame["color"] = frame.label.map(colors)
        # one tooltip row per field. Column names are positional
        # (`tip_0`...) so a field label may contain spaces; `channels`
        # maps label -> column. ``tip_fields`` is the caller's subset of
        # `SPAN_TIP_FIELDS` - it drops "classification" when no
        # classification scan ran, so the tooltip never shows a
        # fabricated "unclassified" for a judge that never existed.
        fields = list(tip_fields)
        for index, field in enumerate(fields):
            frame[f"tip_{index}"] = [title.get(field) for title in titles]
        lane_values = [str(title.get("lane") or "") for title in titles]
        # this chart's worst "lane" and "member votes" tooltip values,
        # for the tip-fit floor (its two wrapping rows)
        max_lane_chars = max((len(v) for v in lane_values), default=0)
        max_votes_chars = max(
            (len(str(title.get("member votes") or "")) for title in titles),
            default=0,
        )
        channels = {field: f"tip_{index}" for index, field in enumerate(fields)}

        frame = span_geometry(frame, n_turns)
        boxed = frame[frame.boxed]
        if len(boxed):
            marks.append(
                rect(
                    Data.from_dataframe(boxed[["x1", "x2", "y1", "y2", "color"]]),
                    x1="x1",
                    x2="x2",
                    y1="y1",
                    y2="y2",
                    fill="color",
                    stroke=_SWIMLANE_BOX_STROKE,
                    stroke_width=_SWIMLANE_BOX_STROKE_WIDTH,
                )
            )
        ticks = frame[~frame.boxed]
        if len(ticks):
            marks.append(
                rule_x(
                    Data.from_dataframe(ticks[["x", "y1", "y2", "color"]]),
                    x="x",
                    y1="y1",
                    y2="y2",
                    stroke="color",  # a per-row hex column - needs the
                    # plot's color_scale="identity" below
                    stroke_width=_TICK_STROKE_WIDTH,
                )
            )
        # the one tip-bearing mark: the box's extent for a boxed row, a
        # hover footprint around the tick otherwise (`span_geometry`)
        marks.append(
            rect(
                Data.from_dataframe(
                    frame[["hx1", "hx2", "y1", "y2", *channels.values()]]
                ),
                x1="hx1",
                x2="hx2",
                y1="y1",
                y2="y2",
                opacity=0,  # hit-target only
                tip=True,  # 2-D pointer: spans stack in rows, so the
                # hovered row matters as much as the turn
                channels=channels,
            )
        )
    if end_markers:
        marker_frame = pd.DataFrame(end_markers, columns=["end_pos", "row_y"])
        markers = Data.from_dataframe(marker_frame)
        marks.append(
            dot(
                markers,
                x="end_pos",
                y="row_y",
                symbol=_END_MARKER_SYMBOL,  # must be a real Plot symbol
                # name - see that constant's own comment
                stroke="#333",
            )
        )
    if yticks:
        # right-anchored into the left margin: x pinned at the turn
        # axis's domain start, nudged further left by dx so the text sits
        # in the margin whitespace rather than over the leftmost marks
        x0, _x1 = turn_xlim(n_turns)
        label_frame = pd.DataFrame(
            {"x": [x0] * len(yticks), "y": yticks, "label": ylabels}
        )
        label_data = Data.from_dataframe(label_frame)
        marks.append(
            text(
                label_data,
                x="x",
                y="y",
                text="label",
                styles=TextStyles(text_anchor="end", font_size=11),
                dx=-8,
            )
        )

    height_px = max(90, 40 + 30 * n_rows)
    component = plot(
        *marks,
        width=chart_width(n_turns),
        height=height_px,
        y_axis=False,  # not None - see module docstring
        y_domain=(n_rows - 0.5, -0.5),
        x_domain=turn_xlim(n_turns),
        x_label="turn",
        margin_left=_SWIMLANE_MARGIN_LEFT,  # also what the row-label
        # text marks above are drawn into
        color_scale=_IDENTITY_SCALE,  # load-bearing - see docstring
    )
    # No widget row here. This chart's tooltip is the report's tallest -
    # `SPAN_TIP_FIELDS` is a fixed row count however few lanes the scan
    # has, so a 1-row chart can pop a tooltip taller than itself and the
    # `_section_height` floor is what covers that. The floor is
    # `_swimlane_tip_fit_height`, not the default `tip_fit_height`, since
    # the "lane" row's wrap height is data-dependent.
    # no spans -> no tip-bearing mark and nothing to fit
    tip_rows = len(tip_fields) if rows else 0
    return component, _section_height(
        height_px,
        tip_rows,
        floor=_swimlane_tip_fit_height(tip_rows, max_lane_chars, max_votes_chars),
    )


def swimlane_min_footprint(n_turns: int) -> float:
    """The minimum horizontal footprint, in turn units, a uniform-mode
    swimlane tick needs for `lanes_layout.pack_lanes`' packing decision:
    the tick's own visual width plus a seam (`_FOOTPRINT_SEAM_PX`),
    converted to turns via this chart's pixel-per-turn rate.

    An approximation, not an exact rendered pixel count (Plot's margins
    make the plotted body narrower than `chart_width`) - acceptable
    because the result only ever widens the footprint a packing decision
    reserves, never the mark actually drawn. What it guarantees, though,
    is unconditional: two spans spawned within about a
    tick-width-plus-seam of each other stack into separate sub-lanes,
    because their ticks would otherwise touch or overlap.

    Callers pass ``0.0`` for proportional-width mode (real recorded span
    ends), which already packs by each span's own observed extent.
    """
    turn_pitch_px = chart_width(n_turns) / n_turns
    tick_width_turns = _TICK_STROKE_WIDTH / turn_pitch_px
    seam_turns = _FOOTPRINT_SEAM_PX / turn_pitch_px
    return tick_width_turns + seam_turns


# The swimlane span tooltip's row labels, in render order. Shared between
# `sections.span_titles` (which fills the cells) and `swimlanes` (which
# binds them as `channels`), so the two cannot drift into disagreeing
# about a field name - a mismatch silently drops that row rather than
# failing.
#
# "classification", not "label": a channel named ``label`` collides with
# Plot's own mark/scale `label` option and blanks the whole chart with
# ``Cannot read properties of undefined``. Channel names containing
# spaces are fine; `_RESERVED_TIP_CHANNELS` is the deny-list every
# `channels=` dict in this module is checked against.
#
# "turns" is the span's first-last observed activity range, in the same
# name and en-dash format `phase_band`'s tooltip uses. On a source that
# records real span ends, `sections.span_titles` appends "(observed
# activity extent)" to this same cell rather than adding a field: the
# field count is the height budget's ceiling
# (`_swimlane_tip_fit_height`); reliability cells (confidence /
# agreement / label source / verifier) are None - and so absent rows -
# where their fact does not apply.
SPAN_TIP_FIELDS = (
    "lane",
    "turns",
    "classification",
    "confidence",
    "agreement",
    "label source",
    "verifier",
    "tool calls",
    "busy",
)


# Tooltip channel names that are also Plot mark/scale options. Using one
# as a `channels=` key blanks the chart. Only ``label`` is confirmed by
# direct repro; the rest are options a plausible row name could collide
# with, listed so a future name gets checked rather than discovered in a
# browser.
_RESERVED_TIP_CHANNELS = frozenset(
    {
        "label",
        "title",
        "text",
        "fill",
        "stroke",
        "opacity",
        "filter",
        "sort",
        "interval",
        "reduce",
        "href",
        "clip",
        "tip",
        "channels",
    }
)


# Swimlanes' left margin: sized to the per-label-block row-name `text`
# marks it draws there (right-anchored, nudged `dx=-8` further left).
# Real-browser measurement, not a guess: this chart's widest observed
# label ("code_implementation (13)") renders ~137px at 11px font, plus
# the 8px nudge, and clears the svg's left edge by ~15px at 160. Raise
# this one number if a longer label ever clips - `text_anchor="end"`
# grows a label leftward, so an over-long one loses its leading
# characters first, keeping the "(N)" suffix visible longest.
_SWIMLANE_MARGIN_LEFT = 160


# The swimlanes "lane" row's own, larger allowance: unlike this chart's
# other `SPAN_TIP_FIELDS` rows (short bounded structured values), a lane
# name is free text up to `lanes_layout.LANE_NAME_MAX_CHARS`, so how
# many lines it wraps to is data-dependent.
#
# Real-browser measurement, not a formula: the value column renders
# ~254.6px wide (tippy caps the popover at 350px, minus the label
# column), so word length decides line count. Short words pack tightly;
# a single very long token overflows its one line rather than wrapping
# tall. The worst packing is ~19-character words: 100px (6 lines) for a
# 121-character value. This adds one wrap-line of headroom over that
# measured peak. A more adversarial construction could still exceed it.
_SWIMLANE_LANE_ROW_PX = 115


def _swimlane_tip_fit_height(
    tip_rows: int, max_lane_chars: int, max_votes_chars: int = 0
) -> int:
    """swimlanes' own tooltip-fit floor: the fixed `SPAN_TIP_FIELDS`
    rows are single-line by construction and get `_TIP_SHORT_ROW_PX`
    each; the two variable-length rows get scaled allowances - "lane"
    from ``max_lane_chars`` (worst observed word packing, capped at
    `_SWIMLANE_LANE_ROW_PX`) and the joined "member votes" row from
    ``max_votes_chars`` via `embed.wrap_row_px` (0 = no such row).
    """
    lane_lines = max(1, -(-min(max_lane_chars, 121) // _LANE_WORST_CHARS_PER_LINE))
    lane_px = min(
        _SWIMLANE_LANE_ROW_PX,
        _TIP_SHORT_ROW_PX + _TIP_EXTRA_LINE_PX * (lane_lines - 1),
    )
    votes_px = wrap_row_px(max_votes_chars) if max_votes_chars else 0
    return _tip_floor(tip_rows, lane_px, second_tall_row_px=votes_px)


# The minimum drawn width of a swimlane box, in pixels, converted to turn
# units through `chart_width`'s pixel-per-turn rate (the same
# approximation `swimlane_min_footprint` uses: the plotted body is a
# little narrower than `chart_width`, so the sliver only ever errs wide).
# A floor in turn units would overstate a short span's wall-clock extent
# on a short axis, where one turn is a hundred pixels; a pixel floor
# keeps a box honest and merely visible.
_MIN_BOX_PX = 6.0


def span_min_box_width(n_turns: int) -> float:
    """The minimum box width in turn units: `_MIN_BOX_PX` at this axis's
    pixel-per-turn rate. Also the hover footprint under a tick, and the
    packing footprint `render._subagent_section` reserves for boxed
    spans, so two sliver-floored boxes never render overlapping."""
    return _MIN_BOX_PX * n_turns / chart_width(n_turns)


def span_geometry(frame: pd.DataFrame, n_turns: int) -> pd.DataFrame:
    """Rendered geometry for swimlane rows (``x_start``, ``width``,
    ``boxed``): ``x1``/``x2`` for a box (its extent floored at
    `span_min_box_width`), ``x`` for a tick, and ``hx1``/``hx2`` for the
    tip-bearing hit rect over either. A box that would overrun the
    axis's right edge (activity clamped at the last orchestrator turn)
    is shifted left to end at the edge, never drawn off-canvas.

    ``hx2`` is one addition from ``hx1``, keeping every tick row's
    data-space hit width exactly the floor whatever ``x_start``'s
    magnitude.
    """
    out = frame.copy()
    floor = span_min_box_width(n_turns)
    right = turn_xlim(n_turns)[1]
    x1 = out.x_start
    x2 = out.x_start + out.width.clip(lower=floor)
    over = (x2 - right).clip(lower=0.0)
    out["x1"] = x1 - over
    out["x2"] = x2 - over
    out["x"] = out.x_start
    out["hx1"] = out.x1.where(out.boxed, out.x_start - floor / 2)
    out["hx2"] = out.x2.where(out.boxed, out.hx1 + floor)
    return out


# The seam both the packing decision (`swimlane_min_footprint`) and the
# browser test hold every rendered mark separation to - one number, so
# "the packer's definition of cannot-touch" and "the test's definition of
# visibly separate" cannot drift apart.
_FOOTPRINT_SEAM_PX = 2.0


# The visual thickness of a uniform-mode tick. `swimlane_min_footprint`
# converts it to turns for the packing decision and `swimlanes` passes it
# straight through as the mark's `stroke_width`, so the packed-for width
# and the drawn width cannot drift apart.
_TICK_STROKE_WIDTH = 2.0


# The swimlane box's half-height, in the same y-unit as one row's pitch
# (1.0 - `pack_lanes` places rows at consecutive integers). 0.3 leaves a
# 0.4-unit gap, so two concurrent same-label spans in adjacent sub-lanes
# - which share one fill colour by construction - read as separate
# boxes rather than fusing into one taller rectangle. Uniform mode's
# ticks reuse it for their own vertical extent.
_SWIMLANE_BOX_HALF_HEIGHT = 0.3


# The white outline every swimlane box gets, so a box's edge stays
# visible where the true gap to its neighbour is a handful of pixels. A
# bare colour string resolves as a paint constant, not a column lookup -
# the string-is-a-column rule in the module docstring is specific to
# symbol names.
_SWIMLANE_BOX_STROKE = "#fff"
_SWIMLANE_BOX_STROKE_WIDTH = 1.5


# The glyph `swimlanes` marks a recorded span completion with, and the
# character `sections.subagent_notes`' how-to-read line names it by.
#
# Check any replacement against the ``Symbol`` literal on
# `inspect_viz.mark.dot`'s own signature, not Plot's prose docs: a name
# that is not one of Plot's symbols resolves as a column reference and
# the resulting binder error blanks the whole plot - silent in Python,
# total in the browser.
_END_MARKER_SYMBOL = "times"
_END_MARKER_GLYPH = "×"


def spend_bars(bars: list[dict]) -> tuple[Component, int]:
    """Horizontal token-spend bars: one per bucket, top-down in the
    caller's (descending-spend) order, category name in the left margin,
    the formatted value at the bar's end, and a tooltip carrying bucket /
    tokens / share.

    ``bars``: `sections.spend_data` rows - label / value / color /
    share (share None on the sub-agent half, whose bars state no
    of-total figure; an absent cell emits no tooltip row). "bucket",
    never "label", as the channel key - see `_RESERVED_TIP_CHANNELS`.
    """
    frame = pd.DataFrame(
        {
            "y": range(len(bars)),
            "bucket": [b["label"] for b in bars],
            "bucket_text": [
                b["label"]
                if len(b["label"]) <= _SPEND_LABEL_MAX_CHARS
                else b["label"][: _SPEND_LABEL_MAX_CHARS - 1] + "\u2026"
                for b in bars
            ],
            "value": [float(b["value"]) for b in bars],
            "color": [b["color"] for b in bars],
            "share": [b["share"] for b in bars],
            "tokens_text": [f"{b['value']:,}" for b in bars],
            # sub-agent bars only (the span tooltips carry no token
            # rollups); absent on the phase half, emitting no row
            "output": [b.get("output") for b in bars],
            "billable": [b.get("billable") for b in bars],
        }
    )
    frame["y1"] = frame.y - 0.38
    frame["y2"] = frame.y + 0.38
    frame["x1"] = 0.0
    data = Data.from_dataframe(
        frame[
            [
                "y",
                "y1",
                "y2",
                "x1",
                "value",
                "bucket",
                "bucket_text",
                "color",
                "share",
                "tokens_text",
                "output",
                "billable",
            ]
        ]
    )
    max_value = float(frame.value.max())
    height_px = _SPEND_MARGIN_TOP + len(bars) * _SPEND_ROW_PX + _SPEND_MARGIN_BOTTOM
    component = plot(
        rect(
            data,
            x1="x1",
            x2="value",
            y1="y1",
            y2="y2",
            fill="color",
            tip=TipOptions(pointer="y"),
            channels=(
                channels := {
                    "bucket": "bucket",
                    "new-work tokens": "tokens_text",
                    "output tokens": "output",
                    "tokens excluding cache reads": "billable",
                    "share": "share",
                }
            ),
        ),
        text(
            data,
            x="value",
            y="y",
            text="tokens_text",
            styles=TextStyles(text_anchor="start", font_size=11),
            dx=6,
        ),
        text(
            data,
            x=sql("0"),
            y="y",
            text="bucket_text",  # gutter-capped; the tooltip's "bucket"
            # row carries the full label
            styles=TextStyles(text_anchor="end", font_size=11),
            dx=-8,
        ),
        width=_SPEND_WIDTH,
        height=height_px,
        y_axis=False,  # not None - see module docstring
        y_domain=(len(bars) - 0.5, -0.5),
        # headroom on the right for the value text
        x_domain=(0, max_value * 1.18),
        x_label="new-work tokens",
        margin_left=_SPEND_MARGIN_LEFT,
        margin_top=_SPEND_MARGIN_TOP,
        margin_bottom=_SPEND_MARGIN_BOTTOM,
        color_scale=_IDENTITY_SCALE,
    )
    floor = _tip_floor(_tip_rows(channels), _TIP_SHORT_ROW_PX + 2 * _TIP_EXTRA_LINE_PX)
    return component, _section_height(
        height_px, tip_rows=_tip_rows(channels), floor=floor
    )


_SPEND_WIDTH = 900
_SPEND_MARGIN_LEFT = 160  # matches the sub-agent chart's own gutter.
# Labels longer than it fits are ellipsis-truncated
# (`_SPEND_LABEL_MAX_CHARS`), same honesty as the lane-name cap.
_SPEND_LABEL_MAX_CHARS = 24
_SPEND_ROW_PX = 30
_SPEND_MARGIN_TOP = 8
_SPEND_MARGIN_BOTTOM = 34  # room for the token x-axis + its label
