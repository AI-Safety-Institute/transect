"""The report orchestrator: `render_report`.

Loads each transcript's frame slices, builds its charts (`charts.py`)
and text fragments (`sections.py`), and assembles them into one titled
section per chart: phase timeline (band + agreement strip), token
telemetry, human interventions, sub-agent activity, token spend, phase
cards, and a reliability & provenance audit.
"""

import base64
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from markupsafe import Markup

from transect.frames import TransectResults
from transect.report import charts, custom, sections
from transect.report._jinja import jinja_env
from transect.report.colors import _UNJUDGED_GREY, _label_colors, _phase_colors
from transect.report.display import MEMBER_NO_VOTE, member_display, multi_roll_models
from transect.report.embed import embed_section
from transect.report.excerpts import (
    SpawnPrompt,
    mark_compaction_turns,
    read_transcript_extras,
)
from transect.report.lanes_layout import pack_lanes
from transect.report.style import _STYLE, _WIDGET_STYLE_CSS
from transect.tags import select_tags

# The built-in sections' stable keys, in default reading order
SECTION_KEYS = (
    "eval_setup",
    "phase_timeline",
    "interventions",
    "token_telemetry",
    "subagents",
    "token_spend",
    "phase_cards",
)


def render_report(
    results: TransectResults,
    title: str = "Transect report",
    viewer_base_url: str | None = None,
    section_order: list[str] | None = None,
    sensitivity: str | None = None,
) -> str:
    validate_section_order(
        section_order, [layer.name for layer in results.extra_layers]
    )
    timeline = results.token_timeline
    order = (
        (
            timeline[["agent", "sample_id", "epoch", "transcript_id"]]
            .drop_duplicates()
            .sort_values(["agent", "sample_id", "epoch"])
            .transcript_id.tolist()
        )
        if len(timeline)
        else []
    )
    phase_colors = _phase_colors(results.phases)
    # family -> declaring layer, the chips' and selectors' provenance
    tag_layer_of: dict[str, str] = {}
    for layer in results.extra_layers:
        layer_frame = results.layer_frames.get(layer.name)
        if layer.tags and layer_frame is not None:
            for family in select_tags(layer, layer_frame):
                tag_layer_of[family] = layer.name
    # read once for all transcripts; empty when the store is unreachable
    # at render time, and the affected surfaces render without extras
    transcript_extras = read_transcript_extras(results.transcripts_location, order)
    transcripts = []
    for idx, transcript_id in enumerate(order):
        one = _mine(timeline, transcript_id)
        my_info = _mine(results.transcript_info, transcript_id)
        my_flushes = _mine(results.flushes, transcript_id).sort_values("turn")
        flush_turns = list(my_flushes.turn)
        my_interventions = _mine(results.interventions, transcript_id).sort_values(
            "turn",
            kind="stable",  # several interventions can share a turn
        )
        intervention_turns = list(my_interventions.turn)
        my_phases = _mine(results.phases, transcript_id)
        my_turn_votes = _mine(results.phase_turn_votes, transcript_id)
        my_definitions = _mine(results.label_definitions, transcript_id)
        tool_lanes = _mine(results.lane_activity, transcript_id)
        my_phase_turns = _mine(results.phase_turns, transcript_id)
        my_subagents = _mine(results.subagents, transcript_id)
        my_tags = (
            _mine(results.turn_tags, transcript_id)
            if results.turn_tags is not None
            else None
        )
        my_extras = transcript_extras.get(transcript_id)
        # sections collect keyed in default reading order; a section
        # with no data is skipped entirely, so the count varies by
        # transcript. section_order rearranges them at assembly below.
        keyed: list[tuple[str, Markup]] = []
        add = keyed.append

        # 0. eval setup: how the run was actually configured, up front
        # (three default-collapsed expandables; the intro sentence
        # renders under the transcript heading via run_meta)
        add(
            (
                "eval_setup",
                sections.section(
                    "Eval setup",
                    [sections.eval_setup_blocks(my_info, my_flushes)],
                ),
            )
        )

        # 1. phase timeline (the cards are built here too)
        phase_cards_section = None
        if len(my_phases):
            # the band's click-to-card href and the cards' ids must
            # agree, and a bare phase_index collides across transcripts
            # on a multi-transcript page - hence the per-transcript
            # prefix (distinct from the "phase-cards-{idx}" scope id)
            card_id_prefix = f"phase-{idx}"
            phase_components, phase_height = charts.phase_band(
                my_phase_turns,
                my_phases.sort_values("phase_index"),
                phase_colors,
                card_id_prefix,
                my_turn_votes,
            )
            my_groups = _mine(results.turn_groups, transcript_id)
            # the scan's own per-turn tool counts when emitted, else
            # the render-time store read
            tool_counts: dict[int, int] | None = None
            if one.n_tool_calls.notna().any():
                tool_counts = {
                    int(turn): int(n)
                    for turn, n in zip(one.turn, one.n_tool_calls, strict=True)
                    if pd.notna(n)
                }
            elif my_extras:
                tool_counts = my_extras.tool_counts
            timeline_section = sections.section(
                "Phase timeline",
                [
                    sections.phase_explanation(),
                    sections.phase_summary_line(my_phases, my_phase_turns),
                    sections.reliability_warnings(
                        sections.phase_reliability_flags(
                            my_phases, my_turn_votes, my_phase_turns
                        )
                    ),
                    _chart(phase_components, phase_height),
                    sections.phase_chips(my_phases, phase_colors, my_definitions),
                    sections.agreement_strip_caption(my_phase_turns, my_phases),
                    sections.phase_definitions(my_definitions),
                ],
            )
            add(("phase_timeline", timeline_section))
            phase_cards_section = sections.section(
                "Phase cards",
                [
                    sections.phase_cards_explanation(),
                    sections.phase_meta_line(my_phases, my_phase_turns),
                    sections.phase_card_controls(
                        f"phase-cards-{idx}",
                        my_phases,
                        phase_colors,
                        turn_tags=my_tags,
                        tag_layer_of=tag_layer_of,
                    ),
                    sections.narrator_line(my_phases),
                    sections.phase_cards(
                        my_phases,
                        my_groups,
                        my_turn_votes,
                        one,
                        flush_turns,
                        intervention_turns,
                        phase_colors,
                        _viewer_link(
                            viewer_base_url,
                            results.transcripts_location,
                            transcript_id,
                        ),
                        f"phase-cards-{idx}",
                        card_id_prefix,
                        # flushes.turn is the first post-flush turn; the
                        # summarization call is the model turn before it
                        mark_compaction_turns(
                            my_extras.excerpts,
                            my_flushes.turn[my_flushes.compaction_prompt.notna()] - 1,
                        )
                        if my_extras
                        else None,
                        lanes=tool_lanes,
                        tool_counts=tool_counts,
                        turn_tags=my_tags,
                        tag_layer_of=tag_layer_of,
                    ),
                ],
            )
        else:
            # the section stays, as a note stating why there is no timeline
            add(
                (
                    "phase_timeline",
                    sections.section(
                        "Phase timeline",
                        [sections.no_phases_note(results.scan_status)],
                    ),
                )
            )
        # 2. human interventions, directly below the phase timeline: the
        # natural cross-read is phase behaviour around an intervention
        # (only when the transcript has any)
        if len(my_interventions):
            intervention_component, intervention_height = charts.interventions_chart(
                my_interventions, int(one.turn.max()) + 1
            )
            interventions_section = sections.section(
                "Human interventions",
                [
                    _chart([intervention_component], intervention_height),
                    sections.intervention_legend(),
                    sections.intervention_line(my_interventions),
                ],
            )
            add(("interventions", interventions_section))
        # 3. token telemetry
        derived = charts.has_derived_token_views(one)
        token_blocks = [
            sections.token_intro(
                derived, coincide=derived and charts.token_measures_coincide(one)
            )
        ]
        threshold = sections.compaction_threshold(my_info)
        components, stack_height = charts.token_stack(
            one, my_flushes, threshold.tokens if threshold is not None else None
        )
        token_blocks.append(
            _chart(components, stack_height)
            if components
            else Markup("<p>No timeline data.</p>")
        )
        # mirrors charts.token_stack's condition for drawing the rule
        drawn_threshold = (
            threshold.tokens
            if threshold is not None and bool(one.context.notna().any())
            else None
        )
        if flush_turns or drawn_threshold is not None:
            token_blocks.append(
                sections.event_legend(
                    derived, flushes=bool(flush_turns), threshold=drawn_threshold
                )
            )
        if flush_turns:
            token_blocks.append(sections.flush_line(my_flushes))
        add(("token_telemetry", sections.section("Token telemetry", token_blocks)))
        # 4. sub-agent activity (the votes slice is shared with the
        # audit section below)
        my_subagent_votes = _mine(results.subagent_votes, transcript_id)
        lanes, render_subagents = _span_lanes(one, tool_lanes)
        if render_subagents:
            subagent_section = _subagent_section(
                one,
                lanes,
                my_subagents,
                my_subagent_votes,
                my_extras.spawn_prompts if my_extras else {},
                classification_ran=bool(results.subagents.status.notna().any()),
                label_definitions=my_definitions,
            )
            add(("subagents", subagent_section))
        # 5. token spend by phase / sub-agent label / custom tag family
        spend_data = sections.spend_data(one, my_phases, phase_colors, my_subagents)
        family_groups = [
            (family, layer_name, bars)
            for family, layer_name in tag_layer_of.items()
            if (bars := sections.tag_spend_bars(one, my_tags, family))
        ]
        spend_blocks = []
        if spend_data:
            phase_bars, subagent_bars, subagent_message = spend_data
            if phase_bars:
                spend_blocks.append(sections.spend_group_heading("By phase label"))
                comp, h = charts.spend_bars(phase_bars)
                spend_blocks.append(_chart([comp], h))
            if subagent_bars:
                spend_blocks.append(sections.spend_group_heading("By sub-agent label"))
                comp, h = charts.spend_bars(subagent_bars)
                spend_blocks.append(_chart([comp], h))
            elif subagent_message:
                spend_blocks.append(sections.spend_group_heading("By sub-agent label"))
                spend_blocks.append(sections.spend_subagent_unavailable())
        if family_groups:
            scope = f"spend-tags-{idx}"
            spend_blocks.append(
                sections.spend_family_control(
                    scope, [(family, name) for family, name, _ in family_groups]
                )
            )
            for pos, (family, _name, bars) in enumerate(family_groups):
                comp, h = charts.spend_bars(bars)
                spend_blocks.append(
                    sections.spend_family_group(
                        scope, family, _chart([comp], h), hidden=pos > 0
                    )
                )
        if spend_blocks:
            add(("token_spend", sections.section("Token spend", spend_blocks)))
        if phase_cards_section is not None:
            add(("phase_cards", phase_cards_section))

        # 6. custom layers, badge-marked; their names are section keys
        for layer in results.extra_layers:
            if not layer.section:
                continue
            layer_frame = _my_layer_frame(results, layer, transcript_id)
            ctx = custom.SectionContext(
                transcript_id=transcript_id,
                frame=layer_frame,
                n_turns=int(one.turn.max()) + 1 if len(one) else 0,
            )
            add(
                (
                    layer.name,
                    custom.layer_section(layer, layer_frame, ctx, my_definitions),
                )
            )

        # 7. reliability & provenance audit (always the report's last)
        layer_audits = [
            {
                "name": layer.name,
                "frame": _my_layer_frame(results, layer, transcript_id),
                "unit_col": layer.audit[0],
                "label_col": layer.audit[1],
            }
            for layer in results.extra_layers
            if layer.audit is not None
        ]
        audit_section = sections.section(
            "Reliability & provenance audit",
            [
                sections.reliability_audit(
                    my_phases,
                    my_phase_turns,
                    my_subagents,
                    my_subagent_votes,
                    my_definitions,
                    phase_turn_votes=my_turn_votes,
                    layer_audits=layer_audits,
                )
            ],
            anchor="reliability-audit" if idx == 0 else None,
        )
        sections_html = [markup for _, markup in _apply_order(keyed, section_order)]
        sections_html.append(audit_section)

        transcripts.append(
            {
                "heading": _transcript_title(one, my_info),
                "run_meta": sections.run_intro_line(my_info),
                "sections": sections_html,
            }
        )

    template = jinja_env().get_template("report.html.j2")
    return template.render(
        title=title,
        style=Markup(_STYLE),
        scan_location=str(results.scan_location),
        scan_status_view=sections.scan_status_view(results.scan_status),
        n_transcripts=len(order),
        transcripts=transcripts,
        sensitivity=sensitivity,
        logo_uri=_brand_asset("logo.svg", "image/svg+xml"),
        favicon_uri=_brand_asset("favicon-32.png", "image/png"),
        package_version=_package_version(),
    )


def validate_section_order(
    section_order: list[str] | None, layer_names: list[str]
) -> None:
    """Reject an order naming unknown or repeated section keys."""
    if section_order is None:
        return
    valid = (*SECTION_KEYS, *layer_names)
    unknown = [key for key in section_order if key not in valid]
    if unknown:
        raise ValueError(
            f"section_order names unknown section(s) {unknown} - valid "
            f"keys are {list(valid)} (built-ins plus each custom "
            "layer's name; the reliability audit is not orderable, it "
            "always renders last, and the run-wide Scan execution & "
            "coverage block always renders once after all transcripts)"
        )
    if len(set(section_order)) != len(section_order):
        raise ValueError("section_order repeats a section key")


def _subagent_section(
    one: pd.DataFrame,
    lanes: list[tuple],
    subagents: pd.DataFrame,
    subagent_votes: pd.DataFrame,
    spawn_prompts: dict[str, SpawnPrompt],
    classification_ran: bool,
    label_definitions: pd.DataFrame | None = None,
) -> Markup:
    """The whole Sub-agent activity section: explanation, notes and
    legend, the swimlanes chart, inline reliability flags, then the
    spawn-prompts expandable.

    One row-block per classification label; concurrent same-label spans
    pack into sub-lanes (`lanes_layout.pack_lanes`). A box marks a
    span's turn extent where the source records span ends, a tick only
    its start turn where it does not. ``lanes`` comes from
    `_span_lanes`, which also decides whether the section renders.
    """
    n_turns = int(one.turn.max()) + 1
    label_of = {
        row.agent_span_id: str(row.label)
        for row in subagents.itertuples()
        if pd.notna(row.label)
    }

    # three honesty cases: (1) no classification scan ran - the label
    # dimension disappears (neutral grey "sub-agents", no chips, no
    # classification tooltip row); (2) it ran but joined zero lanes -
    # "unclassified" plus the loud template warning; (3) it ran and some
    # spans got no label - "unclassified" is the honest fallback
    def _label(span_id) -> str:
        label = label_of.get(span_id)
        if isinstance(label, str) and label:
            return label
        return "unclassified" if classification_ran else "sub-agents"

    by_label: dict[str, list] = {}
    for span_id, lane, has_tokens in lanes:
        by_label.setdefault(_label(span_id), []).append((span_id, lane, has_tokens))
    colors = (
        _label_colors(sorted(by_label))
        if classification_ran
        else dict.fromkeys(by_label, _UNJUDGED_GREY)
    )

    # one flag, four consumers on purpose (end markers, box-vs-tick,
    # the how-to-read sentence, the packing mode)
    has_end_markers = bool(subagents.span_end_recorded.any())
    # uniform-mode marks share a fixed footprint, so extent-only packing
    # would let nearby spans render overlapping; proportional mode packs
    # by true extent and needs no floor
    min_footprint = 0.0 if has_end_markers else charts.swimlane_min_footprint(n_turns)
    packed = pack_lanes(lanes, _label, min_footprint=min_footprint)

    end_of: dict[str, float] = (
        {
            str(sid): float(end)
            for sid, end, recorded in zip(
                subagents.agent_span_id,
                subagents.span_end_turn,
                subagents.span_end_recorded,
                strict=True,
            )
            if recorded and pd.notna(end)
        }
        if has_end_markers
        else {}
    )
    end_markers = [
        (end_of[sid], row_y) for sid, row_y in packed.span_row if sid in end_of
    ]

    # pack_lanes appends rows and span_row in the same iteration, so
    # positional pairing is exact
    span_title_of = sections.span_titles(
        subagents, label_of, span_ends_recorded=has_end_markers
    )
    # one "member votes" tooltip row (voting regimes only), mirroring
    # the agreement strip: every member's own label + confidence (or
    # its reason for producing no vote)
    member_fields: list[str] = []
    if classification_ran and len(subagent_votes):
        pairs = zip(subagent_votes.model, subagent_votes.roll, strict=True)
        member_ids = sorted({(str(m), int(r)) for m, r in pairs})
        multi = multi_roll_models(subagent_votes)
        # a k-roll run has one model: "roll-N" alone says everything
        one_model = len({m for m, _ in member_ids}) == 1 and len(member_ids) > 1
        ballots_of: dict = {}
        for model, roll in member_ids:
            key = f"roll-{roll}" if one_model else member_display(model, roll, multi)
            sub = subagent_votes[
                (subagent_votes.model == model) & (subagent_votes.roll == roll)
            ]
            for row in sub.itertuples():
                if row.agent_span_id not in span_title_of:
                    continue
                voted = str(row.status) == "ok" and pd.notna(row.label)
                if voted:
                    vote = str(row.label)
                    if pd.notna(row.confidence):
                        vote += f" ({float(str(row.confidence)):.2f})"
                else:
                    vote = MEMBER_NO_VOTE.get(str(row.status), str(row.status))
                ballots_of.setdefault(row.agent_span_id, []).append(f"{key}: {vote}")
        if ballots_of:
            member_fields = ["member votes"]
            for span_id, ballots in ballots_of.items():
                span_title_of[span_id]["member votes"] = " · ".join(ballots)
    judged_fields = (
        "classification",
        "confidence",
        "agreement",
        "label source",
        "verifier",
    )
    base_fields = [
        f
        for f in charts.SPAN_TIP_FIELDS
        # honesty case (1): no classifier ran -> no judged cells at all
        if classification_ran or f not in judged_fields
    ]
    insert_at = base_fields.index("verifier") + 1 if "verifier" in base_fields else 2
    tip_fields = tuple(
        base_fields[:insert_at] + member_fields + base_fields[insert_at:]
    )
    row_titles = [span_title_of[span_id] for span_id, _row_y in packed.span_row]

    component, chart_height = charts.swimlanes(
        packed.rows,
        packed.yticks,
        packed.ylabels,
        end_markers,
        n_turns,
        colors,
        titles=row_titles,
        span_ends_recorded=has_end_markers,
        tip_fields=tip_fields,
    )
    flags = sections.subagent_reliability_flags(subagents, subagent_votes)
    # spawn prompts: the full text from the render-time store read wins;
    # the loader's stored copy (capped to what the judge saw, hence its
    # truncated flag) fills in for spans the store could not supply. A
    # span with no resolved prompt is absent, not a blank row.
    stored = {
        str(row.agent_span_id): SpawnPrompt(
            text=str(row.span_task),
            truncated=bool(row.span_task_truncated),
        )
        for row in subagents.itertuples()
        if pd.notna(row.span_task)
    }
    spawn_prompts = {**stored, **spawn_prompts}
    spawn_rows = sorted(
        (
            {
                "turn": int(lane.turn.min()),
                # case (1): no classifier -> no label segment on the row
                "label": _label(span_id) if classification_ran else None,
                "text": spawn_prompts[span_id].text,
                "truncated": spawn_prompts[span_id].truncated,
            }
            for span_id, lane, _has_tokens in lanes
            if span_id in spawn_prompts
        ),
        key=lambda row: int(row["turn"] or 0),
    )
    return sections.section(
        "Sub-agent activity",
        [
            # honesty case (1): with no judge run the explanation would
            # describe a scan that never happened
            sections.subagent_explanation() if classification_ran else None,
            sections.subagent_notes(
                lanes,
                label_of,
                subagents,
                classification_ran,
                has_end_markers,
            ),
            sections.reliability_warnings(flags),
            _chart([component], chart_height),
            sections.subagent_legend(by_label, colors, classification_ran),
            sections.spawn_prompts_block(spawn_rows),
            sections.subagent_definitions(label_definitions),
        ],
    )


def _span_lanes(
    one: pd.DataFrame, tool_lanes: pd.DataFrame
) -> tuple[list[tuple], bool]:
    """This transcript's sub-agent lanes, and whether the Sub-agent
    activity section renders at all.

    One ``(span_id, lane_frame, has_tokens)`` triple per distinct span:
    model turns first, then tool-only spans not already present as a
    model lane. The gate: more than one lane once the orchestrator's
    own lane (span-less model turns) is counted - inspect wraps even a
    solo agent in a span, so a single-agent .eval shows no block.
    """
    lanes = [
        (sid, group, True)
        for sid, group in one.sort_values("turn").groupby("agent_span_id", sort=False)
    ]
    model_ids = {sid for sid, _, _ in lanes}
    lanes += [
        (sid, group, False)
        for sid, group in tool_lanes.groupby("agent_span_id", sort=False)
        if sid not in model_ids
    ]
    main_lane = 1 if one.agent_span_id.isna().any() else 0
    return lanes, len(lanes) + main_lane > 1


def _transcript_title(one: pd.DataFrame, info: pd.DataFrame) -> str:
    """The per-transcript heading: identity labels the source always
    carries (sample_id, agent, epoch - from the token timeline), plus
    ``model`` when the source's own transcript metadata recorded one.
    """
    sample_id = str(one.sample_id.iloc[0])
    agent = str(one.agent.iloc[0])
    epoch = one.epoch.iloc[0]
    model = _info_cell(info, "model")
    labels = [f"sample_id: {sample_id}", f"agent: {agent}"]
    if model is not None:
        labels.append(f"model: {model}")
    return f"{' · '.join(labels)} (epoch {epoch})"


def _viewer_link(base_url, transcripts_location, transcript_id):
    """Scout-viewer deep-link builder for one transcript, or None.

    URL shape:
    ``{base}/#/transcripts/{base64url(location)}/{transcript_id}``
    with an optional ``?event=<uuid>`` anchor. base64url per the
    viewer: standard base64, ``+``->``-``, ``/``->``_``, no padding.
    """
    if not base_url or not transcripts_location:
        return None
    encoded = (
        base64.urlsafe_b64encode(str(transcripts_location).encode())
        .decode()
        .rstrip("=")
    )
    root = f"{base_url.rstrip('/')}/#/transcripts/{encoded}/{transcript_id}"

    def link(anchor_event_id=None):
        if anchor_event_id and pd.notna(anchor_event_id):
            return f"{root}?event={quote(str(anchor_event_id))}"
        return root

    return link


def _info_cell(info: pd.DataFrame, column: str):
    """One transcript_info cell, or None."""
    if not len(info):
        return None
    value = info[column].iloc[0]
    return None if pd.isna(value) else value


def _chart(components: list, height_px: int) -> Markup:
    """One section's chart iframe: the widget style CSS and the
    `Markup` wrapping of `embed_section` done in one place."""
    return Markup(embed_section(components, height_px, extra_head=_WIDGET_STYLE_CSS))


def _mine(frame: pd.DataFrame, transcript_id: str) -> pd.DataFrame:
    """This transcript's rows of one frame."""
    return frame[frame.transcript_id == transcript_id]


def _apply_order(
    keyed: list[tuple[str, Markup]], section_order: list[str] | None
) -> list[tuple[str, Markup]]:
    """Listed sections first, in the given order; unlisted ones follow
    in default reading order."""
    if not section_order:
        return keyed
    listed = [kv for key in section_order for kv in keyed if kv[0] == key]
    return listed + [kv for kv in keyed if kv[0] not in set(section_order)]


def _my_layer_frame(results, layer, transcript_id: str) -> pd.DataFrame | None:
    """One layer's mounted frame cut to this transcript."""
    frame = results.layer_frames.get(layer.name)
    if frame is not None and "transcript_id" in frame.columns:
        return _mine(frame, transcript_id)
    return frame


def _brand_asset(name: str, media_type: str) -> str:
    """One packaged brand asset as a data URI - the report stays
    self-contained."""
    payload = base64.b64encode(
        (Path(__file__).parent / "assets" / name).read_bytes()
    ).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _package_version() -> str:
    try:
        return version("transect")
    except PackageNotFoundError:  # source tree without an install
        return "dev"
