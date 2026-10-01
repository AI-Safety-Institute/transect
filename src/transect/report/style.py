"""The report's CSS: the outer page's stylesheet and the widget-styling
head splice for the embedded chart documents. Values are captured
literals from Inspect View's ts-mono theme (stock bootstrap 5.3.8 plus
its ~20 overrides), each with its provenance recorded next to the
declaration.
"""

# [bootstrap] --bs-font-sans-serif (ts-mono adds no font override).
# Shared by `_STYLE` and `_WIDGET_STYLE_CSS`, so a change touches one
# place.
_FONT_STACK = (
    'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", '
    '"Noto Sans", "Liberation Sans", Arial, sans-serif, '
    '"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"'
)

# Captured literals read against ts-mono@125db36a5f21 + its pinned
# bootstrap@5.3.8; provenance per value: "[bootstrap]" = stock value,
# "[ts-mono]" = that package's override. Named `--transect-*`, not `--bs-*`:
# no bootstrap runtime dependency, only captured values. Plot colours
# live in `colors.py`.
_STYLE = f"""
:root {{
  --transect-color-text: #212529;          /* [bootstrap] --bs-body-color */
  --transect-color-muted: rgba(33, 37, 41, 0.75); /* [bootstrap] --bs-secondary-color */
  --transect-color-border: #dee2e6;        /* [bootstrap] --bs-border-color */
  --transect-color-border-subtle: #e9ecef; /* [bootstrap] --bs-light-border-subtle */
  --transect-color-hover: #f8f9fa;         /* [bootstrap] --bs-light; the one hover
                                          wash, restated as a literal in the
                                          widget CSS (iframes can't see it) */
  --transect-color-link: #104D6A;          /* steel - links + controls */
  --transect-color-link-hover: #0E2841;    /* navy */
  --transect-color-accent: #09A4A7;        /* teal - focus rings */
  --transect-radius: 3px;                  /* [ts-mono] --bs-border-radius override */
  --transect-font-size-title: 1.4rem;      /* [ts-mono] .navbar-brand, 1.4em */
  --transect-font-size-larger: 1.1rem;     /* [ts-mono] --inspect-font-size-larger */
  --transect-font-size-small: 0.9rem;      /* [ts-mono] --inspect-font-size-small */
  --transect-font-size-smaller: 0.8rem;    /* [ts-mono] --inspect-font-size-smaller */
  --transect-chip-bg: #f3f3f6;             /* [ts-mono] --inspect-msg-label-bg */
  --transect-chip-border: #e4e4e9;         /* [ts-mono] --inspect-msg-label-border */
  --transect-chip-text: #5f5f68;           /* [ts-mono] --inspect-msg-label-text */
}}
body {{ font-family: {_FONT_STACK}; color: var(--transect-color-text);
       max-width: 1360px; margin: 2rem auto; padding: 0 3.5rem; }}
a {{ color: var(--transect-color-link); }}
a:hover, a:focus {{ color: var(--transect-color-link-hover); }}
h1, h2, h3, h4 {{ font-weight: 500; line-height: 1.2; }}
h1, h2 {{ color: #0E2841; }}  /* navy */
h1 {{ font-size: var(--transect-font-size-title); }}
h2 {{ font-size: var(--transect-font-size-larger); margin-top: 2rem; }}
h3 {{ font-size: var(--transect-font-size-small); font-weight: 600; }}
/* h4 (audit entity names) bold over h5 (the maps' own titles), so
   the audit's heading hierarchy reads top-down */
h4 {{ font-size: var(--transect-font-size-smaller); font-weight: 600; }}
.phase-cards {{ overflow-wrap: anywhere; }}
.meta {{ color: var(--transect-color-muted); font-size: 0.85rem; }}
/* chart how-to-read lines: one step below .meta, so explainers cede
   visual space to the charts they explain */
.explainer {{ font-size: 0.78rem; }}
.custom-layer-badge {{ font-size: 0.66rem; font-weight: 600;
  color: #6b4fa0; background: #f0ebf8; border: 1px solid #d9cdEE;
  border-radius: 9px; padding: 1px 8px; vertical-align: middle;
  letter-spacing: 0.03em; text-transform: uppercase; }}
/* markdown body text reads as section prose (the .meta register,
   full width); demoted headings render at the section-title (h3)
   style */
.custom-md p, .custom-md li {{ color: var(--transect-color-muted);
  font-size: 0.85rem; }}
.custom-md h5, .custom-md h6 {{ font-size: var(--transect-font-size-small);
  font-weight: 600; color: var(--transect-color-text);
  margin: 1rem 0 0.1rem; }}
/* user tag chips on phase cards: shape only - each chip's colours
   arrive inline, tinted per tag family (colors.chip_tint); purple
   stays reserved for the custom-provenance badge. The slate here is
   the no-style fallback. */
.user-chip {{ font-size: 0.7rem; color: #47525e; background: #eef1f4;
  border: 1px solid #d3dae1; border-radius: 9px; padding: 1px 7px;
  white-space: nowrap; }}
.warning {{ color: #b00a1f; font-size: 0.9rem; font-weight: 600;
           background: #fdeaee; padding: 6px 10px;
           border-radius: var(--transect-radius); }}  /* darkened red */
.warning a {{ color: inherit; }}  /* steel clashes inside the red band */
.warning-amber {{ color: #7a6300; background: #fdf6e0; font-weight: 500; }}
/* the audit's per-classification maps: colour ramps computed in
   sections.py (red = unhealthy, steel = healthy; pale-to-steel for
   shares), values always in the cell text */
.relmap-scroll {{ overflow-x: auto; }}
.relmap {{ border-collapse: collapse; font-size: 0.78rem;
  margin: 0.3rem 0 0.4rem; }}
.relmap th, .relmap td {{ border: 1px solid var(--transect-color-border);
  padding: 0.3rem 0.6rem; text-align: center; }}
/* value cells are exactly the two authored lines ("15% [7-31%]" over
   "5/33 votes"): the <br> is the only permitted break, never a wrap */
.relmap td {{ white-space: nowrap; }}
.relmap tr th:first-child {{ text-align: left; font-weight: 500;
  background: var(--transect-color-hover); white-space: nowrap; }}
.relmap tr:first-child th {{ background: var(--transect-color-hover);
  font-weight: 600; }}
.relmap-sub {{ font-size: 0.7rem; opacity: 0.85; font-weight: 400; }}
.relmap-ci {{ font-size: 0.65rem; opacity: 0.85; }}
.relmap-absent {{ color: var(--transect-color-muted); opacity: 0.6; font-weight: 400; }}
.relmap-group {{ font-weight: 600; }}
.relmap-help {{ color: var(--transect-color-muted); font-weight: 400;
  font-size: 0.7rem; cursor: help; }}
.relmap-legend {{ margin: 0.2rem 0 1rem; }}
/* the audit's explanatory colour scale: a small gradient bar inline
   with its 0/1 end labels */
.ramp-bar {{ display: inline-block; width: 140px; height: 10px;
  border: 1px solid var(--transect-color-border); border-radius: 2px;
  vertical-align: middle; margin: 0 0.35rem; }}
h5 {{ font-size: var(--transect-font-size-smaller); font-weight: 500;
  color: var(--transect-color-muted); margin: 1rem 0 0.1rem; }}
/* collapsed reliability flags: summary = the scannable headline, the
   threshold/remediation prose only unfolds on demand */
details.flag {{ margin: 4px 0; }}
details.flag summary {{ cursor: pointer; }}
details.flag p {{ margin: 0.4rem 0 0; font-weight: 400;
                 font-size: 0.85rem; }}
/* the card summary's classification container: label + confidence,
   neutral until a reliability issue fires, then the issue-chip red
   palette + warning glyph (never colour alone) */
.class-box {{ color: var(--transect-color-text);
  background: var(--transect-color-hover);
  border: 1px solid var(--transect-color-border-subtle); font-size: 0.78rem;
  font-weight: 400; border-radius: 0.25rem; /* [bootstrap] badge radius */
  padding: 0.05rem 0.4rem; }}
.class-box-flagged {{ color: #b00a1f; background: #fdeaee;
  border-color: #f5c6cd; }}  /* matches .warning */
.label-checks {{ display: inline-flex; flex-wrap: wrap; gap: 0.15rem 0.7rem;
  align-items: center; }}
.label-check {{ white-space: nowrap; }}
.label-check .swatch {{ margin: 0 0.25rem 0 0.3rem; }}
/* fixed "Back to top" (bottom-right; hidden until scrolled) - filled
   with the report's link blue so it stands out over page content */
#transect-back-top {{ display: none; position: fixed; right: 1.5rem;
  bottom: 1.5rem; z-index: 10; font-family: inherit;
  font-size: var(--transect-font-size-small); color: #fff;
  background: var(--transect-color-link); border: 1px solid var(--transect-color-link);
  border-radius: var(--transect-radius); padding: 0.35rem 0.7rem;
  cursor: pointer; box-shadow: 0 1px 3px rgba(33, 37, 41, 0.25); }}
#transect-back-top:hover {{ background: var(--transect-color-link-hover);
  border-color: var(--transect-color-link-hover); }}
.event {{ background: #fff; border: 1px solid var(--transect-color-border-subtle);
         border-radius: var(--transect-radius); padding: 0.625rem; margin: 6px 0; }}
/* verbatim prompt text in the eval-setup expandables: preserve the
   prompt's own line breaks, never reflow it into a paragraph */
.prompt-verbatim {{ white-space: pre-wrap; font-size: 0.8rem;
  color: var(--transect-color-muted); margin: 0; }}
.event summary {{ cursor: pointer; margin: -0.625rem -0.625rem 0.625rem;
                  padding: 0.5rem 0.625rem;
                  font-size: var(--transect-font-size-small); font-weight: 600;
                  border-bottom: 1px solid var(--transect-color-border-subtle); }}
.event summary:hover {{ background: var(--transect-color-hover); }}
/* a closed card is just its summary bar: cancel the body padding the
   open state needs, so stacked collapsed expandables carry no empty
   strip under each summary */
.event:not([open]) > summary {{ margin-bottom: -0.625rem; border-bottom: none; }}
.event summary .meta {{ font-weight: 400; }}
.event ul {{ margin: 6px 0; }}
.definitions {{ color: var(--transect-color-muted); font-size: 0.85rem; }}
/* Flush/intervention/spawn event lists: .event card chrome with the
   Phase-definitions typography (muted 0.85rem body, semibold muted
   summary via .event summary + the color rules below). A second class,
   not a .definitions reuse: that one stays scoped to the definitions
   dropdown itself. */
.event-list {{ color: var(--transect-color-muted); font-size: 0.85rem; }}
.event-list summary {{ color: var(--transect-color-muted); }}
.event-list ol {{ margin: 6px 0; padding-left: 1.4rem; }}
.event-list li {{ margin: 3px 0; }}
.intervention-text {{ white-space: pre-wrap; max-height: 14em; overflow: auto;
                      margin: 2px 0 4px; color: var(--transect-color-text); }}
/* per-spawn rows in the spawn-prompts expandable: closed = one
   ellipsis-clamped line, open = the full task text below it */
.spawn-item {{ margin: 3px 0 3px 0.4rem; }}
.spawn-item summary {{ cursor: pointer; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
  font-size: var(--transect-font-size-smaller); }}
.spawn-item summary:hover {{ background: var(--transect-color-hover); }}
.spawn-item .prompt-verbatim {{ margin: 0.3rem 0 0.7rem 1.1rem;
                                max-height: 14em; overflow: auto; }}
.chip {{ display: inline-flex; align-items: center; gap: 4px;
        margin: 0 8px 4px 0; font-size: var(--transect-font-size-smaller);
        color: var(--transect-chip-text); background: var(--transect-chip-bg);
        border: 1px solid var(--transect-chip-border); border-radius: 6px;
        padding: 3px 8px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px;
          border-radius: 2px; }}
.swatch-line {{ display: inline-block; width: 16px; height: 0;
               border-top: 2px solid; vertical-align: middle;
               margin-right: 4px; }}
/* each section reads as a panel: framed box with its h3 title as a
   header strip (steel-tinted wash, navy ink); a titleless section
   keeps the frame only */
/* brand mark: header (beside the h1) and provenance footer */
h1 .brand-logo {{ height: 30px; vertical-align: -7px; margin-right: 6px; }}
.report-footer {{ display: flex; align-items: center; gap: 8px;
  margin: 2.5rem 0 0.8rem; padding-top: 0.9rem;
  border-top: 1px solid var(--transect-color-border-subtle);
  color: var(--transect-color-muted); font-size: 0.8rem; }}
.report-footer img {{ height: 18px; }}
.section {{ background: #fff; border: 1px solid var(--transect-color-border);
  border-radius: 6px; padding: 1rem 1.25rem 1.25rem; margin: 0 0 1.5rem; }}
.section > h3:first-child {{ margin: -1rem -1.25rem 1rem;
  padding: 0.55rem 1.25rem; background: #eef3f6; color: #0E2841;
  border-bottom: 1px solid var(--transect-color-border);
  border-radius: 6px 6px 0 0; }}
/* protective marking: top and bottom of the viewport, every page in
   print; body padding keeps content clear of the fixed banners */
.sensitivity-banner {{ position: fixed; left: 0; right: 0; z-index: 1000;
  text-align: center; font-weight: 700; font-size: 0.8rem;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: #fff; background: #F90D29; padding: 3px 0; }}
.sensitivity-top {{ top: 0; }}
.sensitivity-bottom {{ bottom: 0; }}
body.classified {{ padding-top: 30px; padding-bottom: 30px; }}
.filter-bar {{ display: flex; gap: 10px; align-items: center; margin: 8px 0;
              flex-wrap: wrap; font-size: 0.85rem;
              color: var(--transect-color-muted); }}
.filter-bar select {{ font-family: inherit; font-size: 0.85em;
                      color: var(--transect-color-text); padding: 2px 6px;
                      border: 1px solid var(--transect-color-border);
                      border-radius: var(--transect-radius); }}
.filter-bar select:focus {{ outline: none; border-color: var(--transect-color-accent);
                            box-shadow: 0 0 0 0.25rem rgba(9, 164, 167, 0.25); }}
.filter-bar input[type="checkbox"] {{ accent-color: var(--transect-color-link); }}
.filter-bar label {{ display: inline-flex; align-items: center; gap: 4px; }}
/* the "showing N of M" line: a full-width flex row of its own, so it
   always starts at the bar's left edge instead of floating mid-row
   when the bar wraps */
.filter-bar .count {{ font-size: 0.85em; color: var(--transect-color-muted);
                      flex-basis: 100%; }}
.excerpt-list {{ margin: 4px 0 6px; padding-left: 0; }}
.excerpt-list li {{ list-style: none;
                   border-left: 2px solid var(--transect-color-border-subtle);
                   padding-left: 8px; margin: 5px 0; font-size: 0.9rem; }}
.excerpt-list li > .meta:first-child {{ display: block; }}
.excerpt-list .excerpt-thinking {{ display: block; font-style: italic;
                                  color: var(--transect-color-muted); }}
"""

# `.definitions`/`.event-list` scope the muted small font away from
# `.event` itself, which the phase cards share at body size. `h3`'s
# font-size must stay pinned: an unstyled h3 (~1.17em browser default)
# would render larger than the explicit `h2` rule.

# Spliced into every embedded chart document's head (`embed._embed`'s
# `extra_head`) - inert in a document with no widget. Constraints
# (browser-verified):
#
# - `vconcat` renders one `.mosaic-widget` whose rows are class-less
#   `<div>` siblings with zero margin; `div:has(> fieldset)` is the only
#   way to target the input rows. The spacing rule needs `!important`:
#   inspect_viz writes its stylesheet into `<body>`, which follows
#   `<head>` and so wins on source order.
# - `embed._WIDGET_ROW_HEIGHT` and the `margin-bottom` here came from one
#   measurement; change them together.
# - Literal hex, not `var(--transect-*)`: an embedded document cannot see the
#   outer page's `:root`.
# - tom-select's dropdown (`dropdownParent: "body"`) and the tippy
#   popover escape `.mosaic-widget`, hence the page-level selectors.
# - inspect_viz's `select()` always wraps in Tom Select; a
#   `.mosaic-widget select` rule paints nothing.
# - The radios stay real visible inputs (no Bootstrap `.btn-check`
#   hiding): browser automation clicks `input[type="radio"]` directly.
# - The tippy box's font-size stays untouched: `embed.py`'s measured
#   tooltip-row heights were derived against the unset size.
_WIDGET_STYLE_CSS = (
    "<style>.mosaic-widget, .mosaic-widget input, .mosaic-widget label, "
    ".mosaic-widget fieldset, .mosaic-widget legend "
    f"{{ font-family: {_FONT_STACK}; }}"
    # 0.85rem matches `_STYLE`'s `.meta`, the prose around these widgets
    ".mosaic-widget { font-size: 0.85rem; }"
    ".mosaic-widget div:has(> fieldset) { margin-bottom: 1.25rem !important; }"
    # radio pills. The legend text is styled on `fieldset` itself:
    # `radio_group`'s JS leaves "Measure:" as a bare text node (no
    # `<legend>` element exists), so `fieldset > legend` matches
    # nothing; the `<label>` children are reset back to normal below.
    ".mosaic-widget fieldset { gap: 0.6rem !important; font-weight: 600; "
    "color: rgba(33, 37, 41, 0.75); }"
    ".mosaic-widget fieldset > label { font-weight: 400; color: #212529; "
    "border: 1px solid #dee2e6; "
    "border-radius: 999px; padding: 2px 10px 2px 6px; "
    "transition: background-color .1s, border-color .1s; }"
    ".mosaic-widget fieldset > label:hover { background: #f8f9fa; }"
    ".mosaic-widget fieldset > label:has(> input:checked) "
    "{ border-color: #104D6A; background: #e7f0f4; }"
    ".mosaic-widget fieldset > label > input[type=radio] "
    "{ accent-color: #104D6A; }"
    # tom-select's visible control: strip the vendor chrome, match the
    # outer page's `.filter-bar select` border/radius/font
    ".mosaic-widget label:has(.ts-wrapper) "
    "{ font-weight: 600; color: rgba(33, 37, 41, 0.75); }"
    ".mosaic-widget .ts-wrapper { font-weight: 400; color: #212529; "
    "min-height: 26px !important; }"
    # the dropdown's max-height cap lives in embed._embed, computed per
    # document from the iframe's own height (a page-level rule - the
    # dropdownParent: "body" popup escapes .mosaic-widget)
    ".mosaic-widget .ts-control, "
    ".mosaic-widget .ts-wrapper.single .ts-control { "
    "background: #fff !important; background-image: none !important; "
    "box-shadow: none !important; border: 1px solid #dee2e6 !important; "
    "border-radius: 3px !important; padding: 2px 28px 2px 8px !important; "
    "font-family: inherit !important; font-size: 0.85rem !important; "
    # rem, not em: an em would compound against `.mosaic-widget`'s own
    # 0.85rem font-size
    "min-height: unset !important; }"
    ".mosaic-widget .ts-wrapper.single .ts-control:hover "
    "{ border-color: #b8c3cc !important; }"
    ".mosaic-widget .ts-wrapper.single.focus .ts-control, "
    ".mosaic-widget .ts-wrapper.single.dropdown-active .ts-control { "
    "border-color: #09A4A7 !important; "
    "box-shadow: 0 0 0 0.2rem rgba(9, 164, 167, 0.25) !important; }"
    # tom-select's dropdown popup: page-level, `dropdownParent: "body"`
    # escapes `.mosaic-widget`
    ".ts-dropdown { background: #fff !important; "
    "border: 1px solid #dee2e6 !important; border-radius: 3px !important; "
    "box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important; "
    f"font-family: {_FONT_STACK} !important; font-size: 0.85rem !important; }}"
    ".ts-dropdown .option, .ts-dropdown .optgroup-header, "
    ".ts-dropdown .no-results, .ts-dropdown .create "
    "{ padding: 4px 10px !important; }"
    ".ts-dropdown .active "
    "{ background-color: #f8f9fa !important; color: #212529 !important; }"
    ".tippy-box[data-theme~='inspect'] { font-family: "
    f"{_FONT_STACK} !important; background-color: #f8f9fa !important; "
    "border: 1px solid #dee2e6 !important; border-radius: 4px !important; "
    "box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important; "
    "filter: none !important; }"
    ".tippy-box[data-theme~='inspect'] .tippy-arrow:before "
    "{ color: #f8f9fa !important; }"
    "</style>"
)
