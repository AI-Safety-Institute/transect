"""Chart embedding: one section's components -> one sized iframe.

Owns everything about getting an inspect_viz component tree onto the
page: the srcdoc iframe (+ scroll wrapper), the section height budget
(chart px + widget rows + the tooltip-fit floor), and the two in-iframe
head splices every chart document needs - pointer-transparent tooltips
and the tooltip clamp. No chart builder imports appear here; charts.py
imports this layer, never the reverse.
"""

import html as html_escape
import json

from inspect_viz import Component, Data
from inspect_viz.layout import vconcat


def embed_section(components: list[Component], height_px: int, extra_head: str) -> str:
    """One chart section's components -> that section's single iframe.

    ``components`` is what a builder returned, in draw order: one plot,
    or a plot plus what belongs with it (`token_stack`'s radio widgets
    above its chart, `phase_band`'s agreement strip below its band). They
    are `vconcat`-ed into one tree and embedded together, so a section's
    chart and its inputs stay in one document - a `Param` bound to a
    `radio_group` reaches its plot the same way a `Selection` does, i.e.
    not across an iframe boundary. ``extra_head`` passes through to
    `_embed`.
    """
    return _embed(vconcat(*components), height_px, extra_head=extra_head)


def _embed(component: Component, height_px: int, extra_head: str) -> str:
    """One chart section's component tree -> one iframe.

    `_document` returns a complete document (inspect_viz's `to_html`
    shape), so every embedded chart is a whole srcdoc document.

    `_TIP_POINTER_CSS` and `_TIP_CLAMP_HEAD` are spliced in
    unconditionally (hovering does not work without the first, tooltips
    render clipped without the second), followed by the caller's
    ``extra_head``. All three go into the document's ``<head>`` before
    escaping, as a plain string replace against the one ``</head>``
    `_document` emits - safe because we control the document's shape.

    The iframe is wrapped in a ``<div style="overflow-x:auto">`` so a
    wide chart scrolls inside its own box instead of forcing the outer
    page to scroll horizontally. For that wrapper to have anything to
    scroll, the iframe is sized to this section's widest plot
    (`_widest_plot`) rather than ``width: 100%``: a `width: 100%` iframe
    can never overflow its wrapper, so the only thing that could give was
    the chart's own scale (see `_TIP_POINTER_CSS` for the `max-width`
    half of the same fix). A chart narrower than the page renders at its
    authored width with space to its right rather than stretching.

    The trade-off: once a reader scrolls right, everything drawn into the
    chart's left margin scrolls away with it - the swimlane row labels
    and the widget row. Sticky positioning cannot help, since both live
    inside the one scrolled document. It degrades rather than breaks:
    marks stay colour-coded, `sections`' legend carries the same
    label/swatch pairs, and nothing is hidden at the unscrolled position.
    """
    doc = _document(component)
    dropdown_cap = (
        "<style>.ts-dropdown .ts-dropdown-content "
        f"{{ max-height: {max(80, height_px - _DROPDOWN_OFFSET)}px !important; }}"
        "</style>"
    )
    head = f"{_TIP_POINTER_CSS}{_TIP_CLAMP_HEAD}{dropdown_cap}{extra_head}"
    doc = doc.replace("</head>", f"{head}</head>", 1)
    width = _widest_plot(component.config)
    width_css = f"{width}px" if width else "100%"
    iframe = (
        f'<iframe height="{height_px}" style="width:{width_css};border:none" '
        f'srcdoc="{html_escape.escape(doc, quote=True)}"></iframe>'
    )
    return f'<div class="{CHART_SCROLL_CLASS}" style="overflow-x:auto">{iframe}</div>'


def _document(component: Component) -> str:
    """The component as a complete HTML document, with only the tables it
    reads.

    inspect_viz's `to_html` inlines every tracked `Data`, so a section's
    document would carry every other section's tables and, across renders
    in one process, earlier renders' tables (meridianlabs-ai/inspect_viz#39).
    This mirrors `to_html` without that sweep; replace with `to_html` once
    meridianlabs-ai/inspect_viz#40 is released.
    """
    if not component.spec:
        component.spec = component._create_spec()
    referenced = _referenced_tables(component.spec)
    tables = {
        data.table: data._data
        for data in Data._get_all()
        if data._data and data.table in referenced
    }
    snippet = component._quarto_html(tables_override=tables)
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        f"<body>{snippet}</body></html>"
    )


def _referenced_tables(spec: str) -> set[str]:
    """Names of the tables a spec reads from: its ``from`` values, the only
    key inspect_viz writes a table name into."""
    tables: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            source = node.get("from")
            if isinstance(source, str):
                tables.add(source)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(spec))
    return tables


def _widest_plot(config: object) -> int:
    """The largest ``width`` any plot in this component tree was built
    with, or 0 if none declares one.

    `_embed` needs the section's real pixel width to size its iframe, and
    reading it back off the built components guarantees it agrees with
    what the chart renders rather than restating arithmetic the builder
    already did. A section can hold more than one plot, both deriving
    their width from the same `n_turns`; taking the max rather than the
    first means a future disagreement can only leave spare room in the
    iframe, never clip the widest plot. Input widgets declare no width.
    """
    if isinstance(config, dict):
        # skip the mark list under a plot's own "plot" key: a mark option
        # named "width" there would be a channel, not a pixel width
        widths = [_widest_plot(value) for key, value in config.items() if key != "plot"]
        own = config.get("width")
        if isinstance(own, int):
            widths.append(own)
        return max(widths, default=0)
    if isinstance(config, list):
        return max((_widest_plot(item) for item in config), default=0)
    return 0


# The class on `_embed`'s horizontal-scroll wrapper, and the attribute the
# outer report page writes onto each chart iframe: the visible box of that
# iframe, in the iframe's own coordinates, as ``"left,top,right,bottom"``.
#
# This is how the clamp inside an embedded document learns about a clip
# boundary living in the embedding page (`_TIP_CLAMP_HEAD`, point 2): the
# wrapper scrolls the iframe, so a partially-scrolled wrapper shows a
# sub-box of it, and a popover the tooltip library places comfortably
# inside the iframe can still be cut off by that wrapper (measured at
# scrollLeft=400: 60px past the wrapper's left edge on one hover).
#
# Each side measures its own document and nothing else: the outer page
# writes this attribute (it owns the wrapper and the window), the child
# reads it as a plain string - no cross-document layout. The writer is
# `report.html.j2`'s `_transectChartVisibleBounds`; a missing or malformed
# attribute degrades to the iframe's own viewport, which is the state
# during page load and under any embedding that blocks frame access.
CHART_SCROLL_CLASS = "chart-scroll"
CHART_VISIBLE_ATTR = "data-transect-visible"


# Makes `inspect_viz`'s tippy tooltip pointer-transparent. Spliced into
# every embedded chart document by `_embed`, not left to the caller:
# hovering silently breaks without it. The library creates its one
# tooltip instance with `interactive: true`, so the popover takes pointer
# events; appearing under the pointer it becomes the hit target, the
# chart svg fires `pointerleave`, and Mosaic's `Nearest` interactor
# clears its selection on `pointerleave` - so the hover line vanishes
# the instant its own tooltip appears, and a popover left over the chart
# absorbs every later hover (measured: 1 of 8 exact-centre hovers
# resolved the cursor without this, 8 of 8 with it). The tradeoff is that
# tooltip text is no longer selectable and a link inside it is no longer
# clickable; nothing here puts a link in a tooltip. The
# `[data-tippy-root]` selector stays document-global rather than scoped:
# tippy appends its popovers at document-body level, outside the
# `.mosaic-widget` subtree, so a scoped selector would not reach them.
#
# The second rule is what lets a chart render at the width it was built
# for. Plot ships each plot's svg with a generated class carrying
# ``max-width: 100%``, so a 2043px chart stack inside a ~1520px iframe
# silently renders at 0.74 scale - every label and tick shrunk - and
# the ``overflow-x: auto`` wrapper never has anything to scroll. That
# defeats the point of `chart_width` growing with turn count.
# `.mosaic-widget svg` is enough to win on specificity
# (0,1,1 against the generated class's 0,1,0); `!important` is belt and
# braces against a future Plot stylesheet change, since the failure mode
# is silent. Scoped to `.mosaic-widget` so it cannot reach anything but
# a Mosaic-rendered plot.
#
# ``body { margin: 0 }`` goes with it: the embedded document otherwise
# keeps the browser default 8px, which would push the widest chart past
# the iframe's own width and put a redundant scrollbar *inside* the
# iframe on top of the wrapper's.
_TIP_POINTER_CSS = (
    "<style>[data-tippy-root], [data-tippy-root] * "
    "{ pointer-events: none !important; } "
    ".mosaic-widget svg { max-width: none !important; } "
    "body { margin: 0; } "
    # The embedded document must never scroll: its iframe is sized to the
    # chart's exact height/width budget, so the only thing that can
    # overflow it is a tippy popover near an edge - the clamp translates
    # the visible .tippy-box back inside, but the popper root keeps its
    # unclamped layout position, and a root hanging past the document
    # edge grows scrollable overflow, so the iframe pops its own
    # scrollbars on hover atop the wrapper's real one.
    "html, body { overflow: hidden; }</style>"
)


# Keeps every tippy popover inside the box a reader can actually see.
# Why a clamp rather than a placement option:
#
# - **The library asks tippy for a placement and then leaves it there.**
#   The re-renderer reads Plot's own tip anchor off the hidden svg tip
#   and passes it as tippy's `placement`. Plot picked that anchor to keep
#   *its* tip inside the plot frame, which is neither the same box as the
#   iframe nor the same size as the popover replacing it.
# - **popper's flip/shift cannot rescue it.** It shifts correctly on x,
#   but on y it flips only when the opposite side fits; when neither
#   fits, its fallback keeps the original placement and lets the popover
#   hang outside the boundary. On a short chart that is most hovers, not
#   an edge case (measured: 10 of 15 band hovers, 22 of 22 interventions
#   hovers, 47-50px outside the iframe).
# - **The height budget cannot fix it.** Clearance below the chart only
#   helps a popover placed below its anchor, and does nothing for the
#   second clip boundary below.
# - **The wrapper is a second clip boundary popper knows nothing about.**
#   The iframe is sized to the chart's full width and scrolled by an
#   ``overflow-x: auto`` wrapper in the outer document, so once that
#   wrapper is partially scrolled the visible window is a sub-box of the
#   iframe.
#
# Four details worth knowing before touching the clamp:
#
# 1. **It corrects `.tippy-box` but measures `[data-tippy-root]`.**
#    popper owns the root's inline `transform` and rewrites it on every
#    update, so a correction written there is overwritten or replaces
#    popper's placement rather than composing with it. Measuring the root
#    is what makes the correction a pure function of popper's placement:
#    a transform on a child does not move the parent's layout box, so the
#    root's rect is always the un-clamped position and re-running the
#    clamp computes the same offsets rather than compounding them.
#
#    Measuring the box instead wedges the page. tippy ships
#    ``.tippy-box { transition-property: transform, ... }`` at 300ms, so
#    a correction written there is animated and a `getBoundingClientRect`
#    straight after returns the pre-animation position; each observer
#    pass then reads a stale rect and re-derives a further correction,
#    running away by a fixed step per pass until the main thread
#    saturates. Nothing throws - the symptom is every browser-automation
#    call timing out with one renderer at 100% CPU. The
#    ``transition-property`` override in the clamp's own style block is
#    the belt to that braces.
#
#    For anything measuring where a tooltip landed: measure
#    ``.tippy-box``; the root reads the un-clamped position by design.
# 2. **The visible box is an intersection**, computed fresh each pass:
#    this document's viewport and whatever the embedding page publishes
#    about how much of this iframe it shows (`CHART_VISIBLE_ATTR`). When
#    the popover is larger than the result in an axis, the start edge
#    wins - pinned to the visible top/left, overflowing the far side -
#    so the loss is deterministic (the first rows stay readable) rather
#    than cut at whichever edge popper happened to pick.
#    `_section_height`'s floor keeps that case off the vertical axis.
# 3. **The correction is idempotent, not incremental.** Every pass
#    re-derives the offsets from the root rect and writes them only when
#    they differ from what is on the box, so there is no stored state to
#    go stale and no feedback from the observer's own write.
# 4. **It runs off mutations, deliberately not off scroll.** popper
#    writes the root's style on every reposition, so one
#    `MutationObserver` covers appearance, movement and content changes.
#    The cost of no scroll listener: a popover already up when the
#    wrapper scrolls is not re-clamped and can drift out of the visible
#    window until the next hover. Hiding the popover on a wrapper scroll
#    (what tippy already does for its own window scroll) is the cheaper
#    direction than re-positioning it live.
_TIP_CLAMP_HEAD = (
    "<style>.tippy-box { transform: translate(var(--transect-tip-dx, 0px), "
    "var(--transect-tip-dy, 0px)) !important; "
    "transition-property: visibility, opacity !important; }</style>"
    "<script>"
    """(() => {
  const PAD = 2;
  const readOuter = () => {
    try {
      const attr = window.frameElement
        && window.frameElement.getAttribute('"""
    f"{CHART_VISIBLE_ATTR}"
    """');
      if (!attr) return null;
      const parts = attr.split(',').map(Number);
      if (parts.length !== 4 || parts.some(isNaN)) return null;
      return {
        left: parts[0], top: parts[1], right: parts[2], bottom: parts[3]
      };
    } catch (e) {
      return null;  /* sandboxed or cross-origin: our own box is the bound */
    }
  };
  const visibleBox = () => {
    const root = document.documentElement;
    const box = {
      left: 0, top: 0, right: root.clientWidth, bottom: root.clientHeight
    };
    const outer = readOuter();
    if (outer) {
      box.left = Math.max(box.left, outer.left);
      box.top = Math.max(box.top, outer.top);
      box.right = Math.min(box.right, outer.right);
      box.bottom = Math.min(box.bottom, outer.bottom);
    }
    return box;
  };
  const clamp = () => {
    const roots = document.querySelectorAll('[data-tippy-root]');
    if (!roots.length) return;
    const vis = visibleBox();
    for (const root of roots) {
      const tip = root.querySelector('.tippy-box');
      if (!tip) continue;
      const rect = root.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      let dx = 0;
      let dy = 0;
      if (rect.right > vis.right - PAD) dx = vis.right - PAD - rect.right;
      if (rect.left + dx < vis.left + PAD) dx = vis.left + PAD - rect.left;
      if (rect.bottom > vis.bottom - PAD) dy = vis.bottom - PAD - rect.bottom;
      if (rect.top + dy < vis.top + PAD) dy = vis.top + PAD - rect.top;
      const x = dx + 'px';
      const y = dy + 'px';
      if (tip.style.getPropertyValue('--transect-tip-dx') === x
          && tip.style.getPropertyValue('--transect-tip-dy') === y) continue;
      tip.style.setProperty('--transect-tip-dx', x);
      tip.style.setProperty('--transect-tip-dy', y);
    }
  };
  const start = () => {
    new MutationObserver(clamp).observe(document.documentElement, {
      childList: true, subtree: true, attributes: true,
      attributeFilter: ['style']
    });
    /* no scroll listener here, deliberately - see point 4 above */
    window.addEventListener('resize', clamp);
  };
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();"""
    "</script>"
)


def _section_height(
    chart_px: int, tip_rows: int, widget_rows: int = 0, floor: int | None = None
) -> int:
    """One chart section's iframe height: its own plots, plus a row per
    input widget above them, plus the document-overflow pad - floored at
    what its tooltip needs (`tip_fit_height`, or ``floor`` when a caller
    passes one).

    Every chart builder returns its height through here, so the two
    rules (no inner vertical scrollbar, a tooltip that fits) are stated
    once rather than re-derived per chart. The floor is normally slack -
    a real chart is taller than its own tooltip - and bites on a chart
    whose body collapses on thin data, e.g. a 1-row swimlanes chart with
    its fixed-row span tooltip.

    ``floor`` overrides `tip_fit_height`'s uniform-per-row computation
    for a caller whose tooltip has a row that assumption does not cover
    (`charts._swimlane_tip_fit_height`).
    """
    own = chart_px + widget_rows * _WIDGET_ROW_HEIGHT + _DOC_HEIGHT_PAD
    return max(own, floor if floor is not None else tip_fit_height(tip_rows))


def tip_fit_height(tip_rows: int) -> int:
    """The smallest iframe height a chart whose tooltip has ``tip_rows``
    rows can be given and still show that tooltip whole.

    The clamp (`_TIP_CLAMP_HEAD`) keeps a popover inside the reader's
    visible box, which is a guarantee about *position*, not size: a
    popover taller than that box cannot be moved into it. So each chart
    section's height is floored at what its own tooltip needs
    (`_section_height`). Row count is a build-time fact - every tooltip
    here is an explicit `channels=` dict - so only the per-row pixel
    figure (`_TIP_ROW_PX`) comes from the browser.

    `charts.swimlanes` passes its own floor instead, covering its "lane"
    row's larger, data-dependent wrap allowance.
    """
    return tip_rows * _TIP_ROW_PX + _TIP_FIT_MARGIN


def _tip_floor(n_rows: int, tall_row_px: int, second_tall_row_px: int = 0) -> int:
    """Tooltip-fit floor for a tooltip of ``n_rows`` rows, one or two of
    which wrap: the rest are single-line rows.

    Both callers share this shape and differ only in how they size the
    tall row(s) - `_swimlane_tip_fit_height` derives them from its
    "lane" and "member votes" rows' own longest values, `phase_band`
    fixes its "label basis" row at three lines and derives "member
    votes" the same way - so the arithmetic is stated once here rather
    than re-derived at each site. ``second_tall_row_px`` is 0 when the
    tooltip has only one wrapping row.
    """
    if n_rows == 0:
        return _TIP_FIT_MARGIN
    n_tall = 2 if second_tall_row_px else 1
    return (
        (n_rows - n_tall) * _TIP_SHORT_ROW_PX
        + tall_row_px
        + second_tall_row_px
        + _TIP_FIT_MARGIN
    )


def wrap_row_px(n_chars: int) -> int:
    """Height allowance for one wrapping prose tooltip row of
    ``n_chars``: first line plus extras at the worst observed word
    packing, capped at four extra lines - past the cap a very long
    value's tail lines clip rather than growing every chart's floor,
    a deliberate trade of tooltip tail for page compactness."""
    lines = min(5, max(1, -(-min(n_chars, 200) // _PROSE_WORST_CHARS_PER_LINE)))
    return _TIP_SHORT_ROW_PX + _TIP_EXTRA_LINE_PX * (lines - 1)


def _tip_rows(channels: dict[str, str]) -> int:
    """How many rows a ``channels=`` dict actually pops as tooltip rows.

    Underscore-prefixed keys are excluded, which is not cosmetic:
    passing a dict to a mark mutates it in place - `inspect_viz` records
    its own ``_user_channels`` bookkeeping key on the caller's object -
    so a bare `len()` after the mark is built reads one row too many.
    That key never reaches the popover, so counting this way is right
    whether the dict has been handed to a mark yet or not, which keeps a
    height budget from depending on statement order.
    """
    return sum(1 for key in channels if not key.startswith("_"))


_WIDGET_ROW_HEIGHT = 54  # one select/radio_group row in the vconcat stack

# what the dropdown cap subtracts from the iframe height
_DROPDOWN_OFFSET = 70


_DOC_HEIGHT_PAD = 24  # slack over the sum of a section's own plot heights


# One tooltip row's rendered height, for the floor a chart's iframe has
# to clear so its tallest tooltip fits inside it (`tip_fit_height`).
# Real-browser measurement: an unwrapped row is 25px, a row wrapping to a
# second line is 40px (tippy caps the popover at 350px wide, so a long
# value wraps rather than widening), and the popover's height is exactly
# the sum of its rows - no separate padding to budget for. 40, not 25,
# so the floor covers every row wrapping to two lines at once. A third
# wrapped line on several rows at once is not covered, and is the honest
# limit: a popover taller than the visible box gets pinned to its top
# edge by the clamp and loses its last rows off the bottom.
#
# `charts.swimlanes` does not read this for its own floor - its "lane"
# row carries real user-supplied text, which breaks the "at most one
# wrapped line" assumption (see `charts._SWIMLANE_LANE_ROW_PX`).
_TIP_ROW_PX = 40
_TIP_FIT_MARGIN = 20


# The tighter per-row numbers behind `_tip_floor`, for tooltips whose
# rows are structurally single-line: budgeting `_TIP_ROW_PX`'s two-line
# allowance for those leaves a slim chart with hundreds of pixels of
# blank iframe below it. Real-browser measurement: a full 8-row span
# tooltip with a 2-line lane measures 186px, i.e. ~20px per single-line
# row plus box padding, so a single-line row gets 22px with headroom and
# a wrapping row gets 22px for its first line plus 16px per extra. The
# per-line estimate is the worst observed word packing (~20 chars/line);
# at the lane cap the formula reproduces `charts._SWIMLANE_LANE_ROW_PX`.
#
# Generic tooltip-row metrics, not swimlane-specific: `phase_band`'s
# agreement strip reads the same two numbers for its "label basis" row.
_TIP_SHORT_ROW_PX = 22
_LANE_WORST_CHARS_PER_LINE = 20
# joined prose rows (member votes) wrap better than lane tokens: label
# words break at separators, so the worst packing is looser than
# `_LANE_WORST_CHARS_PER_LINE`'s unbroken-token estimate
_PROSE_WORST_CHARS_PER_LINE = 30
_TIP_EXTRA_LINE_PX = 16
