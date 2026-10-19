"""Chart embedding: one section's components -> one fragment of the report page.

Every chart section is a fragment of the report's one `HtmlPage`
(`render.render_report` makes it and puts its `head()` in the
template's ``<head>``), so the page runs one widget runtime and one
DuckDB database however many sections it has, and embeds each table
once. No chart builder imports appear here.
"""

from inspect_viz import Component
from inspect_viz.layout import vconcat
from inspect_viz.plot import HtmlPage


def embed_section(page: HtmlPage, components: list[Component]) -> str:
    """One chart section's components -> that section's fragment of ``page``.

    ``components`` is what a builder returned, in draw order: one plot,
    or a plot plus what belongs with it (`token_stack`'s radio widgets
    above its chart, `phase_band`'s agreement strip below its band). They
    are `vconcat`-ed into one tree and embedded together.

    Params and selections are shared by id across every fragment on the
    page, so an input reaches any plot that reads its target. What keeps
    one section's inputs off another's charts is that each builder makes
    its own (`Param()` and `Selection.single()` mint a fresh id); a
    module-level `Param` or `Selection` would link every section that
    reads it.

    The fragment sits in a ``chart`` box inside a ``chart-scroll``
    wrapper (`style._CHART_CSS`). By default a chart wider than the
    wrapper scales down to fit; in the zoomed view it keeps its authored
    width and scrolls inside the wrapper instead of forcing the page to
    scroll. Once a zoomed reader scrolls right, what a chart draws into
    its left margin (the swimlane row labels) scrolls away with it; the
    input rows stay in view, and `sections`' legends carry the same
    label/swatch pairs.
    """
    fragment = page.fragment(vconcat(*components))
    return f'<div class="chart-scroll"><div class="chart">{fragment}</div></div>'
