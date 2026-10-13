"""Chart iframes load their documents lazily."""

import pandas as pd
from inspect_viz import Data
from inspect_viz.mark import dot
from inspect_viz.plot import plot

from transect.report.embed import embed_section


def test_a_chart_iframe_carries_its_document_lazily() -> None:
    """The section document rides in data-transect-srcdoc, never in srcdoc:
    the page script loads it when the iframe nears the viewport and evicts
    it when it leaves."""
    data = Data.from_dataframe(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    html = embed_section([plot(dot(data, x="x", y="y"))], 200, extra_head="")
    assert 'data-transect-srcdoc="' in html
    assert ' srcdoc="' not in html
    assert 'class="transect-lazy"' in html
