"""Chart embedding: each section document carries only its own tables."""

import html
import json
import re

import pandas as pd
from inspect_viz import Data
from inspect_viz.mark import dot
from inspect_viz.plot import plot

from transect.report.embed import embed_section


def test_section_document_embeds_only_the_tables_it_reads() -> None:
    """A section carries the tables its chart reads, not every table in the process."""
    used = Data.from_dataframe(pd.DataFrame({"x": [1], "y": [2]}))
    Data.from_dataframe(pd.DataFrame({"x": [3], "y": [4]}))
    document = html.unescape(
        embed_section([plot(dot(used, x="x", y="y"))], height_px=200, extra_head="")
    )

    match = re.search(r"const state = (.+);$", document, re.MULTILINE)
    assert match is not None, "chart bootstrap state not found"
    assert set(json.loads(match[1])["tables"]) == {used.table}
