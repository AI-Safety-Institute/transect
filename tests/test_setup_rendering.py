"""Setup values retain their content and missingness in the rendered rows."""

from xml.etree import ElementTree

import pandas as pd
import pytest

from transect.report.sections import eval_setup_blocks


def _rendered_rows(values):
    markup = eval_setup_blocks(pd.DataFrame([values], dtype=object))
    root = ElementTree.fromstring(f"<div>{markup}</div>")
    rows = {}
    for row in root.findall(".//p[@class='meta']"):
        label = row.find("strong")
        assert label is not None
        rows["".join(label.itertext())] = (label.tail or "").removeprefix(": ")
    return root, rows


@pytest.mark.parametrize(
    "field,label", [("scaffold_prompt", "scaffold prompt"), ("tools", "tool roster")]
)
@pytest.mark.parametrize(
    "value", [[], [None], [False], ["first", "second"], {}, {"mode": "example"}]
)
def test_structured_setup_values_render_verbatim(field, label, value):
    """Recorded containers remain visible, including empty and false-like lists."""
    _, rows = _rendered_rows({"header_available": True, field: value})
    assert rows[label] == str(value)
    assert rows["attempts"] == "scaffold default"


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
@pytest.mark.parametrize("header_available", [True, False])
@pytest.mark.parametrize("with_args", [True, False])
def test_missing_setup_values_keep_their_absence_wording(
    value, header_available, with_args
):
    """The specific missing rows distinguish absent data and unconfigured options."""
    values = {
        "header_available": header_available,
        "scaffold_prompt": value,
        "task_file": value,
    }
    if with_args:
        values["attempts"] = 1
    _, rows = _rendered_rows(values)
    assert rows["scaffold prompt"] == (
        "scaffold default" if with_args else "data not found"
    )
    assert rows["task source file"] == (
        "not set" if header_available else "data not found"
    )


@pytest.mark.parametrize("value", ["example", False, 0])
def test_scalar_setup_values_are_not_treated_as_missing(value):
    """Recorded text, false and zero retain their existing display values."""
    _, rows = _rendered_rows({"scaffold_prompt": value})
    assert rows["scaffold prompt"] == str(value)


def test_structured_setup_content_is_escaped():
    """Markup inside a recorded instruction displays as text rather than HTML."""
    value = ["<script>example()</script>", "a & b"]
    root, rows = _rendered_rows({"scaffold_prompt": value})
    assert rows["scaffold prompt"] == str(value)
    assert root.find(".//script") is None
