"""Token views obey Inspect's normalized usage contract, without model calls."""

import pandas as pd
import pytest
from helpers import model_turn, run_scan
from inspect_ai.model import ModelUsage
from test_frames import raw_row

from transect.frames import token_timeline_df
from transect.scanners.base import token_timeline


def timeline(usages):
    result = run_scan(
        token_timeline(),
        [model_turn(f"t{i}", usage=usage) for i, usage in enumerate(usages)],
    )
    return token_timeline_df(pd.DataFrame([raw_row(result)]))


@pytest.mark.parametrize("reasoning", [0, 40])
def test_reasoning_is_a_component_of_output(reasoning):
    """A reasoning breakdown must not increase output-inclusive totals."""
    row = timeline(
        [
            ModelUsage(
                input_tokens=100,
                output_tokens=60,
                reasoning_tokens=reasoning,
                total_tokens=160,
            )
        ]
    ).iloc[0]
    assert row.reasoning_tokens == reasoning
    assert (row.new_work, row.billable, row.turn_total) == (160, 160, 160)


@pytest.mark.parametrize("cache_read", [200, 1200])
def test_normalized_input_excludes_cache_even_when_reads_are_smaller(cache_read):
    """Small cached prefixes do not change Inspect's uncached-input convention."""
    row = timeline(
        [
            ModelUsage(
                input_tokens=1000,
                output_tokens=50,
                input_tokens_cache_read=cache_read,
                total_tokens=1050 + cache_read,
            )
        ]
    ).iloc[0]
    assert row.cache_semantics == "exclusive"
    assert row.context == 1000 + cache_read
    assert row.turn_total == 1050 + cache_read
    assert row.new_work == row.billable == 1050


def test_later_cache_usage_cannot_change_earlier_derived_values():
    """Appending a cached turn does not retrospectively change existing usage."""
    first = ModelUsage(
        input_tokens=1000,
        output_tokens=50,
        input_tokens_cache_read=200,
        total_tokens=1250,
    )
    later = ModelUsage(
        input_tokens=100,
        output_tokens=50,
        input_tokens_cache_read=2000,
        total_tokens=2150,
    )
    columns = ["context", "new_work", "billable", "turn_total", "cache_semantics"]
    pd.testing.assert_series_equal(
        timeline([first]).iloc[0][columns],
        timeline([first, later]).iloc[0][columns],
    )
