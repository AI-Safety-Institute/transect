"""Solo judge with k-rolls and a verifier.

One judge model labels each phase and sub-agent three times;
the majority vote decides. A stronger verifier re-checks the
judgements the triggers flag as doubtful (low confidence or low
agreement across rolls).

Run from the repo root (needs ANTHROPIC_API_KEY):

    uv run python examples/transect_kroll.py

`--structural` runs the built-in structural scanners only - no
judge, no API key, no model calls; the judged report sections
render empty:

    uv run python examples/transect_kroll.py --structural
"""

import sys

from transect import transect

structural = "--structural" in sys.argv
scans_dir = "examples/scans/structural" if structural else "examples/scans/kroll"
judge = (
    {}
    if structural
    else {
        "judge_models": "anthropic/claude-sonnet-4-6",
        "k_rolls": 3,
        "verify": True,
        "verifier_model": "anthropic/claude-opus-4-7",
    }
)
results = transect(
    "examples/logs",
    "examples/spec.yaml",
    sample="house-price",
    epochs=1,
    scans_dir=scans_dir,
    title="House-price demo - structural only"
    if structural
    else "House-price demo - solo k-roll + verifier",
    **judge,
)

print("\nreport:", results.report_paths[0])
if results.viewer_url:
    print("viewer:", results.viewer_url)
print(
    "\nexplore the dataframes in a notebook:\n\n"
    "    from transect import load\n"
    f'    results = load("{scans_dir}")\n'
    "    frames = results.frames()\n"
    '    frames["phases"]          # labelled, narrated phases\n'
    '    frames["subagents"]       # sub-agent classifications\n'
    '    frames["token_timeline"]  # per-turn token usage\n'
)
