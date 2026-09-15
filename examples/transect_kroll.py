"""Solo judge with k-rolls and a verifier.

One judge model labels each phase and sub-agent three times;
the majority vote decides. A stronger verifier re-checks the
judgements the triggers flag as doubtful (low confidence or low
agreement across rolls).

Run from the repo root (needs ANTHROPIC_API_KEY):

    python examples/transect_kroll.py
"""

from transect import transect

results = transect(
    "examples/logs",
    "examples/spec.yaml",
    sample="house-price",
    epochs=1,
    judge_models="anthropic/claude-sonnet-4-6",
    k_rolls=3,
    verify=True,
    verifier_model="anthropic/claude-opus-4-7",
    scans_dir="examples/scans/kroll",
    title="House-price demo - solo k-roll + verifier",
)

print("\nreport:", results.report_paths[0])
if results.viewer_url:
    print("viewer:", results.viewer_url)
print(
    "\nexplore the dataframes in a notebook:\n\n"
    "    from transect import load\n"
    '    results = load("examples/scans/kroll")\n'
    "    frames = results.frames()\n"
    '    frames["phases"]          # labelled, narrated phases\n'
    '    frames["subagents"]       # sub-agent classifications\n'
    '    frames["token_timeline"]  # per-turn token usage\n'
)
