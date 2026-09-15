"""Cohort of judges.

Three different judge models each label every phase and sub-agent
once; the majority vote across the cohort decides. No verifier.

Run from the repo root (needs ANTHROPIC_API_KEY and OPENAI_API_KEY):

    python examples/transect_cohort.py
"""

from transect import transect

results = transect(
    "examples/logs",
    "examples/spec.yaml",
    sample="house-price",
    epochs=1,
    judge_models=[
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.4",
    ],
    scans_dir="examples/scans/cohort",
    title="House-price demo - judge cohort",
)

print("\nreport:", results.report_paths[0])
if results.viewer_url:
    print("viewer:", results.viewer_url)
print(
    "\nexplore the dataframes in a notebook:\n\n"
    "    from transect import load\n"
    '    results = load("examples/scans/cohort")\n'
    "    frames = results.frames()\n"
    '    frames["phases"]           # labelled, narrated phases\n'
    '    frames["subagent_members"] # per-judge ballots behind each label\n'
    '    frames["token_timeline"]   # per-turn token usage\n'
)
