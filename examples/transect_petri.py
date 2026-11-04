"""A Petri audit: the auditor on the turn axis, the target's branches in
the report.

Runs the shipped scripted audit structurally (no judge, no API key):

    uv run python examples/transect_petri.py

`--full-text` renders every exchange card whole; `--judge` labels the
auditor's phases with a judge model (needs ANTHROPIC_API_KEY).
"""

import argparse

from transect import Spec, transect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full-text", action="store_true", help="whole target replies")
parser.add_argument("--judge", action="store_true", help="label the auditor's phases")
args = parser.parse_args()
judge = {"judge_models": "anthropic/claude-sonnet-4-6"} if args.judge else {}
results = transect(
    "examples/petri_logs",
    spec=Spec.model_validate(
        {
            "phases": ["setup", "probe", "rollback", "closing"],
            "context": "An auditor model probes a target model's response to a "
            "user's rash business plan, rolling the target back once.",
        }
    ),
    epochs="all",
    scans_dir="examples/scans/petri",
    title="Petri demo audit",
    petri_full_text=args.full_text,
    **judge,
)

print("\nreports:", *results.report_paths, sep="\n  ")
if results.viewer_url:
    print("viewer:", results.viewer_url)
