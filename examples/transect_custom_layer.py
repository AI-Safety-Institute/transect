"""Research-skills custom layer: one skill label per reasoning turn.

Run from the repo root (needs ANTHROPIC_API_KEY):

    python examples/transect_custom_layer.py
"""

from inspect_ai.model import Model
from inspect_scout import Scanner, Transcript, scanner

from transect import (
    Layer,
    Spec,
    cohort_llm_scanner,
    reasoning_turns,
    transect,
    turns_frame,
)
from transect.report import Markdown, TurnBand


@scanner(loader=reasoning_turns())
def research_skills(
    spec: Spec,  # injected by transect(): the rubric lives in spec.extra
    judge_models: str | Model | list[str | Model] | None = None,
    k_rolls: int = 1,
    verify: bool | None = None,
    verify_sample: float | None = None,
) -> Scanner[Transcript]:
    """One research-skill label per reasoning turn."""
    skills: dict[str, str] = spec.extra["research_skills"]
    question = (
        "Above is one reasoning turn of an agent, with any sub-agent task "
        "it wrote behind a [DELEGATES] marker. Label the turn with the "
        "best-fitting research-skill category:\n"
        + "\n".join(f"- {label}: {text}" for label, text in skills.items())
        + "\n\nJudge only the text shown. A turn often holds several moves - "
        "label the last substantive one. A [DELEGATES] line counts as the "
        "delegated task's purpose (e.g. delegating a disconfirmation test "
        "= falsification)."
    )
    return cohort_llm_scanner(
        question=question,
        answer=list(skills),
        models=judge_models,
        k_rolls=k_rolls,
        verify=verify,
        verify_sample=verify_sample,
        vocabulary=skills,
    )


results = transect(
    "examples/logs",
    "examples/spec.yaml",
    judge_models="anthropic/claude-sonnet-4-6",
    scans_dir="examples/scans/research_skills",
    title="House-price demo - research skills",
    extra_layers=[
        Layer(
            name="research_skills",
            scanner=research_skills,  # factory: spec + judge roster come from transect()
            frame=turns_frame,  # results -> one judged row per turn
            # phase-card chips: a "skill=<label>" chip on every card whose
            # turn range carries this layer's label, a filter per family,
            # and a "Group by: skill" option in Token spend
            tags={"skill": "label"},
            # an entity block in the reliability audit, keyed by the
            # frame's (unit column, label column)
            audit=("turn", "label"),
            section=[
                Markdown(
                    "#### Research-skill per reasoning turn\n\n"
                    "One research-skill label per orchestrator reasoning turn "
                    "(own text plus each sub-agent spawn task)."
                ),
                TurnBand(label="label"),
            ],
        )
    ],
)

print("\nreport:", results.report_paths[0])
if results.viewer_url:
    print("viewer:", results.viewer_url)
skills = results.layer_frames["research_skills"]
print(
    "\nskill labels over the run's reasoning turns:\n\n"
    + skills.label.value_counts().to_string()
)
print(
    "\nexplore the layer in a notebook (re-declare the layer and pass it\n"
    "to load - it remounts the frame from the stored scan, no rescan):\n\n"
    "    from transect import load\n"
    '    results = load("examples/scans/research_skills", extra_layers=[layer])\n'
    '    results.layer_frames["research_skills"]  # judged per-turn frame\n'
    "    results.turn_tags                        # skill tags, per turn\n"
)
