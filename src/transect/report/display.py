"""Shared display vocabulary: one place that turns frame values into
reader-facing text, so every surface renders the same fact the same way.
"""

import pandas as pd

# Humanised reasons keyed by the scanners' own vocabularies
# (frames/common.py BASES and CALL_STATUSES).
MEMBER_NO_VOTE = {
    "no_answer": "no answer from this judge",
    "refusal": "refused",
    "error": "errored",
    "missing_turn": "no judgement recorded",
    "filled": "no direct judgement (filled)",
}


# suffix naming who a confidence number came from; bare for one judge
CONFIDENCE_QUALIFIER = {"majority_vote": " (mean)", "verifier": " (verifier)"}


def member_display(model, roll, multi_roll: set[str], short: bool = True) -> str:
    """One cohort member's display name: model (shortened unless
    ``short=False``), with a ``roll-N`` suffix only when that model
    judged more than one roll. 0-based, matching the frames."""
    name = _short_model(model) if short else str(model)
    if str(model) in multi_roll:
        name += f" roll-{int(roll)}"
    return name


def multi_roll_models(frame: pd.DataFrame) -> set[str]:
    """Models that appear with more than one roll in a (model, roll)
    frame - the only case the roll suffix distinguishes anything."""
    if not len(frame):
        return set()
    return set(frame.groupby("model").roll.nunique().loc[lambda s: s > 1].index)


def roster(judge_models) -> str:
    """The frames' "+"-joined judge roster as display text."""
    return str(judge_models).replace("+", ", ")


def judge_label(judge_models) -> str:
    """Either "judge" or "judges" for a "+"-joined roster."""
    return "judges" if "+" in str(judge_models) else "judge"


def _short_model(model: str) -> str:
    """Provider prefix stripped: ``anthropic/claude-x`` -> ``claude-x``."""
    return str(model).rsplit("/", 1)[-1]
