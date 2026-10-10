"""Diagnostic helpers for the reliability iteration loop.

The scrambled-vocabulary check measures sensitivity to label names
under a changed prompt. Compare it with unchanged-setup repeats and inspect
prompt coherence and label prevalence before attributing a mechanism. Stable
or changed labels alone establish neither definition-reading nor name anchoring.
The returned labels are diagnostic artifacts, not findings about the transcript.
"""

import random

import pandas as pd

from transect.scanners.phases_common import _DEFAULT_OPS_PHASE, _NONE_PHASE
from transect.scanners.subagents import NONE_OF_THE_ABOVE
from transect.spec import Spec


def scramble_spec(spec: Spec, seed: int) -> tuple[Spec, dict[str, dict[str, str]]]:
    """A label-scrambled copy of ``spec`` plus the descramble mapping.

    Non-reserved phase and sub-agent labels are replaced with neutral tokens
    (``phase_a`` / ``role_a``, ..., neutral-token assignment shuffled by ``seed``);
    descriptions, ``context``, and the ``ops`` flag are untouched, so
    the judges see identical definitions. The phase name ``ops`` and each
    surface's ``none_of_the_above`` keep their names: those names control
    implicit category resolution. A custom-named ``ops=True`` phase is renamed
    but retains its role.

    Returns:
        ``(scrambled, mapping)`` - mapping has one dict per surface
        (``"phases"`` / ``"subagents"``), original label -> token,
        with declared reserved names mapped to themselves, for `descramble`.
    """
    rng = random.Random(seed)

    def tokens(prefix: str, labels: list[str], reserved: set[str]) -> dict[str, str]:
        # base-26 suffixes (a..z, aa, ab, ...) keep tokens alphabetic past 26 labels
        def suffix(i: int) -> str:
            out = ""
            i += 1
            while i:
                i, rem = divmod(i - 1, 26)
                out = chr(ord("a") + rem) + out
            return out

        renamed_labels = [label for label in labels if label not in reserved]
        letters = [suffix(i) for i in range(len(renamed_labels))]
        rng.shuffle(letters)
        renamed = {
            label: f"{prefix}_{letter}"
            for label, letter in zip(renamed_labels, letters, strict=True)
        }
        return {label: renamed.get(label, label) for label in labels}

    phase_map = tokens(
        "phase",
        [p.label for p in spec.phases],
        {_DEFAULT_OPS_PHASE.label, _NONE_PHASE.label},
    )
    role_labels = [r.label for r in spec.subagent_labels]
    role_map = tokens(
        "role",
        role_labels,
        {
            label
            for label in role_labels
            if "_".join(label.strip().lower().split()) == NONE_OF_THE_ABOVE
        },
    )
    scrambled = spec.model_copy(
        update={
            "phases": [
                p.model_copy(update={"label": phase_map[p.label]}) for p in spec.phases
            ],
            "subagent_labels": [
                r.model_copy(update={"label": role_map[r.label]})
                for r in spec.subagent_labels
            ],
        }
    )
    return scrambled, {"phases": phase_map, "subagents": role_map}


def descramble(labels: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map a scrambled run's label column back to the original names.

    ``mapping`` is one surface's dict from `scramble_spec` (original
    -> token). Declared reserved names map to themselves. Values outside
    the mapping, including automatically added reserved buckets and NaN,
    pass through unchanged.
    """
    inverse = {token: original for original, token in mapping.items()}
    return labels.map(lambda v: inverse.get(v, v) if isinstance(v, str) else v)
