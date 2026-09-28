"""Diagnostic helpers for the reliability iteration loop.

The scrambled-vocabulary check measures sensitivity to label names
under a changed prompt. Compare it with unchanged-setup repeats and inspect
prompt coherence and label prevalence before attributing a mechanism. Stable
or changed labels alone establish neither definition-reading nor name anchoring.
The returned labels are diagnostic artifacts, not findings about the transcript.
"""

import random

import pandas as pd

from transect.spec import Spec


def scramble_spec(spec: Spec, seed: int) -> tuple[Spec, dict[str, dict[str, str]]]:
    """A label-scrambled copy of ``spec`` plus the descramble mapping.

    Every phase and sub-agent label is replaced with a neutral token
    (``phase_a`` / ``role_a``, ..., neutral-token assignment shuffled by ``seed``);
    descriptions, ``context``, and the ``ops`` flag are untouched, so
    the judges see identical definitions under meaningless names.

    Returns:
        ``(scrambled, mapping)`` - mapping has one dict per surface
        (``"phases"`` / ``"subagents"``), original label -> token,
        for `descramble`.
    """
    rng = random.Random(seed)

    def tokens(prefix: str, labels: list[str]) -> dict[str, str]:
        # base-26 suffixes (a..z, aa, ab, ...) keep tokens alphabetic past 26 labels
        def suffix(i: int) -> str:
            out = ""
            i += 1
            while i:
                i, rem = divmod(i - 1, 26)
                out = chr(ord("a") + rem) + out
            return out

        letters = [suffix(i) for i in range(len(labels))]
        rng.shuffle(letters)
        return {
            label: f"{prefix}_{letter}"
            for label, letter in zip(labels, letters, strict=True)
        }

    phase_map = tokens("phase", [p.label for p in spec.phases])
    role_map = tokens("role", [r.label for r in spec.subagent_labels])
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
    -> token); values outside it - the reserved buckets
    (``none_of_the_above``, a reserved ``ops``) and NaN - pass through
    unchanged, since they were never scrambled.
    """
    inverse = {token: original for original, token in mapping.items()}
    return labels.map(lambda v: inverse.get(v, v) if isinstance(v, str) else v)
