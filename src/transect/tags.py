"""Tag families: user per-turn labels decorating the report.

A layer declares which columns of its per-turn frame become tag
families (`Layer.tags`); the selected families aggregate into
``results.turn_tags`` - wide, one row per (transcript_id, turn), one
column per family.
"""

import pandas as pd

from transect.frames.common import JUDGE_COLS, VERIFIER_COLS
from transect.layers import Layer

# The bookkeeping columns our own cohort-fed frames emit.
RESERVED_TAG_COLUMNS = (
    frozenset(JUDGE_COLS)
    | frozenset(VERIFIER_COLS)
    | frozenset(
        {
            "answer",
            "confidence",
            "error",
            "explanation",
            "judge_agreement",
            "judge_models",
            "label_source",
            "model",
            "span_task",
            "span_task_source",
            "status",
            "verifier_model",
        }
    )
)

# Neither turn keys nor identity columns are candidate families.
_IGNORED = frozenset(
    {
        "transcript_id",
        "turn",
        "item",
        "sample_id",
        "task_set",
        "epoch",
        "agent",
        "schema_version",
    }
)


def select_tags(layer: Layer, frame: pd.DataFrame) -> dict[str, str]:
    """Resolve one layer's tag declaration to ``{family: column}``.

    ``True`` selects every eligible column (string/categorical, not
    reserved, not identity); a list selects exactly those columns
    (overriding ``RESERVED_TAG_COLUMNS``, never the dtype fence);
    a dict also renames (``{"activity": "label"}``).
    """
    tags = layer.tags
    if tags is False or tags == [] or tags == {}:
        return {}
    where = f"layer {layer.name!r}"
    if "turn" not in frame.columns:
        raise ValueError(f"{where}: tags requires a per-turn frame (no 'turn' column)")
    if tags is True:
        families = {
            c: c
            for c in frame.columns
            if c not in _IGNORED
            and c not in RESERVED_TAG_COLUMNS
            and _taggable(frame[c])
        }
        if not families:
            raise ValueError(
                f"{where}: tags=True selected no eligible columns - a family "
                "is a non-reserved string/categorical column of the layer's "
                "per-turn frame"
            )
        return families
    mapping = {c: c for c in tags} if isinstance(tags, list) else dict(tags)
    for family, column in mapping.items():
        if family in ("transcript_id", "turn"):
            raise ValueError(f"{where}: {family!r} cannot be a family name")
        if column not in frame.columns:
            raise ValueError(f"{where}: tag column {column!r} is not in the frame")
        if not _taggable(frame[column]):
            raise ValueError(
                f"{where}: tag column {column!r} is not string/categorical - "
                "chip vocabularies are closed; bin numeric columns into "
                "categories first"
            )
    return mapping


def turn_tags_frame(
    layers: list[Layer],
    frames: dict[str, pd.DataFrame],
    n_turns: dict[str, int],
) -> pd.DataFrame | None:
    """The ``results.turn_tags`` mount: every tagging layer's selected
    families validated and merged on (transcript_id, turn).

    ``n_turns`` maps each transcript to its turn count. None when no
    layer declares tags.
    """
    family_of: dict[str, str] = {}
    mounted: pd.DataFrame | None = None
    for layer in layers:
        if not layer.tags:
            continue
        where = f"layer {layer.name!r}"
        source = frames[layer.name]
        selection = select_tags(layer, source)
        turns = source["turn"]
        if turns.isna().any() or not all(float(t).is_integer() for t in turns):
            raise ValueError(f"{where}: tag turns must be integers")
        for family in selection:
            if family in family_of:
                raise ValueError(
                    f"tag family {family!r} is declared by both layer "
                    f"{family_of[family]!r} and layer {layer.name!r}"
                )
            family_of[family] = layer.name
        piece = pd.DataFrame({"turn": turns.astype(int)})
        for family, column in selection.items():
            piece[family] = source[column].astype("string").values
        if "transcript_id" in source.columns:
            piece.insert(0, "transcript_id", list(source["transcript_id"]))
        else:
            # no transcript_id: the tags apply to every transcript
            piece = pd.concat(
                [piece.assign(transcript_id=str(tid)) for tid in n_turns],
                ignore_index=True,
            )
        if piece.duplicated(["transcript_id", "turn"]).any():
            raise ValueError(f"{where}: duplicate turn rows in the tags source")
        for tid, group in piece.groupby("transcript_id"):
            limit = n_turns.get(str(tid))
            if limit is not None and (group.turn >= limit).any():
                bad = int(group.turn[group.turn >= limit].iloc[0])
                raise ValueError(
                    f"{where}: turn {bad} out of range for transcript "
                    f"{tid} ({int(limit)} turns)"
                )
        mounted = (
            piece
            if mounted is None
            else mounted.merge(piece, on=["transcript_id", "turn"], how="outer")
        )
    if mounted is None:
        return None
    return mounted.sort_values(["transcript_id", "turn"]).reset_index(drop=True)


def _taggable(column: pd.Series) -> bool:
    """Families are string/categorical only."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.infer_dtype(column.dropna()) in ("string", "empty")
