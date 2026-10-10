"""Closed-label keys shared by scanners and stored-vocabulary projections."""

from collections.abc import Sequence


def normalize_label(raw: object) -> str | None:
    """The existing closed-label convention: lowercase, whitespace to underscores."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return "_".join(raw.strip().lower().split())


def normalize_labels(
    labels: Sequence[str], *, allow_duplicates: bool = False
) -> list[str]:
    """Normalize a roster without merging distinct labels.

    Stored vocabularies may repeat an identical original label; their frame
    projection already keeps the first definition. New answer rosters reject
    all duplicates, and distinct originals that normalize alike always raise.
    """
    seen: dict[str, str] = {}
    result = []
    for raw in labels:
        label = normalize_label(raw)
        if label is None:
            raise ValueError(f"labels must be nonempty strings, got {raw!r}")
        if label in seen and (not allow_duplicates or seen[label] != raw):
            raise ValueError(
                f"labels {seen[label]!r} and {raw!r} collide as {label!r} after "
                "normalization; use distinct labels and re-scan ambiguous stores"
            )
        seen[label] = raw
        result.append(label)
    return result


def normalize_vocabulary(
    entries: Sequence[dict], *, allow_duplicates: bool = False
) -> list[dict]:
    """Copy rubric entries with normalized keys, leaving definitions untouched."""
    labels = normalize_labels(
        [entry["label"] for entry in entries], allow_duplicates=allow_duplicates
    )
    return [
        {**entry, "label": label} for entry, label in zip(entries, labels, strict=True)
    ]
