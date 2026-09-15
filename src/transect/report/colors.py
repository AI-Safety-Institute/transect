"""Categorical palette shared by the phase band, legend chips, and the
sub-agent swimlanes - one assignment so a phase/label keeps the same
color everywhere it appears in a report."""

import pandas as pd

# Validated 12-slot categorical palette (dataviz reference instance,
# light mode, adjacent-pair order; slot order is load-bearing. Labels
# beyond 12 reuse the hues with a hatch texture.
_PHASE_SLOTS = [
    "#1f6d9c",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#09A4A7",
    "#9e330e",
    "#948d00",
    "#9054a3",
]
_UNJUDGED_GREY = "#c9c9c9"
_UNJUDGED_BASES = ("refusal", "no_answer", "missing_turn")

# The agreement strip's single hue
_AGREEMENT_TEAL = "#0b5e60"


def _phase_colors(phases: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    """label -> (hex, hatch) for the whole report, by aggregate turn share.

    Fixed assignment across transcripts (color follows the entity);
    hatch marks the reuse tier past the palette's 12 slots."""
    if not len(phases):
        return {}
    order = phases.groupby("phase").n_turns.sum().sort_values(ascending=False).index
    return {
        label: (
            _PHASE_SLOTS[i % len(_PHASE_SLOTS)],
            None if i < len(_PHASE_SLOTS) else "//",
        )
        for i, label in enumerate(order)
    }


def _label_colors(labels: list[str]) -> dict[str, str]:
    """Label -> hex in the given order (callers pass sorted labels, so
    the assignment is stable); unclassified is always grey."""
    out = {}
    slot = 0
    for label in labels:
        if label == "unclassified":
            out[label] = _UNJUDGED_GREY
            continue
        out[label] = _PHASE_SLOTS[slot % len(_PHASE_SLOTS)]
        slot += 1
    return out


# Per-family tints for the phase cards' user tag chips
_CHIP_TINTS = [
    ("#104D6A", "#e7f0f4", "#c5dae4"),  # steel
    ("#0b6e70", "#e2f4f4", "#b9e2e3"),  # teal
    ("#7a5200", "#faf2dd", "#ecdcae"),  # amber
    ("#a30d20", "#fdeaee", "#f5c6cd"),  # red
    ("#0E2841", "#e9eef3", "#c9d3dd"),  # navy
    ("#4d5966", "#eef1f4", "#d3dae1"),  # slate
]


def chip_tint(index: int) -> str:
    """One tag family's chip style, by the family's position."""
    text, background, border = _CHIP_TINTS[index % len(_CHIP_TINTS)]
    return f"color:{text};background:{background};border-color:{border};"


# The audit maps' two cell ramps. Health: red (0, unhealthy) ->
# pale -> steel (1, healthy) - a red/blue diverging pair, deliberately
# not red->green (the classic colour-vision-deficiency trap); every
# cell also carries its value as text, so colour is never the only
# signal. Share: pale -> steel intensity only - a share has no
# good/bad polarity, so it must not wear the health ramp.
_RAMP_LOW = (212, 11, 35)  # red, deepened for cell-text contrast
_RAMP_MID = (247, 247, 247)
_RAMP_HIGH = (16, 77, 106)  # steel
_CELL_NEUTRAL = "#f4f4f4"

# relative luminance of the body ink
_DARK_INK_LUMINANCE = 0.0172


def ramp_anchors() -> tuple[str, str, str]:
    """The health ramp's low/mid/high anchors as CSS colors, for the
    audit's explanatory scale (the share ramp is the mid -> high half)."""
    low, mid, high = (
        f"rgb({r},{g},{b})" for r, g, b in (_RAMP_LOW, _RAMP_MID, _RAMP_HIGH)
    )
    return low, mid, high


def health_color(x: float | None) -> str:
    """Cell background for a 0-1 health value; neutral grey for None
    (no fact to colour)."""
    if x is None:
        return _CELL_NEUTRAL
    x = max(0.0, min(1.0, float(x)))
    if x < 0.5:
        lo, hi, t = _RAMP_LOW, _RAMP_MID, x / 0.5
    else:
        lo, hi, t = _RAMP_MID, _RAMP_HIGH, (x - 0.5) / 0.5
    r, g, b = (round(a + (b_ - a) * t) for a, b_ in zip(lo, hi, strict=True))
    return f"rgb({r},{g},{b})"


def share_color(x: float | None) -> str:
    """Cell background for a 0-1 share (intensity, no polarity);
    neutral grey for None/zero."""
    if x is None or x <= 0:
        return _CELL_NEUTRAL
    t = max(0.0, min(1.0, float(x)))
    r, g, b = (
        round(a + (b_ - a) * t) for a, b_ in zip(_RAMP_MID, _RAMP_HIGH, strict=True)
    )
    return f"rgb({r},{g},{b})"


def cell_text_color(background: str) -> str:
    """White or body ink, whichever contrasts more against the given
    cell background (an ``rgb(r,g,b)`` string from the ramps, or a hex
    neutral)."""
    if background.startswith("rgb("):
        channels = [int(v) for v in background[4:-1].split(",")]
    else:
        h = background.lstrip("#")
        channels = [int(h[i : i + 2], 16) for i in (0, 2, 4)]

    def _linear(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (_linear(v) for v in channels)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    white = 1.05 / (luminance + 0.05)
    dark = (luminance + 0.05) / (_DARK_INK_LUMINANCE + 0.05)
    return "#fff" if white > dark else "#212529"
