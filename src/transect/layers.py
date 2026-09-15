"""User-injected layers: the custom-interface unit.

A Layer bundles one custom addition to a run: an optional scanner
(joins the scan batch), an optional frame (a fn over the scanner's
raw results, or a ready DataFrame), typed section blocks, and
phase-card tags. Every field is optional but a
layer must do something; users extend the stack via
``transect(..., extra_layers=[Layer(...)])``.

A judged layer scanner is expected to be built on
`transect.cohort_llm_scanner` behind the user's own ``@scanner``-decorated
factory (its own loader + closed vocabulary): that path stamps the
judge-identity block and the loader item's metadata on every result,
so the reliability machinery and the frames' item helpers
(`transect.frames.common.input_id` / `item_metadata`) work unchanged.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd
from inspect_ai._util.registry import registry_info

FrameSource = Callable[[pd.DataFrame], pd.DataFrame] | pd.DataFrame


@dataclass(frozen=True)
class Layer:
    """One user-injected layer.

    Args:
        name: The mount key (``results.layer_frames[name]``) and the
            provenance label on everything this layer renders. Must
            not collide with a built-in frame name.
        scanner: A ``@scanner``-decorated scanner instance, or the
            un-invoked factory - called at the entry point with the
            subset of the judge arguments (``judge_models`` /
            ``k_rolls`` / ``verify`` / ``verifier_model`` /
            ``verify_sample``) its signature declares. A declared
            parameter always takes the entry point's value, so
            hardcode layer-local judge choices in the factory body.
            Joins the scan batch; results feed ``frame``.
        frame: A fn mapping the scanner's raw results table to a tidy
            DataFrame, or a ready DataFrame (bring-your-own data, no
            scanner). Omitted with a scanner: the generic flatten.
        section: Typed report blocks.
        tags: Phase-card tags from this layer's per-turn frame.
        audit: ``(unit_col, label_col)`` opt in to an entity block in
            the report's reliability audit. Needs a frame carrying the
            judge-identity columns (stamped by `transect.cohort_llm_scanner`).
    """

    name: str
    scanner: Any = None
    frame: FrameSource | None = None
    section: list | None = None
    tags: bool | list[str] | dict[str, str] = field(default=False)
    audit: tuple[str, str] | None = None


def validate_layers(layers: list[Layer], builtin_frames: tuple[str, ...] = ()) -> None:
    """Reject a malformed layer list before anything is spent.

    Rules:

    - every layer has a non-empty string name
    - no duplicate layer names
    - no layer name collides with a built-in frame name
    - a layer does something: scanner, frame, section, or tags
    - a scanner is not paired with a ready DataFrame
    - every scanner has registry identity (@scanner-decorated)
    - no two layers share one scanner
    - tags is a bool, list, or dict
    - section blocks are typed blocks with well-formed fields
    - audit is two column names, on a layer that has a frame to audit
    """
    # imported at call time: a top-level import would run the report
    # package init, and the report imports Layer from this module
    from transect.report.blocks import validate_blocks_static
    from transect.report.render import SECTION_KEYS

    seen: set[str] = set()
    seen_scanners: set[str] = set()
    for layer in layers:
        where = f"layer {layer.name!r}"
        if not layer.name or not isinstance(layer.name, str):
            raise ValueError("every layer needs a non-empty string name")
        if layer.name in seen:
            raise ValueError(f"duplicate layer name {layer.name!r}")
        seen.add(layer.name)
        if layer.name in builtin_frames:
            raise ValueError(f"{where} collides with the built-in frame {layer.name!r}")
        # a layer name is also its section_order key, so the report
        # section keys are part of the same namespace
        if layer.name in SECTION_KEYS:
            raise ValueError(
                f"{where} collides with the report section key {layer.name!r}"
            )
        if not isinstance(layer.tags, (bool, list, dict)):
            raise ValueError(
                f"{where}: tags is a bool, a list of family columns, or a "
                "family-to-column dict"
            )
        has_any = (
            layer.scanner is not None
            or layer.frame is not None
            or layer.section
            or (layer.tags is not False and layer.tags != [] and layer.tags != {})
        )
        if not has_any:
            raise ValueError(
                f"{where} is empty - give it a scanner, a frame, a section, or tags"
            )
        if layer.scanner is not None and isinstance(layer.frame, pd.DataFrame):
            raise ValueError(
                f"{where}: pass either a scanner (with an optional frame fn) "
                "or a ready DataFrame, not both - a ready frame cannot be "
                "fed by a scanner"
            )
        if layer.scanner is not None:
            key = scanner_key(layer.scanner)  # raises with the fix named
            if key in seen_scanners:
                raise ValueError(
                    f"two layers share the scanner {key!r} - the scan store "
                    "keys results by scanner name, so each layer needs its "
                    "own @scanner-decorated factory"
                )
            seen_scanners.add(key)
        if layer.section:
            validate_blocks_static(layer)
        if layer.audit is not None:
            malformed = (
                not isinstance(layer.audit, tuple)
                or len(layer.audit) != 2
                or not all(isinstance(c, str) and c for c in layer.audit)
            )
            if malformed:
                raise ValueError(
                    f"{where}: audit is (unit_col, label_col) - two column "
                    "names over the layer's frame"
                )
            if layer.scanner is None and layer.frame is None:
                raise ValueError(
                    f"{where}: audit needs a frame to audit - give the "
                    "layer a scanner or a frame"
                )


def resolve_scanner_factories(layers: list[Layer], judge_args: dict) -> list[Layer]:
    """Instantiate factory-form layer scanners with the run's judge
    configuration.

    A factory (the un-invoked ``@scanner``-decorated function, sync -
    an instance is an async callable) is called with the subset of
    the transect() judge arguments its signature declares: the signature
    is the opt-in, an undeclared knob stays layer-local rather than
    arriving unasked. Declaring ``judge_models`` with none passed to
    ``transect()`` is refused.
    """
    resolved = []
    for layer in layers:
        factory = layer.scanner
        if factory is None or not callable(factory) or _is_instance(factory):
            resolved.append(layer)
            continue
        params = inspect.signature(factory).parameters
        if "judge_models" in params and judge_args.get("judge_models") is None:
            raise ValueError(
                f"layer {layer.name!r}: the scanner factory takes judge_models "
                "but transect() got none - pass judge_models to transect(), or "
                "pass a ready scanner instance carrying its own models"
            )
        kwargs = {k: v for k, v in judge_args.items() if k in params}
        resolved.append(replace(layer, scanner=factory(**kwargs)))
    return resolved


def validate_audit(layer: Layer, frame: pd.DataFrame | None) -> None:
    """Check a layer's mounted frame against everything the audit
    reads: the declared unit/label columns plus the full judged
    projection (judge identity, verifier review, per-unit agreement /
    confidence / label_source) - the columns `transect.turns_frame` carries
    for a scanner built on `transect.cohort_llm_scanner`."""
    # call-time import: frames.common must stay importable without
    # pulling this module's report-side importers along
    from transect.frames.common import JUDGE_COLS, VERIFIER_COLS

    assert layer.audit is not None
    required = (
        *layer.audit,
        *JUDGE_COLS,
        "judge_models",
        "verifier_model",
        *VERIFIER_COLS,
        "judge_agreement",
        "confidence",
        "label_source",
    )
    missing = [c for c in required if frame is None or c not in frame.columns]
    if missing:
        raise ValueError(
            f"layer {layer.name!r}: audit reads the column(s) {missing} "
            "its frame does not carry - the audit needs a judged frame: "
            "build the scanner on transect.cohort_llm_scanner and the frame "
            "with transect.turns_frame (or project the judged columns "
            "yourself), or drop audit"
        )


def scanner_key(scanner: Any) -> str:
    """The scan-store results key for a layer's scanner."""
    try:
        name = registry_info(scanner).name
    except Exception:
        raise ValueError(
            "layer scanner carries no registry identity - decorate your "
            "scanner factory with inspect_scout's @scanner (a bare "
            "cohort_llm_scanner return is the factory's body, not the "
            "factory)"
        ) from None
    return str(name).split("/")[-1]


def _is_instance(scanner: Any) -> bool:
    call = scanner if inspect.isfunction(scanner) else type(scanner).__call__
    return inspect.iscoroutinefunction(call)
