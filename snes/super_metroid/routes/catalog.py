"""Named continuous Super Metroid routes (catalog + segment registry).

Play callables stay in :mod:`super_metroid.routes.continuous`; this module
owns split lists, derived CLI tip views, and the KPDR-style segment registry
for the power-on chain.

**One continuous CLI** (`scripts/record/continuous.py --to <tip>`) covers all
milestones. ``run_to`` dispatches from tip-spec tables only.

CLI identity is owned by :class:`~super_metroid.routes.tips.TipSpec`:

- Early: fields on :data:`~super_metroid.routes.early_continuous.EARLY_TIP_SPECS`
- Super+: fields on :class:`~super_metroid.routes.kpdr.spine_types.TipSegment`,
  copied onto TipSpec in :mod:`super_metroid.routes.kpdr.hops`

:class:`ContinuousTip` and :class:`~retro_harness.adventure.routes.NamedRoute`
are **derived** from registered TipSpecs via :func:`rebuild_from_tip_specs`
(called from :func:`~super_metroid.routes.tips.register_tips`).

Extend a post-Supers tip by:

1. pure controller in ``routes/kpdr/`` (+ ``KPDR_SEGMENTS``)
2. graph edges in ``progression/stages/`` (re-exported via ``progression/data.py``)
3. :class:`~super_metroid.routes.kpdr.spine.SpineHop` (+ tip segment with CLI
   fields) in ``routes/kpdr/spine.py`` / ``tip_segments.py`` — hop tables /
   tip-specs / Super+ split suffixes are derived
4. ``run_to`` wiring stays automatic once the TipSpec is registered

Product tip order follows :data:`~super_metroid.routes.tips.TIP_SPECS` after
registration (tests enforce ContinuousTip order match).

API is tip-id functions: ``play_morph`` / ``run_morph`` / ``run_to("bat_cave")``.
Do **not** add ``start_to_*`` scripts or clone runner pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route,
    list_routes,
    register_routes,
)
from super_metroid.routes.kpdr.spine import hop_ids_to_tip

# Split ids recorded on continuous runs (order matters for reports).
BOMBS_PREFIX_SPLITS = (
    "first_ceres_control",
    "ridley_countdown",
    "zebes_landing",
    "morph_ball",
    "first_missiles",
    "blue_brinstar_missiles",
    "bombs",
    "bomb_torizo_defeated",
    "bomb_torizo_exit",
)

SPORE_EXIT_SPLITS = BOMBS_PREFIX_SPLITS + (
    "terminator_energy_tank",
    "green_brinstar_main_shaft",
    "spore_spawn_activated",
    "spore_spawn_defeated",
    "spore_spawn_exit",
)

SUPERS_SPLITS = SPORE_EXIT_SPLITS + ("spore_supers_collected",)

# Super+ split suffixes: hop_ids along each tip's parent chain (from spine).
# KPDR K1: Super collect → farming → Big Pink main → GHZ → Noob → Red Tower.
RED_TOWER_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("red_tower")

# KPDR K2 first hop: natural Red Tower descent → Bat Room.
BAT_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("bat")

# KPDR K2.1: Bat three-platform crossing → Below Spazer.
BELOW_SPAZER_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("below_spazer")

# KPDR K2.3–K2.6: Below Spazer → West → Glass → East → Warehouse Entrance.
WAREHOUSE_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("warehouse")

# KPDR K2.7–K2.10: Warehouse → Business → Hi-Jump collect.
HIJUMP_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("hijump")

# KPDR K2.11–K2.18: Hi-Jump return → Warehouse approach → natural Kraid entry.
KRAID_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("kraid")

# KPDR K3: Kraid fight → Varia collect.
VARIA_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("varia")

# KPDR K3 return: Varia → Kraid return spine → Business Center.
BUSINESS_RETURN_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("business")

# KPDR K4.0 forward: Business Center → Frog Savestation (side save / Speedway).
FROG_SAVE_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("frog")

# KPDR K4.4 first Bubble: Business → Cathedral climb → Bubble → Bat Cave.
# Sibling of Frog Save (not a prefix of frog): Business → Cathedral path.
BAT_CAVE_SPLITS = SUPERS_SPLITS + hop_ids_to_tip("bat_cave")


@dataclass(frozen=True)
class ContinuousTip:
    """CLI view of one continuous tip (derived from :class:`TipSpec`).

    Play/run/graph live on TipSpec. This type is a thin projection of display
    names, capability flags, and aliases for ``scripts/record/continuous.py
    --to``. Rebuilt by :func:`rebuild_from_tip_specs` after tip registration.
    """

    tip_id: str
    """Canonical short id used by ``--to`` (e.g. ``red_tower``)."""

    artifact_stem: str
    """Recording basename under ``recordings/`` (matches ``tip_id``)."""

    display_name: str
    description: str = ""
    supports_room_timing: bool = False
    supports_unlimited_energy: bool = False
    supports_checkpoint: bool = False
    """When True, ``run_to(..., state_output=)`` may write an integrity-green state."""
    aliases: tuple[str, ...] = ()


# Verified continuous tip (M5): Bat Cave (K4.4 first Bubble) has two matching
# integrity-green power-on reports at 122,304f. Frog Save remains a side tip.
DEFAULT_CONTINUOUS_TIP = "bat_cave"

# Live views — mutated in place by :func:`rebuild_from_tip_specs` so importers
# that bound the name early still see post-registration tips.
CONTINUOUS_TIPS: list[ContinuousTip] = []
CONTINUOUS_TIP_BY_ID: dict[str, ContinuousTip] = {}
ROUTE_REGISTRY: dict[str, NamedRoute] = {}


def continuous_tip_from_spec(spec: Any) -> ContinuousTip:
    """Project a TipSpec (or duck-type) into a ContinuousTip CLI view."""
    return ContinuousTip(
        tip_id=spec.tip_id,
        artifact_stem=getattr(spec, "artifact_stem", spec.tip_id),
        display_name=spec.display_name or f"Power-on → {spec.tip_id}",
        description=spec.description,
        supports_room_timing=bool(spec.supports_room_timing),
        supports_unlimited_energy=bool(spec.supports_unlimited_energy),
        supports_checkpoint=bool(spec.supports_checkpoint),
        aliases=tuple(spec.aliases or ()),
    )


def named_route_from_spec(spec: Any) -> NamedRoute:
    """Project a TipSpec into a NamedRoute (milestones = required_splits)."""
    return NamedRoute(
        route_id=f"sm_{spec.tip_id}",
        display_name=spec.display_name or f"Power-on → {spec.tip_id}",
        description=spec.description,
        milestones=tuple(
            RouteMilestone(sid, sid, sid, sid) for sid in spec.required_splits
        ),
    )


def rebuild_from_tip_specs() -> None:
    """Rebuild ContinuousTip + NamedRoute registries from registered TipSpecs.

    Called from :func:`~super_metroid.routes.tips.register_tips`. Mutates
    :data:`CONTINUOUS_TIPS`, :data:`CONTINUOUS_TIP_BY_ID`, and
    :data:`ROUTE_REGISTRY` in place so early importers stay live.
    """
    from super_metroid.routes.tips import TIP_SPECS

    CONTINUOUS_TIPS.clear()
    CONTINUOUS_TIP_BY_ID.clear()
    ROUTE_REGISTRY.clear()

    for spec in TIP_SPECS:
        tip = continuous_tip_from_spec(spec)
        CONTINUOUS_TIPS.append(tip)
        CONTINUOUS_TIP_BY_ID[tip.tip_id] = tip
        if tip.artifact_stem != tip.tip_id:
            CONTINUOUS_TIP_BY_ID[tip.artifact_stem] = tip
        for alias in tip.aliases:
            CONTINUOUS_TIP_BY_ID[alias] = tip
        route = named_route_from_spec(spec)
        register_routes(ROUTE_REGISTRY, route, tip.tip_id, *tip.aliases)


def get_continuous_tip(tip: str) -> ContinuousTip:
    """Resolve a tip id or alias (case-insensitive)."""
    key = tip.strip().lower().replace("-", "_")
    try:
        return CONTINUOUS_TIP_BY_ID[key]
    except KeyError as exc:
        # Also try the raw lowercased key (preserves dots, e.g. k4.4).
        raw = tip.strip().lower()
        if raw in CONTINUOUS_TIP_BY_ID:
            return CONTINUOUS_TIP_BY_ID[raw]
        known = ", ".join(t.tip_id for t in CONTINUOUS_TIPS) or "(none registered)"
        raise KeyError(
            f"Unknown continuous tip {tip!r}. Known: {known} "
            f"(default tip: {DEFAULT_CONTINUOUS_TIP})"
        ) from exc


def list_continuous_tips() -> list[ContinuousTip]:
    return list(CONTINUOUS_TIPS)


def get_named_route(route_id: str) -> NamedRoute:
    return get_route(ROUTE_REGISTRY, route_id)


def list_named_routes() -> list[NamedRoute]:
    return list_routes(ROUTE_REGISTRY)


SegmentFn = Callable[..., Any]

# Populated by continuous module after play_* functions are defined to avoid
# circular imports at catalog load time.
CONTINUOUS_SEGMENTS: dict[str, SegmentFn] = {}


def register_continuous_segments(segments: dict[str, SegmentFn]) -> None:
    CONTINUOUS_SEGMENTS.clear()
    CONTINUOUS_SEGMENTS.update(segments)


def __getattr__(name: str) -> Any:
    """Derive CONTINUOUS_TIP_ORDER from the live tip list."""
    if name == "CONTINUOUS_TIP_ORDER":
        return tuple(t.tip_id for t in CONTINUOUS_TIPS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
