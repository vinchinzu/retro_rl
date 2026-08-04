"""Named continuous Super Metroid routes (catalog + segment registry).

Play callables stay in :mod:`super_metroid.routes.continuous`; this module
owns identity, split lists, tip metadata, and the KPDR-style segment registry
for the power-on chain.

**One continuous CLI** (`scripts/record/continuous.py --to <tip>`) covers all
milestones. ``run_to`` dispatches from tip-spec tables only:

- Early (morph→supers): :data:`~super_metroid.routes.early_continuous.EARLY_TIP_BY_ID`
- Super+: :data:`~super_metroid.routes.kpdr.hops.POST_SUPERS_TIP_BY_ID`

Extend a post-Supers tip by:

1. pure controller in ``routes/kpdr/`` (+ ``KPDR_SEGMENTS``)
2. graph edges in ``progression/stages/`` (re-exported via ``progression/data.py``)
3. :class:`~super_metroid.routes.kpdr.spine.SpineHop` (+ tip segment) in
   ``routes/kpdr/spine.py`` — hop tables / tip-specs / Super+ split suffixes
   are derived (see spine module docstring)
4. CLI meta row in :data:`_CONTINUOUS_TIP_META` + id in
   :data:`CONTINUOUS_TIP_ORDER` (builds :class:`ContinuousTip`) +
   :class:`NamedRoute` here
5. ``run_to`` wiring stays automatic for tip ids in ``POST_SUPERS_TIP_BY_ID``

Product tip ids: one ordered list (:data:`CONTINUOUS_TIP_ORDER`) must match
:data:`~super_metroid.routes.tips.TIP_SPECS` after registration (tests).

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
    """CLI metadata for one stop on the power-on continuous chain.

    Play/run/graph live on :class:`~super_metroid.routes.tips.TipSpec`.
    This type owns only display names, capability flags, and aliases for
    ``scripts/record/continuous.py --to``.
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


# Product tip order — must match :data:`~super_metroid.routes.tips.TIP_SPECS`
# tip_ids after early + Super+ registration (enforced in tests).
CONTINUOUS_TIP_ORDER: tuple[str, ...] = (
    "morph",
    "bombs",
    "spore",
    "supers",
    "red_tower",
    "bat",
    "below_spazer",
    "warehouse",
    "hijump",
    "kraid",
    "varia",
    "business",
    "frog",
    "bat_cave",
)

# CLI-only fields keyed by tip_id (no second tip-id registry for play/run).
# Flags default False; omit keys that stay at default.
_CONTINUOUS_TIP_META: dict[str, dict[str, object]] = {
    "morph": {
        "display_name": "Power-on → Morph Ball",
        "description": "Ceres → Zebes Morph collect.",
    },
    "bombs": {
        "display_name": "Power-on → Bomb Torizo exit",
        "description": "Morph prefix through natural Bomb Torizo clear.",
        "aliases": ("bomb_torizo", "torizo"),
    },
    "spore": {
        "display_name": "Power-on → Spore Spawn exit",
        "description": "Bombs prefix through natural Spore exit into Super room.",
        "supports_unlimited_energy": True,
        "aliases": ("spore_spawn",),
    },
    "supers": {
        "display_name": "Power-on → Spore Super Missiles",
        "description": "Spore prefix through natural Super Missile collect.",
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("super",),
    },
    "red_tower": {
        "display_name": "Power-on → Red Tower (KPDR K1)",
        "description": (
            "Supers prefix through farming, Big Pink main, GHZ, Noob, "
            "and natural Red Tower entry."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("red", "k1"),
    },
    "bat": {
        "display_name": "Power-on → Bat Room (KPDR K2.0)",
        "description": (
            "Red Tower prefix through natural Red Tower descent and "
            "Bat Room entry (first K2 hop)."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("bat_room", "k2_0"),
    },
    "below_spazer": {
        "display_name": "Power-on → Below Spazer (KPDR K2.1)",
        "description": (
            "Bat prefix through natural three-platform Bat crossing and "
            "Below Spazer entry."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("below", "k2_1"),
    },
    "warehouse": {
        "display_name": "Power-on → Warehouse Entrance (KPDR K2.6)",
        "description": (
            "Below Spazer prefix through West Tunnel, Glass Tunnel, "
            "East Tunnel, and natural Warehouse Entrance."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("warehouse_entrance", "k2_6"),
    },
    "hijump": {
        "display_name": "Power-on → Hi-Jump Boots (KPDR K2.10)",
        "description": (
            "Warehouse prefix through Business Center, Hi-Jump shaft, "
            "and natural Hi-Jump Boots collect."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("hi_jump", "hi-jump", "k2_10"),
    },
    "kraid": {
        "display_name": "Power-on → Kraid entry (KPDR K2.18)",
        "description": (
            "Hi-Jump prefix through return to Warehouse, Zeela/Kihunter/"
            "Baby/Eye approach, and natural Kraid room entry."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "aliases": ("kraid_entry", "k2_18"),
    },
    "varia": {
        "display_name": "Power-on → Varia Suit (KPDR K3)",
        "description": (
            "Kraid-entry prefix through natural Kraid fight, rear exit, "
            "and real Varia PLM collect."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "supports_checkpoint": True,
        "aliases": ("varia_suit", "k3"),
    },
    "business": {
        "display_name": "Power-on → Business Center return (KPDR K3→K4)",
        "description": (
            "Varia prefix through the natural Kraid return spine and the "
            "right-ledge Warehouse reverse stack into Business Center."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "supports_checkpoint": True,
        "aliases": ("business_center", "k3_return"),
    },
    "frog": {
        "display_name": "Power-on → Frog Savestation (KPDR K4.0)",
        "description": (
            "Business return plus the elevator descent and blue-door exit to "
            "Frog Savestation."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "supports_checkpoint": True,
        "aliases": ("frog_save", "k4_0"),
    },
    "bat_cave": {
        "display_name": "Power-on → Bat Cave (KPDR K4.4 first Bubble)",
        "description": (
            "Business return through Cathedral Entrance → Cathedral → Rising "
            "Tide → Bubble Mountain (R19 double-WJ fire + Super door) into "
            "ordinary Bat Cave. Sibling of Frog Save, not a frog prefix."
        ),
        "supports_room_timing": True,
        "supports_unlimited_energy": True,
        "supports_checkpoint": True,
        "aliases": ("norfair_bat", "bubble_bat", "k4_4", "k4.4"),
    },
}


def _build_continuous_tips() -> tuple[ContinuousTip, ...]:
    """Build CLI tips from order + meta table (artifact_stem == tip_id)."""
    missing = [tid for tid in CONTINUOUS_TIP_ORDER if tid not in _CONTINUOUS_TIP_META]
    if missing:
        raise RuntimeError(f"CONTINUOUS_TIP_ORDER missing meta for: {missing}")
    extra = set(_CONTINUOUS_TIP_META) - set(CONTINUOUS_TIP_ORDER)
    if extra:
        raise RuntimeError(f"_CONTINUOUS_TIP_META has unused tip ids: {sorted(extra)}")
    tips: list[ContinuousTip] = []
    for tip_id in CONTINUOUS_TIP_ORDER:
        meta = _CONTINUOUS_TIP_META[tip_id]
        raw_aliases = meta.get("aliases", ())
        aliases = tuple(raw_aliases) if isinstance(raw_aliases, tuple) else ()
        tips.append(
            ContinuousTip(
                tip_id=tip_id,
                artifact_stem=tip_id,
                display_name=str(meta["display_name"]),
                description=str(meta.get("description", "")),
                supports_room_timing=bool(meta.get("supports_room_timing", False)),
                supports_unlimited_energy=bool(
                    meta.get("supports_unlimited_energy", False)
                ),
                supports_checkpoint=bool(meta.get("supports_checkpoint", False)),
                aliases=aliases,
            )
        )
    return tuple(tips)


# Ordered prefix chain; the default is the furthest integrity-green tip.
CONTINUOUS_TIPS: tuple[ContinuousTip, ...] = _build_continuous_tips()

# Verified continuous tip (M5): Bat Cave (K4.4 first Bubble) has two matching
# integrity-green power-on reports at 122,304f. Frog Save remains a side tip.
DEFAULT_CONTINUOUS_TIP = "bat_cave"


def _tip_lookup() -> dict[str, ContinuousTip]:
    table: dict[str, ContinuousTip] = {}
    for tip in CONTINUOUS_TIPS:
        table[tip.tip_id] = tip
        table[tip.artifact_stem] = tip
        for alias in tip.aliases:
            table[alias] = tip
    return table


CONTINUOUS_TIP_BY_ID: dict[str, ContinuousTip] = _tip_lookup()


def get_continuous_tip(tip: str) -> ContinuousTip:
    """Resolve a tip id or alias (case-insensitive)."""
    key = tip.strip().lower().replace("-", "_")
    try:
        return CONTINUOUS_TIP_BY_ID[key]
    except KeyError as exc:
        known = ", ".join(t.tip_id for t in CONTINUOUS_TIPS)
        raise KeyError(
            f"Unknown continuous tip {tip!r}. Known: {known} "
            f"(default tip: {DEFAULT_CONTINUOUS_TIP})"
        ) from exc


def list_continuous_tips() -> list[ContinuousTip]:
    return list(CONTINUOUS_TIPS)


ROUTE_MORPH = NamedRoute(
    route_id="sm_morph",
    display_name="Power-on → Morph Ball",
    description="Continuous Ceres → Zebes Morph collect (assisted energy).",
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid)
        for sid in (
            "first_ceres_control",
            "ridley_countdown",
            "zebes_landing",
            "morph_ball",
        )
    ),
)

ROUTE_BOMBS = NamedRoute(
    route_id="sm_bombs",
    display_name="Power-on → Bomb Torizo exit",
    description="Morph prefix through natural Bomb Torizo clear and exit.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BOMBS_PREFIX_SPLITS),
)

ROUTE_SPORE = NamedRoute(
    route_id="sm_spore",
    display_name="Power-on → Spore Spawn exit",
    description="Bombs prefix through natural Spore Spawn clear into Super room.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in SPORE_EXIT_SPLITS),
)

ROUTE_SUPERS = NamedRoute(
    route_id="sm_supers",
    display_name="Power-on → Spore Super Missiles",
    description="Spore prefix through natural Super Missile collect in 0x9B5B.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in SUPERS_SPLITS),
)

ROUTE_RED_TOWER = NamedRoute(
    route_id="sm_red_tower",
    display_name="Power-on → Red Tower (KPDR K1)",
    description=(
        "Supers prefix through farming, Big Pink main, GHZ, Noob Bridge, "
        "and natural Red Tower entry (Charge return side trip not included)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in RED_TOWER_SPLITS),
)

ROUTE_BAT = NamedRoute(
    route_id="sm_bat",
    display_name="Power-on → Bat Room (KPDR K2.0)",
    description=(
        "Red Tower prefix through natural Red Tower descent and Bat Room "
        "entry (first continuous K2 hop)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BAT_SPLITS),
)

ROUTE_BELOW_SPAZER = NamedRoute(
    route_id="sm_below_spazer",
    display_name="Power-on → Below Spazer (KPDR K2.1)",
    description=(
        "Bat prefix through natural three-platform Bat crossing and Below Spazer entry."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BELOW_SPAZER_SPLITS),
)

ROUTE_WAREHOUSE = NamedRoute(
    route_id="sm_warehouse",
    display_name="Power-on → Warehouse Entrance (KPDR K2.6)",
    description=(
        "Below Spazer prefix through West/Glass/East tunnels and natural "
        "Warehouse Entrance (KPDR K2.3–K2.6)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in WAREHOUSE_SPLITS),
)

ROUTE_HIJUMP = NamedRoute(
    route_id="sm_hijump",
    display_name="Power-on → Hi-Jump Boots (KPDR K2.10)",
    description=(
        "Warehouse prefix through Business Center and natural Hi-Jump collect."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in HIJUMP_SPLITS),
)

ROUTE_KRAID = NamedRoute(
    route_id="sm_kraid",
    display_name="Power-on → Kraid entry (KPDR K2.18)",
    description=(
        "Hi-Jump prefix through return and Warehouse approach to natural "
        "Kraid room entry."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in KRAID_SPLITS),
)

ROUTE_VARIA = NamedRoute(
    route_id="sm_varia",
    display_name="Power-on → Varia Suit (KPDR K3)",
    description=("Kraid-entry prefix through natural fight and Varia Suit collect."),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in VARIA_SPLITS),
)

ROUTE_BUSINESS = NamedRoute(
    route_id="sm_business",
    display_name="Power-on → Business Center return (KPDR K3→K4)",
    description=(
        "Varia prefix through the natural Kraid return spine and Warehouse "
        "reverse stack into Business Center."
    ),
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid) for sid in BUSINESS_RETURN_SPLITS
    ),
)

ROUTE_FROG = NamedRoute(
    route_id="sm_frog",
    display_name="Power-on → Frog Savestation (KPDR K4.0)",
    description=(
        "Business return plus the natural Business Center elevator descent and "
        "blue-door exit to Frog Savestation."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in FROG_SAVE_SPLITS),
)

ROUTE_BAT_CAVE = NamedRoute(
    route_id="sm_bat_cave",
    display_name="Power-on → Bat Cave (KPDR K4.4 first Bubble)",
    description=(
        "Business return through Cathedral climb and Bubble Mountain into "
        "ordinary Bat Cave (first Bubble visit, no Speed)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BAT_CAVE_SPLITS),
)

ROUTE_REGISTRY: dict[str, NamedRoute] = {}
register_routes(ROUTE_REGISTRY, ROUTE_MORPH, "morph")
register_routes(ROUTE_REGISTRY, ROUTE_BOMBS, "bombs")
register_routes(ROUTE_REGISTRY, ROUTE_SPORE, "spore")
register_routes(ROUTE_REGISTRY, ROUTE_SUPERS, "supers")
register_routes(ROUTE_REGISTRY, ROUTE_RED_TOWER, "red_tower", "k1")
register_routes(ROUTE_REGISTRY, ROUTE_BAT, "bat", "bat_room", "k2_0")
register_routes(ROUTE_REGISTRY, ROUTE_BELOW_SPAZER, "below_spazer", "below", "k2_1")
register_routes(ROUTE_REGISTRY, ROUTE_WAREHOUSE, "warehouse", "warehouse_entrance", "k2_6")
register_routes(ROUTE_REGISTRY, ROUTE_HIJUMP, "hijump", "hi_jump", "k2_10")
register_routes(ROUTE_REGISTRY, ROUTE_KRAID, "kraid", "kraid_entry", "k2_18")
register_routes(ROUTE_REGISTRY, ROUTE_VARIA, "varia", "varia_suit", "k3")
register_routes(ROUTE_REGISTRY, ROUTE_BUSINESS, "business", "business_center", "k3_return")
register_routes(ROUTE_REGISTRY, ROUTE_FROG, "frog", "frog_save", "k4_0")
register_routes(ROUTE_REGISTRY, ROUTE_BAT_CAVE, "bat_cave", "norfair_bat", "k4_4")


SegmentFn = Callable[..., Any]

# Populated by continuous module after play_* functions are defined to avoid
# circular imports at catalog load time.
CONTINUOUS_SEGMENTS: dict[str, SegmentFn] = {}


def register_continuous_segments(segments: dict[str, SegmentFn]) -> None:
    CONTINUOUS_SEGMENTS.clear()
    CONTINUOUS_SEGMENTS.update(segments)


def get_named_route(route_id: str) -> NamedRoute:
    return get_route(ROUTE_REGISTRY, route_id)


def list_named_routes() -> list[NamedRoute]:
    return list_routes(ROUTE_REGISTRY)
