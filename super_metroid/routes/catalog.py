"""Named continuous Super Metroid routes (catalog + segment registry).

Play callables stay in :mod:`super_metroid.routes.continuous`; this module
owns identity, split lists, tip metadata, and the KPDR-style segment registry
for the power-on chain.

**One continuous CLI** (`scripts/record/continuous.py --to <tip>`) covers all
milestones. Extend a post-Supers tip by:

1. pure controller in ``routes/kpdr/`` (+ ``KPDR_SEGMENTS``)
2. graph edges in ``progression.py``
3. split tuple + :class:`ContinuousTip` + :class:`NamedRoute` here
4. ``RouteHop`` / tip-spec rows in ``continuous.py`` (data-driven post-Supers
   composition; capability flags on :class:`ContinuousTip` — no hard-coded
   ``run_to`` allowlists)

Do **not** add a new ``start_to_*.py`` script or copy another full ``run_*``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from adventure_common.routes import (
    NamedRoute,
    RouteMilestone,
    get_route,
    list_routes,
    register_routes,
)

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

# KPDR K1: Super collect → farming → Big Pink main → GHZ → Noob → Red Tower.
RED_TOWER_SPLITS = SUPERS_SPLITS + (
    "super_to_farming",
    "farming_to_big_pink",
    "big_pink_main",
    "big_pink_to_ghz",
    "ghz_to_noob",
    "noob_to_red_tower",
)

# KPDR K2 first hop: natural Red Tower descent → Bat Room.
BAT_SPLITS = RED_TOWER_SPLITS + ("red_tower_to_bat",)

# KPDR K2.1: Bat three-platform crossing → Below Spazer.
BELOW_SPAZER_SPLITS = BAT_SPLITS + ("bat_to_below_spazer",)

# KPDR K2.3–K2.6: Below Spazer → West → Glass → East → Warehouse Entrance.
WAREHOUSE_SPLITS = BELOW_SPAZER_SPLITS + (
    "below_spazer_to_west",
    "west_to_glass",
    "glass_to_east",
    "east_to_warehouse",
)

# KPDR K2.7–K2.10: Warehouse → Business → Hi-Jump collect.
HIJUMP_SPLITS = WAREHOUSE_SPLITS + (
    "warehouse_to_business",
    "business_to_hj_shaft",
    "hj_shaft_to_hj_room",
    "hijump_collected",
)

# KPDR K2.11–K2.18: Hi-Jump return → Warehouse approach → natural Kraid entry.
KRAID_SPLITS = HIJUMP_SPLITS + (
    "hj_room_to_shaft",
    "hj_shaft_to_business",
    "business_to_warehouse",
    "warehouse_to_zeela",
    "zeela_to_kihunter",
    "kihunter_to_baby_kraid",
    "baby_kraid_to_eye",
    "eye_to_kraid",
)

# KPDR K3: Kraid fight → Varia collect.
VARIA_SPLITS = KRAID_SPLITS + ("kraid_to_varia",)

# KPDR K3 return: Varia → Kraid return spine → Business Center.
BUSINESS_RETURN_SPLITS = VARIA_SPLITS + (
    "varia_to_kraid_return",
    "kraid_to_eye_return",
    "eye_to_baby_return",
    "baby_to_kihunter_return",
    "kihunter_to_zeela_return",
    "zeela_to_warehouse_return",
    "warehouse_to_business_return",
)

# KPDR K4.0 forward: Business Center → Frog Savestation (side save / Speedway).
FROG_SAVE_SPLITS = BUSINESS_RETURN_SPLITS + ("business_to_frog_save",)

# KPDR K4.4 first Bubble: Business → Cathedral climb → Bubble → Bat Cave.
# Sibling of Frog Save (not a prefix of frog): Business → Cathedral path.
BAT_CAVE_SPLITS = BUSINESS_RETURN_SPLITS + (
    "business_to_cathedral_entrance",
    "cathedral_entrance_to_cathedral",
    "cathedral_to_rising_tide",
    "rising_tide_to_bubble",
    "bubble_to_bat_cave",
)


@dataclass(frozen=True)
class ContinuousTip:
    """One stop on the power-on continuous chain (CLI ``--to`` target)."""

    tip_id: str
    """Canonical short id used by ``--to`` (e.g. ``red_tower``)."""

    artifact_stem: str
    """Recording basename under ``recordings/`` (historical ``start_to_*`` stems)."""

    display_name: str
    description: str = ""
    supports_room_timing: bool = False
    supports_unlimited_energy: bool = False
    supports_checkpoint: bool = False
    """When True, ``run_to(..., state_output=)`` may write an integrity-green state."""
    aliases: tuple[str, ...] = ()


# Ordered prefix chain; the default is the furthest integrity-green tip.
CONTINUOUS_TIPS: tuple[ContinuousTip, ...] = (
    ContinuousTip(
        tip_id="morph",
        artifact_stem="start_to_morph",
        display_name="Power-on → Morph Ball",
        description="Ceres → Zebes Morph collect.",
        aliases=("start_to_morph",),
    ),
    ContinuousTip(
        tip_id="bombs",
        artifact_stem="start_to_bomb_torizo",
        display_name="Power-on → Bomb Torizo exit",
        description="Morph prefix through natural Bomb Torizo clear.",
        aliases=("start_to_bombs", "bomb_torizo", "torizo"),
    ),
    ContinuousTip(
        tip_id="spore",
        artifact_stem="start_to_spore_spawn",
        display_name="Power-on → Spore Spawn exit",
        description="Bombs prefix through natural Spore exit into Super room.",
        supports_unlimited_energy=True,
        aliases=("start_to_spore", "start_to_spore_spawn", "spore_spawn"),
    ),
    ContinuousTip(
        tip_id="supers",
        artifact_stem="start_to_supers",
        display_name="Power-on → Spore Super Missiles",
        description="Spore prefix through natural Super Missile collect.",
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_supers", "super"),
    ),
    ContinuousTip(
        tip_id="red_tower",
        artifact_stem="start_to_red_tower",
        display_name="Power-on → Red Tower (KPDR K1)",
        description=(
            "Supers prefix through farming, Big Pink main, GHZ, Noob, "
            "and natural Red Tower entry."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_red_tower", "red", "k1"),
    ),
    ContinuousTip(
        tip_id="bat",
        artifact_stem="start_to_bat",
        display_name="Power-on → Bat Room (KPDR K2.0)",
        description=(
            "Red Tower prefix through natural Red Tower descent and "
            "Bat Room entry (first K2 hop)."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_bat", "bat_room", "k2_0"),
    ),
    ContinuousTip(
        tip_id="below_spazer",
        artifact_stem="start_to_below_spazer",
        display_name="Power-on → Below Spazer (KPDR K2.1)",
        description=(
            "Bat prefix through natural three-platform Bat crossing and "
            "Below Spazer entry."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_below_spazer", "below", "k2_1"),
    ),
    ContinuousTip(
        tip_id="warehouse",
        artifact_stem="start_to_warehouse",
        display_name="Power-on → Warehouse Entrance (KPDR K2.6)",
        description=(
            "Below Spazer prefix through West Tunnel, Glass Tunnel, "
            "East Tunnel, and natural Warehouse Entrance."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_warehouse", "warehouse_entrance", "k2_6"),
    ),
    ContinuousTip(
        tip_id="hijump",
        artifact_stem="start_to_hijump",
        display_name="Power-on → Hi-Jump Boots (KPDR K2.10)",
        description=(
            "Warehouse prefix through Business Center, Hi-Jump shaft, "
            "and natural Hi-Jump Boots collect."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_hijump", "hi_jump", "hi-jump", "k2_10"),
    ),
    ContinuousTip(
        tip_id="kraid",
        artifact_stem="start_to_kraid",
        display_name="Power-on → Kraid entry (KPDR K2.18)",
        description=(
            "Hi-Jump prefix through return to Warehouse, Zeela/Kihunter/"
            "Baby/Eye approach, and natural Kraid room entry."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        aliases=("start_to_kraid", "kraid_entry", "k2_18"),
    ),
    ContinuousTip(
        tip_id="varia",
        artifact_stem="start_to_varia",
        display_name="Power-on → Varia Suit (KPDR K3)",
        description=(
            "Kraid-entry prefix through natural Kraid fight, rear exit, "
            "and real Varia PLM collect."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        supports_checkpoint=True,
        aliases=("start_to_varia", "varia_suit", "k3"),
    ),
    ContinuousTip(
        tip_id="business",
        artifact_stem="start_to_business",
        display_name="Power-on → Business Center return (KPDR K3→K4)",
        description=(
            "Varia prefix through the natural Kraid return spine and the "
            "right-ledge Warehouse reverse stack into Business Center."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        supports_checkpoint=True,
        aliases=("start_to_business", "business_center", "k3_return"),
    ),
    ContinuousTip(
        tip_id="frog",
        artifact_stem="start_to_frog_save",
        display_name="Power-on → Frog Savestation (KPDR K4.0)",
        description=(
            "Business return plus the elevator descent and blue-door exit to "
            "Frog Savestation."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        supports_checkpoint=True,
        aliases=("start_to_frog_save", "frog_save", "k4_0"),
    ),
    ContinuousTip(
        tip_id="bat_cave",
        artifact_stem="start_to_bat_cave",
        display_name="Power-on → Bat Cave (KPDR K4.4 first Bubble)",
        description=(
            "Business return through Cathedral Entrance → Cathedral → Rising "
            "Tide → Bubble Mountain (R19 double-WJ fire + Super door) into "
            "ordinary Bat Cave. Sibling of Frog Save, not a frog prefix."
        ),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        supports_checkpoint=True,
        aliases=(
            "start_to_bat_cave",
            "norfair_bat",
            "bubble_bat",
            "k4_4",
            "k4.4",
        ),
    ),
)

# Verified continuous tip (M5): Frog Save (K4.0) has two matching
# integrity-green power-on reports at 114,923f. Bat Cave tip is wired for
# compose; promote DEFAULT only after dual integrity-green start_to_bat_cave.
DEFAULT_CONTINUOUS_TIP = "frog"


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


ROUTE_START_TO_MORPH = NamedRoute(
    route_id="sm_start_to_morph",
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

ROUTE_START_TO_BOMBS = NamedRoute(
    route_id="sm_start_to_bombs",
    display_name="Power-on → Bomb Torizo exit",
    description="Morph prefix through natural Bomb Torizo clear and exit.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BOMBS_PREFIX_SPLITS),
)

ROUTE_START_TO_SPORE = NamedRoute(
    route_id="sm_start_to_spore",
    display_name="Power-on → Spore Spawn exit",
    description="Bombs prefix through natural Spore Spawn clear into Super room.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in SPORE_EXIT_SPLITS),
)

ROUTE_START_TO_SUPERS = NamedRoute(
    route_id="sm_start_to_supers",
    display_name="Power-on → Spore Super Missiles",
    description="Spore prefix through natural Super Missile collect in 0x9B5B.",
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in SUPERS_SPLITS),
)

ROUTE_START_TO_RED_TOWER = NamedRoute(
    route_id="sm_start_to_red_tower",
    display_name="Power-on → Red Tower (KPDR K1)",
    description=(
        "Supers prefix through farming, Big Pink main, GHZ, Noob Bridge, "
        "and natural Red Tower entry (Charge return side trip not included)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in RED_TOWER_SPLITS),
)

ROUTE_START_TO_BAT = NamedRoute(
    route_id="sm_start_to_bat",
    display_name="Power-on → Bat Room (KPDR K2.0)",
    description=(
        "Red Tower prefix through natural Red Tower descent and Bat Room "
        "entry (first continuous K2 hop)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BAT_SPLITS),
)

ROUTE_START_TO_BELOW_SPAZER = NamedRoute(
    route_id="sm_start_to_below_spazer",
    display_name="Power-on → Below Spazer (KPDR K2.1)",
    description=(
        "Bat prefix through natural three-platform Bat crossing and Below Spazer entry."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BELOW_SPAZER_SPLITS),
)

ROUTE_START_TO_WAREHOUSE = NamedRoute(
    route_id="sm_start_to_warehouse",
    display_name="Power-on → Warehouse Entrance (KPDR K2.6)",
    description=(
        "Below Spazer prefix through West/Glass/East tunnels and natural "
        "Warehouse Entrance (KPDR K2.3–K2.6)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in WAREHOUSE_SPLITS),
)

ROUTE_START_TO_HIJUMP = NamedRoute(
    route_id="sm_start_to_hijump",
    display_name="Power-on → Hi-Jump Boots (KPDR K2.10)",
    description=(
        "Warehouse prefix through Business Center and natural Hi-Jump collect."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in HIJUMP_SPLITS),
)

ROUTE_START_TO_KRAID = NamedRoute(
    route_id="sm_start_to_kraid",
    display_name="Power-on → Kraid entry (KPDR K2.18)",
    description=(
        "Hi-Jump prefix through return and Warehouse approach to natural "
        "Kraid room entry."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in KRAID_SPLITS),
)

ROUTE_START_TO_VARIA = NamedRoute(
    route_id="sm_start_to_varia",
    display_name="Power-on → Varia Suit (KPDR K3)",
    description=("Kraid-entry prefix through natural fight and Varia Suit collect."),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in VARIA_SPLITS),
)

ROUTE_START_TO_BUSINESS = NamedRoute(
    route_id="sm_start_to_business",
    display_name="Power-on → Business Center return (KPDR K3→K4)",
    description=(
        "Varia prefix through the natural Kraid return spine and Warehouse "
        "reverse stack into Business Center."
    ),
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid) for sid in BUSINESS_RETURN_SPLITS
    ),
)

ROUTE_START_TO_FROG_SAVE = NamedRoute(
    route_id="sm_start_to_frog_save",
    display_name="Power-on → Frog Savestation (KPDR K4.0)",
    description=(
        "Business return plus the natural Business Center elevator descent and "
        "blue-door exit to Frog Savestation."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in FROG_SAVE_SPLITS),
)

ROUTE_START_TO_BAT_CAVE = NamedRoute(
    route_id="sm_start_to_bat_cave",
    display_name="Power-on → Bat Cave (KPDR K4.4 first Bubble)",
    description=(
        "Business return through Cathedral climb and Bubble Mountain into "
        "ordinary Bat Cave (first Bubble visit, no Speed)."
    ),
    milestones=tuple(RouteMilestone(sid, sid, sid, sid) for sid in BAT_CAVE_SPLITS),
)

ROUTE_REGISTRY: dict[str, NamedRoute] = {}
register_routes(ROUTE_REGISTRY, ROUTE_START_TO_MORPH, "morph", "start_to_morph")
register_routes(ROUTE_REGISTRY, ROUTE_START_TO_BOMBS, "bombs", "start_to_bombs")
register_routes(ROUTE_REGISTRY, ROUTE_START_TO_SPORE, "spore", "start_to_spore")
register_routes(ROUTE_REGISTRY, ROUTE_START_TO_SUPERS, "supers", "start_to_supers")
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_RED_TOWER,
    "red_tower",
    "start_to_red_tower",
    "k1",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_BAT,
    "bat",
    "start_to_bat",
    "bat_room",
    "k2_0",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_BELOW_SPAZER,
    "below_spazer",
    "start_to_below_spazer",
    "below",
    "k2_1",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_WAREHOUSE,
    "warehouse",
    "start_to_warehouse",
    "warehouse_entrance",
    "k2_6",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_HIJUMP,
    "hijump",
    "start_to_hijump",
    "hi_jump",
    "k2_10",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_KRAID,
    "kraid",
    "start_to_kraid",
    "kraid_entry",
    "k2_18",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_VARIA,
    "varia",
    "start_to_varia",
    "varia_suit",
    "k3",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_BUSINESS,
    "business",
    "start_to_business",
    "business_center",
    "k3_return",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_FROG_SAVE,
    "frog",
    "start_to_frog_save",
    "frog_save",
    "k4_0",
)
register_routes(
    ROUTE_REGISTRY,
    ROUTE_START_TO_BAT_CAVE,
    "bat_cave",
    "start_to_bat_cave",
    "norfair_bat",
    "k4_4",
)

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
