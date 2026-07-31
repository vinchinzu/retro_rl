"""Named continuous Super Metroid routes (catalog + segment registry).

Play callables stay in :mod:`super_metroid.routes.continuous`; this module
owns identity, split lists, tip metadata, and the KPDR-style segment registry
for the power-on chain.

**One continuous CLI** (`scripts/record/continuous.py --to <tip>`) covers all
milestones. Extend a post-Supers tip by:

1. pure controller in ``routes/kpdr/`` (+ ``KPDR_SEGMENTS``)
2. graph edges in ``progression.py``
3. split tuple + :class:`ContinuousTip` + :class:`NamedRoute` here
4. ``RouteHop`` rows + thin ``play_*`` / ``run_post_supers_tip`` wrapper in
   ``continuous.py``

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
    aliases: tuple[str, ...] = ()


# Ordered prefix chain; last entry is the current continuous tip.
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
)

# Current continuous tip (extend CONTINUOUS_TIPS when attaching the next leg).
DEFAULT_CONTINUOUS_TIP = CONTINUOUS_TIPS[-1].tip_id


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
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid) for sid in BOMBS_PREFIX_SPLITS
    ),
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
        "Bat prefix through natural three-platform Bat crossing and "
        "Below Spazer entry."
    ),
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid) for sid in BELOW_SPAZER_SPLITS
    ),
)

ROUTE_START_TO_WAREHOUSE = NamedRoute(
    route_id="sm_start_to_warehouse",
    display_name="Power-on → Warehouse Entrance (KPDR K2.6)",
    description=(
        "Below Spazer prefix through West/Glass/East tunnels and natural "
        "Warehouse Entrance (KPDR K2.3–K2.6)."
    ),
    milestones=tuple(
        RouteMilestone(sid, sid, sid, sid) for sid in WAREHOUSE_SPLITS
    ),
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
