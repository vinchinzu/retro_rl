"""Post-Supers continuous hop tables + tip-spec data.

Extracted from :mod:`super_metroid.routes.continuous` so tip composition data
lives next to KPDR controllers. Runners (``play_hops``, ``play_post_supers_tip_spec``,
``run_post_supers_tip``) stay in ``continuous`` and import these tables.

**No tip-order or frame-semantics ownership here** — this module is data only.
Extend a tip: pure controller → graph → catalog ContinuousTip → append a
:class:`PostSupersTipSpec` row (parent + hops + report fields).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from super_metroid.progression import (
    START_TO_BAT_GRAPH,
    START_TO_BELOW_SPAZER_GRAPH,
    START_TO_HIJUMP_GRAPH,
    START_TO_KRAID_GRAPH,
    START_TO_RED_TOWER_GRAPH,
    START_TO_SPEED_GRAPH,
    START_TO_VARIA_GRAPH,
    START_TO_WAREHOUSE_GRAPH,
    RoomProgressionGraph,
)
from super_metroid.ram import HI_JUMP_MASK, VARIA_MASK
from super_metroid.routes.catalog import (
    BAT_CAVE_SPLITS,
    BAT_SPLITS,
    BELOW_SPAZER_SPLITS,
    BUSINESS_RETURN_SPLITS,
    FROG_SAVE_SPLITS,
    HIJUMP_SPLITS,
    KRAID_SPLITS,
    RED_TOWER_SPLITS,
    VARIA_SPLITS,
    WAREHOUSE_SPLITS,
)
from super_metroid.routes.kpdr.big_pink import play_big_pink_to_ghz
from super_metroid.routes.kpdr.big_pink_shaft import play_big_pink_into_main_shaft
from super_metroid.routes.kpdr.green_hill import (
    play_ghz_to_noob,
    play_noob_to_red_tower,
)
from super_metroid.routes.kpdr.hijump import (
    play_business_to_hj_shaft,
    play_business_to_warehouse,
    play_hj_room_collect,
    play_hj_room_to_shaft,
    play_hj_shaft_to_business,
    play_hj_shaft_to_hj_room,
)
from super_metroid.routes.kpdr.warehouse import play_warehouse_to_business
from super_metroid.routes.kpdr.bubble_mountain import play_bubble_to_bat_cave
from super_metroid.routes.kpdr.k4_norfair import (
    play_business_to_cathedral_entrance,
    play_business_to_frog_save,
    play_cathedral_entrance_to_cathedral,
    play_cathedral_to_rising_tide,
    play_rising_tide_to_bubble,
)
from super_metroid.routes.kpdr.kraid_approach import (
    play_baby_kraid_to_eye,
    play_eye_to_kraid,
    play_kihunter_to_baby_kraid,
    play_kraid_entry_to_varia,
    play_warehouse_to_zeela_with_hijump,
    play_zeela_to_kihunter,
)
from super_metroid.routes.kpdr.kraid_return import (
    play_baby_to_kihunter_return,
    play_eye_to_baby_return,
    play_kihunter_to_zeela_return,
    play_zeela_to_warehouse_return,
)
from super_metroid.routes.kpdr.red_tower import (
    play_bat_to_below_spazer,
    play_below_spazer_to_west,
    play_east_to_warehouse,
    play_glass_to_east,
    play_red_tower_to_bat,
    play_west_to_glass,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BAT_CAVE,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_EAST_TUNNEL,
    ROOM_FARMING,
    ROOM_FROG_SAVE,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_RISING_TIDE,
    ROOM_VARIA,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.kpdr.super_room import (
    play_farming_to_big_pink,
    play_super_room_to_farming,
)
from super_metroid.routes.kpdr.varia_return import (
    play_kraid_to_eye_return,
    play_varia_to_kraid,
)
from super_metroid.routes.runtime import RouteSession

__all__ = [
    "RouteHop",
    "PostSupersTipSpec",
    "POST_SUPERS_TIP_SPECS",
    "POST_SUPERS_TIP_BY_ID",
    "RED_TOWER_HOPS",
    "BAT_HOPS",
    "BELOW_SPAZER_HOPS",
    "WAREHOUSE_HOPS",
    "HIJUMP_HOPS",
    "KRAID_HOPS",
    "VARIA_HOPS",
    "BUSINESS_RETURN_HOPS",
    "FROG_ONLY_HOPS",
    "FROG_SAVE_HOPS",
    "BAT_CAVE_ONLY_HOPS",
    # Underscore aliases kept for continuous re-export / historical tests.
    "_RED_TOWER_HOPS",
    "_BAT_HOPS",
    "_BELOW_SPAZER_HOPS",
    "_WAREHOUSE_HOPS",
    "_HIJUMP_HOPS",
    "_KRAID_HOPS",
    "_VARIA_HOPS",
    "_BUSINESS_RETURN_HOPS",
    "_FROG_ONLY_HOPS",
    "_FROG_SAVE_HOPS",
    "_BAT_CAVE_ONLY_HOPS",
]


@dataclass(frozen=True)
class RouteHop:
    """One controller leg: play, record split, assert destination room."""

    split_id: str
    play: Callable[[RouteSession], Any]
    from_room: int
    to_room: int
    room_label: str
    #: When False, skip door-edge split lookup (in-room milestones only).
    use_transition_split: bool = True
    #: Optional extra assert after play (before room-id check).
    after: Callable[[RouteSession], None] | None = None


def _require_big_pink_main_shaft(session: RouteSession) -> None:
    if session.state.room_id != ROOM_BIG_PINK or session.state.samus_x > 750:
        raise RuntimeError(
            f"Big Pink main shaft not reached: room=0x{session.state.room_id:04X} "
            f"x={session.state.samus_x}"
        )


def _require_hijump_collected(session: RouteSession) -> None:
    if not session.state.collected_items & HI_JUMP_MASK:
        raise RuntimeError(
            f"Hi-Jump not collected: items=0x{session.state.collected_items:04X}"
        )


def _require_varia_collected(session: RouteSession) -> None:
    if not session.state.collected_items & VARIA_MASK:
        raise RuntimeError(
            f"Varia not collected: items=0x{session.state.collected_items:04X}"
        )


# K1 legs after Super collect (Charge return intentionally omitted).
_RED_TOWER_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "super_to_farming",
        play_super_room_to_farming,
        0x9B5B,
        ROOM_FARMING,
        "farming",
    ),
    RouteHop(
        "farming_to_big_pink",
        play_farming_to_big_pink,
        ROOM_FARMING,
        ROOM_BIG_PINK,
        "Big Pink",
    ),
    RouteHop(
        "big_pink_main",
        play_big_pink_into_main_shaft,
        ROOM_BIG_PINK,
        ROOM_BIG_PINK,
        "Big Pink main shaft",
        use_transition_split=False,
        after=_require_big_pink_main_shaft,
    ),
    RouteHop(
        "big_pink_to_ghz",
        play_big_pink_to_ghz,
        ROOM_BIG_PINK,
        ROOM_GHZ,
        "GHZ",
    ),
    RouteHop(
        "ghz_to_noob",
        play_ghz_to_noob,
        ROOM_GHZ,
        ROOM_NOOB,
        "Noob",
    ),
    RouteHop(
        "noob_to_red_tower",
        play_noob_to_red_tower,
        ROOM_NOOB,
        ROOM_RED_TOWER,
        "Red Tower",
    ),
)

_BAT_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "red_tower_to_bat",
        play_red_tower_to_bat,
        ROOM_RED_TOWER,
        ROOM_BAT,
        "Bat Room",
    ),
)

_BELOW_SPAZER_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "bat_to_below_spazer",
        play_bat_to_below_spazer,
        ROOM_BAT,
        ROOM_BELOW_SPAZER,
        "Below Spazer",
    ),
)

_WAREHOUSE_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "below_spazer_to_west",
        play_below_spazer_to_west,
        ROOM_BELOW_SPAZER,
        ROOM_WEST_TUNNEL,
        "West Tunnel",
    ),
    RouteHop(
        "west_to_glass",
        play_west_to_glass,
        ROOM_WEST_TUNNEL,
        ROOM_GLASS,
        "Glass Tunnel",
    ),
    RouteHop(
        "glass_to_east",
        play_glass_to_east,
        ROOM_GLASS,
        ROOM_EAST_TUNNEL,
        "East Tunnel",
    ),
    RouteHop(
        "east_to_warehouse",
        play_east_to_warehouse,
        ROOM_EAST_TUNNEL,
        ROOM_WAREHOUSE,
        "Warehouse Entrance",
    ),
)

# K2.7–K2.10: Warehouse elevator → Business → HJ shaft → HJ room collect.
_HIJUMP_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "warehouse_to_business",
        play_warehouse_to_business,
        ROOM_WAREHOUSE,
        ROOM_BUSINESS,
        "Business Center",
    ),
    RouteHop(
        "business_to_hj_shaft",
        play_business_to_hj_shaft,
        ROOM_BUSINESS,
        ROOM_HJ_SHAFT,
        "Hi-Jump shaft",
    ),
    RouteHop(
        "hj_shaft_to_hj_room",
        play_hj_shaft_to_hj_room,
        ROOM_HJ_SHAFT,
        ROOM_HJ,
        "Hi-Jump Room",
    ),
    RouteHop(
        "hijump_collected",
        play_hj_room_collect,
        ROOM_HJ,
        ROOM_HJ,
        "Hi-Jump collect",
        use_transition_split=False,
        after=_require_hijump_collected,
    ),
)

# K2.11–K2.18: HJ return → Warehouse → Zeela → … → natural Kraid entry.
_KRAID_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "hj_room_to_shaft",
        play_hj_room_to_shaft,
        ROOM_HJ,
        ROOM_HJ_SHAFT,
        "Hi-Jump shaft return",
    ),
    RouteHop(
        "hj_shaft_to_business",
        play_hj_shaft_to_business,
        ROOM_HJ_SHAFT,
        ROOM_BUSINESS,
        "Business Center return",
    ),
    RouteHop(
        "business_to_warehouse",
        play_business_to_warehouse,
        ROOM_BUSINESS,
        ROOM_WAREHOUSE,
        "Warehouse return",
    ),
    RouteHop(
        "warehouse_to_zeela",
        play_warehouse_to_zeela_with_hijump,
        ROOM_WAREHOUSE,
        ROOM_ZEELA,
        "Warehouse Zeela",
    ),
    RouteHop(
        "zeela_to_kihunter",
        play_zeela_to_kihunter,
        ROOM_ZEELA,
        ROOM_WAREHOUSE_KIHUNTER,
        "Warehouse Kihunter",
    ),
    RouteHop(
        "kihunter_to_baby_kraid",
        play_kihunter_to_baby_kraid,
        ROOM_WAREHOUSE_KIHUNTER,
        ROOM_BABY_KRAID,
        "Baby Kraid",
    ),
    RouteHop(
        "baby_kraid_to_eye",
        play_baby_kraid_to_eye,
        ROOM_BABY_KRAID,
        ROOM_KRAID_EYE,
        "Kraid Eye Door",
    ),
    RouteHop(
        "eye_to_kraid",
        play_eye_to_kraid,
        ROOM_KRAID_EYE,
        ROOM_KRAID,
        "Kraid's Room",
    ),
)

# K3: fight + rear exit + Varia PLM (multi-room; graph has kraid→varia edge).
_VARIA_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "kraid_to_varia",
        play_kraid_entry_to_varia,
        ROOM_KRAID,
        ROOM_VARIA,
        "Varia Suit Room",
        use_transition_split=False,
        after=_require_varia_collected,
    ),
)

# K3 return: two matching integrity-green ``run_start_to_business`` power-on
# reports validate this entire return spine.
_BUSINESS_RETURN_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "varia_to_kraid_return",
        play_varia_to_kraid,
        ROOM_VARIA,
        ROOM_KRAID,
        "Kraid's Room return",
    ),
    RouteHop(
        "kraid_to_eye_return",
        play_kraid_to_eye_return,
        ROOM_KRAID,
        ROOM_KRAID_EYE,
        "Kraid Eye Door return",
    ),
    RouteHop(
        "eye_to_baby_return",
        play_eye_to_baby_return,
        ROOM_KRAID_EYE,
        ROOM_BABY_KRAID,
        "Baby Kraid return",
    ),
    RouteHop(
        "baby_to_kihunter_return",
        play_baby_to_kihunter_return,
        ROOM_BABY_KRAID,
        ROOM_WAREHOUSE_KIHUNTER,
        "Warehouse Kihunter return",
    ),
    RouteHop(
        "kihunter_to_zeela_return",
        play_kihunter_to_zeela_return,
        ROOM_WAREHOUSE_KIHUNTER,
        ROOM_ZEELA,
        "Warehouse Zeela return",
    ),
    RouteHop(
        "zeela_to_warehouse_return",
        play_zeela_to_warehouse_return,
        ROOM_ZEELA,
        ROOM_WAREHOUSE,
        "Warehouse Entrance return",
    ),
    RouteHop(
        "warehouse_to_business_return",
        play_warehouse_to_business,
        ROOM_WAREHOUSE,
        ROOM_BUSINESS,
        "Business Center return",
    ),
)

_FROG_ONLY_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "business_to_frog_save",
        play_business_to_frog_save,
        ROOM_BUSINESS,
        ROOM_FROG_SAVE,
        "Frog Savestation",
    ),
)

# Historical full hop list (business return + frog); prefer tip-spec parents.
_FROG_SAVE_HOPS: tuple[RouteHop, ...] = _BUSINESS_RETURN_HOPS + _FROG_ONLY_HOPS

# K4.4 first Bubble: Business → Cathedral → Rising Tide → Bubble → Bat Cave.
# Sibling of Frog Save (parent business); includes R19 Bubble double-WJ product.
_BAT_CAVE_ONLY_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "business_to_cathedral_entrance",
        play_business_to_cathedral_entrance,
        ROOM_BUSINESS,
        ROOM_CATHEDRAL_ENTRANCE,
        "Cathedral Entrance",
    ),
    RouteHop(
        "cathedral_entrance_to_cathedral",
        play_cathedral_entrance_to_cathedral,
        ROOM_CATHEDRAL_ENTRANCE,
        ROOM_CATHEDRAL,
        "Cathedral",
    ),
    RouteHop(
        "cathedral_to_rising_tide",
        play_cathedral_to_rising_tide,
        ROOM_CATHEDRAL,
        ROOM_RISING_TIDE,
        "Rising Tide",
    ),
    RouteHop(
        "rising_tide_to_bubble",
        play_rising_tide_to_bubble,
        ROOM_RISING_TIDE,
        ROOM_BUBBLE,
        "Bubble Mountain",
    ),
    RouteHop(
        "bubble_to_bat_cave",
        play_bubble_to_bat_cave,
        ROOM_BUBBLE,
        ROOM_BAT_CAVE,
        "Bat Cave",
    ),
)

# Public names (same objects as underscore aliases).
RED_TOWER_HOPS = _RED_TOWER_HOPS
BAT_HOPS = _BAT_HOPS
BELOW_SPAZER_HOPS = _BELOW_SPAZER_HOPS
WAREHOUSE_HOPS = _WAREHOUSE_HOPS
HIJUMP_HOPS = _HIJUMP_HOPS
KRAID_HOPS = _KRAID_HOPS
VARIA_HOPS = _VARIA_HOPS
BUSINESS_RETURN_HOPS = _BUSINESS_RETURN_HOPS
FROG_ONLY_HOPS = _FROG_ONLY_HOPS
FROG_SAVE_HOPS = _FROG_SAVE_HOPS
BAT_CAVE_ONLY_HOPS = _BAT_CAVE_ONLY_HOPS


@dataclass(frozen=True)
class PostSupersTipSpec:
    """Declarative Super+ continuous tip (drives play + run_post_supers_tip)."""

    tip_id: str
    parent_tip_id: str | None
    """``None`` → compose on top of :func:`~super_metroid.routes.continuous.play_start_to_supers`."""
    hops: tuple[RouteHop, ...]
    graph: RoomProgressionGraph
    kind: str
    required_splits: tuple[str, ...]
    final_room: int
    success_outcome: str
    route_label: str
    source_policy: str
    timing_source: str
    entry_condition_key: str
    ordinary_condition_key: str
    require_hi_jump: bool = False
    require_varia: bool = False


POST_SUPERS_TIP_SPECS: tuple[PostSupersTipSpec, ...] = (
    PostSupersTipSpec(
        tip_id="red_tower",
        parent_tip_id=None,
        hops=_RED_TOWER_HOPS,
        graph=START_TO_RED_TOWER_GRAPH,
        kind="red_tower",
        required_splits=RED_TOWER_SPLITS,
        final_room=ROOM_RED_TOWER,
        success_outcome="red_tower_entry",
        route_label="start-to-Red-Tower",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1 controllers "
            "(Super→farm→Big Pink main→GHZ→Noob→Red) + phase-guarded resources"
        ),
        timing_source="start_to_red_tower",
        entry_condition_key="natural_red_tower_entry",
        ordinary_condition_key="post_red_ordinary",
    ),
    PostSupersTipSpec(
        tip_id="bat",
        parent_tip_id="red_tower",
        hops=_BAT_HOPS,
        graph=START_TO_BAT_GRAPH,
        kind="bat",
        required_splits=BAT_SPLITS,
        final_room=ROOM_BAT,
        success_outcome="bat_room_entry",
        route_label="start-to-Bat-Room",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2.0 "
            "controllers (Super→farm→Big Pink main→GHZ→Noob→Red→Bat) + "
            "phase-guarded resources"
        ),
        timing_source="start_to_bat",
        entry_condition_key="natural_bat_room_entry",
        ordinary_condition_key="post_bat_ordinary",
    ),
    PostSupersTipSpec(
        tip_id="below_spazer",
        parent_tip_id="bat",
        hops=_BELOW_SPAZER_HOPS,
        graph=START_TO_BELOW_SPAZER_GRAPH,
        kind="below_spazer",
        required_splits=BELOW_SPAZER_SPLITS,
        final_room=ROOM_BELOW_SPAZER,
        success_outcome="below_spazer_entry",
        route_label="start-to-Below-Spazer",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Red→Bat→Below Spazer) + phase-guarded resources"
        ),
        timing_source="start_to_below_spazer",
        entry_condition_key="natural_below_spazer_entry",
        ordinary_condition_key="post_below_spazer_ordinary",
    ),
    PostSupersTipSpec(
        tip_id="warehouse",
        parent_tip_id="below_spazer",
        hops=_WAREHOUSE_HOPS,
        graph=START_TO_WAREHOUSE_GRAPH,
        kind="warehouse",
        required_splits=WAREHOUSE_SPLITS,
        final_room=ROOM_WAREHOUSE,
        success_outcome="warehouse_entry",
        route_label="start-to-Warehouse-Entrance",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Bat→Below Spazer→West→Glass→East→Warehouse) + "
            "phase-guarded resources"
        ),
        timing_source="start_to_warehouse",
        entry_condition_key="natural_warehouse_entry",
        ordinary_condition_key="post_warehouse_ordinary",
    ),
    PostSupersTipSpec(
        tip_id="hijump",
        parent_tip_id="warehouse",
        hops=_HIJUMP_HOPS,
        graph=START_TO_HIJUMP_GRAPH,
        kind="hijump",
        required_splits=HIJUMP_SPLITS,
        final_room=ROOM_HJ,
        success_outcome="hijump_collected",
        route_label="start-to-Hi-Jump",
        source_policy=(
            "accepted Warehouse continuous prefix + KPDR Hi-Jump controllers "
            "(Warehouse→Business→shaft→HJ room collect) + phase-guarded resources"
        ),
        timing_source="start_to_hijump",
        entry_condition_key="natural_hijump_room",
        ordinary_condition_key="post_hijump_ordinary",
        require_hi_jump=True,
    ),
    PostSupersTipSpec(
        tip_id="kraid",
        parent_tip_id="hijump",
        hops=_KRAID_HOPS,
        graph=START_TO_KRAID_GRAPH,
        kind="kraid",
        required_splits=KRAID_SPLITS,
        final_room=ROOM_KRAID,
        success_outcome="kraid_entry",
        route_label="start-to-Kraid-entry",
        source_policy=(
            "accepted Warehouse prefix + Hi-Jump collect/return + KPDR Kraid "
            "approach controllers + phase-guarded resources"
        ),
        timing_source="start_to_kraid",
        entry_condition_key="natural_kraid_entry",
        ordinary_condition_key="post_kraid_ordinary",
        require_hi_jump=True,
    ),
    PostSupersTipSpec(
        tip_id="varia",
        parent_tip_id="kraid",
        hops=_VARIA_HOPS,
        graph=START_TO_VARIA_GRAPH,
        kind="varia",
        required_splits=VARIA_SPLITS,
        final_room=ROOM_VARIA,
        success_outcome="varia_collected",
        route_label="start-to-Varia",
        source_policy=(
            "accepted Kraid-entry continuous chain + combat.kraid fight/Varia "
            "policy + phase-guarded resources"
        ),
        timing_source="start_to_varia",
        entry_condition_key="natural_varia_room",
        ordinary_condition_key="post_varia_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    PostSupersTipSpec(
        tip_id="business",
        parent_tip_id="varia",
        hops=_BUSINESS_RETURN_HOPS,
        graph=START_TO_SPEED_GRAPH,
        kind="business",
        required_splits=BUSINESS_RETURN_SPLITS,
        final_room=ROOM_BUSINESS,
        success_outcome="business_return",
        route_label="start-to-Business-return",
        source_policy=(
            "accepted Varia continuous chain + natural K3 return controllers "
            "(Varia→Kraid→Eye→Baby→Kihunter→Zeela→Warehouse→Business) + "
            "phase-guarded resources"
        ),
        timing_source="start_to_business",
        entry_condition_key="natural_business_return",
        ordinary_condition_key="post_business_return_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    PostSupersTipSpec(
        tip_id="frog",
        parent_tip_id="business",
        hops=_FROG_ONLY_HOPS,
        graph=START_TO_SPEED_GRAPH,
        kind="frog_save",
        required_splits=FROG_SAVE_SPLITS,
        final_room=ROOM_FROG_SAVE,
        success_outcome="frog_save_reached",
        route_label="start-to-Frog-Save",
        source_policy=(
            "accepted Business return chain + natural Business elevator descent "
            "and Frog blue-door controller + phase-guarded resources"
        ),
        timing_source="start_to_frog_save",
        entry_condition_key="natural_frog_save",
        ordinary_condition_key="post_frog_save_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    PostSupersTipSpec(
        tip_id="bat_cave",
        parent_tip_id="business",
        hops=_BAT_CAVE_ONLY_HOPS,
        graph=START_TO_SPEED_GRAPH,
        kind="bat_cave",
        required_splits=BAT_CAVE_SPLITS,
        final_room=ROOM_BAT_CAVE,
        success_outcome="bat_cave_reached",
        route_label="start-to-Bat-Cave",
        source_policy=(
            "accepted Business return chain + Cathedral first-Bubble pure "
            "controllers (CATH-01…04 + Bubble R19 double-WJ fire / Super door) "
            "+ phase-guarded resources"
        ),
        timing_source="start_to_bat_cave",
        entry_condition_key="natural_bat_cave_entry",
        ordinary_condition_key="post_bat_cave_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
)

POST_SUPERS_TIP_BY_ID: dict[str, PostSupersTipSpec] = {
    spec.tip_id: spec for spec in POST_SUPERS_TIP_SPECS
}
