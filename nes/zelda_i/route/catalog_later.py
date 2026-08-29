"""NamedRoutes for Zelda I Level 3–9.

L3–L5 are first-class room/item/TF milestones from ``docs/LEVELN_ROUTE.md``
(observed / assisted — not Clean). L6–L8 stay planning stubs. L9 is the
fixture endgame suffix (``route_eligible=false``).

L1/L2 published routes remain in ``zelda_i.route.catalog``.
"""

from __future__ import annotations

from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route as _get_route,
    list_routes as _list_routes,
    register_routes,
)
from zelda_i.route.nodes import (
    NODE_LEVEL3_BOSS,
    NODE_LEVEL3_COMPLETE,
    NODE_LEVEL3_ENTRANCE,
    NODE_LEVEL3_ENTRY_ROOM,
    NODE_LEVEL3_RAFT,
    NODE_LEVEL3_WEST_KEY,
    NODE_LEVEL4_BOSS,
    NODE_LEVEL4_COMPLETE,
    NODE_LEVEL4_ENTRANCE,
    NODE_LEVEL4_ENTRY_ROOM,
    NODE_LEVEL4_STEPLADDER,
    NODE_LEVEL5_BOSS,
    NODE_LEVEL5_COMPLETE,
    NODE_LEVEL5_EAST_77,
    NODE_LEVEL5_ENTRANCE,
    NODE_LEVEL5_ENTRY_ROOM,
    NODE_LEVEL5_KEY_66,
    NODE_LEVEL5_WHISTLE,
    NODE_LEVEL6_COMPLETE,
    NODE_LEVEL6_DUNGEON,
    NODE_LEVEL6_ENTRANCE,
    NODE_LEVEL7_COMPLETE,
    NODE_LEVEL7_DUNGEON,
    NODE_LEVEL7_ENTRANCE,
    NODE_LEVEL8_COMPLETE,
    NODE_LEVEL8_DUNGEON,
    NODE_LEVEL8_ENTRANCE,
    NODE_LEVEL9_CELLAR_67,
    NODE_LEVEL9_DUNGEON,
    NODE_LEVEL9_ENTRANCE,
    NODE_LEVEL9_GANON,
    NODE_LEVEL9_PATRA,
    NODE_LEVEL9_ROOM_03,
    NODE_LEVEL9_ROOM_04,
    NODE_LEVEL9_ROOM_30,
    NODE_LEVEL9_ROOM_31,
    NODE_LEVEL9_ROOM_41,
    NODE_LEVEL9_ZELDA,
    NODE_LOST_HILLS,
    NODE_RAFT_L4_DOCK,
    TF_BIT_L6,
    TF_BIT_L7,
    TF_BIT_L8,
    TF_BITS_ALL,
    TRIFORCE_BITS_BY_LEVEL,
)
from zelda_i.route.legs_later import (
    LEVEL3_COMPLETE_LEGS,
    LEVEL4_COMPLETE_LEGS,
    LEVEL5_COMPLETE_LEGS,
    LEVEL9_FIXTURE_LEGS,
    build_later_route_graph,
    level3_complete_route_plan,
    level4_complete_route_plan,
    level5_complete_route_plan,
    level9_fixture_route_plan,
)

__all__ = [
    "ROUTE_LEVEL3_COMPLETE",
    "ROUTE_LEVEL4_COMPLETE",
    "ROUTE_LEVEL5_COMPLETE",
    "ROUTE_LEVEL6_COMPLETE",
    "ROUTE_LEVEL7_COMPLETE",
    "ROUTE_LEVEL8_COMPLETE",
    "ROUTE_LEVEL9_GANON",
    "ROUTE_REGISTRY_LATER",
    "TRIFORCE_BITS_BY_LEVEL",
    "TF_BITS_ALL",
    "LEVEL3_COMPLETE_LEGS",
    "LEVEL4_COMPLETE_LEGS",
    "LEVEL5_COMPLETE_LEGS",
    "LEVEL9_FIXTURE_LEGS",
    "build_later_route_graph",
    "get_later_route",
    "level3_complete_route_plan",
    "level4_complete_route_plan",
    "level5_complete_route_plan",
    "level9_fixture_route_plan",
    "list_later_routes",
]


_STUB_NOTE = (
    "PLANNING STUB — not route-ready. Door screens and stop predicates are "
    "placeholders until live recon (OVERWORLD_DOORS.md). No Clean claim."
)
_ASSISTED_NOTE = (
    "Observed / assisted from LEVEL{n}_ROUTE.md. Not Clean STATUS; "
    "not route-ready as a power-on compose."
)
_L9_FIXTURE_NOTE = (
    "Fixture-only endgame suffix. route_eligible=false. Composed inventory "
    "and room loader — not a natural Level 9 route. Observed rooms from "
    "LEVEL9_ROUTE.md. No Clean claim."
)


def _door_milestones(
    *,
    level: int,
    name: str,
    entrance_node: str,
    dungeon_node: str,
    door_pred: str,
    enter_pred: str,
) -> tuple[RouteMilestone, ...]:
    return (
        RouteMilestone(
            f"level{level}_entrance",
            entrance_node,
            f"Level {level} ({name}) overworld door (source candidate)",
            door_pred,
        ),
        RouteMilestone(
            f"level{level}_dungeon",
            dungeon_node,
            f"Inside Level {level} ({name})",
            enter_pred,
        ),
    )


def _complete_milestones(
    *,
    level: int,
    name: str,
    entrance_node: str,
    dungeon_node: str,
    complete_node: str,
    tf_bit: int,
    door_pred: str,
    enter_pred: str,
    item_label: str,
) -> tuple[RouteMilestone, ...]:
    return (
        *_door_milestones(
            level=level,
            name=name,
            entrance_node=entrance_node,
            dungeon_node=dungeon_node,
            door_pred=door_pred,
            enter_pred=enter_pred,
        ),
        RouteMilestone(
            f"level{level}_item",
            dungeon_node,
            f"Level {level} item: {item_label} (planning)",
            f"level{level}_item_collected",
        ),
        RouteMilestone(
            f"level{level}_complete",
            complete_node,
            f"Level {level} Triforce shard {level}",
            f"triforce & 0x{tf_bit:02x}",
        ),
    )


ROUTE_LEVEL3_COMPLETE = NamedRoute(
    route_id="zelda_level3_complete",
    display_name="Level 3 Manji → Triforce 0x04",
    description=(
        f"{_ASSISTED_NOTE.format(n=3)} OW 0x74 → entry 0x7c → west key 0x7b → "
        "Raft 0x0f → Manhandla 0x4d → TF 0x3d / triforce & 0x04."
    ),
    milestones=(
        RouteMilestone(
            "level3_entrance",
            NODE_LEVEL3_ENTRANCE,
            "Level 3 (Manji) overworld door 0x74",
            "level3_door_screen",
        ),
        RouteMilestone(
            "level3_entry_room",
            NODE_LEVEL3_ENTRY_ROOM,
            "Level 3 entry room 0x7c",
            "level3_entrance_success",
        ),
        RouteMilestone(
            "level3_west_key",
            NODE_LEVEL3_WEST_KEY,
            "Level 3 west key room 0x7b",
            "level3_room_7b_key_success",
        ),
        RouteMilestone(
            "level3_raft",
            NODE_LEVEL3_RAFT,
            "Level 3 Raft (ADDR_RAFT)",
            "level3_raft_collected",
        ),
        RouteMilestone(
            "level3_manhandla",
            NODE_LEVEL3_BOSS,
            "Level 3 Manhandla 0x4d",
            "level3_boss_cleared",
        ),
        RouteMilestone(
            "level3_complete",
            NODE_LEVEL3_COMPLETE,
            "Level 3 Triforce shard 3 (room 0x3d)",
            "triforce & 0x04",
        ),
    ),
)

ROUTE_LEVEL4_COMPLETE = NamedRoute(
    route_id="zelda_level4_complete",
    display_name="Level 4 Snake → Triforce 0x08",
    description=(
        f"{_ASSISTED_NOTE.format(n=4)} Requires Raft. Dock 0x55 → island 0x45 → "
        "entry 0x71 → Stepladder 0x60 → Gleeok 0x13 → TF 0x03 / triforce & 0x08."
    ),
    milestones=(
        RouteMilestone(
            "level4_dock",
            NODE_RAFT_L4_DOCK,
            "Level 4 raft dock 0x55",
            "level4_dock_screen",
        ),
        RouteMilestone(
            "level4_entrance",
            NODE_LEVEL4_ENTRANCE,
            "Level 4 island door 0x45",
            "level4_door_screen",
        ),
        RouteMilestone(
            "level4_entry_room",
            NODE_LEVEL4_ENTRY_ROOM,
            "Level 4 entry room 0x71",
            "level4_entrance_success",
        ),
        RouteMilestone(
            "level4_stepladder",
            NODE_LEVEL4_STEPLADDER,
            "Level 4 Stepladder (room 0x60)",
            "level4_stepladder_collected",
        ),
        RouteMilestone(
            "level4_gleeok",
            NODE_LEVEL4_BOSS,
            "Level 4 Gleeok 0x13",
            "level4_boss_cleared",
        ),
        RouteMilestone(
            "level4_complete",
            NODE_LEVEL4_COMPLETE,
            "Level 4 Triforce shard 4 (room 0x03)",
            "triforce & 0x08",
        ),
    ),
)

ROUTE_LEVEL5_COMPLETE = NamedRoute(
    route_id="zelda_level5_complete",
    display_name="Level 5 Lizard → Triforce 0x10",
    description=(
        f"{_ASSISTED_NOTE.format(n=5)} Lost Hills 0x1B → door 0x0B → entry 0x76 → "
        "0x66 key → east 0x77 → Recorder/Whistle → Digdogger 0x24 → "
        "TF 0x14 / triforce & 0x10."
    ),
    milestones=(
        RouteMilestone(
            "level5_lost_hills",
            NODE_LOST_HILLS,
            "Lost Hills 0x1B (UP×4)",
            "level5_lost_hills_screen",
        ),
        RouteMilestone(
            "level5_entrance",
            NODE_LEVEL5_ENTRANCE,
            "Level 5 door screen 0x0B",
            "level5_door_screen",
        ),
        RouteMilestone(
            "level5_entry_room",
            NODE_LEVEL5_ENTRY_ROOM,
            "Level 5 entry room 0x76",
            "level5_entrance_success",
        ),
        RouteMilestone(
            "level5_key_66",
            NODE_LEVEL5_KEY_66,
            "Level 5 first key room 0x66",
            "level5_room_66_cleared",
        ),
        RouteMilestone(
            "level5_east_77",
            NODE_LEVEL5_EAST_77,
            "Level 5 east key room 0x77",
            "level5_room_77_key_success",
        ),
        RouteMilestone(
            "level5_whistle",
            NODE_LEVEL5_WHISTLE,
            "Level 5 Recorder / Whistle (room 0x04)",
            "level5_whistle_collected",
        ),
        RouteMilestone(
            "level5_digdogger",
            NODE_LEVEL5_BOSS,
            "Level 5 Digdogger 0x24",
            "level5_boss_cleared",
        ),
        RouteMilestone(
            "level5_complete",
            NODE_LEVEL5_COMPLETE,
            "Level 5 Triforce shard 5 (room 0x14)",
            "triforce & 0x10",
        ),
    ),
)

ROUTE_LEVEL6_COMPLETE = NamedRoute(
    route_id="zelda_level6_complete",
    display_name="Level 6 Dragon (stub) → Triforce 0x20",
    description=(
        f"{_STUB_NOTE} Source door 0x22 near graveyard. Item Magical Rod. "
        "Boss Gohma (arrow eye). TF bit 0x20."
    ),
    milestones=_complete_milestones(
        level=6,
        name="Dragon",
        entrance_node=NODE_LEVEL6_ENTRANCE,
        dungeon_node=NODE_LEVEL6_DUNGEON,
        complete_node=NODE_LEVEL6_COMPLETE,
        tf_bit=TF_BIT_L6,
        door_pred="level6_door_screen",
        enter_pred="level6_entrance_success",
        item_label="Magical Rod",
    ),
)

ROUTE_LEVEL7_COMPLETE = NamedRoute(
    route_id="zelda_level7_complete",
    display_name="Level 7 Demon (stub) → Triforce 0x40",
    description=(
        f"{_STUB_NOTE} Source door 0x42 whistle pond. Item Red Candle. "
        "Boss Aquamentus. TF bit 0x40. Requires Whistle + Bait."
    ),
    milestones=_complete_milestones(
        level=7,
        name="Demon",
        entrance_node=NODE_LEVEL7_ENTRANCE,
        dungeon_node=NODE_LEVEL7_DUNGEON,
        complete_node=NODE_LEVEL7_COMPLETE,
        tf_bit=TF_BIT_L7,
        door_pred="level7_door_screen",
        enter_pred="level7_entrance_success",
        item_label="Red Candle",
    ),
)

ROUTE_LEVEL8_COMPLETE = NamedRoute(
    route_id="zelda_level8_complete",
    display_name="Level 8 Lion (stub) → Triforce 0x80",
    description=(
        f"{_STUB_NOTE} Source door 0x6D candle bush. Items Book of Magic + "
        "Magical Key. Boss Gleeok 4-head. TF bit 0x80."
    ),
    milestones=_complete_milestones(
        level=8,
        name="Lion",
        entrance_node=NODE_LEVEL8_ENTRANCE,
        dungeon_node=NODE_LEVEL8_DUNGEON,
        complete_node=NODE_LEVEL8_COMPLETE,
        tf_bit=TF_BIT_L8,
        door_pred="level8_door_screen",
        enter_pred="level8_entrance_success",
        item_label="Book of Magic / Magical Key",
    ),
)

ROUTE_LEVEL9_GANON = NamedRoute(
    route_id="zelda_level9_ganon",
    display_name="Level 9 fixture suffix → Ganon / Zelda",
    description=_L9_FIXTURE_NOTE,
    milestones=(
        RouteMilestone(
            "level9_entrance",
            NODE_LEVEL9_ENTRANCE,
            "Level 9 Spectacle Rock 0x05 (not route-ready)",
            "level9_door_screen",
        ),
        RouteMilestone(
            "level9_dungeon",
            NODE_LEVEL9_DUNGEON,
            "Inside Level 9 (fixture loader — route_eligible=false)",
            "level9_entrance_success",
        ),
        RouteMilestone(
            "level9_room_41",
            NODE_LEVEL9_ROOM_41,
            "Fixture suffix start 0x41 (route_eligible=false)",
            "level9_in_room_41",
        ),
        RouteMilestone(
            "level9_room_31",
            NODE_LEVEL9_ROOM_31,
            "Level 9 room 0x31 (fixture)",
            "level9_in_room_31",
        ),
        RouteMilestone(
            "level9_room_30",
            NODE_LEVEL9_ROOM_30,
            "Level 9 room 0x30 block-stairs (fixture)",
            "level9_in_room_30",
        ),
        RouteMilestone(
            "level9_cellar_67",
            NODE_LEVEL9_CELLAR_67,
            "Level 9 cellar 0x67 (fixture)",
            "level9_in_cellar_67",
        ),
        RouteMilestone(
            "level9_room_04",
            NODE_LEVEL9_ROOM_04,
            "Level 9 room 0x04 bomb-west (fixture)",
            "level9_in_room_04",
        ),
        RouteMilestone(
            "level9_room_03",
            NODE_LEVEL9_ROOM_03,
            "Level 9 room 0x03 Patra stairs (fixture)",
            "level9_in_room_03",
        ),
        RouteMilestone(
            "level9_patra",
            NODE_LEVEL9_PATRA,
            "Final Patra 0x52 (fixture_only)",
            "level9_patra_cleared",
        ),
        RouteMilestone(
            "level9_ganon",
            NODE_LEVEL9_GANON,
            "Ganon 0x42 defeated (fixture_only, route_eligible=false)",
            "level9_ganon_defeated",
        ),
        RouteMilestone(
            "level9_zelda",
            NODE_LEVEL9_ZELDA,
            "Princess Zelda 0x32 / credits (fixture_only)",
            "level9_ending",
        ),
    ),
)


ROUTE_REGISTRY_LATER: dict[str, NamedRoute] = {}
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL3_COMPLETE,
    "level3",
    "level3_complete",
    "triforce_3",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL4_COMPLETE,
    "level4",
    "level4_complete",
    "triforce_4",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL5_COMPLETE,
    "level5",
    "level5_complete",
    "triforce_5",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL6_COMPLETE,
    "level6",
    "level6_complete",
    "triforce_6",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL7_COMPLETE,
    "level7",
    "level7_complete",
    "triforce_7",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL8_COMPLETE,
    "level8",
    "level8_complete",
    "triforce_8",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL9_GANON,
    "level9",
    "level9_ganon",
    "ganon",
)


def get_later_route(route_id: str) -> NamedRoute:
    return _get_route(ROUTE_REGISTRY_LATER, route_id)


def list_later_routes() -> list[NamedRoute]:
    return _list_routes(ROUTE_REGISTRY_LATER)
