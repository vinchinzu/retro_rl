"""Named routes for Zelda I full-run scaffolding."""

from __future__ import annotations

from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route as _get_route,
    list_routes as _list_routes,
)
from zelda_i.overworld import (
    NODE_LEVEL1_DUNGEON,
    NODE_LEVEL1_COMPLETE,
    NODE_LEVEL1_ENTRANCE,
    NODE_LEVEL1_ENTRY_ROOM,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL1_FIRST_KEY,
    NODE_LEVEL1_FIRST_KEY_ROOM,
    NODE_LEVEL1_NORTH_CLEARED,
    NODE_LEVEL1_NORTH_ROOM,
    NODE_LEVEL1_ROOM_53_CLEARED,
    NODE_LEVEL1_ROOM_54_CLEARED,
    NODE_LEVEL2_PATH_4A,
    NODE_START,
    NODE_SWORD_CAVE,
)
from zelda_i.route_legs import (
    level1_clear53_route_legs,
    level1_clear54_route_legs,
    level1_complete_route_legs,
    level1_clear63_route_legs,
    level1_first_key_route_legs,
    level1_north_route_legs,
    level1_route_legs,
    level2_path_prefix_route_legs,
    sword_cave_route_legs,
)


ROUTE_SWORD_CAVE = NamedRoute(
    route_id="zelda_sword_cave",
    display_name="Wooden Sword Cave",
    description=(
        "From overworld start (0x77), enter the NW cave, take the wooden sword, "
        "return to the start screen with sword>=1."
    ),
    milestones=(
        RouteMilestone(
            "start_overworld",
            NODE_START,
            "Start overworld",
            "is_on_start_overworld",
        ),
        RouteMilestone(
            "sword_cave",
            NODE_SWORD_CAVE,
            "Sword cave",
            "in_cave",
        ),
        RouteMilestone(
            "sword_obtained",
            NODE_START,
            "Start screen with wooden sword",
            "has_sword_on_start",
        ),
    ),
)

ROUTE_TO_LEVEL1 = NamedRoute(
    route_id="zelda_to_level1",
    display_name="Start → Level 1 Entrance",
    description=(
        "Sword cave clear, then east-north overworld path "
        "0x77→0x78→0x68→0x58→0x48→0x38→0x37 into the Level 1 tree door. "
        "Not a straight col-7 north run (0x67 is a dead end; 0x47 is a lake)."
    ),
    milestones=(
        *ROUTE_SWORD_CAVE.milestones,
        RouteMilestone(
            "level1_overworld",
            NODE_LEVEL1_ENTRANCE,
            "Level 1 overworld screen",
            "level1_screen_reached",
        ),
        RouteMilestone(
            "level1_dungeon",
            NODE_LEVEL1_DUNGEON,
            "Inside Level 1 (Eagle)",
            "level1_entrance_success",
        ),
    ),
)

ROUTE_LEVEL1_FIRST_KEY = NamedRoute(
    route_id="zelda_level1_first_key",
    display_name="Start → Level 1 First Key",
    description=(
        "Power-on route through the wooden sword and Level 1 entrance, then "
        "east from room 0x73 into 0x74 to collect the carried room key."
    ),
    milestones=(
        *ROUTE_TO_LEVEL1.milestones,
        RouteMilestone(
            "level1_entry_room",
            NODE_LEVEL1_ENTRY_ROOM,
            "Level 1 entrance room 0x73",
            "level1_entry_room_ready",
        ),
        RouteMilestone(
            "level1_first_key_room",
            NODE_LEVEL1_FIRST_KEY_ROOM,
            "Level 1 east room 0x74",
            "level1_first_key_room_ready",
        ),
        RouteMilestone(
            "level1_first_key",
            NODE_LEVEL1_FIRST_KEY,
            "First Level 1 key",
            "level1_first_key_success",
        ),
    ),
)

ROUTE_LEVEL1_NORTH = NamedRoute(
    route_id="zelda_level1_north",
    display_name="Start → Level 1 North Room",
    description=(
        "Power-on through the first Level 1 key, return west to room 0x73, "
        "spend the key on the north door, and settle in room 0x63."
    ),
    milestones=(
        *ROUTE_LEVEL1_FIRST_KEY.milestones,
        RouteMilestone(
            "level1_north_room",
            NODE_LEVEL1_NORTH_ROOM,
            "Level 1 north room 0x63",
            "level1_north_room_success",
        ),
    ),
)

ROUTE_LEVEL1_CLEAR63 = NamedRoute(
    route_id="zelda_level1_clear63",
    display_name="Start → Level 1 Room 0x63 Cleared",
    description=(
        "Power-on through room 0x63 entry, then clear its three Stalfos. "
        "No inventory reward; north door remains open into room 0x53."
    ),
    milestones=(
        *ROUTE_LEVEL1_NORTH.milestones,
        RouteMilestone(
            "level1_room_63_cleared",
            NODE_LEVEL1_NORTH_CLEARED,
            "Level 1 room 0x63 cleared",
            "level1_room_63_cleared",
        ),
    ),
)

ROUTE_LEVEL1_CLEAR53 = NamedRoute(
    route_id="zelda_level1_clear53",
    display_name="Start → Level 1 Room 0x53 Key",
    description=(
        "Power-on through the room 0x63 clear, enter 0x53, clear five "
        "Stalfos, and collect the fixed room key at (128, 109)."
    ),
    milestones=(
        *ROUTE_LEVEL1_CLEAR63.milestones,
        RouteMilestone(
            "level1_room_53_cleared",
            NODE_LEVEL1_ROOM_53_CLEARED,
            "Level 1 room 0x53 cleared and key collected",
            "level1_room_53_cleared",
        ),
    ),
)

ROUTE_LEVEL1_CLEAR54 = NamedRoute(
    route_id="zelda_level1_clear54",
    display_name="Start → Level 1 Room 0x54 Cleared",
    description=(
        "Power-on through the room 0x53 key, take the east branch, and clear "
        "eight Keese in room 0x54."
    ),
    milestones=(
        *ROUTE_LEVEL1_CLEAR53.milestones,
        RouteMilestone(
            "level1_room_54_cleared",
            NODE_LEVEL1_ROOM_54_CLEARED,
            "Level 1 room 0x54 cleared",
            "dungeon_room_cleared(ROOM_54_SPEC)",
        ),
    ),
)

ROUTE_LEVEL1_COMPLETE = NamedRoute(
    route_id="zelda_level1_complete",
    display_name="Start → Level 1 Triforce Shard 1",
    description=(
        "Power-on through the room 0x53 key, take the required west route, "
        "defeat Aquamentus, collect the Heart Container, and collect the "
        "first Triforce shard in room 0x36."
    ),
    milestones=(
        *ROUTE_LEVEL1_CLEAR53.milestones,
        RouteMilestone(
            "level1_complete",
            NODE_LEVEL1_COMPLETE,
            "Level 1 Triforce shard 1",
            "triforce & 0x01",
        ),
    ),
)

ROUTE_LEVEL2_PATH_PREFIX = NamedRoute(
    route_id="zelda_level2_path_prefix",
    display_name="Start → Level 2 Walk Prefix (0x4A)",
    description=(
        "Power-on through Level 1 Triforce, idle the fanfare settle onto "
        "overworld 0x37, then walk 0x37→38→48→58→59→49→4A. Avoids the "
        "0x79 rocky dead-end. Continuation to 0x3C needs heart-safe combat."
    ),
    milestones=(
        *ROUTE_LEVEL1_COMPLETE.milestones,
        RouteMilestone(
            "level1_exit_overworld",
            NODE_LEVEL1_EXIT_OVERWORLD,
            "Post-Triforce overworld at Level 1 mouth",
            "post_triforce_overworld_ready",
        ),
        RouteMilestone(
            "level2_path_4a",
            NODE_LEVEL2_PATH_4A,
            "Level 2 path screen 0x4A",
            "level2_path_prefix_success",
        ),
    ),
)

ROUTE_REGISTRY: dict[str, NamedRoute] = {
    ROUTE_SWORD_CAVE.route_id: ROUTE_SWORD_CAVE,
    "sword": ROUTE_SWORD_CAVE,
    "sword_cave": ROUTE_SWORD_CAVE,
    ROUTE_TO_LEVEL1.route_id: ROUTE_TO_LEVEL1,
    "level1": ROUTE_TO_LEVEL1,
    "to_level1": ROUTE_TO_LEVEL1,
    ROUTE_LEVEL1_FIRST_KEY.route_id: ROUTE_LEVEL1_FIRST_KEY,
    "first_key": ROUTE_LEVEL1_FIRST_KEY,
    "level1_first_key": ROUTE_LEVEL1_FIRST_KEY,
    ROUTE_LEVEL1_NORTH.route_id: ROUTE_LEVEL1_NORTH,
    "level1_north": ROUTE_LEVEL1_NORTH,
    "north_room": ROUTE_LEVEL1_NORTH,
    ROUTE_LEVEL1_CLEAR63.route_id: ROUTE_LEVEL1_CLEAR63,
    "level1_clear63": ROUTE_LEVEL1_CLEAR63,
    "clear63": ROUTE_LEVEL1_CLEAR63,
    ROUTE_LEVEL1_CLEAR53.route_id: ROUTE_LEVEL1_CLEAR53,
    "level1_clear53": ROUTE_LEVEL1_CLEAR53,
    "clear53": ROUTE_LEVEL1_CLEAR53,
    ROUTE_LEVEL1_CLEAR54.route_id: ROUTE_LEVEL1_CLEAR54,
    "level1_clear54": ROUTE_LEVEL1_CLEAR54,
    "clear54": ROUTE_LEVEL1_CLEAR54,
    ROUTE_LEVEL1_COMPLETE.route_id: ROUTE_LEVEL1_COMPLETE,
    "level1_complete": ROUTE_LEVEL1_COMPLETE,
    "triforce_1": ROUTE_LEVEL1_COMPLETE,
    ROUTE_LEVEL2_PATH_PREFIX.route_id: ROUTE_LEVEL2_PATH_PREFIX,
    "level2_prefix": ROUTE_LEVEL2_PATH_PREFIX,
    "to_level2_prefix": ROUTE_LEVEL2_PATH_PREFIX,
}


def get_route(route_id: str) -> NamedRoute:
    return _get_route(ROUTE_REGISTRY, route_id)


def list_routes() -> list[NamedRoute]:
    return _list_routes(ROUTE_REGISTRY)


# Re-export for segment policies
SWORD_CAVE_LEGS = sword_cave_route_legs()
LEVEL1_LEGS = level1_route_legs()
LEVEL1_FIRST_KEY_LEGS = level1_first_key_route_legs()
LEVEL1_NORTH_LEGS = level1_north_route_legs()
LEVEL1_CLEAR63_LEGS = level1_clear63_route_legs()
LEVEL1_CLEAR53_LEGS = level1_clear53_route_legs()
LEVEL1_CLEAR54_LEGS = level1_clear54_route_legs()
LEVEL1_COMPLETE_LEGS = level1_complete_route_legs()
LEVEL2_PATH_PREFIX_LEGS = level2_path_prefix_route_legs()
