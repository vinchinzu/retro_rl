"""Named routes for Zelda I full-run scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.overworld import (
    NODE_LEVEL1_DUNGEON,
    NODE_LEVEL1_ENTRANCE,
    NODE_LEVEL1_ENTRY_ROOM,
    NODE_LEVEL1_FIRST_KEY,
    NODE_LEVEL1_FIRST_KEY_ROOM,
    NODE_LEVEL1_NORTH_CLEARED,
    NODE_LEVEL1_NORTH_ROOM,
    NODE_LEVEL1_ROOM_53_CLEARED,
    NODE_LEVEL1_ROOM_54_CLEARED,
    NODE_START,
    NODE_SWORD_CAVE,
    level1_clear53_route_legs,
    level1_clear54_route_legs,
    level1_clear63_route_legs,
    level1_first_key_route_legs,
    level1_north_route_legs,
    level1_route_legs,
    sword_cave_route_legs,
)


@dataclass(frozen=True)
class RouteMilestone:
    milestone_id: str
    node_id: str
    label: str
    stop_predicate: str
    """Machine name of the stop check (documented in STATUS / segment code)."""


@dataclass(frozen=True)
class NamedRoute:
    route_id: str
    display_name: str
    milestones: tuple[RouteMilestone, ...]
    description: str = ""


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
}


def get_route(route_id: str) -> NamedRoute:
    key = route_id.strip().lower()
    if key not in ROUTE_REGISTRY:
        available = sorted({r.route_id for r in ROUTE_REGISTRY.values()})
        raise KeyError(f"Unknown route {route_id!r}. Available: {available}")
    return ROUTE_REGISTRY[key]


def list_routes() -> list[NamedRoute]:
    seen: set[str] = set()
    out: list[NamedRoute] = []
    for route in ROUTE_REGISTRY.values():
        if route.route_id not in seen:
            seen.add(route.route_id)
            out.append(route)
    return out


# Re-export for segment policies
SWORD_CAVE_LEGS = sword_cave_route_legs()
LEVEL1_LEGS = level1_route_legs()
LEVEL1_FIRST_KEY_LEGS = level1_first_key_route_legs()
LEVEL1_NORTH_LEGS = level1_north_route_legs()
LEVEL1_CLEAR63_LEGS = level1_clear63_route_legs()
LEVEL1_CLEAR53_LEGS = level1_clear53_route_legs()
LEVEL1_CLEAR54_LEGS = level1_clear54_route_legs()
