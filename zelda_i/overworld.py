"""Zelda I overworld screen graph and early route anchors.

Overworld is a 16×8 screen grid. Screen id = (row << 4) | col.
Caves and dungeons hang off screens as portal nodes in the route graph.
"""

from __future__ import annotations

from typing import Iterable

from adventure_common.graph import (
    GraphEdge,
    GraphNode,
    RouteGraph,
    RouteLeg,
)

# Grid geometry
OVERWORLD_COLS = 16
OVERWORLD_ROWS = 8

# Anchors (first quest)
SCREEN_START = 0x77
SCREEN_NORTH_OF_START = 0x67
SCREEN_LEVEL1 = 0x37

NODE_START = "ow_77"
NODE_SWORD_CAVE = "cave_sword_77"
NODE_START_EXIT = "ow_77_post_sword"
NODE_LEVEL1_ENTRANCE = "ow_37"
NODE_LEVEL1_DUNGEON = "dungeon_level1"
NODE_LEVEL1_ENTRY_ROOM = "level1_room_73"
NODE_LEVEL1_FIRST_KEY_ROOM = "level1_room_74"
NODE_LEVEL1_FIRST_KEY = "level1_first_key"
NODE_LEVEL1_NORTH_ROOM = "level1_room_63"
NODE_LEVEL1_NORTH_CLEARED = "level1_room_63_cleared"
NODE_LEVEL1_ROOM_53 = "level1_room_53"
NODE_LEVEL1_ROOM_53_CLEARED = "level1_room_53_cleared"
NODE_LEVEL1_ROOM_52 = "level1_room_52"
NODE_LEVEL1_ROOM_54 = "level1_room_54"
NODE_LEVEL1_ROOM_54_CLEARED = "level1_room_54_cleared"

# Probe-verified path screens (2026-07-28): east-then-north, not col-7 straight
LEVEL1_PATH_SCREENS: tuple[int, ...] = (
    0x77,
    0x78,
    0x68,
    0x58,
    0x48,
    0x38,
    0x37,
)

SCREEN_LABELS: dict[int, str] = {
    0x77: "start",
    0x78: "east_of_start",
    0x68: "bush_field",
    0x58: "bush_grid",
    0x48: "pre_level1_east",
    0x38: "level1_east",
    0x37: "level1_entrance",
    0x67: "north_of_start_deadend",
    0x47: "central_lake",
}


def screen_id(col: int, row: int) -> int:
    if not (0 <= col < OVERWORLD_COLS and 0 <= row < OVERWORLD_ROWS):
        raise ValueError(f"screen col/row out of range: {col},{row}")
    return (row << 4) | col


def screen_to_grid(screen: int) -> tuple[int, int]:
    s = int(screen) & 0x7F
    return s & 0x0F, (s >> 4) & 0x0F


def neighbor_screens(screen: int) -> dict[str, int | None]:
    col, row = screen_to_grid(screen)
    return {
        "north": screen_id(col, row - 1) if row > 0 else None,
        "south": screen_id(col, row + 1) if row < OVERWORLD_ROWS - 1 else None,
        "west": screen_id(col - 1, row) if col > 0 else None,
        "east": screen_id(col + 1, row) if col < OVERWORLD_COLS - 1 else None,
    }


def direction_between(current: int, target: int) -> str | None:
    c0, r0 = screen_to_grid(current)
    c1, r1 = screen_to_grid(target)
    if r1 < r0:
        return "UP"
    if r1 > r0:
        return "DOWN"
    if c1 < c0:
        return "LEFT"
    if c1 > c0:
        return "RIGHT"
    return None


def node_id_for_screen(screen: int) -> str:
    return f"ow_{int(screen) & 0x7F:02x}"


def build_overworld_grid_graph(
    *,
    screens: Iterable[int] | None = None,
) -> RouteGraph:
    """Build a bidirectional grid graph for the given screens (default: full 128)."""
    if screens is None:
        screen_list = [
            screen_id(c, r)
            for r in range(OVERWORLD_ROWS)
            for c in range(OVERWORLD_COLS)
        ]
    else:
        screen_list = list(screens)

    nodes = [
        GraphNode(
            node_id=node_id_for_screen(s),
            name=SCREEN_LABELS.get(s, f"screen_{s:02X}"),
            area="overworld",
            tags=frozenset({"overworld", "screen"}),
            meta={"screen": s, "col": screen_to_grid(s)[0], "row": screen_to_grid(s)[1]},
        )
        for s in screen_list
    ]
    present = {n.node_id for n in nodes}
    edges: list[GraphEdge] = []
    for s in screen_list:
        for direction, neighbor in neighbor_screens(s).items():
            if neighbor is None:
                continue
            src = node_id_for_screen(s)
            dst = node_id_for_screen(neighbor)
            if dst not in present:
                continue
            edges.append(
                GraphEdge(
                    source_id=src,
                    target_id=dst,
                    direction=direction,
                    verification="planned",
                    provenance="overworld_grid",
                    meta={"from_screen": s, "to_screen": neighbor},
                )
            )
    return RouteGraph(nodes, edges)


def build_early_route_graph() -> RouteGraph:
    """Early-game graph: start, sword cave, verified path to Level 1.

    Level 1 is **not** a straight col-7 north run (0x67 is a dead-end grove;
    0x47 is a lake). Verified path::

        0x77 → E 0x78 → N 0x68 → N 0x58 → N 0x48 → N 0x38 → W 0x37 → dungeon
    """
    seed_screens = set(LEVEL1_PATH_SCREENS)
    seed_screens.update(
        {
            SCREEN_NORTH_OF_START,  # 0x67 dead-end (documented trap)
            0x76,
            0x66,
            0x47,
            0x57,
            0x59,
        }
    )
    # Neighborhood around the verified path for expansion
    for sc in LEVEL1_PATH_SCREENS:
        for neighbor in neighbor_screens(sc).values():
            if neighbor is not None:
                seed_screens.add(neighbor)

    graph = build_overworld_grid_graph(screens=sorted(seed_screens))

    # Promote verified path edges
    verified_hops = list(zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]))
    promoted: list[GraphEdge] = []
    for edge in graph.edges:
        pair = (edge.meta.get("from_screen"), edge.meta.get("to_screen"))
        if pair in verified_hops or (
            edge.meta.get("from_screen"),
            edge.meta.get("to_screen"),
        ) in {(a, b) for a, b in verified_hops}:
            promoted.append(
                GraphEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_id=edge.edge_id,
                    direction=edge.direction,
                    requires=edge.requires,
                    cost=edge.cost,
                    verification="observed",
                    provenance="emulator_probe",
                    meta={**dict(edge.meta), "segment": "to_level1"},
                )
            )
        else:
            promoted.append(edge)

    # Portal nodes
    extra_nodes = [
        GraphNode(
            node_id=NODE_SWORD_CAVE,
            name="wooden_sword_cave",
            area="cave",
            tags=frozenset({"cave", "item", "sword"}),
            meta={"overworld_screen": SCREEN_START, "item": "wooden_sword"},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_DUNGEON,
            name="level1_eagle",
            area="dungeon",
            tags=frozenset({"dungeon", "level1"}),
            meta={"overworld_screen": SCREEN_LEVEL1, "level": 1},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ENTRY_ROOM,
            name="level1_entrance_room",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room"}),
            meta={"level": 1, "room": 0x73},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            name="level1_first_key_room",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "combat"}),
            meta={"level": 1, "room": 0x74, "enemies": "stalfos"},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_FIRST_KEY,
            name="level1_first_key",
            area="level1",
            tags=frozenset({"dungeon", "level1", "item", "key"}),
            meta={"level": 1, "room": 0x74, "item": "small_key"},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_NORTH_ROOM,
            name="level1_north_stalfos_room",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "combat"}),
            meta={"level": 1, "room": 0x63, "enemies": "3_stalfos"},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_NORTH_CLEARED,
            name="level1_north_stalfos_cleared",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "cleared"}),
            meta={
                "level": 1,
                "room": 0x63,
                "reward": "none",
                "doors": {"south": 0x73, "north": 0x53},
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ROOM_53,
            name="level1_room_53_five_stalfos",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "combat"}),
            meta={"level": 1, "room": 0x53, "enemies": "5_stalfos", "item_id": 0x19},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ROOM_53_CLEARED,
            name="level1_room_53_cleared_key_collected",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "cleared", "key"}),
            meta={
                "level": 1,
                "room": 0x53,
                "reward": "small_key",
                "doors": {"south": 0x63, "west": 0x52, "east": 0x54},
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ROOM_52,
            name="level1_room_52_six_keese",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "combat"}),
            meta={"level": 1, "room": 0x52, "enemies": "6_keese", "item_id": 0x03},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ROOM_54,
            name="level1_room_54_eight_keese",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "combat"}),
            meta={"level": 1, "room": 0x54, "enemies": "8_keese", "item_id": 0x16},
        ),
        GraphNode(
            node_id=NODE_LEVEL1_ROOM_54_CLEARED,
            name="level1_room_54_eight_keese_cleared",
            area="level1",
            tags=frozenset({"dungeon", "level1", "room", "cleared"}),
            meta={
                "level": 1,
                "room": 0x54,
                "reward": "no_known_inventory_change",
                "item_id": 0x16,
                "doors": {"west": 0x53, "east": "blocked"},
            },
        ),
    ]
    extra_edges = [
        GraphEdge(
            source_id=NODE_START,
            target_id=NODE_SWORD_CAVE,
            edge_id="enter_sword_cave",
            direction="IN",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "sword_cave"},
        ),
        GraphEdge(
            source_id=NODE_SWORD_CAVE,
            target_id=NODE_START,
            edge_id="exit_sword_cave",
            direction="OUT",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "sword_cave", "acquires_on_clear": ["wooden_sword"]},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ENTRANCE,
            target_id=NODE_LEVEL1_DUNGEON,
            edge_id="enter_level1",
            direction="IN",
            requires=frozenset({"wooden_sword"}),
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "to_level1", "door": "tree_mouth", "approach": (112, 140)},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_DUNGEON,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            edge_id="settle_level1_entrance",
            direction="IN",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "level1_first_key", "room": 0x73},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            edge_id="enter_level1_first_key_room",
            direction="RIGHT",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "level1_first_key", "from_room": 0x73, "to_room": 0x74},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY,
            edge_id="collect_level1_first_key",
            direction="ITEM",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "level1_first_key", "item_id": 0x19},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_FIRST_KEY,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            edge_id="resume_after_level1_first_key",
            direction="ROOM",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "level1_north", "room": 0x74},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            edge_id="return_to_level1_entrance",
            direction="LEFT",
            verification="observed",
            provenance="emulator_probe",
            meta={"segment": "level1_north", "from_room": 0x74, "to_room": 0x73},
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_NORTH_ROOM,
            edge_id="unlock_level1_north",
            direction="UP",
            requires=frozenset({"keys"}),
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_north",
                "from_room": 0x73,
                "to_room": 0x63,
                "consumes": {"keys": 1},
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_NORTH_ROOM,
            target_id=NODE_LEVEL1_NORTH_CLEARED,
            edge_id="clear_level1_room_63",
            direction="COMBAT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear63",
                "room": 0x63,
                "enemies": "3_stalfos",
                "reward": "none",
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_NORTH_CLEARED,
            target_id=NODE_LEVEL1_ROOM_53,
            edge_id="enter_level1_room_53",
            direction="UP",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear63",
                "from_room": 0x63,
                "to_room": 0x53,
                "note": "north door open without clear; clear preferred",
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ROOM_53,
            target_id=NODE_LEVEL1_ROOM_53_CLEARED,
            edge_id="clear_level1_room_53",
            direction="COMBAT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear53",
                "room": 0x53,
                "enemies": "5_stalfos",
                "reward": "small_key",
                "item_id": 0x19,
                "item_position": (128, 109),
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_ROOM_52,
            edge_id="enter_level1_room_52",
            direction="LEFT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear53",
                "from_room": 0x53,
                "to_room": 0x52,
                "enemies": "6_keese",
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_ROOM_54,
            edge_id="enter_level1_room_54",
            direction="RIGHT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear53",
                "from_room": 0x53,
                "to_room": 0x54,
                "enemies": "8_keese",
                "item_id": 0x16,
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_ROOM_54,
            target_id=NODE_LEVEL1_ROOM_54_CLEARED,
            edge_id="clear_level1_room_54",
            direction="COMBAT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_clear54",
                "room": 0x54,
                "enemies": "8_keese",
                "reward": "no_known_inventory_change",
                "item_id": 0x16,
            },
        ),
    ]
    nodes = list(graph.nodes.values()) + extra_nodes
    edges = list(promoted) + extra_edges
    return RouteGraph(nodes, edges)


def sword_cave_route_legs() -> tuple[RouteLeg, ...]:
    """Canonical M3 segment: start overworld → sword cave → back with sword."""
    return (
        RouteLeg(
            leg_id="enter_sword_cave",
            source_id=NODE_START,
            target_id=NODE_SWORD_CAVE,
            acquires=frozenset(),
            goal="enter_wooden_sword_cave",
        ),
        RouteLeg(
            leg_id="take_wooden_sword_and_exit",
            source_id=NODE_SWORD_CAVE,
            target_id=NODE_START,
            acquires=frozenset({"wooden_sword"}),
            goal="wooden_sword_on_start_screen",
        ),
    )


def level1_route_legs() -> tuple[RouteLeg, ...]:
    """Sword cave + overworld path into Level 1 dungeon."""
    hops = list(zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]))
    ow_legs = tuple(
        RouteLeg(
            leg_id=f"ow_{src:02x}_to_{dst:02x}",
            source_id=node_id_for_screen(src),
            target_id=node_id_for_screen(dst),
            requires=frozenset({"wooden_sword"}),
            goal=f"reach_screen_{dst:02X}",
        )
        for src, dst in hops
    )
    return (
        *sword_cave_route_legs(),
        *ow_legs,
        RouteLeg(
            leg_id="enter_level1_dungeon",
            source_id=NODE_LEVEL1_ENTRANCE,
            target_id=NODE_LEVEL1_DUNGEON,
            requires=frozenset({"wooden_sword"}),
            goal="inside_level1",
        ),
    )


def level1_first_key_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the first Level 1 key."""
    return (
        *level1_route_legs(),
        RouteLeg(
            leg_id="settle_level1_entrance",
            source_id=NODE_LEVEL1_DUNGEON,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_73_ready",
        ),
        RouteLeg(
            leg_id="enter_level1_first_key_room",
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_74",
        ),
        RouteLeg(
            leg_id="collect_level1_first_key",
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY,
            requires=frozenset({"wooden_sword"}),
            acquires=frozenset({"keys"}),
            goal="level1_keys_at_least_1",
        ),
    )


def level1_north_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the north door into room 0x63."""
    return (
        *level1_first_key_route_legs(),
        RouteLeg(
            leg_id="resume_after_level1_first_key",
            source_id=NODE_LEVEL1_FIRST_KEY,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="level1_first_key_room_with_key",
        ),
        RouteLeg(
            leg_id="return_to_level1_entrance",
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="return_level1_room_73",
        ),
        RouteLeg(
            leg_id="unlock_level1_north",
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_NORTH_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="level1_room_63_ready",
            constraints=("consumes_one_key",),
        ),
    )


def level1_clear63_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix that clears the three Stalfos in room 0x63."""
    return (
        *level1_north_route_legs(),
        RouteLeg(
            leg_id="clear_level1_room_63",
            source_id=NODE_LEVEL1_NORTH_ROOM,
            target_id=NODE_LEVEL1_NORTH_CLEARED,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_63_cleared",
        ),
    )


def level1_clear53_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the room 0x53 clear and fixed key."""
    return (
        *level1_clear63_route_legs(),
        RouteLeg(
            leg_id="enter_level1_room_53",
            source_id=NODE_LEVEL1_NORTH_CLEARED,
            target_id=NODE_LEVEL1_ROOM_53,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_53",
        ),
        RouteLeg(
            leg_id="clear_level1_room_53",
            source_id=NODE_LEVEL1_ROOM_53,
            target_id=NODE_LEVEL1_ROOM_53_CLEARED,
            requires=frozenset({"wooden_sword"}),
            acquires=frozenset({"keys"}),
            goal="level1_room_53_cleared_and_key_collected",
        ),
    )


def level1_clear54_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the east branch and room 0x54 clear."""
    return (
        *level1_clear53_route_legs(),
        RouteLeg(
            leg_id="enter_level1_room_54",
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_ROOM_54,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_54",
        ),
        RouteLeg(
            leg_id="clear_level1_room_54",
            source_id=NODE_LEVEL1_ROOM_54,
            target_id=NODE_LEVEL1_ROOM_54_CLEARED,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_54_cleared",
        ),
    )


def early_route_plan(graph: RouteGraph | None = None):
    """Plan the sword-cave legs on the early graph (empty initial inventory)."""
    g = graph or build_early_route_graph()
    return g.plan_legs(sword_cave_route_legs(), initial_capabilities=frozenset())


def level1_route_plan(graph: RouteGraph | None = None):
    """Plan sword + overworld path into Level 1."""
    g = graph or build_early_route_graph()
    return g.plan_legs(level1_route_legs(), initial_capabilities=frozenset())


def level1_first_key_route_plan(graph: RouteGraph | None = None):
    """Plan sword + overworld + Level 1 first-key segment."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_first_key_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_north_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the first locked Level 1 door."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_north_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear63_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the room 0x63 Stalfos clear."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear63_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear53_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the room 0x53 Stalfos clear and room key."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear53_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear54_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the observed east branch room 0x54 clear."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear54_route_legs(),
        initial_capabilities=frozenset(),
    )
