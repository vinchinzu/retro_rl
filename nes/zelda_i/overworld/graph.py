"""Zelda I overworld screen graph and early route anchors.

Overworld is a 16×8 screen grid. Screen id = (row << 4) | col.
Caves and dungeons hang off screens as portal nodes in the route graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    RouteGraph,
)


@dataclass(frozen=True)
class ScreenHop:
    """One overworld screen transition with optional alignment constraints."""

    target: int
    direction: str  # UP / DOWN / LEFT / RIGHT
    align_x: int | None = None
    align_y: int | None = None
    y_band_lo: int | None = None
    y_band_hi: int | None = None

    @property
    def y_band(self) -> tuple[int, int] | None:
        if self.y_band_lo is None or self.y_band_hi is None:
            return None
        return (self.y_band_lo, self.y_band_hi)


def path_screens_from_hops(
    start: int,
    hops: tuple[ScreenHop, ...],
) -> tuple[int, ...]:
    return (start,) + tuple(hop.target for hop in hops)

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
NODE_LEVEL1_COMPLETE = "level1_triforce_shard_1"
NODE_LEVEL1_EXIT_OVERWORLD = "ow_37_post_triforce"
NODE_LEVEL2_PATH_4A = "ow_4a_level2_path"
NODE_LEVEL2_ENTRANCE = "ow_3c"
NODE_LEVEL2_DUNGEON = "dungeon_level2"

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

# Post-Triforce walk prefix (health-stable; verified 3/3). Avoids 0x79.
# Geometry drives level2_overworld controller default stop.
LEVEL2_PATH_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x38, "RIGHT", align_y=140),
    ScreenHop(0x48, "DOWN", align_x=120),
    ScreenHop(0x58, "DOWN", align_x=112),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)
LEVEL2_PATH_SCREENS: tuple[int, ...] = path_screens_from_hops(
    0x37, LEVEL2_PATH_HOPS
)

# Full walk to Moon door 0x3C (probe-verified 2026-07-29).
# Avoids 0x4B→0x5B north-entry (east sealed). Uses 0x5A west entry into 0x5B.
# 0x5C needs a mid-screen maze: east along y≈88, then down to y≈128, then east.
# 0x5D north exit only around x≈48–56. Clean health management still open.
LEVEL2_DOOR_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x38, "RIGHT", align_y=140),
    ScreenHop(0x48, "DOWN", align_x=120),
    ScreenHop(0x58, "DOWN", align_x=112),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x5A, "RIGHT", y_band_lo=120, y_band_hi=145),
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    # North bush corridor out of 0x5B (y≈80–95); not the south pocket.
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    # 0x5C→0x5D requires maze waypoints (see LEVEL2_ROUTE.md); hop alone is not enough.
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x4D, "UP", align_x=52),
    ScreenHop(0x4C, "LEFT", y_band_lo=120, y_band_hi=170),
    ScreenHop(0x3C, "UP", align_x=112),
)
LEVEL2_DOOR_SCREENS: tuple[int, ...] = path_screens_from_hops(
    0x37, LEVEL2_DOOR_HOPS
)
# 0x5C maze waypoints (pixel) from BFS: east on y≈88 to x≈184, then down/east.
# Wired into OverworldPathController / L2+L8 door paths (rr-gfx).
LEVEL2_5C_MAZE_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (20, 92),
    (40, 92),
    (56, 92),
    (80, 92),
    (104, 92),
    (120, 92),
    (144, 92),
    (168, 92),
    (184, 92),
    (192, 84),
    (192, 108),
    (192, 132),
    (200, 132),
    (224, 132),
    (240, 132),
)
# Screen that owns the maze hop (0x5C → 0x5D east).
SCREEN_5C_MAZE = 0x5C
MAZE_HOP_TARGET = 0x5D
MAZE_WAYPOINT_TOL = 6


def is_5c_maze_hop(hop: ScreenHop) -> bool:
    """True for the 0x5C→0x5D east hop that requires maze waypoints."""
    return hop.target == MAZE_HOP_TARGET and hop.direction == "RIGHT"


def _n(node_id: str, name: str, area: str, *tags: str, **meta: object) -> GraphNode:
    return GraphNode(
        node_id=node_id, name=name, area=area, tags=frozenset(tags), meta=meta
    )


def _e(
    source_id: str,
    target_id: str,
    direction: str,
    *,
    edge_id: str = "",
    requires: frozenset[str] | None = None,
    verification: str = "observed",
    provenance: str = "emulator_probe",
    **meta: object,
) -> GraphEdge:
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_id=edge_id,
        direction=direction,
        requires=requires or frozenset(),
        verification=verification,
        provenance=provenance,
        meta=meta,
    )


def _copy_edge(
    edge: GraphEdge, *, verification: str, provenance: str, **extra: object
) -> GraphEdge:
    return GraphEdge(
        source_id=edge.source_id,
        target_id=edge.target_id,
        edge_id=edge.edge_id,
        direction=edge.direction,
        requires=edge.requires,
        cost=edge.cost,
        verification=verification,
        provenance=provenance,
        meta={**dict(edge.meta), **extra},
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
    0x59: "bush_east_of_58",
    0x49: "north_of_59",
    0x4A: "east_of_49",
    0x4B: "east_of_4a",
    0x5A: "door_path_west_of_5b",
    0x5B: "bush_corridor_to_5c",
    0x5C: "maze_before_5d",
    0x5D: "north_exit_to_4d",
    0x4D: "north_of_5d",
    0x4C: "west_of_4d_pre_moon",
    0x3C: "level2_entrance",
    0x79: "rocky_deadend_east_of_78",
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

    Level 2 door-path screens (0x37→…→0x3C via 0x5A/5C maze) are seeded as
    **planned** hop edges (geometry probe-mapped; Clean transit not verified).
    Shared prefix hops with the 0x4A walk remain ``observed``.
    """
    seed_screens = (
        set(LEVEL1_PATH_SCREENS)
        | set(LEVEL2_PATH_SCREENS)
        | set(LEVEL2_DOOR_SCREENS)
    )
    seed_screens.update(
        {
            SCREEN_NORTH_OF_START,  # 0x67 dead-end (documented trap)
            0x76,
            0x66,
            0x47,
            0x57,
            0x79,  # rocky dead-end trap east of 0x78
            0x4B,  # partial probe past prefix / north-entry trap into 0x5B
        }
    )
    # Neighborhood around verified + door paths for expansion
    for sc in (
        list(LEVEL1_PATH_SCREENS)
        + list(LEVEL2_PATH_SCREENS)
        + list(LEVEL2_DOOR_SCREENS)
    ):
        for neighbor in neighbor_screens(sc).values():
            if neighbor is not None:
                seed_screens.add(neighbor)

    graph = build_overworld_grid_graph(screens=sorted(seed_screens))

    # Promote path edges: observed L1 / L2-prefix first; door-only hops stay planned.
    verified_l1 = {
        (a, b) for a, b in zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:])
    }
    verified_l2 = {
        (a, b) for a, b in zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:])
    }
    # Forward door-path hops (geometry probe-mapped; not Clean continuous).
    door_path_hops = {
        (a, b) for a, b in zip(LEVEL2_DOOR_SCREENS, LEVEL2_DOOR_SCREENS[1:])
    }
    promoted: list[GraphEdge] = []
    for edge in graph.edges:
        pair = (edge.meta.get("from_screen"), edge.meta.get("to_screen"))
        if pair in verified_l1:
            promoted.append(
                _copy_edge(
                    edge,
                    verification="observed",
                    provenance="emulator_probe",
                    segment="to_level1",
                )
            )
        elif pair in verified_l2:
            promoted.append(
                _copy_edge(
                    edge,
                    verification="observed",
                    provenance="emulator_probe",
                    segment="to_level2_prefix",
                )
            )
        elif pair in door_path_hops:
            extra: dict[str, object] = {
                "segment": "to_level2_door",
                "note": "geometry_probe_mapped; clean_health_not_verified",
            }
            if pair in ((0x5B, 0x5C), (0x5C, 0x5D)):
                extra["maze"] = "0x5c_waypoints"
            if pair == (0x5C, 0x5D):
                extra["requires_maze"] = True
            if pair == (0x5D, 0x4D):
                extra["align_x"] = 52
            promoted.append(
                _copy_edge(
                    edge, verification="planned", provenance="probe_geometry", **extra
                )
            )
        else:
            promoted.append(edge)

    extra_nodes = [
        _n(NODE_SWORD_CAVE, "wooden_sword_cave", "cave", "cave", "item", "sword",
           overworld_screen=SCREEN_START, item="wooden_sword"),
        _n(NODE_LEVEL1_DUNGEON, "level1_eagle", "dungeon", "dungeon", "level1",
           overworld_screen=SCREEN_LEVEL1, level=1),
        _n(NODE_LEVEL1_ENTRY_ROOM, "level1_entrance_room", "level1",
           "dungeon", "level1", "room", level=1, room=0x73),
        _n(NODE_LEVEL1_FIRST_KEY_ROOM, "level1_first_key_room", "level1",
           "dungeon", "level1", "room", "combat", level=1, room=0x74, enemies="stalfos"),
        _n(NODE_LEVEL1_FIRST_KEY, "level1_first_key", "level1",
           "dungeon", "level1", "item", "key", level=1, room=0x74, item="small_key"),
        _n(NODE_LEVEL1_NORTH_ROOM, "level1_north_stalfos_room", "level1",
           "dungeon", "level1", "room", "combat",
           level=1, room=0x63, enemies="3_stalfos"),
        _n(NODE_LEVEL1_NORTH_CLEARED, "level1_north_stalfos_cleared", "level1",
           "dungeon", "level1", "room", "cleared",
           level=1, room=0x63, reward="none", doors={"south": 0x73, "north": 0x53}),
        _n(NODE_LEVEL1_ROOM_53, "level1_room_53_five_stalfos", "level1",
           "dungeon", "level1", "room", "combat",
           level=1, room=0x53, enemies="5_stalfos", item_id=0x19),
        _n(NODE_LEVEL1_ROOM_53_CLEARED, "level1_room_53_cleared_key_collected",
           "level1", "dungeon", "level1", "room", "cleared", "key",
           level=1, room=0x53, reward="small_key",
           doors={"south": 0x63, "west": 0x52, "east": 0x54}),
        _n(NODE_LEVEL1_ROOM_52, "level1_room_52_six_keese", "level1",
           "dungeon", "level1", "room", "combat",
           level=1, room=0x52, enemies="6_keese", item_id=0x03),
        _n(NODE_LEVEL1_ROOM_54, "level1_room_54_eight_keese", "level1",
           "dungeon", "level1", "room", "combat",
           level=1, room=0x54, enemies="8_keese", item_id=0x16),
        _n(NODE_LEVEL1_ROOM_54_CLEARED, "level1_room_54_eight_keese_cleared",
           "level1", "dungeon", "level1", "room", "cleared",
           level=1, room=0x54, reward="no_known_inventory_change", item_id=0x16,
           doors={"west": 0x53, "east": "blocked"}),
        _n(NODE_LEVEL1_COMPLETE, "level1_triforce_shard_1", "level1",
           "dungeon", "level1", "boss", "triforce", "complete",
           level=1, room=0x36, boss="aquamentus", reward="triforce_shard_1",
           stop_predicate="triforce & 0x01"),
        _n(NODE_LEVEL1_EXIT_OVERWORLD, "post_triforce_level1_mouth", "overworld",
           "overworld", "post_triforce", screen=0x37,
           note="engine returns here after shard-1 fanfare; then walk"),
        _n(NODE_LEVEL2_PATH_4A, "level2_path_4a", "overworld",
           "overworld", "level2_path", screen=0x4A, segment="to_level2_prefix",
           stop_predicate="level2_path_prefix_success"),
        _n(NODE_LEVEL2_ENTRANCE, "level2_overworld_door", "overworld",
           "overworld", "level2", "door", screen=0x3C, segment="to_level2_door",
           source="probe_geometry_plus_walkthrough", verification="planned",
           note="door_screen_reached_in_probe_states; clean_walk_not_verified"),
        _n(NODE_LEVEL2_DUNGEON, "level2_moon", "dungeon", "dungeon", "level2",
           overworld_screen=0x3C, level=2, entry_room=0x7D, verification="planned"),
    ]
    sword = frozenset({"wooden_sword"})
    sword_tf = frozenset({"wooden_sword", "triforce_shard_1"})
    extra_edges = [
        _e(NODE_START, NODE_SWORD_CAVE, "IN", edge_id="enter_sword_cave",
           segment="sword_cave"),
        _e(NODE_SWORD_CAVE, NODE_START, "OUT", edge_id="exit_sword_cave",
           segment="sword_cave", acquires_on_clear=["wooden_sword"]),
        _e(NODE_LEVEL1_ENTRANCE, NODE_LEVEL1_DUNGEON, "IN", edge_id="enter_level1",
           requires=sword, segment="to_level1", door="tree_mouth",
           approach=(112, 140)),
        _e(NODE_LEVEL1_DUNGEON, NODE_LEVEL1_ENTRY_ROOM, "IN",
           edge_id="settle_level1_entrance", segment="level1_first_key", room=0x73),
        _e(NODE_LEVEL1_ENTRY_ROOM, NODE_LEVEL1_FIRST_KEY_ROOM, "RIGHT",
           edge_id="enter_level1_first_key_room", segment="level1_first_key",
           from_room=0x73, to_room=0x74),
        _e(NODE_LEVEL1_FIRST_KEY_ROOM, NODE_LEVEL1_FIRST_KEY, "ITEM",
           edge_id="collect_level1_first_key", segment="level1_first_key",
           item_id=0x19),
        _e(NODE_LEVEL1_FIRST_KEY, NODE_LEVEL1_FIRST_KEY_ROOM, "ROOM",
           edge_id="resume_after_level1_first_key", segment="level1_north",
           room=0x74),
        _e(NODE_LEVEL1_FIRST_KEY_ROOM, NODE_LEVEL1_ENTRY_ROOM, "LEFT",
           edge_id="return_to_level1_entrance", segment="level1_north",
           from_room=0x74, to_room=0x73),
        _e(NODE_LEVEL1_ENTRY_ROOM, NODE_LEVEL1_NORTH_ROOM, "UP",
           edge_id="unlock_level1_north", requires=frozenset({"keys"}),
           segment="level1_north", from_room=0x73, to_room=0x63,
           consumes={"keys": 1}),
        _e(NODE_LEVEL1_NORTH_ROOM, NODE_LEVEL1_NORTH_CLEARED, "COMBAT",
           edge_id="clear_level1_room_63", segment="level1_clear63", room=0x63,
           enemies="3_stalfos", reward="none"),
        _e(NODE_LEVEL1_NORTH_CLEARED, NODE_LEVEL1_ROOM_53, "UP",
           edge_id="enter_level1_room_53", segment="level1_clear63",
           from_room=0x63, to_room=0x53,
           note="north door open without clear; clear preferred"),
        _e(NODE_LEVEL1_ROOM_53, NODE_LEVEL1_ROOM_53_CLEARED, "COMBAT",
           edge_id="clear_level1_room_53", segment="level1_clear53", room=0x53,
           enemies="5_stalfos", reward="small_key", item_id=0x19,
           item_position=(128, 109)),
        _e(NODE_LEVEL1_ROOM_53_CLEARED, NODE_LEVEL1_ROOM_52, "LEFT",
           edge_id="enter_level1_room_52", segment="level1_clear53",
           from_room=0x53, to_room=0x52, enemies="6_keese"),
        _e(NODE_LEVEL1_ROOM_53_CLEARED, NODE_LEVEL1_ROOM_54, "RIGHT",
           edge_id="enter_level1_room_54", segment="level1_clear53",
           from_room=0x53, to_room=0x54, enemies="8_keese", item_id=0x16),
        _e(NODE_LEVEL1_ROOM_54, NODE_LEVEL1_ROOM_54_CLEARED, "COMBAT",
           edge_id="clear_level1_room_54", segment="level1_clear54", room=0x54,
           enemies="8_keese", reward="no_known_inventory_change", item_id=0x16),
        _e(NODE_LEVEL1_ROOM_53_CLEARED, NODE_LEVEL1_COMPLETE, "ROUTE",
           edge_id="complete_level1_eagle", requires=sword,
           segment="level1_complete",
           rooms=(0x52, 0x42, 0x41, 0x43, 0x33, 0x23, 0x44, 0x45, 0x35, 0x36),
           source="walkthrough_correlated_then_live_verified"),
        _e(NODE_LEVEL1_COMPLETE, NODE_LEVEL1_EXIT_OVERWORLD, "OUT",
           edge_id="settle_post_triforce_overworld", segment="to_level2",
           note="idle through mode-18 fanfare; engine places Link on 0x37",
           approx_frames=704),
        _e(NODE_LEVEL1_EXIT_OVERWORLD, NODE_LEVEL2_PATH_4A, "ROUTE",
           edge_id="walk_level2_path_prefix", requires=sword_tf,
           segment="to_level2_prefix", screens=LEVEL2_PATH_SCREENS, avoid=(0x79,)),
        _e(NODE_LEVEL1_EXIT_OVERWORLD, NODE_LEVEL2_ENTRANCE, "ROUTE",
           edge_id="walk_level2_door_path", requires=sword_tf,
           verification="planned", provenance="probe_geometry",
           segment="to_level2_door", screens=LEVEL2_DOOR_SCREENS,
           hops="LEVEL2_DOOR_HOPS", maze="0x5c_waypoints",
           blocker="overworld_health_management", avoid=(0x79, 0x4B),
           note="geometry probe-mapped via 0x5A west entry; "
                "Clean dies on 0x5C with 0 hearts; not 2/2 continuous"),
        _e(NODE_LEVEL2_PATH_4A, NODE_LEVEL2_ENTRANCE, "ROUTE",
           edge_id="walk_level2_suffix", requires=sword_tf,
           verification="planned", provenance="probe_geometry",
           segment="to_level2_door", planned_screens=LEVEL2_DOOR_SCREENS,
           from_prefix=0x4A, blocker="overworld_health_management",
           maze="0x5c_waypoints",
           note="continuation from verified prefix; door hops still planned"),
        _e(NODE_LEVEL2_ENTRANCE, NODE_LEVEL2_DUNGEON, "IN", edge_id="enter_level2",
           requires=sword_tf, verification="planned", provenance="probe_geometry",
           segment="to_level2_door", door="moon_mouth", entry_room=0x7D,
           note="dev fixtures Level2Entrance/Level2EntryFresh; not Clean natural"),
    ]
    nodes = list(graph.nodes.values()) + extra_nodes
    edges = list(promoted) + extra_edges
    return RouteGraph(nodes, edges)

