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
# Wired into OverworldToLevel2Controller for the 0x5C→0x5D hop (rr-gfx).
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
        elif pair in verified_l2:
            # Shared with door path (0x37→…→0x59); Clean prefix wins over planned door.
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
                    meta={**dict(edge.meta), "segment": "to_level2_prefix"},
                )
            )
        elif pair in door_path_hops:
            # Geometry probe-mapped (incl. death-on-0x5C Clean timing); not 2/2 Clean.
            hop_meta = {
                **dict(edge.meta),
                "segment": "to_level2_door",
                "note": "geometry_probe_mapped; clean_health_not_verified",
            }
            if pair == (0x5B, 0x5C):
                hop_meta["maze"] = "0x5c_waypoints"
            if pair == (0x5C, 0x5D):
                hop_meta["maze"] = "0x5c_waypoints"
                hop_meta["requires_maze"] = True
            if pair == (0x5D, 0x4D):
                hop_meta["align_x"] = 52
            promoted.append(
                GraphEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_id=edge.edge_id,
                    direction=edge.direction,
                    requires=edge.requires,
                    cost=edge.cost,
                    verification="planned",
                    provenance="probe_geometry",
                    meta=hop_meta,
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
        GraphNode(
            node_id=NODE_LEVEL1_COMPLETE,
            name="level1_triforce_shard_1",
            area="level1",
            tags=frozenset(
                {"dungeon", "level1", "boss", "triforce", "complete"}
            ),
            meta={
                "level": 1,
                "room": 0x36,
                "boss": "aquamentus",
                "reward": "triforce_shard_1",
                "stop_predicate": "triforce & 0x01",
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL1_EXIT_OVERWORLD,
            name="post_triforce_level1_mouth",
            area="overworld",
            tags=frozenset({"overworld", "post_triforce"}),
            meta={
                "screen": 0x37,
                "note": "engine returns here after shard-1 fanfare; then walk",
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL2_PATH_4A,
            name="level2_path_4a",
            area="overworld",
            tags=frozenset({"overworld", "level2_path"}),
            meta={
                "screen": 0x4A,
                "segment": "to_level2_prefix",
                "stop_predicate": "level2_path_prefix_success",
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL2_ENTRANCE,
            name="level2_overworld_door",
            area="overworld",
            tags=frozenset({"overworld", "level2", "door"}),
            meta={
                "screen": 0x3C,
                "segment": "to_level2_door",
                "source": "probe_geometry_plus_walkthrough",
                "verification": "planned",
                "note": "door_screen_reached_in_probe_states; clean_walk_not_verified",
            },
        ),
        GraphNode(
            node_id=NODE_LEVEL2_DUNGEON,
            name="level2_moon",
            area="dungeon",
            tags=frozenset({"dungeon", "level2"}),
            meta={
                "overworld_screen": 0x3C,
                "level": 2,
                "entry_room": 0x7D,
                "verification": "planned",
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
        GraphEdge(
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_COMPLETE,
            edge_id="complete_level1_eagle",
            direction="ROUTE",
            requires=frozenset({"wooden_sword"}),
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "level1_complete",
                "rooms": (
                    0x52,
                    0x42,
                    0x41,
                    0x43,
                    0x33,
                    0x23,
                    0x44,
                    0x45,
                    0x35,
                    0x36,
                ),
                "source": "walkthrough_correlated_then_live_verified",
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_COMPLETE,
            target_id=NODE_LEVEL1_EXIT_OVERWORLD,
            edge_id="settle_post_triforce_overworld",
            direction="OUT",
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "to_level2",
                "note": "idle through mode-18 fanfare; engine places Link on 0x37",
                "approx_frames": 704,
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_EXIT_OVERWORLD,
            target_id=NODE_LEVEL2_PATH_4A,
            edge_id="walk_level2_path_prefix",
            direction="ROUTE",
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            verification="observed",
            provenance="emulator_probe",
            meta={
                "segment": "to_level2_prefix",
                "screens": LEVEL2_PATH_SCREENS,
                "avoid": (0x79,),
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL1_EXIT_OVERWORLD,
            target_id=NODE_LEVEL2_ENTRANCE,
            edge_id="walk_level2_door_path",
            direction="ROUTE",
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            verification="planned",
            provenance="probe_geometry",
            meta={
                "segment": "to_level2_door",
                "screens": LEVEL2_DOOR_SCREENS,
                "hops": "LEVEL2_DOOR_HOPS",
                "maze": "0x5c_waypoints",
                "blocker": "overworld_health_management",
                "avoid": (0x79, 0x4B),
                "note": (
                    "geometry probe-mapped via 0x5A west entry; "
                    "Clean dies on 0x5C with 0 hearts; not 2/2 continuous"
                ),
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL2_PATH_4A,
            target_id=NODE_LEVEL2_ENTRANCE,
            edge_id="walk_level2_suffix",
            direction="ROUTE",
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            verification="planned",
            provenance="probe_geometry",
            meta={
                "segment": "to_level2_door",
                "planned_screens": LEVEL2_DOOR_SCREENS,
                "from_prefix": 0x4A,
                "blocker": "overworld_health_management",
                "maze": "0x5c_waypoints",
                "note": "continuation from verified prefix; door hops still planned",
            },
        ),
        GraphEdge(
            source_id=NODE_LEVEL2_ENTRANCE,
            target_id=NODE_LEVEL2_DUNGEON,
            edge_id="enter_level2",
            direction="IN",
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            verification="planned",
            provenance="probe_geometry",
            meta={
                "segment": "to_level2_door",
                "door": "moon_mouth",
                "entry_room": 0x7D,
                "note": "dev fixtures Level2Entrance/Level2EntryFresh; not Clean natural",
            },
        ),
    ]
    nodes = list(graph.nodes.values()) + extra_nodes
    edges = list(promoted) + extra_edges
    return RouteGraph(nodes, edges)

