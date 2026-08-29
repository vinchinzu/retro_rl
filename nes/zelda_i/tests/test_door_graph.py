"""Pure unit tests for dungeon door-graph pathfinding (no emulator)."""

from __future__ import annotations

from zelda_i.door_graph import (
    L2_BOMB_N,
    L2_COMPASS,
    L2_ENTRY,
    L3_BOMB_SHORTCUT,
    L3_COMPASS,
    L3_DARKNUTS,
    L3_RAFT_PASSAGE,
    L3_SOUTH_DARKNUTS,
    L3_WEST_DARKNUTS,
    LEVEL_2_DOOR_GRAPH,
    LEVEL_3_DOOR_GRAPH,
    LEVEL_4_DOOR_GRAPH,
    LEVEL_5_DOOR_GRAPH,
    L4_ENTRY,
    L4_TRIFORCE,
    L5_ENTRY,
    L5_TRIFORCE,
    L5_WHISTLE_ITEM,
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    InventoryCaps,
    RoomExit,
    dirs_from_mask,
)
from zelda_i.level2_puzzles import BOMB_WALL_6F_NORTH


def _toy_graph() -> DungeonDoorGraph:
    return DungeonDoorGraph.from_exits(
        {
            0x10: (
                RoomExit(DoorDir.RIGHT, 0x11, GateKind.OPEN),
                RoomExit(DoorDir.UP, 0x12, GateKind.KEY, key_cost=1),
                RoomExit(DoorDir.LEFT, None, GateKind.SEALED),
            ),
            0x11: (
                RoomExit(DoorDir.LEFT, 0x10, GateKind.OPEN),
                RoomExit(DoorDir.RIGHT, 0x13, GateKind.BOMB, bomb_stand=(64, 100)),
                RoomExit(DoorDir.UP, 0x14, GateKind.KILL_CLEAR),
            ),
            0x12: (RoomExit(DoorDir.DOWN, 0x10, GateKind.OPEN),),
            0x13: (RoomExit(DoorDir.LEFT, 0x11, GateKind.OPEN),),
            0x14: (RoomExit(DoorDir.DOWN, 0x11, GateKind.OPEN),),
        },
        level=99,
        name="toy",
    )


def test_dirs_from_mask_and_sealed_not_pathfinding() -> None:
    assert dirs_from_mask(0x02) == frozenset({DoorDir.LEFT})
    assert dirs_from_mask(0x04) == frozenset({DoorDir.DOWN})
    assert dirs_from_mask(0x0F) == frozenset(DoorDir)
    e = RoomExit(DoorDir.LEFT, None, GateKind.SEALED)
    assert not e.is_pathfinding


def test_bfs_path_consumes_key_and_reachable_with_bomb_and_kill() -> None:
    g = _toy_graph()
    path = g.bfs_path(0x10, 0x12, InventoryCaps(keys=1))
    assert path is not None
    assert len(path) == 1
    assert path[0].gate is GateKind.KEY
    assert path[0].target_room == 0x12
    assert g.bfs_path(0x10, 0x12, InventoryCaps(keys=0)) is None

    reached = g.bfs_reachable(
        0x10, InventoryCaps(keys=0, bombs=1, can_clear=True)
    )
    assert reached == frozenset({0x10, 0x11, 0x13, 0x14})


def test_level2_bomb_north_6f_stand_matches_puzzles() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(L2_COMPASS, L2_BOMB_N, direction=DoorDir.UP)
    assert e is not None
    assert e.gate is GateKind.BOMB
    assert e.bomb_stand == BOMB_WALL_6F_NORTH.stand == (120, 101)


def test_level2_path_blocked_without_bomb() -> None:
    assert (
        LEVEL_2_DOOR_GRAPH.bfs_path(
            L2_ENTRY, L2_BOMB_N, InventoryCaps(keys=2, bombs=0, can_clear=True)
        )
        is None
    )


def test_level3_raft_bfs_5b_to_passage_needs_key() -> None:
    caps = InventoryCaps(keys=1, bombs=0, can_clear=True)
    path = LEVEL_3_DOOR_GRAPH.bfs_path(L3_DARKNUTS, L3_RAFT_PASSAGE, caps)
    assert path is not None
    rooms = [L3_DARKNUTS, *[e.target_room for e in path]]
    assert rooms[-1] == L3_RAFT_PASSAGE == 0x0F
    assert L3_COMPASS in rooms
    pairs = list(zip(rooms, rooms[1:]))
    assert (L3_COMPASS, L3_WEST_DARKNUTS) in pairs
    assert (L3_WEST_DARKNUTS, L3_SOUTH_DARKNUTS) in pairs
    assert (L3_SOUTH_DARKNUTS, L3_RAFT_PASSAGE) in pairs
    assert any(
        e.gate is GateKind.KEY and e.target_room == L3_WEST_DARKNUTS for e in path
    )
    assert all(e.gate is not GateKind.BOMB for e in path)
    assert L3_BOMB_SHORTCUT not in LEVEL_3_DOOR_GRAPH.bfs_reachable(L3_DARKNUTS, caps)

    no_key = InventoryCaps(keys=0, bombs=0, can_clear=True)
    assert LEVEL_3_DOOR_GRAPH.bfs_path(L3_DARKNUTS, L3_RAFT_PASSAGE, no_key) is None


def test_level4_bfs_entry_reaches_tf() -> None:
    caps = InventoryCaps(keys=2, bombs=2, can_clear=True)
    reached = LEVEL_4_DOOR_GRAPH.bfs_reachable(L4_ENTRY, caps)
    assert L4_TRIFORCE in reached
    path = LEVEL_4_DOOR_GRAPH.bfs_path(L4_ENTRY, L4_TRIFORCE, caps)
    assert path is not None
    assert path[-1].target_room == L4_TRIFORCE


def test_level5_bfs_entry_reaches_tf_or_whistle_with_bombs() -> None:
    tf_caps = InventoryCaps(keys=2, bombs=2, can_clear=True)
    reached = LEVEL_5_DOOR_GRAPH.bfs_reachable(L5_ENTRY, tf_caps)
    assert L5_TRIFORCE in reached
    path = LEVEL_5_DOOR_GRAPH.bfs_path(L5_ENTRY, L5_TRIFORCE, tf_caps)
    assert path is not None
    assert path[-1].target_room == L5_TRIFORCE

    whistle_caps = InventoryCaps(keys=3, bombs=2, can_clear=True)
    assert L5_WHISTLE_ITEM in LEVEL_5_DOOR_GRAPH.bfs_reachable(
        L5_ENTRY, whistle_caps
    )
