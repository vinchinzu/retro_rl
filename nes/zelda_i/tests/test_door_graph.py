"""Pure unit tests for dungeon door-graph pathfinding (no emulator)."""

from __future__ import annotations

import pytest

from zelda_i.door_graph import (
    L2_BOMB_N,
    L2_BOOM,
    L2_COMPASS,
    L2_DODONGO,
    L2_EAST_KEY,
    L2_EAST_OF_ROPES,
    L2_ENTRY,
    L2_GORIYA_BOMBS,
    L2_GORIYA_WEST,
    L2_MOLDORM,
    L2_ROPES,
    L2_ROPES_NORTH,
    L2_ROPES_UNLOCK,
    L2_TRAPS_KEESE,
    L2_WEST_KEY,
    L2_WEST_OF_BOSS,
    L3_BOMB_SHORTCUT,
    L3_COMPASS,
    L3_DARKNUTS,
    L3_ENTRY,
    L3_RAFT_PASSAGE,
    L3_SOUTH_DARKNUTS,
    L3_WEST_DARKNUTS,
    L3_ZOL_KEY_4B,
    LEVEL_2_DOOR_GRAPH,
    LEVEL_3_DOOR_GRAPH,
    LEVEL_4_DOOR_GRAPH,
    LEVEL_5_DOOR_GRAPH,
    LEVEL_9_DOOR_GRAPH,
    L4_ENTRY,
    L4_STEPLADDER,
    L4_TRIFORCE,
    L5_ENTRY,
    L5_TRIFORCE,
    L5_WHISTLE_ITEM,
    L9_ROOM_41,
    L9_ZELDA,
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    InventoryCaps,
    RoomExit,
    dirs_from_mask,
    door_dir_from_label,
    level_2_door_graph,
    level_3_door_graph,
    level_4_door_graph,
    level_5_door_graph,
    level_9_door_graph,
)


# ---------------------------------------------------------------------------
# Enums / helpers
# ---------------------------------------------------------------------------


def test_door_dir_bits_match_ram_layout() -> None:
    assert DoorDir.RIGHT == 0x01
    assert DoorDir.LEFT == 0x02
    assert DoorDir.DOWN == 0x04
    assert DoorDir.UP == 0x08
    assert DoorDir.LEFT.opposite is DoorDir.RIGHT
    assert DoorDir.UP.opposite is DoorDir.DOWN


def test_dirs_from_mask() -> None:
    # After 0x6d clear: LEFT bit only.
    assert dirs_from_mask(0x02) == frozenset({DoorDir.LEFT})
    # Bomb entry 0x5f often only DOWN.
    assert dirs_from_mask(0x04) == frozenset({DoorDir.DOWN})
    assert dirs_from_mask(0x0F) == frozenset(DoorDir)


def test_door_dir_from_label() -> None:
    assert door_dir_from_label("LEFT") is DoorDir.LEFT
    assert door_dir_from_label("r") is DoorDir.RIGHT
    with pytest.raises(ValueError):
        door_dir_from_label("north-east")


def test_room_exit_key_cost_defaults() -> None:
    e = RoomExit(DoorDir.RIGHT, 0x6F, GateKind.KEY)
    assert e.key_cost == 1
    open_e = RoomExit(DoorDir.UP, 0x6D, GateKind.OPEN, key_cost=3)
    assert open_e.key_cost == 0  # non-KEY strips cost


def test_sealed_not_pathfinding() -> None:
    e = RoomExit(DoorDir.LEFT, None, GateKind.SEALED)
    assert not e.is_pathfinding


# ---------------------------------------------------------------------------
# Small synthetic graph
# ---------------------------------------------------------------------------


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


def test_edges_from_and_pathfinding_filter() -> None:
    g = _toy_graph()
    all_exits = g.edges_from(0x10)
    assert len(all_exits) == 3
    pf = g.pathfinding_edges_from(0x10)
    assert len(pf) == 2
    assert all(e.gate is not GateKind.SEALED for e in pf)


def test_bfs_reachable_open_only() -> None:
    g = _toy_graph()
    # No keys/bombs, can_clear False → only OPEN chain 0x10↔0x11.
    reached = g.bfs_reachable(0x10, InventoryCaps(keys=0, bombs=0, can_clear=False))
    assert reached == frozenset({0x10, 0x11})


def test_bfs_reachable_with_key() -> None:
    g = _toy_graph()
    reached = g.bfs_reachable(0x10, InventoryCaps(keys=1, bombs=0, can_clear=False))
    assert 0x12 in reached
    assert 0x13 not in reached


def test_bfs_reachable_bomb_and_kill() -> None:
    g = _toy_graph()
    reached = g.bfs_reachable(
        0x10, InventoryCaps(keys=0, bombs=1, can_clear=True)
    )
    assert reached == frozenset({0x10, 0x11, 0x13, 0x14})


def test_bfs_path_consumes_key() -> None:
    g = _toy_graph()
    path = g.bfs_path(0x10, 0x12, InventoryCaps(keys=1))
    assert path is not None
    assert len(path) == 1
    assert path[0].gate is GateKind.KEY
    assert path[0].target_room == 0x12
    # Zero keys: unreachable.
    assert g.bfs_path(0x10, 0x12, InventoryCaps(keys=0)) is None


def test_inventory_caps_from_mapping() -> None:
    caps = InventoryCaps.from_mapping({"keys": 2, "bombs": 3, "can_clear": False})
    assert caps == InventoryCaps(keys=2, bombs=3, can_clear=False)


# ---------------------------------------------------------------------------
# Level 2 seed vs LEVEL2_ROUTE.md
# ---------------------------------------------------------------------------

_L2_CORE_ROOMS = frozenset(
    {
        L2_ENTRY,
        L2_ROPES,
        L2_WEST_KEY,
        L2_EAST_KEY,
        L2_EAST_OF_ROPES,
        L2_COMPASS,
        L2_BOMB_N,
        L2_GORIYA_WEST,
        L2_ROPES_NORTH,
        L2_BOOM,
        L2_TRAPS_KEESE,
        L2_MOLDORM,
        L2_ROPES_UNLOCK,
        L2_GORIYA_BOMBS,
        L2_DODONGO,
        L2_WEST_OF_BOSS,
    }
)


def test_level2_seed_has_required_rooms() -> None:
    g = LEVEL_2_DOOR_GRAPH
    assert g.level == 2
    assert g.room_ids() == _L2_CORE_ROOMS


def test_level2_entry_geometry() -> None:
    g = LEVEL_2_DOOR_GRAPH
    up = g.exit_between(L2_ENTRY, L2_ROPES, direction=DoorDir.UP)
    assert up is not None
    assert up.gate is GateKind.OPEN
    east = g.exit_between(L2_ENTRY, L2_EAST_KEY, direction=DoorDir.RIGHT)
    assert east is not None
    assert east.approach_xy == (208, 141)
    # West sealed (no pathfinding target).
    sealed = [e for e in g.edges_from(L2_ENTRY) if e.direction is DoorDir.LEFT]
    assert len(sealed) == 1 and sealed[0].gate is GateKind.SEALED


def test_level2_ropes_kill_clear_left() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(L2_ROPES, L2_WEST_KEY, direction=DoorDir.LEFT)
    assert e is not None
    assert e.gate is GateKind.KILL_CLEAR
    assert e.approach_xy == (48, 141)


def test_level2_key_door_6e_to_6f() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(
        L2_EAST_OF_ROPES, L2_COMPASS, direction=DoorDir.RIGHT
    )
    assert e is not None
    assert e.gate is GateKind.KEY
    assert e.key_cost == 1


def test_level2_bomb_north_6f() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(L2_COMPASS, L2_BOMB_N, direction=DoorDir.UP)
    assert e is not None
    assert e.gate is GateKind.BOMB
    assert e.bomb_stand == (120, 101)


def test_level2_5f_key_left_to_goriya() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(
        L2_BOMB_N, L2_GORIYA_WEST, direction=DoorDir.LEFT
    )
    assert e is not None
    assert e.gate is GateKind.KEY


def test_level2_5f_bomb_north_to_boom() -> None:
    e = LEVEL_2_DOOR_GRAPH.exit_between(L2_BOMB_N, L2_BOOM, direction=DoorDir.UP)
    assert e is not None
    assert e.gate is GateKind.BOMB
    assert e.bomb_stand == (120, 101)


def test_level2_1e_bomb_north_to_dodongo() -> None:
    """rr-n5i: walk-UP solid; bomb-N @(120,101) → boss 0x0e."""
    e = LEVEL_2_DOOR_GRAPH.exit_between(
        L2_GORIYA_BOMBS, L2_DODONGO, direction=DoorDir.UP
    )
    assert e is not None
    assert e.gate is GateKind.BOMB
    assert e.bomb_stand == (120, 101)
    boom_n = LEVEL_2_DOOR_GRAPH.exit_between(
        L2_BOOM, L2_TRAPS_KEESE, direction=DoorDir.UP
    )
    assert boom_n is not None and boom_n.gate is GateKind.BOMB


def test_level2_reachable_without_resources() -> None:
    """Open subgraph from entry: 7d, 6d, 7e, 6e — not west key (kill) or past key door."""
    g = LEVEL_2_DOOR_GRAPH
    reached = g.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=0, bombs=0, can_clear=False)
    )
    assert L2_ENTRY in reached
    assert L2_ROPES in reached
    assert L2_EAST_KEY in reached
    assert L2_EAST_OF_ROPES in reached
    assert L2_WEST_KEY not in reached
    assert L2_COMPASS not in reached
    assert L2_BOMB_N not in reached


def test_level2_reachable_clear_opens_west_key() -> None:
    reached = LEVEL_2_DOOR_GRAPH.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=0, bombs=0, can_clear=True)
    )
    assert L2_WEST_KEY in reached
    assert L2_COMPASS not in reached


def test_level2_one_key_reaches_compass_not_goriya() -> None:
    # One key: 6e→6f, but 5f→5e also needs a key after bomb.
    reached = LEVEL_2_DOOR_GRAPH.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=1, bombs=0, can_clear=True)
    )
    assert L2_COMPASS in reached
    assert L2_BOMB_N not in reached  # needs bomb
    assert L2_GORIYA_WEST not in reached


def test_level2_two_keys_and_bomb_reach_goriya() -> None:
    """Route doc: carry ≥2 keys into 0x6e; bomb N; another key for 0x5f LEFT.

    With keys=2 bombs=1: reach Goriya + 0x4e (free UP → Moldorm). Boom 0x4f is
    also reachable via 0x4e→0x3e→0x3f→DOWN (open hole) without a second bomb.
    """
    reached = LEVEL_2_DOOR_GRAPH.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=2, bombs=1, can_clear=True)
    )
    assert {
        L2_ENTRY,
        L2_ROPES,
        L2_WEST_KEY,
        L2_EAST_KEY,
        L2_EAST_OF_ROPES,
        L2_COMPASS,
        L2_BOMB_N,
        L2_GORIYA_WEST,
        L2_ROPES_NORTH,
        L2_MOLDORM,  # free UP from 0x4e
        L2_BOOM,  # via 0x3e↔0x3f open graph after 0x4e
    } <= reached


def test_level2_boom_reachable_with_extra_bomb_or_key() -> None:
    """Magical Boomerang room: bomb-N from 0x5f or key-RIGHT from 0x4e."""
    via_bomb = LEVEL_2_DOOR_GRAPH.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=2, bombs=2, can_clear=True)
    )
    assert L2_BOOM in via_bomb
    via_key = LEVEL_2_DOOR_GRAPH.bfs_reachable(
        L2_ENTRY, InventoryCaps(keys=3, bombs=1, can_clear=True)
    )
    assert L2_BOOM in via_key


def test_level2_path_entry_to_goriya() -> None:
    path = LEVEL_2_DOOR_GRAPH.bfs_path(
        L2_ENTRY,
        L2_GORIYA_WEST,
        InventoryCaps(keys=2, bombs=1, can_clear=True),
    )
    assert path is not None
    rooms = [L2_ENTRY, *[e.target_room for e in path]]
    assert rooms[-1] == L2_GORIYA_WEST
    assert L2_COMPASS in rooms
    assert L2_BOMB_N in rooms
    gates = [e.gate for e in path]
    assert GateKind.KEY in gates
    assert GateKind.BOMB in gates
    # Exactly two key spends and one bomb.
    assert sum(1 for g in gates if g is GateKind.KEY) == 2
    assert sum(1 for g in gates if g is GateKind.BOMB) == 1


def test_level2_path_blocked_without_bomb() -> None:
    assert (
        LEVEL_2_DOOR_GRAPH.bfs_path(
            L2_ENTRY, L2_BOMB_N, InventoryCaps(keys=2, bombs=0, can_clear=True)
        )
        is None
    )


def test_level2_fresh_copy_independent() -> None:
    g = level_2_door_graph()
    assert g.room_ids() == LEVEL_2_DOOR_GRAPH.room_ids()
    # Mutating the copy must not touch the module seed.
    g.rooms[L2_ENTRY] = ()
    assert LEVEL_2_DOOR_GRAPH.edges_from(L2_ENTRY)


# ---------------------------------------------------------------------------
# Level 3 seed (Manji past-Darknuts LIVE 2026-08-07)
# ---------------------------------------------------------------------------


def test_level3_seed_core_rooms() -> None:
    g = LEVEL_3_DOOR_GRAPH
    assert g.level == 3
    for rid in (
        L3_ENTRY,
        L3_DARKNUTS,
        L3_ZOL_KEY_4B,
        L3_COMPASS,
        L3_SOUTH_DARKNUTS,
        L3_RAFT_PASSAGE,
    ):
        assert rid in g.room_ids()


def test_level3_darknuts_exits() -> None:
    g = LEVEL_3_DOOR_GRAPH
    up = g.exit_between(L3_DARKNUTS, L3_ZOL_KEY_4B, direction=DoorDir.UP)
    assert up is not None and up.gate is GateKind.OPEN
    left = g.exit_between(L3_DARKNUTS, L3_COMPASS, direction=DoorDir.LEFT)
    assert left is not None and left.gate is GateKind.OPEN
    bomb = [
        e
        for e in g.edges_from(L3_DARKNUTS)
        if e.gate is GateKind.BOMB
    ]
    assert len(bomb) == 1
    assert bomb[0].bomb_stand == (192, 141)


def test_level3_raft_path_with_key_and_clear() -> None:
    """Compass path needs 1 key + kill clears to reach stairs room."""
    path = LEVEL_3_DOOR_GRAPH.bfs_path(
        L3_DARKNUTS,
        L3_SOUTH_DARKNUTS,
        InventoryCaps(keys=1, bombs=0, can_clear=True),
    )
    assert path is not None
    rooms = [L3_DARKNUTS, *[e.target_room for e in path]]
    assert L3_COMPASS in rooms
    assert rooms[-1] == L3_SOUTH_DARKNUTS
    assert any(e.gate is GateKind.KEY for e in path)


def test_level3_raft_bfs_5b_to_passage() -> None:
    """Compass-west Raft: 0x5b → 0x0f with keys=1 bombs=0 can_clear.

    Bomb-R 0x5b→0x5c is the boss shortcut, not this path. Raft is room 0x0f
    (mode-9 passage), not a cardinal item pickup.
    """
    caps = InventoryCaps(keys=1, bombs=0, can_clear=True)
    path = LEVEL_3_DOOR_GRAPH.bfs_path(L3_DARKNUTS, L3_RAFT_PASSAGE, caps)
    assert path is not None
    rooms = [L3_DARKNUTS, *[e.target_room for e in path]]
    assert rooms[-1] == L3_RAFT_PASSAGE == 0x0F
    assert L3_COMPASS in rooms
    pairs = list(zip(rooms, rooms[1:]))
    assert (L3_COMPASS, L3_WEST_DARKNUTS) in pairs  # KEY-LEFT 0x5a → 0x59
    assert (L3_WEST_DARKNUTS, L3_SOUTH_DARKNUTS) in pairs  # KILL_CLEAR 0x59 → 0x69
    assert (L3_SOUTH_DARKNUTS, L3_RAFT_PASSAGE) in pairs  # KILL_CLEAR 0x69 → 0x0f
    assert any(
        e.gate is GateKind.KEY and e.target_room == L3_WEST_DARKNUTS for e in path
    )
    assert any(
        e.gate is GateKind.KILL_CLEAR and e.target_room == L3_SOUTH_DARKNUTS
        for e in path
    )
    assert any(
        e.gate is GateKind.KILL_CLEAR and e.target_room == L3_RAFT_PASSAGE
        for e in path
    )
    assert all(e.gate is not GateKind.BOMB for e in path)

    reached = LEVEL_3_DOOR_GRAPH.bfs_reachable(L3_DARKNUTS, caps)
    assert L3_RAFT_PASSAGE in reached
    assert L3_BOMB_SHORTCUT not in reached  # bomb-R needs a bomb

    pickup = [
        e
        for e in LEVEL_3_DOOR_GRAPH.edges_from(L3_RAFT_PASSAGE)
        if e.target_room is None
    ]
    assert pickup
    assert all(not e.is_pathfinding for e in pickup)


def test_level3_raft_bfs_blocked_without_key() -> None:
    """KEY-LEFT 0x5a → 0x59 spends a key; keys=0 cannot reach Raft."""
    caps = InventoryCaps(keys=0, bombs=0, can_clear=True)
    assert (
        LEVEL_3_DOOR_GRAPH.bfs_path(L3_DARKNUTS, L3_RAFT_PASSAGE, caps) is None
    )
    assert (
        LEVEL_3_DOOR_GRAPH.bfs_path(L3_COMPASS, L3_WEST_DARKNUTS, caps) is None
    )
    reached = LEVEL_3_DOOR_GRAPH.bfs_reachable(L3_DARKNUTS, caps)
    assert L3_COMPASS in reached  # OPEN west
    assert L3_WEST_DARKNUTS not in reached
    assert L3_RAFT_PASSAGE not in reached


def test_level3_fresh_copy() -> None:
    g = level_3_door_graph()
    assert g.room_ids() == LEVEL_3_DOOR_GRAPH.room_ids()
    g.rooms[L3_ENTRY] = ()
    assert LEVEL_3_DOOR_GRAPH.edges_from(L3_ENTRY)


# ---------------------------------------------------------------------------
# Level 4 / 5 / 9
# ---------------------------------------------------------------------------


def test_level4_bfs_entry_reaches_tf_and_stepladder() -> None:
    caps = InventoryCaps(keys=2, bombs=2, can_clear=True)
    reached = LEVEL_4_DOOR_GRAPH.bfs_reachable(L4_ENTRY, caps)
    assert L4_TRIFORCE in reached
    assert L4_STEPLADDER in reached
    path = LEVEL_4_DOOR_GRAPH.bfs_path(L4_ENTRY, L4_TRIFORCE, caps)
    assert path is not None
    assert path[-1].target_room == L4_TRIFORCE


def test_level4_tf_blocked_without_resources() -> None:
    assert (
        LEVEL_4_DOOR_GRAPH.bfs_path(
            L4_ENTRY, L4_TRIFORCE, InventoryCaps(keys=0, bombs=0, can_clear=True)
        )
        is None
    )


def test_level4_fresh_copy() -> None:
    g = level_4_door_graph()
    assert g.room_ids() == LEVEL_4_DOOR_GRAPH.room_ids()
    g.rooms[L4_ENTRY] = ()
    assert LEVEL_4_DOOR_GRAPH.edges_from(L4_ENTRY)


def test_level5_bfs_entry_reaches_tf() -> None:
    caps = InventoryCaps(keys=2, bombs=2, can_clear=True)
    reached = LEVEL_5_DOOR_GRAPH.bfs_reachable(L5_ENTRY, caps)
    assert L5_TRIFORCE in reached
    path = LEVEL_5_DOOR_GRAPH.bfs_path(L5_ENTRY, L5_TRIFORCE, caps)
    assert path is not None
    assert path[-1].target_room == L5_TRIFORCE


def test_level5_bfs_can_reach_whistle_with_bombs() -> None:
    caps = InventoryCaps(keys=3, bombs=2, can_clear=True)
    reached = LEVEL_5_DOOR_GRAPH.bfs_reachable(L5_ENTRY, caps)
    assert L5_WHISTLE_ITEM in reached


def test_level5_fresh_copy() -> None:
    g = level_5_door_graph()
    assert g.room_ids() == LEVEL_5_DOOR_GRAPH.room_ids()
    g.rooms[L5_ENTRY] = ()
    assert LEVEL_5_DOOR_GRAPH.edges_from(L5_ENTRY)


def test_level9_bfs_41_to_32() -> None:
    caps = InventoryCaps(keys=0, bombs=2, can_clear=True)
    reached = LEVEL_9_DOOR_GRAPH.bfs_reachable(L9_ROOM_41, caps)
    assert L9_ZELDA in reached
    path = LEVEL_9_DOOR_GRAPH.bfs_path(L9_ROOM_41, L9_ZELDA, caps)
    assert path is not None
    rooms = [L9_ROOM_41, *[e.target_room for e in path]]
    assert rooms[-1] == L9_ZELDA
    assert all("fixture_only" in (e.notes or "") for e in path)
    assert all(e.verification == "observed" for e in path)


def test_level9_fresh_copy() -> None:
    g = level_9_door_graph()
    assert g.room_ids() == LEVEL_9_DOOR_GRAPH.room_ids()
    g.rooms[L9_ROOM_41] = ()
    assert LEVEL_9_DOOR_GRAPH.edges_from(L9_ROOM_41)
