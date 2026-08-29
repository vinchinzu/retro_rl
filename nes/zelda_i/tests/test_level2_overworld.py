from __future__ import annotations

import numpy as np

from zelda_i.level2.overworld import (
    LEVEL2_ENTRY_ROOM,
    LEVEL2_PATH_SCREENS,
    OverworldToLevel2Controller,
    is_5c_maze_hop,
    level2_door_hops_from,
    level2_entrance_success,
)
from zelda_i.overworld.graph import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    LEVEL2_DOOR_HOPS,
    LEVEL2_DOOR_SCREENS,
    LEVEL2_PATH_HOPS,
    LEVEL2_PATH_SCREENS as OW_LEVEL2_SCREENS,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL2_DUNGEON,
    NODE_LEVEL2_ENTRANCE,
    build_early_route_graph,
    neighbor_screens,
    node_id_for_screen,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL1_ENTRANCE)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x01)
    return ram


def test_level2_path_screens_chain() -> None:
    assert LEVEL2_PATH_SCREENS[0] == 0x37
    assert LEVEL2_PATH_SCREENS[-1] == 0x4A
    assert LEVEL2_PATH_SCREENS == OW_LEVEL2_SCREENS
    assert len(LEVEL2_PATH_HOPS) == len(LEVEL2_PATH_SCREENS) - 1
    assert all(
        hop.target == screen
        for hop, screen in zip(LEVEL2_PATH_HOPS, LEVEL2_PATH_SCREENS[1:])
    )
    for a, b in zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()


def test_level2_door_path_geometry() -> None:
    assert LEVEL2_DOOR_SCREENS[0] == 0x37
    assert LEVEL2_DOOR_SCREENS[-1] == 0x3C
    assert 0x5A in LEVEL2_DOOR_SCREENS  # west entry into 0x5B
    assert 0x4B not in LEVEL2_DOOR_SCREENS  # north-entry trap
    assert 0x79 not in LEVEL2_DOOR_SCREENS
    assert len(LEVEL2_DOOR_HOPS) == len(LEVEL2_DOOR_SCREENS) - 1
    assert len(LEVEL2_5C_MAZE_WAYPOINTS) >= 10
    hop_4d = next(h for h in LEVEL2_DOOR_HOPS if h.target == 0x4D)
    assert hop_4d.direction == "UP"
    assert hop_4d.align_x is not None and hop_4d.align_x < 80

    graph = build_early_route_graph()
    for screen in LEVEL2_DOOR_SCREENS:
        assert node_id_for_screen(screen) in graph.nodes
    for a, b in zip(LEVEL2_DOOR_SCREENS, LEVEL2_DOOR_SCREENS[1:]):
        assert graph.edge_for(node_id_for_screen(a), node_id_for_screen(b)) is not None
    abstract = graph.edge_for(NODE_LEVEL1_EXIT_OVERWORLD, NODE_LEVEL2_ENTRANCE)
    assert abstract is not None
    assert abstract.verification == "planned"
    assert abstract.meta.get("segment") == "to_level2_door"
    enter = graph.edge_for(NODE_LEVEL2_ENTRANCE, NODE_LEVEL2_DUNGEON)
    assert enter is not None
    assert enter.verification == "planned"


def test_maze_phase_follows_waypoints() -> None:
    """On 0x5C with door hop to 0x5D, controller tracks maze waypoints."""
    hops = level2_door_hops_from(0x5C)
    ctrl = OverworldToLevel2Controller(hops=hops)
    assert is_5c_maze_hop(ctrl.hops[0])

    # Entry on 0x5C west edge near first maze corridor y.
    snap = read_snapshot(_ram(screen=0x5C, x=16, y=93, sword=1, triforce=0x01))
    act = ctrl.step(snap)
    assert "maze" in act.reason
    assert "maze_start" in ctrl.notes
    assert ctrl.report()["hop"]["maze"] is True

    # Near first waypoint → advance index and push toward next.
    tx, ty = LEVEL2_5C_MAZE_WAYPOINTS[0]
    snap = read_snapshot(_ram(screen=0x5C, x=tx, y=ty, sword=1, triforce=0x01))
    ctrl.step(snap)
    assert ctrl.maze_wp_index >= 1

    # Default prefix controller does not treat 0x4A path as maze.
    prefix = OverworldToLevel2Controller()
    assert not any(is_5c_maze_hop(h) for h in prefix.hops)


def test_level2_entrance_success_requires_room_ready() -> None:
    """Moon entry is level==2 + mode 5 + room 0x7d (not mid mode-16 door)."""
    assert LEVEL2_ENTRY_ROOM == 0x7D
    assert level2_entrance_success(
        _ram(level=2, mode=PLAY_MODE, screen=0x7D, x=120, y=205)
    )
    # Mid cave-enter still on OW screen id — not room-ready.
    assert not level2_entrance_success(
        _ram(level=2, mode=16, screen=0x3C, x=112, y=125)
    )
    assert not level2_entrance_success(
        _ram(level=0, mode=PLAY_MODE, screen=0x3C, sword=1, triforce=0x01)
    )
