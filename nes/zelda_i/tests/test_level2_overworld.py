from __future__ import annotations

import numpy as np

from zelda_i.level2_overworld import (
    LEVEL2_ENTRY_ROOM,
    LEVEL2_PATH_SCREENS,
    LEVEL2_REJOIN_4A_TO_5A,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
    is_5c_maze_hop,
    level2_door_hops_from,
    level2_entrance_success,
    level2_path_prefix_success,
    post_triforce_overworld_ready,
)
from zelda_i.overworld import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    LEVEL2_DOOR_HOPS,
    LEVEL2_PATH_HOPS,
    LEVEL2_PATH_SCREENS as OW_LEVEL2_SCREENS,
    NODE_LEVEL1_COMPLETE,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL2_PATH_4A,
    ScreenHop,
    build_early_route_graph,
    neighbor_screens,
)
from zelda_i.route_legs import level2_path_prefix_route_plan
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


def test_post_triforce_overworld_ready() -> None:
    assert post_triforce_overworld_ready(
        _ram(screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x01, level=0)
    )
    assert not post_triforce_overworld_ready(
        _ram(screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x00)
    )
    assert not post_triforce_overworld_ready(_ram(level=1, triforce=0x01))


def test_level2_path_prefix_success() -> None:
    assert level2_path_prefix_success(
        _ram(screen=0x4A, triforce=0x01, sword=1, level=0)
    )
    assert not level2_path_prefix_success(
        _ram(screen=0x37, triforce=0x01, sword=1)
    )


def test_settle_controller_accepts_overworld() -> None:
    ctrl = PostTriforceSettleController()
    from zelda_i.ram import read_snapshot

    snap = read_snapshot(
        _ram(mode=PLAY_MODE, level=0, screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x01)
    )
    ctrl.step(snap)
    assert ctrl.success
    assert ctrl.phase.name == "DONE"


def test_nav_controller_reports_hop() -> None:
    ctrl = OverworldToLevel2Controller()
    assert ctrl.hop_index == 0
    rep = ctrl.report()
    assert rep["hop"]["target"] == 0x38


def test_level2_path_prefix_route_plan() -> None:
    planned = level2_path_prefix_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL2_PATH_4A
    assert "triforce_shard_1" in planned[-1].capabilities_before
    assert any(leg.leg.leg_id == "settle_post_triforce_overworld" for leg in planned)


def test_graph_has_level2_prefix_edge() -> None:
    graph = build_early_route_graph()
    assert NODE_LEVEL1_EXIT_OVERWORLD in graph.nodes
    assert NODE_LEVEL2_PATH_4A in graph.nodes
    edge = graph.edge_for(NODE_LEVEL1_COMPLETE, NODE_LEVEL1_EXIT_OVERWORLD)
    assert edge is not None
    assert edge.verification == "observed"
    edge2 = graph.edge_for(NODE_LEVEL1_EXIT_OVERWORLD, NODE_LEVEL2_PATH_4A)
    assert edge2 is not None
    assert edge2.verification == "observed"
    # Grid hops along the verified walk prefix are promoted to observed.
    hop_edge = graph.edge_for("ow_37", "ow_38")
    assert hop_edge is not None
    assert hop_edge.verification == "observed"
    assert hop_edge.meta.get("segment") == "to_level2_prefix"


def test_level2_door_path_geometry() -> None:
    from zelda_i.overworld import (
        LEVEL2_DOOR_SCREENS,
        NODE_LEVEL2_DUNGEON,
        NODE_LEVEL2_ENTRANCE,
        node_id_for_screen,
    )

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


def test_door_path_flag_wires_level2_door_hops() -> None:
    ctrl = OverworldToLevel2Controller(door_path=True)
    assert ctrl.hops is LEVEL2_DOOR_HOPS or ctrl.hops == LEVEL2_DOOR_HOPS
    assert ctrl.hops[-1].target == 0x3C
    assert 0x4B not in {h.target for h in ctrl.hops}
    maze_hops = [h for h in ctrl.hops if is_5c_maze_hop(h)]
    assert len(maze_hops) == 1
    assert maze_hops[0].target == 0x5D


def test_level2_door_hops_from_rejoin_4a() -> None:
    from zelda_i.level2_overworld import (
        LEVEL2_CLEAN_FROM_4A_TO_5A,
        LEVEL2_CLEAN_FROM_5A_TO_3C,
        LEVEL2_REJOIN_4A_HOPS,
    )

    from_4a = level2_door_hops_from(0x4A)
    # 0x4A has no south exit; rejoin west→0x49 then south→0x59→door path.
    assert from_4a[0].target == 0x49
    assert from_4a[0].direction == "LEFT"
    assert from_4a[1].target == 0x59
    assert from_4a[1].direction == "DOWN"
    assert from_4a[2].target == 0x5A
    assert from_4a[-1].target == 0x3C
    assert from_4a[:2] == LEVEL2_REJOIN_4A_HOPS
    assert LEVEL2_REJOIN_4A_TO_5A.target == 0x49  # alias = first rejoin hop
    # Never routes through the north-entry trap.
    assert 0x4B not in {h.target for h in from_4a}
    from_37 = level2_door_hops_from(0x37)
    assert from_37 == LEVEL2_DOOR_HOPS
    from_5c = level2_door_hops_from(0x5C)
    assert from_5c[0].target == 0x5D
    assert is_5c_maze_hop(from_5c[0])
    # Clean hop tables: y≈140 into/out of 0x5A, maze still ends at 0x3C.
    assert LEVEL2_CLEAN_FROM_4A_TO_5A[-1].target == 0x5A
    assert LEVEL2_CLEAN_FROM_4A_TO_5A[-1].align_y == 140
    assert LEVEL2_CLEAN_FROM_5A_TO_3C[0].target == 0x5B
    assert LEVEL2_CLEAN_FROM_5A_TO_3C[0].align_y == 140
    assert LEVEL2_CLEAN_FROM_5A_TO_3C[-1].target == 0x3C
    assert is_5c_maze_hop(LEVEL2_CLEAN_FROM_5A_TO_3C[2])


def test_maze_phase_follows_waypoints() -> None:
    """On 0x5C with door hop to 0x5D, controller tracks maze waypoints."""
    from zelda_i.ram import read_snapshot

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


def test_is_5c_maze_hop() -> None:
    assert is_5c_maze_hop(ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140))
    assert not is_5c_maze_hop(ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95))
    assert not is_5c_maze_hop(ScreenHop(0x5D, "UP", align_x=52))


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


def test_require_dungeon_stop_and_door_align() -> None:
    """require_dungeon idles settle and aligns door x before UP."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel2Controller(hops=(), require_dungeon=True)
    # Settled entry room → success.
    snap = read_snapshot(_ram(level=2, mode=PLAY_MODE, screen=0x7D, x=120, y=205))
    act = ctrl.step(snap)
    assert ctrl.success
    assert act.reason == "done"

    ctrl = OverworldToLevel2Controller(hops=(), require_dungeon=True)
    # Off-center on door screen: re-align x before UP.
    snap = read_snapshot(
        _ram(level=0, mode=PLAY_MODE, screen=0x3C, x=80, y=180, sword=1, triforce=0x01)
    )
    act = ctrl.step(snap)
    assert "door_ax" in act.reason

    ctrl = OverworldToLevel2Controller(hops=(), require_dungeon=True)
    snap = read_snapshot(
        _ram(level=0, mode=PLAY_MODE, screen=0x3C, x=112, y=180, sword=1, triforce=0x01)
    )
    act = ctrl.step(snap)
    assert "door_hunt" in act.reason

    # Mid enter: idle settle.
    ctrl = OverworldToLevel2Controller(hops=(), require_dungeon=True)
    snap = read_snapshot(_ram(level=2, mode=16, screen=0x3C, x=112, y=125))
    act = ctrl.step(snap)
    assert act.reason in ("dungeon_settle", "scroll_idle")
