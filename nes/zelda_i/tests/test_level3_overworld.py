"""Unit tests for Level 3 overworld hop tables and stop predicates."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_overworld import (
    LEVEL3,
    LEVEL3_DOOR_HOPS_FROM_66,
    LEVEL3_HOPS_FROM_POST_L2,
    LEVEL3_PATH_HOPS,
    LEVEL3_PATH_SCREENS,
    LEVEL3_POST_L2_SCREENS,
    LEVEL3_SOURCE_PATH_SCREENS,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL3_ENTRY_ROOM,
    SCREEN_POST_L2_RETURN,
    OverworldPostL2ToLevel3Controller,
    OverworldToLevel3Controller,
    is_5c_maze_reverse_hop,
    level3_entrance_success,
    level3_path_success,
    post_l2_overworld_ready,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_COLLIDING_TILE,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL3_ENTRANCE)
    ram[ADDR_LINK_X] = fields.get("x", 128)
    ram[ADDR_LINK_Y] = fields.get("y", 140)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_COLLIDING_TILE] = fields.get("colliding_tile", 0)
    return ram


def test_level3_path_screens_chain() -> None:
    assert LEVEL3_PATH_SCREENS[0] == 0x77
    assert LEVEL3_PATH_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE == 0x74
    assert len(LEVEL3_PATH_HOPS) == len(LEVEL3_PATH_SCREENS) - 1
    for a, b in zip(LEVEL3_PATH_SCREENS, LEVEL3_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_level3_door_hops_from_66_neighbors() -> None:
    screens = (0x66,) + tuple(h.target for h in LEVEL3_DOOR_HOPS_FROM_66)
    assert screens[-1] == 0x74
    for a, b in zip(screens, screens[1:]):
        assert b in neighbor_screens(a).values()


def test_source_path_documented_but_not_default() -> None:
    """Source arithmetic ends on 0x74 but is not the controller default hops."""
    assert LEVEL3_SOURCE_PATH_SCREENS[-1] == 0x74
    assert LEVEL3_SOURCE_PATH_SCREENS[1] == 0x67
    assert LEVEL3_PATH_SCREENS[1] != 0x67


def test_level3_path_success() -> None:
    assert level3_path_success(_ram(screen=0x74, sword=1))
    assert not level3_path_success(_ram(screen=0x74, sword=0))
    assert not level3_path_success(_ram(screen=0x73, sword=1))


def test_level3_entrance_success() -> None:
    assert level3_entrance_success(
        _ram(level=LEVEL3, screen=SCREEN_LEVEL3_ENTRY_ROOM, mode=PLAY_MODE)
    )
    assert not level3_entrance_success(_ram(level=0, screen=0x74))
    assert not level3_entrance_success(_ram(level=LEVEL3, screen=0x7d))


def test_controller_defaults() -> None:
    nav = OverworldToLevel3Controller()
    assert nav.hops[-1].target == 0x74
    assert nav.entry_room == 0x7C
    assert nav.door_x == 128


def test_post_l2_path_screens_chain() -> None:
    assert LEVEL3_POST_L2_SCREENS[0] == SCREEN_POST_L2_RETURN == 0x3C
    assert LEVEL3_POST_L2_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE == 0x74
    assert len(LEVEL3_HOPS_FROM_POST_L2) == len(LEVEL3_POST_L2_SCREENS) - 1
    for a, b in zip(LEVEL3_POST_L2_SCREENS, LEVEL3_POST_L2_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_post_l2_maze_reverse_pred() -> None:
    maze_hops = [h for h in LEVEL3_HOPS_FROM_POST_L2 if is_5c_maze_reverse_hop(h)]
    assert len(maze_hops) == 1
    assert maze_hops[0].target == 0x5B


def test_post_l2_controller_defaults() -> None:
    nav = OverworldPostL2ToLevel3Controller(require_dungeon=True)
    assert nav.hops[0].target == 0x4C
    assert nav.hops[-1].target == 0x74
    assert nav.require_dungeon is True
    assert is_5c_maze_reverse_hop is nav.maze_hop_pred or nav.maze_hop_pred(
        nav.hops[4]
    )


def test_post_l2_overworld_ready() -> None:
    ram = _ram(screen=0x3C, sword=1)
    ram[ADDR_TRIFORCE] = 0x03
    assert post_l2_overworld_ready(ram)
    ram[ADDR_TRIFORCE] = 0x01
    assert not post_l2_overworld_ready(ram)
    ram[ADDR_TRIFORCE] = 0x03
    ram[ADDR_SCREEN] = 0x37
    assert not post_l2_overworld_ready(ram)


def test_post_l2_trap_bands_match_docs() -> None:
    """L2→L3 path traps must stay aligned with AGENTS / LEVEL3_ROUTE."""
    by_target = {h.target: h for h in LEVEL3_HOPS_FROM_POST_L2}
    # 0x4C east → 0x4D: y∈[133,145] only (y=149 solid)
    east = by_target[0x4D]
    assert east.direction == "RIGHT"
    assert east.y_band_lo == 133 and east.y_band_hi == 145
    # 0x5C reverse → 0x5B: no y_band (waypoints own the corridor)
    maze = by_target[0x5B]
    assert maze.direction == "LEFT"
    assert maze.y_band_lo is None and maze.y_band_hi is None
    assert is_5c_maze_reverse_hop(maze)
    # 0x64 west → 0x63: y≈125–150
    west = by_target[0x63]
    assert west.direction == "LEFT"
    assert west.y_band_lo == 125 and west.y_band_hi == 150


def test_post_l2_leave_64_inland_when_wrong_y() -> None:
    """West rock face at wrong y must step inland before band align."""
    from zelda_i.ram import ZeldaSnapshot

    nav = OverworldPostL2ToLevel3Controller()
    hop = next(h for h in nav.hops if h.target == 0x63)
    snap = ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=0x64,
        next_screen=0x64,
        link_x=24,
        link_y=109,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=127,
        triforce=0x03,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    act = nav._extra_hop_action(snap, hop)
    assert act is not None
    assert "64" in act.reason or "inland" in act.reason


def test_post_l2_63_east_obstruction_boundary_moves_inland() -> None:
    nav = OverworldPostL2ToLevel3Controller()
    snap = read_snapshot(
        _ram(screen=0x63, x=200, y=125, colliding_tile=0xC6)
    )

    action = nav._leave_63_south(snap)

    assert snap.mode == PLAY_MODE
    assert snap.colliding_tile == 0xC6
    assert action.reason.startswith("63_inland")
