"""0x6b strand seed + 0x5c/0x5d $6530 BLOCK_TILES seed (no emulator)."""

from __future__ import annotations

import numpy as np

from types import SimpleNamespace

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.tilemap import (
    ADDR_ROOM_TILE_MAP,
    TILE_ROWS,
    WRAM_BASE,
    WRAM_RAM_OFFSET,
)
from zelda_i.level3.boss_path import UP_5D_SPEC, L3DoorHopController
from zelda_i.level3.dungeon import ROOM_L3_BOSS_PREP, ROOM_L3_NORTH_ZOLS as ROOM_6B
from zelda_i.level3.geometry import NORTH_DOOR_X, ROOM_6B_BAND_Y
from zelda_i.level3.occupancy import link_block_pixels, seed_block_cells
from zelda_i.level3.path import Level3NorthExit6bController
from zelda_i.tests.ram_helpers import make_ram
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker, WALK_DELTA
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = ROOM_6B
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return ram


def test_miss_blocks_ahead_and_replans() -> None:
    # Off the door column — x≈120 inland is a leave-column residual (v6).
    ctrl = Level3NorthExit6bController()
    start = read_snapshot(_ram(x=96, y=141))
    first = ctrl.step(start)
    assert first.reason == "north6b_path"
    assert ctrl.walker.last_dir in WALK_DELTA
    blocked_ahead = {
        "UP": (96, 140),
        "DOWN": (96, 142),
        "LEFT": (95, 141),
        "RIGHT": (97, 141),
    }[ctrl.walker.last_dir]
    second = ctrl.step(start)
    assert ctrl.misses == 1
    assert blocked_ahead in ctrl.grid.blocked
    assert second.reason in {"north6b_path", "north6b_thread", "north6b_thread_up"}
    path = ctrl.grid.shortest_path((96, 141), (NORTH_DOOR_X, ROOM_6B_BAND_Y))
    assert path is not None
    assert blocked_ahead not in path


def test_south_mouth_is_not_occupancy_graded() -> None:
    """Combat can leave Link on the south door; do not miss-block inland UP.

    Cardinals stick at (120,181) (v2). LEFT+UP is the door clip that moves
    (v4); once off x≈120, UP inland. Occupancy does not grade the residual.
    """
    ctrl = Level3NorthExit6bController()
    door = ctrl.step(read_snapshot(_ram(x=120, y=181)))
    assert door.reason == "north6b_leave_mouth_clip"
    assert list(door.action) == list(nes_action("LEFT", "UP"))
    assert ctrl.misses == 0
    assert ctrl.walker.last_dir is None
    inland = ctrl.step(read_snapshot(_ram(x=100, y=181)))
    assert inland.reason == "north6b_leave_mouth"
    assert list(inland.action) == list(nes_action("UP"))
    assert ctrl.misses == 0


_BLOCK_QUAD = (0xB0, 0xB2, 0xB1, 0xB3)
# Measured 0x5c solids: waist leftover (96,141); south UP at x=144/176/192 is 0xB3.
_DIAMONDS_5C = ((96, 144), (144, 176), (176, 176), (192, 176))
# 0x5d center 2x2 on the north-door column. miss_UP_120_165 is 1px south.
_PLUS_5D = ((112, 144), (128, 144), (112, 160), (128, 160))


def _set_cell(ram: np.ndarray, x: int, y: int, quad) -> None:
    base = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    col, row = x // 8, (y - 64) // 8
    for dc, dr, value in (
        (0, 0, quad[0]),
        (1, 0, quad[1]),
        (0, 1, quad[2]),
        (1, 1, quad[3]),
    ):
        ram[base + (col + dc) * TILE_ROWS + (row + dr)] = value


def _wram_5c(diamonds=_DIAMONDS_5C) -> np.ndarray:
    ram = np.zeros(10240, dtype=np.uint8)
    for cx, cy in diamonds:
        _set_cell(ram, cx, cy, _BLOCK_QUAD)
    return ram


def test_5c_seed_is_spec_declared_not_inferred() -> None:
    grid = OccupancyGrid()
    n = seed_block_cells(grid, _wram_5c())
    assert n > 0
    waist = link_block_pixels(96, 144)
    assert (97, 141) in waist
    assert (97, 141) in grid.blocked
    assert (97, 141) not in grid.inferred
    assert grid.passable(208, 141)


def test_5c_seed_bfs_leaves_waist_diamond() -> None:
    """(96,141) on a diamond: BFS goes around, never RIGHT through the cell."""
    grid = OccupancyGrid()
    seed_block_cells(grid, _wram_5c())
    grid.blocked.discard((208, 141))
    path = grid.shortest_path((96, 141), (208, 141))
    assert path is not None
    assert (97, 141) not in path
    assert path[-1] == (208, 141)


def test_5c_seed_bfs_does_not_up_into_se_diamond() -> None:
    """(144,179) / (192,181) UP is tile 0xB3. BFS stays south of those cells."""
    grid = OccupancyGrid()
    seed_block_cells(grid, _wram_5c())
    grid.blocked.discard((208, 141))
    path = grid.shortest_path((144, 179), (208, 141))
    assert path is not None
    assert (144, 178) not in path
    se = grid.shortest_path((192, 181), (208, 141))
    assert se is not None
    assert (192, 180) not in se


def test_5c_spec_blocks_survive_inferred_forget() -> None:
    grid = OccupancyGrid()
    seed_block_cells(grid, _wram_5c())
    walker = OccupancyWalker(grid=grid)
    for direction in ("RIGHT", "LEFT", "DOWN", "UP"):
        grid.mark_blocked_ahead(96, 141, direction)
    step = walker.next_dir((96, 141), (208, 141))
    assert step is not None
    assert (97, 141) in grid.blocked
    assert walker.forgets == 1


def test_seed_block_cells_noops_without_cart_wram() -> None:
    grid = OccupancyGrid()
    assert seed_block_cells(grid, np.zeros(0x800, dtype=np.uint8)) == 0
    assert not grid.blocked


def _wram_5d_plus(*, x: int = 120, y: int = 166) -> np.ndarray:
    cpu = make_ram(
        {
            "mode": PLAY_MODE,
            "level": 3,
            "x": x,
            "y": y,
            "triforce": 0x03,
            "bombs": 4,
            "keys": 4,
            "health": 0x71,
        },
        mode=PLAY_MODE,
        level=3,
        screen=ROOM_L3_BOSS_PREP,
        x=x,
        y=y,
        doors=10,
        bombs=4,
        keys=4,
        health=0x71,
        triforce=0x03,
    )
    ram = np.zeros(10240, dtype=np.uint8)
    ram[:0x800] = cpu
    for cx, cy in _PLUS_5D:
        _set_cell(ram, cx, cy, _BLOCK_QUAD)
    return ram


def _assert_5d_around(path: list[tuple[int, int]] | None, dest: tuple[int, int]) -> None:
    assert path is not None
    assert path[-1] == dest
    plus = link_block_pixels(112, 160) | link_block_pixels(112, 144)
    assert (120, 164) in plus
    assert not any(cell in plus and cell != path[0] for cell in path)
    assert any(px != 120 for px, _py in path)


def test_5d_seed_bfs_goes_around_center_plus() -> None:
    """(120,166)/(120,175) UP through the plus is miss_UP_120_165. BFS around."""
    grid = OccupancyGrid()
    seed_block_cells(grid, _wram_5d_plus())
    grid.blocked.discard((120, 93))
    grid.blocked.discard((120, 109))
    assert (120, 164) in grid.blocked
    assert (120, 164) not in grid.inferred
    assert grid.passable(120, 93)
    assert grid.passable(120, 109)
    for start in ((120, 166), (120, 175)):
        _assert_5d_around(grid.shortest_path(start, (120, 109)), (120, 109))
        _assert_5d_around(grid.shortest_path(start, (120, 93)), (120, 93))


def test_5d_spec_blocks_survive_inferred_forget() -> None:
    grid = OccupancyGrid()
    seed_block_cells(grid, _wram_5d_plus())
    walker = OccupancyWalker(grid=grid)
    for direction in ("UP", "LEFT", "RIGHT", "DOWN"):
        grid.mark_blocked_ahead(120, 166, direction)
    step = walker.next_dir((120, 166), (120, 109))
    assert step is not None
    assert (120, 164) in grid.blocked
    assert (120, 164) not in grid.inferred
    assert walker.forgets == 1
    path = grid.shortest_path((120, 166), (120, 109))
    _assert_5d_around(path, (120, 109))


def test_up_5d_bind_env_seeds_plus_not_door_column() -> None:
    ram = _wram_5d_plus(x=120, y=166)
    ctl = L3DoorHopController(UP_5D_SPEC)
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    assert ctl._blocks_seeded
    assert any(n.startswith("seed_blocks_") for n in ctl.notes)
    assert UP_5D_SPEC.goal == (120, 93)
    assert ctl.walker.grid.passable(120, 93)
    assert ctl.walker.grid.passable(120, 109)
    assert (120, 164) in ctl.walker.grid.blocked
    assert (120, 164) not in ctl.walker.grid.inferred
    leftover = read_snapshot(ram)
    first = ctl.step(leftover)
    assert not ctl.failed
    assert ctl.goal in {(120, 93), (120, 109)}
    assert list(first.action) != list(nes_idle_action())
    _assert_5d_around(
        ctl.walker.grid.shortest_path((120, 166), ctl.goal), ctl.goal
    )
    _assert_5d_around(
        ctl.walker.grid.shortest_path((120, 175), ctl.goal), ctl.goal
    )
    stuck = ctl.step(leftover)
    assert list(stuck.action) != list(nes_action("UP"))
    assert (120, 164) in ctl.walker.grid.blocked
    assert (120, 164) not in ctl.walker.grid.inferred


def test_up_5d_south_mouth_does_not_push_down() -> None:
    """Serial red (103,189): south_band DOWN-pushed the south door. Occupancy dest owns leave."""
    assert UP_5D_SPEC.south_band is False
    assert UP_5D_SPEC.align == "dest"
    assert UP_5D_SPEC.goal == (120, 93)
    ram = _wram_5d_plus(x=103, y=189)
    ctl = L3DoorHopController(UP_5D_SPEC)
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    leftover = read_snapshot(ram)
    first = ctl.step(leftover)
    assert not ctl.failed
    assert ctl.goal == (120, 109)
    assert list(first.action) != list(nes_action("DOWN"))
    assert first.reason == "north_path"
    assert list(first.action) != list(nes_idle_action())
    _assert_5d_around(
        ctl.walker.grid.shortest_path((103, 189), ctl.goal), ctl.goal
    )
    _assert_5d_around(
        ctl.walker.grid.shortest_path((120, 175), ctl.goal), ctl.goal
    )
    reasons = [ctl.step(leftover).reason for _ in range(8)]
    assert "south_push" not in reasons
    assert "south_align" not in reasons
