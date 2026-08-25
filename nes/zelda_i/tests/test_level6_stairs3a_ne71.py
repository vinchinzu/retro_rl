"""Unit tests for Level 6 0x3A stairs: LEFT around tile 119 at x=160 onto 0x71."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_path import BLOCK_OBJECT_TYPE, south_face_stand
from zelda_i.level6_spine import L6_THROUGH
from zelda_i.level6_stairs3a_ne71 import (
    AROUND_X,
    DATED_LEFTOVER,
    HOLE_COLUMN,
    HOLE_TILE,
    SOUTH_AROUND_Y,
    V1_TO_NE,
    V2_AROUND,
    V2_TO_NE,
    V3_AROUND,
    V3_HOLE,
    WARP_TILE,
    WEST_UP_X,
    level6_stairs3a_ne71_stages,
    level6_stairs3a_ne71_success,
    make_stairs_3a_ne71_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", 144)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0)
    return ram


def _plant_block(ram: np.ndarray, slot: int, x: int, y: int) -> None:
    ram[ADDR_OBJ_TYPE + slot] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def _ne_ctl(*, phase=None, hole_x: int | None = 112, misses: int = 0):
    ctl = make_stairs_3a_ne71_controller()
    if phase is not None:
        ctl.phase = phase
    ctl.hole_x = hole_x
    ctl.walker.misses = misses
    return ctl


def test_level6_stairs3a_ne71_left_around_119_at_x160_then_tile_71() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_stairs3a import center_block_0x68
    from zelda_i.level6_stairs3a_ne71 import Stairs3ANE71Phase

    stages = level6_stairs3a_ne71_stages()
    assert [name for name, _, _ in stages] == ["level6_stairs_0x3a_ne71"]
    leftover = _ram(level=6, screen=0x3A, x=144, y=141, keys=4, tile=118)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    _plant_block(leftover, 11, 112, 144)
    ctl = make_stairs_3a_ne71_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "stand_path"
    assert list(act.action) in (
        list(nes_action("LEFT")),
        list(nes_action("DOWN")),
    )
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("B"))
    assert DATED_LEFTOVER == (144, 141)
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "stand_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert not ctl.failed
    aligned = _ram(level=6, screen=0x3A, x=114, y=149, tile=116)
    aligned[ADDR_ROD] = 1
    _plant_block(aligned, 11, 112, 144)
    act = ctl.step(read_snapshot(aligned))
    assert act.reason == "stand_y"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert not ctl.failed
    assert ctl.walker.misses == 1
    south = _ram(level=6, screen=0x3A, x=114, y=160)
    south[ADDR_ROD] = 1
    _plant_block(south, 11, 112, 144)
    act = ctl.step(read_snapshot(south))
    assert act.reason == "push_block"
    assert list(act.action) == list(nes_action("UP"))
    assert not ctl.failed
    block = center_block_0x68(read_snapshot(leftover))
    assert block is not None
    assert south_face_stand(block) == (112, 160)
    at_stand = make_stairs_3a_ne71_controller()
    ram_stand = _ram(level=6, screen=0x3A, x=112, y=160)
    ram_stand[ADDR_ROD] = 1
    _plant_block(ram_stand, 11, 112, 144)
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "push_block"
    assert list(act.action) == list(nes_action("UP"))
    ram_stand[ADDR_LINK_Y + 11] = 136
    act = at_stand.step(read_snapshot(ram_stand))
    assert any(n.startswith("pushed_112_144_to_112_136") for n in at_stand.notes)
    assert act.reason != "hole_idle"
    assert act.reason != "hole_y"
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_idle_action())
    assert act.reason in ("ne_sidestep", "wait_ne_block")
    _plant_block(ram_stand, 11, 208, 96)
    ram_stand[ADDR_COLLIDING_TILE] = HOLE_TILE
    ram_stand[ADDR_LINK_X] = 112
    ram_stand[ADDR_LINK_Y] = 146
    hole = _ne_ctl(phase=at_stand.phase, hole_x=112)
    hole.notes = list(at_stand.notes)
    act = hole.step(read_snapshot(ram_stand))
    assert act.reason == "ne_sidestep"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    inland = _ram(level=6, screen=0x3A, x=128, y=160)
    inland[ADDR_ROD] = 1
    _plant_block(inland, 11, 208, 96)
    to_ne = _ne_ctl(phase=hole.phase, hole_x=112)
    act = to_ne.step(read_snapshot(inland))
    assert act.reason == "ne_sidestep"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    west_south = _ram(level=6, screen=0x3A, x=72, y=165, tile=116)
    west_south[ADDR_ROD] = 1
    _plant_block(west_south, 11, 208, 96)
    from_v2 = _ne_ctl(phase=hole.phase, hole_x=112, misses=1)
    act = from_v2.step(read_snapshot(west_south))
    assert act.reason == "ne_sidestep"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    assert not from_v2.failed
    assert from_v2.walker.misses == 1
    # v1 leftover (114,149) tile 116 last_dir=RIGHT after push: clip, no halt.
    v1_ne = _ram(level=6, screen=0x3A, x=V1_TO_NE[0], y=V1_TO_NE[1], tile=116)
    v1_ne[ADDR_ROD] = 1
    _plant_block(v1_ne, 11, 112, 136)
    skip_halt = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=1)
    skip_halt.walker.last_dir = "RIGHT"
    skip_halt.walker.last_xy = (112, 149)
    act = skip_halt.step(read_snapshot(v1_ne))
    assert not skip_halt.failed
    assert skip_halt.walker.misses == 2
    assert act.reason == "ne_around_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert V1_TO_NE == (114, 149)
    assert SOUTH_AROUND_Y == 160
    # v2 leftover (122,149) tile 118: same clip, no occupancy_halt.
    v2_ne = _ram(level=6, screen=0x3A, x=V2_TO_NE[0], y=V2_TO_NE[1], tile=118)
    v2_ne[ADDR_ROD] = 1
    _plant_block(v2_ne, 11, 208, 96)
    skip_v2 = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=2)
    skip_v2.walker.last_dir = "RIGHT"
    skip_v2.walker.last_xy = (114, 149)
    act = skip_v2.step(read_snapshot(v2_ne))
    assert not skip_v2.failed
    assert skip_v2.walker.misses == 3
    assert act.reason == "ne_around_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert V2_TO_NE == (122, 149)
    # v1 leftover (158,149) tile 118 last_dir=LEFT: still RIGHT+DOWN to
    # AROUND_X. Do not LEFT around west of 160. Do not occupancy_halt.
    v1_early = _ram(level=6, screen=0x3A, x=158, y=149, tile=118)
    v1_early[ADDR_ROD] = 1
    _plant_block(v1_early, 11, 208, 96)
    skip_early = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=1)
    skip_early.walker.last_dir = "LEFT"
    skip_early.walker.last_xy = (158, 149)
    act = skip_early.step(read_snapshot(v1_early))
    assert not skip_early.failed
    assert skip_early.walker.misses == 2
    assert not skip_early.passed_around
    assert act.reason == "ne_around_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    # v2 leftover: passed_around after AROUND_X, LEFT miss at (158,149).
    # Clip LEFT around; do not occupancy_halt that cell.
    v2_over = _ram(level=6, screen=0x3A, x=V2_AROUND[0], y=V2_AROUND[1], tile=118)
    v2_over[ADDR_ROD] = 1
    _plant_block(v2_over, 11, 208, 96)
    skip_over = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=1)
    skip_over.passed_around = True
    skip_over.walker.last_dir = "LEFT"
    skip_over.walker.last_xy = (160, 149)
    act = skip_over.step(read_snapshot(v2_over))
    assert not skip_over.failed
    assert skip_over.walker.misses == 2
    assert skip_over.passed_around
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert skip_over.walker.last_dir is None
    assert V2_AROUND == (158, 149)
    # Tile 119 on y=149 west of AROUND_X is still the corridor, not LEFT.
    early_119 = _ram(level=6, screen=0x3A, x=152, y=149, tile=HOLE_TILE)
    early_119[ADDR_ROD] = 1
    _plant_block(early_119, 11, 208, 96)
    keep_clip = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=1)
    act = keep_clip.step(read_snapshot(early_119))
    assert not keep_clip.failed
    assert not keep_clip.passed_around
    assert act.reason == "ne_around_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    south_around = _ram(level=6, screen=0x3A, x=114, y=SOUTH_AROUND_Y)
    south_around[ADDR_ROD] = 1
    _plant_block(south_around, 11, 208, 96)
    around_push = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=2)
    act = around_push.step(read_snapshot(south_around))
    assert act.reason == "ne_sidestep"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert not around_push.failed
    # v3 leftover (160,147) tile 119: LEFT around, never UP/RIGHT.
    v3_around = _ram(
        level=6, screen=0x3A, x=V3_AROUND[0], y=V3_AROUND[1], tile=HOLE_TILE
    )
    v3_around[ADDR_ROD] = 1
    _plant_block(v3_around, 11, 208, 96)
    around = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=2)
    around.walker.last_dir = "UP"
    around.walker.last_xy = (160, 149)
    act = around.step(read_snapshot(v3_around))
    assert not around.failed
    assert around.walker.misses == 3
    assert around.passed_around
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert V3_AROUND == (160, 147)
    assert AROUND_X == 160
    # Tile 119 at x=184 still LEFT, never RIGHT (stairs3a-71 v3).
    v3_hole = _ram(level=6, screen=0x3A, x=V3_HOLE[0], y=V3_HOLE[1], tile=HOLE_TILE)
    v3_hole[ADDR_ROD] = 1
    _plant_block(v3_hole, 11, 208, 96)
    around184 = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=1)
    act = around184.step(read_snapshot(v3_hole))
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    assert not around184.failed
    assert V3_HOLE == (184, 147)
    assert HOLE_COLUMN == 184
    # Hole column x=184 y=165: LEFT around, not UP.
    ne_col = _ram(level=6, screen=0x3A, x=HOLE_COLUMN, y=165)
    ne_col[ADDR_ROD] = 1
    _plant_block(ne_col, 11, 208, 96)
    left_col = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    act = left_col.step(read_snapshot(ne_col))
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    # AROUND_X=160 y=165: LEFT around, not UP (stairs3a-ne v3 UP onto 119).
    mid = _ram(level=6, screen=0x3A, x=AROUND_X, y=165)
    mid[ADDR_ROD] = 1
    _plant_block(mid, 11, 208, 96)
    left_mid = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    act = left_mid.step(read_snapshot(mid))
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert left_mid.passed_around
    # v2 proved UP at (144,149) is solid. Continue LEFT to WEST_UP_X=136.
    v2_up_miss = _ram(level=6, screen=0x3A, x=144, y=149, tile=116)
    v2_up_miss[ADDR_ROD] = 1
    _plant_block(v2_up_miss, 11, 208, 96)
    continue_left = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    continue_left.passed_around = True
    act = continue_left.step(read_snapshot(v2_up_miss))
    assert act.reason == "ne_around"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    # After that extra cell, climb west of tile 119 and east of the stairs.
    west_up = _ram(level=6, screen=0x3A, x=WEST_UP_X, y=165)
    west_up[ADDR_ROD] = 1
    _plant_block(west_up, 11, 208, 96)
    up_west = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    up_west.passed_around = True
    act = up_west.step(read_snapshot(west_up))
    assert act.reason == "ne_y"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert WEST_UP_X == 136
    north = _ram(level=6, screen=0x3A, x=128, y=112)
    north[ADDR_ROD] = 1
    _plant_block(north, 11, 208, 96)
    across = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    across.passed_around = True
    act = across.step(read_snapshot(north))
    assert act.reason == "ne_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    ne_stand = _ram(level=6, screen=0x3A, x=208, y=112)
    ne_stand[ADDR_ROD] = 1
    _plant_block(ne_stand, 11, 208, 96)
    push_ne = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112)
    push_ne.passed_around = True
    act = push_ne.step(read_snapshot(ne_stand))
    assert act.reason == "push_ne_block"
    assert list(act.action) == list(nes_action("UP"))
    # NE 0x68 does not y-move; keep UP until tile 0x71.
    still_block = _ram(level=6, screen=0x3A, x=208, y=112)
    still_block[ADDR_ROD] = 1
    _plant_block(still_block, 11, 208, 96)
    hold = _ne_ctl(phase=Stairs3ANE71Phase.PUSH_NE, hole_x=112)
    hold.block_slot = 11
    hold.block_x0 = 208
    hold.block_y0 = 96
    act = hold.step(read_snapshot(still_block))
    assert act.reason == "push_ne_block"
    assert list(act.action) == list(nes_action("UP"))
    assert not hold.failed
    ne_stand[ADDR_COLLIDING_TILE] = WARP_TILE
    ne_stand[ADDR_LINK_Y] = 93
    act = push_ne.step(read_snapshot(ne_stand))
    assert act.reason == "warp_idle"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    assert WARP_TILE == 0x71
    assert any("warp_tile_71" in n for n in push_ne.notes)
    east = _ram(level=6, screen=0x3A, x=208, y=141, mode=6)
    east[ADDR_ROD] = 1
    door = make_stairs_3a_ne71_controller()
    act = door.step(read_snapshot(east))
    assert door.failed
    assert act.reason.startswith("east_door_")
    dest39 = _ram(level=6, screen=0x39, x=16, y=141)
    dest39[ADDR_ROD] = 1
    west = make_stairs_3a_ne71_controller()
    act = west.step(read_snapshot(dest39))
    assert west.failed
    assert act.reason.startswith("west_door_")
    assert not level6_stairs3a_ne71_success(read_snapshot(dest39))
    north29 = _ram(level=6, screen=0x29, x=120, y=205)
    north29[ADDR_ROD] = 1
    n29 = make_stairs_3a_ne71_controller()
    act = n29.step(read_snapshot(north29))
    assert n29.failed
    assert act.reason.startswith("north_29_")
    assert not level6_stairs3a_ne71_success(read_snapshot(north29))
    key09 = _ram(level=6, screen=0x09, x=120, y=205)
    key09[ADDR_ROD] = 1
    k09 = make_stairs_3a_ne71_controller()
    act = k09.step(read_snapshot(key09))
    assert k09.failed
    assert act.reason.startswith("key_up_09_")
    ram = _ram(level=6, screen=0x3A, x=208, y=93, mode=9, tile=WARP_TILE)
    ram[ADDR_ROD] = 1
    arrive = make_stairs_3a_ne71_controller()
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "warped_9"
    assert level6_stairs3a_ne71_success(read_snapshot(ram))
    still = _ram(level=6, screen=0x3A, x=144, y=141)
    still[ADDR_ROD] = 1
    assert not level6_stairs3a_ne71_success(read_snapshot(still))
    dest = _ram(level=6, screen=0x35, x=120, y=141)
    dest[ADDR_ROD] = 1
    assert level6_stairs3a_ne71_success(read_snapshot(dest))
    dest[ADDR_ROD] = 0
    assert not level6_stairs3a_ne71_success(read_snapshot(dest))
    boxed = _ram(level=6, screen=0x3A, x=128, y=133)
    boxed[ADDR_ROD] = 1
    _plant_block(boxed, 11, 112, 144)
    halt = make_stairs_3a_ne71_controller()
    halt.step(read_snapshot(leftover))
    halt.step(read_snapshot(leftover))
    halt.walker.last_dir = "DOWN"
    halt.walker.last_xy = (128, 133)
    act = halt.step(read_snapshot(boxed))
    assert halt.failed
    assert act.reason == "occupancy_halt_128_133"
    # After leaving tile 119, occupancy halt at first new miss.
    off119 = _ram(level=6, screen=0x3A, x=144, y=133, tile=116)
    off119[ADDR_ROD] = 1
    _plant_block(off119, 11, 208, 96)
    after = _ne_ctl(phase=Stairs3ANE71Phase.TO_NE, hole_x=112, misses=3)
    after.passed_around = True
    after.walker.last_dir = "UP"
    after.walker.last_xy = (144, 147)
    act = after.step(read_snapshot(off119))
    assert after.failed
    assert act.reason == "occupancy_halt_144_133"
    snap = read_snapshot(leftover)
    assert snap.bow == 0
    assert snap.arrows == 0
    leftover[ADDR_BOW] = 1
    leftover[ADDR_ARROWS] = 1
    armed = read_snapshot(leftover)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-stairs3a-ne71", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_stairs_0x3a_ne71"
    assert "level6-stairs3a-ne71" in L6_THROUGH
    assert L6_THROUGH[L6_THROUGH.index("level6-clear3a") + 1] == (
        "level6-stairs3a-neunder"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-neunder") + 1] == (
        "level6-stairs3a-neclip"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-neclip") + 1] == (
        "level6-stairs3a-ne71"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-ne71") + 1] == (
        "level6-stairs3a-ne"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-ne") + 1] == (
        "level6-stairs3a-71"
    )
    assert L6_THROUGH.index("level6-stairs3a-ne71") < L6_THROUGH.index(
        "level6-stairs3a-ne"
    )
    assert L6_THROUGH.index("level6-stairs3a-ne71") < L6_THROUGH.index(
        "level6-stairs3a"
    )
    assert not hasattr(ctl, "bomb")
