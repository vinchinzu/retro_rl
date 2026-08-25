"""Unit tests for Level 6 play 0x38 west census occupancy."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import INVULN_MOVER_OBJECT_TYPE
from zelda_i.level6_spine import L6_THROUGH
from zelda_i.level6_west38 import (
    EAST_DOOR,
    SOUTH_DOOR_Y,
    WEST_DOOR,
    WEST_SPAWN_XMIN,
    WEST_XMAX,
    level6_west38_stages,
    level6_west38_success,
    make_west38_controller,
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
    ADDR_OBJ_HP,
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x38)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 93)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    return ram


def test_level6_west38_occupancy_to_west_then_halts_next_miss() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_west38_stages()
    assert [name for name, _, _ in stages] == ["level6_west_0x38"]
    leftover = _ram(level=6, screen=0x38, x=120, y=93, keys=4)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    ctl = make_west38_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "to_west"
    # v1 LEFT at y=93 boxed tile 118. v2 DOWN inland 109 first.
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("B", "DOWN"))
    assert list(act.action) != list(nes_action("B"))
    assert not hasattr(ctl, "bomb")
    inland = _ram(level=6, screen=0x38, x=120, y=109, keys=4)
    inland[ADDR_ROD] = 1
    inland[ADDR_BOMBS] = 8
    band = make_west38_controller()
    act = band.step(read_snapshot(inland))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    statue_band = _ram(level=6, screen=0x38, x=120, y=141, keys=4)
    statue_band[ADDR_ROD] = 1
    statue_band[ADDR_BOMBS] = 8
    statue = make_west38_controller()
    act = statue.step(read_snapshot(statue_band))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert not statue.failed
    assert ctl.walker.grid.xmin == WEST_SPAWN_XMIN == 16
    assert ctl.walker.grid.xmax == WEST_XMAX == 120
    trap = _ram(level=6, screen=0x38, x=120, y=93, keys=4)
    trap[ADDR_ROD] = 1
    trap[ADDR_BOMBS] = 8
    trap[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_OBJECT_TYPE
    trap[ADDR_OBJ_HP + 1] = 240
    skip = make_west38_controller()
    act = skip.step(read_snapshot(trap))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("DOWN"))
    assert not any("peel" in n or n.startswith("reclear_") for n in skip.notes)
    mouth = _ram(level=6, screen=0x38, x=120, y=SOUTH_DOOR_Y, keys=4)
    mouth[ADDR_ROD] = 1
    mouth[ADDR_BOMBS] = 8
    mouth[ADDR_COLLIDING_TILE] = 170
    back = make_west38_controller()
    act = back.step(read_snapshot(mouth))
    assert act.reason == "mouth_back"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    wx, wy = WEST_DOOR
    west = _ram(level=6, screen=0x38, x=wx, y=wy, keys=4)
    west[ADDR_ROD] = 1
    west[ADDR_BOMBS] = 8
    pusher = make_west38_controller()
    act = pusher.step(read_snapshot(west))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    ex, ey = EAST_DOOR
    east = _ram(level=6, screen=0x38, x=ex, y=ey, keys=4)
    east[ADDR_ROD] = 1
    east[ADDR_BOMBS] = 8
    east_halt = make_west38_controller()
    act = east_halt.step(read_snapshot(east))
    assert east_halt.failed
    assert act.reason.startswith("occupancy_halt_208_141")
    assert list(act.action) != list(nes_action("RIGHT"))
    mid = _ram(level=6, screen=0x38, x=64, y=93, keys=4)
    mid[ADDR_ROD] = 1
    mid[ADDR_BOMBS] = 8
    halt = make_west38_controller()
    for x in range(16, 121):
        for y in range(77, 206):
            halt.walker.grid.blocked.add((x, y))
    act = halt.step(read_snapshot(mid))
    assert halt.failed
    assert act.reason.startswith("occupancy_halt_64_93")
    dest = _ram(level=6, screen=0x37, x=208, y=141, keys=4)
    dest[ADDR_ROD] = 1
    dest[ADDR_BOMBS] = 8
    arrive = make_west38_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_37"
    assert list(act.action) == list(nes_idle_action())
    assert level6_west38_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x38, x=120, y=93, keys=4)
    still[ADDR_ROD] = 1
    still[ADDR_BOMBS] = 8
    assert not level6_west38_success(read_snapshot(still))
    south = _ram(level=6, screen=0x48, x=120, y=77, keys=4)
    south[ADDR_ROD] = 1
    assert not level6_west38_success(read_snapshot(south))
    south_fail = make_west38_controller()
    south_fail.keys = 4
    act = south_fail.step(read_snapshot(south))
    assert south_fail.failed
    assert act.reason.startswith("south_dated_48")
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    north_fail = make_west38_controller()
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(key_up))
    assert north_fail.failed
    assert act.reason.startswith("key_up_09")
    back28 = _ram(level=6, screen=0x28, x=120, y=205, keys=4)
    back28[ADDR_ROD] = 1
    north = make_west38_controller()
    act = north.step(read_snapshot(back28))
    assert north.failed
    assert act.reason.startswith("backtrack_28")
    east_room = _ram(level=6, screen=0x39, x=16, y=141, keys=4)
    east_room[ADDR_ROD] = 1
    east_fail = make_west38_controller()
    east_fail.keys = 4
    act = east_fail.step(read_snapshot(east_room))
    assert east_fail.failed
    assert act.reason.startswith("east_dated_39")
    assert not level6_west38_success(read_snapshot(east_room))
    cellar = _ram(level=6, screen=0x38, x=120, y=96, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_west38_controller()
    act = warp.step(read_snapshot(cellar))
    assert warp.failed
    assert act.reason.startswith("warped_cellar")
    snap = read_snapshot(leftover)
    assert snap.bow == 0
    assert snap.arrows == 0
    leftover[ADDR_BOW] = 1
    leftover[ADDR_ARROWS] = 1
    armed = read_snapshot(leftover)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-west38", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x38"
    assert "level6-west38" in L6_THROUGH
    assert L6_THROUGH[-9:] == (
        "level6-west38",
        "level6-east38",
        "level6-east38-lane",
        "level6-bomb38-south",
        "level6-south38",
        "level6-clear38-south",
        "level6-aisle28",
        "level6-south28",
        "level6-exit-ow",
    )
