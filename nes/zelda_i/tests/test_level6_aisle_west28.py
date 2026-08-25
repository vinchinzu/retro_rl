"""Unit tests for Level 6 play 0x28 west-aisle then west occupancy."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import (
    INVULN_MOVER_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
)
from zelda_i.level6_spine import L6_THROUGH
from zelda_i.level6_aisle_west28 import (
    AISLE_X,
    AISLE_Y,
    EAST_DOOR,
    NORTH_MOUTH,
    NORTH_MOUTH_Y,
    SOUTH_DOOR_Y,
    WEST_CLIP_NOOP,
    WEST_DOOR,
    WEST_SPAWN_XMIN,
    WEST_XMAX,
    level6_aisle_west28_stages,
    level6_aisle_west28_success,
    make_aisle_west28_controller,
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x28)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 77)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    return ram


def test_level6_aisle_west28_occupancy_down_then_left_halts_on_miss() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_aisle_west28_stages()
    assert [name for name, _, _ in stages] == ["level6_aisle_west_0x28"]
    leftover = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    ctl = make_aisle_west28_controller()
    act = ctl.step(read_snapshot(leftover))
    # 1-frame DOWN off north mouth; not occupancy-DOWN (v1 2px miss).
    assert NORTH_MOUTH == (120, NORTH_MOUTH_Y) == (120, 77)
    assert act.reason == "mouth_step"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("B", "DOWN"))
    assert list(act.action) != list(nes_action("B"))
    assert not hasattr(ctl, "bomb")
    assert not ctl.failed
    assert AISLE_X == 64
    assert AISLE_Y == 141
    assert ctl.aisle == (AISLE_X, AISLE_Y)
    assert ctl.goal == WEST_DOOR
    assert ctl.walker.grid.xmin == WEST_SPAWN_XMIN == 16
    assert ctl.walker.grid.xmax == WEST_XMAX == 120
    inland = _ram(level=6, screen=0x28, x=120, y=79, keys=4)
    inland[ADDR_ROD] = 1
    inland[ADDR_BOMBS] = 8
    inland_snap = read_snapshot(inland)
    act = ctl.step(inland_snap)
    # Occupancy from dated miss pose (120,79) to (64,141): DOWN, not LEFT y=93.
    assert act.reason == "to_aisle"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert not ctl.failed
    # v2 2px overshoot (120,79)→(120,81) is leftover startup, not a wall.
    over = _ram(level=6, screen=0x28, x=120, y=81, keys=4)
    over[ADDR_ROD] = 1
    over[ADDR_BOMBS] = 8
    over_snap = read_snapshot(over)
    act = ctl.step(over_snap)
    assert not ctl.failed
    assert act.reason == "to_aisle"
    assert list(act.action) == list(nes_action("DOWN"))
    assert any(n.startswith("overshoot_f") for n in ctl.notes)
    act = ctl.step(over_snap)
    assert ctl.failed
    assert act.reason.startswith("occupancy_halt_120_81")
    stuck = make_aisle_west28_controller()
    act = stuck.step(read_snapshot(leftover))
    assert act.reason == "mouth_step"
    act = stuck.step(inland_snap)
    assert act.reason == "to_aisle"
    assert not stuck.failed
    act = stuck.step(inland_snap)
    assert stuck.failed
    assert act.reason.startswith("occupancy_halt_120_79")
    door_band = _ram(level=6, screen=0x28, x=120, y=AISLE_Y, keys=4)
    door_band[ADDR_ROD] = 1
    door_band[ADDR_BOMBS] = 8
    band = make_aisle_west28_controller()
    act = band.step(read_snapshot(door_band))
    assert act.reason == "to_aisle"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    north_diamond = _ram(level=6, screen=0x28, x=96, y=109, keys=4)
    north_diamond[ADDR_ROD] = 1
    north_diamond[ADDR_BOMBS] = 8
    south_first = make_aisle_west28_controller()
    act = south_first.step(read_snapshot(north_diamond))
    assert act.reason == "to_aisle"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    aisle = _ram(level=6, screen=0x28, x=AISLE_X, y=AISLE_Y, keys=4)
    aisle[ADDR_ROD] = 1
    aisle[ADDR_BOMBS] = 8
    west = make_aisle_west28_controller()
    act = west.step(read_snapshot(aisle))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    trap = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    trap[ADDR_ROD] = 1
    trap[ADDR_BOMBS] = 8
    trap[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_OBJECT_TYPE
    trap[ADDR_OBJ_HP + 1] = 240
    skip = make_aisle_west28_controller()
    act = skip.step(read_snapshot(trap))
    assert act.reason == "mouth_step"
    assert list(act.action) == list(nes_action("DOWN"))
    assert not any("peel" in n or n.startswith("reclear_") for n in skip.notes)
    live = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    live[ADDR_ROD] = 1
    live[ADDR_BOMBS] = 8
    live[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    live[ADDR_OBJ_HP + 1] = 64
    no_reclear = make_aisle_west28_controller()
    act = no_reclear.step(read_snapshot(live))
    assert act.reason == "mouth_step"
    assert list(act.action) == list(nes_action("DOWN"))
    assert not any(n.startswith("reclear_") for n in no_reclear.notes)
    assert not hasattr(no_reclear, "fighter") or no_reclear.fighter is None
    mouth = _ram(level=6, screen=0x28, x=120, y=SOUTH_DOOR_Y, keys=4)
    mouth[ADDR_ROD] = 1
    mouth[ADDR_BOMBS] = 8
    mouth[ADDR_COLLIDING_TILE] = 170
    back = make_aisle_west28_controller()
    act = back.step(read_snapshot(mouth))
    assert act.reason == "mouth_back"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    south_live = _ram(level=6, screen=0x28, x=120, y=SOUTH_DOOR_Y, keys=4)
    south_live[ADDR_ROD] = 1
    south_live[ADDR_BOMBS] = 8
    south_live[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    south_live[ADDR_OBJ_HP + 1] = 64
    south_guard = make_aisle_west28_controller()
    act = south_guard.step(read_snapshot(south_live))
    assert act.reason == "mouth_back"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    wx, wy = WEST_DOOR
    west_mouth = _ram(level=6, screen=0x28, x=wx, y=wy, keys=4)
    west_mouth[ADDR_ROD] = 1
    west_mouth[ADDR_BOMBS] = 8
    pusher = make_aisle_west28_controller()
    act = pusher.step(read_snapshot(west_mouth))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("DOWN"))
    clipper = make_aisle_west28_controller()
    west_snap = read_snapshot(west_mouth)
    act = clipper.step(west_snap)
    assert act.reason == "west_push"
    assert not clipper.failed
    for _ in range(WEST_CLIP_NOOP - 1):
        act = clipper.step(west_snap)
        assert act.reason == "west_clip"
        assert not clipper.failed
    act = clipper.step(west_snap)
    assert clipper.failed
    assert act.reason.startswith("west_clip_noop_32_141")
    ex, ey = EAST_DOOR
    east = _ram(level=6, screen=0x28, x=ex, y=ey, keys=4)
    east[ADDR_ROD] = 1
    east[ADDR_BOMBS] = 8
    east_halt = make_aisle_west28_controller()
    act = east_halt.step(read_snapshot(east))
    assert east_halt.failed
    assert act.reason.startswith("occupancy_halt_208_141")
    assert list(act.action) != list(nes_action("RIGHT"))
    mid = _ram(level=6, screen=0x28, x=64, y=109, keys=4)
    mid[ADDR_ROD] = 1
    mid[ADDR_BOMBS] = 8
    halt = make_aisle_west28_controller()
    for x in range(16, 121):
        for y in range(77, 182):
            halt.walker.grid.blocked.add((x, y))
    act = halt.step(read_snapshot(mid))
    assert halt.failed
    assert act.reason.startswith("occupancy_halt_64_109")
    dest = _ram(level=6, screen=0x27, x=208, y=141, keys=4)
    dest[ADDR_ROD] = 1
    dest[ADDR_BOMBS] = 8
    arrive = make_aisle_west28_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_27"
    assert list(act.action) == list(nes_idle_action())
    assert level6_aisle_west28_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    still[ADDR_ROD] = 1
    still[ADDR_BOMBS] = 8
    assert not level6_aisle_west28_success(read_snapshot(still))
    south = _ram(level=6, screen=0x38, x=120, y=77, keys=4)
    south[ADDR_ROD] = 1
    assert not level6_aisle_west28_success(read_snapshot(south))
    south_fail = make_aisle_west28_controller()
    south_fail.keys = 4
    act = south_fail.step(read_snapshot(south))
    assert south_fail.failed
    assert act.reason.startswith("south_trap_38")
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    north_fail = make_aisle_west28_controller()
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(key_up))
    assert north_fail.failed
    assert act.reason.startswith("key_up_09")
    back18 = _ram(level=6, screen=0x18, x=120, y=205, keys=4)
    back18[ADDR_ROD] = 1
    north = make_aisle_west28_controller()
    act = north.step(read_snapshot(back18))
    assert north.failed
    assert act.reason.startswith("backtrack_18")
    cellar = _ram(level=6, screen=0x28, x=120, y=96, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_aisle_west28_controller()
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
    run = SpineRun(through="level6-aisle-west28", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_aisle_west_0x28"
    assert "level6-aisle-west28" in L6_THROUGH
    assert L6_THROUGH[L6_THROUGH.index("level6-south18") + 1] == "level6-aisle-west28"
    assert L6_THROUGH[L6_THROUGH.index("level6-aisle-west28") + 1] == "level6-west28"
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
