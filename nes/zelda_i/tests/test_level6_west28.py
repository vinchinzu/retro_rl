"""Unit tests for Level 6 play 0x28 west census occupancy."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import (
    INVULN_MOVER_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
)
from zelda_i.level6_spine import L6_THROUGH
from zelda_i.level6_west28 import (
    CLIP_PAST_Y,
    EAST_DOOR,
    SOUTH_DOOR_Y,
    WEST_CLIP_NOOP,
    WEST_DOOR,
    WEST_SPAWN_XMIN,
    WEST_XMAX,
    level6_west28_stages,
    level6_west28_success,
    make_west28_controller,
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


def test_level6_west28_occupancy_to_west_then_halts_on_miss() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_west28_stages()
    assert [name for name, _, _ in stages] == ["level6_west_0x28"]
    leftover = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    ctl = make_west28_controller()
    act = ctl.step(read_snapshot(leftover))
    # v1 LEFT leftover y=77 miss f2; occupancy y=93 halt (88,93) tile 117.
    # LEFT+DOWN clip past north face after that live miss.
    assert act.reason == "diamond_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("B", "DOWN"))
    assert list(act.action) != list(nes_action("B"))
    assert not hasattr(ctl, "bomb")
    assert CLIP_PAST_Y == 109
    assert ctl.walker.grid.xmin == WEST_SPAWN_XMIN == 16
    assert ctl.walker.grid.xmax == WEST_XMAX == 120
    door_band = _ram(level=6, screen=0x28, x=120, y=141, keys=4)
    door_band[ADDR_ROD] = 1
    door_band[ADDR_BOMBS] = 8
    band = make_west28_controller()
    act = band.step(read_snapshot(door_band))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    west_wall = _ram(level=6, screen=0x28, x=32, y=77, keys=4)
    west_wall[ADDR_ROD] = 1
    west_wall[ADDR_BOMBS] = 8
    drop = make_west28_controller()
    act = drop.step(read_snapshot(west_wall))
    assert act.reason == "to_west"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    trap = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    trap[ADDR_ROD] = 1
    trap[ADDR_BOMBS] = 8
    trap[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_OBJECT_TYPE
    trap[ADDR_OBJ_HP + 1] = 240
    skip = make_west28_controller()
    act = skip.step(read_snapshot(trap))
    assert act.reason == "diamond_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert not any("peel" in n or n.startswith("reclear_") for n in skip.notes)
    live = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    live[ADDR_ROD] = 1
    live[ADDR_BOMBS] = 8
    live[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    live[ADDR_OBJ_HP + 1] = 64
    no_reclear = make_west28_controller()
    act = no_reclear.step(read_snapshot(live))
    assert act.reason == "diamond_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert not any(n.startswith("reclear_") for n in no_reclear.notes)
    assert not hasattr(no_reclear, "fighter") or no_reclear.fighter is None
    mouth = _ram(level=6, screen=0x28, x=120, y=SOUTH_DOOR_Y, keys=4)
    mouth[ADDR_ROD] = 1
    mouth[ADDR_BOMBS] = 8
    mouth[ADDR_COLLIDING_TILE] = 170
    back = make_west28_controller()
    act = back.step(read_snapshot(mouth))
    assert act.reason == "mouth_back"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    south_live = _ram(level=6, screen=0x28, x=120, y=SOUTH_DOOR_Y, keys=4)
    south_live[ADDR_ROD] = 1
    south_live[ADDR_BOMBS] = 8
    south_live[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    south_live[ADDR_OBJ_HP + 1] = 64
    south_guard = make_west28_controller()
    act = south_guard.step(read_snapshot(south_live))
    assert act.reason == "mouth_back"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    wx, wy = WEST_DOOR
    west = _ram(level=6, screen=0x28, x=wx, y=wy, keys=4)
    west[ADDR_ROD] = 1
    west[ADDR_BOMBS] = 8
    pusher = make_west28_controller()
    act = pusher.step(read_snapshot(west))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("DOWN"))
    stuck = make_west28_controller()
    west_snap = read_snapshot(west)
    act = stuck.step(west_snap)
    assert act.reason == "west_push"
    assert not stuck.failed
    for _ in range(WEST_CLIP_NOOP - 1):
        act = stuck.step(west_snap)
        assert act.reason == "west_clip"
        assert not stuck.failed
    act = stuck.step(west_snap)
    assert stuck.failed
    assert act.reason.startswith("west_clip_noop_32_141")
    ex, ey = EAST_DOOR
    east = _ram(level=6, screen=0x28, x=ex, y=ey, keys=4)
    east[ADDR_ROD] = 1
    east[ADDR_BOMBS] = 8
    east_halt = make_west28_controller()
    act = east_halt.step(read_snapshot(east))
    assert east_halt.failed
    assert act.reason.startswith("occupancy_halt_208_141")
    assert list(act.action) != list(nes_action("RIGHT"))
    mid = _ram(level=6, screen=0x28, x=64, y=109, keys=4)
    mid[ADDR_ROD] = 1
    mid[ADDR_BOMBS] = 8
    halt = make_west28_controller()
    for x in range(16, 121):
        for y in range(77, 206):
            halt.walker.grid.blocked.add((x, y))
    act = halt.step(read_snapshot(mid))
    assert halt.failed
    assert act.reason.startswith("occupancy_halt_64_109")
    dest = _ram(level=6, screen=0x27, x=208, y=141, keys=4)
    dest[ADDR_ROD] = 1
    dest[ADDR_BOMBS] = 8
    arrive = make_west28_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_27"
    assert list(act.action) == list(nes_idle_action())
    assert level6_west28_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    still[ADDR_ROD] = 1
    still[ADDR_BOMBS] = 8
    assert not level6_west28_success(read_snapshot(still))
    south = _ram(level=6, screen=0x38, x=120, y=77, keys=4)
    south[ADDR_ROD] = 1
    assert not level6_west28_success(read_snapshot(south))
    south_fail = make_west28_controller()
    south_fail.keys = 4
    act = south_fail.step(read_snapshot(south))
    assert south_fail.failed
    assert act.reason.startswith("south_trap_38")
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    north_fail = make_west28_controller()
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(key_up))
    assert north_fail.failed
    assert act.reason.startswith("key_up_09")
    back18 = _ram(level=6, screen=0x18, x=120, y=205, keys=4)
    back18[ADDR_ROD] = 1
    north = make_west28_controller()
    act = north.step(read_snapshot(back18))
    assert north.failed
    assert act.reason.startswith("backtrack_18")
    cellar = _ram(level=6, screen=0x28, x=120, y=96, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_west28_controller()
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
    run = SpineRun(through="level6-west28", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x28"
    assert "level6-west28" in L6_THROUGH
    assert L6_THROUGH[L6_THROUGH.index("level6-south18") + 1] == "level6-aisle-west28"
    assert L6_THROUGH[L6_THROUGH.index("level6-aisle-west28") + 1] == "level6-west28"
    assert L6_THROUGH[L6_THROUGH.index("level6-west28") + 1] == "level6-east28"
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
