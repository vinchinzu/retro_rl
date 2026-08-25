"""Unit tests for Level 6 reclear 0x39 then occupancy y=141 LEFT."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import INVULN_MOVER_OBJECT_TYPE, VIRE_OBJECT_TYPE
from zelda_i.level6_clear39_west import (
    DATED_DOWN,
    DATED_LEFT,
    DATED_LEFT2,
    LANE_Y,
    WEST_DOOR,
    WEST_SPAWN_XMIN,
    level6_clear39_west_stages,
    level6_clear39_west_success,
    make_clear39_west_controller,
)
from zelda_i.level6_path import WEST_CLIP_X
from zelda_i.level6_spine import L6_THROUGH
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", 144)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    return ram


def test_level6_clear39_west_reuses_v3_enter_then_y141_left() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_clear39_west_stages()
    assert [name for name, _, _ in stages] == ["level6_clear39_west_0x39"]
    leftover = _ram(level=6, screen=0x3A, x=144, y=141, keys=4)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    ctl = make_clear39_west_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "left_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("B"))
    assert not hasattr(ctl, "bomb")
    assert not ctl.failed
    assert ctl.room == 0x3A
    assert ctl._goal() == (WEST_CLIP_X, LANE_Y)
    assert ctl.walker.grid.xmin == WEST_SPAWN_XMIN == 16
    replan = make_clear39_west_controller()
    act = replan.step(read_snapshot(leftover))
    assert act.reason == "left_path"
    act = replan.step(read_snapshot(leftover))
    assert not replan.failed
    assert any(n.startswith("miss_f2_LEFT_144_141") for n in replan.notes)
    assert act.reason == "left_path"
    assert list(act.action) != list(nes_idle_action())
    assert list(act.action) != list(nes_action("LEFT"))
    for _ in range(20):
        act = replan.step(read_snapshot(leftover))
        if act.reason == "occupancy_stand":
            break
    assert act.reason == "occupancy_stand"
    assert not replan.failed
    assert list(act.action) == list(nes_idle_action())
    dest39 = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    dest39[ADDR_ROD] = 1
    dest39[ADDR_BOMBS] = 8
    act = ctl.step(read_snapshot(dest39))
    assert not ctl.failed
    assert not ctl.success
    assert ctl.room == 0x39
    assert any(n.startswith("arrived_39_") for n in ctl.notes)
    assert act.reason == "room_settle"
    assert ctl.walker.grid.ymin == LANE_Y
    assert ctl.walker.grid.ymax == LANE_Y
    vires = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    vires[ADDR_ROD] = 1
    vires[ADDR_BOMBS] = 8
    vires[ADDR_OBJ_TYPE + 1] = VIRE_OBJECT_TYPE
    vires[ADDR_OBJ_HP + 1] = 144
    reclear = make_clear39_west_controller()
    reclear.room = 0x39
    reclear.keys = 4
    act = reclear.step(read_snapshot(vires))
    assert any(n.startswith("reclear_39_") for n in reclear.notes)
    assert reclear.fighter is not None
    assert act.reason != "north_push"
    assert list(act.action) != list(nes_action("UP"))
    assert not reclear.success
    empty39 = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    empty39[ADDR_ROD] = 1
    empty39[ADDR_BOMBS] = 8
    west = make_clear39_west_controller()
    west.room = 0x39
    west.keys = 4
    west.walker = make_clear39_west_controller().walker
    west._arrive_39(read_snapshot(empty39))
    act = west.step(read_snapshot(empty39))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert west._goal() == WEST_DOOR
    assert west.walker.grid.ymin == LANE_Y
    assert not west.failed
    dated = _ram(level=6, screen=0x39, x=DATED_DOWN[0], y=DATED_DOWN[1], keys=4)
    dated[ADDR_ROD] = 1
    dated[ADDR_COLLIDING_TILE] = 118
    clip = make_clear39_west_controller()
    clip.room = 0x39
    clip.keys = 4
    clip._arrive_39(read_snapshot(dated))
    act = clip.step(read_snapshot(dated))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert not clip.failed
    act = clip.step(read_snapshot(dated))
    assert act.reason == "west_clip"
    assert not clip.failed
    assert "west_clip" in clip.notes
    dated_left = _ram(
        level=6, screen=0x39, x=DATED_LEFT[0], y=DATED_LEFT[1], keys=4
    )
    dated_left[ADDR_ROD] = 1
    dated_left[ADDR_COLLIDING_TILE] = 119
    lane = make_clear39_west_controller()
    lane.room = 0x39
    lane.keys = 4
    lane._arrive_39(read_snapshot(dated_left))
    act = lane.step(read_snapshot(dated_left))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert not lane.failed
    act = lane.step(read_snapshot(dated_left))
    assert act.reason == "west_clip"
    assert not lane.failed
    past = _ram(level=6, screen=0x39, x=140, y=141, keys=4)
    past[ADDR_ROD] = 1
    act = lane.step(read_snapshot(past))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert not lane.failed
    act = lane.step(read_snapshot(past))
    assert lane.failed
    assert act.reason.startswith("occupancy_halt_140_141")
    assert list(act.action) == list(nes_idle_action())
    dated_left2 = _ram(
        level=6, screen=0x39, x=DATED_LEFT2[0], y=DATED_LEFT2[1], keys=4
    )
    dated_left2[ADDR_ROD] = 1
    dated_left2[ADDR_COLLIDING_TILE] = 119
    lane2 = make_clear39_west_controller()
    lane2.room = 0x39
    lane2.keys = 4
    lane2._arrive_39(read_snapshot(dated_left2))
    act = lane2.step(read_snapshot(dated_left2))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert not lane2.failed
    act = lane2.step(read_snapshot(dated_left2))
    assert act.reason == "west_clip"
    assert not lane2.failed
    past2 = _ram(level=6, screen=0x39, x=136, y=141, keys=4)
    past2[ADDR_ROD] = 1
    act = lane2.step(read_snapshot(past2))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert not lane2.failed
    act = lane2.step(read_snapshot(past2))
    assert lane2.failed
    assert act.reason.startswith("occupancy_halt_136_141")
    stuck = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    stuck[ADDR_ROD] = 1
    stuck[ADDR_COLLIDING_TILE] = 118
    halt = make_clear39_west_controller()
    halt.room = 0x39
    halt.keys = 4
    halt._arrive_39(read_snapshot(stuck))
    act = halt.step(read_snapshot(stuck))
    assert act.reason == "west_lane"
    assert not halt.failed
    act = halt.step(read_snapshot(stuck))
    assert halt.failed
    assert act.reason.startswith("occupancy_halt_208_141")
    assert list(act.action) == list(nes_idle_action())
    door = _ram(level=6, screen=0x39, x=32, y=141, keys=4)
    door[ADDR_ROD] = 1
    push = make_clear39_west_controller()
    push.room = 0x39
    push.keys = 4
    push._arrive_39(read_snapshot(door))
    act = push.step(read_snapshot(door))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    mouth = _ram(level=6, screen=0x3A, x=32, y=93, keys=4)
    mouth[ADDR_ROD] = 1
    align = make_clear39_west_controller()
    act = align.step(read_snapshot(mouth))
    assert act.reason == "west_align"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert not align.failed
    invuln = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    invuln[ADDR_ROD] = 1
    invuln[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_OBJECT_TYPE
    invuln[ADDR_OBJ_HP + 1] = 240
    ignore = make_clear39_west_controller()
    ignore.room = 0x39
    ignore.keys = 4
    ignore._arrive_39(read_snapshot(invuln))
    act = ignore.step(read_snapshot(invuln))
    assert ignore.fighter is None
    assert act.reason == "west_lane"
    ram_dest = _ram(level=6, screen=0x38, x=208, y=141, keys=4)
    ram_dest[ADDR_ROD] = 1
    arrive = make_clear39_west_controller()
    arrive.room = 0x39
    arrive.keys = 4
    act = arrive.step(read_snapshot(ram_dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_38"
    assert list(act.action) == list(nes_idle_action())
    assert level6_clear39_west_success(read_snapshot(ram_dest))
    north = _ram(level=6, screen=0x29, x=120, y=205, keys=4)
    north[ADDR_ROD] = 1
    north_fail = make_clear39_west_controller()
    north_fail.room = 0x39
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(north))
    assert north_fail.failed
    assert act.reason.startswith("north_29_")
    assert not level6_clear39_west_success(read_snapshot(north))
    still = _ram(level=6, screen=0x3A, x=144, y=141, keys=4)
    still[ADDR_ROD] = 1
    assert not level6_clear39_west_success(read_snapshot(still))
    via = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    via[ADDR_ROD] = 1
    assert not level6_clear39_west_success(read_snapshot(via))
    back = _ram(level=6, screen=0x3A, x=16, y=141, keys=4)
    back[ADDR_ROD] = 1
    back_fail = make_clear39_west_controller()
    back_fail.room = 0x39
    back_fail.keys = 4
    act = back_fail.step(read_snapshot(back))
    assert back_fail.failed
    assert act.reason.startswith("backtrack_3a_")
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    up_fail = make_clear39_west_controller()
    up_fail.room = 0x39
    up_fail.keys = 4
    act = up_fail.step(read_snapshot(key_up))
    assert up_fail.failed
    assert act.reason.startswith("key_up_09_")
    cellar = _ram(level=6, screen=0x3A, x=144, y=141, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_clear39_west_controller()
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
    run = SpineRun(through="level6-clear39-west", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_clear39_west_0x39"
    assert "level6-clear39-west" in L6_THROUGH
    assert L6_THROUGH[L6_THROUGH.index("level6-clear3a") + 1] == "level6-west39-upclip"
    assert L6_THROUGH[L6_THROUGH.index("level6-west39-upclip") + 1] == "level6-west39"
    assert L6_THROUGH[L6_THROUGH.index("level6-west39") + 1] == "level6-clear39-west"
    assert L6_THROUGH[L6_THROUGH.index("level6-clear39-west") + 1] == "level6-stairs3a"
    assert "level6-north39" not in L6_THROUGH[: L6_THROUGH.index("level6-clear39-west")]
    assert L6_THROUGH[L6_THROUGH.index("level6-clear39")] != "level6-clear39-west"
