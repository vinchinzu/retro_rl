"""Unit tests for Level 6 reclear 0x39 then DOWN onto y=141 at (125,133)."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import INVULN_MOVER_OBJECT_TYPE, VIRE_OBJECT_TYPE
from zelda_i.level6_path import WEST_CLIP_X
from zelda_i.level6_spine import L6_THROUGH
from zelda_i.level6_west39_reband import (
    DATED_DOWN,
    DATED_LEFT,
    DATED_LEFT2,
    DATED_LEFT3,
    DATED_LEFT4,
    DATED_LEFT5,
    DATED_LEFT6,
    LANE_Y,
    WEST_DOOR,
    WEST_SPAWN_XMIN,
    level6_west39_reband_stages,
    level6_west39_reband_success,
    make_west39_reband_controller,
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", 144)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    return ram


def test_level6_west39_reband_reuses_prefix_then_down_at_125_133() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_west39_reband_stages()
    assert [name for name, _, _ in stages] == ["level6_west39_reband_0x39"]
    leftover = _ram(level=6, screen=0x3A, x=144, y=141, keys=4)
    leftover[ADDR_ROD] = 1
    leftover[ADDR_BOMBS] = 8
    ctl = make_west39_reband_controller()
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
    replan = make_west39_reband_controller()
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
    reclear = make_west39_reband_controller()
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
    west = make_west39_reband_controller()
    west.room = 0x39
    west.keys = 4
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
    clip = make_west39_reband_controller()
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
    lane = make_west39_reband_controller()
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
    dated_left2 = _ram(
        level=6, screen=0x39, x=DATED_LEFT2[0], y=DATED_LEFT2[1], keys=4
    )
    dated_left2[ADDR_ROD] = 1
    dated_left2[ADDR_COLLIDING_TILE] = 119
    lane2 = make_west39_reband_controller()
    lane2.room = 0x39
    lane2.keys = 4
    lane2._arrive_39(read_snapshot(dated_left2))
    act = lane2.step(read_snapshot(dated_left2))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert not lane2.failed
    dated_left3 = _ram(
        level=6, screen=0x39, x=DATED_LEFT3[0], y=DATED_LEFT3[1], keys=4
    )
    dated_left3[ADDR_ROD] = 1
    dated_left3[ADDR_COLLIDING_TILE] = 117
    upclip = make_west39_reband_controller()
    upclip.room = 0x39
    upclip.keys = 4
    upclip._arrive_39(read_snapshot(dated_left3))
    act = upclip.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert not upclip.failed
    assert upclip.upclipped
    assert "west_upclip" in upclip.notes
    act = upclip.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert not upclip.failed
    band = _ram(level=6, screen=0x39, x=133, y=138, keys=4)
    band[ADDR_ROD] = 1
    act = upclip.step(read_snapshot(band))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert upclip.walker.grid.ymin == 138
    assert upclip.walker.grid.ymax == 138
    assert upclip._goal() == (WEST_DOOR[0], 138)
    assert "upclip_band_138" in upclip.notes
    assert not upclip.failed
    act = upclip.step(read_snapshot(band))
    assert upclip.failed
    assert act.reason.startswith("occupancy_halt_133_138")
    assert list(act.action) == list(nes_idle_action())
    dated_left4 = _ram(
        level=6, screen=0x39, x=DATED_LEFT4[0], y=DATED_LEFT4[1], keys=4
    )
    dated_left4[ADDR_ROD] = 1
    dated_left4[ADDR_COLLIDING_TILE] = 116
    clip4 = make_west39_reband_controller()
    clip4.room = 0x39
    clip4.keys = 4
    clip4._arrive_39(read_snapshot(dated_left3))
    act = clip4.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    assert clip4.upclipped
    act = clip4.step(read_snapshot(dated_left4))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert not clip4.failed
    assert "west_clip" in clip4.notes
    act = clip4.step(read_snapshot(dated_left4))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert not clip4.failed
    new_band = _ram(level=6, screen=0x39, x=130, y=136, keys=4)
    new_band[ADDR_ROD] = 1
    act = clip4.step(read_snapshot(new_band))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert clip4.walker.grid.ymin == 136
    assert clip4.walker.grid.ymax == 136
    assert clip4._goal() == (WEST_DOOR[0], 136)
    assert "upclip_band_136" in clip4.notes
    assert not clip4.failed
    act = clip4.step(read_snapshot(new_band))
    assert clip4.failed
    assert act.reason.startswith("occupancy_halt_130_136")
    assert list(act.action) == list(nes_idle_action())
    dated_left5 = _ram(
        level=6, screen=0x39, x=DATED_LEFT5[0], y=DATED_LEFT5[1], keys=4
    )
    dated_left5[ADDR_ROD] = 1
    dated_left5[ADDR_COLLIDING_TILE] = 116
    clip5 = make_west39_reband_controller()
    clip5.room = 0x39
    clip5.keys = 4
    clip5._arrive_39(read_snapshot(dated_left3))
    act = clip5.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    assert clip5.upclipped
    act = clip5.step(read_snapshot(dated_left4))
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert not clip5.failed
    act = clip5.step(read_snapshot(dated_left5))
    assert act.reason == "west_upclip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert not clip5.failed
    assert "west_upclip" in clip5.notes
    act = clip5.step(read_snapshot(dated_left5))
    assert act.reason == "west_upclip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert not clip5.failed
    new_band5 = _ram(level=6, screen=0x39, x=127, y=130, keys=4)
    new_band5[ADDR_ROD] = 1
    act = clip5.step(read_snapshot(new_band5))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert clip5.walker.grid.ymin == 130
    assert clip5.walker.grid.ymax == 130
    assert clip5._goal() == (WEST_DOOR[0], 130)
    assert "upclip_band_130" in clip5.notes
    assert not clip5.failed
    act = clip5.step(read_snapshot(new_band5))
    assert clip5.failed
    assert act.reason.startswith("occupancy_halt_127_130")
    assert list(act.action) == list(nes_idle_action())
    dated_left6 = _ram(
        level=6, screen=0x39, x=DATED_LEFT6[0], y=DATED_LEFT6[1], keys=4
    )
    dated_left6[ADDR_ROD] = 1
    dated_left6[ADDR_COLLIDING_TILE] = 118
    reband = make_west39_reband_controller()
    reband.room = 0x39
    reband.keys = 4
    reband._arrive_39(read_snapshot(dated_left3))
    act = reband.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    act = reband.step(read_snapshot(dated_left4))
    assert act.reason == "west_clip"
    act = reband.step(read_snapshot(dated_left5))
    assert act.reason == "west_upclip"
    act = reband.step(read_snapshot(dated_left6))
    assert act.reason == "west_reband"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("UP"))
    assert not reband.failed
    assert reband.rebanded
    assert "west_reband" in reband.notes
    assert any(n.startswith("reband_125_133_tile=") for n in reband.notes)
    act = reband.step(read_snapshot(dated_left6))
    assert reband.failed
    assert act.reason.startswith("occupancy_halt_125_133")
    assert list(act.action) == list(nes_idle_action())
    lane141 = make_west39_reband_controller()
    lane141.room = 0x39
    lane141.keys = 4
    lane141._arrive_39(read_snapshot(dated_left3))
    act = lane141.step(read_snapshot(dated_left3))
    assert act.reason == "west_upclip"
    act = lane141.step(read_snapshot(dated_left4))
    assert act.reason == "west_clip"
    act = lane141.step(read_snapshot(dated_left5))
    assert act.reason == "west_upclip"
    act = lane141.step(read_snapshot(dated_left6))
    assert act.reason == "west_reband"
    assert not lane141.failed
    door_band = _ram(level=6, screen=0x39, x=125, y=LANE_Y, keys=4)
    door_band[ADDR_ROD] = 1
    act = lane141.step(read_snapshot(door_band))
    assert act.reason == "west_lane"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert lane141.walker.grid.ymin == LANE_Y
    assert lane141.walker.grid.ymax == LANE_Y
    assert lane141._goal() == WEST_DOOR
    assert "reband_lane_141" in lane141.notes
    assert not lane141.failed
    act = lane141.step(read_snapshot(door_band))
    assert lane141.failed
    assert act.reason.startswith("occupancy_halt_125_141")
    assert list(act.action) == list(nes_idle_action())
    direct = _ram(
        level=6, screen=0x39, x=DATED_LEFT6[0], y=DATED_LEFT6[1], keys=4
    )
    direct[ADDR_ROD] = 1
    direct[ADDR_COLLIDING_TILE] = 118
    drop = make_west39_reband_controller()
    drop.room = 0x39
    drop.keys = 4
    drop._arrive_39(read_snapshot(direct))
    act = drop.step(read_snapshot(direct))
    assert act.reason == "west_reband"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    assert not drop.failed
    stuck = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    stuck[ADDR_ROD] = 1
    stuck[ADDR_COLLIDING_TILE] = 118
    halt = make_west39_reband_controller()
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
    push = make_west39_reband_controller()
    push.room = 0x39
    push.keys = 4
    push._arrive_39(read_snapshot(door))
    act = push.step(read_snapshot(door))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    mouth = _ram(level=6, screen=0x3A, x=32, y=93, keys=4)
    mouth[ADDR_ROD] = 1
    align = make_west39_reband_controller()
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
    ignore = make_west39_reband_controller()
    ignore.room = 0x39
    ignore.keys = 4
    ignore._arrive_39(read_snapshot(invuln))
    act = ignore.step(read_snapshot(invuln))
    assert ignore.fighter is None
    assert act.reason == "west_lane"
    ram_dest = _ram(level=6, screen=0x38, x=208, y=141, keys=4)
    ram_dest[ADDR_ROD] = 1
    arrive = make_west39_reband_controller()
    arrive.room = 0x39
    arrive.keys = 4
    act = arrive.step(read_snapshot(ram_dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_38"
    assert list(act.action) == list(nes_idle_action())
    assert level6_west39_reband_success(read_snapshot(ram_dest))
    north = _ram(level=6, screen=0x29, x=120, y=205, keys=4)
    north[ADDR_ROD] = 1
    north_fail = make_west39_reband_controller()
    north_fail.room = 0x39
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(north))
    assert north_fail.failed
    assert act.reason.startswith("north_29_")
    assert not level6_west39_reband_success(read_snapshot(north))
    still = _ram(level=6, screen=0x3A, x=144, y=141, keys=4)
    still[ADDR_ROD] = 1
    assert not level6_west39_reband_success(read_snapshot(still))
    via = _ram(level=6, screen=0x39, x=208, y=141, keys=4)
    via[ADDR_ROD] = 1
    assert not level6_west39_reband_success(read_snapshot(via))
    back = _ram(level=6, screen=0x3A, x=16, y=141, keys=4)
    back[ADDR_ROD] = 1
    back_fail = make_west39_reband_controller()
    back_fail.room = 0x39
    back_fail.keys = 4
    act = back_fail.step(read_snapshot(back))
    assert back_fail.failed
    assert act.reason.startswith("backtrack_3a_")
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    up_fail = make_west39_reband_controller()
    up_fail.room = 0x39
    up_fail.keys = 4
    act = up_fail.step(read_snapshot(key_up))
    assert up_fail.failed
    assert act.reason.startswith("key_up_09_")
    cellar = _ram(level=6, screen=0x3A, x=144, y=141, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_west39_reband_controller()
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
    run = SpineRun(through="level6-west39-reband", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west39_reband_0x39"
    assert "level6-west39-reband" in L6_THROUGH
    assert L6_THROUGH[L6_THROUGH.index("level6-clear3a") + 1] == (
        "level6-stairs3a-ne71"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-ne71") + 1] == (
        "level6-stairs3a-ne"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-ne") + 1] == (
        "level6-stairs3a-71"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-stairs3a-71") + 1] == (
        "level6-west39-reband"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-west39-reband") + 1] == (
        "level6-west39-upclip"
    )
    assert L6_THROUGH[L6_THROUGH.index("level6-west39-upclip") + 1] == "level6-west39"
    assert "level6-north39" not in L6_THROUGH[: L6_THROUGH.index("level6-west39-reband")]
    assert L6_THROUGH[L6_THROUGH.index("level6-clear39")] != "level6-west39-reband"
