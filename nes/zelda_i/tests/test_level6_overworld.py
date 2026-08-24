"""Unit tests for Level 6 overworld hops, L5 settle, and spine stages."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_overworld import (
    HILLS_AISLE_X,
    HILLS_CHANNEL_Y_HI,
    HILLS_CHANNEL_Y_LO,
    HILLS_NOTCH_X,
    HILLS_ROCK_X,
    LEVEL6_POST_L5_SCREENS,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_TO_LEVEL6_HOPS,
    SCREEN_POST_L5_RETURN,
    OverworldToLevel6Controller,
    PostL5SettlePhase,
    PostL5TriforceSettleController,
    level6_hops_from,
    lost_hills_west_dir,
    make_post_l5_level6_controller,
    post_l5_overworld_ready,
)
from zelda_i.level6_path import (
    BLOCK_OBJECT_TYPE,
    SETTLE_18_IDLE_FRAMES,
    Level6North68Controller,
    left_block_0x68,
    make_north_18_controller,
    make_north_28_controller,
    make_north_38_controller,
    make_north_48_controller,
    make_north_58_controller,
    make_settle_18_controller,
    south_face_stand,
)
from zelda_i.level6_spine import (
    L6_THROUGH,
    Level6Return79Controller,
    level6_clear28_stages,
    level6_clear28_success,
    level6_clear38_stages,
    level6_clear38_success,
    level6_room18_stages,
    level6_room18_success,
    level6_settle18_stages,
    level6_settle18_success,
    level6_gleeok18_stages,
    level6_gleeok18_success,
    level6_postgleeok18_stages,
    level6_postgleeok18_success,
    level6_stairs18_stages,
    level6_stairs18_success,
    level6_room28_stages,
    level6_room28_success,
    level6_clear58_stages,
    level6_clear58_success,
    level6_clear68_stages,
    level6_clear68_success,
    level6_compass_stages,
    level6_compass_success,
    level6_keese_stages,
    level6_keese_success,
    level6_room38_stages,
    level6_room38_success,
    level6_room48_stages,
    level6_room48_success,
    level6_east_key_stages,
    level6_east_key_success,
    level6_entry_stages,
    level6_entry_success,
    level6_west_stages,
    level6_west_success,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_COMPASS,
    ADDR_CUR_OPENED_DOORS,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_RAFT,
    ADDR_ROOM_ITEM_ID,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    ADDR_KEYS,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x0B)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_RAFT] = fields.get("raft", 1)
    ram[ADDR_LADDER] = fields.get("ladder", 1)
    ram[ADDR_WHISTLE] = fields.get("whistle", 1)
    ram[ADDR_KEYS] = fields.get("keys", 5)
    return ram


def test_post_l5_path_is_contiguous_and_skips_lost_hills_south() -> None:
    assert SCREEN_POST_L5_RETURN == 0x0B
    assert LEVEL6_POST_L5_SCREENS[0] == 0x0B
    assert LEVEL6_POST_L5_SCREENS[1] == 0x1B
    assert LEVEL6_POST_L5_SCREENS[-1] == 0x22
    assert [h.target for h in POST_L5_TO_LEVEL6_HOPS[:3]] == [0x1B, 0x1A, 0x19]
    assert POST_L5_TO_LEVEL6_HOPS[0].direction == "DOWN"
    assert POST_L5_TO_LEVEL6_HOPS[1].direction == "LEFT"
    assert POST_L5_TO_LEVEL6_HOPS[1].align_y == 141
    assert POST_L5_TO_LEVEL6_HOPS[2].target == 0x19
    assert POST_L5_TO_LEVEL6_HOPS[2].align_y == 141
    assert all(
        h.align_y == 141
        for h in POST_L5_TO_LEVEL6_HOPS[1:7]
        if h.direction == "LEFT"
    )
    assert POST_L5_TO_LEVEL6_HOPS[7].target == 0x14
    assert POST_L5_TO_LEVEL6_HOPS[7].y_band == (165, 189)
    assert POST_L5_TO_LEVEL6_HOPS[8].target == 0x24
    assert POST_L5_TO_LEVEL6_HOPS[8].direction == "DOWN"
    assert POST_L5_TO_LEVEL6_HOPS[8].align_x == 160
    assert POST_L5_TO_LEVEL6_HOPS[10].target == 0x33
    assert POST_L5_TO_LEVEL6_HOPS[10].align_x == 208
    assert POST_L5_TO_LEVEL6_HOPS[-1].target == 0x22
    assert POST_L5_TO_LEVEL6_HOPS[-1].direction == "UP"
    assert len(POST_L5_TO_LEVEL6_HOPS) == len(LEVEL6_POST_L5_SCREENS) - 1
    for before, after in zip(LEVEL6_POST_L5_SCREENS, LEVEL6_POST_L5_SCREENS[1:]):
        assert after in neighbor_screens(before).values(), f"{before:02x}->{after:02x}"


def test_level6_hops_from_l5_door_uses_post_l5_path() -> None:
    assert level6_hops_from(0x0B) == POST_L5_TO_LEVEL6_HOPS
    assert level6_hops_from(0x1B)[0].target == 0x1A
    assert level6_hops_from(0x32)[0].target == 0x22


def test_post_l5_settle_idles_fanfare_then_l5_door() -> None:
    ctl = PostL5TriforceSettleController()
    fanfare = read_snapshot(_ram(mode=18, level=5, screen=0x14, triforce=0x1F))
    act = ctl.step(fanfare)
    assert act.reason == "settle_wait"
    assert ctl.success is False
    ready = read_snapshot(_ram())
    act = ctl.step(ready)
    assert ctl.success
    assert ctl.phase is PostL5SettlePhase.DONE
    assert act.reason == "settle_done"
    assert post_l5_overworld_ready(ready)
    assert not post_l5_overworld_ready(read_snapshot(_ram(triforce=0x0F)))
    assert not post_l5_overworld_ready(read_snapshot(_ram(screen=0x22)))


def test_level6_entry_attaches_post_l5_hops() -> None:
    stages = level6_entry_stages()
    assert [name for name, _, _ in stages] == ["settle_l5_tf", "enter_level6"]
    assert isinstance(stages[0][1], PostL5TriforceSettleController)
    enter = stages[1][1]
    assert enter.hops == POST_L5_TO_LEVEL6_HOPS
    assert enter.require_dungeon is True
    assert stages[1][2] == POST_L5_PATH_MAX_FRAMES
    ctl = make_post_l5_level6_controller()
    assert ctl.hops[0].target == 0x1B
    assert ctl.hops[0].direction == "DOWN"
    run = SpineRun(through="level6-entry", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_entry_0x79"


def test_level6_entry_stop_requires_l5_inventory() -> None:
    ram = _ram(level=6, screen=0x79, x=120, y=205)
    snap = read_snapshot(ram)
    assert level6_entry_success(snap, whistle=1)
    assert not level6_entry_success(snap, whistle=0)
    ram[ADDR_LADDER] = 0
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_LADDER] = 1
    ram[ADDR_RAFT] = 0
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_RAFT] = 1
    ram[ADDR_TRIFORCE] = 0x0F
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_TRIFORCE] = 0x1F
    ram[ADDR_SCREEN] = 0x7A
    assert not level6_entry_success(read_snapshot(ram), whistle=1)


def test_lost_hills_west_channel_gates_burned_leftovers() -> None:
    """Screenshot occupancy: west sand y=136–151 x<72. Not north-edge LEFT."""
    from retro_harness.nes import nes_action

    assert lost_hills_west_dir(112, 61) == "DOWN"  # v25 north arrival
    assert lost_hills_west_dir(96, 141) == "DOWN"  # v1 rock at x≈72
    assert lost_hills_west_dir(96, 165) == "DOWN"  # v26 bottom rock row
    assert lost_hills_west_dir(96, 185) == "LEFT"  # south sand of both rows
    assert lost_hills_west_dir(71, 189) == "LEFT"  # v27 under the south rock
    assert lost_hills_west_dir(48, 189) == "UP"  # v28 SW mountain, notch north
    assert lost_hills_west_dir(64, 165) == "DOWN"  # still east of notch
    assert lost_hills_west_dir(48, 165) == "LEFT"  # v29 toward v17 aisle
    assert lost_hills_west_dir(40, 181) == "UP"  # v30 west wall, climb notch
    assert lost_hills_west_dir(32, 165) == "UP"  # v17 SW notch / west aisle
    assert lost_hills_west_dir(32, 189) == "UP"
    assert lost_hills_west_dir(40, 93) == "DOWN"  # v8 NW alcove
    assert lost_hills_west_dir(64, 125) == "DOWN"  # v15 north of channel
    assert lost_hills_west_dir(64, 149) == "LEFT"  # v12 in channel band
    assert lost_hills_west_dir(32, 141) == "LEFT"
    assert HILLS_CHANNEL_Y_LO <= 141 <= HILLS_CHANNEL_Y_HI
    assert HILLS_ROCK_X == 72
    assert HILLS_AISLE_X == 32
    assert HILLS_NOTCH_X == 48

    ctl = make_post_l5_level6_controller()
    hop = POST_L5_TO_LEVEL6_HOPS[1]
    assert hop.target == 0x1A
    assert hop.align_y == 141
    north = read_snapshot(_ram(screen=0x1B, x=112, y=61))
    act = ctl._extra_hop_action(north, hop)
    assert act is not None
    assert act.reason.startswith("hills_down")
    channel = read_snapshot(_ram(screen=0x1B, x=32, y=141))
    act = ctl._extra_hop_action(channel, hop)
    assert list(act.action) == list(nes_action("LEFT"))
    assert act.reason == "hills_west_left"
    v31 = read_snapshot(_ram(screen=0x1B, x=24, y=149))
    act = ctl._extra_hop_action(v31, hop)
    assert list(act.action) == list(nes_action("UP"))
    assert act.reason == "hills_west_ay"
    north_ch = read_snapshot(_ram(screen=0x1B, x=24, y=136))
    act = ctl._extra_hop_action(north_ch, hop)
    assert list(act.action) == list(nes_action("DOWN"))
    sw = read_snapshot(_ram(screen=0x1B, x=32, y=165))
    act = ctl._extra_hop_action(sw, hop)
    assert act.reason.startswith("hills_up")
    ctl.stuck = 181
    solid = ctl._extra_hop_action(sw, hop)
    assert solid.reason.startswith("hills_solid_32_165")
    assert ctl.phase.name == "FAILED"

    ctl = make_post_l5_level6_controller()
    hop14 = POST_L5_TO_LEVEL6_HOPS[7]
    assert hop14.target == 0x14
    east = read_snapshot(_ram(screen=0x15, x=232, y=109))
    act = ctl._extra_hop_action(east, hop14)
    assert act is not None
    assert act.reason.startswith("ow15_inland")
    mid = read_snapshot(_ram(screen=0x15, x=104, y=141))
    act = ctl._extra_hop_action(mid, hop14)
    assert act.reason.startswith("ow15_south")


def test_level6_east_key_attaches_right_then_fight() -> None:
    stages = level6_east_key_stages()
    assert [name for name, _, _ in stages] == [
        "level6_right_0x7a",
        "level6_east_key_0x7a",
    ]
    ram = _ram(level=6, screen=0x7A, x=120, y=141, keys=6)
    snap = read_snapshot(ram)
    assert level6_east_key_success(snap, keys_before=5)
    assert not level6_east_key_success(snap, keys_before=6)
    run = SpineRun(through="level6-east-key", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_east_key_0x7a"


def test_level6_west_attaches_return_then_key_then_clear() -> None:
    from retro_harness.nes import nes_action

    stages = level6_west_stages()
    assert [name for name, _, _ in stages] == [
        "level6_return_0x79",
        "level6_west_key_0x78",
        "level6_west_clear_0x78",
    ]
    assert isinstance(stages[0][1], Level6Return79Controller)
    ctl = Level6Return79Controller()
    ram = _ram(level=6, screen=0x7A, x=120, y=141)
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_SCREEN] = 0x79
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "arrived_79"
    ram[ADDR_SCREEN] = 0x78
    ram[ADDR_LEVEL] = 6
    snap = read_snapshot(ram)
    assert level6_west_success(snap)
    ram[ADDR_SCREEN] = 0x68
    assert not level6_west_success(read_snapshot(ram))
    run = SpineRun(through="level6-west", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x78"
    assert "level6-west" in L6_THROUGH
    assert not hasattr(OverworldToLevel6Controller(), "bfs")


def test_level6_compass_occupancy_up_from_0x78() -> None:
    from retro_harness.nes import nes_action

    stages = level6_compass_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x68"]
    assert isinstance(stages[0][1], Level6North68Controller)
    assert not hasattr(stages[0][1], "bfs")

    ctl = Level6North68Controller()
    leftover = read_snapshot(_ram(level=6, screen=0x78, x=144, y=141))
    act = ctl.step(leftover)
    assert act.reason == "north_path"
    assert list(act.action) in (
        list(nes_action("LEFT")),
        list(nes_action("UP")),
    )

    push = Level6North68Controller()
    act = push.step(read_snapshot(_ram(level=6, screen=0x78, x=120, y=101)))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))

    arrive = Level6North68Controller()
    ram = _ram(level=6, screen=0x68, x=120, y=205)
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_68"
    assert level6_compass_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x78
    assert not level6_compass_success(read_snapshot(ram))
    run = SpineRun(through="level6-compass", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_compass_0x68"
    assert "level6-compass" in L6_THROUGH


def test_level6_clear68_attaches_occupancy_fight() -> None:
    from zelda_i.dungeon_ids import ZOL_OBJECT_TYPE
    from zelda_i.level6_dungeon import LEVEL6_COMPASS_BIT, make_compass_68_controller

    stages = level6_clear68_stages()
    assert [name for name, _, _ in stages] == ["level6_clear_0x68"]
    fight = stages[0][1]
    assert fight.spec.room_id == 0x68
    assert fight.spec.combat.occupancy_patrol
    assert fight is not make_compass_68_controller()
    ram = _ram(level=6, screen=0x68, x=120, y=205)
    ram[ADDR_COMPASS] = 0x1F | LEVEL6_COMPASS_BIT
    snap = read_snapshot(ram)
    assert level6_clear68_success(snap)
    ram[ADDR_COMPASS] = 0x1F
    assert not level6_clear68_success(read_snapshot(ram))
    ram[ADDR_COMPASS] = 0x1F | LEVEL6_COMPASS_BIT
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_clear68_success(read_snapshot(ram))
    run = SpineRun(through="level6-clear68", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_clear_0x68"
    assert "level6-clear68" in L6_THROUGH


def test_level6_keese_occupancy_up_from_0x68() -> None:
    from retro_harness.nes import nes_action

    stages = level6_keese_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x58"]
    ctl = make_north_58_controller()
    assert ctl.source_room == 0x68
    assert ctl.dest_room == 0x58
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x68, x=120, y=149)))
    assert act.reason == "north_path"
    assert list(act.action) == list(nes_action("UP"))
    arrive = make_north_58_controller()
    ram = _ram(level=6, screen=0x58, x=120, y=205)
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_58"
    assert level6_keese_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x68
    assert not level6_keese_success(read_snapshot(ram))
    run = SpineRun(through="level6-keese", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_keese_0x58"
    assert "level6-keese" in L6_THROUGH


def test_level6_clear58_attaches_occupancy_keese() -> None:
    from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE

    stages = level6_clear58_stages()
    assert [name for name, _, _ in stages] == ["level6_clear_0x58"]
    fight = stages[0][1]
    assert fight.spec.room_id == 0x58
    assert fight.spec.combat.occupancy_patrol
    assert fight.spec.alive_rule.name == "TYPE"
    ram = _ram(level=6, screen=0x58, x=120, y=205)
    assert level6_clear58_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 0
    assert not level6_clear58_success(read_snapshot(ram))
    run = SpineRun(through="level6-clear58", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_clear_0x58"
    assert "level6-clear58" in L6_THROUGH


def test_level6_room48_occupancy_up_from_0x58() -> None:
    from retro_harness.nes import nes_action

    stages = level6_room48_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x48"]
    ctl = make_north_48_controller()
    assert ctl.source_room == 0x58
    assert ctl.dest_room == 0x48
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x58, x=112, y=167)))
    assert act.reason == "north_path"
    push = make_north_48_controller()
    act = push.step(read_snapshot(_ram(level=6, screen=0x58, x=120, y=101)))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))
    ram = _ram(level=6, screen=0x48, x=120, y=205)
    arrive = make_north_48_controller()
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_48"
    assert level6_room48_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x58
    assert not level6_room48_success(read_snapshot(ram))
    run = SpineRun(through="level6-room48", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_room_0x48"


def test_level6_room38_occupancy_up_from_0x48() -> None:
    from retro_harness.nes import nes_action

    stages = level6_room38_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x38"]
    ctl = make_north_38_controller()
    assert ctl.source_room == 0x48
    assert ctl.dest_room == 0x38
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x48, x=120, y=205)))
    assert act.reason == "north_path"
    assert list(act.action) == list(nes_action("UP"))
    ram = _ram(level=6, screen=0x38, x=120, y=205)
    arrive = make_north_38_controller()
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_38"
    assert level6_room38_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x48
    assert not level6_room38_success(read_snapshot(ram))
    run = SpineRun(through="level6-room38", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_room_0x38"


def test_level6_clear38_attaches_occupancy() -> None:
    from zelda_i.dungeon_ids import LIKE_LIKE_OBJECT_TYPE

    stages = level6_clear38_stages()
    assert [name for name, _, _ in stages] == ["level6_clear_0x38"]
    fight = stages[0][1]
    assert fight.spec.room_id == 0x38
    assert fight.spec.combat.occupancy_patrol
    ram = _ram(level=6, screen=0x38, x=120, y=189)
    assert level6_clear38_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_clear38_success(read_snapshot(ram))
    run = SpineRun(through="level6-clear38", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_clear_0x38"
    assert "level6-clear38" in L6_THROUGH


def _plant_block(ram: np.ndarray, slot: int, x: int, y: int) -> None:
    ram[ADDR_OBJ_TYPE + slot] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def test_level6_room28_push_then_north_from_0x38_west() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level6_path import Level6Push38Controller

    stages = level6_room28_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x28"]
    ctl = make_north_28_controller()
    assert isinstance(ctl, Level6Push38Controller)
    assert ctl.source_room == 0x38
    assert ctl.dest_room == 0x28
    assert not hasattr(ctl, "walker")

    west = _ram(level=6, screen=0x38, x=32, y=149)
    _plant_block(west, 11, 112, 117)
    _plant_block(west, 12, 144, 117)
    snap = read_snapshot(west)
    left = left_block_0x68(snap)
    assert left is not None
    assert (left.x, left.y) == (112, 117)
    stand = south_face_stand(left)
    assert stand == (112, 133)

    act = ctl.step(snap)
    assert act.reason == "west_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))

    south_of_face = make_north_28_controller()
    inland = _ram(level=6, screen=0x38, x=48, y=149)
    _plant_block(inland, 11, 112, 117)
    _plant_block(inland, 12, 144, 117)
    act = south_of_face.step(read_snapshot(inland))
    assert act.reason == "stand_x"
    assert list(act.action) == list(nes_action("RIGHT"))

    north_of_face = make_north_28_controller()
    high = _ram(level=6, screen=0x38, x=48, y=125)
    _plant_block(high, 11, 112, 117)
    _plant_block(high, 12, 144, 117)
    act = north_of_face.step(read_snapshot(high))
    assert act.reason == "stand_y"
    assert list(act.action) == list(nes_action("DOWN"))

    at_stand = make_north_28_controller()
    ram_stand = _ram(level=6, screen=0x38, x=stand[0], y=stand[1])
    _plant_block(ram_stand, 11, 112, 117)
    _plant_block(ram_stand, 12, 144, 117)
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "push_left_block"
    assert list(act.action) == list(nes_action("UP"))
    assert at_stand.phase.name == "PUSH"
    assert any(n.startswith("at_push_") for n in at_stand.notes)
    assert any(n.startswith("left_block_") for n in at_stand.notes)

    ram_stand[ADDR_LINK_Y + 11] = 101
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "north_west"
    assert list(act.action) == list(nes_action("LEFT"))
    assert at_stand.phase.name == "NORTH"
    assert any(n.startswith("pushed_112_117_to_112_101") for n in at_stand.notes)

    ram_stand[ADDR_LINK_X] = 64
    ram_stand[ADDR_LINK_Y] = 165
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "north_aisle"
    assert list(act.action) == list(nes_action("UP"))
    ram_stand[ADDR_LINK_Y] = 100
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "north_align"
    assert list(act.action) == list(nes_action("RIGHT"))
    ram_stand[ADDR_LINK_X] = 120
    act = at_stand.step(read_snapshot(ram_stand))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))

    ram = _ram(level=6, screen=0x28, x=120, y=205)
    arrive = make_north_28_controller()
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_28"
    assert level6_room28_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x38
    assert not level6_room28_success(read_snapshot(ram))
    run = SpineRun(through="level6-room28", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_room_0x28"


def test_level6_clear28_attaches_occupancy() -> None:
    from zelda_i.level6_overworld import WIZZROBE_ORANGE_TYPE

    stages = level6_clear28_stages()
    assert [name for name, _, _ in stages] == ["level6_clear_0x28"]
    fight = stages[0][1]
    assert fight.spec.room_id == 0x28
    assert fight.spec.combat.occupancy_patrol
    assert fight.spec.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    ram = _ram(level=6, screen=0x28, x=120, y=189)
    assert level6_clear28_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_clear28_success(read_snapshot(ram))
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_OBJ_TYPE + 2] = 0x40
    ram[ADDR_OBJ_HP + 2] = 64
    assert level6_clear28_success(read_snapshot(ram))
    run = SpineRun(through="level6-clear28", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_clear_0x28"
    assert "level6-clear28" in L6_THROUGH


def test_level6_room18_occupancy_up_from_0x28() -> None:
    from retro_harness.nes import nes_action

    stages = level6_room18_stages()
    assert [name for name, _, _ in stages] == ["level6_north_0x18"]
    ctl = make_north_18_controller()
    assert ctl.source_room == 0x28
    assert ctl.dest_room == 0x18
    assert ctl.use_occupancy is False
    assert ctl.clip_left_up is True
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x28, x=120, y=181)))
    assert act.reason == "diamond_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    past = make_north_18_controller()
    act = past.step(read_snapshot(_ram(level=6, screen=0x28, x=96, y=173)))
    assert act.reason == "north_hold"
    assert list(act.action) == list(nes_action("UP"))
    hold = make_north_18_controller()
    act = hold.step(read_snapshot(_ram(level=6, screen=0x28, x=80, y=165)))
    assert act.reason == "north_hold"
    assert list(act.action) == list(nes_action("UP"))
    band = make_north_18_controller()
    act = band.step(read_snapshot(_ram(level=6, screen=0x28, x=96, y=109)))
    assert act.reason == "door_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    push = make_north_18_controller()
    act = push.step(read_snapshot(_ram(level=6, screen=0x28, x=120, y=101)))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))
    ram = _ram(level=6, screen=0x18, x=120, y=205)
    arrive = make_north_18_controller()
    act = arrive.step(read_snapshot(ram))
    assert arrive.success
    assert act.reason == "arrived_18"
    assert level6_room18_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x28
    assert not level6_room18_success(read_snapshot(ram))
    run = SpineRun(through="level6-room18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_room_0x18"
    assert "level6-room18" in L6_THROUGH


def test_level6_settle18_idles_and_censuses_spawn() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    stages = level6_settle18_stages()
    assert [name for name, _, _ in stages] == ["level6_settle_0x18"]
    ctl = make_settle_18_controller()
    assert ctl.idle_frames == SETTLE_18_IDLE_FRAMES
    ram = _ram(level=6, screen=0x18, x=120, y=189)
    ram[ADDR_ROOM_ITEM_ID] = 0x03
    ram[ADDR_CUR_OPENED_DOORS] = 0x04
    ram[ADDR_OPEN_DOORWAY_MASK] = 0x04
    ram[ADDR_OBJ_TYPE + 1] = 0x43
    ram[ADDR_OBJ_HP + 1] = 160
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 93
    ram[ADDR_OBJ_TYPE + 2] = 0x2B
    ram[ADDR_OBJ_HP + 2] = 240
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "spawn_idle"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    assert ctl.saw_0x43
    assert not ctl.saw_0x46
    assert ctl.type_histogram.get("0x43") == 1
    assert "0x2b" not in ctl.type_histogram
    assert ctl.samples
    assert ctl.samples[0]["objects"][0]["type"] == 0x43
    for _ in range(SETTLE_18_IDLE_FRAMES - 1):
        act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "settled"
    assert level6_settle18_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = 0x3F
    assert not level6_settle18_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = 0x1F
    ram[ADDR_SCREEN] = 0x28
    assert not level6_settle18_success(read_snapshot(ram))
    empty = make_settle_18_controller()
    empty.idle_frames = 2
    empty.max_frames = 8
    bare = _ram(level=6, screen=0x18, x=120, y=189)
    empty.step(read_snapshot(bare))
    empty.step(read_snapshot(bare))
    assert empty.success
    assert not empty.saw_0x43
    assert empty.type_histogram == {}
    report = ctl.report()
    assert report["saw_0x43"] is True
    assert report["leftover"]["x"] == 120
    assert report["leftover"]["y"] == 189
    assert report["cur_opened_doors"] == 0x04
    run = SpineRun(through="level6-settle18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_settle_0x18"
    assert "level6-settle18" in L6_THROUGH


def test_level6_gleeok18_clips_then_south_stands_0x44() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_gleeok18 import make_gleeok_18_controller

    stages = level6_gleeok18_stages()
    assert [name for name, _, _ in stages] == ["level6_gleeok_0x18"]
    ctl = make_gleeok_18_controller()
    ram = _ram(level=6, screen=0x18, x=120, y=189)
    ram[ADDR_OBJ_TYPE + 1] = 0x44
    ram[ADDR_OBJ_HP + 1] = 160
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "diamond_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    inland = make_gleeok_18_controller()
    ram[ADDR_LINK_Y] = 133
    ram[ADDR_LINK_X] = 124
    act = inland.step(read_snapshot(ram))
    assert act.reason == "south_stand"
    assert list(act.action) == list(nes_action("UP", "A"))
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    gone = make_gleeok_18_controller()
    gone.saw_0x44 = True
    act = gone.step(read_snapshot(ram))
    assert gone.success
    assert act.reason == "body_gone"
    assert list(act.action) == list(nes_idle_action())
    assert level6_gleeok18_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0x44
    ram[ADDR_OBJ_HP + 1] = 160
    assert not level6_gleeok18_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0x43
    assert level6_gleeok18_success(read_snapshot(ram))
    run = SpineRun(through="level6-gleeok18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_gleeok_0x18"
    assert "level6-gleeok18" in L6_THROUGH


def test_level6_postgleeok18_censuses_residual_and_doors() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_gleeok18 import (
        POSTGLEEOK_STAND_X,
        POSTGLEEOK_STAND_Y,
        STAIRS_KEEP_Y,
        make_postgleeok_18_controller,
    )

    stages = level6_postgleeok18_stages()
    assert [name for name, _, _ in stages] == ["level6_postgleeok_0x18"]
    assert POSTGLEEOK_STAND_X == 120
    assert POSTGLEEOK_STAND_Y == 133
    assert STAIRS_KEEP_Y == 125
    leftover = _ram(level=6, screen=0x18, x=121, y=133)
    idle = make_postgleeok_18_controller()
    idle.census_frames = 3
    idle.after_heads = 2
    act = idle.step(read_snapshot(leftover))
    assert act.reason == "residual_idle"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    idle.step(read_snapshot(leftover))
    act = idle.step(read_snapshot(leftover))
    assert idle.success
    assert act.reason == "heads_gone"
    assert level6_postgleeok18_success(read_snapshot(leftover))

    heads = make_postgleeok_18_controller()
    ram = _ram(level=6, screen=0x18, x=121, y=133)
    ram[ADDR_OBJ_TYPE + 1] = 0x46
    ram[ADDR_OBJ_HP + 1] = 80
    ram[ADDR_LINK_X + 1] = 64
    ram[ADDR_LINK_Y + 1] = 100
    act = heads.step(read_snapshot(ram))
    assert act.reason == "south_stand"
    assert list(act.action) == list(nes_action("UP", "A"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert not heads.success
    assert not level6_postgleeok18_success(read_snapshot(ram))

    fb = make_postgleeok_18_controller()
    ball = _ram(level=6, screen=0x18, x=121, y=133)
    ball[ADDR_OBJ_TYPE + 1] = 0x56
    ball[ADDR_OBJ_HP + 1] = 16
    ball[ADDR_LINK_X + 1] = 130
    ball[ADDR_LINK_Y + 1] = 133
    act = fb.step(read_snapshot(ball))
    assert act.reason == "fb_dodge"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))

    east = make_postgleeok_18_controller()
    ram_e = _ram(level=6, screen=0x18, x=121, y=133)
    ram_e[ADDR_CUR_OPENED_DOORS] = 0x05
    act = east.step(read_snapshot(ram_e))
    assert act.reason == "residual_idle"
    assert not east.success
    ram_e[ADDR_OPEN_DOORWAY_MASK] = 0x01
    opened = make_postgleeok_18_controller()
    act = opened.step(read_snapshot(ram_e))
    assert opened.success
    assert act.reason == "east_open"
    assert level6_postgleeok18_success(read_snapshot(ram_e))

    stairs = make_postgleeok_18_controller()
    ram_s = _ram(level=6, screen=0x18, x=120, y=109, mode=9)
    act = stairs.step(read_snapshot(ram_s))
    assert stairs.success
    assert act.reason == "stairs"
    assert level6_postgleeok18_success(read_snapshot(ram_s))

    ram[ADDR_OBJ_TYPE + 1] = 0x44
    ram[ADDR_OBJ_HP + 1] = 160
    assert not level6_postgleeok18_success(read_snapshot(ram))
    run = SpineRun(through="level6-postgleeok18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_postgleeok_0x18"
    assert "level6-postgleeok18" in L6_THROUGH


def test_level6_stairs18_occupancy_then_up() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_stairs18 import STAIRS_18_GOAL, make_stairs_18_controller

    stages = level6_stairs18_stages()
    assert [name for name, _, _ in stages] == ["level6_stairs_0x18"]
    assert STAIRS_18_GOAL == (120, 96)
    leftover = make_stairs_18_controller()
    act = leftover.step(read_snapshot(_ram(level=6, screen=0x18, x=156, y=133)))
    assert act.reason == "stairs_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    aligned = make_stairs_18_controller()
    act = aligned.step(read_snapshot(_ram(level=6, screen=0x18, x=120, y=133)))
    assert act.reason == "stairs_path"
    assert list(act.action) == list(nes_action("UP"))
    south = make_stairs_18_controller()
    act = south.step(read_snapshot(_ram(level=6, screen=0x18, x=120, y=109)))
    assert act.reason == "stairs_path"
    assert list(act.action) == list(nes_action("UP"))
    hole = make_stairs_18_controller()
    act = hole.step(read_snapshot(_ram(level=6, screen=0x18, x=120, y=96)))
    assert act.reason == "stairs_idle"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    cellar = make_stairs_18_controller()
    ram_c = _ram(level=6, screen=0x18, x=120, y=109, mode=9)
    act = cellar.step(read_snapshot(ram_c))
    assert cellar.success
    assert act.reason == "stairs"
    assert level6_stairs18_success(read_snapshot(ram_c))
    other = _ram(level=6, screen=0x07, x=128, y=141)
    assert level6_stairs18_success(read_snapshot(other))
    still = _ram(level=6, screen=0x18, x=120, y=109)
    assert not level6_stairs18_success(read_snapshot(still))
    run = SpineRun(through="level6-stairs18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_stairs_0x18"
    assert "level6-stairs18" in L6_THROUGH
    assert list(nes_idle_action()) != list(nes_action("UP"))


def test_level6_compass_fails_closed_on_east_return() -> None:
    ctl = Level6North68Controller()
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x79, x=32, y=141)))
    assert ctl.failed
    assert act.reason == "left_0x78"


def test_level6_east_key_live_enemies_block_stop() -> None:
    from zelda_i.level6_overworld import WIZZROBE_ORANGE_TYPE

    ram = _ram(level=6, screen=0x7A, x=120, y=141, keys=6)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_east_key_success(read_snapshot(ram), keys_before=5)
