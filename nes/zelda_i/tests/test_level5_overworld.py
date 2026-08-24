from types import SimpleNamespace

import numpy as np

from zelda_i.level5_overworld import (
    LEVEL5_PATH_HOPS,
    POST_L4_PATH_MAX_FRAMES,
    POST_L4_TO_LEVEL5_HOPS,
    SCREEN_POST_L4_RETURN,
    Level5NavPhase,
    OverworldToLevel5Controller,
    PostL4SettlePhase,
    PostL4TriforceSettleController,
    level5_hops_from,
    make_post_l4_level5_controller,
    post_l4_overworld_ready,
)
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level5_dungeon import ROOM_66_SPEC
from zelda_i.level5_spine import (
    ROOM_66_SPINE_SPEC,
    Level5EastKeyNavController,
    level5_clear66_stages,
    level5_clear66_success,
    level5_east77_stages,
    level5_east77_success,
    level5_entry_stages,
    level5_entry_success,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SpineRun


def test_post_l4_path_preserves_verified_level5_suffix() -> None:
    suffix = POST_L4_TO_LEVEL5_HOPS[-len(LEVEL5_PATH_HOPS) :]
    assert suffix == LEVEL5_PATH_HOPS


def test_post_l4_path_returns_from_raft_island_and_joins_4a() -> None:
    targets = [hop.target for hop in POST_L4_TO_LEVEL5_HOPS]
    assert targets[:7] == [0x55, 0x56, 0x57, 0x58, 0x59, 0x49, 0x4A]
    assert POST_L4_TO_LEVEL5_HOPS[0].direction == "DOWN"
    assert POST_L4_TO_LEVEL5_HOPS[0].align_x == 128


def test_post_l4_56_entry_uses_open_center_channel() -> None:
    hop = POST_L4_TO_LEVEL5_HOPS[2]
    assert hop.target == 0x57
    assert hop.direction == "RIGHT"
    assert hop.align_y == 141


def test_level5_hops_from_l4_island_uses_post_l4_path() -> None:
    assert SCREEN_POST_L4_RETURN == 0x45
    assert level5_hops_from(0x45) == POST_L4_TO_LEVEL5_HOPS
    assert level5_hops_from(0x4A) == LEVEL5_PATH_HOPS


def _l4_ow_ram(*, mode: int = PLAY_MODE, screen: int = 0x45, tf: int = 0x0F, raft: int = 1):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = 128
    ram[ADDR_LINK_Y] = 125
    ram[ADDR_TRIFORCE] = tf
    ram[ADDR_RAFT] = raft
    ram[ADDR_LADDER] = 1
    return ram


def test_post_l4_settle_idles_fanfare_then_island() -> None:
    ctl = PostL4TriforceSettleController()
    fanfare = read_snapshot(_l4_ow_ram(mode=18, screen=0x03, tf=0x0F))
    act = ctl.step(fanfare)
    assert act.reason == "settle_wait"
    assert ctl.success is False
    ready = read_snapshot(_l4_ow_ram())
    act = ctl.step(ready)
    assert ctl.success
    assert ctl.phase is PostL4SettlePhase.DONE
    assert act.reason == "settle_done"
    assert post_l4_overworld_ready(ready)
    assert not post_l4_overworld_ready(read_snapshot(_l4_ow_ram(tf=0x07)))
    assert not post_l4_overworld_ready(read_snapshot(_l4_ow_ram(raft=0)))


def test_level5_entry_attaches_post_l4_hops_not_old_at4a() -> None:
    stages = level5_entry_stages()
    assert [name for name, _, _ in stages] == ["settle_l4_tf", "enter_level5"]
    assert isinstance(stages[0][1], PostL4TriforceSettleController)
    enter = stages[1][1]
    assert enter.hops == POST_L4_TO_LEVEL5_HOPS
    assert enter.require_dungeon is True
    assert stages[1][2] == POST_L4_PATH_MAX_FRAMES
    ctl = make_post_l4_level5_controller()
    assert ctl.hops[0].target == 0x55
    assert ctl.hops[0].direction == "DOWN"
    run = SpineRun(through="level5-entry", success=True, boot_frames=199)
    assert run.report()["stop"] == "level5_entry_0x76"


def test_level5_entry_stop_requires_l4_inventory() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x76
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    snap = read_snapshot(ram)
    assert level5_entry_success(snap)
    ram[ADDR_LADDER] = 0
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 1
    ram[ADDR_RAFT] = 0
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_RAFT] = 1
    ram[ADDR_TRIFORCE] = 0x07
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_SCREEN] = 0x66
    assert not level5_entry_success(read_snapshot(ram))


def test_level5_clear66_attaches_existing_gibdo_controller() -> None:
    stages = level5_clear66_stages()
    assert [name for name, _, _ in stages] == ["level5_clear_0x66"]
    ctl = stages[0][1]
    assert isinstance(ctl, GenericDungeonRoomController)
    assert ctl.spec is ROOM_66_SPINE_SPEC
    assert ctl.spec.combat.occupancy_patrol is True
    assert ROOM_66_SPEC.combat.occupancy_patrol is False
    assert ctl.phase is DungeonPhase.ROUTE_ENTRY
    assert stages[0][2] == ROOM_66_SPINE_SPEC.max_frames == 20000
    assert "bfs" not in ctl.report()
    run = SpineRun(through="level5-clear66", success=True, boot_frames=199)
    assert run.report()["stop"] == "level5_clear_0x66"


def test_level5_clear66_stop_is_empty_66_with_east_door() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x66
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    ram[ADDR_CUR_OPENED_DOORS] = 0x08
    ram[ADDR_ROOM_ALL_DEAD] = 20
    snap = read_snapshot(ram)
    assert level5_clear66_success(snap)
    ram[ADDR_CUR_OPENED_DOORS] = 0
    assert not level5_clear66_success(read_snapshot(ram))
    ram[ADDR_CUR_OPENED_DOORS] = 0x08
    ram[ADDR_SCREEN] = 0x76
    assert not level5_clear66_success(read_snapshot(ram))


def test_level5_east77_attaches_nav_then_pols_voice() -> None:
    from zelda_i.level5_dungeon import Level5PolsVoiceController
    from retro_harness.nes import nes_action

    stages = level5_east77_stages()
    assert [name for name, _, _ in stages] == [
        "level5_east_key_0x77",
        "level5_clear_0x77",
    ]
    assert isinstance(stages[0][1], Level5EastKeyNavController)
    assert isinstance(stages[1][1], Level5PolsVoiceController)
    assert "bfs" not in stages[0][1].report()
    run = SpineRun(through="level5-east77", success=True, boot_frames=199)
    assert run.report()["stop"] == "level5_east_key_0x77"
    ctl = Level5EastKeyNavController()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x66
    ram[ADDR_LINK_X] = 32
    ram[ADDR_LINK_Y] = 101
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT"))
    assert act.reason == "east_key_to_ladder_x"
    ram[ADDR_SCREEN] = 0x77
    ram[ADDR_LINK_X] = 16
    ram[ADDR_LINK_Y] = 141
    for _ in range(ctl.settle_frames):
        act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "arrived_77"
    ram[ADDR_SCREEN] = 0x77
    assert level5_east77_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x66
    assert not level5_east77_success(read_snapshot(ram))


def test_level5_whistle_stop_requires_recorder() -> None:
    from zelda_i.level5_spine import level5_whistle_success

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x04
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    snap = read_snapshot(ram)
    assert level5_whistle_success(snap, whistle=1)
    assert not level5_whistle_success(snap, whistle=0)
    ram[ADDR_SCREEN] = 0x77
    ram[ADDR_MODE] = PLAY_MODE
    assert not level5_whistle_success(read_snapshot(ram), whistle=1)
    run = SpineRun(through="level5-whistle", success=True, boot_frames=199)
    assert run.report()["stop"] == "level5_whistle_0x04"


def test_level5_exit04_stop_is_play_05_not_cellar() -> None:
    from zelda_i.level5_boss_path import STOP_EXIT04, TF_SUFFIX_STOPS
    from zelda_i.level5_spine import L5_THROUGH, level5_exit04_success

    assert "level5-exit04" in L5_THROUGH
    assert STOP_EXIT04 in TF_SUFFIX_STOPS
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x05
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    snap = read_snapshot(ram)
    assert level5_exit04_success(snap, whistle=1)
    assert not level5_exit04_success(snap, whistle=0)
    ram[ADDR_SCREEN] = 0x04
    ram[ADDR_MODE] = 9
    assert not level5_exit04_success(read_snapshot(ram), whistle=1)
    run = SpineRun(through="level5-exit04", success=True, boot_frames=199)
    assert run.report()["stop"] == "level5_exit_0x04"


def test_level5_tf_stop_requires_bit_0x10_in_room_0x14() -> None:
    from zelda_i.anchors import LEVEL5_TF_ROOM, TF_BIT_L5
    from zelda_i.level5_spine import L5_STOPS, level5_tf_success

    assert LEVEL5_TF_ROOM == 0x14
    assert TF_BIT_L5 == 0x10
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x14
    ram[ADDR_TRIFORCE] = 0x1F
    snap = read_snapshot(ram)
    assert level5_tf_success(snap)
    ram[ADDR_TRIFORCE] = 0x0F
    assert not level5_tf_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = 0x1F
    ram[ADDR_SCREEN] = 0x24
    assert not level5_tf_success(read_snapshot(ram))
    run = SpineRun(through="level5", success=True, boot_frames=199)
    assert run.report()["stop"] == L5_STOPS["level5"] == "level5_triforce_0x10"


def test_lost_hills_east_ledge_steps_left_before_down() -> None:
    controller = OverworldToLevel5Controller()
    controller.phase = Level5NavPhase.FREE_POCKET
    controller.phase_frames = 1

    action = controller._free_pocket(
        SimpleNamespace(screen=0x1B, link_x=240, link_y=141)
    )

    assert action.reason.startswith("pocket_unwedge")
