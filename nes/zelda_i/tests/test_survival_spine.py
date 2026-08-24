"""Continuous Survival spine — no seamed viewing compose."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import numpy as np

from zelda_i.level1_dungeon import ROOM_45_SPEC, ROOM_45_SURVIVAL_SPEC
from zelda_i.level1_finish import level1_triforce_stages
from zelda_i.level2_overworld import OverworldToLevel2Controller, PostTriforceSettleController
from zelda_i.level2_spine import (
    level2_boom_success,
    level2_through_success,
    level2_to_boom_stages,
)
from zelda_i.level2_tf_spine import level2_tf_stages
from zelda_i.dungeon_ids import (
    INVULN_MOVER_OBJECT_TYPE as INVULN_MOVER_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    VIRE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.level3_spine import (
    level3_compass_stages,
    level3_west_darknuts_stages,
    level3_south_darknuts_stages,
    level3_raft_stages,
    level3_dest_6b_stages,
    level3_entry_stages,
    level3_west_key_stages,
)
from zelda_i.survival_spine import (
    BOOT_POLICY,
    SPINE_BOMB_RETOPUP,
    SPINE_THROUGH,
    SpineRun,
    level4_entry_stages,
    level4_first_key_stages,
    level4_first_key_success,
    level4_room40_key_stages,
    level4_north_30_stages,
    level4_north_30_success,
    level4_key_right_31_stages,
    level4_key_right_31_success,
    level4_clear_31_stages,
    level4_clear_31_success,
    level4_east_32_stages,
    level4_east_32_success,
    level4_clear_32_stages,
    level4_clear_32_success,
    level4_stepladder_stages,
    level4_stepladder_success,
    level4_exit60_stages,
    level4_exit60_success,
    level4_west31_stages,
    level4_west31_success,
    level4_keyup20_stages,
    level4_keyup20_success,
    level4_bomb11_stages,
    level4_bomb11_success,
    level4_key01_stages,
    level4_key01_success,
    level4_map21_stages,
    level4_map21_success,
    level4_mappick_stages,
    level4_mappick_success,
    level2_entry_stages,
    merge_inventory_assist,
    spine_final_fields,
    topup_owned_inventory,
    topup_owned_bombs,
    validate_l5_endpoint,
    _run_level3_boss_suffix,
)


def test_spine_through_is_continuous_only() -> None:
    assert SPINE_THROUGH == (
        "level1", "level2", "level3", "level4-entry", "level4-key",
        "level4-clear50",
        "level4-room40-key",
        "level4-room30",
        "level4-room31",
        "level4-clear31",
        "level4-room32",
        "level4-clear32",
        "level4-stepladder",
        "level4-exit60",
        "level4-west31",
        "level4-keyup20",
        "level4-room21",
        "level4-map",
        "level4-bomb11",
        "level4-key01",
        "level4-clear12",
        "level4-gleeok13",
        "level4",
        "level5-entry",
        "level5-clear66",
        "level5-east77",
        "level5-whistle",
        "level5-exit04",
        "level5",
        "level6-entry",
        "level6-east-key",
        "level6-west",
    )


def test_level4_entry_attaches_after_level3_tf() -> None:
    assert [name for name, _, _ in level4_entry_stages()] == [
        "settle_l3_tf",
        "enter_level4",
    ]
    run = SpineRun(through="level4-entry", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_entry_0x71"


def test_level4_first_key_is_a_deterministic_one_env_sequence() -> None:
    stages = level4_first_key_stages()
    assert [name for name, _, _ in stages] == [
        "level4_entry_up_0x61",
        "level4_bomb_north_0x61",
        "level4_key_0x51",
    ]
    assert stages[-1][1].phase.name == "FIGHT"
    run = SpineRun(through="level4-key", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_natural_key_0x51"


def test_level4_first_key_stop_uses_inventory_delta_not_room_timer() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x51
    ram[ADDR_KEYS] = 5
    snap = read_snapshot(ram)
    assert level4_first_key_success(snap, keys_before=4)
    assert not level4_first_key_success(snap, keys_before=5)


def test_level4_room40_key_sequence_skips_compass_and_preserves_keys() -> None:
    assert [name for name, _, _ in level4_room40_key_stages()] == [
        "level4_north_0x40",
        "level4_key_0x40",
    ]


def test_level4_north_30_attaches_existing_controller_from_room40() -> None:
    stages = level4_north_30_stages()
    assert [name for name, _, _ in stages] == ["level4_north_0x30"]
    assert stages[0][1].phase.name == "ALIGN"
    run = SpineRun(through="level4-room30", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_enter_0x30"


def test_level4_north_30_stop_is_enter_room_not_clear() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x30
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    snap = read_snapshot(ram)
    assert level4_north_30_success(snap)
    ram[ADDR_SCREEN] = 0x40
    assert not level4_north_30_success(read_snapshot(ram))


def test_level4_key_right_31_attaches_clear_then_door() -> None:
    stages = level4_key_right_31_stages()
    assert [name for name, _, _ in stages] == [
        "level4_clear_0x30",
        "level4_key_right_0x31",
    ]
    assert stages[0][1].phase.name == "TO_BAND"
    assert stages[1][1].clear_vires is False
    assert stages[1][1].phase.name == "ALIGN"
    run = SpineRun(through="level4-room31", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_enter_0x31"


def test_level4_key_right_31_stop_is_enter_room_after_key() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x31
    ram[ADDR_LINK_X] = 16
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_KEYS] = 5
    snap = read_snapshot(ram)
    assert level4_key_right_31_success(snap, keys_before=6)
    assert not level4_key_right_31_success(snap, keys_before=5)
    ram[ADDR_SCREEN] = 0x30
    ram[ADDR_KEYS] = 6
    assert not level4_key_right_31_success(read_snapshot(ram), keys_before=6)


def test_level4_clear_31_attaches_existing_maze_vire_controller() -> None:
    stages = level4_clear_31_stages()
    assert [name for name, _, _ in stages] == [
        "level4_inland_0x31",
        "level4_clear_0x31",
    ]
    assert stages[0][1].phase.name == "CLIP"
    assert stages[0][2] == 4000
    assert stages[1][1].phase.name == "FIGHT"
    assert stages[1][1].spec.room_id == 0x31
    run = SpineRun(through="level4-clear31", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_clear_0x31"


def test_level4_clear_31_stop_is_empty_maze_not_east_door() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x31
    ram[ADDR_LINK_X] = 128
    ram[ADDR_LINK_Y] = 133
    snap = read_snapshot(ram)
    assert level4_clear_31_success(snap)
    ram[ADDR_OBJ_TYPE + 1] = VIRE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level4_clear_31_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_SCREEN] = 0x32
    assert not level4_clear_31_success(read_snapshot(ram))


def test_level4_east_32_attaches_coordinate_maze_not_state_bfs() -> None:
    stages = level4_east_32_stages()
    assert [name for name, _, _ in stages] == ["level4_east_0x32"]
    controller = stages[0][1]
    assert controller.phase.name == "JOIN"
    assert stages[0][2] == 4000
    assert controller.report()["waypoints"][0] == [160, 173]
    run = SpineRun(through="level4-room32", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_enter_0x32"


def test_level4_east_32_stop_is_enter_room_not_clear() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_LINK_X] = 16
    ram[ADDR_LINK_Y] = 141
    snap = read_snapshot(ram)
    assert level4_east_32_success(snap)
    ram[ADDR_SCREEN] = 0x31
    assert not level4_east_32_success(read_snapshot(ram))


def test_level4_clear_32_attaches_existing_zol_likelike_controller() -> None:
    stages = level4_clear_32_stages()
    assert [name for name, _, _ in stages] == ["level4_clear_0x32"]
    assert stages[0][1].phase.name == "FIGHT"
    assert stages[0][1].spec.room_id == 0x32
    run = SpineRun(through="level4-clear32", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_clear_0x32"


def test_level4_clear_32_stop_is_empty_room_not_stairs() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 141
    snap = read_snapshot(ram)
    assert level4_clear_32_success(snap)
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level4_clear_32_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level4_clear_32_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert level4_clear_32_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0x68
    ram[ADDR_OBJ_HP + 1] = 0
    assert level4_clear_32_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_SCREEN] = 0x60
    assert not level4_clear_32_success(read_snapshot(ram))


def test_level4_stepladder_attaches_existing_push_controller() -> None:
    stages = level4_stepladder_stages()
    assert [name for name, _, _ in stages] == ["level4_stepladder"]
    controller = stages[0][1]
    assert controller.clear_first is False
    assert controller.phase.name == "ALIGN_PUSH"
    assert stages[0][2] == controller.max_frames
    report = controller.report()
    assert report["segment"] == "level4_stepladder"
    assert report["path_len"] >= 40
    assert report["push_stand"] == [120, 141]
    assert "bfs" not in report
    run = SpineRun(through="level4-stepladder", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_stepladder_0x60"


def test_level4_stepladder_stop_is_addr_ladder_not_exit() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_LINK_X] = 136
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_LADDER] = 1
    snap = read_snapshot(ram)
    assert level4_stepladder_success(snap)
    ram[ADDR_LADDER] = 0
    assert not level4_stepladder_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 1
    ram[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ram[ADDR_OBJ_TYPE + 2] = 0x68
    assert level4_stepladder_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_MODE] = PLAY_MODE
    assert not level4_stepladder_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 3
    assert not level4_stepladder_success(read_snapshot(ram))


def test_level4_exit60_attaches_after_stepladder() -> None:
    stages = level4_exit60_stages()
    assert [name for name, _, _ in stages] == ["level4_exit_0x60"]
    controller = stages[0][1]
    assert controller.phase.name == "SETTLE"
    report = controller.report()
    assert report["segment"] == "level4_exit_0x60"
    assert "bfs" not in report
    assert report["waypoints"][0] == [175, 141]
    assert report["waypoints"][-1] == [48, 69]
    run = SpineRun(through="level4-exit60", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_exit_0x60"


def test_level4_exit60_stop_is_play32_with_ladder() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_LINK_X] = 192
    ram[ADDR_LINK_Y] = 189
    ram[ADDR_LADDER] = 1
    assert level4_exit60_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 0
    assert not level4_exit60_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 1
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_MODE] = 9
    assert not level4_exit60_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x31
    ram[ADDR_MODE] = PLAY_MODE
    assert not level4_exit60_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_OBJ_TYPE + 1] = INVULN_MOVER_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ram[ADDR_OBJ_TYPE + 2] = 0x68
    assert level4_exit60_success(read_snapshot(ram))


def test_level4_west31_attaches_after_exit60() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level4_west31 import West31Phase, make_west31_controller

    stages = level4_west31_stages()
    assert [name for name, _, _ in stages] == ["level4_west_0x31"]
    report = stages[0][1].report()
    assert report["segment"] == "level4_west_0x31"
    assert "bfs" not in report
    assert report["waypoints"][0] == [48, 189]
    run = SpineRun(through="level4-west31", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_west_0x31"

    ctl = make_west31_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x32
    ram[ADDR_LINK_X] = 192
    ram[ADDR_LINK_Y] = 189
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 48
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 141
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 16
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "west_push_left"
    ram[ADDR_SCREEN] = 0x31
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert ctl.phase is West31Phase.DONE
    ram[ADDR_SCREEN] = 0x32
    assert not level4_west31_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x31
    assert level4_west31_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 0
    assert not level4_west31_success(read_snapshot(ram))


def test_level4_keyup20_attaches_maze_west_then_key_up() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level4_keyup20 import make_maze_31_west_controller

    stages = level4_keyup20_stages()
    assert [name for name, _, _ in stages] == [
        "level4_maze_west_0x30",
        "level4_key_up_0x20",
    ]
    report = stages[0][1].report()
    assert report["segment"] == "level4_maze_west_0x30"
    assert "bfs" not in report
    assert report["waypoints"][0] == [192, 141]
    run = SpineRun(through="level4-keyup20", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_key_up_0x20"

    ctl = make_maze_31_west_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x31
    ram[ADDR_LINK_X] = 208
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 192
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_SCREEN] = 0x30
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    ram[ADDR_SCREEN] = 0x20
    ram[ADDR_LADDER] = 1
    assert level4_keyup20_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x30
    assert not level4_keyup20_success(read_snapshot(ram))


def test_level4_map21_attaches_after_keyup20() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level4_map21 import make_map21_controller

    stages = level4_map21_stages()
    assert [name for name, _, _ in stages] == [
        "level4_clear_0x20",
        "level4_map_0x21",
    ]
    report = stages[1][1].report()
    assert report["segment"] == "level4_map_0x21"
    assert "bfs" not in report
    assert stages[0][1].report()["segment"] == "level4_clear_0x20"
    run = SpineRun(through="level4-room21", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_enter_0x21"
    ctl = make_map21_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x20
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 96
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    ram[ADDR_LINK_X] = 136
    ram[ADDR_LINK_Y] = 94
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    ram[ADDR_LINK_Y] = 96
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_LINK_X] = 200
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    ram[ADDR_LINK_Y] = 128
    ram[ADDR_LINK_X] = 208
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_Y] = 133
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_Y] = 141
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_SCREEN] = 0x21
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert level4_map21_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x20
    assert not level4_map21_success(read_snapshot(ram))


def test_level4_mappick_attaches_after_room21() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level4_dungeon import LEVEL4_MAP_BIT, MAP_21_PICKUP_XY
    from zelda_i.level4_mappick import make_mappick_controller
    from zelda_i.ram import ADDR_MAP

    stages = level4_mappick_stages()
    assert [name for name, _, _ in stages] == ["level4_map_pickup_0x21"]
    report = stages[0][1].report()
    assert report["segment"] == "level4_map_pickup_0x21"
    assert "bfs" not in report
    assert report["pickup_xy"] == list(MAP_21_PICKUP_XY)
    run = SpineRun(through="level4-map", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_map_pickup_0x21"
    ctl = make_mappick_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x21
    ram[ADDR_LINK_X] = 16
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    ram[ADDR_LINK_X] = 32
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 93
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    ram[ADDR_LINK_X] = 80
    ram[ADDR_LINK_Y] = 125
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_LINK_X] = 208
    ram[ADDR_LINK_Y] = 189
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 181
    ram[ADDR_MAP] = LEVEL4_MAP_BIT
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "done"
    assert level4_mappick_success(read_snapshot(ram))
    ram[ADDR_MAP] = 0
    assert not level4_mappick_success(read_snapshot(ram))


def test_level4_bomb11_attaches_after_map() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.bomb_wall_path import BombWallController
    from zelda_i.level4_bomb11 import make_bomb_21_north_controller
    from zelda_i.level4_dungeon import BOMB_21_NORTH_STAND
    from zelda_i.level4_occupancy import ROOM_21_BOMB_WAYPOINTS

    stages = level4_bomb11_stages()
    assert [name for name, _, _ in stages] == ["level4_bomb_north_0x21"]
    ctl = stages[0][1]
    assert isinstance(ctl, BombWallController)
    assert ctl.wall.opens_to == 0x11
    assert ctl.stand == BOMB_21_NORTH_STAND
    assert ctl.approach_waypoints == ROOM_21_BOMB_WAYPOINTS
    assert ctl.south_band_first is False
    assert "bfs" not in ctl.report()
    run = SpineRun(through="level4-bomb11", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_enter_0x11"
    ctl = make_bomb_21_north_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x21
    ram[ADDR_LINK_X] = 208
    ram[ADDR_LINK_Y] = 181
    ram[ADDR_LADDER] = 1
    ram[ADDR_BOMBS] = 15
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 93
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "approach_next"
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 105
    act = ctl.step(read_snapshot(ram))
    assert act.reason in ("approach_next", "stand_ready")
    for _ in range(10):
        act = ctl.step(read_snapshot(ram))
        if act.reason == "place_bomb":
            break
    assert act.reason == "place_bomb"
    ram[ADDR_SCREEN] = 0x11
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "done"
    assert level4_bomb11_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x21
    assert not level4_bomb11_success(read_snapshot(ram))


def test_level4_key01_attaches_after_bomb11() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.bomb_wall_path import BombWallController
    from zelda_i.dungeon import DungeonPhase
    from zelda_i.level4_key01 import BOMB_11_NORTH_STAND, make_bomb_11_north_controller

    stages = level4_key01_stages()
    assert [name for name, _, _ in stages] == [
        "level4_bomb_north_0x11",
        "level4_key_0x01",
    ]
    ctl = stages[0][1]
    assert isinstance(ctl, BombWallController)
    assert ctl.wall.opens_to == 0x01
    assert ctl.stand == BOMB_11_NORTH_STAND == (120, 105)
    assert "bfs" not in ctl.report()
    assert stages[-1][1].phase is DungeonPhase.FIGHT
    run = SpineRun(through="level4-key01", success=True, boot_frames=199)
    assert run.report()["stop"] == "level4_natural_key_0x01"
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x11
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 189
    ram[ADDR_KEYS] = 4
    ram[ADDR_BOMBS] = 15
    ctl = make_bomb_11_north_controller()
    assert list(ctl.step(read_snapshot(ram)).action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 105
    reasons = [ctl.step(read_snapshot(ram)).reason for _ in range(10)]
    assert "place_bomb" in reasons
    ram[ADDR_SCREEN] = 0x01
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert not level4_key01_success(read_snapshot(ram), keys_before=4)
    ram[ADDR_KEYS] = 5
    assert level4_key01_success(read_snapshot(ram), keys_before=4)
    ram[ADDR_SCREEN] = 0x11
    assert not level4_key01_success(read_snapshot(ram), keys_before=4)



def test_through_level3_attaches_boss_suffix_after_natural_raft() -> None:
    names = [name for name, _, _ in level3_entry_stages()]
    assert names == ["settle_l2_tf", "enter_level3"]
    west = [name for name, _, _ in level3_west_key_stages()]
    assert west == ["west_key"]
    dest_names = [name for name, _, _ in level3_dest_6b_stages()]
    assert dest_names == ["west_key", "north_chain"]
    assert "north_chain" not in names
    assert [name for name, _, _ in level3_compass_stages()] == ["compass_0x5a"]
    assert [name for name, _, _ in level3_west_darknuts_stages()] == ["west_darknuts_0x59"]
    assert [name for name, _, _ in level3_south_darknuts_stages()] == ["south_darknuts_0x69"]
    assert [name for name, _, _ in level3_raft_stages()] == ["raft_0x0f"]
    run = SpineRun(through="level3", success=True, boot_frames=199)
    assert run.report()["stop"] == "level3_triforce_0x04"
    assert "l3_entry" in run.report()


def test_level3_boss_suffix_uses_carried_bombs_without_poke(monkeypatch) -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = 0x0F
    ram[ADDR_MODE] = 9
    ram[ADDR_BOMBS] = 8
    seen = {}

    class FakeBoss:
        def __init__(self, *, poke_bombs, tag, continuous_mode):
            seen["poke_bombs"] = poke_bombs
            seen["continuous_mode"] = continuous_mode
            self.success = False
            self.failed = False

        def path_to_5d(self, env, assist, total):
            total[0] += 10
            return {"ok": True}

        def open_5d_up(self, env, assist, total):
            total[0] += 20
            return {"ok": True}

        def fight_manhandla(self, env, assist, total, *, max_frames):
            total[0] += 30
            ram[ADDR_TRIFORCE] |= 0x04
            self.success = True
            return {"ok": True, "tf04": True}

        def report(self):
            return {"poke_bombs": seen["poke_bombs"]}

    monkeypatch.setattr("zelda_i.survival_spine.Level3BossPathController", FakeBoss)
    env = SimpleNamespace(get_ram=lambda: ram)
    run = SpineRun(through="level3", success=True, boot_frames=199, end_frame=100)
    assert _run_level3_boss_suffix(env, run, assist=object())
    assert seen["poke_bombs"] is None
    assert seen["continuous_mode"] is True
    assert run.end_frame == 160
    assert run.stages[-1].name == "level3_boss_tf"
    assert run.stages[-1].success


def test_level3_boss_suffix_fails_closed_before_verified_wall_budget(monkeypatch) -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = 0x0F
    ram[ADDR_MODE] = 9
    ram[ADDR_BOMBS] = 1

    class FakeBoss:
        def __init__(self, *, poke_bombs, tag, continuous_mode):
            self.success = False
            self.failed = False
            self.last_error = None

        def _fail(self, error):
            self.failed = True
            self.last_error = error

        def report(self):
            return {"last_error": self.last_error}

    monkeypatch.setattr("zelda_i.survival_spine.Level3BossPathController", FakeBoss)
    run = SpineRun(through="level3", success=True, boot_frames=199, end_frame=100)
    assert not _run_level3_boss_suffix(SimpleNamespace(get_ram=lambda: ram), run, assist=object())
    assert run.failed_stage == "level3_boss_tf"
    assert "bomb_budget_gate" in run.stages[-1].controller.last_error


def _l2_snap(
    *,
    room: int = 0x7D,
    boom: int = 0,
    bombs: int = 0,
    keys: int = 0,
    triforce: int = 0x01,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_MAGIC_BOOMERANG] = boom
    ram[ADDR_TRIFORCE] = triforce
    ram[ADDR_HEALTH] = 0x2F
    return read_snapshot(ram)


def test_through_level2_requires_triforce_bit() -> None:
    """through=level2 is TF 0x02, not merely boom or Moon entry."""
    assert not level2_through_success(_l2_snap(room=0x4F, boom=1, triforce=0x01))
    assert not level2_boom_success(_l2_snap(room=0x4F, boom=0))
    assert level2_boom_success(_l2_snap(room=0x4F, boom=1))
    assert level2_through_success(_l2_snap(room=0x0D, boom=1, triforce=0x03))


def test_level2_tf_stages_follow_isolated_boss_path() -> None:
    names = [name for name, _, _ in level2_tf_stages()]
    assert names == [
        "bomb_north_4f",
        "clear3f",
        "enter_3e",
        "clear3e",
        "enter_2e",
        "clear2e",
        "enter_1e",
        "clear1e",
        "bomb_north_1e",
        "fight_dodongo",
        "collect_tf",
    ]


def test_spine_retopup_covers_first_l2_bomb_wall() -> None:
    """Power-on L2 entry is bombs=0; 0x6f north must get the Survival top-up."""
    names = [name for name, _, _ in level2_to_boom_stages()]
    assert "bomb_north_6f" in names
    assert "bomb_north_6f" in SPINE_BOMB_RETOPUP
    assert "bomb_north_5f" in SPINE_BOMB_RETOPUP
    tf_names = [name for name, _, _ in level2_tf_stages()]
    assert "bomb_north_4f" in tf_names
    assert "bomb_north_4f" in SPINE_BOMB_RETOPUP
    assert "bomb_north_1e" in SPINE_BOMB_RETOPUP
    assert "fight_dodongo" in SPINE_BOMB_RETOPUP


def test_merge_inventory_assist_appends_writes() -> None:
    first = {
        "writes": [{"field": "bombs", "from": 0, "to": 16}],
        "notes": ["bombs=16"],
        "poke_bombs": 16,
        "poke_keys": None,
    }
    extra = {
        "writes": [{"field": "keys", "from": 1, "to": 2}],
        "notes": ["keys=2"],
        "poke_bombs": 16,
        "poke_keys": 2,
    }
    merged = merge_inventory_assist(first, extra)
    assert len(merged["writes"]) == 2
    assert merged["notes"] == ["bombs=16", "keys=2"]
    assert merged["poke_bombs"] == 16
    assert merged["poke_keys"] == 2
    assert merge_inventory_assist(None, extra) is extra


def test_topup_owned_inventory_records_poke_on_run() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_BOMBS] = 0
    ram[ADDR_KEYS] = 1
    values: dict[str, int] = {}

    class _Data:
        memory = None

        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=_Data(), em=None),
    )
    run = SpineRun(through="level3", success=True, boot_frames=199)
    topup_owned_inventory(env, run)
    assert run.inventory_assist is not None
    assert run.inventory_assist["poke_bombs"] == 16
    assert run.inventory_assist["poke_keys"] == 2
    assert values["bombs"] == 16
    assert values["keys"] == 2
    report = run.report()
    assert report["poke_bombs"] == 16
    assert report["poke_keys"] == 2


def test_l3_boss_topup_preserves_carried_keys() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_BOMBS] = 8
    ram[ADDR_KEYS] = 4
    values: dict[str, int] = {}

    class _Data:
        memory = None

        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=_Data(), em=None),
    )
    run = SpineRun(through="level3", success=True, boot_frames=199)
    topup_owned_bombs(env, run)
    assert values["bombs"] == 16
    assert "keys" not in values
    assert run.inventory_assist["poke_bombs"] == 16
    assert run.inventory_assist["poke_keys"] is None


def test_spine_report_includes_l2_entry_bomb_plan() -> None:
    run = SpineRun(through="level2", success=True, boot_frames=199)
    run.l2_entry = spine_final_fields(_l2_snap(bombs=4, keys=0))
    from zelda_i.level2_bombs import spine_bomb_report

    run.bombs = spine_bomb_report(4, through="tf", bombs_out=1)
    run.inventory_assist = {
        "poke_bombs": 16,
        "poke_keys": 2,
        "writes": [{"field": "bombs", "from": 2, "to": 16}],
        "progression_writes": 0,
        "capacity_writes": 0,
    }
    report = run.report()
    assert report["poke_bombs"] == 16
    assert report["poke_keys"] == 2
    assert report["inventory_assist"]["poke_bombs"] == 16
    assert report["l2_entry"]["bombs"] == 4
    assert report["bombs"]["bombs_in"] == 4
    assert report["bombs"]["bombs_out"] == 1
    assert report["bombs"]["action"] == "farm"


def test_spine_final_fields_record_bombs_and_keys() -> None:
    fields = spine_final_fields(_l2_snap(room=0x4F, bombs=4, keys=2))
    assert fields["bombs"] == 4
    assert fields["keys"] == 2
    assert fields["room"] == 0x4F
    assert fields["level"] == 2
    assert fields["triforce"] == 0x01


def test_level2_entry_stages_settle_then_moon_door() -> None:
    names = [name for name, _, _ in level2_entry_stages()]
    assert names == ["settle_l1_tf", "enter_level2"]
    stages = {name: ctl for name, ctl, _ in level2_entry_stages()}
    assert isinstance(stages["settle_l1_tf"], PostTriforceSettleController)
    enter = stages["enter_level2"]
    assert isinstance(enter, OverworldToLevel2Controller)
    assert enter.door_path is True
    assert enter.require_dungeon is True


def test_level1_stages_survival_uses_off_wall_overlay() -> None:
    clean = {name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True)}
    survival = {
        name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True, survival=True)
    }
    assert "clear45_key" in clean
    assert clean["clear45_key"].spec is ROOM_45_SPEC
    assert survival["clear45_key"].spec is ROOM_45_SURVIVAL_SPEC
    assert ROOM_45_SPEC.combat.avoid_walls is False
    assert ROOM_45_SURVIVAL_SPEC.combat.avoid_walls is True
    assert ROOM_45_SPEC.reward.waypoints[0] == (160, 141)
    assert (152, 189) in ROOM_45_SPEC.reward.waypoints
    assert ROOM_45_SURVIVAL_SPEC.reward.waypoints[0] == (208, 157)
    assert (152, 189) in ROOM_45_SURVIVAL_SPEC.reward.waypoints
    assert (208, 189) in ROOM_45_SURVIVAL_SPEC.reward.waypoints


def test_survival_aquamentus_tanks_fireballs() -> None:
    clean = {name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True)}
    survival = {
        name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True, survival=True)
    }
    assert clean["aquamentus_heart"].tank_hits is False
    assert survival["aquamentus_heart"].tank_hits is True


def test_spine_boot_policy_is_first_slot_first_quest() -> None:
    assert BOOT_POLICY == {
        "file_slot": 1,
        "quest": 1,
        "playthrough": "first",
        "file_menu_select": False,
    }


def test_validate_l5_endpoint_requires_continuous_session() -> None:
    with pytest.raises(ValueError, match="continuous"):
        validate_l5_endpoint(
            {
                "ok": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 0, "capacity_writes": 0},
            }
        )
    with pytest.raises(ValueError, match="seamed"):
        validate_l5_endpoint(
            {
                "ok": True,
                "continuous_emulator_session": True,
                "seamed": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 0, "capacity_writes": 0},
            }
        )
    validate_l5_endpoint(
        {
            "ok": True,
            "continuous_emulator_session": True,
            "seamed": False,
            "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
            "assist": {"progression_writes": 0, "capacity_writes": 0},
        }
    )
    with pytest.raises(ValueError, match="progression writes"):
        validate_l5_endpoint(
            {
                "ok": True,
                "continuous_emulator_session": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 1, "capacity_writes": 0},
            }
        )
    validate_l5_endpoint(
        {
            "ok": True,
            "continuous_emulator_session": True,
            "seamed": False,
            "final": {"level": 5, "room": 0x14, "triforce": 0x1C},
            "assist": {"progression_writes": 0, "capacity_writes": 0},
        }
    )

def test_seamed_compose_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("zelda_i.scripts.compose_honest_route_recording")

