"""Unit tests for Level 4 live interior anchors (rr-5lu / rr-2ysf / rr-9so0)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.dungeon import ROOM_SPECS, ensure_default_specs
from zelda_i.dungeon_ids import VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE, object_name
from zelda_i.level4_dungeon import (
    BOMB_61_NORTH_STAND,
    BOMB_61_OPENS_TO,
    BombWall61North,
    COMPASS_PICKUP_XY,
    EXIT_60_HOLD,
    EXIT_60_SAMPLE_PATH,
    GEL_SPLIT_OBJECT_TYPE,
    INVULN_MOVER_TYPE,
    KEY_30_EAST_Y,
    KEY_40_PICKUP_XY,
    KEY_61_EAST_Y,
    KEY_61_OPENS_TO,
    LEVEL4_COMPASS_BIT,
    MAZE_31_EAST_X_MIN,
    MAZE_31_EAST_Y,
    MAZE_31_HOLD,
    MAZE_40_KEY_HOLD,
    MAZE_40_TO_KEY,
    MAZE_50_HOLD,
    MAZE_50_LONG_UP,
    MAZE_50_TO_NORTH,
    MAZE_62_RETURN_WEST,
    MAZE_62_TO_COMPASS,
    MAZE_IN_HOLD,
    MAZE_OUT_HOLD,
    POST_LADDER_ITEM_SETTLE,
    ROOM_L4_COMPASS_62,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_NORTH_30,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_ZOLS_40,
    LIKE_LIKE_OBJECT_TYPE,
    LADDER_60_PICKUP_XY,
    MAZE_60_HOLD,
    MAZE_60_TO_LADDER,
    PUSH_32_DIR,
    PUSH_32_STAND,
    ROOM_30_SPEC,
    ROOM_31_SPEC,
    ROOM_32_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_62_SPEC,
    ROOM_71_SPEC,
    ROOM_ITEM_COMPASS,
    ROOM_ITEM_STEPLADDER,
    ROOM_L4_STEPLADDER,
    STAIRS_32_APPROACH,
    VIRE_OBJECT_TYPE as L4_VIRE,
    WEST_31_HOLD,
    WEST_31_SAMPLE_PATH,
    ZOL_OBJECT_TYPE,
    GEL_OBJECT_TYPE,
    LEVEL4_MAP_BIT,
    MAP_21_HOLD,
    MAP_21_PICKUP_XY,
    MAP_21_SAMPLE_PATH,
    ROOM_ITEM_MAP,
    ROOM_L4_MAP_21,
    ROOM_L4_WATER_NORTH_20,
    RIGHT_20_STAND,
    level4_map_success,
    level4_map_room_success,
    make_bomb_61_north_controller,
    make_compass_62_controller,
    make_entry_up_controller,
    make_key_right_31_controller,
    make_key_right_62_controller,
    make_left_50_controller,
    make_north_30_controller,
    make_north_40_controller,
    make_room_30_clear_controller,
    make_room_31_clear_controller,
    make_room_32_clear_controller,
    make_room_40_clear_controller,
    make_room_40_key_controller,
    make_room_50_clear_controller,
    make_room_51_key_controller,
    make_room_61_clear_controller,
    make_room_62_clear_controller,
    make_stepladder_controller,
    planning_interior_report,
)
from zelda_i.level4_overworld import LEVEL4


def test_live_room_ids() -> None:
    assert ROOM_L4_ENTRY == 0x71
    assert ROOM_L4_VIRES_61 == 0x61
    assert ROOM_L4_KEESE_KEY_51 == 0x51
    assert ROOM_L4_VIRES_50 == 0x50
    assert ROOM_L4_COMPASS_62 == 0x62
    assert ROOM_L4_ZOLS_40 == 0x40
    assert ROOM_L4_NORTH_30 == 0x30
    assert ROOM_L4_EAST_31 == 0x31
    assert ROOM_L4_EAST_32 == 0x32
    assert ROOM_L4_STEPLADDER == 0x60
    assert ROOM_L4_WATER_NORTH_20 == 0x20
    assert ROOM_L4_MAP_21 == 0x21
    assert L4_VIRE == VIRE_OBJECT_TYPE == 0x12
    assert GEL_OBJECT_TYPE == 0x15
    assert ROOM_ITEM_MAP == 0x17
    assert LEVEL4_MAP_BIT == 0x08
    assert MAP_21_HOLD == 6
    assert MAP_21_PICKUP_XY == (208, 181)
    assert len(MAP_21_SAMPLE_PATH) >= 20
    assert RIGHT_20_STAND == (208, 141)
    # Gleeok approach anchors (rr-rvae dual-green enter)
    from zelda_i.level4_dungeon import (
        BOMB_21_NORTH_STAND,
        BOMB_21_OPENS_TO,
        GLEEOK_OBJECT_TYPE,
        MID_11_OBJECT_TYPE,
        PATH_12_TO_GLEEOK,
        PUSH_12_BLOCK_FROM,
        PUSH_12_BLOCK_TO,
        PUSH_12_DIR,
        PUSH_12_STAND,
        RIGHT_12_HOLD,
        ROOM_12_SPEC,
        ROOM_ITEM_HEART_CONTAINER,
        ROOM_L4_GLEEOK_13,
        ROOM_L4_KEY_01,
        ROOM_L4_MANHANDLA_10,
        ROOM_L4_MID_11,
        ROOM_L4_TRAPS_02,
        ROOM_L4_VIRES_12,
        make_room_12_clear_controller,
    )

    assert ROOM_L4_MID_11 == 0x11
    assert ROOM_L4_KEY_01 == 0x01
    assert ROOM_L4_VIRES_12 == 0x12
    assert ROOM_L4_TRAPS_02 == 0x02
    assert ROOM_L4_GLEEOK_13 == 0x13
    assert ROOM_L4_MANHANDLA_10 == 0x10
    assert BOMB_21_OPENS_TO == 0x11
    assert BOMB_21_NORTH_STAND == (120, 105)
    assert GLEEOK_OBJECT_TYPE == 0x43
    assert MID_11_OBJECT_TYPE == 0x35
    assert ROOM_ITEM_HEART_CONTAINER == 0x1A
    assert PUSH_12_STAND == (112, 144)
    assert PUSH_12_DIR == "LEFT"
    assert PUSH_12_BLOCK_FROM == (96, 144)
    assert PUSH_12_BLOCK_TO == (80, 144)
    assert RIGHT_12_HOLD == 4
    assert len(PATH_12_TO_GLEEOK) == 31
    assert PATH_12_TO_GLEEOK[0] == "RIGHT"
    assert PATH_12_TO_GLEEOK[-1] == "RIGHT"
    assert ROOM_12_SPEC.room_id == 0x12
    assert ROOM_12_SPEC.expected_enemy_count == 5
    clear12 = make_room_12_clear_controller()
    assert clear12.spec is ROOM_12_SPEC
    rep = planning_interior_report()
    assert rep["status"] == "gleeok_tf08_dual_green"
    assert "0x13" in rep["live_graph"]
    assert rep["live_graph"]["0x21"]["BOMB_UP"] == "0x11"
    assert rep["right_13"]["dual_green"] is True
    assert rep["right_13"]["path_len"] == 31
    assert rep["gleeok_tf"]["dual_green"] is True
    assert rep["gleeok_tf"]["tf_bit"] == "0x8"
    assert rep["gleeok_tf"]["tf_room"] == "0x3"
    from zelda_i.level4_dungeon import (
        GLEEOK_HEAD_OBJECT_TYPE,
        LEVEL4_TRIFORCE_BIT,
        ROOM_L4_TRIFORCE,
        level4_triforce_stop,
    )
    from zelda_i.level4_boss_combat import (
        FIREBALL_DODGE_DIST,
        GLEEOK_HEAD_OBJECT_TYPE as HEAD_BOSS,
        Level4GleeokFightController,
        STAND_DY,
        level4_complete_success,
        level4_tf08,
        make_gleeok_fight_controller,
    )
    import numpy as np

    assert GLEEOK_HEAD_OBJECT_TYPE == HEAD_BOSS == 0x46
    assert ROOM_L4_TRIFORCE == 0x03
    assert LEVEL4_TRIFORCE_BIT == 0x08
    assert STAND_DY == 22  # rr-vdnc Clean south-stand
    assert FIREBALL_DODGE_DIST == 14
    ctl = make_gleeok_fight_controller(tag="unit")
    assert isinstance(ctl, Level4GleeokFightController)
    assert ctl.stand_dy == STAND_DY
    ram = np.zeros(0x800, dtype=np.uint8)
    assert level4_complete_success(ram) is False
    assert level4_tf08(ram) is False
    ram[0x0671] = 0x08
    assert level4_tf08(ram) is True
    assert level4_complete_success(ram) is True
    from zelda_i.ram import read_snapshot

    snap = read_snapshot(ram)
    assert level4_triforce_stop(snap) is True


def test_map_success_predicates_false_on_zeros() -> None:
    import numpy as np

    ram = np.zeros(0x800, dtype=np.uint8)
    assert level4_map_success(ram) is False
    assert level4_map_room_success(ram) is False
    assert VIRE_SPLIT_KEESE_TYPE == 0x1C
    assert ZOL_OBJECT_TYPE == 0x13
    assert GEL_SPLIT_OBJECT_TYPE == 0x14
    assert LIKE_LIKE_OBJECT_TYPE == 0x17
    assert INVULN_MOVER_TYPE == 0x2B
    assert ROOM_ITEM_COMPASS == 0x16
    assert ROOM_ITEM_STEPLADDER == 0x0D
    assert LEVEL4_COMPASS_BIT == 0x08
    assert KEY_61_EAST_Y == 141
    assert KEY_61_OPENS_TO == 0x62
    assert KEY_30_EAST_Y == 141
    assert KEY_40_PICKUP_XY == (120, 117)
    assert PUSH_32_STAND == (120, 141)
    assert PUSH_32_DIR == "LEFT"
    assert STAIRS_32_APPROACH == (208, 96)
    assert LADDER_60_PICKUP_XY == (136, 141)
    assert POST_LADDER_ITEM_SETTLE >= 100
    assert EXIT_60_HOLD == 4
    assert len(EXIT_60_SAMPLE_PATH) >= 40
    assert WEST_31_HOLD == 4
    assert len(WEST_31_SAMPLE_PATH) >= 10
    assert EXIT_60_SAMPLE_PATH[0] == "RIGHT"
    assert WEST_31_SAMPLE_PATH[0] == "LEFT"


def test_object_names() -> None:
    assert object_name(0x12) == "vire"
    assert object_name(0x1C) == "vire_split_keese"
    assert object_name(0x13) == "zol"
    assert object_name(0x14) == "gel_or_zol_split_residual"
    assert object_name(0x17) == "like_like"


def test_bomb_wall_geometry() -> None:
    wall = BombWall61North()
    assert wall.room == 0x61
    assert wall.stand == BOMB_61_NORTH_STAND == (120, 105)
    assert wall.face == "UP"
    assert wall.opens_to == BOMB_61_OPENS_TO == 0x51


def test_specs_register() -> None:
    ensure_default_specs()
    assert ROOM_71_SPEC.level == LEVEL4
    assert ROOM_61_SPEC.enemy_types[0] == 0x12
    assert VIRE_SPLIT_KEESE_TYPE in ROOM_61_SPEC.type_only_enemy_types
    assert ROOM_61_SPEC.object_slot_max == 12
    assert ROOM_51_SPEC.room_item_id == 0x19
    assert ROOM_50_SPEC.room_id == 0x50
    assert ROOM_50_SPEC.expected_enemy_count == 5
    assert ROOM_62_SPEC.room_id == 0x62
    assert ROOM_62_SPEC.room_item_id == 0x16
    assert ROOM_40_SPEC.room_id == 0x40
    assert ROOM_40_SPEC.enemy_types[0] == ZOL_OBJECT_TYPE
    assert GEL_SPLIT_OBJECT_TYPE in ROOM_40_SPEC.enemy_types
    assert GEL_SPLIT_OBJECT_TYPE in ROOM_40_SPEC.type_only_enemy_types
    assert ROOM_40_SPEC.room_item_id == 0x19
    assert ROOM_40_SPEC.reward.settle_all_dead == 0
    assert ROOM_30_SPEC.room_id == 0x30
    assert ROOM_30_SPEC.expected_enemy_count == 3
    assert ROOM_30_SPEC.reward.settle_all_dead == 0
    assert INVULN_MOVER_TYPE not in ROOM_30_SPEC.enemy_types
    assert ROOM_31_SPEC.room_id == 0x31
    assert ROOM_31_SPEC.expected_enemy_count == 5
    assert ROOM_31_SPEC.reward.settle_all_dead == 0
    assert ROOM_32_SPEC.room_id == 0x32
    assert ROOM_32_SPEC.expected_enemy_count == 4
    assert LIKE_LIKE_OBJECT_TYPE in ROOM_32_SPEC.enemy_types
    assert ZOL_OBJECT_TYPE in ROOM_32_SPEC.enemy_types
    assert ROOM_32_SPEC.reward.settle_all_dead == 0
    assert ROOM_SPECS[0x71] is ROOM_71_SPEC
    assert ROOM_SPECS[0x61] is ROOM_61_SPEC
    assert ROOM_SPECS[0x51] is ROOM_51_SPEC
    assert ROOM_SPECS[0x50] is ROOM_50_SPEC
    assert ROOM_SPECS[0x62] is ROOM_62_SPEC
    assert ROOM_SPECS[0x40] is ROOM_40_SPEC
    assert ROOM_SPECS[0x30] is ROOM_30_SPEC
    assert ROOM_SPECS[0x31] is ROOM_31_SPEC
    assert ROOM_SPECS[0x32] is ROOM_32_SPEC


def test_factories() -> None:
    up = make_entry_up_controller()
    assert up.max_frames > 0
    clear = make_room_61_clear_controller()
    assert clear.spec is ROOM_61_SPEC
    bomb = make_bomb_61_north_controller(clear_vires=True)
    assert bomb.level == LEVEL4
    assert bomb.to_room == 0x51
    assert bomb.clear_spec is ROOM_61_SPEC
    key = make_room_51_key_controller()
    assert key.spec is ROOM_51_SPEC
    left = make_left_50_controller()
    assert left.max_frames > 0
    c50 = make_room_50_clear_controller()
    assert c50.spec is ROOM_50_SPEC
    kr = make_key_right_62_controller(clear_vires=True)
    assert kr.clear_vires is True
    c62 = make_room_62_clear_controller()
    assert c62.spec is ROOM_62_SPEC
    compass = make_compass_62_controller()
    assert compass.max_frames > 0
    assert compass.phase.name == "MAZE_IN"
    north = make_north_40_controller()
    assert north.max_frames > 0
    assert north.phase.name == "WAYPOINTS"
    c40 = make_room_40_clear_controller()
    assert c40.spec is ROOM_40_SPEC
    k40 = make_room_40_key_controller()
    assert k40.max_frames > 0
    assert k40.phase.name == "FIGHT"
    n30 = make_north_30_controller()
    assert n30.max_frames > 0
    assert n30.phase.name == "ALIGN"
    c30 = make_room_30_clear_controller()
    assert c30.max_frames > 0
    assert c30.phase.name == "TO_BAND"
    kr31 = make_key_right_31_controller(clear_vires=True)
    assert kr31.clear_vires is True
    assert kr31.phase.name == "CLEAR"
    c31 = make_room_31_clear_controller()
    assert c31.spec is ROOM_31_SPEC
    c32 = make_room_32_clear_controller()
    assert c32.spec is ROOM_32_SPEC
    ladder = make_stepladder_controller(clear_first=True)
    assert ladder.max_frames > 0
    assert ladder.phase.name == "CLEAR"
    preclear = make_stepladder_controller(clear_first=False)
    assert preclear.phase.name == "ALIGN_PUSH"
    assert preclear.clear_first is False


def test_stepladder_settle_walks_west_aisle_to_spawn() -> None:
    """Continuous 0x60 leftover (48,133) must PATH south to the notch, not hunt."""
    import numpy as np

    from zelda_i.level4_stepladder import MAZE_60_SETTLE, StepladderPhase
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        read_snapshot,
    )

    ctl = make_stepladder_controller(clear_first=False)
    ctl.phase = StepladderPhase.SETTLE_STAIRS
    ctl.phase_frames = MAZE_60_SETTLE
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 133
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "path_from_west_aisle"
    assert ctl.phase is StepladderPhase.PATH
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_clip_y"
    ram[ADDR_LINK_Y] = 69
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_clip_y"
    ne = make_stepladder_controller(clear_first=False)
    ne.phase = StepladderPhase.SETTLE_STAIRS
    ne.phase_frames = MAZE_60_SETTLE
    ram[ADDR_LINK_X] = 208
    ram[ADDR_LINK_Y] = 93
    act = ne.step(read_snapshot(ram))
    assert act.reason == "wait_nw_resettle"
    assert ne.phase is StepladderPhase.SETTLE_STAIRS


def test_stepladder_notch_stall_fails_without_bfs() -> None:
    """SW-notch miss fail-closes; do not fall through to emulator BFS."""
    import numpy as np

    from zelda_i.level4_occupancy import ROOM_60_CLIP_BUDGET
    from zelda_i.level4_stepladder import StepladderPhase
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        read_snapshot,
    )

    ctl = make_stepladder_controller(clear_first=False)
    ctl.phase = StepladderPhase.PATH
    ctl._last_xy = (48, 161)
    ctl._stall = ROOM_60_CLIP_BUDGET
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 161
    act = ctl.step(read_snapshot(ram))
    assert ctl.phase is StepladderPhase.FAILED
    assert act.reason.startswith("notch161_solid_48_161")


def test_maze_62_paths() -> None:
    assert MAZE_IN_HOLD == 6
    assert MAZE_OUT_HOLD == 4
    assert MAZE_62_TO_COMPASS[0] == "DOWN"
    assert "RIGHT" in MAZE_62_TO_COMPASS
    assert MAZE_62_RETURN_WEST[0] == "DOWN"
    assert MAZE_62_RETURN_WEST.count("LEFT") >= 10


def test_maze_50_north_path() -> None:
    from zelda_i.level4_dungeon import MAZE_50_WAYPOINTS

    assert MAZE_50_HOLD == 6
    assert MAZE_50_LONG_UP >= 100
    assert MAZE_50_TO_NORTH[0] == "DOWN"
    assert MAZE_50_TO_NORTH.count("UP") >= 10
    assert MAZE_50_TO_NORTH.count("LEFT") >= 5
    assert MAZE_50_WAYPOINTS[0][1] >= 170  # south first
    assert MAZE_50_WAYPOINTS[-1][1] <= 72  # north band
    assert make_north_40_controller().phase.name == "WAYPOINTS"
    assert COMPASS_PICKUP_XY == (136, 132)


def test_maze_50_live_corner_turns_at_observed_coordinates() -> None:
    def snap(x: int, y: int) -> SimpleNamespace:
        return SimpleNamespace(
            level=4, screen=ROOM_L4_VIRES_50, mode=5,
            transitioning=False, link_x=x, link_y=y,
        )

    below_corner = make_north_40_controller()
    below_corner.path_index = 4
    assert below_corner.step(snap(128, 101)).reason == "maze50_seek_4_UP"

    at_corner = make_north_40_controller()
    at_corner.path_index = 4
    assert at_corner.step(snap(128, 93)).reason == "maze50_seek_4_LEFT"

    door_column = make_north_40_controller()
    door_column.path_index = 4
    assert door_column.step(snap(120, 93)).reason == "maze50_seek_4_UP"


def test_maze_40_key_path() -> None:
    assert MAZE_40_KEY_HOLD == 6
    assert MAZE_40_TO_KEY[0] == "UP"
    assert MAZE_40_TO_KEY.count("RIGHT") >= 4
    assert MAZE_40_TO_KEY.count("LEFT") >= 4
    assert make_room_40_key_controller().phase.name == "FIGHT"
    assert KEY_40_PICKUP_XY == (120, 117)


def test_north_30_from_room40_key_pose_aligns_then_pushes_up() -> None:
    """Continuous v7 leftover is (136,125); isolated north_30 used that pose."""

    def snap(x: int, y: int, *, screen: int = ROOM_L4_ZOLS_40) -> SimpleNamespace:
        return SimpleNamespace(
            level=4, screen=screen, mode=5,
            transitioning=False, link_x=x, link_y=y,
        )

    from_key = make_north_30_controller()
    assert from_key.step(snap(136, 125)).reason == "align_x"

    door_column = make_north_30_controller()
    assert door_column.step(snap(120, 125)).reason == "push_up_north"

    north_band = make_north_30_controller()
    assert north_band.step(snap(120, 68)).reason == "push_up_north"

    entered = make_north_30_controller()
    action = entered.step(snap(120, 205, screen=ROOM_L4_NORTH_30))
    assert action.reason == "done"
    assert entered.success


def test_clear_30_from_south_mouth_walks_north_band() -> None:
    """Continuous leftover is (120,205); isolated clear_30 used that pose."""

    def snap(x: int, y: int):
        return SimpleNamespace(
            level=4, screen=ROOM_L4_NORTH_30, mode=5,
            transitioning=False, link_x=x, link_y=y, objects=(),
        )

    south = make_room_30_clear_controller()
    assert south.step(snap(120, 205)).reason == "walk_north_band"

    on_band = make_room_30_clear_controller()
    action = on_band.step(snap(120, 140))
    assert on_band.phase.name == "FIGHT"
    assert action.reason == "wait_spawn"


def test_key_right_31_aligns_y141_from_clear_leftover() -> None:
    """Isolated clear leftover is (80,133); KEY-RIGHT is y=141 then RIGHT."""

    def snap(x: int, y: int, *, screen: int = ROOM_L4_NORTH_30, keys: int = 6):
        return SimpleNamespace(
            level=4, screen=screen, mode=5,
            transitioning=False, link_x=x, link_y=y, keys=keys, objects=(),
        )

    from_band = make_key_right_31_controller(clear_vires=False)
    assert from_band.step(snap(80, 133)).reason == "align_y"

    at_door = make_key_right_31_controller(clear_vires=False)
    assert at_door.step(snap(80, 141)).reason == "push_key_right"

    entered = make_key_right_31_controller(clear_vires=False)
    action = entered.step(snap(16, 141, screen=ROOM_L4_EAST_31, keys=5))
    assert action.reason == "done"
    assert entered.success


def test_room_31_west_alcove_clip_is_right_up() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4_maze_path import make_maze_31_inland_controller

    def snap(x: int, y: int, *, screen: int = ROOM_L4_EAST_31):
        return SimpleNamespace(
            mode=5, level=4, screen=screen, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    alcove = make_maze_31_inland_controller()
    action = alcove.step(snap(32, 141))
    assert action.reason == "maze31_alcove_clip"
    assert list(action.action) == list(nes_action("RIGHT", "UP"))

    corridor = make_maze_31_inland_controller()
    first = corridor.step(snap(48, 133))
    assert first.reason == "maze31_thread_UP"
    assert not corridor.success
    north = make_maze_31_inland_controller()
    assert north.step(snap(48, 109)).reason == "maze31_thread_RIGHT"
    corner = make_maze_31_inland_controller()
    assert corner.step(snap(80, 109)).reason == "maze31_thread_DOWN"
    drop = make_maze_31_inland_controller()
    assert drop.step(snap(80, 141)).reason == "maze31_thread_DOWN"

    inland = make_maze_31_inland_controller()
    done = inland.step(snap(128, 133))
    assert done.reason == "done"
    assert inland.success
    assert list(done.action) == list(nes_idle_action())


def test_room_31_east_leftover_goes_up_not_through_water() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4_maze_path import make_maze_31_east_controller

    def snap(x: int, y: int, *, screen: int = ROOM_L4_EAST_31):
        return SimpleNamespace(
            mode=5, level=4, screen=screen, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    leftover = make_maze_31_east_controller()
    action = leftover.step(snap(112, 141))
    assert action.reason == "maze31_east_join_UP"
    assert list(action.action) == list(nes_action("UP"))

    clip = make_maze_31_east_controller()
    clipped = clip.step(snap(112, 113))
    assert clipped.reason == "maze31_east_se_clip"
    assert list(clipped.action) == list(nes_action("RIGHT", "DOWN"))

    band = make_maze_31_east_controller()
    assert band.step(snap(200, 136)).reason == "maze31_east_push"
    assert list(band.step(snap(200, 136)).action) == list(nes_action("RIGHT"))

    entered = make_maze_31_east_controller()
    done = entered.step(snap(16, 141, screen=ROOM_L4_EAST_32))
    assert done.reason == "done"
    assert entered.success
    assert list(done.action) == list(nes_idle_action())


def test_room_31_west_door_occupancy_includes_alcove() -> None:
    """Continuous leftover (16,141)/(33,141) is west of default xmin=40."""
    from zelda_i.walk_physics import OccupancyWalker

    assert ROOM_31_SPEC.combat.occupancy_patrol is True
    assert ROOM_31_SPEC.combat.occupancy_bounds is not None
    assert ROOM_31_SPEC.combat.occupancy_bounds[0] <= 16
    default = OccupancyWalker()
    assert not default.grid.in_bounds(16, 141)
    assert not default.grid.in_bounds(33, 141)
    ctl = make_room_31_clear_controller()
    assert ctl.walker.grid.in_bounds(16, 141)
    assert ctl.walker.grid.in_bounds(33, 141)
    direction = ctl.walker.next_dir((33, 141), (64, 109))
    assert direction in ("UP", "RIGHT", "DOWN")


def test_room_30_live_enemies_ignore_invuln_0x2b() -> None:
    invuln = SimpleNamespace(slot=1, type_id=INVULN_MOVER_TYPE, x=80, y=133, hp=64)
    vire = SimpleNamespace(slot=3, type_id=VIRE_OBJECT_TYPE, x=160, y=100, hp=64)
    assert ROOM_30_SPEC.live_enemies(SimpleNamespace(objects=(invuln,))) == ()
    live = ROOM_30_SPEC.live_enemies(SimpleNamespace(objects=(invuln, vire)))
    assert len(live) == 1
    assert live[0].type_id == VIRE_OBJECT_TYPE


def test_room_32_live_enemies_ignore_invuln_0x2b_and_block_0x68() -> None:
    invuln = SimpleNamespace(slot=1, type_id=INVULN_MOVER_TYPE, x=80, y=133, hp=64)
    block = SimpleNamespace(slot=2, type_id=0x68, x=80, y=144, hp=0)
    zol = SimpleNamespace(slot=3, type_id=ZOL_OBJECT_TYPE, x=160, y=100, hp=64)
    like = SimpleNamespace(slot=4, type_id=LIKE_LIKE_OBJECT_TYPE, x=100, y=141, hp=64)
    assert ROOM_32_SPEC.live_enemies(SimpleNamespace(objects=(invuln, block))) == ()
    live = ROOM_32_SPEC.live_enemies(SimpleNamespace(objects=(invuln, block, zol, like)))
    types = {obj.type_id for obj in live}
    assert types == {ZOL_OBJECT_TYPE, LIKE_LIKE_OBJECT_TYPE}


def test_planning_interior_report() -> None:
    r = planning_interior_report()
    assert r["bead"] == "rr-5lu"
    assert r["tip"] == "rr-rvae"
    assert r["map_21"]["room"] == "0x21"
    assert r["map_21"]["bead"] == "rr-rvae"
    assert r["live_graph"]["0x20"]["RIGHT_after_clear"] == "0x21"
    assert r["live_graph"]["0x21"]["map_bit"] == "0x8"
    assert r["entry_room"] == "0x71"
    assert r["live_graph"]["0x71"]["UP"] == "0x61"
    assert r["live_graph"]["0x61"]["BOMB_UP"] == "0x51"
    assert r["live_graph"]["0x61"]["KEY_RIGHT"] == "0x62"
    assert r["live_graph"]["0x61"]["RIGHT_reenter"] == "0x62"
    assert r["live_graph"]["0x51"]["LEFT"] == "0x50"
    assert r["live_graph"]["0x51"]["UP"] == "sealed"
    assert r["live_graph"]["0x51"]["RIGHT"] == "sealed"
    assert r["live_graph"]["0x50"]["UP_scripted"] == "0x40"
    assert r["live_graph"]["0x40"]["DOWN"] == "0x50"
    assert r["live_graph"]["0x40"]["UP"] == "0x30"
    assert r["live_graph"]["0x40"]["enemies"]["0x13"] == 5
    assert r["live_graph"]["0x30"]["DOWN"] == "0x40"
    assert r["live_graph"]["0x30"]["KEY_RIGHT"] == "0x31"
    assert r["live_graph"]["0x31"]["enemies"]["0x12"] == 5
    assert r["live_graph"]["0x31"]["RIGHT_after_clear"] == "0x32"
    assert r["live_graph"]["0x32"]["LEFT"] == "0x31"
    assert r["live_graph"]["0x32"]["push_left_stairs"] == "0x60"
    assert r["live_graph"]["0x60"]["room_item"] == "0xd"
    assert r["live_graph"]["0x62"]["room_item"] == "0x16"
    assert r["live_graph"]["0x62"]["compass_bit"] == "0x8"
    assert r["segments"]["clear_vires_61"] == "rr-yr77"
    assert r["segments"]["clear_50"] == "rr-2ysf"
    assert r["segments"]["key_right_62"] == "rr-2ysf"
    assert r["segments"]["compass_62"] == "rr-9so0"
    assert r["segments"]["north_40"] == "rr-xc3x"
    assert r["segments"]["key_40"] == "rr-q8eq"
    assert r["segments"]["north_30"] == "rr-q8eq"
    assert r["segments"]["clear_30"] == "rr-n1wn"
    assert r["segments"]["key_right_31"] == "rr-n1wn"
    assert r["segments"]["clear_31"] == "rr-resv"
    assert r["segments"]["east_32"] == "rr-resv"
    assert r["segments"]["clear_32"] == "rr-tib8"
    assert r["segments"]["stepladder"] == "rr-tib8"
    assert r["segments"]["stepladder_path"] == "rr-tib8"
    assert r["post_compass"]["bead"] == "rr-o0nn"
    assert r["post_compass"]["first_outside"] == "0x40"
    assert r["post_compass"]["next_outside"] == "0x30"
    assert r["post_compass"]["ladder"] == 1
    assert "0x62" in r["post_compass"]["early_component"]
    assert r["key_61_east"]["opens_to"] == "0x62"
    assert r["maze_62"]["pickup_xy"] == [136, 132]
    assert r["maze_50_north"]["opens_to"] == "0x40"
    assert r["key_40"]["opens_north"] == "0x30"
    assert r["key_right_31"]["opens_to"] == "0x31"
    assert r["key_right_31"]["y"] == 141
    assert r["clear_31"]["checkpoint"] == "Level4Room31Cleared"
    assert r["east_32"]["opens_to"] == "0x32"
    assert r["east_32"]["hold"] == MAZE_31_HOLD == 4
    assert MAZE_31_EAST_X_MIN == 200
    assert MAZE_31_EAST_Y == 136
    assert r["clear_32"]["checkpoint"] == "Level4Room32Cleared"
    assert r["stepladder"]["checkpoint"] == "Level4Stepladder"
    assert r["stepladder"]["stairs_room"] == "0x60"
    assert r["stepladder"]["path_hold"] == MAZE_60_HOLD == 4
    assert r["stepladder"]["path_len"] == len(MAZE_60_TO_LADDER)
    assert MAZE_60_TO_LADDER[0] == "UP"
    assert MAZE_60_TO_LADDER.count("DOWN") >= 10
    assert MAZE_60_TO_LADDER.count("RIGHT") >= 10
