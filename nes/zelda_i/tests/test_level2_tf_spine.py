"""Unit tests for L2 boom → Dodongo → TF spine table (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.bomb_wall_path import BombWallController
from zelda_i.dungeon import GenericDungeonRoomController, RewardKind
from zelda_i.level2_spine import Level2RoomWalkController
from zelda_i.level2_enter_1e import Level2Enter1eController
from zelda_i.level2_tf_spine import (
    DodongoPhase,
    Level2ClearDoorController,
    Level2DodongoController,
    Level2SouthBandUpController,
    Level2ToSouthCenterController,
    Level2TfCollectController,
    ROOM_3E_SPINE_SPEC,
    SOUTH_BAND_Y,
    SPINE_TF_BOMB_POKE,
    SPINE_TF_KEY_POKE,
    SouthBandUpPhase,
    level2_boom_owned,
    level2_tf_stages,
    level2_through_success,
)
from retro_harness.nes import nes_action
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_CUR_OPENED_DOORS,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    room: int = 0x4F,
    x: int = 120,
    y: int = 141,
    bombs: int = 8,
    keys: int = 2,
    boom: int = 1,
    triforce: int = 0x01,
    mode: int = PLAY_MODE,
    dodo_hp: int | None = None,
    doors: int = 0,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_MAGIC_BOOMERANG] = boom
    ram[ADDR_TRIFORCE] = triforce
    ram[ADDR_HEALTH] = 0x2F
    ram[ADDR_CUR_OPENED_DOORS] = doors
    if dodo_hp is not None:
        ram[ADDR_OBJ_TYPE + 1] = 0x32
        ram[ADDR_LINK_X + 1] = 140
        ram[ADDR_LINK_Y + 1] = 141
        ram[ADDR_OBJ_HP + 1] = dodo_hp
    return read_snapshot(ram)


def test_tf_stage_names_and_types() -> None:
    stages = list(level2_tf_stages())
    names = [name for name, _, _ in stages]
    assert names[0] == "bomb_north_4f"
    assert names[-2] == "fight_dodongo"
    assert names[-1] == "collect_tf"
    by_name = {name: ctl for name, ctl, _ in stages}
    assert isinstance(by_name["bomb_north_4f"], BombWallController)
    assert by_name["bomb_north_4f"].from_room == 0x4F
    assert by_name["bomb_north_4f"].to_room == 0x3F
    assert isinstance(by_name["clear3f"], GenericDungeonRoomController)
    assert isinstance(by_name["enter_3e"], Level2RoomWalkController)
    assert isinstance(by_name["enter_2e"], Level2SouthBandUpController)
    assert by_name["enter_2e"].dest_room == 0x2E
    assert isinstance(by_name["enter_1e"], Level2Enter1eController)
    assert by_name["enter_1e"].dest_room == 0x1E
    from zelda_i.level2_bomb_path import Level2BombNorth1eSpineController

    assert isinstance(by_name["bomb_north_1e"], Level2BombNorth1eSpineController)
    assert by_name["bomb_north_1e"].to_room == 0x0E
    assert by_name["bomb_north_1e"].approach_waypoints == (
        (96, 117),
        (96, 101),
        (120, 101),
    )
    assert isinstance(by_name["clear2e"], Level2ClearDoorController)
    assert by_name["clear2e"].door_bit == 0x08
    assert isinstance(by_name["fight_dodongo"], Level2DodongoController)
    assert isinstance(by_name["collect_tf"], Level2TfCollectController)


def test_3e_spine_spec_is_clear_only() -> None:
    assert ROOM_3E_SPINE_SPEC.reward.kind is RewardKind.CLEAR_ONLY
    assert ROOM_3E_SPINE_SPEC.expected_enemy_count == 1


def test_through_success_is_tf_bit() -> None:
    assert not level2_through_success(_snap(room=0x4F, boom=1, triforce=0x01))
    assert level2_boom_owned(_snap(room=0x4F, boom=1))
    assert level2_through_success(_snap(room=0x0D, triforce=0x03))


def test_south_band_dives_then_aligns_then_up() -> None:
    ctl = Level2SouthBandUpController(dest_room=0x2E)
    act = ctl.step(_snap(room=0x3E, x=80, y=141))
    assert act.reason == "diamond_free"
    act = ctl.step(_snap(room=0x3E, x=48, y=141))
    assert act.reason == "side_north"
    act = ctl.step(_snap(room=0x3E, x=48, y=101))
    assert act.reason == "north_align_x"
    act = ctl.step(_snap(room=0x3E, x=120, y=109))
    assert act.reason == "push_up"
    act = ctl.step(_snap(room=0x2E, x=120, y=205))
    assert ctl.success is True
    assert ctl.phase is SouthBandUpPhase.DONE
    assert act.reason == "done"


def test_south_band_frees_live_timeout_poses() -> None:
    """v1 (120,185) UP-hold; v2 (154,141) DOWN into the diamond."""
    south = Level2SouthBandUpController(dest_room=0x2E)
    act = south.step(_snap(room=0x3E, x=120, y=185))
    assert act.reason == "south_band"
    mid = Level2SouthBandUpController(dest_room=0x2E)
    act = mid.step(_snap(room=0x3E, x=154, y=141))
    assert act.reason == "diamond_free"
    assert act.action == nes_action("RIGHT")
    ne = Level2SouthBandUpController(dest_room=0x2E)
    act = ne.step(_snap(room=0x3E, x=175, y=109))
    assert act.reason == "north_align_x"
    assert act.action == nes_action("LEFT")


def test_enter_1e_gutter_tries_right_then_clips() -> None:
    """v5 LEFT / v6 DOWN / v7 UP no-op at (96,141). Occupancy RIGHT, then clip."""
    ctl = Level2Enter1eController()
    snap = _snap(room=0x2E, x=96, y=141)
    act = ctl.step(snap)
    assert act.reason == "gutter_right"
    assert list(act.action) == list(nes_action("RIGHT"))
    act = ctl.step(snap)
    assert act.reason == "gutter_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))


def test_bomb_north_1e_peels_west_from_north_pinch() -> None:
    """v8 leftover (120, 117): do not UP into the closed bomb wall."""
    from zelda_i.bomb_wall_path import BombWallPhase
    from zelda_i.level2_bomb_path import Level2BombNorth1eSpineController

    ctl = Level2BombNorth1eSpineController()
    act = ctl.step(_snap(room=0x1E, x=120, y=117, bombs=16))
    assert ctl.phase is BombWallPhase.SOUTH_BAND
    assert act.reason == "approach_x"
    assert list(act.action) == list(nes_action("LEFT"))


def test_bomb_north_1e_clips_right_up_at_stand_y() -> None:
    """v9 leftover (96, 101): cardinal RIGHT is solid."""
    from zelda_i.level2_bomb_path import Level2BombNorth1eSpineController

    ctl = Level2BombNorth1eSpineController()
    act = ctl.step(_snap(room=0x1E, x=96, y=101, bombs=16))
    assert act.reason == "stand_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))


def test_enter_1e_north_band_pushes_up() -> None:
    ctl = Level2Enter1eController()
    act = ctl.step(_snap(room=0x2E, x=120, y=93))
    assert act.reason == "push_up"
    assert list(act.action) == list(nes_action("UP"))
    act = ctl.step(_snap(room=0x1E, x=120, y=205))
    assert ctl.success is True
    assert act.reason == "done"


def test_dodongo_fails_without_bombs() -> None:
    ctl = Level2DodongoController(settle_frames=1)
    ctl.step(_snap(room=0x0E, bombs=0, dodo_hp=0x20))
    act = ctl.step(_snap(room=0x0E, bombs=0, dodo_hp=0x20))
    assert ctl.phase is DodongoPhase.FAILED
    assert act.reason == "out_of_bombs"
    assert ctl.success is False


def test_dodongo_controller_does_not_poke() -> None:
    ctl = Level2DodongoController()
    assert ctl.report()["poke"] is False


def test_prep_1e_takes_east_aisle_from_north() -> None:
    """v6/v7 west-column DOWN from the north band is solid."""
    pocket = Level2ToSouthCenterController()
    act = pocket.step(_snap(room=0x1E, x=48, y=93))
    assert act.reason == "north_to_east"
    mid = Level2ToSouthCenterController()
    act = mid.step(_snap(room=0x1E, x=72, y=93))
    assert act.reason == "north_to_east"
    edge = Level2ToSouthCenterController()
    act = edge.step(_snap(room=0x1E, x=168, y=93))
    assert act.reason == "north_to_east"
    east = Level2ToSouthCenterController()
    act = east.step(_snap(room=0x1E, x=176, y=93))
    assert act.reason == "east_south"
    act = east.step(_snap(room=0x1E, x=176, y=189))
    assert act.reason == "south_align_x"
    act = east.step(_snap(room=0x1E, x=120, y=189))
    assert east.success is True


def test_clear2e_succeeds_when_up_door_opens() -> None:
    """v4 timed out with 1 rope left; north door was already open."""
    from zelda_i.dungeon import GenericDungeonRoomController
    from zelda_i.level2_tf_spine import ROOM_2E_SPINE_SPEC

    ctl = Level2ClearDoorController(
        inner=GenericDungeonRoomController(ROOM_2E_SPINE_SPEC),
        door_bit=0x08,
        room_id=0x2E,
    )
    act = ctl.step(_snap(room=0x2E, x=58, y=101, doors=0x08))
    assert ctl.success is True
    assert act.reason == "done"


def test_inventory_poke_constants_are_owned_counts() -> None:
    assert SPINE_TF_BOMB_POKE == 16
    assert SPINE_TF_KEY_POKE == 2
