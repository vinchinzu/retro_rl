"""Unit tests for the L2 Survival-spine 0x7d → Magical Boomerang table."""

from __future__ import annotations

import numpy as np

from zelda_i.bomb_wall_path import BombWallController
from zelda_i.dungeon import GenericDungeonRoomController
from zelda_i.level2_dungeon import (
    ROOM_4F_SPEC,
    ROOM_6C_SPEC,
    ROOM_6D_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    ROOM_7E_SPEC,
)
from zelda_i.level2_spine import (
    Level2BacktrackTo7dController,
    Level2Enter6fKeyController,
    Level2NavPhase,
    Level2WestEnter6eController,
    ROOM_7E_SPINE_SPEC,
    level2_through_success,
    level2_to_boom_stages,
)
from zelda_i.nav_common import DIAMOND_BAND_6E
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    room: int,
    x: int = 120,
    y: int = 141,
    keys: int = 2,
    bombs: int = 0,
    boom: int = 0,
    mode: int = PLAY_MODE,
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
    ram[ADDR_TRIFORCE] = 0x01
    ram[ADDR_HEALTH] = 0x2F
    return read_snapshot(ram)


def test_level2_to_boom_stage_names_and_types() -> None:
    stages = list(level2_to_boom_stages())
    names = [name for name, _, _ in stages]
    assert names == [
        "clear6d",
        "clear6c_key",
        "backtrack_7d",
        "clear7e_key",
        "enter_6e_west",
        "clear6e",
        "enter_6f_key",
        "clear6f_compass",
        "bomb_north_6f",
        "bomb_north_5f",
        "clear4f_boom",
    ]
    by_name = {name: ctl for name, ctl, _ in stages}
    assert isinstance(by_name["clear6d"], GenericDungeonRoomController)
    assert by_name["clear6d"].spec is ROOM_6D_SPEC
    assert isinstance(by_name["clear6c_key"], GenericDungeonRoomController)
    assert by_name["clear6c_key"].spec is ROOM_6C_SPEC
    assert isinstance(by_name["backtrack_7d"], Level2BacktrackTo7dController)
    assert isinstance(by_name["clear7e_key"], GenericDungeonRoomController)
    assert by_name["clear7e_key"].spec is ROOM_7E_SPINE_SPEC
    assert by_name["clear7e_key"].spec is not ROOM_7E_SPEC
    assert ROOM_7E_SPINE_SPEC.reward.waypoints
    assert ROOM_7E_SPINE_SPEC.reward.reward_while_live is True
    assert isinstance(by_name["enter_6e_west"], Level2WestEnter6eController)
    assert isinstance(by_name["clear6e"], GenericDungeonRoomController)
    assert by_name["clear6e"].spec is ROOM_6E_SPEC
    assert isinstance(by_name["enter_6f_key"], Level2Enter6fKeyController)
    assert isinstance(by_name["clear6f_compass"], GenericDungeonRoomController)
    assert by_name["clear6f_compass"].spec is ROOM_6F_SPEC
    assert isinstance(by_name["bomb_north_6f"], BombWallController)
    assert by_name["bomb_north_6f"].from_room == 0x6F
    assert by_name["bomb_north_6f"].to_room == 0x5F
    assert isinstance(by_name["bomb_north_5f"], BombWallController)
    assert by_name["bomb_north_5f"].from_room == 0x5F
    assert by_name["bomb_north_5f"].to_room == 0x4F
    assert isinstance(by_name["clear4f_boom"], GenericDungeonRoomController)
    assert by_name["clear4f_boom"].spec is ROOM_4F_SPEC


def test_backtrack_7d_walks_east_then_south() -> None:
    ctl = Level2BacktrackTo7dController()
    act = ctl.step(_snap(room=0x6C, x=80, y=141))
    assert act.reason == "push_door"
    act = ctl.step(_snap(room=0x6C, x=80, y=120))
    assert act.reason == "align_door_y"
    act = ctl.step(_snap(room=0x6D, x=80, y=141))
    assert act.reason == "align_door_x"
    act = ctl.step(_snap(room=0x6D, x=120, y=141))
    assert act.reason == "push_door"
    act = ctl.step(_snap(room=0x7D, x=120, y=93))
    assert ctl.success is True
    assert ctl.phase is Level2NavPhase.DONE
    assert act.reason == "done"


def test_backtrack_7d_recenters_live_timeout_pose() -> None:
    """survival_spine_l2_boom timed out in 0x6c at (128, 133) with ALIGN_TOL=6."""
    ctl = Level2BacktrackTo7dController()
    act = ctl.step(_snap(room=0x6C, x=128, y=133))
    assert act.reason == "align_door_y"
    act = ctl.step(_snap(room=0x6C, x=136, y=136))
    assert act.reason == "align_door_y"
    act = ctl.step(_snap(room=0x6C, x=136, y=141))
    assert act.reason == "push_door"


def test_west_enter_6e_goes_north_from_7e() -> None:
    ctl = Level2WestEnter6eController()
    act = ctl.step(_snap(room=0x7E, x=80, y=141))
    assert act.reason == "align_door_x"
    act = ctl.step(_snap(room=0x7E, x=120, y=141))
    assert act.reason == "push_door"
    act = ctl.step(_snap(room=0x6E, x=120, y=205))
    assert ctl.success is True
    assert ctl.phase is Level2NavPhase.DONE
    assert act.reason == "done"


def test_enter_6f_fails_without_keys() -> None:
    ctl = Level2Enter6fKeyController()
    act = ctl.step(_snap(room=0x6E, keys=0))
    assert ctl.phase is Level2NavPhase.FAILED
    assert act.reason == "no_keys"
    assert ctl.success is False


def test_enter_6f_uses_diamond_band_and_arrives() -> None:
    ctl = Level2Enter6fKeyController()
    assert ctl.band_y == DIAMOND_BAND_6E
    act = ctl.step(_snap(room=0x6E, x=120, y=141, keys=2))
    assert ctl.phase is Level2NavPhase.WALK
    assert act.reason != "no_keys"
    act = ctl.step(_snap(room=0x6F, x=32, y=141, keys=1))
    assert ctl.success is True
    assert ctl.phase is Level2NavPhase.DONE


def test_through_success_is_boom_inventory() -> None:
    assert not level2_through_success(_snap(room=0x7D, boom=0))
    assert level2_through_success(_snap(room=0x4F, boom=1))
