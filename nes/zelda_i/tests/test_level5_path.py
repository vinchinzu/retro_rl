"""Unit tests for Level 5 leftover geometry that burned."""

from __future__ import annotations

import numpy as np

from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
)
from zelda_i.level5.path import (
    level5_east_key_step,
    level5_room66_west_aisle_north_step,
    level5_west65_step,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    return ram


def test_room66_west_aisle_prefights_north_of_river() -> None:
    """TF suffix leftover (32,141) UP; (79,165)/(152,189) still south."""
    hole = level5_room66_west_aisle_north_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=32, y=141))
    )
    assert hole.reason == "66_west_aisle_up"
    south = level5_room66_west_aisle_north_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=79, y=165))
    )
    assert south.reason == "66_west_aisle_up"
    se = level5_room66_west_aisle_north_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=152, y=189))
    )
    assert se.reason == "66_west_aisle_up"
    bank = level5_room66_west_aisle_north_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=32, y=101))
    )
    assert bank.reason == "66_west_aisle_x"
    parked = level5_room66_west_aisle_north_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=48, y=101))
    )
    assert parked.reason == "66_north_bank"


def test_east_key_route_returns_south_from_cleared_66() -> None:
    snap = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=117, keys=1))
    action = level5_east_key_step(snap)
    assert action.reason == "east_key_finish_ladder"
    north_bank = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=32, y=101, keys=6))
    )
    assert north_bank.reason == "east_key_to_ladder_x"
    off_ladder = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=149, keys=1))
    )
    assert off_ladder.reason == "east_key_align_south_x"


def test_east_key_route_uses_wall_before_door_channel() -> None:
    approach = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=180, y=157, keys=1))
    )
    still_approach = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=200, y=157, keys=1))
    )
    channel = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=208, y=157, keys=1))
    )
    assert approach.reason == "east_key_approach_wall"
    assert still_approach.reason == "east_key_approach_wall"
    assert channel.reason == "east_key_align_channel_y"


def test_west65_uses_statue_bypass_on_76() -> None:
    doorway = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=224, y=141, keys=2))
    )
    assert doorway.reason == "west65_leave_east_mouth"
    east_pocket = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=200, y=141, keys=2))
    )
    assert east_pocket.reason == "west65_align_approach_y"
    leave = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=200, y=157, keys=2))
    )
    assert leave.reason == "west65_leave_east_door"
    north = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=120, y=157, keys=2))
    )
    assert north.reason == "west65_enter_66"


def test_take_center_stairs_06_is_center_tile_not_spawn() -> None:
    from zelda_i.level5.path import take_center_stairs_06, cellar_to_64
    assert "120,141" in (take_center_stairs_06.__doc__ or "")
    assert "return take_block_stairs_06" not in (take_center_stairs_06.__doc__ or "")
    assert "189" in (cellar_to_64.__doc__ or "") or "pit" in (cellar_to_64.__doc__ or "").lower()


def test_whistle_tf_stand_geometry() -> None:
    """Whistle stand is (120, 141); 0x04 exit is 135,141, 0x06 stairs 128/120,141."""
    from zelda_i.level5.boss_path import (
        WHISTLE_STAND,
        fight_digdogger,
        path_exit_whistle_04,
        take_stairs_06,
    )

    assert WHISTLE_STAND == (120, 141)
    assert "0x04" in (path_exit_whistle_04.__doc__ or "")
    assert "135,141" in (path_exit_whistle_04.__doc__ or "")
    assert "0x38" in (fight_digdogger.__doc__ or "")
    assert "128,141" in (take_stairs_06.__doc__ or "")
    assert "120,141" in (take_stairs_06.__doc__ or "")
