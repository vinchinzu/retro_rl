"""Unit tests for Level 5 east-key path policy (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
)
from zelda_i.level5_path import (
    EAST_DOOR_APPROACH_Y,
    EAST_DOOR_CHANNEL_Y,
    EAST_DOOR_WALL_X,
    ROOM_06_BLOCK_PUSHED_Y,
    ROOM_06_BLOCK_REST_Y,
    ROOM_06_BLOCK_X,
    ROOM_06_STAIRS_X,
    ROOM_06_STAIRS_Y,
    WHISTLE_04_LADDER_X,
    WHISTLE_04_MOUTH_X,
    WHISTLE_04_PIT_Y,
    level5_east_key_step,
    level5_west65_step,
    should_force_keys_zero,
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


def test_east_door_geometry() -> None:
    assert EAST_DOOR_APPROACH_Y == 157
    assert EAST_DOOR_CHANNEL_Y == 141
    assert EAST_DOOR_WALL_X == 208


def test_east_key_route_returns_south_from_cleared_66() -> None:
    snap = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=117, keys=1))
    action = level5_east_key_step(snap)
    assert action.reason == "east_key_finish_ladder"
    off_ladder = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=149, keys=1))
    )
    assert off_ladder.reason == "east_key_align_south_x"


def test_east_key_route_uses_wall_before_door_channel() -> None:
    approach = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=180, y=157, keys=1))
    )
    # x=200 is still short of the wall; y-slide here hits the statue.
    still_approach = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=200, y=157, keys=1))
    )
    channel = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=208, y=157, keys=1))
    )
    assert approach.reason == "east_key_approach_wall"
    assert still_approach.reason == "east_key_approach_wall"
    assert channel.reason == "east_key_align_channel_y"


def test_should_force_keys_zero() -> None:
    assert should_force_keys_zero("L5_Room_77") is True
    assert should_force_keys_zero("Level5Cleared66") is False
    assert should_force_keys_zero("Level5EntranceFromL4") is False
    assert should_force_keys_zero("L5_Room_77", keep_keys=True) is False
    assert should_force_keys_zero("Level5Cleared66", force_keys_zero=True) is True

def test_west65_returns_from_77() -> None:
    start = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=136, y=165, keys=2))
    )
    assert start.reason == "west65_align_77_south_y"
    mid_block = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=136, y=141, keys=2))
    )
    assert mid_block.reason == "west65_align_77_south_y"
    south = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=136, y=173, keys=2))
    )
    assert south.reason == "west65_pass_77_blocks"
    door_y = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=40, y=173, keys=2))
    )
    assert door_y.reason == "west65_align_77_y"
    channel = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=40, y=141, keys=2))
    )
    assert channel.reason == "west65_return_76"


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


def test_return_66_stops_in_cleared_66() -> None:
    """East-key return must not take the free UP into Dodongos 0x56."""
    from zelda_i.level5_path import level5_return_66_step

    south = level5_return_66_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=205, keys=2))
    )
    assert south.reason == "return66_leave_south"
    ready = level5_return_66_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=141, keys=2))
    )
    assert ready.reason == "return66_arrived"
    from_77 = level5_return_66_step(
        read_snapshot(_ram(room=ROOM_L5_POLS_77, x=136, y=165, keys=2))
    )
    assert from_77.reason == "west65_align_77_south_y"


def test_bomb_west_66_stand() -> None:
    """0x66 west bomb bricks sit at the west-door column, south of the river lock."""
    from zelda_i.level5_path import BOMB_WEST_66_STAND, bomb_west_from_66

    assert BOMB_WEST_66_STAND == (32, 141)
    doc = bomb_west_from_66.__doc__ or ""
    assert "0x65" in doc
    assert "189" in doc
    assert "poke" in doc.lower()


def test_west65_goes_north_from_66() -> None:
    south = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=205, keys=2))
    )
    assert south.reason == "west65_leave_66_south"
    north = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=141, keys=2))
    )
    assert north.reason == "west65_enter_56"
    ladder = level5_west65_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=117, keys=2))
    )
    assert ladder.reason == "west65_finish_ladder"


def test_whistle_04_exit_geometry() -> None:
    """Alcove drops only at x=176; left mouth is the 0x05 return."""
    assert WHISTLE_04_LADDER_X == 176
    assert WHISTLE_04_PIT_Y == 189
    assert WHISTLE_04_MOUTH_X == 48


def test_whistle_06_07_return_geometry() -> None:
    """0x06 return is block-stairs, not the south key drop. 0x07 left is 0x64."""
    from zelda_i.level5_path import (
        L5_CELLAR_FLOOR_Y,
        L5_CELLAR_LEFT_X,
        L5_CELLAR_RIGHT_X,
        ROOM_L5_BLUE_64,
        ROOM_L5_CELLAR_07,
        ROOM_L5_PASSAGE_06,
        cellar_to_64,
        take_block_stairs_06,
        take_center_stairs_06,
        walk_east_from_05,
    )

    assert ROOM_L5_PASSAGE_06 == 0x06
    assert ROOM_L5_CELLAR_07 == 0x07
    assert ROOM_L5_BLUE_64 == 0x64
    assert L5_CELLAR_FLOOR_Y == 189
    assert L5_CELLAR_LEFT_X == 48
    assert L5_CELLAR_RIGHT_X == 192
    assert take_center_stairs_06 is not take_block_stairs_06
    assert "96,133" in (take_block_stairs_06.__doc__ or "")
    assert "0x16" in (take_block_stairs_06.__doc__ or "")
    assert "left" in (cellar_to_64.__doc__ or "").lower()
    assert walk_east_from_05.__doc__


def test_room06_stairs_geometry() -> None:
    """Return stairs sit by the pushed 0x68, not diamond-center tiles."""
    assert ROOM_06_BLOCK_X == 96
    assert ROOM_06_BLOCK_REST_Y == 144
    assert ROOM_06_BLOCK_PUSHED_Y == 128
    assert ROOM_06_STAIRS_X == 96
    assert ROOM_06_STAIRS_Y == 133

def test_walk_east_from_65_geometry() -> None:
    """0x65 east tries the existing hole first; north shutter is not a path."""
    from zelda_i.level5_path import (
        EAST65_TO_66_PATHS,
        bomb_east_from_65,
        walk_east_from_65,
    )

    assert EAST65_TO_66_PATHS[0][0] == "north109_east"
    assert ("y", 109) in EAST65_TO_66_PATHS[0][1]
    assert ("x", 224) in EAST65_TO_66_PATHS[0][1]
    doc = walk_east_from_65.__doc__ or ""
    assert "bomb" in doc.lower()
    assert "UP" in doc or "one-way" in doc.lower() or "shutter" in doc.lower()
    assert walk_east_from_65 is not bomb_east_from_65



def test_take_center_stairs_06_is_center_tile_not_spawn() -> None:
    from zelda_i.level5_path import take_center_stairs_06, cellar_to_64
    assert "120,141" in (take_center_stairs_06.__doc__ or "")
    assert "return take_block_stairs_06" not in (take_center_stairs_06.__doc__ or "")
    assert "189" in (cellar_to_64.__doc__ or "") or "pit" in (cellar_to_64.__doc__ or "").lower()


def test_walk_north_from_57_geometry() -> None:
    """0x57 north is open; do not clear Zols (0x5f seals y≈125)."""
    from zelda_i.level5_path import (
        DIAMOND_NORTH_Y,
        NORTH_DOOR_X,
        NORTH_DOOR_Y,
        ROOM_L5_EAST_ZOLS,
        ROOM_L5_NORTH_GIBDOS,
        STATUE_5F_TYPE,
        STATUE_5F_X,
        STATUE_5F_Y,
        walk_north_from_57,
    )

    assert ROOM_L5_EAST_ZOLS == 0x57
    assert ROOM_L5_NORTH_GIBDOS == 0x47
    assert NORTH_DOOR_X == 120
    assert NORTH_DOOR_Y == 93
    assert DIAMOND_NORTH_Y == 109
    assert STATUE_5F_TYPE == 0x5F
    assert (STATUE_5F_X, STATUE_5F_Y) == (128, 128)
    doc = walk_north_from_57.__doc__ or ""
    assert "not" in doc.lower() and "zol" in doc.lower()
    assert "key-north" in doc or "key_north" in doc
    assert "0x5f" in doc
    assert "diamond" in doc.lower()


def test_level5_path_reexports_split_modules() -> None:
    """Public names stay on the facade after the rr-fcyg split."""
    import zelda_i.level5_cellar_path as cellar
    import zelda_i.level5_path as facade
    import zelda_i.level5_tf_path as tf
    import zelda_i.level5_west_path as west
    import zelda_i.level5_whistle_path as whistle

    assert facade.level5_east_key_step.__module__ == "zelda_i.level5_path"
    assert facade.make_west65_controller.__module__ == "zelda_i.level5_path"
    assert facade.level5_return_66_step.__module__ == "zelda_i.level5_path"
    assert facade.walk_west_from_27 is west.walk_west_from_27
    assert facade.bomb_west_from_65 is whistle.bomb_west_from_65
    assert facade.bomb_west_from_66 is whistle.bomb_west_from_66
    assert facade.take_block_stairs_06 is cellar.take_block_stairs_06
    assert facade.walk_north_from_57 is tf.walk_north_from_57
    assert facade.BLUE_DARKNUT_TYPE is whistle.BLUE_DARKNUT_TYPE
