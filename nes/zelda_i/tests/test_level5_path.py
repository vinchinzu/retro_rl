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
    make_return_66_controller,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
) -> np.ndarray:
    return make_ram(
        {}, level=level, screen=room, x=x, y=y, mode=mode, keys=keys
    )


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


def test_block_stairs_06_warp_is_walked_not_spawn_tile() -> None:
    """0x06 warps from the walked (128,141) tile, not the (96,133) spawn."""
    from zelda_i.level5.cellar_path import cellar_to_64, take_block_stairs_06

    doc = take_block_stairs_06.__doc__ or ""
    assert "96,133" in doc
    assert "do not warp" in doc.lower()
    assert "189" in (cellar_to_64.__doc__ or "") or "pit" in (cellar_to_64.__doc__ or "").lower()


def test_nav_rows_keep_settle_and_frame_budgets() -> None:
    """0x66 return / 0x77 east key are rows of one settle-on-arrival nav."""
    from zelda_i.level5.path import (
        EAST_KEY_77_NAV,
        RETURN_66_NAV,
        level5_east_key_step,
        level5_return_66_step,
    )

    assert (RETURN_66_NAV.max_frames, RETURN_66_NAV.settle_frames) == (8000, 30)
    assert RETURN_66_NAV.step is level5_return_66_step
    assert RETURN_66_NAV.spec_id == "level5_return66_from_east_key"
    assert (EAST_KEY_77_NAV.max_frames, EAST_KEY_77_NAV.settle_frames) == (8000, 40)
    assert EAST_KEY_77_NAV.step is level5_east_key_step
    assert EAST_KEY_77_NAV.spec_id == "level5_east_key_nav_0x77"


def test_return_66_controller_settles_then_stops() -> None:
    from zelda_i.level5.path import RETURN_66_NAV, make_return_66_controller

    ctl = make_return_66_controller()
    snap = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=141))
    for _ in range(RETURN_66_NAV.settle_frames - 1):
        assert ctl.step(snap).reason == "settle_66"
        assert not ctl.success
    assert ctl.step(snap).reason == "arrived_66"
    assert ctl.success
    assert ctl.report()["spec_id"] == "level5_return66_from_east_key"


def test_return_66_controller_waits_in_the_south_mouth() -> None:
    """y>185 is the 0x66 south mouth; the hop is not done until Link is in."""
    ctl = make_return_66_controller()
    mouth = ctl.step(read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=205)))
    assert mouth.reason == "return66_leave_south"
    assert not ctl.success


def test_bomb_wall_rows_keep_stands_and_approaches() -> None:
    from zelda_i.level5.whistle_path import BOMB_EAST_65, BOMB_WEST_65, BOMB_WEST_66

    assert (BOMB_WEST_66.stand, BOMB_WEST_66.face) == ((32, 141), "LEFT")
    assert BOMB_WEST_66.dest_room == 0x65
    assert BOMB_WEST_66.leave_south_mouth
    assert BOMB_WEST_66.probe_paths[0] == (("y", 189), ("x", 32), ("y", 141))
    assert len(BOMB_WEST_66.probe_paths) == 3
    assert (BOMB_WEST_65.stand, BOMB_WEST_65.dest_room) == ((32, 141), 0x64)
    assert BOMB_WEST_65.approach == (
        ("y", 109, 400), ("x", 32, 400), ("y", 141, 400), ("x", 32, 200),
    )
    assert (BOMB_EAST_65.stand, BOMB_EAST_65.face) == ((224, 141), "RIGHT")
    assert BOMB_EAST_65.dest_room == 0x66
    assert BOMB_EAST_65.approach == (
        ("y", 109, 400), ("x", 208, 500), ("y", 141, 400), ("x", 224, 200),
    )


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


def test_whistle_path_has_no_idle_n_on_clean_path() -> None:
    """Clean L5 whistle hops wait on RAM; idle(n)/push_dir(frames=N) are gone."""
    import inspect

    from zelda_i.level5 import whistle_path

    src = inspect.getsource(whistle_path)
    assert "idle(env" not in src
    assert "push_dir(" not in src
    assert "wait_ram" in src
    assert "door_band_goal" in src


def test_ram_wait_hop_arrives_on_dest_room_not_frame_count() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level5.path import RamWaitHop

    hop = RamWaitHop(
        pred=lambda snap: snap.screen == 0x65 and snap.mode == PLAY_MODE,
        hold="LEFT",
        max_frames=20,
        spec_id="l5_west_65",
    )
    origin = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=32, y=141))
    act = hop.step(origin)
    assert list(act.action) == list(nes_action("LEFT"))
    assert hop.success is False
    dest = read_snapshot(_ram(room=0x65, x=224, y=141))
    done = hop.step(dest)
    assert hop.success
    assert list(done.action) == list(nes_idle_action())
    assert done.reason == "done"


def test_l5_west_door_band_is_leftover_relative() -> None:
    from zelda_i.dungeon.door_hop import door_band_goal

    assert door_band_goal("LEFT", (48, 141), (32, 141)) == (32, 141)
    assert door_band_goal("LEFT", (120, 189), (32, 141)) == (32, 141)
    assert door_band_goal("UP", (48, 189), (48, 93))[0] == 48
    assert door_band_goal("UP", (208, 189), (48, 93))[0] == 48


def test_whistle_04_exit_geometry() -> None:
    from zelda_i.level5.whistle_path import (
        L5_CELLAR_FLOOR_Y,
        L5_CELLAR_LEFT_X,
        L5_CELLAR_RIGHT_X,
        WHISTLE_04_LADDER_X,
        WHISTLE_04_MOUTH_X,
        WHISTLE_04_PIT_Y,
    )

    assert (WHISTLE_04_LADDER_X, WHISTLE_04_PIT_Y, WHISTLE_04_MOUTH_X) == (176, 189, 48)
    assert (L5_CELLAR_FLOOR_Y, L5_CELLAR_LEFT_X, L5_CELLAR_RIGHT_X) == (189, 48, 192)


def test_bomb_wall_blast_stand_waits_for_hole_open() -> None:
    """Fuse wait must stand (hold=None / idle) until door opens or room changes."""
    from retro_harness.nes import nes_idle_action
    from zelda_i.level5.path import RamWaitHop
    from zelda_i.ram import ADDR_CUR_OPENED_DOORS

    # West door bit is 0x02
    door_bit = 0x02
    hop = RamWaitHop(
        pred=lambda snap: snap.screen == 0x65 or bool(snap.cur_opened_doors & door_bit),
        hold=None,
        max_frames=140,
        spec_id="blast_test",
    )
    # Origin room 0x66, door closed -> must stand (idle), not push into bomb
    closed_ram = _ram(room=ROOM_L5_GIBDO_66, x=32, y=141)
    closed_ram[ADDR_CUR_OPENED_DOORS] = 0
    snap_closed = read_snapshot(closed_ram)
    act = hop.step(snap_closed)
    assert list(act.action) == list(nes_idle_action())
    assert hop.success is False

    # Blast finishes -> door bit 0x02 opens
    open_ram = _ram(room=ROOM_L5_GIBDO_66, x=32, y=141)
    open_ram[ADDR_CUR_OPENED_DOORS] = door_bit
    snap_open = read_snapshot(open_ram)
    act2 = hop.step(snap_open)
    assert hop.success is True
    assert list(act2.action) == list(nes_idle_action())

