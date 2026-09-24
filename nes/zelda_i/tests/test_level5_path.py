"""Unit tests for Level 5 leftover geometry that burned."""

from __future__ import annotations

import numpy as np

from retro_harness.controls import pressed_nes_buttons

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


def _press(step, **ram) -> list[str]:
    return pressed_nes_buttons(list(step(read_snapshot(_ram(**ram))).action))


def test_room66_west_aisle_prefights_north_of_river() -> None:
    """TF suffix leftover (32,141) UP; (79,165)/(152,189) still south."""
    step = level5_room66_west_aisle_north_step
    assert _press(step, room=ROOM_L5_GIBDO_66, x=32, y=141) == ["UP"]
    assert _press(step, room=ROOM_L5_GIBDO_66, x=79, y=165) == ["UP"]
    assert _press(step, room=ROOM_L5_GIBDO_66, x=152, y=189) == ["UP"]
    assert _press(step, room=ROOM_L5_GIBDO_66, x=32, y=101) == ["RIGHT"]
    # Parked on the north bank: stand still and let the fight come.
    assert _press(step, room=ROOM_L5_GIBDO_66, x=48, y=101) == []


def test_east_key_route_returns_south_from_cleared_66() -> None:
    step = level5_east_key_step
    assert _press(step, room=ROOM_L5_GIBDO_66, x=56, y=117, keys=1) == ["DOWN"]
    assert _press(step, room=ROOM_L5_GIBDO_66, x=32, y=101, keys=6) == ["RIGHT"]
    assert _press(step, room=ROOM_L5_GIBDO_66, x=56, y=149, keys=1) == ["RIGHT"]


def test_east_key_route_uses_wall_before_door_channel() -> None:
    step = level5_east_key_step
    assert _press(step, room=ROOM_L5_ENTRY, x=180, y=157, keys=1) == ["RIGHT"]
    assert _press(step, room=ROOM_L5_ENTRY, x=200, y=157, keys=1) == ["RIGHT"]
    assert _press(step, room=ROOM_L5_ENTRY, x=208, y=157, keys=1) == ["UP"]


def test_west65_uses_statue_bypass_on_76() -> None:
    step = level5_west65_step
    assert _press(step, room=ROOM_L5_ENTRY, x=224, y=141, keys=2) == ["LEFT"]
    assert _press(step, room=ROOM_L5_ENTRY, x=200, y=141, keys=2) == ["DOWN"]
    assert _press(step, room=ROOM_L5_ENTRY, x=200, y=157, keys=2) == ["LEFT"]
    assert _press(step, room=ROOM_L5_ENTRY, x=120, y=157, keys=2) == ["UP"]


def test_return_66_controller_settles_then_stops() -> None:
    from zelda_i.level5.path import RETURN_66_NAV, make_return_66_controller

    ctl = make_return_66_controller()
    snap = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=141))
    for _ in range(RETURN_66_NAV.settle_frames - 1):
        assert ctl.step(snap).reason == "settle_66"
        assert not ctl.success
    assert ctl.step(snap).reason == "arrived_66"
    assert ctl.success


def test_return_66_controller_waits_in_the_south_mouth() -> None:
    """y>185 is the 0x66 south mouth; the hop is not done until Link is in."""
    ctl = make_return_66_controller()
    mouth = ctl.step(read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=205)))
    assert mouth.reason == "return66_leave_south"
    assert not ctl.success


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
