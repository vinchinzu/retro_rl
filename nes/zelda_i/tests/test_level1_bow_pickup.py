"""Unit tests for L1 mode-9 bow pickup and return to play 0x23."""

from __future__ import annotations

import numpy as np

from zelda_i.level1_bow_cellar import LEVEL1_BOW_CELLAR_ROOM
from zelda_i.level1_bow_pickup import (
    BOW_PEDESTAL,
    EAST_X,
    EXIT_STAIRS,
    FLOOR_Y,
    SETTLE_FRAMES,
    WEST_X,
    PickupPhase,
    level1_bow_pickup_success,
    make_bow_pickup_controller,
)
from zelda_i.level1_bow_rejoin import (
    REJOIN_DEST,
    REJOIN_EAST_X,
    REJOIN_NORTH_Y,
    REJOIN_PLUS_X,
    REJOIN_WEST_COL,
    level1_bow_rejoin_success,
    make_bow_rejoin_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_IS_UPDATING_MODE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PASSAGE_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 1)
    ram[ADDR_SCREEN] = fields.get("screen", LEVEL1_BOW_CELLAR_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", 128)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    ram[ADDR_KEYS] = fields.get("keys", 0)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_IS_UPDATING_MODE] = fields.get("updating", 1)
    return ram


def test_bow_pickup_does_not_right_into_pit_at_y141() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    ctl = make_bow_pickup_controller()
    warp = _ram(x=128, y=141, updating=1)
    act = ctl.step(read_snapshot(warp))
    assert act.reason == "wait_spawn"
    assert list(act.action) == list(nes_idle_action())
    spit = _ram(x=WEST_X, y=93, updating=1)
    for _ in range(SETTLE_FRAMES + 1):
        act = ctl.step(read_snapshot(spit))
    assert act.reason == "west_floor"
    assert list(act.action) == list(nes_action("DOWN"))
    pit = _ram(x=WEST_X, y=141, updating=1)
    hunt = make_bow_pickup_controller()
    hunt.phase = PickupPhase.HUNT
    act = hunt.step(read_snapshot(pit))
    assert act.reason == "west_floor"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    floor = _ram(x=WEST_X, y=FLOOR_Y, updating=1)
    act = hunt.step(read_snapshot(floor))
    assert act.reason == "floor_east"
    assert list(act.action) == list(nes_action("RIGHT"))
    east = _ram(x=EAST_X, y=FLOOR_Y, updating=1)
    act = hunt.step(read_snapshot(east))
    assert act.reason == "east_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    overshoot = _ram(x=208, y=FLOOR_Y, updating=1)
    act = hunt.step(read_snapshot(overshoot))
    assert act.reason == "east_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    at = _ram(x=BOW_PEDESTAL[0], y=BOW_PEDESTAL[1], updating=1)
    act = hunt.step(read_snapshot(at))
    assert act.reason == "bow_stand_idle"


def test_bow_pickup_exits_stairs_then_east_after_bow() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    ctl = make_bow_pickup_controller()
    got = _ram(x=136, y=141, bow=1, updating=1)
    act = ctl.step(read_snapshot(got))
    assert ctl.phase is PickupPhase.EXIT_STAIRS
    assert act.reason == "exit_to_east"
    assert list(act.action) == list(nes_action("RIGHT"))
    drop = _ram(x=EAST_X, y=141, bow=1, updating=1)
    act = ctl.step(read_snapshot(drop))
    assert act.reason == "exit_drop"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    stairs = _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], bow=1, updating=1)
    act = ctl.step(read_snapshot(stairs))
    assert act.reason == "exit_up"
    assert list(act.action) == list(nes_action("UP"))
    back = _ram(mode=PLAY_MODE, screen=0x22, x=96, y=173, bow=1)
    act = ctl.step(read_snapshot(back))
    assert ctl.phase is PickupPhase.EAST_22
    assert act.reason == "east_peel"
    assert list(act.action) == list(nes_action("DOWN"))
    mouth = _ram(mode=PLAY_MODE, screen=0x22, x=96, y=FLOOR_Y, bow=1)
    act = ctl.step(read_snapshot(mouth))
    assert act.reason == "east_22"
    assert list(act.action) == list(nes_action("RIGHT"))
    brick = _ram(mode=PLAY_MODE, screen=0x22, x=208, y=FLOOR_Y, bow=1)
    act = ctl.step(read_snapshot(brick))
    assert act.reason == "east_column"
    assert list(act.action) == list(nes_action("UP"))


def test_bow_pickup_requires_play_23_with_bow() -> None:
    dest = _ram(mode=PLAY_MODE, screen=0x23, x=32, y=141, bow=1)
    assert level1_bow_pickup_success(read_snapshot(dest))
    cellar = _ram(bow=1)
    assert not level1_bow_pickup_success(read_snapshot(cellar))
    empty = _ram(mode=PLAY_MODE, screen=0x23, x=32, y=141, bow=0)
    assert not level1_bow_pickup_success(read_snapshot(empty))
    ctl = make_bow_pickup_controller()
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert act.reason == "arrived_23_bow"


def test_bow_rejoin_does_not_up_out_the_west_door() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    ctl = make_bow_rejoin_controller()
    mouth = _ram(mode=PLAY_MODE, screen=0x23, x=16, y=141, bow=1)
    act = ctl.step(read_snapshot(mouth))
    assert act.reason == "rejoin_inland"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    col = _ram(mode=PLAY_MODE, screen=0x23, x=REJOIN_WEST_COL, y=141, bow=1)
    act = ctl.step(read_snapshot(col))
    assert act.reason == "rejoin_up"
    assert list(act.action) == list(nes_action("UP"))
    north = _ram(
        mode=PLAY_MODE, screen=0x23, x=REJOIN_WEST_COL, y=REJOIN_NORTH_Y, bow=1
    )
    act = ctl.step(read_snapshot(north))
    assert act.reason == "rejoin_north"
    assert list(act.action) == list(nes_action("RIGHT"))
    overshoot = _ram(
        mode=PLAY_MODE, screen=0x23, x=REJOIN_EAST_X, y=REJOIN_NORTH_Y, bow=1
    )
    act = ctl.step(read_snapshot(overshoot))
    assert act.reason == "rejoin_north"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    plus = _ram(
        mode=PLAY_MODE, screen=0x23, x=REJOIN_PLUS_X, y=REJOIN_NORTH_Y, bow=1
    )
    act = ctl.step(read_snapshot(plus))
    assert act.reason == "rejoin_drop"
    assert list(act.action) == list(nes_action("DOWN"))
    channel = _ram(
        mode=PLAY_MODE, screen=0x23, x=REJOIN_PLUS_X, y=REJOIN_DEST[1], bow=1
    )
    act = ctl.step(read_snapshot(channel))
    assert act.reason == "rejoin_east"
    assert list(act.action) == list(nes_action("RIGHT"))
    at = _ram(
        mode=PLAY_MODE,
        screen=0x23,
        x=REJOIN_DEST[0],
        y=REJOIN_DEST[1],
        bow=1,
    )
    act = ctl.step(read_snapshot(at))
    assert ctl.success
    assert act.reason == "rejoin_at"
    assert list(act.action) == list(nes_idle_action())
    assert level1_bow_rejoin_success(read_snapshot(at))
    empty = _ram(mode=PLAY_MODE, screen=0x23, x=16, y=141, bow=0)
    assert not level1_bow_rejoin_success(read_snapshot(empty))
