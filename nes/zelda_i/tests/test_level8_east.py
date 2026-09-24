"""Durable L8 east gate from play 0x3E leftover (120,93). No emulator.

Idle until RIGHT door bit, stay north of statues, RIGHT toward the east
door, y-align, RIGHT push. OccupancyWalker banned. Dest is RAM; fail
cellar 0x0F and Gleeok 0x3C. No RAM writes.
make_gleeok_passage_controller stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.path import (
    CELLAR_ROOM,
    EAST_3E_DEST,
    EAST_3E_DEST_HYP,
    EAST_3E_DEST_POSE,
    EAST_3E_ORIGIN,
    EAST_3E_ORIGIN_POSE,
    EAST_DOOR,
    EAST_STATUE_CLEAR_X,
    GLEEOK_HYP,
    east_3e_step,
    make_east_3e_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
DOWN = list(nes_action("DOWN"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())

# Arrival doors 0x0C (UP+DOWN); idle later raises RIGHT (0x0D).
DOORS_ARRIVAL = 0x0C
DOORS_RIGHT_OPEN = 0x0D


_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": EAST_3E_ORIGIN,
    "x": EAST_3E_ORIGIN_POSE[0],
    "y": EAST_3E_ORIGIN_POSE[1],
    "tile": 0,
    "keys": 8,
    "bombs": 6,
    "magic_key": 1,
    "triforce": 0x7F,
    "doors": DOORS_RIGHT_OPEN,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "east 3E controllers must not write RAM"
    return act


def test_knockback_y_aligns_to_leftover_door_band_not_south() -> None:
    """Past statue-clear x, leftover y=157 y-aligns UP to the door row, never DOWN."""
    ctl = make_east_3e_controller(dest=None)
    act = _step(ctl, _ram(x=208, y=157, doors=DOORS_RIGHT_OPEN))
    assert not ctl.failed
    assert list(act.action) == UP
    assert list(act.action) != DOWN
    assert act.reason == "east_align"


def test_leftover_emits_right_not_down_or_up() -> None:
    """(120,93) with RIGHT bit goes RIGHT. Not DOWN south or UP north."""
    snap = read_snapshot(_ram(x=120, y=93, doors=DOORS_RIGHT_OPEN))
    act = east_3e_step(snap)
    assert list(act.action) == RIGHT
    assert list(act.action) != DOWN
    assert list(act.action) != UP
    assert act.reason == "east_approach"

    ctl = make_east_3e_controller(dest=None)
    act = _step(ctl, _ram(x=120, y=93, doors=DOORS_RIGHT_OPEN))
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert list(act.action) != DOWN
    assert list(act.action) != UP
    assert act.reason == "east_approach"


def test_arrival_without_right_bit_idles() -> None:
    snap = read_snapshot(_ram(x=120, y=93, doors=DOORS_ARRIVAL))
    act = east_3e_step(snap)
    assert list(act.action) == IDLE
    assert list(act.action) != DOWN
    assert list(act.action) != UP
    assert act.reason == "east_wait_right_bit"


def test_statue_row_emits_up_not_right_into_x144() -> None:
    """Do not walk y=141 RIGHT into the x=144 statue."""
    ctl = make_east_3e_controller(dest=None)
    act = _step(ctl, _ram(x=120, y=141, doors=DOORS_RIGHT_OPEN))
    assert not ctl.failed
    assert list(act.action) == UP
    assert list(act.action) != RIGHT
    assert act.reason == "east_north_band"


def test_past_statue_y_aligns_then_east_door_pushes_right() -> None:
    align = make_east_3e_controller(dest=None)
    act = _step(
        align, _ram(x=EAST_STATUE_CLEAR_X, y=93, doors=DOORS_RIGHT_OPEN)
    )
    assert not align.failed
    assert list(act.action) == DOWN
    assert act.reason == "east_align"

    ctl = make_east_3e_controller(dest=None)
    act = _step(
        ctl,
        _ram(x=EAST_DOOR[0], y=EAST_DOOR[1], doors=DOORS_RIGHT_OPEN),
    )
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert act.reason == "east_push"


def test_dest_none_first_play_succeeds_gleeok_and_cellar_fail() -> None:
    ok = make_east_3e_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=EAST_3E_DEST_HYP, x=16, y=141))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    gleeok = make_east_3e_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE

    cellar_mode = make_east_3e_controller(dest=None)
    act = _step(
        cellar_mode, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_mode.success and cellar_mode.failed
    assert list(act.action) == IDLE

    cellar_screen = make_east_3e_controller(dest=None)
    act = _step(
        cellar_screen, _ram(mode=PLAY_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_screen.success and cellar_screen.failed
    assert list(act.action) == IDLE


def test_dest_live_accepts_3f_rejects_3c() -> None:
    ok = make_east_3e_controller(dest=EAST_3E_DEST)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x3F, x=32, y=141))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    none = make_east_3e_controller(dest=None)
    act = _step(none, _ram(mode=PLAY_MODE, screen=0x3F, x=32, y=141))
    assert none.success and not none.failed

    bad = make_east_3e_controller(dest=EAST_3E_DEST)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE

    cellar = make_east_3e_controller(dest=EAST_3E_DEST)
    act = _step(cellar, _ram(mode=PLAY_MODE, screen=CELLAR_ROOM, x=136, y=141))
    assert not cellar.success and cellar.failed
    assert list(act.action) == IDLE
    assert EAST_3E_DEST_POSE == (32, 141)
