"""Durable L8 south gate from play 0x1E leftover (208,141). No emulator.

Cardinal x-align to 120, then DOWN push. Occupancy BFS first dir is DOWN
along x=208 into the SE statue; 1px-grade also false-misses 2px dungeon
steps (west G1). Dest is RAM; fail cellar 0x0F and Gleeok 0x3C. No RAM
writes. make_gleeok_passage_controller stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.path import (
    CELLAR_ROOM,
    GLEEOK_HYP,
    Level8South1EController,
    Level8South2EController,
    UnverifiedLevel8PathController,
    SOUTH_2E_DEST,
    SOUTH_2E_DEST_HYP,
    SOUTH_2E_DEST_POSE,
    SOUTH_2E_ORIGIN,
    SOUTH_2E_ORIGIN_POSE,
    SOUTH_DEST,
    SOUTH_DEST_POSE,
    SOUTH_DOOR,
    SOUTH_ORIGIN,
    SOUTH_ORIGIN_POSE,
    make_gleeok_passage_controller,
    make_magic_key_stairs_controller,
    make_south_1e_controller,
    make_south_2e_controller,
    south_1e_step,
    south_2e_step,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": SOUTH_ORIGIN,
    "x": SOUTH_ORIGIN_POSE[0],
    "y": SOUTH_ORIGIN_POSE[1],
    "tile": 0,
    "keys": 8,
    "bombs": 6,
    "magic_key": 1,
    "triforce": 0x7F,
}

LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
DOWN = list(nes_action("DOWN"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "south 1E controllers must not write RAM"
    return act


def test_off_column_knockback_aligns_left_not_down_or_up() -> None:
    """x=208 leftover must LEFT onto the aisle, never DOWN/UP into a statue."""
    ctl = make_south_1e_controller()
    act = _step(ctl, _ram(x=208, y=157))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert list(act.action) != DOWN
    assert list(act.action) != UP
    assert act.reason == "south_align"


def test_on_column_leftover_keeps_leftover_x() -> None:
    """x=118 is on the door column: DOWN, not RIGHT 2px onto a frozen 120."""
    ctl = make_south_1e_controller()
    act = _step(ctl, _ram(x=118, y=141))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert list(act.action) != RIGHT
    assert act.reason == "south_approach"


def test_leftover_emits_left_align_not_up_or_west_push() -> None:
    """(208,141) x-aligns LEFT toward 120. Not UP into the north door."""
    snap = read_snapshot(_ram(x=208, y=141))
    act = south_1e_step(snap)
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert list(act.action) != DOWN
    assert act.reason == "south_align"

    ctl = make_south_1e_controller()
    act = _step(ctl, _ram(x=208, y=141))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert act.reason == "south_align"


def test_center_aisle_emits_down_not_left_into_west_wall() -> None:
    """Once x-aligned, DOWN is the hop. LEFT is not the whole hop."""
    ctl = make_south_1e_controller()
    act = _step(ctl, _ram(x=SOUTH_DOOR[0], y=141))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert list(act.action) != LEFT
    assert act.reason == "south_approach"


def test_south_door_emits_down_push() -> None:
    ctl = make_south_1e_controller()
    act = _step(ctl, _ram(x=SOUTH_DOOR[0], y=SOUTH_DOOR[1]))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert act.reason == "south_push"


def test_right_of_center_aligns_left_left_of_center_aligns_right() -> None:
    left = make_south_1e_controller()
    act = _step(left, _ram(x=160, y=141))
    assert not left.failed
    assert list(act.action) == LEFT
    assert act.reason == "south_align"

    right = make_south_1e_controller()
    act = _step(right, _ram(x=80, y=141))
    assert not right.failed
    assert list(act.action) == RIGHT
    assert act.reason == "south_align"


def test_dest_none_first_play_succeeds_gleeok_and_cellar_fail() -> None:
    ok = make_south_1e_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x2E, x=120, y=93))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    gleeok = make_south_1e_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE

    cellar_mode = make_south_1e_controller(dest=None)
    act = _step(
        cellar_mode, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_mode.success and cellar_mode.failed
    assert list(act.action) == IDLE

    cellar_screen = make_south_1e_controller(dest=None)
    act = _step(
        cellar_screen, _ram(mode=PLAY_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_screen.success and cellar_screen.failed
    assert list(act.action) == IDLE


def test_dest_0x2e_accepts_2e_rejects_3c() -> None:
    ok = make_south_1e_controller(dest=0x2E)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x2E, x=120, y=77))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    bad = make_south_1e_controller(dest=0x2E)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_south_1e_controller()
    assert isinstance(ctl, Level8South1EController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "DOWN"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level8_south_1e"
    assert report["dest_screen"] == SOUTH_DEST == 0x2E
    assert SOUTH_DEST_POSE == (120, 77)
    assert SOUTH_DEST != 0x3C
    assert SOUTH_DOOR == (120, 205)
    assert SOUTH_ORIGIN == 0x1E
    assert SOUTH_ORIGIN_POSE == (208, 141)
    assert SOUTH_ORIGIN != 0x3C


def test_gleeok_passage_factory_stays_unverified() -> None:
    ctl = make_gleeok_passage_controller()
    assert isinstance(ctl, UnverifiedLevel8PathController)
    assert not isinstance(ctl, Level8South1EController)


def test_2e_leftover_emits_down_not_align() -> None:
    """(120,77) is already on the south aisle. Hold DOWN, no LEFT/RIGHT."""
    snap = read_snapshot(_ram(screen=SOUTH_2E_ORIGIN, x=120, y=77))
    act = south_2e_step(snap)
    assert list(act.action) == DOWN
    assert list(act.action) != LEFT
    assert list(act.action) != RIGHT
    assert list(act.action) != UP
    assert act.reason == "south_approach"

    ctl = make_south_2e_controller()
    act = _step(ctl, _ram(screen=SOUTH_2E_ORIGIN, x=120, y=77))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert list(act.action) != LEFT
    assert act.reason == "south_approach"


def test_2e_off_aisle_aligns_then_south_door_pushes_down() -> None:
    left = make_south_2e_controller()
    act = _step(left, _ram(screen=SOUTH_2E_ORIGIN, x=160, y=77))
    assert not left.failed
    assert list(act.action) == LEFT
    assert act.reason == "south_align"

    right = make_south_2e_controller()
    act = _step(right, _ram(screen=SOUTH_2E_ORIGIN, x=80, y=77))
    assert not right.failed
    assert list(act.action) == RIGHT
    assert act.reason == "south_align"

    ctl = make_south_2e_controller()
    act = _step(
        ctl, _ram(screen=SOUTH_2E_ORIGIN, x=SOUTH_DOOR[0], y=SOUTH_DOOR[1])
    )
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert act.reason == "south_push"


def test_2e_dest_none_first_play_succeeds_gleeok_and_cellar_fail() -> None:
    ok = make_south_2e_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=SOUTH_2E_DEST_HYP, x=120, y=77))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    gleeok = make_south_2e_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE

    cellar_mode = make_south_2e_controller(dest=None)
    act = _step(
        cellar_mode, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_mode.success and cellar_mode.failed
    assert list(act.action) == IDLE

    cellar_screen = make_south_2e_controller(dest=None)
    act = _step(
        cellar_screen, _ram(mode=PLAY_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_screen.success and cellar_screen.failed
    assert list(act.action) == IDLE


def test_2e_dest_live_accepts_3e_rejects_3c() -> None:
    ok = make_south_2e_controller(dest=SOUTH_2E_DEST)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x3E, x=120, y=93))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    none = make_south_2e_controller(dest=None)
    act = _step(none, _ram(mode=PLAY_MODE, screen=0x3E, x=120, y=93))
    assert none.success and not none.failed

    bad = make_south_2e_controller(dest=SOUTH_2E_DEST)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE
    assert SOUTH_2E_DEST_POSE == (120, 93)


def test_2e_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_south_2e_controller()
    assert isinstance(ctl, Level8South2EController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "DOWN"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level8_south_2e"
    assert report["dest_screen"] == SOUTH_2E_DEST == 0x3E
    assert SOUTH_2E_DEST_HYP == SOUTH_2E_DEST == 0x3E
    assert SOUTH_2E_DEST != GLEEOK_HYP
    assert SOUTH_2E_DEST != CELLAR_ROOM
    assert SOUTH_2E_DEST_POSE == (120, 93)
    assert SOUTH_DOOR == (120, 205)
    assert SOUTH_2E_ORIGIN == 0x2E
    assert SOUTH_2E_ORIGIN_POSE == (120, 77)
    assert SOUTH_2E_ORIGIN != GLEEOK_HYP


def test_2e_gleeok_and_magic_key_factories_stay_unverified() -> None:
    gleeok = make_gleeok_passage_controller()
    assert isinstance(gleeok, UnverifiedLevel8PathController)
    assert not isinstance(gleeok, Level8South2EController)
    # rr-6o7.2: magic_key_stairs is now live (see test_level8_cellar).
    mk = make_magic_key_stairs_controller()
    assert not isinstance(mk, UnverifiedLevel8PathController)
    assert not isinstance(mk, Level8South2EController)
