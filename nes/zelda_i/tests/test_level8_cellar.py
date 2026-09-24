"""Durable L8 Magical Key cellar 0x0F return: east-drop, fail-closed, no writes.

No emulator. Fake snapshots at leftover (136,141) must emit RIGHT toward
the east ladder, never LEFT into pit tile 250. F1 cardinal DOWN did not
move. Spine magic-key factory stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.cellar import (
    CELLAR_ROOM,
    EAST_X,
    EXIT_STAIRS,
    FLOOR_Y,
    PAD,
    PIT_TILE,
    WEST_X,
    magic_key_cellar_return_step,
    make_magic_key_cellar_return_controller,
)
from zelda_i.level8.path import (
    make_magic_key_stairs_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PASSAGE_MODE,
    "level": 8,
    "screen": CELLAR_ROOM,
    "x": PAD[0],
    "y": PAD[1],
    "tile": 36,
    "keys": 8,
    "bombs": 6,
    "magic_key": 1,
    "triforce": 0x7F,
}

LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())
LEFT_DOWN = list(nes_action("LEFT", "DOWN"))


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "cellar controllers must not write RAM"
    return act


def _no_strafe(act) -> None:
    assert list(act.action) != LEFT, act.reason
    assert list(act.action) != RIGHT, act.reason


def test_leftover_pad_goes_east_never_left_into_pit() -> None:
    snap = read_snapshot(_ram(x=136, y=141, tile=36))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=136, y=141, tile=36))
    assert not ctl.failed
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT


def test_east_column_drops_left_down() -> None:
    snap = read_snapshot(_ram(x=EAST_X, y=141))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == LEFT_DOWN

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EAST_X, y=141))
    assert not ctl.failed
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == LEFT_DOWN


def test_mid_ledge_keeps_right_until_east_column() -> None:
    """F2: x=160 is still the pit. Do not LEFT+DOWN until x>=174."""
    snap = read_snapshot(_ram(x=160, y=141, tile=36))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT_DOWN

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=160, y=141, tile=36))
    assert not ctl.failed
    assert act.reason == "cellar_to_east"


def test_pit_tile_250_fails_closed_idle() -> None:
    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=112, y=141, tile=PIT_TILE))
    assert ctl.failed and not ctl.success
    assert "pit_tile_250" in ctl.notes
    assert act.reason == "pit_tile_250"
    assert list(act.action) == IDLE


def test_floor_walks_west() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=100, y=FLOOR_Y)))
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=100, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT


def test_west_floor_climbs() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=WEST_X, y=FLOOR_Y)))
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_west_climb_holds_up() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=WEST_X, y=120)))
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=120))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP


def test_west_lip_tile_0x6f_keeps_up() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP
    assert list(act.action) != IDLE

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    assert not ctl.failed
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP


def test_stairs_tile_idles_exit_warp() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    assert not ctl.failed
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE


def test_play_mode_dest_none_succeeds_on_first_play() -> None:
    ctl = make_magic_key_cellar_return_controller(dest=None)
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=0x1F, x=120, y=205))
    assert ctl.success and not ctl.failed
    assert list(act.action) == IDLE
    assert act.reason == "left_0x0f_stairs"


def test_play_mode_dest_0x1f_rejects_0x3c() -> None:
    ok = make_magic_key_cellar_return_controller(dest=0x1F)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x1F, x=120, y=205))
    assert ok.success and not ok.failed
    assert act.reason == "left_0x0f_stairs"

    bad = make_magic_key_cellar_return_controller(dest=0x1F)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=0x3C, x=120, y=205))
    assert not bad.success
    assert bad.failed
    assert "unexpected_play_0x3c" in bad.notes
    assert list(act.action) == IDLE


def test_west_column_at_y141_goes_up_never_strafe() -> None:
    snap = read_snapshot(_ram(x=WEST_X, y=141))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP
    _no_strafe(act)

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=141))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP
    _no_strafe(act)


class _FakeEnv:
    """Minimal env so the controller's ADDR_MAGIC_KEY reads resolve."""

    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram

    def get_ram(self) -> np.ndarray:
        return self._ram


def _mk_ram(**fields: int) -> np.ndarray:
    base = {
        "mode": PLAY_MODE,
        "level": 8,
        "screen": 0x1F,
        "x": 120,
        "y": 141,
        "tile": 0,
        "keys": 1,
        "bombs": 15,
        "magic_key": 0,
        "triforce": 0x7F,
    }
    return make_ram(base, **fields)


def test_magic_key_stairs_fails_closed_off_room() -> None:
    ctl = make_magic_key_stairs_controller()
    ram = _mk_ram(screen=0x6D)
    ctl.bind_env(_FakeEnv(ram))
    before = ram.copy()
    ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before)
    assert ctl.failed and not ctl.success
    assert any("unexpected_room" in n for n in ctl.notes)


def test_magic_key_stairs_push_routes_out_of_the_boxed_south_pose() -> None:
    # power-on clear can leave Link south-east of the diamond (144,165);
    # LEFT there is a wall.  The push route must head east/up, not LEFT.
    ctl = make_magic_key_stairs_controller()
    ram = _mk_ram(x=144, y=165)
    ctl.bind_env(_FakeEnv(ram))
    ctl.phase = "push"
    ctl.block_xy0 = (96, 144)
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert list(act.action) in (RIGHT, UP), act.reason
    assert list(act.action) != LEFT


def test_magic_key_stairs_cellar_pickup_loop_then_return_handoff() -> None:
    ctl = make_magic_key_stairs_controller()
    ram = _mk_ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=128, y=141, tile=113)
    ctl.bind_env(_FakeEnv(ram))
    ctl.phase = "cellar"
    ctl.mk_before = 0
    # before the key: first move is DOWN off the entry-stairs warp tile.
    act = _step(ctl, ram)
    assert list(act.action) == list(nes_action("DOWN"))
    assert not ctl.failed and not ctl.mk_gained

    # key acquired mid-cellar -> hand off to the two-ladder return navigator.
    ram2 = _mk_ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=141, y=141, tile=36, magic_key=1)
    ctl.bind_env(_FakeEnv(ram2))
    act = _step(ctl, ram2)
    assert ctl.mk_gained
    assert act.reason.startswith("cellar_")
