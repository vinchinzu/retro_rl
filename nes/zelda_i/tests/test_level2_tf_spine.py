"""Unit tests for L2 boom → Dodongo → TF spine table (no emulator)."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.bomb_wall_path import BombWallPhase
from zelda_i.level2_bomb_path import Level2BombNorth1eSpineController
from zelda_i.level2_tf_spine import (
    DodongoPhase,
    Level2DodongoController,
    Level2SouthBandUpController,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
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
    if dodo_hp is not None:
        ram[ADDR_OBJ_TYPE + 1] = 0x32
        ram[ADDR_LINK_X + 1] = 140
        ram[ADDR_LINK_Y + 1] = 141
        ram[ADDR_OBJ_HP + 1] = dodo_hp
    return read_snapshot(ram)


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


def test_bomb_north_1e_skips_to_stand_after_clip_overshoot() -> None:
    """Continuous v10: RIGHT+UP from (96,101) lands in the alcove; FACE."""
    ctl = Level2BombNorth1eSpineController()
    clip = ctl.step(_snap(room=0x1E, x=96, y=101, bombs=16))
    assert clip.reason == "stand_clip"
    act = ctl.step(_snap(room=0x1E, x=120, y=93, bombs=16))
    assert list(act.action) == list(nes_action("UP"))
    assert act.reason == "face_up"
    assert ctl.phase is BombWallPhase.FACE


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
