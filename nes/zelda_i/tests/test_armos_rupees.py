"""Unit tests for ArmosRupeeController (0x3D 30R and 0x4E 10R caves)."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from retro_harness.nes import nes_action
from zelda_i.dungeon.tilemap import (
    ADDR_ROOM_TILE_MAP,
    PLAYFIELD_TOP_Y,
    TILE_PX,
    TILE_ROWS,
    WRAM_BASE,
    WRAM_RAM_OFFSET,
)
from zelda_i.overworld.armos_rupees import (
    ARMOS_3D_CAVE_ID,
    ARMOS_3D_PAYOUT,
    ARMOS_3D_SCREEN,
    ARMOS_3D_STAND,
    ARMOS_3D_TILE,
    ARMOS_4E_CAVE_ID,
    ARMOS_4E_PAYOUT,
    ARMOS_4E_SCREEN,
    ARMOS_4E_STAND,
    ARMOS_4E_TILE,
    ArmosRupeeController,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RUPEES,
    ADDR_SCREEN,
    CAVE_MODE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, screen: int, x: int, y: int, mode: int = PLAY_MODE, level: int = 0, rupees: int = 20) -> np.ndarray:
    ram = np.zeros(10240, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_RUPEES] = rupees
    return ram


def _set_tile(ram: np.ndarray, x: int, y: int, tile: int) -> None:
    col = int(x) // TILE_PX
    row = (int(y) - PLAYFIELD_TOP_Y) // TILE_PX
    idx = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE + col * TILE_ROWS + row
    ram[idx] = tile & 0xFF


def _env(ram: np.ndarray):
    return SimpleNamespace(get_ram=lambda: ram)


def test_armos_rupees_constants() -> None:
    assert ARMOS_3D_SCREEN == 0x3D
    assert ARMOS_3D_STAND == (128, 125)
    assert ARMOS_3D_TILE == (144, 128)
    assert ARMOS_3D_CAVE_ID == 0x21
    assert ARMOS_3D_PAYOUT == 30

    assert ARMOS_4E_SCREEN == 0x4E
    assert ARMOS_4E_STAND == (144, 125)
    assert ARMOS_4E_TILE == (160, 128)
    assert ARMOS_4E_CAVE_ID == 0x23
    assert ARMOS_4E_PAYOUT == 10


def test_armos_rupees_wrong_screen_fails() -> None:
    ctl = ArmosRupeeController()
    ram = _ram(screen=0x3C, x=128, y=125)
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed is True
    assert act.reason == "armos_wrong_screen"


def test_armos_rupees_walks_to_stand_and_touches() -> None:
    ctl = ArmosRupeeController()
    ram = _ram(screen=0x3D, x=120, y=125)
    ctl.bind_env(_env(ram))

    # link at x=120, stand is (128, 125) -> walks RIGHT
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "armos_stand"
    assert act.action == nes_action("RIGHT")

    # link reaches stand (128, 125) -> phase becomes touch
    ram[ADDR_LINK_X] = 128
    act2 = ctl.step(read_snapshot(ram))
    assert act2.reason == "armos_touch"
    assert act2.action == nes_action("RIGHT")
    assert ctl._phase == "touch"

    # stairs revealed -> phase becomes enter
    _set_tile(ram, *ARMOS_3D_TILE, 0x70)
    act3 = ctl.step(read_snapshot(ram))
    assert ctl._phase == "enter"
    assert act3.action == nes_action("RIGHT")


def test_armos_rupees_cave_collection_success() -> None:
    ctl = ArmosRupeeController()
    # Inside cave mode
    ram = _ram(screen=0x3D, x=120, y=200, mode=CAVE_MODE, level=0, rupees=20)
    ctl.bind_env(_env(ram))

    act = ctl.step(read_snapshot(ram))
    assert ctl.success is False
    assert ctl._cave_rupees == 20

    # Payout credited: 20 + 30 = 50 rupees
    ram[ADDR_RUPEES] = 50
    act2 = ctl.step(read_snapshot(ram))
    assert ctl.success is True
    assert "armos_0x3d_rupees_20_to_50" in ctl.notes
