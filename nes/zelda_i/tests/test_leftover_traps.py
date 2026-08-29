"""One leftover trap per burned hop. Independent RAM/screenshot evidence."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.overworld.graph import LEVEL2_PATH_SCREENS, neighbor_screens
from zelda_i.overworld.nav import OverworldToLevel1Controller
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    CAVE_MODE,
    PLAY_MODE,
    SCREEN_START,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_HEALTH] = fields.get("health", 0x22)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_cave_exit_at_64_77_goes_down_first() -> None:
    """After sword cave exit ~(64,77) on 0x77, first travel is DOWN to y≈140."""
    ctrl = OverworldToLevel1Controller()
    act = ctrl.step(read_snapshot(_ram(screen=0x77, x=64, y=77, sword=1)))
    assert list(act.action) == list(nes_action("DOWN"))


def test_sword_pickup_at_x120_walks_up() -> None:
    """Cave pickup: x≈120 then UP. Mode 11, floor spawn, no sword yet."""
    ctrl = SwordCaveController()
    snap = read_snapshot(_ram(mode=CAVE_MODE, x=120, y=213, sword=0))
    act = None
    for _ in range(40):
        act = ctrl.step(snap)
    assert act is not None
    assert list(act.action) == list(nes_action("UP"))


def test_l2_prefix_never_enters_79() -> None:
    """L2 prefix is 37→38→48→58→59→49→4A. 0x79 is a rocky dead-end."""
    assert LEVEL2_PATH_SCREENS == (0x37, 0x38, 0x48, 0x58, 0x59, 0x49, 0x4A)
    assert 0x79 not in LEVEL2_PATH_SCREENS
    for a, b in zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()
