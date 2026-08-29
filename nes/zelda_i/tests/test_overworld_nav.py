from __future__ import annotations

import numpy as np

from zelda_i.overworld.nav import (
    NavPhase,
    OverworldToLevel1Controller,
    level1_entrance_success,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_START,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 160)
    ram[ADDR_HEALTH] = fields.get("health", 0x22)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_level1_entrance_success_dungeon() -> None:
    assert level1_entrance_success(_ram(level=1, screen=0x73), require_dungeon=True)
    assert not level1_entrance_success(
        _ram(level=0, screen=SCREEN_LEVEL1_ENTRANCE, sword=1),
        require_dungeon=True,
    )


def test_controller_advances_on_screen_78() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    # Force a step on 77 first so phase is set
    ctrl.step(read_snapshot(_ram(screen=SCREEN_START, x=120, y=140, sword=1)))
    snap = read_snapshot(_ram(screen=0x78, x=20, y=140, sword=1))
    ctrl.step(snap)
    assert ctrl.phase is NavPhase.NORTH_78
