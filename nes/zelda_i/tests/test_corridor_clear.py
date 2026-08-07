"""Unit tests for Clean corridor-clear controller."""

from __future__ import annotations

import numpy as np

from zelda_i.corridor_clear import CorridorClearController, CorridorClearPhase
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x5A)
    ram[ADDR_LINK_X] = fields.get("x", 40)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    filled = fields.get("filled", 2)
    ram[ADDR_HEALTH] = (3 << 4) | (filled & 0x0F)
    return ram


def test_clear_completes_after_max_frames() -> None:
    ctrl = CorridorClearController(max_frames=5, clear_screen=0x5A)
    for _ in range(5):
        snap = read_snapshot(_ram())
        ctrl.step(snap)
    assert ctrl.success
    assert ctrl.phase is CorridorClearPhase.DONE
    assert any(n.startswith("clear_done_") for n in ctrl.notes)


def test_clear_death_fails() -> None:
    ctrl = CorridorClearController(max_frames=100)
    snap = read_snapshot(_ram(mode=17, filled=0))
    ctrl.step(snap)
    assert not ctrl.success
    assert ctrl.phase is CorridorClearPhase.FAILED
    assert "link_death" in ctrl.notes


def test_clear_idle_without_enemies() -> None:
    ctrl = CorridorClearController(max_frames=100)
    snap = read_snapshot(_ram(x=40, y=141))
    act = ctrl.step(snap)
    assert "clear" in act.reason
    assert ctrl.phase is CorridorClearPhase.CLEAR
