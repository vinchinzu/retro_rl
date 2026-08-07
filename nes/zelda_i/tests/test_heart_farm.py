"""Unit tests for Clean heart-farm controller (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.heart_farm import HeartFarmController, HeartFarmPhase
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x4A)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    # high nibble = containers-1 (3 → 4 containers), low = filled
    health = fields.get("health")
    if health is None:
        filled = fields.get("filled", 2)
        containers_minus_1 = fields.get("containers", 4) - 1
        health = (containers_minus_1 << 4) | (filled & 0x0F)
    ram[ADDR_HEALTH] = health
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x01)
    return ram


def test_farm_skips_when_min_zero() -> None:
    ctrl = HeartFarmController(min_filled=0)
    snap = read_snapshot(_ram(filled=1))
    act = ctrl.step(snap)
    assert ctrl.success
    assert ctrl.phase is HeartFarmPhase.DONE
    assert "farm_skipped" in ctrl.notes
    assert act.reason == "farm_done"


def test_farm_already_satisfied() -> None:
    ctrl = HeartFarmController(min_filled=2)
    snap = read_snapshot(_ram(filled=3))
    ctrl.step(snap)
    assert ctrl.success
    assert ctrl.phase is HeartFarmPhase.DONE
    assert any(n.startswith("farm_ok_") for n in ctrl.notes)


def test_farm_patrols_until_hearts() -> None:
    ctrl = HeartFarmController(min_filled=3, max_frames=100)
    # Start under threshold — patrol action.
    snap = read_snapshot(_ram(filled=1, x=10, y=141))
    act = ctrl.step(snap)
    assert not ctrl.success
    assert ctrl.phase is HeartFarmPhase.FARM
    assert "farm" in act.reason

    # Reach threshold.
    snap = read_snapshot(_ram(filled=3, x=10, y=141))
    ctrl.step(snap)
    assert ctrl.success
    rep = ctrl.report()
    assert rep["start_filled"] == 1
    assert rep["peak_filled"] == 3


def test_farm_death_fails() -> None:
    ctrl = HeartFarmController(min_filled=3)
    snap = read_snapshot(_ram(mode=17, filled=0))
    ctrl.step(snap)
    assert not ctrl.success
    assert ctrl.phase is HeartFarmPhase.FAILED
    assert "link_death" in ctrl.notes


def test_farm_timeout_under_min() -> None:
    ctrl = HeartFarmController(min_filled=3, max_frames=5)
    for _ in range(5):
        snap = read_snapshot(_ram(filled=1))
        ctrl.step(snap)
    assert not ctrl.success
    assert ctrl.phase is HeartFarmPhase.FAILED
    assert any("farm_timeout" in n for n in ctrl.notes)


def test_already_satisfied_helper() -> None:
    ctrl = HeartFarmController(min_filled=3)
    assert ctrl.already_satisfied(read_snapshot(_ram(filled=3)))
    assert not ctrl.already_satisfied(read_snapshot(_ram(filled=2)))
