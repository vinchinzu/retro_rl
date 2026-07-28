from __future__ import annotations

import numpy as np

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
)
from zelda_i.sword_cave import (
    SwordCaveController,
    SwordPhase,
    sword_segment_success,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_HEALTH] = fields.get("health", 0x22)
    ram[ADDR_SWORD] = fields.get("sword", 0)
    return ram


def test_sword_segment_success_predicate() -> None:
    assert sword_segment_success(_ram(sword=1)) is True
    assert sword_segment_success(_ram(sword=0)) is False
    assert sword_segment_success(_ram(sword=1, mode=CAVE_MODE)) is False


def test_controller_moves_toward_cave_approach() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = SwordCaveController()
    snap = read_snapshot(_ram(x=120, y=141))
    action = ctrl.step(snap)
    assert ctrl.phase is SwordPhase.APPROACH_DOOR
    assert action.reason in {"approach_x", "approach_y"}


def test_controller_detects_done_with_sword_on_start() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = SwordCaveController()
    snap = read_snapshot(_ram(sword=1, mode=PLAY_MODE, screen=SCREEN_START))
    action = ctrl.step(snap)
    assert ctrl.success is True
    assert ctrl.phase is SwordPhase.DONE
    assert action.reason == "done"
