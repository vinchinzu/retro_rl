from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_ii.north_palace import LeavePalacePolicy
from zelda_ii.ram import (
    ADDR_ENGINE_MODE,
    ADDR_HEALTH,
    ADDR_LIFE,
    MODE_OVERWORLD,
    MODE_SIDESCROLL,
)


def _ram(mode: int, *, health: int = 127, magic: int = 127) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_ENGINE_MODE] = mode
    ram[ADDR_HEALTH] = health
    ram[ADDR_LIFE] = magic
    return ram


def test_walk_left_while_side_scroll() -> None:
    pol = LeavePalacePolicy()
    tick = pol.tick(_ram(MODE_SIDESCROLL))
    assert tick.reason == "walk_left"
    assert list(tick.action) == list(nes_action("LEFT"))


def test_idle_on_transition_overworld_and_death() -> None:
    pol = LeavePalacePolicy()
    trans = pol.tick(_ram(16))
    assert trans.reason == "transition"
    assert list(trans.action) == list(nes_idle_action())

    done = pol.tick(_ram(MODE_OVERWORLD))
    assert done.reason == "clear_hold"
    assert list(done.action) == list(nes_idle_action())

    dead = pol.tick(_ram(MODE_SIDESCROLL, health=0))
    assert dead.reason == "dead"
