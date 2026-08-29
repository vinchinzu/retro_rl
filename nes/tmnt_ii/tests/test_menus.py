from __future__ import annotations

import numpy as np

from tmnt_ii.menus import (
    BOOT_LEFTOVER_WALK_FRAMES,
    BOOT_MAX_FRAMES,
    boot_to_leftover,
    boot_to_level1_script,
    leftover_walk_script,
)
from tmnt_ii.ram import ADDR_HEALTH, ADDR_LIVES


def test_boot_script_is_bounded_and_nonempty() -> None:
    script = list(boot_to_level1_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_start" in reasons


def test_leftover_walk_holds_right() -> None:
    from retro_harness.nes import nes_action

    frames = list(leftover_walk_script())
    assert len(frames) == BOOT_LEFTOVER_WALK_FRAMES == 40
    assert {frame.reason for frame in frames} == {"leftover_walk"}
    assert frames[0].action == nes_action("RIGHT")
    assert list(leftover_walk_script(walk_frames=0)) == []


class _FakeBootEnv:
    def __init__(self, ready_at: int) -> None:
        self.ready_at = ready_at
        self.steps = 0
        self.ram = np.zeros(0x800, dtype=np.uint8)

    def step(self, _action: object) -> tuple[np.ndarray, int, bool, dict]:
        self.steps += 1
        if self.steps >= self.ready_at:
            self.ram[ADDR_HEALTH] = 60
            self.ram[ADDR_LIVES] = 2
        obs = np.full((8, 8, 3), 80, dtype=np.uint8)
        return obs, 0, False, {}

    def get_ram(self) -> np.ndarray:
        return self.ram


def test_boot_to_leftover_walks_after_ready() -> None:
    env = _FakeBootEnv(ready_at=5)
    _obs, frames, ready = boot_to_leftover(env, walk_frames=3)
    assert ready is True
    assert frames == 5 + 3
    assert env.steps == 8


def test_boot_to_leftover_reports_unready() -> None:
    env = _FakeBootEnv(ready_at=10_000)
    _obs, frames, ready = boot_to_leftover(env, walk_frames=3)
    assert ready is False
    assert frames == BOOT_MAX_FRAMES
    assert env.steps == BOOT_MAX_FRAMES
