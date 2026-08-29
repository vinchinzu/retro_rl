"""Deterministic reset-to-Level-1 sequence for Super Mario Bros. (NES)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script
from smb.ram import is_level1_ready

BOOT_MAX_FRAMES = 2000
MIN_BOOT_FRAME = 200
STABLE_BOOT_FRAMES = 20


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title inputs toward World 1-1 play."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=120,
        pulses=(PeriodPulse(20, 28, nes_action("START"), "boot_start"),),
        idle=nes_idle_action(),
    )


def boot_to_ready(
    env: Any,
    *,
    min_frame: int = MIN_BOOT_FRAME,
    stable_frames: int = STABLE_BOOT_FRAMES,
) -> tuple[object, int]:
    """Step the title script until Level 1 is stably ready."""
    frame = 0
    obs = None
    stable = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        mean = float(np.asarray(obs).mean())
        if frame >= min_frame and is_level1_ready(env.get_ram(), obs_mean=mean):
            stable += 1
        else:
            stable = 0
        if stable >= stable_frames:
            return obs, frame
    return obs, frame


def idle_n(env: Any, n: int) -> object:
    """Hold idle for ``n`` frames (natural-entry phase align)."""
    obs = None
    idle = np.asarray(nes_idle_action(), dtype=np.int8)
    action_size = int(env.action_space.shape[0])
    if idle.shape[0] != action_size:
        idle = np.zeros(action_size, dtype=np.int8)
    for _ in range(n):
        obs, *_ = env.step(idle)
    return obs
