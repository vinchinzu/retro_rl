"""Deterministic reset-to-Level-1 sequence for TMNT II (NES)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script
from tmnt_ii.ram import is_level1_ready

BOOT_MAX_FRAMES = 4000
# Matches boot_probe leftover: 40f RIGHT after first Level 1 ready.
BOOT_LEFTOVER_WALK_FRAMES = 40


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/select inputs toward Stage 1 combat."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=180,
        pulses=(
            PeriodPulse(20, 30, nes_action("START"), "boot_start"),
            PeriodPulse(80, 90, nes_action("A"), "boot_confirm"),
        ),
        idle=nes_idle_action(),
    )


def leftover_walk_script(
    *, walk_frames: int = BOOT_LEFTOVER_WALK_FRAMES
) -> Iterator[FrameAction]:
    """Hold RIGHT after menus so play starts on the boot_probe leftover."""
    action = nes_action("RIGHT")
    for _ in range(walk_frames):
        yield FrameAction(action, "leftover_walk")


def boot_to_leftover(
    env: Any, *, walk_frames: int = BOOT_LEFTOVER_WALK_FRAMES
) -> tuple[Any, int, bool]:
    """Menus until Level 1 is live, then leftover RIGHT walk.

    Returns ``(obs, frames, ready)``. ``ready`` is False if menus expire.
    """
    obs: Any = None
    frame = 0
    ready = False
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            ready = True
            break
    if not ready:
        return obs, frame, False
    for step in leftover_walk_script(walk_frames=walk_frames):
        obs, *_ = env.step(step.action)
        frame += 1
    return obs, frame, True
