"""Deterministic reset-to-Level-1 sequence for Kirby's Adventure (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction

BOOT_MAX_FRAMES = 6000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/intro inputs toward Vegetable Valley hub."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 200
        if 20 <= slot < 32:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 100 <= slot < 110:
            yield FrameAction(nes_action("A"), "boot_confirm")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")
