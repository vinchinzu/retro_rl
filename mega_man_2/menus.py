"""Deterministic reset-to-Level-1 sequence for Mega Man 2 (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction

BOOT_MAX_FRAMES = 5000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/menu inputs toward first controllable play."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 180
        if 20 <= slot < 30:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 70 <= slot < 80:
            yield FrameAction(nes_action("A"), "boot_confirm")

        elif 130 <= slot < 134:
            yield FrameAction(nes_action("SELECT"), "boot_select")
        elif 140 <= slot < 146:
            yield FrameAction(nes_action("DOWN"), "boot_nav")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")
