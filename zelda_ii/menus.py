"""Deterministic reset-to-Level-1 sequence for Zelda II (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction

BOOT_MAX_FRAMES = 5000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/file inputs toward North Palace."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 180
        if 20 <= slot < 28:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 80 <= slot < 86:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 120 <= slot < 124:
            yield FrameAction(nes_action("SELECT"), "boot_select")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")

