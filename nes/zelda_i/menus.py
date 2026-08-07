"""Deterministic reset-to-Level-1 sequence for Zelda I (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

# Safety cap; ready is typically ~500–650 with BOOT_PERIOD below.
BOOT_MAX_FRAMES = 2000
# Button cycle period. Period 180 reached play ~1749f (~29s). Period 50 still
# clears is_level1_ready (~567f verified); 45 works; 40 fails menu debounce.
BOOT_PERIOD = 50


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/file inputs toward overworld play."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % BOOT_PERIOD
        # START ~8f, A ~6f, SELECT ~4f with idle between for menu debounce.
        if 6 <= slot < 14:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 22 <= slot < 28:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 33 <= slot < 37:
            yield FrameAction(nes_action("SELECT"), "boot_select")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")
