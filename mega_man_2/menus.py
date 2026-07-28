"""Deterministic reset-to-Level-1 sequence for Mega Man 2 (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction

BOOT_MAX_FRAMES = 2500


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/stage-select inputs toward Air Man stage play."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 200
        if 20 <= slot < 28:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 60 <= slot < 66:
            yield FrameAction(nes_action("UP"), "boot_nav")
        elif 90 <= slot < 98:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 140 <= slot < 148:
            yield FrameAction(nes_action("START"), "boot_enter_stage")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")
