"""Deterministic reset-to-level input sequence for The Magical Quest."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction

BOOT_SCRIPT_FRAMES = 2400


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Select one-player/default Mickey and reach the Stage 1 opening."""
    for frame in range(1, BOOT_SCRIPT_FRAMES + 1):
        slot = frame % 240
        if 20 <= slot < 30:
            yield FrameAction(buttons("START"), "boot_start")
        elif 100 <= slot < 108:
            yield FrameAction(buttons("B"), "boot_confirm")
        elif 160 <= slot < 166:
            yield FrameAction(buttons("Y"), "boot_confirm_alt")
        else:
            yield FrameAction(idle_action(), "boot_wait")

