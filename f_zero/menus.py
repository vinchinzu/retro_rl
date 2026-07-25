"""Deterministic reset-to-Mute-City input sequence for F-Zero."""

from __future__ import annotations

from collections.abc import Iterator

from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.primitives import FrameAction

BOOT_SCRIPT_FRAMES = 1080


def boot_to_mute_city_script() -> Iterator[FrameAction]:
    """Select Grand Prix, Blue Falcon, beginner league, and Mute City I."""
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

