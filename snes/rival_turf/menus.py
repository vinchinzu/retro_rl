"""Deterministic reset-to-Stage-1 input sequence for Rival Turf!."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction

BOOT_SCRIPT_FRAMES = 2000


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield the verified USA-ROM title/menu sequence.

    The conservative cadence advances the publisher screens, selects the
    default one-player mode and Jack Flak, skips the stage map, and leaves the
    game unpaused at the Stage 1 opening.
    """
    for frame in range(1, BOOT_SCRIPT_FRAMES + 1):
        slot = frame % 240
        if 20 <= slot < 30:
            yield FrameAction(buttons("START"), "boot_start")
        elif 100 <= slot < 108:
            yield FrameAction(buttons("Y"), "boot_confirm")
        else:
            yield FrameAction(idle_action(), "boot_wait")

