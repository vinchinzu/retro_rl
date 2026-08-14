"""Deterministic reset-to-Stage-1 input sequence for Rival Turf!."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_SCRIPT_FRAMES = 2000


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield the verified USA-ROM title/menu sequence.

    The conservative cadence advances the publisher screens, selects the
    default one-player mode and Jack Flak, skips the stage map, and leaves the
    game unpaused at the Stage 1 opening.
    """
    yield from period_script(
        max_frames=BOOT_SCRIPT_FRAMES,
        period=240,
        pulses=(
            PeriodPulse(20, 30, buttons("START"), "boot_start"),
            PeriodPulse(100, 108, buttons("Y"), "boot_confirm"),
        ),
        idle=idle_action(),
    )
