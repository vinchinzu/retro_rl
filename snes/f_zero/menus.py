"""Deterministic reset-to-Mute-City input sequence for F-Zero."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_SCRIPT_FRAMES = 1080


def boot_to_mute_city_script() -> Iterator[FrameAction]:
    """Select Grand Prix, Blue Falcon, beginner league, and Mute City I."""
    yield from period_script(
        max_frames=BOOT_SCRIPT_FRAMES,
        period=240,
        pulses=(
            PeriodPulse(20, 30, buttons("START"), "boot_start"),
            PeriodPulse(100, 108, buttons("B"), "boot_confirm"),
            PeriodPulse(160, 166, buttons("Y"), "boot_confirm_alt"),
        ),
        idle=idle_action(),
    )
