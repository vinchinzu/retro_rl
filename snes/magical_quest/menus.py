"""Deterministic reset-to-level input sequence for The Magical Quest."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_SCRIPT_FRAMES = 2400


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Select one-player/default Mickey and reach the Stage 1 opening."""
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
