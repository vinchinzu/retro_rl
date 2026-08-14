"""Deterministic reset-to-Lesson-1 input sequence for Pilotwings."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_SCRIPT_FRAMES = 1920


def boot_to_lesson1_plane_script() -> Iterator[FrameAction]:
    """Yield the verified USA-ROM sequence to the first light-plane lesson.

    A conservative input cadence advances the intro, accepts Lesson 1, and
    chooses the default light-plane objective. It ends airborne and unpaused.
    """
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
