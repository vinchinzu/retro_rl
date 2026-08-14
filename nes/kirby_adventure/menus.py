"""Deterministic reset-to-Level-1 sequence for Kirby's Adventure (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_MAX_FRAMES = 6000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/intro inputs toward Vegetable Valley hub."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=200,
        pulses=(
            PeriodPulse(20, 32, nes_action("START"), "boot_start"),
            PeriodPulse(100, 110, nes_action("A"), "boot_confirm"),
        ),
        idle=nes_idle_action(),
    )
