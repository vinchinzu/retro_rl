"""Deterministic reset-to-Level-1 sequence for Super Mario Bros. 3 (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_MAX_FRAMES = 3000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title inputs toward World 1 map control."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=160,
        pulses=(
            PeriodPulse(20, 30, nes_action("START"), "boot_start"),
            PeriodPulse(90, 100, nes_action("A"), "boot_confirm"),
        ),
        idle=nes_idle_action(),
    )
