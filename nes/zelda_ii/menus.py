"""Deterministic reset-to-Level-1 sequence for Zelda II (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_MAX_FRAMES = 5000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/file inputs toward North Palace."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=180,
        pulses=(
            PeriodPulse(20, 28, nes_action("START"), "boot_start"),
            PeriodPulse(80, 86, nes_action("A"), "boot_confirm"),
            PeriodPulse(120, 124, nes_action("SELECT"), "boot_select"),
        ),
        idle=nes_idle_action(),
    )
