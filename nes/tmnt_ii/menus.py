"""Deterministic reset-to-Level-1 sequence for TMNT II (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction, PeriodPulse, period_script

BOOT_MAX_FRAMES = 4000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/select inputs toward Stage 1 combat."""
    yield from period_script(
        max_frames=BOOT_MAX_FRAMES,
        period=180,
        pulses=(
            PeriodPulse(20, 30, nes_action("START"), "boot_start"),
            PeriodPulse(80, 90, nes_action("A"), "boot_confirm"),
        ),
        idle=nes_idle_action(),
    )
