"""Deterministic reset-to-Brinstar sequence for Metroid (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

BOOT_MAX_FRAMES = 4000


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title inputs toward Brinstar play.

    Title defaults to START selected. Pulse START a few times early, then
    idle while the engine leaves title mode and settles into play (mode 3).
    Avoid holding START after play begins (it pauses).
    """
    # Wait for title to paint.
    for _ in range(90):
        yield FrameAction(nes_idle_action(), "boot_title_wait")
    # Confirm START (may need a couple pulses if attract is mid-cycle).
    for pulse in range(6):
        for _ in range(4):
            yield FrameAction(nes_action("START"), "boot_start")
        for _ in range(50):
            yield FrameAction(nes_idle_action(), "boot_post_start")
        # Remaining frames: pure idle so we never pause mid-gameplay.
    for _ in range(BOOT_MAX_FRAMES - 90 - 6 * 54):
        yield FrameAction(nes_idle_action(), "boot_settle")
