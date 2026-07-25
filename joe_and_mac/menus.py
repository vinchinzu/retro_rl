"""Deterministic reset-to-Stage-1 sequence for Joe & Mac."""

from __future__ import annotations

from collections.abc import Iterator

from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.primitives import FrameAction

BOOT_SCRIPT_FRAMES = 2820


def _repeat(action: FrameAction, frames: int) -> Iterator[FrameAction]:
    for _ in range(frames):
        yield action


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield the verified 1P/menu/map route to controllable Stage 1."""
    for frame in range(1, 801):
        slot = frame % 240
        if 20 <= slot < 30:
            yield FrameAction(buttons("START"), "boot_start")
        elif 100 <= slot < 108:
            yield FrameAction(buttons("B"), "boot_confirm")
        elif 160 <= slot < 166:
            yield FrameAction(buttons("Y"), "boot_confirm_alt")
        else:
            yield FrameAction(idle_action(), "boot_wait")

    yield from _repeat(FrameAction(buttons("UP"), "map_up"), 30)
    yield from _repeat(FrameAction(idle_action(), "map_wait"), 180)
    yield from _repeat(FrameAction(buttons("RIGHT"), "map_right"), 30)
    yield from _repeat(FrameAction(idle_action(), "map_wait"), 240)
    yield from _repeat(FrameAction(buttons("UP"), "map_select_node"), 30)
    yield from _repeat(FrameAction(idle_action(), "map_wait"), 600)
    yield from _repeat(FrameAction(buttons("B"), "map_confirm"), 10)
    yield from _repeat(FrameAction(idle_action(), "stage_wait"), 900)
