"""Deterministic reset-to-stage sequences for Mega Man 2 (NES).

Stage-select cursor (``$002A``), clockwise from Bubble Man per Data Crystal:

```
1 Bubble   2 Air     3 Quick
8 Heat     0 Wily    4 Wood
7 Metal    6 Flash   5 Crash
```

After password → robot select, cursor starts on **Wily (0)**.
``UP`` → Air (2); ``LEFT`` → Heat (8).
"""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

BOOT_MAX_FRAMES = 2500

# Cursor values (stage select)
CURSOR_WILY = 0
CURSOR_BUBBLE = 1
CURSOR_AIR = 2
CURSOR_QUICK = 3
CURSOR_WOOD = 4
CURSOR_CRASH = 5
CURSOR_FLASH = 6
CURSOR_METAL = 7
CURSOR_HEAT = 8


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/stage-select inputs toward Air Man stage play."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 200
        if 20 <= slot < 28:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 60 <= slot < 66:
            yield FrameAction(nes_action("UP"), "boot_nav")
        elif 90 <= slot < 98:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 140 <= slot < 148:
            yield FrameAction(nes_action("START"), "boot_enter_stage")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")


def boot_to_heat_man_script() -> Iterator[FrameAction]:
    """Yield title → password → stage select → Heat Man stage play.

    Same 200-frame period pattern as ``boot_to_level1_script`` (Air), but
    navigates **LEFT** from Wily → Heat (cursor 8) instead of UP → Air (2).

    Probe-verified (2026-08-10): password mean~102 → robot select mean~119
    at Wily(0) → LEFT → Heat(8) → A + START → READY ~f900+.
    """
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % 200
        if 20 <= slot < 28:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 60 <= slot < 66:
            # Wily(0) → Heat(8). Extra LEFT frames are ignored once selected.
            yield FrameAction(nes_action("LEFT"), "boot_nav_heat")
        elif 90 <= slot < 98:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 140 <= slot < 148:
            yield FrameAction(nes_action("START"), "boot_enter_stage")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")
