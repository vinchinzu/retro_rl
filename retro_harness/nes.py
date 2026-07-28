"""Focused public API for starting a new NES game with minimal imports.

Mirrors :mod:`retro_harness.snes` for fceumm-backed stable-retro integrations.
Genre-specific policy remains game-local until a second NES consumer needs a
shared package.
"""

from retro_harness.actions import (
    NES_ACTION_SIZE,
    NES_BUTTON_NAMES,
    nes_action,
    nes_buttons,
    nes_idle_action,
)
from retro_harness.env import GameSpec

__all__ = [
    "GameSpec",
    "NES_ACTION_SIZE",
    "NES_BUTTON_NAMES",
    "nes_action",
    "nes_buttons",
    "nes_idle_action",
]
