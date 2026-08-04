"""SNES action builders for golf menus and swings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_BUTTON_NAME_TO_INDEX,
    SNES_DOWN,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_START,
    SNES_UP,
    SNES_X,
    SNES_Y,
)

ACTION_SIZE = 12


def idle() -> np.ndarray:
    """Return a no-input action."""
    return np.zeros(ACTION_SIZE, dtype=np.int8)


def press(*buttons: int) -> np.ndarray:
    """Return an action with the given button indices held."""
    action = idle()
    for button in buttons:
        if 0 <= button < ACTION_SIZE:
            action[button] = 1
    return action


def press_named(*names: str) -> np.ndarray:
    """Return an action from SNES button name labels."""
    indices: list[int] = []
    for name in names:
        idx = SNES_BUTTON_NAME_TO_INDEX.get(name.upper())
        if idx is not None:
            indices.append(idx)
    return press(*indices)


def tap_sequence(
    button: int,
    *,
    hold: int = 2,
    gap: int = 8,
    times: int = 1,
) -> list[np.ndarray]:
    """Build a hold/release sequence for one button."""
    frames: list[np.ndarray] = []
    for _ in range(times):
        frames.extend(press(button) for _ in range(hold))
        frames.extend(idle() for _ in range(gap))
    return frames


def named_script(steps: Sequence[tuple[str, int]]) -> list[np.ndarray]:
    """Expand ``(button_name|IDLE, frames)`` into per-frame actions."""
    frames: list[np.ndarray] = []
    for label, count in steps:
        action = idle() if label.upper() == "IDLE" else press_named(label)
        frames.extend(action for _ in range(max(0, count)))
    return frames


def flatten_scripts(scripts: Iterable[Sequence[np.ndarray]]) -> list[np.ndarray]:
    """Concatenate multiple frame scripts."""
    out: list[np.ndarray] = []
    for script in scripts:
        out.extend(script)
    return out


# Hal's Hole in One uses Japanese-style confirms on menus (B=ENTER).
CONFIRM = SNES_B
CANCEL = SNES_A
ALT_CONFIRM = SNES_A
ALT_CANCEL = SNES_X
START = SNES_START
UP = SNES_UP
DOWN = SNES_DOWN
LEFT = SNES_LEFT
RIGHT = SNES_RIGHT
