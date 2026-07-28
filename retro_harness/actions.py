"""Canonical named-button action builders for stable-retro controllers.

Keep controller layout knowledge here.  Game and genre packages should describe
inputs by button name instead of copying SNES indices or zero-vector builders.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from retro_harness.controls import (
    NES_ACTION_SIZE,
    NES_BUTTON_NAMES,
    SNES_BUTTON_NAMES,
    action_from_nes_button_names,
    action_from_snes_button_names,
    pressed_snes_buttons,
)

SNES_ACTION_SIZE = len(SNES_BUTTON_NAMES)


def _active_names(
    button_names: Iterable[str],
    button_states: dict[str, bool],
) -> tuple[str, ...]:
    names = [str(name).strip().upper() for name in button_names]
    names.extend(
        name.strip().upper() for name, active in button_states.items() if active
    )
    return tuple(name for name in names if name)


def snes_action(
    *button_names: str,
    action_size: int = SNES_ACTION_SIZE,
    dtype: Any | None = None,
    **button_states: bool,
):
    """Build one action from positional names and/or truthy named flags.

    ``snes_action("RIGHT", "Y")`` and
    ``snes_action(right=True, y=True)`` are equivalent.  Lists are returned by
    default.  Pass a NumPy dtype when an environment or task API requires an
    ndarray, for example ``dtype=np.int32``.
    """

    names = _active_names(button_names, button_states)
    action = action_from_snes_button_names(names, action_size=action_size)
    if dtype is None:
        return action

    import numpy as np

    return np.asarray(action, dtype=dtype)


def indexed_action(
    button_indices: Iterable[int] = (),
    *,
    action_size: int = SNES_ACTION_SIZE,
    dtype: Any | None = None,
):
    """Build an action from indices for optimized tables and imported traces."""

    action = [0] * action_size
    for index in button_indices:
        if not 0 <= index < action_size:
            raise ValueError(f"button index {index} outside action size {action_size}")
        action[index] = 1
    if dtype is None:
        return action

    import numpy as np

    return np.asarray(action, dtype=dtype)


def idle_action(*, action_size: int = SNES_ACTION_SIZE, dtype: Any | None = None):
    """Return a released-controller action."""

    return snes_action(action_size=action_size, dtype=dtype)


def buttons(*names: str) -> list[int]:
    """Compatibility-friendly shorthand for a one-player SNES action."""

    return snes_action(*names)


def nes_action(
    *button_names: str,
    action_size: int = NES_ACTION_SIZE,
    dtype: Any | None = None,
    **button_states: bool,
):
    """Build one NES action from positional names and/or truthy named flags."""

    names = _active_names(button_names, button_states)
    action = action_from_nes_button_names(names, action_size=action_size)
    if dtype is None:
        return action

    import numpy as np

    return np.asarray(action, dtype=dtype)


def nes_buttons(*names: str) -> list[int]:
    """Shorthand for a one-player NES action."""

    return nes_action(*names)


def nes_idle_action(
    *,
    action_size: int = NES_ACTION_SIZE,
    dtype: Any | None = None,
):
    """Return a released-controller NES action."""

    return nes_action(action_size=action_size, dtype=dtype)


def multiplayer_action(
    *players: Iterable[str],
    action_size: int = SNES_ACTION_SIZE,
    dtype: Any | None = None,
):
    """Build a flat action vector from one iterable of button names per player."""

    if not players:
        raise ValueError("at least one player is required")
    action: list[int] = []
    for player in players:
        action.extend(snes_action(*player, action_size=action_size))
    if dtype is None:
        return action

    import numpy as np

    return np.asarray(action, dtype=dtype)


def idle_action_multi(
    *,
    players: int = 2,
    action_size: int = SNES_ACTION_SIZE,
    dtype: Any | None = None,
):
    """Return a released-controller vector for multiple players."""

    if players < 1:
        raise ValueError("players must be >= 1")
    return multiplayer_action(
        *(tuple() for _ in range(players)),
        action_size=action_size,
        dtype=dtype,
    )


def buttons_multi(
    p1: tuple[str, ...] = (),
    p2: tuple[str, ...] = (),
) -> list[int]:
    """Build the historical two-player 24-button SNES action vector."""

    return multiplayer_action(p1, p2)


def action_names(action: Iterable[int]) -> tuple[str, ...]:
    """Return the canonical names of pressed buttons in a one-player action."""

    return tuple(pressed_snes_buttons(list(action)))


class ActionBuilder:
    """Small fluent builder retained for policies that compose an action."""

    def __init__(self, *, action_size: int = SNES_ACTION_SIZE) -> None:
        self.action_size = action_size
        self._names: list[str] = []

    def press(self, *names: str) -> ActionBuilder:
        """Hold the named buttons."""

        # Validate now so errors point at the call that added the bad button.
        snes_action(*names, action_size=self.action_size)
        self._names.extend(name.upper() for name in names)
        return self

    def clear(self) -> ActionBuilder:
        """Release all buttons."""

        self._names.clear()
        return self

    def build(self) -> list[int]:
        """Return a new action vector."""

        return snes_action(*self._names, action_size=self.action_size)


__all__ = [
    "ActionBuilder",
    "NES_ACTION_SIZE",
    "NES_BUTTON_NAMES",
    "SNES_ACTION_SIZE",
    "action_names",
    "buttons",
    "buttons_multi",
    "idle_action",
    "idle_action_multi",
    "indexed_action",
    "multiplayer_action",
    "nes_action",
    "nes_buttons",
    "nes_idle_action",
    "snes_action",
]
