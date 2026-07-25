"""Compatibility imports for actions now owned by :mod:`retro_harness`."""

from retro_harness.actions import (
    ActionBuilder,
    SNES_ACTION_SIZE,
    action_names,
    buttons,
    buttons_multi,
    idle_action,
    idle_action_multi,
    multiplayer_action,
    snes_action,
)

NUM_BUTTONS = SNES_ACTION_SIZE
NUM_BUTTONS_MULTI = SNES_ACTION_SIZE * 2

__all__ = [
    "ActionBuilder",
    "NUM_BUTTONS",
    "NUM_BUTTONS_MULTI",
    "action_names",
    "buttons",
    "buttons_multi",
    "idle_action",
    "idle_action_multi",
    "multiplayer_action",
    "snes_action",
]
