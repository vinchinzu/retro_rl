"""Focused public API for starting a new SNES game with minimal imports.

Use this module for game identity, environment creation, named actions, and
title/menu scripts.  Genre-specific policy remains in packages such as
``platformer_common`` and ``snes_oneshot``.
"""

from retro_harness.actions import (
    ActionBuilder,
    SNES_ACTION_SIZE,
    action_names,
    buttons,
    buttons_multi,
    idle_action,
    idle_action_multi,
    indexed_action,
    multiplayer_action,
    snes_action,
)
from retro_harness.env import GameSpec
from retro_harness.input_script import (
    FrameAction,
    InputStep,
    ScriptResult,
    StartupPlan,
    input_step,
    iter_input_steps,
    parse_input_script,
    press_button_sequence,
    run_input_steps,
    run_startup,
)

__all__ = [
    "ActionBuilder",
    "FrameAction",
    "GameSpec",
    "InputStep",
    "SNES_ACTION_SIZE",
    "ScriptResult",
    "StartupPlan",
    "action_names",
    "buttons",
    "buttons_multi",
    "idle_action",
    "idle_action_multi",
    "indexed_action",
    "input_step",
    "iter_input_steps",
    "multiplayer_action",
    "parse_input_script",
    "press_button_sequence",
    "run_input_steps",
    "run_startup",
    "snes_action",
]
