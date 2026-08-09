"""Shared fighting-game utilities.

The public facade is deliberately lazy.  Importing :mod:`retro_harness.fighters`
must remain safe in the core (non-ML) installation; Gymnasium, OpenCV, Torch,
and Stable-Baselines are only imported when a caller asks for an ML-facing
attribute.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "FightingGameConfig": "retro_harness.fighters.fighting_env",
    "FightingEnv": "retro_harness.fighters.fighting_env",
    "DirectRAMReader": "retro_harness.fighters.fighting_env",
    "FrameSkip": "retro_harness.fighters.fighting_env",
    "FrameStack": "retro_harness.fighters.fighting_env",
    "GrayscaleResize": "retro_harness.fighters.fighting_env",
    "DiscreteAction": "retro_harness.fighters.fighting_env",
    "FIGHTING_ACTIONS": "retro_harness.fighters.fighting_env",
    "make_fighting_env": "retro_harness.fighters.fighting_env",
    "RamObservation": "retro_harness.fighters.ram_observation",
    "make_ram_fighting_env": "retro_harness.fighters.ram_observation",
    "build_eval_env": "retro_harness.fighters.ram_observation",
    "MK1_RAM_FEATURES": "retro_harness.fighters.ram_observation",
    "MenuNavigator": "retro_harness.fighters.menu_nav",
    "navigate_to_fight": "retro_harness.fighters.menu_nav",
    "create_fight_state": "retro_harness.fighters.menu_nav",
    "GAME_REGISTRY": "retro_harness.fighters.game_configs",
    "get_game_config": "retro_harness.fighters.game_configs",
}


def __getattr__(name: str) -> Any:
    """Resolve public attributes without importing optional stacks eagerly."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))

__all__ = [
    "FightingGameConfig",
    "FightingEnv",
    "DirectRAMReader",
    "FrameSkip",
    "FrameStack",
    "GrayscaleResize",
    "DiscreteAction",
    "FIGHTING_ACTIONS",
    "make_fighting_env",
    "RamObservation",
    "make_ram_fighting_env",
    "build_eval_env",
    "MK1_RAM_FEATURES",
    "MenuNavigator",
    "navigate_to_fight",
    "create_fight_state",
    "GAME_REGISTRY",
    "get_game_config",
]
