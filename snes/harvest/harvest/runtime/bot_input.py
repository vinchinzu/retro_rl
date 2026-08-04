"""Controller/keyboard input helpers for Harvest play sessions."""

from __future__ import annotations

import os

import numpy as np
import pygame

from retro_harness import (
    SNES_L,
    SNES_R,
    SNES_SELECT,
    controller_action,
    describe_input_mapping,
    format_input_mapping,
    init_controller as _init_controller,
    keyboard_action,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
HOTSWAP_KEYS = {pygame.K_a, pygame.K_s, pygame.K_TAB}


def init_controller():
    return _init_controller(pygame)


def get_controller_action(joystick, action):
    controller_action(joystick, action)


def get_keyboard_action(keys, action):
    keyboard_action(keys, action, pygame)


def check_hotswap_chord(joystick, keys):
    """Check if hotswap chord (L+R+SELECT) is pressed."""
    if all(keys[k] for k in HOTSWAP_KEYS):
        return True
    if joystick is not None:
        try:
            action = np.zeros(12, dtype=np.int32)
            controller_action(joystick, action)
            if action[SNES_L] and action[SNES_R] and action[SNES_SELECT]:
                return True
        except Exception:
            pass
    return False


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def print_controls(joystick=None):
    """Print Harvest Moon control scheme."""
    print("\nControls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print(f"    Mapping: {format_input_mapping(describe_input_mapping(joystick=joystick))}")
        print("    D-Pad/Stick: Movement")
        print("    B: Run/Cancel | A: Confirm/Talk | Y: Use Item | X: Menu")
        print("    L/R: Cycle Items")
        print("    L+R+SELECT: Toggle Human/Bot Mode")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Cancel (B) | C: Confirm (A) | V: Menu (X) | X: Use Item (Y)")
    print("    A/S: Cycle Items (L/R)")
    print("    A+S+TAB: Toggle Human/Bot Mode")
    print("    P: Mark current tile as no-go (debug)")
    print("    F2: Search RAM for value | F3: Narrow search after change")
    print("    F5: Save recording when --record, otherwise save as 'latest' | F9: Load quicksave")
