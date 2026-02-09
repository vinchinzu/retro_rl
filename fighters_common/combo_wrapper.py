"""
ComboAction wrapper - adds special move macro actions to the action space.

When the agent selects a combo action, this wrapper bypasses FrameSkip and
executes the precise frame-by-frame button sequence needed for the special
move. Normal actions pass through unchanged.

Usage:
    # After building the normal env stack:
    env = ComboFrameSkip(base_env, combos=LIUKANG_COMBOS, n_skip=4)
    # Use this INSTEAD of regular FrameSkip

The combo action indices are appended after the normal action space.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# SNES button indices
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)


def _btns(**kwargs):
    b = np.zeros(12, dtype=np.int8)
    for k, v in kwargs.items():
        b[int(k)] = v
    return b

_NOOP = _btns()
_FWD  = _btns(**{str(_RIGHT): 1})
_BACK = _btns(**{str(_LEFT): 1})
_HP   = _btns(**{str(_Y): 1})
_HK   = _btns(**{str(_B): 1})
_LP   = _btns(**{str(_L): 1})


# Combo definitions: list of (button_array, n_frames)
# Timing from empirical testing: F(4) gap(2) F(4) gap(1) attack(4) = 15 frames
LIUKANG_COMBOS = [
    {
        "name": "Fireball",
        "sequence": [
            (_FWD, 4),    # Forward tap 1
            (_NOOP, 2),   # Release
            (_FWD, 4),    # Forward tap 2
            (_NOOP, 1),   # Release
            (_HP, 4),     # High Punch
        ],
    },
    {
        "name": "Flying Kick",
        "sequence": [
            (_FWD, 4),    # Forward tap 1
            (_NOOP, 2),   # Release
            (_FWD, 4),    # Forward tap 2
            (_NOOP, 1),   # Release
            (_HK, 4),     # High Kick
        ],
    },
]


class ComboFrameSkip(gym.Wrapper):
    """
    FrameSkip wrapper with combo macro support.

    Replaces the normal FrameSkip wrapper. For regular actions, repeats the
    action for n_skip frames (standard frame skip). For combo actions (indices
    >= n_normal_actions), executes a precise frame-by-frame button sequence.

    Must be placed in the same position as FrameSkip in the wrapper stack
    (directly on top of the base env or DirectRAMReader).

    The action space is MultiBinary(12) from the base env. The combo indices
    are tracked internally - upstream DiscreteAction maps combo action indices
    to a sentinel value that this wrapper intercepts.
    """

    # Sentinel button arrays that signal combo execution.
    # DiscreteAction maps combo indices to these, and ComboFrameSkip intercepts them.
    COMBO_SENTINEL_BIT = _SELECT  # Use SELECT button as sentinel (never used in fighting)

    def __init__(self, env, combos: list[dict] | None = None, n_skip: int = 4):
        super().__init__(env)
        self.n_skip = n_skip
        self.combos = combos or []

    def step(self, action):
        """Step with either normal frame-skip or combo execution."""
        # Check if this is a combo action (SELECT bit set as sentinel)
        is_array = isinstance(action, np.ndarray)
        if is_array and action[self.COMBO_SENTINEL_BIT] == 1:
            # Combo index encoded in remaining bits
            combo_idx = int(action[_START])  # Use START bit for combo index
            if 0 <= combo_idx < len(self.combos):
                if hasattr(self, '_debug') and self._debug:
                    print(f"  [ComboFrameSkip] Executing combo {combo_idx}: {self.combos[combo_idx]['name']}")
                return self._execute_combo(combo_idx)

        # Normal frame-skip
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = None
        for _ in range(self.n_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def _execute_combo(self, combo_idx):
        """Execute a combo sequence frame-by-frame."""
        combo = self.combos[combo_idx]
        sequence = combo["sequence"]

        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = None

        for buttons, n_frames in sequence:
            for _ in range(n_frames):
                obs, reward, terminated, truncated, info = self.env.step(buttons)
                total_reward += reward
                if terminated or truncated:
                    return obs, total_reward, terminated, truncated, info

        return obs, total_reward, terminated, truncated, info


def get_combo_actions(combos: list[dict]) -> list[dict]:
    """
    Generate DiscreteAction entries for combo actions.

    Returns action dicts that use SELECT as a sentinel bit (never used in
    normal fighting) with START encoding the combo index.
    """
    actions = []
    for i, combo in enumerate(combos):
        # SELECT=1 signals combo, START=combo_index
        action = {_SELECT: 1, _START: i}
        actions.append(action)
    return actions
