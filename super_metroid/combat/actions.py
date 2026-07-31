"""Discrete combat action table for structured-state RL.

Maps integer actions ↔ SNES button-name tuples used by controllers and the
feature-vector Gym env. Keep the table small and shared so strategy, RL, and
distillation all speak the same language.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from retro_harness.actions import buttons, idle_action

# Fixed discrete set for Bomb Torizo structured RL (and future bosses).
# Index is the Gym Discrete action id.
COMBAT_ACTION_NAMES: tuple[tuple[str, ...], ...] = (
    (),  # 0 idle
    ("LEFT",),  # 1
    ("RIGHT",),  # 2
    ("LEFT", "A"),  # 3 jump left
    ("RIGHT", "A"),  # 4 jump right
    ("LEFT", "X"),  # 5 fire left
    ("RIGHT", "X"),  # 6 fire right
    ("LEFT", "A", "X"),  # 7 jump+fire left
    ("RIGHT", "A", "X"),  # 8 jump+fire right
    ("X",),  # 9 fire only (face from pose)
    ("A",),  # 10 jump only
    ("LEFT", "B"),  # 11 dash left
    ("RIGHT", "B"),  # 12 dash right
)

N_COMBAT_ACTIONS = len(COMBAT_ACTION_NAMES)


def action_names(action_id: int) -> tuple[str, ...]:
    """Button names for a discrete combat action id."""
    if not 0 <= action_id < N_COMBAT_ACTIONS:
        raise ValueError(f"action_id {action_id} not in [0, {N_COMBAT_ACTIONS})")
    return COMBAT_ACTION_NAMES[action_id]


def action_vector(action_id: int) -> np.ndarray:
    """12-button multi-binary vector for stable-retro."""
    names = action_names(action_id)
    raw = idle_action() if not names else buttons(*names)
    return np.asarray(raw, dtype=np.int8)


def nearest_action_id(names: Sequence[str]) -> int:
    """Map free-form button names to the closest discrete table entry.

    Used to project the deterministic strategy onto the RL action space for
    eval / distillation baselines.
    """
    wanted = tuple(dict.fromkeys(n.upper() for n in names if n))
    if not wanted:
        return 0
    wanted_set = set(wanted)
    best_id = 0
    best_score = -1
    for index, candidate in enumerate(COMBAT_ACTION_NAMES):
        cand_set = set(candidate)
        # Prefer exact match, then max Jaccard with same face direction.
        if cand_set == wanted_set:
            return index
        inter = len(cand_set & wanted_set)
        union = len(cand_set | wanted_set) or 1
        score = inter / union
        # Slight preference for same horizontal intent.
        if ("LEFT" in wanted_set) == ("LEFT" in cand_set) and (
            "RIGHT" in wanted_set
        ) == ("RIGHT" in cand_set):
            score += 0.05
        if score > best_score:
            best_score = score
            best_id = index
    return best_id
