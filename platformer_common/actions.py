"""Reduced action space for SNES platformer optimization.

SNES env button order: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
Indices:                 0  1    2      3     4    5     6     7    8  9  10 11

The DEFAULT_PLATFORMER_ACTIONS table works for most SNES side-scrollers
(DKC, Super Mario World, etc.). Games can override via LevelConfig.action_table.
"""

from retro_harness.controls import (
    SNES_B,
    SNES_Y,
    SNES_RIGHT,
    SNES_LEFT,
    SNES_DOWN,
    SNES_UP,
    SNES_A,
)

NUM_BUTTONS = 12


def _make(*, buttons: list[int]) -> list[int]:
    """Create a 12-element action array with given buttons pressed."""
    action = [0] * NUM_BUTTONS
    for b in buttons:
        action[b] = 1
    return action


# Default reduced action set for SNES platforming
DEFAULT_PLATFORMER_ACTIONS = [
    _make(buttons=[]),                              # 0: NOTHING
    _make(buttons=[SNES_RIGHT]),                    # 1: RIGHT
    _make(buttons=[SNES_RIGHT, SNES_Y]),            # 2: RIGHT + Y (run right)
    _make(buttons=[SNES_RIGHT, SNES_Y, SNES_B]),   # 3: RIGHT + Y + B (run + jump)
    _make(buttons=[SNES_RIGHT, SNES_B]),            # 4: RIGHT + B (walk + jump)
    _make(buttons=[SNES_B]),                        # 5: JUMP
    _make(buttons=[SNES_LEFT]),                     # 6: LEFT
    _make(buttons=[SNES_LEFT, SNES_Y]),             # 7: LEFT + Y (run left)
    _make(buttons=[SNES_LEFT, SNES_Y, SNES_B]),    # 8: LEFT + Y + B (run left + jump)
    _make(buttons=[SNES_LEFT, SNES_B]),             # 9: LEFT + B (walk left + jump)
    _make(buttons=[SNES_DOWN]),                     # 10: DOWN (duck/dismount)
    _make(buttons=[SNES_A]),                        # 11: A (roll/special)
    _make(buttons=[SNES_RIGHT, SNES_A]),            # 12: RIGHT + A (roll right)
    _make(buttons=[SNES_UP]),                       # 13: UP (enter door)
]


def action_index_to_buttons(idx: int, action_table: list[list[int]] | None = None) -> list[int]:
    """Convert action index to 12-element button array."""
    table = action_table or DEFAULT_PLATFORMER_ACTIONS
    if 0 <= idx < len(table):
        return list(table[idx])
    return [0] * NUM_BUTTONS


def buttons_to_action_index(buttons: list[int], action_table: list[list[int]] | None = None) -> int:
    """Find the closest action index for a raw 12-element button array."""
    table = action_table or DEFAULT_PLATFORMER_ACTIONS
    best_idx = 0
    best_dist = float("inf")
    for idx, ref in enumerate(table):
        dist = sum(abs(a - b) for a, b in zip(buttons, ref))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
        if dist == 0:
            break
    return best_idx
