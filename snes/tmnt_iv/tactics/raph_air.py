"""Raphael Starbase jump-kick: B+Y flying kick with a 3-frame steer gap.

Never A. Never grounded Y+B as a power attack. ``buttons("B", "Y", *dir)``
is the flying kick.
"""

from __future__ import annotations

from retro_harness.actions import buttons
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameState
from tmnt_iv.stages import (
    RAPH_CHAR,
    RAPH_STARBASE_CLOSE_CHARS,
    RAPH_STARBASE_GROUND_CHARS,
    STARBASE_WAVES,
)

_ADX = 80
_ADY = 36
_Y_STEER = 8
# Steer-without-hit period. %4 KEEP; %3 timeout; %2 jump-lock. Never hold B+Y.
_JUMP_PERIOD = 4


def raph_starbase_jump_action(state: GameState) -> FrameAction | None:
    """Close-range Starbase jump-kick, or None to fall through."""
    if (
        state.stage != STARBASE_WAVES
        or state.boss_active
        or int(state.extras.get("char_id", -1)) != RAPH_CHAR
    ):
        return None
    living = state.living_enemies
    if any(enemy.kind in RAPH_STARBASE_GROUND_CHARS for enemy in living):
        return None
    targets = [enemy for enemy in living if enemy.kind in RAPH_STARBASE_CLOSE_CHARS]
    if not targets:
        return None
    target = min(
        targets,
        key=lambda enemy: abs(enemy.x - state.player_x)
        + abs(enemy.y - state.player_y),
    )
    dx = target.x - state.player_x
    dy = target.y - state.player_y
    if abs(dx) > _ADX or abs(dy) > _ADY:
        return None
    toward = "RIGHT" if dx > 0 else "LEFT"
    steering = [toward]
    if abs(dy) > _Y_STEER:
        steering.append("DOWN" if dy > 0 else "UP")
    if state.frame % _JUMP_PERIOD:
        return FrameAction(
            action=buttons(*steering),
            reason="raph_starbase_close_gap",
        )
    return FrameAction(
        action=buttons("B", "Y", *steering),
        reason="raph_starbase_jump",
    )
