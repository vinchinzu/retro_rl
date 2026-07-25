"""Scene-segment policy: move Waldo cursor toward a target coordinate."""

from __future__ import annotations

from snes_oneshot.cursor import (
    CursorPose,
    CursorTarget,
    at_target,
    plan_cursor_path,
    step_toward_target,
)
from snes_oneshot.game_state import GameMode, GameState

__all__ = [
    "CursorPose",
    "CursorTarget",
    "at_target",
    "cursor_from_state",
    "plan_cursor_path",
    "playing_state",
    "step_toward_target",
]


def cursor_from_state(state: GameState) -> CursorPose:
    """Read cursor coords from player_x/player_y (Waldo adapter convention)."""
    return CursorPose(x=state.player_x, y=state.player_y)


def playing_state(
    *,
    frame: int,
    cursor_x: int,
    cursor_y: int,
    scene_id: int = 0,
) -> GameState:
    """Build a PLAYING GameState with Waldo cursor in player fields."""
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        player_x=cursor_x,
        player_y=cursor_y,
        room=scene_id,
        stage=scene_id,
        extras={"scene_id": scene_id},
    )
