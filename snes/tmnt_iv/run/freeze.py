"""Frozen-X abort for the continuous full Hard run.

Enemyless frozen X this long is an infinite dumpster/rail loop, not a
finish. Pin dumpsters recover in hundreds of frames; Diag rail skip
recovered after ~600f. The 180k–230k encode stall was this hole.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from retro_harness.env import save_state
from retro_harness.ram_state import GameState
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR

FREEZE_ABORT_FRAMES = 12_000


class FreezeWatch:
    """Count consecutive enemyless frames at the same X and abort at 12_000."""

    def __init__(self, abort_frames: int = FREEZE_ABORT_FRAMES) -> None:
        self.abort_frames = abort_frames
        self._x = -1
        self._stage = -1
        self.frames = 0

    def tick(
        self,
        *,
        armed: bool,
        state: GameState,
        frame: int,
        reason: str,
        damage: int,
        obs: np.ndarray,
        env: Any,
    ) -> None:
        """Advance the freeze counter; dump PNG+state and raise on abort."""
        if not armed:
            self._x = -1
            self._stage = -1
            self.frames = 0
            return
        if state.player_x == self._x and state.stage == self._stage:
            self.frames += 1
        else:
            self._x = state.player_x
            self._stage = state.stage
            self.frames = 0
        if self.frames in {2_000, 5_000} or (
            self.frames >= 2_000 and self.frames % 5_000 == 0
        ):
            print(
                f"FREEZE {self.frames}f  frame={frame} "
                f"stage={state.stage} "
                f"p=({state.player_x},{state.player_y}) "
                f"cam={state.camera_x} reason={reason} "
                f"dmg={damage}",
                flush=True,
            )
        if self.frames >= self.abort_frames:
            shot = RECORDINGS_DIR / (
                f"scratch_freeze_s{state.stage}_"
                f"x{state.player_x}_f{frame}.png"
            )
            Image.fromarray(np.asarray(obs, dtype=np.uint8)).save(shot)
            save_state(
                env,
                GAME_DIR,
                GAME,
                f"ScratchFreeze_s{state.stage}_x{state.player_x}",
            )
            raise RuntimeError(
                f"frozen X for {self.frames}f at frame {frame}: "
                f"stage={state.stage} "
                f"p=({state.player_x},{state.player_y}) "
                f"cam={state.camera_x} reason={reason} "
                f"dmg={damage} shot={shot}"
            )
