"""Mode-7 Neon Night Riders lane hold (drift / wait)."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.stages import KRANG_CHAR, NEON_MIN_FIGHT_Y, is_neon_highway


class NeonLaneTactics:
    """Hold the Mode-7 lane when no near-band enemy is in play.

    Near-band (Krang or y >= ``NEON_MIN_FIGHT_Y``) falls through to fight.
    Far-band living enemies still use this hold so the combat tree runs and
    ``CombatPositionStall`` can overlay a freeze.
    """

    def next(self, state: GameState) -> FrameAction | None:
        """Return a lane hold, or ``None`` off neon / when fight owns the frame."""
        if not is_neon_highway(state) or state.mode is not GameMode.PLAYING:
            return None
        if any(
            e.kind == KRANG_CHAR or e.y >= NEON_MIN_FIGHT_Y
            for e in state.living_enemies
        ):
            return None
        if state.player_x < 90:
            return FrameAction(action=buttons("RIGHT"), reason="neon_drift_right")
        if state.player_x > 180:
            return FrameAction(action=buttons("LEFT"), reason="neon_drift_left")
        return FrameAction(action=idle_action(), reason="neon_wait")
