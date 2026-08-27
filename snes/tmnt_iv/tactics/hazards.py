"""Stage-1 wrecking-ball dodge (offline in tick) and Sewer spike jump."""

from __future__ import annotations

from retro_harness.actions import buttons
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.stages import is_sewer

_SEWER_SPIKE_CHARS: frozenset[int] = frozenset({0x1C, 0x2C})
# A/B: adx 56 → 1 residual 16-dmg hit (wider bands added more spikes).
_SEWER_SPIKE_ADX = 56
# Wrecking-ball / ceiling hazard band (Stage 1 Big Apple).
# Ceiling 0x36 hitbox is wide (~30px). Bands are tight enough to dodge −24
# without hijacking combat for thousands of frames (wide bands caused Clean
# deaths by starving DPS / pizza routing).
_HAZARD_DODGE_ADX = 48
_HAZARD_CEILING_CLEAR_ADX = 40


class SewerSpikeAvoid:
    """Dodge Sewer Surfin' hanging spike props (char 0x1C / 0x2C).

    These are HP-0 props exposed via ``extras["hazards"]`` (−16). Surfboard
    Y is lane depth (py stays ~192 through B), but A/B still found
    jump-right when adx ≤ 56 cuts empty-band spikes 3→1 (timing/hitbox).
    LEFT thrash regressed to 4 spikes. Mid-pack: only when the spike is
    as close as the nearest Foot.
    """

    def next(self, state: GameState) -> FrameAction | None:
        """Return jump-right past a near spike column, or ``None``."""
        if state.mode is not GameMode.PLAYING or not is_sewer(state):
            return None
        # Rat King: keep the long poke; spikes are a wave problem.
        if state.boss_active:
            return None
        hazards = state.extras.get("hazards") or ()
        spikes = [
            (int(h[0]), int(h[1]), int(h[2]))
            for h in hazards
            if int(h[2]) in _SEWER_SPIKE_CHARS
        ]
        if not spikes:
            return None
        hx, _hy, _ch = min(spikes, key=lambda t: abs(t[0] - state.player_x))
        adx = abs(hx - state.player_x)
        if adx > _SEWER_SPIKE_ADX:
            return None
        if state.living_enemies:
            nearest = min(
                state.living_enemies,
                key=lambda e: abs(e.x - state.player_x) + abs(e.y - state.player_y),
            )
            enemy_adx = abs(nearest.x - state.player_x)
            if enemy_adx + 8 < adx:
                return None
        return FrameAction(action=buttons("B", "RIGHT"), reason="sewer_spike_jump")


class HazardAvoid:
    """Avoid Stage 1 wrecking-ball / ceiling chip (path-RNG stable).

    ``extras["hazards"]`` exposes char ``0x32`` / ``0x36`` props (HP 0).
    Ceiling ``0x36`` is −24; ground swing ``0x32`` is −16.

    Jump-right past the column when healthy (progress + dumpster). When HP
    is low, **wait outside** the column on the open side instead of leaping
    through the ball again (that is the power-on Clean chain-death).
    """

    def __init__(self) -> None:
        self._close_frames = 0
        self._prev_hx: int | None = None
        self._vx = 0

    def next(self, state: GameState) -> FrameAction | None:
        """Return a jump-past / micro-dodge, or ``None`` when clear."""
        if state.mode is not GameMode.PLAYING or state.stage != 0:
            self._close_frames = 0
            self._prev_hx = None
            self._vx = 0
            return None
        hazards = state.extras.get("hazards") or ()
        # Stage-1 only — ignore sewer 0x1C/0x2C (handled by SewerSpikeAvoid).
        hazards = tuple(
            h for h in hazards if int(h[2]) in {0x32, 0x36}
        )
        if not hazards:
            self._close_frames = 0
            self._prev_hx = None
            return None
        hx, hy, char = min(
            ((int(h[0]), int(h[1]), int(h[2])) for h in hazards),
            key=lambda t: abs(t[0] - state.player_x),
        )
        adx = abs(hx - state.player_x)
        if self._prev_hx is not None:
            self._vx = hx - self._prev_hx
        self._prev_hx = hx
        if adx > _HAZARD_DODGE_ADX:
            self._close_frames = 0
            return None
        self._close_frames += 1
        overhead = hy + 24 < state.player_y or char == 0x36
        closing = (self._vx > 0 and hx < state.player_x) or (
            self._vx < 0 and hx > state.player_x
        )
        clear = _HAZARD_CEILING_CLEAR_ADX + (6 if closing else 0)

        if overhead and adx <= clear:
            # Always open right when possible — proven Clean Stage1 clear path.
            # (HP-conditional wait/pocket variants regressed multi-entry suite.)
            if state.player_x < 200:
                return FrameAction(
                    action=buttons("B", "RIGHT"),
                    reason="hazard_jump",
                )
            return FrameAction(
                action=buttons("B", "DOWN", "LEFT"),
                reason="hazard_dodge",
            )

        if not state.living_enemies and adx <= _HAZARD_DODGE_ADX:
            return FrameAction(
                action=buttons("B", "RIGHT"),
                reason="hazard_jump",
            )

        if not overhead and adx <= 28 and self._close_frames < 20:
            side = (
                "LEFT"
                if state.player_x <= hx and state.player_x > 40
                else "RIGHT"
            )
            return FrameAction(
                action=buttons("B", side),
                reason="hazard_dodge",
            )
        return None
