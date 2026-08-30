"""Alleycat pack spacing: left shoulder, no jumper Y-chase, no 0x76 grab walk-in.

REACH (2026-08-27): full ``Stage2`` now arrives at the post-pizza ``0x5E``
window at HP 80 (was life_loss ~4.4k on 0x68/0x76). Residual: 0x5E still
lands 24-dmg when Leo closes from the right into a left clump
(~progress 21.4k / x≈164).

Already burned (do not re-open): pack jump-hop, 0x5E jump-slash, global
min_range, LEFT-forever on overlap, dense Y on 0x60 sandwich (wave-1 freeze),
adx≤80 hold, LEFT+Y close. No B, no A.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.combat import AttackCadence
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import EnemyState, GameMode, GameState

# Street Foot + Alleycat grabber. Metalhead (0x46) is excluded on purpose.
_PACK_CHARS: frozenset[int] = frozenset({0x5E, 0x60, 0x62, 0x68, 0x76})
_GRABBER_CHARS: frozenset[int] = frozenset({0x76})
_ALLEY_STAGE = 1
_ATTACK_RANGE = 65
# Clearly on the right of the target — not the overlap band. Point-blank
# LEFT is the rejected AlleycatPackSpace / min_range>0 death (frozen x,
# never pokes). Overlap must plant-poke (min_range 0 KEEP).
_WRONG_SIDE_SLACK = 8
_OVERLAP_ADX = 16
_SANDWICH_ADX = 12
_GRAB_CLOSE = 40
_JUMPER_ELEV_MAX = 18
_Y_BAND = 24
_PACK_HOLD_X = 176
_RIGHT_WALL_X = 200
# 0x5E jump-kick connects from ~27px on the right (REACH 24-dmg at x=199).
# Stay under the 36px left-shoulder standoff so a normal poke does not flee.
_KICK_RIGHT_ADX = 32


class AlleycatPackTactics:
    """Stage-1-only overrides for pack pile-ons. Returns ``None`` otherwise."""

    def __init__(self) -> None:
        self._cadence = AttackCadence(hold_frames=1, gap_frames=2)

    def reset(self) -> None:
        """Restart the one-frame poke cadence."""
        self._cadence.reset()

    def next(self, state: GameState) -> FrameAction | None:
        """Return a pack action, or ``None`` to fall through to the combat tree."""
        if state.mode is not GameMode.PLAYING:
            return None
        if state.stage != _ALLEY_STAGE or state.boss_active or state.health <= 0:
            return None
        living = [e for e in state.living_enemies if e.kind in _PACK_CHARS]
        if not living:
            return None

        grab = self._grabber_action(state, living)
        if grab is not None:
            return grab

        target = _select_pack_target(state, living)
        if target is None:
            return None

        # Nearest body, not a farther-left slot — releft-through-overlap
        # is the 0x5E 24-dmg pile (player walks through a kick to chase x=78).
        has_5e = any(e.kind == 0x5E for e in living)
        if any(_overlapping(state, e) for e in living):
            return self._poke(dense=has_5e)
        # Standing in a 0x5E jump-kick from the right is the 24-dmg hit.
        # Walk left out of it before sandwich-plant (screenshot x=199 / 226).
        if has_5e and any(_right_kicker(state, e) for e in living):
            return FrameAction(action=buttons("LEFT"), reason="alley_releft")
        if _sandwiched(state, living) and any(
            abs(e.x - state.player_x) <= _ATTACK_RANGE for e in living
        ):
            return self._poke(dense=has_5e)

        in_range = _in_slash_range(state, target)
        wrong_side = state.player_x > target.x + _WRONG_SIDE_SLACK
        # 0x5E in poke range: plant. Relieft walks through their 24-dmg
        # kick (REACH x=160). Other Foot still releft — dense-plant froze
        # the opening 0x60 wave at x=113.
        if wrong_side and has_5e and in_range and state.player_x < _RIGHT_WALL_X:
            return self._poke(dense=True)
        # Left 0x5E clump, player on the right shoulder (REACH x=164 /
        # enemies 69–96): generic releft walks through the kick. Exit
        # right until the hold line, then plant — never LEFT through.
        if has_5e and _left_5e_clump(state, living):
            if state.player_x < _PACK_HOLD_X:
                return FrameAction(
                    action=buttons("RIGHT"), reason="alley_right_exit"
                )
            return self._poke(dense=True)
        if wrong_side:
            return FrameAction(action=buttons("LEFT"), reason="alley_releft")

        # In-range Y-lock: do not align_up/down into a kick (ady=12 → 24 dmg).
        dy = abs(target.y - state.player_y)
        if (
            _in_slash_range(state, target)
            and 6 < dy <= _JUMPER_ELEV_MAX
        ):
            return self._poke()
        if _in_slash_range(state, target) and _jumper_above(state, target):
            return self._poke()

        # Do not walk righter into a cluster already in poke range.
        ahead = [
            e
            for e in living
            if e.x > state.player_x + _SANDWICH_ADX
            and abs(e.y - state.player_y) <= _Y_BAND
        ]
        if (
            len(ahead) >= 2
            and state.player_x >= _PACK_HOLD_X
            and _in_slash_range(state, target)
        ):
            return self._poke()
        return None

    def _grabber_action(
        self,
        state: GameState,
        living: list[EnemyState],
    ) -> FrameAction | None:
        grabbers = [e for e in living if e.kind in _GRABBER_CHARS]
        if not grabbers:
            return None
        grabber = min(
            grabbers,
            key=lambda e: abs(e.x - state.player_x) + abs(e.y - state.player_y),
        )
        if abs(grabber.y - state.player_y) > _Y_BAND:
            return None
        dx = grabber.x - state.player_x
        if state.player_x > grabber.x + _WRONG_SIDE_SLACK:
            return FrameAction(action=buttons("LEFT"), reason="alley_grab_space")
        if abs(dx) < _OVERLAP_ADX or 0 < dx <= _GRAB_CLOSE:
            return self._poke(dense=True)
        return None

    def _poke(self, *, dense: bool = False) -> FrameAction:
        if dense:
            return FrameAction(action=buttons("Y"), reason="alley_poke")
        action = self._cadence.next_attack(button="Y")
        if action.reason == "attack":
            return FrameAction(action=action.action, reason="alley_poke")
        return FrameAction(action=idle_action(), reason="alley_gap")


def _select_pack_target(
    state: GameState,
    living: list[EnemyState],
) -> EnemyState | None:
    if not living:
        return None
    return min(
        living,
        key=lambda e: abs(e.x - state.player_x) + abs(e.y - state.player_y),
    )


def _overlapping(state: GameState, enemy: EnemyState) -> bool:
    return (
        abs(enemy.x - state.player_x) < _OVERLAP_ADX
        and abs(enemy.y - state.player_y) <= _Y_BAND
    )


def _sandwiched(state: GameState, living: list[EnemyState]) -> bool:
    left_any = any(e.x < state.player_x - _SANDWICH_ADX for e in living)
    right_any = any(e.x > state.player_x + _SANDWICH_ADX for e in living)
    return left_any and right_any


def _in_slash_range(state: GameState, target: EnemyState) -> bool:
    return abs(target.x - state.player_x) <= _ATTACK_RANGE


def _jumper_above(state: GameState, target: EnemyState) -> bool:
    elev = state.player_y - target.y
    return 0 < elev <= _JUMPER_ELEV_MAX


def _right_kicker(state: GameState, enemy: EnemyState) -> bool:
    if enemy.kind != 0x5E:
        return False
    dx = enemy.x - state.player_x
    return 0 < dx <= _KICK_RIGHT_ADX and abs(enemy.y - state.player_y) <= _Y_BAND


def _left_5e_clump(state: GameState, living: list[EnemyState]) -> bool:
    """True when every 0x5E is on the left and none is a right-side kicker."""
    pack = [e for e in living if e.kind == 0x5E]
    if not pack:
        return False
    if any(_right_kicker(state, e) for e in pack):
        return False
    return all(e.x <= state.player_x for e in pack)
