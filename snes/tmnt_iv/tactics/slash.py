"""Slash (Prehistoric) tactics.

Raph Hard wiki (RetroMaggedon): do **not** shoulder-ram — Hard Slash
blocks. Jump **over** (B+toward), land just behind, short grounded combo,
hop away. Jump-kick (B+Y) only once airborne / to meet his hop. Spin
``0xEE`` and claw ``0x83``/``0x09`` still hop away. Production
spin_dodge_adx stays **52** (do not port probe KEEP 40).

One production path: the Raph jump-over behind-combo. Leo hybrid-whiplash
probes live in ``lab.slash_lab`` / HybridWhiplash, not this class.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import EnemyState, GameState
from tmnt_iv.grind_knobs import GrindKnobs, active_knobs

SLASH_CHAR = 0x50

_SPIN = 0xEE
_CLAW = frozenset({0x83, 0x09})
_PUNISH = frozenset({0x3E, 0x2E, 0x17, 0xB7, 0x23, 0x40})

# Same-Y jump-over is B-only (lab cross). B+Y is only for meeting an
# elevated Slash — same-frame Y+B on the ground is the HP-drain special.
_OVER_ADX_MIN = 24
_OVER_ADX_MAX = 72
_BEHIND_ADX_MIN = 12
_BEHIND_ADX_MAX = 40
_BODY_ADX = 20
_ELEV_DY = 16
_OVER_RISE_FRAMES = 8
_OVER_TOTAL_FRAMES = 22
_COMBO_FRAMES = 36
_HOP_FRAMES = 8


def _act(*names: str, reason: str) -> FrameAction:
    """Build a Slash action. Callers must never pass A."""
    return FrameAction(action=buttons(*names), reason=reason)


def _idle(reason: str) -> FrameAction:
    return FrameAction(action=idle_action(), reason=reason)


class SlashTactics:
    """Wiki jump-over behind-combo. One production FSM.

    RaphFullHardBoss5 (char 8) production spin_dodge_adx=52: **11,386f /
    478 dmg / 6 heals**. Probe KEEP spin=40 is **6,765f / 226 / 3** (5/5)
    but continuous dry-runs lose the sub-hour damage baseline via later
    RNG — keep 52 until a full-route re-tune. Rules
    (also ``docs/SLASH_VULN_MAP.md``):

    * Claw ``0x83``/``0x09`` → hop away when already close.
    * Spin ``0xEE`` close → hop away.
    * Punish ``0x3E``/``0x2E``/``0x17``/``0xB7``/``0x23``/``0x40`` or
      iframes → mash toward+Y (stay glued on hitstun).
    * Else: B-only jump-over → grounded combo → hop away. B+Y only
      to meet an elevated Slash. Never A.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget cycle phase / facing."""
        self._active = False
        self._phase = "approach"
        self._timer = 0
        self._punish_tick = 0
        self._facing = 0
        self._last_slash_x: int | None = None
        self._jump_toward = "RIGHT"
        self._jump_meet = False
        self._jump_start_y = 0

    def next(self, state: GameState) -> FrameAction | None:
        """Return a Slash-specific action, or ``None`` outside his fight."""
        if state.stage != 4 or int(state.extras.get("event", -1)) != 0x0A:
            if self._active:
                self.reset()
            return None

        slash = next(
            (enemy for enemy in state.living_enemies if enemy.kind == SLASH_CHAR),
            None,
        )
        if slash is None:
            if not self._active:
                return None
            return _idle("slash_wait")

        self._active = True
        self._update_facing(slash, state.player_x)
        knobs = active_knobs()
        dy = slash.y - state.player_y
        dx = slash.x - state.player_x
        adx = abs(dx)
        status = int(slash.animation)
        iframes = int(state.extras.get("iframes", 0))
        toward = "RIGHT" if dx > 0 else "LEFT"
        away = "LEFT" if dx >= 0 else "RIGHT"

        if slash.x > 256:
            if state.player_x > 180:
                return _act("LEFT", reason="slash_approach")
            if state.player_x < 90:
                return _act("RIGHT", reason="slash_approach")
            return _idle("slash_wait")

        if (
            status == _SPIN
            and iframes <= 0
            and adx < knobs.slash_spin_dodge_adx
            and abs(dy) <= knobs.slash_spin_dodge_ady
        ):
            self._phase = "approach"
            return _act("B", away, reason="slash_dodge")
        if (
            status in _CLAW
            and iframes <= 0
            and adx < knobs.slash_claw_dodge_adx
        ):
            self._phase = "approach"
            return _act("B", away, reason="slash_dodge")

        if iframes > 0 or status in _PUNISH:
            return self._punish(knobs, dy, adx, toward, away, iframes)

        return self._raph_cycle(state, slash, knobs, dy, adx, toward, away)

    def _update_facing(self, slash: EnemyState, player_x: int) -> None:
        if self._last_slash_x is not None and slash.x != self._last_slash_x:
            self._facing = 1 if slash.x > self._last_slash_x else -1
        elif self._facing == 0:
            self._facing = -1 if player_x < slash.x else 1
        self._last_slash_x = slash.x

    def _player_behind(self, slash: EnemyState, player_x: int) -> bool:
        if self._facing > 0:
            return player_x < slash.x
        if self._facing < 0:
            return player_x > slash.x
        return False

    def _punish(
        self,
        knobs: GrindKnobs,
        dy: int,
        adx: int,
        toward: str,
        away: str,
        iframes: int,
    ) -> FrameAction:
        if abs(dy) > 14 and adx < 40 and iframes <= 0:
            return _act("UP" if dy < 0 else "DOWN", reason="slash_align")
        if adx > knobs.slash_punish_approach_adx:
            return _act(toward, reason="slash_approach")
        if adx < knobs.slash_back_attack_adx:
            return _act(away, "Y", reason="slash_back_attack")
        self._punish_tick = (self._punish_tick + 1) % knobs.slash_punish_cycle
        if self._punish_tick < knobs.slash_punish_cross and adx < 40:
            return _act("B", toward, reason="slash_cross")
        return _act(toward, "Y", reason="slash_back_attack")

    def _raph_cycle(
        self,
        state: GameState,
        slash: EnemyState,
        knobs: GrindKnobs,
        dy: int,
        adx: int,
        toward: str,
        away: str,
    ) -> FrameAction:
        behind = self._player_behind(slash, state.player_x)

        if self._phase == "hop_away":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "approach"
            return _act("B", away, reason="slash_hop_away")

        if self._phase == "jump_over":
            return self._continue_jump(state, toward)

        if self._phase == "combo":
            return self._continue_combo(toward, away, adx)

        if abs(dy) >= _ELEV_DY:
            return self._start_jump(state, toward, meet=True)

        if behind and _BEHIND_ADX_MIN <= adx <= _BEHIND_ADX_MAX:
            return self._start_combo(toward)

        if not behind and adx < _BODY_ADX:
            return _act(away, reason="slash_space")

        if _OVER_ADX_MIN <= adx <= _OVER_ADX_MAX:
            return self._start_jump(state, toward, meet=False)

        if adx > knobs.slash_approach_band:
            return _act(toward, reason="slash_approach")
        return _idle("slash_bait")

    def _start_jump(
        self, state: GameState, toward: str, *, meet: bool
    ) -> FrameAction:
        self._phase = "jump_over"
        self._timer = _OVER_TOTAL_FRAMES - 1
        self._jump_toward = toward
        self._jump_meet = meet
        self._jump_start_y = state.player_y
        return _act("B", toward, reason="slash_jump_over")

    def _continue_jump(self, state: GameState, toward: str) -> FrameAction:
        self._timer -= 1
        side = self._jump_toward or toward
        if self._timer <= 0:
            return self._start_combo(toward)
        airborne = state.player_y < self._jump_start_y - 3
        risen = self._timer < (_OVER_TOTAL_FRAMES - _OVER_RISE_FRAMES)
        # Same-Y over: B only (lab-winner cross). Meet his hop: B+Y after
        # leaving the ground, or after rise frames when Y is frozen in tests.
        if self._jump_meet and (airborne or risen):
            return _act("B", "Y", side, reason="slash_jump_kick")
        return _act("B", side, reason="slash_jump_over")

    def _start_combo(self, toward: str) -> FrameAction:
        self._phase = "combo"
        self._timer = _COMBO_FRAMES - 1
        return _act(toward, "Y", reason="slash_back_attack")

    def _continue_combo(self, toward: str, away: str, adx: int) -> FrameAction:
        self._timer -= 1
        if self._timer <= 0:
            self._phase = "hop_away"
            self._timer = _HOP_FRAMES
            return _act("B", away, reason="slash_hop_away")
        if adx < 8:
            return _act(away, "Y", reason="slash_back_attack")
        return _act(toward, "Y", reason="slash_back_attack")
