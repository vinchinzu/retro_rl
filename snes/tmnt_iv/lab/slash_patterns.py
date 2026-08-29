"""Standalone Slash (char 0x50) pattern controllers for the pattern lab.

Reference controllers for the implementer to port into policy. Does **not**
import ``tmnt_iv.policy`` or the lab trial runner. ``ProductionSlash`` wraps
``SlashTactics`` so the lab can run production as a named pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.ram_state import GameState  # noqa: E402
from retro_harness.input_script import FrameAction  # noqa: E402
from tmnt_iv.grind_knobs import active_knobs
from tmnt_iv.tactics.slash import SLASH_CHAR, SlashTactics

_SPIN_STATUS = 0xEE
_PUNISH_STATUS = frozenset({0x3E, 0x2E, 0x17})

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _slash_enemy(state: GameState):
    return next(
        (e for e in state.living_enemies if e.kind == SLASH_CHAR),
        None,
    )


def _geom(state: GameState, slash) -> tuple[int, int, int, str, str, int, int]:
    """dx, dy, adx, toward, away, status, iframes."""
    dx = slash.x - state.player_x
    dy = slash.y - state.player_y
    adx = abs(dx)
    toward = "RIGHT" if dx > 0 else "LEFT"
    away = "LEFT" if dx >= 0 else "RIGHT"
    status = int(slash.animation)
    iframes = int(state.extras.get("iframes", 0))
    return dx, dy, adx, toward, away, status, iframes


def _offscreen_park(state: GameState, slash) -> FrameAction | None:
    """Hold mid-lane when Slash is parked off the right edge."""
    if slash.x <= 256:
        return None
    if state.player_x > 180:
        return FrameAction(action=buttons("LEFT"), reason="slash_park_left")
    if state.player_x < 90:
        return FrameAction(action=buttons("RIGHT"), reason="slash_park_right")
    return FrameAction(action=idle_action(), reason="slash_park_wait")


def _align_y(state: GameState, dy: int) -> FrameAction | None:
    if abs(dy) <= 10:
        return None
    return FrameAction(
        action=buttons("UP" if dy < 0 else "DOWN"),
        reason="slash_align",
    )


def _assert_no_a(action: list[int] | Any) -> None:
    # SNES action layout: B Y SELECT START UP DOWN LEFT RIGHT A X L R
    if int(action[8]) != 0:
        raise RuntimeError("forbidden A press in Slash pattern lab")


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class SlashPattern(ABC):
    """One named scripted controller."""

    name: str = "base"
    description: str = ""

    def reset(self) -> None:
        """Reset phase timers between trials."""

    @abstractmethod
    def next(self, state: GameState) -> FrameAction:
        """Return the next frame action (never A)."""


class ClassicThrash(SlashPattern):
    """Approach 42 → jump-cross B+toward 22f → toward+Y 36f. No dodge."""

    name = "classic_thrash"
    description = "approach 42, cross B+toward 22f, toward+Y 36f (no flee)"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = "approach"
        self._timer = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, _away, _status, _iframes = _geom(state, slash)
        align = _align_y(state, dy)
        if align is not None and self._phase == "approach":
            return align

        if self._phase == "approach":
            if adx > 42:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            self._phase = "cross"
            self._timer = 22
            return FrameAction(
                action=buttons("B", toward), reason="slash_cross"
            )

        if self._phase == "cross":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "attack"
                self._timer = 36
            return FrameAction(
                action=buttons("B", toward), reason="slash_cross"
            )

        self._timer -= 1
        if self._timer <= 0:
            self._phase = "approach"
        return FrameAction(
            action=buttons(toward, "Y"), reason="slash_back_attack"
        )


class ThrashFleeSpin(SlashPattern):
    """Classic thrash + flee only on status 0xEE when adx < 48."""

    name = "thrash_flee_spin"
    description = "thrash + flee 0xEE when adx<48"

    def __init__(self) -> None:
        self._thrash = ClassicThrash()
        self.reset()

    def reset(self) -> None:
        self._thrash.reset()

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        _dx, dy, adx, _toward, away, status, iframes = _geom(state, slash)
        if (
            status == _SPIN_STATUS
            and iframes <= 0
            and adx < 48
            and abs(dy) <= 20
        ):
            self._thrash.reset()
            return FrameAction(
                action=buttons("B", away), reason="slash_dodge"
            )
        return self._thrash.next(state)


class StatusAware(SlashPattern):
    """Flee 0xEE; mash toward+Y on punish statuses when close; else standoff 70."""

    name = "status_aware"
    description = "flee 0xEE; mash 0x3E/0x2E/0x17 close; else standoff 70px"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._timer = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, away, status, iframes = _geom(state, slash)

        if (
            status == _SPIN_STATUS
            and iframes <= 0
            and adx < 64
            and abs(dy) <= 24
        ):
            return FrameAction(
                action=buttons("B", away), reason="slash_dodge"
            )

        align = _align_y(state, dy)
        if align is not None and status not in _PUNISH_STATUS and iframes <= 0:
            return align

        if status in _PUNISH_STATUS or iframes > 0:
            if adx > 52:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            if adx < 10:
                return FrameAction(
                    action=buttons(away), reason="slash_space"
                )
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_punish"
            )

        # Neutral: hold ~70 px standoff and chip if he walks in.
        if adx > 78:
            return FrameAction(
                action=buttons(toward), reason="slash_close_in"
            )
        if adx < 62:
            return FrameAction(
                action=buttons(away), reason="slash_standoff"
            )
        # At band: light poke, no jump commit.
        self._timer = (self._timer + 1) % 12
        if self._timer < 3:
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_poke"
            )
        return FrameAction(action=buttons(toward), reason="slash_hold")


class IframeAggressive(SlashPattern):
    """When player iframes > 0 always mash; else thrash + flee spin."""

    name = "iframe_aggressive"
    description = "iframes>0 always mash toward+Y; else thrash+flee spin"

    def __init__(self) -> None:
        self._base = ThrashFleeSpin()
        self.reset()

    def reset(self) -> None:
        self._base.reset()

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, away, status, iframes = _geom(state, slash)
        if iframes > 0:
            align = _align_y(state, dy)
            if align is not None and adx < 80:
                # Prefer horizontal pressure while invuln; skip heavy align.
                pass
            if adx > 56:
                return FrameAction(
                    action=buttons(toward), reason="slash_iframe_chase"
                )
            if adx < 8:
                return FrameAction(
                    action=buttons(away, "Y"), reason="slash_iframe_mash"
                )
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_iframe_mash"
            )
        return self._base.next(state)


class JumpSlashPunish(SlashPattern):
    """Jump-slash (B+Y toward) only on punish statuses; otherwise kite."""

    name = "jump_slash_punish"
    description = "B+Y jump-slash only when status in {0x3E,0x2E,0x17}"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = "kite"
        self._timer = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, away, status, iframes = _geom(state, slash)

        if status == _SPIN_STATUS and adx < 56 and iframes <= 0:
            self._phase = "kite"
            return FrameAction(
                action=buttons("B", away), reason="slash_dodge"
            )

        if status in _PUNISH_STATUS or iframes > 0:
            if self._phase != "jump_slash":
                self._phase = "jump_slash"
                self._timer = 28
            self._timer -= 1
            if adx > 60:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            # Jump-slash commit.
            if self._timer > 14:
                return FrameAction(
                    action=buttons("B", "Y", toward),
                    reason="slash_jump_slash",
                )
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_follow"
            )

        # Kite at ~80 px, align Y lightly.
        self._phase = "kite"
        align = _align_y(state, dy)
        if align is not None:
            return align
        if adx < 70:
            return FrameAction(
                action=buttons(away), reason="slash_kite"
            )
        if adx > 90:
            return FrameAction(
                action=buttons(toward), reason="slash_close_in"
            )
        return FrameAction(action=idle_action(), reason="slash_wait_window")


class HybridWhiplash(SlashPattern):
    """Hybrid: iframe mash + punish stick + thrash cross only mid-HP + flee spin.

    Design notes (from pattern goals):
    - Natural iframes and punish statuses (0x3E/0x2E/0x17) are the real
      damage windows — stay glued and mash toward+Y.
    - Shell spin 0xEE: hop away when close without iframes.
    - Outside windows: short approach→cross→attack thrash, but keep a
      slightly wider approach (48) so we enter from behind more often.
    - Low boss HP (<48): shorter cross (16f) to finish faster.
    - Never A.
    """

    name = "hybrid_whiplash"
    description = (
        "iframe/punish mash; flee 0xEE; thrash cross@48/22→Y36; low-HP short cross"
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = "approach"
        self._timer = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, away, status, iframes = _geom(state, slash)

        # 1) Spin dodge
        if (
            status == _SPIN_STATUS
            and iframes <= 0
            and adx < 52
            and abs(dy) <= 22
        ):
            self._phase = "approach"
            return FrameAction(
                action=buttons("B", away), reason="slash_dodge"
            )

        # 2) Iframe / punish windows — full aggression
        if iframes > 0 or status in _PUNISH_STATUS:
            if abs(dy) > 14 and adx < 40 and iframes <= 0:
                return FrameAction(
                    action=buttons("UP" if dy < 0 else "DOWN"),
                    reason="slash_align",
                )
            if adx > 54:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            if adx < 8:
                return FrameAction(
                    action=buttons(away, "Y"), reason="slash_back_attack"
                )
            # Mix a brief jump-cross every ~40f of punish to re-flank.
            self._timer = (self._timer + 1) % 48
            if 0 <= self._timer < 10 and adx < 40:
                return FrameAction(
                    action=buttons("B", toward), reason="slash_cross"
                )
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_back_attack"
            )

        # 3) Y-align outside windows
        if abs(dy) > 10:
            self._phase = "approach"
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="slash_align",
            )

        # 4) Thrash cycle — combo length from grind knobs (lab adapter seam)
        knobs = active_knobs()
        approach_band = 48
        low = slash.health <= knobs.slash_low_hp
        cross_frames = (
            knobs.slash_cross_frames_low if low else knobs.slash_cross_frames
        )
        attack_frames = (
            knobs.slash_attack_frames_low if low else knobs.slash_attack_frames
        )

        if self._phase == "approach":
            if adx > approach_band:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            self._phase = "cross"
            self._timer = cross_frames
            return FrameAction(
                action=buttons("B", toward), reason="slash_cross"
            )

        if self._phase == "cross":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "attack"
                self._timer = attack_frames
            return FrameAction(
                action=buttons("B", toward), reason="slash_cross"
            )

        self._timer -= 1
        if self._timer <= 0:
            self._phase = "approach"
        return FrameAction(
            action=buttons(toward, "Y"), reason="slash_back_attack"
        )


class HybridStickAndMove(SlashPattern):
    """Alternate hybrid: standoff until punish/spin-end, then sticky thrash.

    Extra candidate — often lower chip than pure thrash if punish windows
    are real, worse if Slash never opens.
    """

    name = "hybrid_stick_move"
    description = (
        "standoff 64 until punish/iframe; then sticky thrash 30f; flee 0xEE"
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sticky = 0
        self._phase = "approach"
        self._timer = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        park = _offscreen_park(state, slash)
        if park is not None:
            return park
        dx, dy, adx, toward, away, status, iframes = _geom(state, slash)

        if (
            status == _SPIN_STATUS
            and iframes <= 0
            and adx < 50
            and abs(dy) <= 20
        ):
            self._sticky = 0
            self._phase = "approach"
            return FrameAction(
                action=buttons("B", away), reason="slash_dodge"
            )

        if iframes > 0 or status in _PUNISH_STATUS:
            self._sticky = 45

        if self._sticky > 0:
            self._sticky -= 1
            if abs(dy) > 12:
                return FrameAction(
                    action=buttons("UP" if dy < 0 else "DOWN"),
                    reason="slash_align",
                )
            if adx > 44:
                return FrameAction(
                    action=buttons(toward), reason="slash_approach"
                )
            if self._phase == "approach":
                self._phase = "cross"
                self._timer = 18
            if self._phase == "cross":
                self._timer -= 1
                if self._timer <= 0:
                    self._phase = "attack"
                    self._timer = 30
                return FrameAction(
                    action=buttons("B", toward), reason="slash_cross"
                )
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "approach"
            return FrameAction(
                action=buttons(toward, "Y"), reason="slash_back_attack"
            )

        # Passive standoff
        self._phase = "approach"
        if abs(dy) > 12:
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="slash_align",
            )
        if adx < 56:
            return FrameAction(action=buttons(away), reason="slash_standoff")
        if adx > 72:
            return FrameAction(
                action=buttons(toward), reason="slash_close_in"
            )
        return FrameAction(action=idle_action(), reason="slash_hold")


class VulnReactive(SlashPattern):
    """Status-reactive FSM (Raph Hard). Better chip than KEEP, slower.

    Emergency CLEAR 10,260f / 342 dmg / 5 heals vs KEEP 9,595 / 435 / 6.
    Not ported — KEEP still wins frames. No jump-over, no facing guess.
    """

    name = "vuln_reactive"
    description = (
        "status FSM: hop claw>=80/spin>=64; left 0x3E/0x40 [40,56]; "
        "glue 0x17/0x2E [14,22]; 0xB7 [16,24]; never A"
    )
    _CLAW = frozenset({0x83, 0x09})
    _OPENER = frozenset({0x3E, 0x40})
    _STUN = frozenset({0x17, 0x2E, 0x9F, 0x74, 0x42})
    _BIG = frozenset({0x23, 0xB7})

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._glue = 0

    def next(self, state: GameState) -> FrameAction:
        slash = _slash_enemy(state)
        if slash is None:
            return self._fa(reason="slash_wait")
        parked = _offscreen_park(state, slash)
        if parked is not None:
            return parked
        _dx, dy, adx, toward, away, status, iframes = _geom(state, slash)
        if status == 0x00:
            return self._lane(state)
        if status in self._CLAW and iframes <= 0:
            self._glue = 0
            if adx < 80:
                return self._hop(state, away)
            return self._fa(reason="slash_hold")
        if status == _SPIN_STATUS:
            self._glue = 0
            if iframes <= 0 and adx < 64:
                if adx < 48:
                    return self._hop(state, away)
                return self._fa(away, reason="slash_dodge")
            if abs(dy) > 10:
                return self._fa("UP" if dy < 0 else "DOWN", reason="slash_align")
            if adx > 80:
                return self._fa(toward, reason="slash_close_in")
            return self._fa(reason="slash_hold")
        if status in {0x17, 0x2E}:
            self._glue = 100
        if status in self._BIG:
            return self._band(state, dy, adx, toward, away, 16, 24)
        if status in self._STUN or self._glue > 0:
            if self._glue:
                self._glue -= 1
            return self._band(state, dy, adx, toward, away, 14, 22)
        if status in self._OPENER:
            return self._opener(state, slash, dy, adx, toward, away)
        if iframes > 0:
            return self._band(state, dy, adx, toward, away, 14, 22)
        return self._neutral(state, slash, dy, adx, toward, away)

    @staticmethod
    def _fa(*names: str, reason: str) -> FrameAction:
        act = buttons(*names) if names else idle_action()
        return FrameAction(action=act, reason=reason)

    def _hop(self, state: GameState, away: str) -> FrameAction:
        if away == "LEFT" and state.player_x < 24:
            away = "RIGHT"
        elif away == "RIGHT" and state.player_x > 232:
            away = "LEFT"
        return self._fa("B", away, reason="slash_dodge")

    def _lane(self, state: GameState) -> FrameAction:
        if state.player_x > 180:
            return self._fa("LEFT", reason="slash_park_left")
        if state.player_x < 90:
            return self._fa("RIGHT", reason="slash_park_right")
        return self._fa(reason="slash_park_wait")

    def _meet(self, state: GameState, toward: str, dy: int) -> FrameAction | None:
        if dy >= -16:
            return None
        if state.player_y < 164:
            return self._fa("B", "Y", toward, reason="slash_meet")
        return self._fa("B", toward, reason="slash_meet")

    def _band(
        self,
        state: GameState,
        dy: int,
        adx: int,
        toward: str,
        away: str,
        lo: int,
        hi: int,
    ) -> FrameAction:
        meet = self._meet(state, toward, dy)
        if meet is not None:
            return meet
        if abs(dy) > 10 and adx < 48:
            return self._fa("UP" if dy < 0 else "DOWN", reason="slash_align")
        if adx < lo:
            return self._fa(away, "Y", reason="slash_space")
        if adx > hi:
            extra = ("Y",) if adx <= hi + 28 else ()
            return self._fa(toward, *extra, reason="slash_approach")
        return self._fa(toward, "Y", reason="slash_punish")

    def _opener(
        self,
        state: GameState,
        slash,
        dy: int,
        adx: int,
        toward: str,
        away: str,
    ) -> FrameAction:
        meet = self._meet(state, toward, dy)
        if meet is not None:
            return meet
        if abs(dy) > 10:
            return self._fa("UP" if dy < 0 else "DOWN", reason="slash_align")
        on_left = state.player_x < slash.x
        if not on_left and slash.x >= 70 and adx > 56:
            return self._fa("LEFT", reason="slash_flank")
        if adx > 56:
            return self._fa(toward, reason="slash_approach")
        if adx < 40:
            return self._fa(away, "Y", reason="slash_space")
        return self._fa(toward, "Y", reason="slash_punish")

    def _neutral(
        self,
        state: GameState,
        slash,
        dy: int,
        adx: int,
        toward: str,
        away: str,
    ) -> FrameAction:
        if abs(dy) > 10:
            return self._fa("UP" if dy < 0 else "DOWN", reason="slash_align")
        on_left = state.player_x < slash.x
        if not on_left and slash.x >= 70:
            if adx < 40:
                return self._fa("B", "LEFT", reason="slash_flank")
            return self._fa("LEFT", reason="slash_flank")
        if adx < 64:
            return self._fa(away, reason="slash_standoff")
        if adx > 80:
            return self._fa(toward, reason="slash_close_in")
        return self._fa(reason="slash_hold")


class ProductionSlash(SlashPattern):
    """Lab adapter wrapping production ``SlashTactics``.

    ``SlashTactics.next`` returns ``None`` outside the fight; the lab
    contract is always a ``FrameAction``, so ``None`` maps to idle
    ``slash_wait`` (same as ClassicThrash when no Slash entity).
    """

    name = "production"
    description = "production SlashTactics (wiki jump-over behind-combo)"

    def __init__(self) -> None:
        self._tactics = SlashTactics()
        self.reset()

    def reset(self) -> None:
        self._tactics.reset()

    def next(self, state: GameState) -> FrameAction:
        action = self._tactics.next(state)
        if action is None:
            return FrameAction(action=idle_action(), reason="slash_wait")
        return action
