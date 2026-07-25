"""Slash (char 0x50) attack-pattern lab from FullHardBoss5.

Standalone reference controllers for the implementer to port into policy.
Does **not** import or mutate ``tmnt_iv.policy`` — keep that free for the
production agent.

Examples::

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.slash_pattern_lab --list

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.slash_pattern_lab \\
        --pattern classic_thrash --heal emergency

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.slash_pattern_lab --all --heal full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from snes_oneshot.actions import buttons, idle_action  # noqa: E402
from snes_oneshot.game_state import GameState  # noqa: E402
from snes_oneshot.primitives import FrameAction  # noqa: E402
from snes_oneshot.segment_runner import configure_headless  # noqa: E402
from retro_harness.env import make_env  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

_SLASH_CHAR = 0x50
_SPIN_STATUS = 0xEE
_PUNISH_STATUS = frozenset({0x3E, 0x2E, 0x17})
_DEFAULT_STATE = "FullHardBoss5"
_MAX_FRAMES_DEFAULT = 35_000
_EMERGENCY_HP_THRESHOLD = 16
_EMERGENCY_HP_RESTORE = 80
_FULL_HEAL_HP = 96


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _slash_enemy(state: GameState):
    return next(
        (e for e in state.living_enemies if e.kind == _SLASH_CHAR),
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

        # 4) Thrash cycle
        approach_band = 48
        cross_frames = 16 if slash.health <= 48 else 22
        attack_frames = 40 if slash.health <= 48 else 36

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


PATTERNS: dict[str, Callable[[], SlashPattern]] = {
    ClassicThrash.name: ClassicThrash,
    ThrashFleeSpin.name: ThrashFleeSpin,
    StatusAware.name: StatusAware,
    IframeAggressive.name: IframeAggressive,
    JumpSlashPunish.name: JumpSlashPunish,
    HybridWhiplash.name: HybridWhiplash,
    HybridStickAndMove.name: HybridStickAndMove,
}


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    pattern: str
    heal_mode: str
    state: str
    outcome: str
    frames: int
    boss_hp_start: int
    boss_hp_end: int
    dmg_taken: int
    heals: int
    min_hp: int | None
    end_hp: int
    end_stage: int
    event: str
    elapsed_s: float
    top_reasons: list[tuple[str, int]] = field(default_factory=list)
    description: str = ""

    @property
    def boss_damage(self) -> int:
        return max(0, self.boss_hp_start - self.boss_hp_end)

    @property
    def dps(self) -> float:
        if self.frames <= 0:
            return 0.0
        return self.boss_damage * 60.0 / self.frames

    @property
    def dmg_per_boss_hp(self) -> float:
        bd = self.boss_damage
        if bd <= 0:
            return float("inf") if self.dmg_taken else 0.0
        return self.dmg_taken / bd


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def run_trial(
    *,
    pattern: SlashPattern,
    state_name: str = _DEFAULT_STATE,
    max_frames: int = _MAX_FRAMES_DEFAULT,
    heal_mode: str = "emergency",
    stop_stage_gt: int = 4,
) -> TrialResult:
    """Run one controller from ``state_name`` and collect metrics.

    heal_mode:
      - ``none``: no HP writes (pure survival stress)
      - ``emergency``: restore to 80 when HP <= 16 (production-like)
      - ``full``: restore to 96 on any damage (pure DPS ranking)
    """
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    pattern.reset()
    _reset(env)
    start = parse_game_state(env.get_ram(), frame=0)
    prev_hp = start.health if 0 < start.health <= 0x60 else None
    prev_lives = start.lives
    damage = 0
    heals = 0
    min_hp = prev_hp
    reasons: dict[str, int] = {}
    boss_hp_start = int(start.extras.get("boss_hp", 0))
    # Also track Slash entity HP directly (boss_hp extras may drop when
    # status filters flicker).
    slash0 = _slash_enemy(start)
    if slash0 is not None:
        boss_hp_start = max(boss_hp_start, slash0.health)
    final = start
    outcome = "timeout"
    t0 = time.perf_counter()
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            # Natural damage from HP drops (before any heal write).
            if (
                prev_hp is not None
                and 0 <= state.health <= 0x60
                and prev_hp <= 0x60
                and state.health < prev_hp
            ):
                damage += prev_hp - max(0, state.health)
            if state.health == 0 and prev_hp is not None and prev_hp > 0:
                damage += prev_hp

            if 0 < state.health <= 0x60:
                if min_hp is None or state.health < min_hp:
                    min_hp = state.health

            # Heal assists.
            if heal_mode == "full" and 0 < state.health <= 0x60:
                if state.health < _FULL_HEAL_HP:
                    env.set_value("player_hp", _FULL_HEAL_HP)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
            elif heal_mode == "emergency":
                if state.health == 0 or (
                    0 < state.health <= _EMERGENCY_HP_THRESHOLD
                ):
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state

            if 0 < state.health <= 0x60:
                prev_hp = state.health
            elif state.health == 0:
                prev_hp = 0

            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives

            if state.stage > stop_stage_gt:
                outcome = "stage_advance"
                break

            slash = _slash_enemy(state)
            if (
                start.boss_active
                and not state.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19}
            ) or (
                slash is None
                and start.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19, 0x04}
            ):
                outcome = "boss_down"
            if outcome == "boss_down" and frame % 30 == 0:
                if state.stage > start.stage or int(
                    state.extras.get("event", 0)
                ) in {0x19, 0x04, 0x0B}:
                    if slash is None or slash.health <= 0:
                        outcome = "cleared"
                        break

            if state.mode.name in {"CONTINUE", "GAME_OVER"}:
                if heal_mode != "none":
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    prev_hp = (
                        state.health if 0 < state.health <= 0x60 else prev_hp
                    )
                else:
                    outcome = "life_loss"
                    break

            fa = pattern.next(state)
            action = fa.action if fa is not None else idle_action()
            reason = fa.reason if fa is not None else "idle"
            reasons[reason] = reasons.get(reason, 0) + 1
            _assert_no_a(action)
            env.step(action)
        else:
            if outcome not in {"cleared", "boss_down"}:
                outcome = "timeout"
    finally:
        env.close()

    elapsed = time.perf_counter() - t0
    slash_f = _slash_enemy(final)
    boss_hp_end = (
        slash_f.health
        if slash_f is not None
        else int(final.extras.get("boss_hp", 0))
    )
    if outcome in {"cleared", "stage_advance", "boss_down"} and slash_f is None:
        boss_hp_end = 0

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:10]
    return TrialResult(
        pattern=pattern.name,
        heal_mode=heal_mode,
        state=state_name,
        outcome=outcome,
        frames=final.frame,
        boss_hp_start=boss_hp_start,
        boss_hp_end=boss_hp_end,
        dmg_taken=damage,
        heals=heals,
        min_hp=min_hp,
        end_hp=final.health,
        end_stage=final.stage,
        event=hex(int(final.extras.get("event", -1))),
        elapsed_s=round(elapsed, 2),
        top_reasons=top,
        description=pattern.description,
    )


def _score(r: TrialResult) -> tuple:
    """Rank key: clear first, then less damage, fewer frames, fewer heals."""
    cleared = r.outcome in {"cleared", "stage_advance", "boss_down"}
    # Prefer full clear / stage advance over mere boss_down
    tier = {
        "stage_advance": 3,
        "cleared": 3,
        "boss_down": 2,
        "timeout": 1,
        "life_loss": 0,
        "forbidden_a": -1,
    }.get(r.outcome, 0)
    return (
        tier,
        r.boss_damage,  # more boss damage better when not cleared
        -r.dmg_taken,
        -r.frames,
        -r.heals,
    )


def _print_result(r: TrialResult) -> None:
    print(
        f"[{r.pattern}/{r.heal_mode}] outcome={r.outcome} "
        f"frames={r.frames} boss={r.boss_hp_start}->{r.boss_hp_end} "
        f"dmg_taken={r.dmg_taken} heals={r.heals} "
        f"dps={r.dps:.2f} dmg/bossHP={r.dmg_per_boss_hp:.2f} "
        f"min_hp={r.min_hp} event={r.event} ({r.elapsed_s}s)"
    )
    if r.top_reasons:
        brief = ", ".join(f"{k}:{v}" for k, v in r.top_reasons[:5])
        print(f"  reasons: {brief}")


def _markdown_table(results: list[TrialResult]) -> str:
    lines = [
        "| pattern | heal | frames | boss_hp | dmg_taken | heals | outcome | dps | dmg/bhp |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in sorted(results, key=_score, reverse=True):
        boss = f"{r.boss_hp_start}->{r.boss_hp_end}"
        if r.outcome in {"cleared", "stage_advance"} and r.boss_hp_end == 0:
            boss = f"{r.boss_hp_start}->0 (clear)"
        lines.append(
            f"| `{r.pattern}` | {r.heal_mode} | {r.frames} | {boss} | "
            f"{r.dmg_taken} | {r.heals} | {r.outcome} | {r.dps:.2f} | "
            f"{r.dmg_per_boss_hp:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="pattern name (repeatable). default: all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every registered pattern",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list patterns and exit",
    )
    parser.add_argument(
        "--heal",
        choices=("none", "emergency", "full"),
        action="append",
        dest="heals",
        default=None,
        help="heal mode (repeatable). default: emergency + full",
    )
    parser.add_argument("--state", default=_DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=_MAX_FRAMES_DEFAULT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write full JSON results",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="optional path to write markdown table only",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in PATTERNS.items():
            inst = cls()
            print(f"  {name:22s}  {inst.description}")
        return 0

    names = args.patterns
    if args.all or not names:
        names = list(PATTERNS.keys())
    for n in names:
        if n not in PATTERNS:
            print(f"unknown pattern: {n}", file=sys.stderr)
            print(f"choose from: {', '.join(PATTERNS)}", file=sys.stderr)
            return 2

    heals = args.heals or ["emergency", "full"]
    results: list[TrialResult] = []
    for heal in heals:
        for name in names:
            print(f"\n=== RUN {name} heal={heal} ===", flush=True)
            ctrl = PATTERNS[name]()
            result = run_trial(
                pattern=ctrl,
                state_name=args.state,
                max_frames=args.max_frames,
                heal_mode=heal,
            )
            _print_result(result)
            results.append(result)

    print("\n======== SUMMARY ========")
    print(_markdown_table(results))

    ranked = sorted(results, key=_score, reverse=True)
    if ranked:
        best = ranked[0]
        print(
            f"\nWINNER: {best.pattern} ({best.heal_mode}) "
            f"outcome={best.outcome} frames={best.frames} "
            f"dmg_taken={best.dmg_taken} heals={best.heals} "
            f"boss={best.boss_hp_start}->{best.boss_hp_end}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_markdown_table(results) + "\n")
        print(f"wrote {args.md_out}")

    # Non-zero only if every trial life-lost / forbidden.
    if results and all(
        r.outcome in {"life_loss", "forbidden_a"} for r in results
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
