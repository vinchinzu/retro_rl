"""State-gated short skills for stitchless 8-3 (not long RLE tapes).

Skills are control/land-pin relative: they read RAM and emit short button
sequences. Prefer these over absolute FM2 offsets or natural_82 mid-splices.

- ``hammer_bro_absorber``: reactive RIGHT+B + timed A when a Hammer Bro / hammer
  is nearby; resumes into a short pure flagpole approach when clear.
- ``flagpole_macro``: multi-hop stair climb → flag grab (ps=4) → idle for auto.
- ``fpg_fireworks_hold``: explicit B-hold / jump-edge skeleton for FPG timing
  (timer digit class); used after leave polish, not as a mid-level body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from smb.ram import (
    ENEMY_TYPE_HAMMER,
    ENEMY_TYPE_HAMMER_BRO,
    PLAYER_STATE_AUTO_WALK,
    PLAYER_STATE_DYING,
    PLAYER_STATE_FLAGPOLE,
    read_enemy_slots,
    read_snapshot,
    rich_handoff_fingerprint,
)

# nes9 button order: B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A
IDLE: list[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
RUN: list[int] = [1, 0, 0, 0, 0, 0, 0, 1, 0]  # B + RIGHT
RUN_JUMP: list[int] = [1, 0, 0, 0, 0, 0, 0, 1, 1]  # B + RIGHT + A
RIGHT_ONLY: list[int] = [0, 0, 0, 0, 0, 0, 0, 1, 0]
JUMP_ONLY: list[int] = [0, 0, 0, 0, 0, 0, 0, 0, 1]

# 8-3 landmarks (absolute x, from annotated HL/nat traces)
X_STAIR_APPROACH = 3050
X_STAIR_LAND = 3120
X_FLAG_GRAB = 3380
X_FLAG_POLE = 3410

# Hammer Bro common spawn band on 8-3 (approximate)
X_HB_BAND = (900, 2200)


def clone_frame(fr: Sequence[int]) -> list[int]:
    return [int(x) for x in fr[:9]] + [0] * max(0, 9 - len(fr))


def hop_pattern(
    *,
    run0: int = 4,
    jhold: int = 28,
    gap: int = 10,
    hops: int = 3,
    run_tail: int = 80,
) -> list[list[int]]:
    """Short multi-hop RIGHT+B+A macro (stair / flag approach)."""
    out: list[list[int]] = [list(RUN) for _ in range(max(0, run0))]
    for _ in range(max(1, hops)):
        out.extend(list(RUN_JUMP) for _ in range(max(1, jhold)))
        out.extend(list(RUN) for _ in range(max(0, gap)))
    out.extend(list(RUN) for _ in range(max(0, run_tail)))
    return out


def flagpole_macro(
    *,
    style: str = "mid",
) -> list[list[int]]:
    """Pure short flagpole approach macros (no natural_82 body).

    Styles encode stair multi-hop + final lip jump. Auto-walk after ps=4 is
    handled by the consumer (idle pad), not this macro.
    """
    presets: dict[str, dict[str, int]] = {
        "short": {"run0": 2, "jhold": 20, "gap": 8, "hops": 2, "run_tail": 60},
        "mid": {"run0": 4, "jhold": 28, "gap": 12, "hops": 3, "run_tail": 100},
        "tall": {"run0": 6, "jhold": 36, "gap": 10, "hops": 4, "run_tail": 120},
        "lip": {"run0": 0, "jhold": 48, "gap": 0, "hops": 1, "run_tail": 140},
        # Natural-shaped geometry without splicing natural frames: long climb.
        "stairs": {"run0": 8, "jhold": 24, "gap": 14, "hops": 5, "run_tail": 100},
    }
    p = presets.get(style, presets["mid"])
    return hop_pattern(**p)


def fpg_fireworks_hold(
    *,
    b_hold: int = 40,
    jump_at: int = 12,
    jump_hold: int = 28,
    total: int = 90,
) -> list[list[int]]:
    """FPG / fireworks B-hold + jump-edge skeleton (timer digit class later).

    Emits RIGHT+B for ``b_hold``, with A edge starting at ``jump_at`` for
    ``jump_hold`` frames. Pure structural macro — phase-tune via timer_mod21.
    """
    out: list[list[int]] = []
    for t in range(max(1, total)):
        fr = list(RUN)
        if jump_at <= t < jump_at + jump_hold:
            fr[8] = 1  # A
        if t >= b_hold:
            fr[0] = 0  # release B
        out.append(fr)
    return out


@dataclass
class SkillStep:
    """One closed-loop skill decision."""

    buttons: list[int]
    skill_id: str
    reason: str


def hammer_bro_nearby(ram: np.ndarray, *, max_dx: int = 80) -> bool:
    snap = read_snapshot(ram, 0)
    px = int(snap.player_x)
    for e in read_enemy_slots(ram):
        if e["type"] not in (ENEMY_TYPE_HAMMER_BRO, ENEMY_TYPE_HAMMER):
            continue
        if abs(int(e["x"]) - px) <= max_dx:
            return True
    return False


class HammerBroAbsorber:
    """Reactive phase absorber: micro-jumps while HB/hammer near, else resume.

    Records emitted frames so a pure tape can be exported after a leave.
    """

    def __init__(
        self,
        *,
        jump_hold: int = 16,
        jump_period: int = 28,
        max_jumps: int = 8,
        resume: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.jump_hold = jump_hold
        self.jump_period = jump_period
        self.max_jumps = max_jumps
        self.resume = [clone_frame(f) for f in (resume or flagpole_macro(style="mid"))]
        self._jump_left = 0
        self._jumps = 0
        self._t = 0
        self._resume_i = 0
        self.recorded: list[list[int]] = []
        self.done = False

    def next_frame(self, ram: np.ndarray) -> list[int]:
        snap = read_snapshot(ram, 0)
        ps = int(snap.player_state)
        if ps in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
            fr = list(IDLE)
            self.recorded.append(fr)
            return fr
        if ps == PLAYER_STATE_DYING:
            self.done = True
            fr = list(IDLE)
            self.recorded.append(fr)
            return fr

        px = int(snap.player_x)
        g = bool(snap.grounded)
        near = hammer_bro_nearby(ram) or (X_HB_BAND[0] <= px <= X_HB_BAND[1] and self._jumps < 2)

        if self._jump_left > 0:
            fr = list(RUN_JUMP)
            self._jump_left -= 1
        elif near and g and self._jumps < self.max_jumps and self._t % self.jump_period == 0:
            fr = list(RUN_JUMP)
            self._jump_left = self.jump_hold - 1
            self._jumps += 1
        elif px >= X_STAIR_APPROACH:
            # Resume into short pure flagpole macro
            if self._resume_i < len(self.resume):
                fr = list(self.resume[self._resume_i])
                self._resume_i += 1
            else:
                fr = list(RUN)
        else:
            fr = list(RUN)

        self._t += 1
        self.recorded.append(list(fr))
        return fr


class FlagpoleSkill:
    """State-gated stair→flag skill from a land-pin seat.

    When ``player_x`` is in the stair/flag band and grounded (or forced), emit
    multi-hop until flag grab; then idle.
    """

    def __init__(
        self,
        *,
        style: str = "stairs",
        min_x: int = X_STAIR_APPROACH,
    ) -> None:
        self.body = flagpole_macro(style=style)
        self.min_x = min_x
        self._i = 0
        self.recorded: list[list[int]] = []
        self.flag_at: int | None = None
        self.done = False

    def next_frame(self, ram: np.ndarray) -> list[int]:
        snap = read_snapshot(ram, 0)
        ps = int(snap.player_state)
        if ps == PLAYER_STATE_FLAGPOLE:
            if self.flag_at is None:
                self.flag_at = self._i
            fr = list(IDLE)
            self.recorded.append(fr)
            self._i += 1
            return fr
        if ps == PLAYER_STATE_AUTO_WALK:
            fr = list(IDLE)
            self.recorded.append(fr)
            self._i += 1
            return fr
        if ps == PLAYER_STATE_DYING:
            self.done = True
            fr = list(IDLE)
            self.recorded.append(fr)
            return fr

        if self._i < len(self.body):
            fr = list(self.body[self._i])
        else:
            fr = list(RUN)
        self.recorded.append(list(fr))
        self._i += 1
        return fr


def score_trial(result: dict[str, Any]) -> tuple:
    """Score: leave first, then max_x, survival, favorable timer_mod21, not raw len."""
    leave = result.get("leave")
    max_x = int(result.get("max_x") or 0)
    death = result.get("death")
    survival = 1 if death is None else 0
    # Prefer timer_mod21 near 0 at leave (framerule-friendly); unknown → mid.
    tmod = result.get("timer_mod21")
    tmod_score = 0 if tmod is None else -min(int(tmod), 21 - int(tmod))
    leave_bonus = 1 if leave else 0
    leave_neg = -(int(leave) if leave else 10**9)
    return (leave_bonus, max_x, survival, tmod_score, leave_neg)


FLAGPOLE_STYLES: tuple[str, ...] = ("short", "mid", "tall", "lip", "stairs")


def open_skill_catalog() -> dict[str, Any]:
    """Named skills for docs / tests (no emulator)."""
    return {
        "hammer_bro_absorber": {
            "kind": "reactive",
            "resume": "flagpole_macro:mid",
            "notes": "RIGHT+B micro-jumps on HB/hammer proximity; resume flagpole",
        },
        "flagpole_macro": {
            "kind": "pure_macro",
            "styles": list(FLAGPOLE_STYLES),
            "land_pin_x": X_STAIR_LAND,
            "flag_x": X_FLAG_POLE,
        },
        "fpg_fireworks_hold": {
            "kind": "pure_macro",
            "params": ["b_hold", "jump_at", "jump_hold", "total"],
            "notes": "B-hold + A-edge; tune with timer_mod21 after leave",
        },
        "fingerprint": "smb.ram.rich_handoff_fingerprint",
        "score": "leave > max_x > survival > timer_mod21 closeness; not raw length",
    }
