"""Deterministic RAM-scripted Liu Kang: zone with fireball / flying kick."""

from __future__ import annotations

from collections import deque

import numpy as np

from mortal_kombat.ram import (
    ADDR_P1_X,
    ADDR_P2_X,
    ADDR_P2_Y,
    MAX_HEALTH,
    PUNCH_RANGE,
    Screen,
    parse_ram,
)
from mortal_kombat.fight1_tape import FIGHT1_RLE

KIND_SCRIPT = "script"
SCRIPT_NAME = "scripted"

B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R = range(12)

FIREBALL_RANGE = 72
ANTI_AIR_RANGE = 56
AIRBORNE_Y = 140
SPECIAL_COOLDOWN = 55
# FIGHT! banner eats specials for ~90f on Fight_LiuKang; fireball then hits for 25.
INTRO_FRAMES = 90


class DeterministicFight1Policy:
    """Exact no-model replay for the ``Fight_LiuKang`` save state."""

    kind = KIND_SCRIPT
    name = "fight1_tape"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._run = 0
        self._remaining = FIGHT1_RLE[0][1]

    def act(
        self,
        ram: np.ndarray,
        rgb: np.ndarray | None,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        del ram, rgb, deterministic
        if self._run >= len(FIGHT1_RLE):
            return zeros()
        mask, _frames = FIGHT1_RLE[self._run]
        out = zeros()
        for button in range(12):
            out[button] = (mask >> button) & 1
        self._remaining -= 1
        if self._remaining == 0:
            self._run += 1
            if self._run < len(FIGHT1_RLE):
                self._remaining = FIGHT1_RLE[self._run][1]
        return out


def zeros() -> np.ndarray:
    return np.zeros(12, dtype=np.int8)


def forward(facing: int) -> np.ndarray:
    out = zeros()
    out[RIGHT if facing >= 0 else LEFT] = 1
    return out


def back(facing: int) -> np.ndarray:
    out = zeros()
    out[LEFT if facing >= 0 else RIGHT] = 1
    return out


def expand(buttons: np.ndarray, n: int) -> list[np.ndarray]:
    return [buttons.copy() for _ in range(n)]


def fireball_sequence(facing: int) -> list[np.ndarray]:
    """F(4) noop(2) F(4) noop(1) HP(4). Facing-relative forward, not always RIGHT."""
    hp = zeros()
    hp[Y] = 1
    fwd = forward(facing)
    return (
        expand(fwd, 4)
        + expand(zeros(), 2)
        + expand(fwd, 4)
        + expand(zeros(), 1)
        + expand(hp, 4)
    )


def flying_kick_sequence(facing: int) -> list[np.ndarray]:
    """F(4) noop(2) F(4) noop(1) HK(4). Facing-relative forward."""
    hk = zeros()
    hk[B] = 1
    fwd = forward(facing)
    return (
        expand(fwd, 4)
        + expand(zeros(), 2)
        + expand(fwd, 4)
        + expand(zeros(), 1)
        + expand(hk, 4)
    )


class ScriptedPolicy:
    """Frame-by-frame Liu Kang: fireball / flying kick / anti-air / block / re-space."""

    kind = KIND_SCRIPT
    name = SCRIPT_NAME

    def __init__(self, intro_frames: int = INTRO_FRAMES) -> None:
        self.intro_frames = intro_frames
        self._queue: deque[np.ndarray] = deque()
        self._cooldown = 0
        self._intro = 0
        self._hurt = 0
        self._prev_hp = 0
        self._was_full = False

    def reset(self) -> None:
        self._queue.clear()
        self._cooldown = 0
        self._intro = 0
        self._hurt = 0
        self._prev_hp = 0
        self._was_full = False

    def act(
        self,
        ram: np.ndarray,
        rgb: np.ndarray | None,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        del rgb, deterministic
        if self._queue:
            return self._queue.popleft()
        if self._cooldown > 0:
            self._cooldown -= 1
        return self._choose(parse_ram(ram), ram)

    def _enqueue(self, sequence: list[np.ndarray]) -> np.ndarray:
        self._queue.extend(sequence)
        self._cooldown = SPECIAL_COOLDOWN
        return self._queue.popleft()

    def _choose(self, snap, ram: np.ndarray) -> np.ndarray:
        if snap.screen is not Screen.FIGHT:
            return zeros()
        full = snap.p1_health == snap.p2_health == MAX_HEALTH
        if full and not self._was_full:
            self._intro = 0
        self._was_full = full
        if snap.p1_health < self._prev_hp:
            self._hurt = 24
        self._prev_hp = snap.p1_health
        if self._intro < self.intro_frames:
            self._intro += 1
            return zeros()
        if snap.p1.state != 0:
            return zeros()
        p1_x = int(ram[ADDR_P1_X]) & 0xFF if ADDR_P1_X < len(ram) else 0
        p2_x = int(ram[ADDR_P2_X]) & 0xFF if ADDR_P2_X < len(ram) else 0
        p2_y = int(ram[ADDR_P2_Y]) & 0xFF if ADDR_P2_Y < len(ram) else 0
        del ram
        dist = abs(p2_x - p1_x)
        facing = 1 if p1_x <= p2_x else -1
        if self._hurt > 0:
            self._hurt -= 1
            if dist > PUNCH_RANGE:
                jump = zeros()
                jump[UP] = 1
                return jump
            block = zeros()
            block[X] = 1
            return block
        if p2_y < AIRBORNE_Y and dist <= ANTI_AIR_RANGE:
            uppercut = zeros()
            uppercut[DOWN] = 1
            uppercut[L] = 1
            return uppercut
        if snap.p2.state != 0 and dist <= ANTI_AIR_RANGE:
            block = zeros()
            block[X] = 1
            return block
        cornered = p1_x < 28 or p1_x > 228
        if dist > FIREBALL_RANGE:
            if self._cooldown == 0:
                return self._enqueue(fireball_sequence(facing))
            return zeros()
        if cornered:
            jump_in = zeros()
            jump_in[UP] = 1
            jump_in[RIGHT if facing >= 0 else LEFT] = 1
            return jump_in
        return back(facing)
