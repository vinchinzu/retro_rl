"""Deterministic Liu Kang fireball loop (facing-right P1 default)."""

from __future__ import annotations

from collections import deque

import numpy as np

from mortal_kombat_ii.ram import parse_ram

KIND_SCRIPT = "script"
SCRIPT_NAME = "scripted"

B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R = range(12)
INTRO_FRAMES = 90
SPECIAL_GAP = 24


def zeros() -> np.ndarray:
    return np.zeros(12, dtype=np.int8)


def expand(buttons: np.ndarray, n: int) -> list[np.ndarray]:
    return [buttons.copy() for _ in range(n)]


def fireball_sequence() -> list[np.ndarray]:
    """QCF + HP: Liu Kang high fireball while P1 faces right."""
    down = zeros()
    down[DOWN] = 1
    down_forward = zeros()
    down_forward[DOWN] = 1
    down_forward[RIGHT] = 1
    forward = zeros()
    forward[RIGHT] = 1
    hp = zeros()
    hp[Y] = 1
    return (
        expand(down, 4)
        + expand(down_forward, 4)
        + expand(forward, 4)
        + expand(hp, 6)
        + expand(zeros(), SPECIAL_GAP)
    )


class ScriptedPolicy:
    """Health-aware fireball spam. Match-win is scored by ``eval_match``."""

    kind = KIND_SCRIPT
    name = SCRIPT_NAME

    def __init__(self, intro_frames: int = INTRO_FRAMES) -> None:
        self.intro_frames = intro_frames
        self._queue: deque[np.ndarray] = deque()
        self._intro = 0

    def reset(self) -> None:
        self._queue.clear()
        self._intro = 0

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
        snap = parse_ram(ram)
        if snap.p1_health == 0 or snap.p2_health == 0:
            return zeros()
        if self._intro < self.intro_frames:
            self._intro += 1
            return zeros()
        sequence = fireball_sequence()
        self._queue.extend(sequence[1:])
        return sequence[0]
