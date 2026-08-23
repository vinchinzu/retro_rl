#!/usr/bin/env python3
"""Replay the model-free power-on Liu Kang Match 1 input tape."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "snes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mortal_kombat.natural_fight1_tape import (  # noqa: E402
    NATURAL_FIGHT1_FRAMES,
    NATURAL_FIGHT1_RLE,
)
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import LIU_KANG_ID, parse_ram  # noqa: E402
from retro_harness.env import make_env, reset_obs  # noqa: E402


def buttons_from_mask(mask: int) -> np.ndarray:
    buttons = np.zeros(12, dtype=np.int8)
    for button in range(12):
        buttons[button] = (mask >> button) & 1
    return buttons


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    p1_kos = p2_kos = frame = 0
    prev_health: tuple[int, int] | None = None
    fight_started = False
    try:
        reset_obs(env)
        for mask, count in NATURAL_FIGHT1_RLE:
            buttons = buttons_from_mask(mask)
            for _ in range(count):
                snap = parse_ram(env.unwrapped.get_ram())
                health = (snap.p1_health, snap.p2_health)
                fight_started = fight_started or (
                    snap.match_counter == 0
                    and snap.p1_character == LIU_KANG_ID
                    and snap.timer > 50
                    and health == (161, 161)
                )
                if fight_started and snap.match_counter == 0 and prev_health is not None:
                    p1_kos += int(prev_health[1] > 0 and health[1] == 0)
                    p2_kos += int(prev_health[0] > 0 and health[0] == 0)
                prev_health = health if fight_started and snap.match_counter == 0 else None
                env.step(buttons)
                frame += 1
        snap = parse_ram(env.unwrapped.get_ram())
    finally:
        env.close()
    transitioned = snap.match_counter >= 1 and snap.p1_character == LIU_KANG_ID
    print(
        f"transitioned={transitioned} frame={frame}/{NATURAL_FIGHT1_FRAMES} "
        f"match={snap.match_counter} char={snap.p1_character} p2={snap.p2_character} "
        f"kos={p1_kos}-{p2_kos} hp={snap.p1_health}/{snap.p2_health}"
    )
    return 0 if transitioned else 1


if __name__ == "__main__":
    raise SystemExit(main())
