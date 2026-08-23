#!/usr/bin/env python3
"""Dump fighter-object bytes and punch/walk diffs from Fight_LiuKang."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

import numpy as np  # noqa: E402

from retro_harness.actions import snes_action  # noqa: E402
from retro_harness.env import make_env, reset_obs  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    FIGHTER_STRIDE,
    OFF_STATE,
    P1_OBJ,
    P2_OBJ,
    parse_ram,
)


def _hold(env, *names: str, frames: int = 30) -> np.ndarray:
    action = snes_action(*names, dtype=np.int8) if names else snes_action(dtype=np.int8)
    last = env.unwrapped.get_ram().copy()
    for _ in range(frames):
        env.step(action)
        last = env.unwrapped.get_ram().copy()
    return last


def _obj(ram: np.ndarray, base: int) -> bytes:
    return bytes(int(x) & 0xFF for x in ram[base : base + FIGHTER_STRIDE])


def _diff(before: bytes, after: bytes) -> list[tuple[int, int, int]]:
    rows = []
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if a != b:
            rows.append((i, a, b))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Fight_LiuKang")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_env(GAME_ID, args.state, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        idle = _hold(env, frames=20)
        snap = parse_ram(idle)
        print(
            f"idle hp={snap.p1_health}/{snap.p2_health} "
            f"xy=({snap.p1.x},{snap.p1.y})-({snap.p2.x},{snap.p2.y}) "
            f"state={snap.p1.state}/{snap.p2.state} "
            f"dist={snap.distance_x} ram_len={snap.ram_len} "
            f"off_state=0x{OFF_STATE:02X}"
        )
        idle_p1 = _obj(idle, P1_OBJ)
        right = _hold(env, "RIGHT", frames=40)
        right_p1 = _obj(right, P1_OBJ)
        print("walk_right p1 object diffs (offset, before, after):")
        for off, a, b in _diff(idle_p1, right_p1)[:20]:
            print(f"  +0x{off:02X} {a:3d} -> {b:3d}")
        punch = _hold(env, "Y", frames=20)
        punch_p1 = _obj(punch, P1_OBJ)
        punch_p2 = _obj(punch, P2_OBJ)
        idle_p2 = _obj(idle, P2_OBJ)
        print("high_punch p1 object diffs:")
        for off, a, b in _diff(right_p1, punch_p1)[:20]:
            print(f"  +0x{off:02X} {a:3d} -> {b:3d}")
        print("high_punch p2 object diffs:")
        for off, a, b in _diff(idle_p2, punch_p2)[:20]:
            print(f"  +0x{off:02X} {a:3d} -> {b:3d}")
        after = parse_ram(punch)
        print(
            f"after punch xy=({after.p1.x},{after.p1.y})-({after.p2.x},{after.p2.y}) "
            f"state={after.p1.state}/{after.p2.state} overlap={after.bodies_overlap} "
            f"p1_hit={after.p1_hit_connects}"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
