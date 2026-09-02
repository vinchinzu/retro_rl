"""Scratch: hunt elevator wall-jump latches (pose 131/132) in a TAS replay.

The frame bases in ``tas/bodies`` and the verified lsnes log disagree, so
locate the event by its WRAM fingerprint rather than by a frame number.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from pathlib import Path

from retro_harness.env import make_env
from super_metroid.paths import GAME, GAME_DIR
from super_metroid.tas.slice import REF_DIR, REF_100, load_movie_frames
from super_metroid.tas.trace import _pad12

ELEV = 0xDF45
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 70_000
MOVIE = Path(sys.argv[2]) if len(sys.argv) > 2 else REF_100
if not MOVIE.is_absolute() and not MOVIE.exists():
    MOVIE = REF_DIR / MOVIE.name


def main() -> None:
    frames = load_movie_frames(MOVIE, MOVIE.suffix.lstrip("."))
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    t0 = time.time()
    latches: list[tuple[int, int, int, int]] = []
    best_y = 10_000
    try:
        env.reset()
        for i in range(min(LIMIT, len(frames))):
            env.step(np.asarray(_pad12(frames[i]), dtype=np.int8))
            ram = env.get_ram()
            if (int(ram[0x079B]) | (int(ram[0x079C]) << 8)) != ELEV:
                continue
            pose = int(ram[0x0A1C]) | (int(ram[0x0A1D]) << 8)
            x = int(ram[0x0AF6]) | (int(ram[0x0AF7]) << 8)
            y = int(ram[0x0AFA]) | (int(ram[0x0AFB]) << 8)
            if pose in (131, 132):
                latches.append((i, x, y, pose))
            if y < best_y and y > 0:
                best_y = y
            if i % 5000 == 0:
                print(f"  f{i} x={x} y={y} pose={pose} "
                      f"latches={len(latches)} {time.time() - t0:.0f}s", flush=True)
    finally:
        env.close()
    print(f"latch frames ({len(latches)}):")
    prev = -99
    for f, x, y, p in latches:
        if f - prev > 1:
            print(f"  --")
        print(f"  f{f} x={x} y={y} pose={p}")
        prev = f
    print(f"min y in elev = {best_y}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
