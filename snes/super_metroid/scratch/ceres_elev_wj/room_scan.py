"""Scratch: room-id timeline for a TAS replay — desync check before any WRAM claim."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.env import make_env
from super_metroid.paths import GAME, GAME_DIR
from pathlib import Path

from super_metroid.tas.slice import REF_DIR, REF_100, load_movie_frames
from super_metroid.tas.trace import _pad12

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
MOVIE = Path(sys.argv[2]) if len(sys.argv) > 2 else REF_100
if not MOVIE.is_absolute() and not MOVIE.exists():
    MOVIE = REF_DIR / MOVIE.name
KIND = MOVIE.suffix.lstrip(".")


def main() -> None:
    frames = load_movie_frames(MOVIE, KIND)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    t0 = time.time()
    prev = None
    try:
        env.reset()
        for i in range(min(LIMIT, len(frames))):
            env.step(np.asarray(_pad12(frames[i]), dtype=np.int8))
            ram = env.get_ram()
            room = int(ram[0x079B]) | (int(ram[0x079C]) << 8)
            if room != prev:
                x = int(ram[0x0AF6]) | (int(ram[0x0AF7]) << 8)
                y = int(ram[0x0AFA]) | (int(ram[0x0AFB]) << 8)
                gs = int(ram[0x0998]) | (int(ram[0x0999]) << 8)
                print(f"f{i:6d} room={room:#06x} gs={gs:2d} x={x:4d} y={y:4d}", flush=True)
                prev = room
    finally:
        env.close()
    print(f"{time.time() - t0:.1f}s over {min(LIMIT, len(frames))} frames")


if __name__ == "__main__":
    main()
