"""Scratch: replay the Sniq 100% BK2 from power-on and dump raw WRAM windows.

The elevator wall-jump chain lives around TAS frame 13074-13148. Everything
about the contract (pose ladder, speed, hitbox width at contact, the frame
window) has to come off the authentic tape, so this steps the movie with no
sanitize and snapshots low WRAM every frame inside the window.

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.tas_wram
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.env import make_env
from super_metroid.paths import GAME, GAME_DIR
from super_metroid.tas.slice import REF_100, load_movie_frames
from super_metroid.tas.trace import _pad12, frame_button_names

HERE = Path(__file__).resolve().parent
OUT = HERE / "tas_wram.json"

# Window to capture densely (movie frame index, power-on relative).
WIN_LO = int(os.environ.get("WIN_LO", 12_900))
WIN_HI = int(os.environ.get("WIN_HI", 13_260))

# Raw WRAM slices kept per frame: (name, lo, hi) — hi exclusive.
SLICES = (
    ("dp", 0x0000, 0x0100),      # direct page: controller latches live here
    ("samus", 0x0A00, 0x0B60),   # pose / position / radii / speed
    ("room", 0x0780, 0x07C0),    # room id, door, scroll
    ("gs", 0x0990, 0x09B0),
)


def grab(ram: np.ndarray) -> dict[str, str]:
    return {n: ram[lo:hi].tobytes().hex() for n, lo, hi in SLICES}


def main() -> None:
    frames = load_movie_frames(REF_100, "bk2")
    print(f"movie frames={len(frames)}", flush=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    rows: list[dict] = []
    t0 = time.time()
    try:
        env.reset()
        for i in range(min(WIN_HI, len(frames))):
            env.step(np.asarray(_pad12(frames[i]), dtype=np.int8))
            if i >= WIN_LO:
                ram = env.get_ram()
                rows.append(
                    {
                        "f": i,
                        "btn": frame_button_names(frames[i]),
                        "ram": grab(ram),
                    }
                )
            if i % 2000 == 0:
                ram = env.get_ram()
                room = int(ram[0x079B]) | (int(ram[0x079C]) << 8)
                print(
                    f"  f{i} room={room:#06x} "
                    f"x={int(ram[0x0AF6]) | (int(ram[0x0AF7]) << 8)} "
                    f"y={int(ram[0x0AFA]) | (int(ram[0x0AFB]) << 8)} "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )
    finally:
        env.close()
    OUT.write_text(json.dumps({"window": [WIN_LO, WIN_HI], "rows": rows}) + "\n")
    print(f"rows={len(rows)} report: {OUT} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
