"""Scratch: map every wall-jumpable spot in the elevator shaft.

Uses the latch itself as the probe. Park an airborne spin state, poke Samus to
(x, y), run the measured kick shape (two frames of *away* with A off, then
away+A) and record whether pose 131/132 fired. The result is the surface map
the chain search has been missing: where a second wall jump is available at all.

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.wjmap
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons
from super_metroid.ram import (
    ADDR_SAMUS_X,
    ADDR_SAMUS_X_SUB,
    ADDR_SAMUS_Y,
    ADDR_SAMUS_Y_SUB,
    parse_state,
    write_wram_u16,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "wjmap.json"

XS = list(range(32, 229, 4))
YS = list(range(168, 649, 8))


def probe(env, session, blob: bytes, x: int, y: int, away: str) -> int:
    """0 = no latch, 131/132 = the pose that fired."""
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(8):
        session.step(buttons("RIGHT", "A"), "rise")
    write_wram_u16(env, ADDR_SAMUS_X, x)
    write_wram_u16(env, ADDR_SAMUS_X_SUB, 0)
    write_wram_u16(env, ADDR_SAMUS_Y, y)
    write_wram_u16(env, ADDR_SAMUS_Y_SUB, 0)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(2):
        session.step(buttons(away), "release")
    for _ in range(3):
        session.step(buttons(away, "A"), "kick")
        pose = int(session.state.pose)
        if pose in (131, 132):
            return pose
    return 0


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    grid: dict[int, dict[str, list[int]]] = {}
    t0 = time.time()
    try:
        for y in YS:
            row = {"LEFT": [], "RIGHT": []}
            for away in ("LEFT", "RIGHT"):
                for x in XS:
                    if probe(env, session, blob, x, y, away):
                        row[away].append(x)
            grid[y] = row
            left = f"{min(row['RIGHT']):3d}-{max(row['RIGHT']):3d}" if row["RIGHT"] else "  -  "
            right = f"{min(row['LEFT']):3d}-{max(row['LEFT']):3d}" if row["LEFT"] else "  -  "
            print(f"y={y:3d}  kick-right(from left wall) x={left}   "
                  f"kick-left(from right wall) x={right}   "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        env.close()
    OUT.write_text(json.dumps({str(k): v for k, v in grid.items()}, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
