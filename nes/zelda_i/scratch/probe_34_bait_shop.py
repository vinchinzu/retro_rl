"""Scratch recon: poke ADDR_SCREEN=0x34 from a safe OW fixture and observe.

Not a route claim -- pure recon for rr-8t4.5 (the natural L6->0x34 overworld
walk is a separate open bead, rr-8t4.4). This script:

1. Loads a plain overworld fixture state.
2. Pokes ADDR_SCREEN=0x34, a central link_x/y, and a recon rupee count.
3. Settles a few idle frames and screenshots/dumps the resulting RAM to see
   whether the poke actually loads screen 0x34's terrain/object data.
4. If it looks like real 0x34 data, sweeps for a cave mouth (mode 16) and,
   if found, walks in and dumps the shop pedestal row (type 0x40 objects)
   to locate the Bait pedestal without touching an adjacent one.

    QT_QPA_PLATFORM=offscreen uv run python \
      nes/zelda_i/scratch/probe_34_bait_shop.py --dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_RUPEES,
    ADDR_SCREEN,
    CAVE_MODE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

SCRATCH_DIR = Path(__file__).resolve().parent
SOURCE_STATE = "PostSwordStart"
TARGET_SCREEN = 0x34
RECON_RUPEES = 200
STAND_X = 120
STAND_Y = 141


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def leftover_row(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "mode": int(snap.mode),
        "level": int(snap.level),
        "screen": int(snap.screen),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "rupees": int(snap.rupees),
        "tile": int(snap.colliding_tile),
        "objects": [
            {"slot": o.slot, "type": o.type_id, "x": o.x, "y": o.y, "hp": o.hp}
            for o in snap.objects
            if o.type_id
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--tag", default="probe_34")
    args = parser.parse_args()
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        before = leftover_row(env)
        print(f"before poke: {before}")

        _assign(env, ADDR_SCREEN, TARGET_SCREEN)
        _assign(env, ADDR_LINK_X, STAND_X)
        _assign(env, ADDR_LINK_Y, STAND_Y)
        if before["rupees"] < RECON_RUPEES:
            _assign(env, ADDR_RUPEES, RECON_RUPEES)

        for i in range(60):
            obs, *_ = env.step(nes_idle_action())
            if i in (0, 5, 29, 59):
                png = RECORDINGS_DIR / f"{args.tag}_settle_f{i}.png"
                save_rgb_png(obs, png)

        after = leftover_row(env)
        print(f"after poke + settle: {after}")
        png = RECORDINGS_DIR / f"{args.tag}_after_poke.png"
        save_rgb_png(obs, png)

        result = {"before": before, "after": after, "screenshot": str(png)}
        path = SCRATCH_DIR / f"{args.tag}_poke.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
