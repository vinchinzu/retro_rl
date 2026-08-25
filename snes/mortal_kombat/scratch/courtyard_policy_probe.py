#!/usr/bin/env python3
"""Drive CourtyardKanoPolicy from the leftover Fight 7 pin for one match."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "snes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from retro_harness.env import make_env  # noqa: E402
from retro_harness.snapshot import get_emulator_state, set_emulator_state  # noqa: E402
from mortal_kombat.boot import BootController, Phase, action_from_buttons  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import ADDR_P1_X, ADDR_P1_Y, ADDR_P2_X, Screen, parse_ram  # noqa: E402
from mortal_kombat.scripted import zeros  # noqa: E402
from mortal_kombat.scripts.capture_natural_endurance1 import (  # noqa: E402
    ADDR_KNIFE_X,
    CourtyardKanoPolicy,
    describe,
    replay_through_fight7,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main() -> int:
    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replay through fight 7…", flush=True)
        replay_through_fight7(env)
        pin = get_emulator_state(env.unwrapped)
        print(f"leftover {describe(parse_ram(env.unwrapped.get_ram()))}", flush=True)
        set_emulator_state(env.unwrapped, pin)
        boot = BootController(allow_continue=False)
        policy = CourtyardKanoPolicy()
        prev = parse_ram(env.unwrapped.get_ram())
        p1_kos = p2_kos = 0
        for frame in range(1, 8000):
            ram = env.unwrapped.get_ram()
            snap = parse_ram(ram)
            live = (
                snap.screen is Screen.FIGHT
                and snap.p1_health > 0
                and snap.p2_health > 0
                and snap.timer > 50
            )
            if live:
                out = policy.act(ram, None)
                env.step(out)
            else:
                policy.reset()
                phase, names = boot.decide(snap, frame)
                env.step(zeros() if phase is Phase.FIGHT else action_from_buttons(names))
            snap = parse_ram(env.unwrapped.get_ram())
            ram = env.unwrapped.get_ram()
            p1_x = int(ram[ADDR_P1_X]) & 0xFF
            p1_y = int(ram[ADDR_P1_Y]) & 0xFF
            p2_x = int(ram[ADDR_P2_X]) & 0xFF
            knife = int(ram[ADDR_KNIFE_X]) & 0xFF
            if prev.p1_health > 0 and snap.p1_health == 0:
                p2_kos += 1
            if prev.p2_health > 0 and snap.p2_health == 0:
                p1_kos += 1
            hp_drop = snap.p1_health < prev.p1_health
            p2_drop = snap.p2_health < prev.p2_health
            if hp_drop or p2_drop or frame % 400 == 0:
                mark = ""
                if hp_drop:
                    mark += " P1HIT"
                if p2_drop:
                    mark += " P2HIT"
                if 40 < p1_y < 140:
                    mark += " AIR"
                if p1_x > p2_x:
                    mark += " CROSS"
                print(
                    f"  f={frame} kos={p1_kos}-{p2_kos} {describe(snap)} "
                    f"x={p1_x}/{p2_x} y={p1_y} knife={knife}{mark}",
                    flush=True,
                )
            if snap.screen is Screen.CONTINUE:
                print(f"continue kos={p1_kos}-{p2_kos} {describe(snap)}", flush=True)
                return 0
            prev = snap
        print(f"timeout kos={p1_kos}-{p2_kos} {describe(prev)}", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
