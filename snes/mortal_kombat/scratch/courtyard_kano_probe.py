#!/usr/bin/env python3
"""From leftover Fight 7 pin, try jump clocks after 161/161 (capture path)."""

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
from mortal_kombat.scripted import B, RIGHT, UP, zeros  # noqa: E402
from mortal_kombat.scripts.capture_natural_endurance1 import (  # noqa: E402
    ADDR_KNIFE_X,
    describe,
    replay_through_fight7,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def trial(env, pin, jump_at: int, hold: int, forward: bool) -> None:
    set_emulator_state(env.unwrapped, pin)
    boot = BootController(allow_continue=False)
    clock = 0
    armed = False
    air = None
    prev = parse_ram(env.unwrapped.get_ram())
    print(f"\n=== leftover jump_at={jump_at} hold={hold} fwd={forward} ===", flush=True)
    for frame in range(1, 2500):
        ram = env.unwrapped.get_ram()
        snap = parse_ram(ram)
        live = (
            snap.screen is Screen.FIGHT
            and snap.p1_health > 0
            and snap.p2_health > 0
            and snap.timer > 50
        )
        out = zeros()
        if live:
            if not armed:
                armed = True
                clock = 0
                print(f"  armed f={frame} {describe(snap)}", flush=True)
            clock += 1
            p1_y = int(ram[ADDR_P1_Y]) & 0xFF
            if jump_at <= clock < jump_at + hold:
                out[UP] = 1
                if forward:
                    out[RIGHT] = 1
            elif air is not None and clock < jump_at + 80:
                out[B] = 1
            env.step(out)
        else:
            phase, names = boot.decide(snap, frame)
            if phase is Phase.FIGHT:
                env.step(zeros())
            else:
                env.step(action_from_buttons(names))
        snap = parse_ram(env.unwrapped.get_ram())
        ram = env.unwrapped.get_ram()
        p1_x = int(ram[ADDR_P1_X]) & 0xFF
        p1_y = int(ram[ADDR_P1_Y]) & 0xFF
        p2_x = int(ram[ADDR_P2_X]) & 0xFF
        knife = int(ram[ADDR_KNIFE_X]) & 0xFF
        if p1_y < 140 and air is None:
            air = (frame, clock, p1_y)
            print(f"  AIR f={frame} clock={clock} y={p1_y} x={p1_x}/{p2_x}", flush=True)
        hp_drop = snap.p1_health < prev.p1_health
        p2_drop = snap.p2_health < prev.p2_health
        if hp_drop or p2_drop or (armed and clock in (jump_at, jump_at + hold, 273, 324)):
            mark = ""
            if hp_drop:
                mark += " P1HIT"
            if p2_drop:
                mark += " P2HIT"
            if p1_y < 140:
                mark += " AIR"
            print(
                f"  f={frame} clock={clock} hp={snap.p1_health}/{snap.p2_health} "
                f"x={p1_x}/{p2_x} y={p1_y} knife={knife}{mark}",
                flush=True,
            )
        if armed and (snap.p1_health == 0 or snap.p2_health == 0):
            print(f"  round over {describe(snap)} air={air}", flush=True)
            return
        prev = snap
    print(f"  timeout {describe(prev)} air={air}", flush=True)


def main() -> int:
    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replay through fight 7…", flush=True)
        replay_through_fight7(env)
        pin = get_emulator_state(env.unwrapped)
        print(f"leftover {describe(parse_ram(env.unwrapped.get_ram()))}", flush=True)
        for at, hold, fwd in (
            (240, 10, True),
            (250, 10, True),
            (280, 10, True),
            (291, 10, True),
            (296, 10, True),
            (296, 10, False),
            (296, 20, True),
            (310, 10, True),
            (330, 10, True),
        ):
            trial(env, pin, at, hold, fwd)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
