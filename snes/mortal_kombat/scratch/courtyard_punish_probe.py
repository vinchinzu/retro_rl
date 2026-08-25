#!/usr/bin/env python3
"""Fight 7 pin: air-HK opener plus recovery-gated follow-ups."""

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
from mortal_kombat.scripted import (  # noqa: E402
    A,
    B,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    X,
    fireball_sequence,
    flying_kick_sequence,
    zeros,
)
from mortal_kombat.scripts.capture_natural_endurance1 import (  # noqa: E402
    ADDR_KNIFE_X,
    describe,
    knife_incoming,
    replay_through_fight7,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

JUMP_AT = 296
HOLD = 10


def trial(env, pin, name: str, *, kind: str, dump_from: int = 0, dump_to: int = 0, max_live: int = 1600):
    set_emulator_state(env.unwrapped, pin)
    boot = BootController(allow_continue=False)
    clock = 0
    armed = False
    air_at = None
    land_at = None
    first_p2 = None
    p2_hits = []
    crossed = None
    queue: list = []
    tap_left = 0
    wait_left = 0
    jumped = False
    prev = parse_ram(env.unwrapped.get_ram())
    print(f"\n=== {name} ===", flush=True)
    for frame in range(1, 2800):
        ram = env.unwrapped.get_ram()
        snap = parse_ram(ram)
        live = (
            snap.screen is Screen.FIGHT
            and snap.p1_health > 0
            and snap.p2_health > 0
            and snap.timer > 50
        )
        p1_x = int(ram[ADDR_P1_X]) & 0xFF
        p1_y = int(ram[ADDR_P1_Y]) & 0xFF
        p2_x = int(ram[ADDR_P2_X]) & 0xFF
        facing = 1 if p1_x <= p2_x else -1
        dist = abs(p2_x - p1_x)
        out = zeros()
        if live:
            if not armed:
                armed = True
                clock = 0
            clock += 1
            if air_at is None and 40 < p1_y < 140:
                air_at = clock
            if air_at is not None and land_at is None and p1_y >= 140:
                land_at = clock
            if not jumped and JUMP_AT <= clock < JUMP_AT + HOLD:
                out[UP] = 1
                out[RIGHT] = 1
            elif (
                not jumped
                and kind == "duck-knife"
                and knife_incoming(ram, p1_x, p2_x)
            ):
                out[DOWN] = 1
            elif not jumped and air_at is not None and p1_y < 140:
                if kind.startswith("air-lk"):
                    out[A] = 1
                elif kind.startswith("air-hk"):
                    out[B] = 1
            elif not jumped and land_at is not None:
                jumped = True
                if kind in ("baseline-land-hk", "duck-knife"):
                    tap_left = 8
                    out[B] = 1
                    tap_left -= 1
            if jumped:
                out, queue, tap_left, wait_left = follow(
                    kind,
                    snap,
                    ram,
                    p1_x,
                    p1_y,
                    p2_x,
                    facing,
                    dist,
                    queue,
                    tap_left,
                    wait_left,
                    land_at,
                    clock,
                )
            env.step(out)
        else:
            phase, names = boot.decide(snap, frame)
            env.step(zeros() if phase is Phase.FIGHT else action_from_buttons(names))
        snap = parse_ram(env.unwrapped.get_ram())
        ram = env.unwrapped.get_ram()
        p1_x = int(ram[ADDR_P1_X]) & 0xFF
        p1_y = int(ram[ADDR_P1_Y]) & 0xFF
        p2_x = int(ram[ADDR_P2_X]) & 0xFF
        knife = int(ram[ADDR_KNIFE_X]) & 0xFF
        if armed and crossed is None and p1_x > p2_x:
            crossed = clock
        hp_drop = snap.p1_health < prev.p1_health
        p2_drop = snap.p2_health < prev.p2_health
        if p2_drop:
            p2_hits.append((clock, snap.p2_health, p1_x, p2_x, p1_y, snap.p1.state))
            if first_p2 is None:
                first_p2 = p2_hits[-1]
        dump = dump_from and dump_from <= clock <= dump_to
        if hp_drop or p2_drop or dump:
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
                f"  f={frame} clock={clock} hp={snap.p1_health}/{snap.p2_health} "
                f"x={p1_x}/{p2_x} y={p1_y} st={snap.p1.state}/{snap.p2.state} "
                f"knife={knife} dist={abs(p2_x - p1_x)}{mark}",
                flush=True,
            )
        if armed and (snap.p1_health == 0 or snap.p2_health == 0):
            print(
                f"  round over {describe(snap)} air={air_at} land={land_at} "
                f"cross={crossed} p2hits={p2_hits}",
                flush=True,
            )
            return
        if armed and clock >= max_live:
            print(
                f"  stop {describe(snap)} air={air_at} land={land_at} "
                f"cross={crossed} p2hits={p2_hits}",
                flush=True,
            )
            return
        prev = snap
    print(f"  timeout {describe(prev)}", flush=True)


def follow(kind, snap, ram, p1_x, p1_y, p2_x, facing, dist, queue, tap_left, wait_left, land_at, clock):
    if tap_left > 0:
        tap_left -= 1
        out = zeros()
        out[B] = 1
        return out, queue, tap_left, wait_left
    if wait_left > 0:
        wait_left -= 1
        return zeros(), queue, tap_left, wait_left
    if queue:
        return queue.pop(0), queue, tap_left, wait_left
    if kind in ("air-hk-dump", "baseline-land-hk", "duck-knife"):
        return zeros(), queue, tap_left, wait_left
    if 40 < p1_y < 140:
        if kind.endswith("+rejump"):
            out = zeros()
            out[B] = 1
            return out, queue, tap_left, wait_left
        return zeros(), queue, tap_left, wait_left
    if snap.p1.state != 0 and kind not in ("air-hk+mashhk",):
        return zeros(), queue, tap_left, wait_left

    if kind == "air-hk+fly":
        queue = flying_kick_sequence(facing)
        return queue.pop(0), queue, tap_left, 40
    if kind == "air-lk+fly":
        queue = flying_kick_sequence(facing)
        return queue.pop(0), queue, tap_left, 40
    if kind == "air-hk+hk":
        out = zeros()
        out[B] = 1
        return out, queue, 3, 36
    if kind == "air-hk+mashhk":
        out = zeros()
        out[B] = 1
        return out, queue, tap_left, wait_left
    if kind == "air-hk+fb":
        if dist >= 72:
            queue = fireball_sequence(facing)
            return queue.pop(0), queue, tap_left, 50
        out = zeros()
        out[LEFT if facing >= 0 else RIGHT] = 1
        return out, queue, tap_left, wait_left
    if kind == "air-hk+rejump":
        out = zeros()
        out[UP] = 1
        if facing >= 0:
            out[RIGHT] = 1
        else:
            out[LEFT] = 1
        return out, queue, tap_left, 30
    if kind == "air-hk+react":
        if knife_incoming(ram, p1_x, p2_x) and dist > 40:
            out = zeros()
            out[DOWN] = 1
            return out, queue, tap_left, wait_left
        if dist <= 44:
            out = zeros()
            out[B] = 1
            return out, queue, 3, 28
        if dist >= 90:
            queue = fireball_sequence(facing)
            return queue.pop(0), queue, tap_left, 45
        out = zeros()
        if facing >= 0:
            out[RIGHT] = 1
        else:
            out[LEFT] = 1
        return out, queue, tap_left, wait_left
    if kind == "air-hk+block":
        out = zeros()
        out[DOWN] = 1
        out[X] = 1
        return out, queue, tap_left, wait_left
    if kind == "air-hk+upper":
        if dist <= 50:
            out = zeros()
            out[DOWN] = 1
            out[10] = 1  # L = LP
            return out, queue, tap_left, 40
        out = zeros()
        if facing >= 0:
            out[RIGHT] = 1
        else:
            out[LEFT] = 1
        return out, queue, tap_left, wait_left
    return zeros(), queue, tap_left, wait_left


def main() -> int:
    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replay through fight 7…", flush=True)
        replay_through_fight7(env)
        pin = get_emulator_state(env.unwrapped)
        print(f"leftover {describe(parse_ram(env.unwrapped.get_ram()))}", flush=True)
        trial(env, pin, "air-hk-dump", kind="air-hk-dump", dump_from=300, dump_to=420, max_live=430)
        trial(env, pin, "duck-knife", kind="duck-knife", dump_from=300, dump_to=400, max_live=420)
        trial(env, pin, "baseline-land-hk", kind="baseline-land-hk", dump_from=340, dump_to=380, max_live=420)
        for name in (
            "air-hk+fly",
            "air-lk+fly",
            "air-hk+hk",
            "air-hk+fb",
            "air-hk+rejump",
            "air-hk+react",
            "air-hk+block",
            "air-hk+upper",
        ):
            trial(env, pin, name, kind=name, dump_from=340, dump_to=520, max_live=1800)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
