#!/usr/bin/env python3
"""Replay the model-free power-on Liu Kang Match 1–7 input tape."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "snes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mortal_kombat.natural_fight7_tape import (  # noqa: E402
    NATURAL_FIGHT7_FRAMES,
    NATURAL_FIGHT7_RLE,
)
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    ADDR_P1_X,
    ADDR_P2_X,
    LIU_KANG_ID,
    char_name,
    parse_ram,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask  # noqa: E402
from mortal_kombat.scripts.replay_natural_fight6 import (  # noqa: E402
    NATURAL_THROUGH_FIGHT6_FRAMES,
    NATURAL_THROUGH_FIGHT6_RLE,
)
from retro_harness.env import make_env, reset_obs  # noqa: E402

NATURAL_THROUGH_FIGHT7_FRAMES = NATURAL_THROUGH_FIGHT6_FRAMES + NATURAL_FIGHT7_FRAMES
NATURAL_THROUGH_FIGHT7_RLE = NATURAL_THROUGH_FIGHT6_RLE + NATURAL_FIGHT7_RLE
MATCHES = (0, 1, 2, 3, 4, 5, 6)


def ram_signature(snap, ram) -> bytes:
    p1_x = int(ram[ADDR_P1_X]) & 0xFF if ADDR_P1_X < len(ram) else 0
    p2_x = int(ram[ADDR_P2_X]) & 0xFF if ADDR_P2_X < len(ram) else 0
    return bytes(
        (
            snap.match_counter,
            snap.p1_character,
            snap.p2_character,
            snap.p1_health,
            snap.p2_health,
            snap.timer,
            snap.p1_rounds,
            snap.p2_rounds,
            int(snap.screen),
            p1_x,
            p2_x,
        )
    )


def play_tape(env) -> tuple[object, int, dict[int, tuple[int, int]], str]:
    """Play the concatenated tape. Returns snap, frames, kos-by-match, digest."""
    p1_kos = {match: 0 for match in MATCHES}
    p2_kos = {match: 0 for match in MATCHES}
    started = {match: False for match in MATCHES}
    prev_health: dict[int, tuple[int, int] | None] = {match: None for match in MATCHES}
    digest = hashlib.sha256()
    frame = 0
    reset_obs(env)
    for mask, count in NATURAL_THROUGH_FIGHT7_RLE:
        buttons = buttons_from_mask(mask)
        for _ in range(count):
            ram = env.unwrapped.get_ram()
            snap = parse_ram(ram)
            match = snap.match_counter
            health = (snap.p1_health, snap.p2_health)
            if match in started:
                started[match] = started[match] or (
                    snap.p1_character == LIU_KANG_ID
                    and snap.timer > 50
                    and health == (161, 161)
                )
                if started[match] and prev_health[match] is not None:
                    p1_kos[match] += int(prev_health[match][1] > 0 and health[1] == 0)
                    p2_kos[match] += int(prev_health[match][0] > 0 and health[0] == 0)
                prev_health[match] = health if started[match] else None
            digest.update(ram_signature(snap, ram))
            env.step(buttons)
            frame += 1
    snap = parse_ram(env.unwrapped.get_ram())
    kos = {match: (p1_kos[match], p2_kos[match]) for match in MATCHES}
    return snap, frame, kos, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    first_sig: str | None = None
    all_ok = True
    for attempt in range(args.repeat):
        # Cold boot each attempt. env.reset() after a long NONE run does not
        # restore the power-on pin the tape was captured from.
        env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
        try:
            snap, frame, kos, sig = play_tape(env)
        finally:
            env.close()
        transitioned = snap.match_counter >= 7 and snap.p1_character == LIU_KANG_ID
        identical = first_sig is None or sig == first_sig
        if first_sig is None:
            first_sig = sig
        all_ok = all_ok and transitioned and identical
        print(
            f"attempt={attempt + 1}/{args.repeat} transitioned={transitioned} "
            f"identical={identical} frame={frame}/{NATURAL_THROUGH_FIGHT7_FRAMES} "
            f"match={snap.match_counter} char={snap.p1_character} "
            f"p2={snap.p2_character}/{char_name(snap.p2_character)} "
            f"m1_kos={kos[0][0]}-{kos[0][1]} m2_kos={kos[1][0]}-{kos[1][1]} "
            f"m3_kos={kos[2][0]}-{kos[2][1]} m4_kos={kos[3][0]}-{kos[3][1]} "
            f"m5_kos={kos[4][0]}-{kos[4][1]} m6_kos={kos[5][0]}-{kos[5][1]} "
            f"m7_kos={kos[6][0]}-{kos[6][1]} "
            f"hp={snap.p1_health}/{snap.p2_health}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
