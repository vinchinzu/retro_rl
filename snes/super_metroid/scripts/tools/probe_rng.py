#!/usr/bin/env python3
"""Dump $05E5 through first Ceres control, or advance a seed offline.

See ``snes/super_metroid/docs/RNG.md``.

```bash
# Offline: next k seeds (no ROM)
uv run python snes/super_metroid/scripts/tools/probe_rng.py --advance 5 --seed 0x5705

# Live: power-on TAS mash → first gs=8 on Ceres Elevator
uv run python snes/super_metroid/scripts/tools/probe_rng.py
uv run python snes/super_metroid/scripts/tools/probe_rng.py \\
  --json snes/super_metroid/scratch/rng_boot.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from super_metroid.ram import (
    ADDR_GAME_STATE,
    ADDR_RNG,
    ADDR_ROOM_ID,
    RNG_BOOT_SEED,
    rng1,
    rng1_rolls_between,
    read_wram_u16,
)

ROOM_CERES_ELEVATOR = 0xDF45
# Match play_boot_to_ceres_tas mash: START/A then A/idle.
_MASH_FRAMES = 400
_MAX_FRAMES = 12_000


def _parse_seed(text: str) -> int:
    return int(text, 0) & 0xFFFF


def advance_report(*, seed: int, steps: int) -> dict[str, Any]:
    values = [seed]
    cur = seed
    for _ in range(steps):
        cur = rng1(cur)
        values.append(cur)
    return {
        "kind": "advance",
        "seed": f"0x{seed:04X}",
        "steps": steps,
        "final": f"0x{values[-1]:04X}",
        "values": [f"0x{v:04X}" for v in values],
    }


def probe_boot(*, max_frames: int, mash_frames: int) -> dict[str, Any]:
    from retro_harness.actions import buttons, idle_action
    from retro_harness.env import make_env
    from super_metroid.paths import GAME, GAME_DIR

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    events: list[dict[str, Any]] = []
    prev_rng = read_wram_u16(env, ADDR_RNG)
    prev_gs = read_wram_u16(env, ADDR_GAME_STATE)
    seeded_frame: int | None = None
    first_roll_frame: int | None = None
    first_control: dict[str, Any] | None = None
    extra_roll_frames = 0
    extra_rolls = 0
    skipped_frames = 0

    events.append(
        {
            "kind": "reset",
            "frame": 0,
            "rng": f"0x{prev_rng:04X}",
            "game_state": prev_gs,
            "room": f"0x{read_wram_u16(env, ADDR_ROOM_ID):04X}",
        }
    )

    try:
        for i in range(max_frames):
            if i < mash_frames:
                act = buttons("START" if (i % 2) == 0 else "A")
            elif (i % 2) == 0:
                act = buttons("A")
            else:
                act = idle_action()
            env.step(act)
            frame = i + 1
            rng = read_wram_u16(env, ADDR_RNG)
            gs = read_wram_u16(env, ADDR_GAME_STATE)
            room = read_wram_u16(env, ADDR_ROOM_ID)

            if seeded_frame is None and rng == RNG_BOOT_SEED:
                seeded_frame = frame
                events.append(
                    {
                        "kind": "boot_seed_0061",
                        "frame": frame,
                        "rng": f"0x{RNG_BOOT_SEED:04X}",
                        "game_state": gs,
                        "room": f"0x{room:04X}",
                        "prev_rng": f"0x{prev_rng:04X}",
                    }
                )

            if rng != prev_rng:
                n = rng1_rolls_between(prev_rng, rng)
                if first_roll_frame is None and prev_rng == RNG_BOOT_SEED:
                    first_roll_frame = frame
                    events.append(
                        {
                            "kind": "first_main_loop_roll",
                            "frame": frame,
                            "from": f"0x{RNG_BOOT_SEED:04X}",
                            "to": f"0x{rng:04X}",
                            "rolls": n,
                            "expected_rng1": f"0x{rng1(RNG_BOOT_SEED):04X}",
                            "game_state": gs,
                        }
                    )
                if n > 1:
                    extra_roll_frames += 1
                    extra_rolls += n - 1
                    if extra_roll_frames <= 40:
                        events.append(
                            {
                                "kind": "extra_rolls",
                                "frame": frame,
                                "from": f"0x{prev_rng:04X}",
                                "to": f"0x{rng:04X}",
                                "rolls": n,
                                "game_state": gs,
                                "room": f"0x{room:04X}",
                            }
                        )
                elif n < 0 and seeded_frame is not None:
                    events.append(
                        {
                            "kind": "non_formula_write",
                            "frame": frame,
                            "from": f"0x{prev_rng:04X}",
                            "to": f"0x{rng:04X}",
                            "game_state": gs,
                            "room": f"0x{room:04X}",
                        }
                    )
            elif seeded_frame is not None and first_roll_frame is not None:
                skipped_frames += 1

            if gs != prev_gs:
                events.append(
                    {
                        "kind": "game_state",
                        "frame": frame,
                        "game_state": gs,
                        "prev_game_state": prev_gs,
                        "rng": f"0x{rng:04X}",
                        "room": f"0x{room:04X}",
                    }
                )

            if first_control is None and room == ROOM_CERES_ELEVATOR and gs == 8:
                first_control = {
                    "kind": "first_ceres_control",
                    "frame": frame,
                    "rng": f"0x{rng:04X}",
                    "game_state": gs,
                    "room": f"0x{room:04X}",
                    "seeded_frame": seeded_frame,
                    "first_roll_frame": first_roll_frame,
                }
                events.append(first_control)
                break

            prev_rng = rng
            prev_gs = gs
    finally:
        env.close()

    return {
        "kind": "boot",
        "addr": f"0x{ADDR_RNG:04X}",
        "boot_seed_expected": f"0x{RNG_BOOT_SEED:04X}",
        "seeded_frame": seeded_frame,
        "first_roll_frame": first_roll_frame,
        "first_control": first_control,
        "extra_roll_frames": extra_roll_frames,
        "extra_rolls": extra_rolls,
        "skipped_frames_after_loop": skipped_frames,
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        default=f"0x{RNG_BOOT_SEED:04X}",
        help="Starting $05E5 (hex or int). Default: boot seed 0x0061",
    )
    parser.add_argument(
        "--advance",
        type=int,
        default=None,
        metavar="K",
        help="Offline: print seed after K rng1 steps (no ROM)",
    )
    parser.add_argument("--json", type=Path, help="Write the report JSON here")
    parser.add_argument("--max-frames", type=int, default=_MAX_FRAMES)
    parser.add_argument("--mash-frames", type=int, default=_MASH_FRAMES)
    args = parser.parse_args(argv)

    seed = _parse_seed(args.seed)
    if args.advance is not None:
        report = advance_report(seed=seed, steps=args.advance)
    else:
        report = probe_boot(max_frames=args.max_frames, mash_frames=args.mash_frames)

    text = json.dumps(report, indent=2) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if report.get("kind") == "advance" or report.get("first_control") else 1


if __name__ == "__main__":
    raise SystemExit(main())
