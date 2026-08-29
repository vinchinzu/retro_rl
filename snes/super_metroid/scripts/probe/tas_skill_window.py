#!/usr/bin/env python3
"""Replay one offline-detected TAS skill window from a live pin.

Assist off. Halt at the first RAM miss. Not a STATUS pin and not a dual.

```bash
uv run python snes/super_metroid/scripts/probe/tas_skill_window.py \\
  --slice hero_bubbleroom_full --skill arm_pump \\
  --state-path snes/super_metroid/custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.state
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.controls import SNES_LEFT, SNES_RIGHT
from retro_harness.env import make_env, read_state_bytes
from super_metroid.paths import GAME, GAME_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.tas.rle import expand_snes12_rle, load_snes12_rle_seed
from super_metroid.tas.skills_extract import detect_slice_skills
from super_metroid.tas.slice import SLICE_DIR
from super_metroid.tas.trace import action_array, frame_button_names

SCRATCH = GAME_DIR / "scratch"


def _run_dir(frame) -> str | None:
    left = bool(frame[SNES_LEFT])
    right = bool(frame[SNES_RIGHT])
    if left and not right:
        return "LEFT"
    if right and not left:
        return "RIGHT"
    return None


def _pin_brief(state) -> dict[str, object]:
    pin = probe_pin(state)
    return {
        "room": pin["room"],
        "pose": pin["pose"],
        "x": pin["x"],
        "y": pin["y"],
        "facing": pin["facing"],
        "game_state": int(state.game_state),
        "door_transition": pin["door_transition"],
        "phase": pin["phase"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice", dest="slice_id", required=True)
    p.add_argument("--skill", required=True, help="arm_pump | mockball")
    p.add_argument("--state-path", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Scratch JSON (default scratch/tas_skill_<slice>_<skill>.json)",
    )
    args = p.parse_args(argv)

    if not args.state_path.exists():
        print(f"skip: missing pin {args.state_path}", file=sys.stderr)
        return 2

    try:
        windows = detect_slice_skills(args.slice_id)
    except FileNotFoundError as exc:
        print(f"skip: {exc}", file=sys.stderr)
        return 2
    match = [w for w in windows if w.skill == args.skill]
    if not match:
        print(f"error: no {args.skill} window in {args.slice_id}", file=sys.stderr)
        return 1
    window = match[0]
    frames = expand_snes12_rle(load_snes12_rle_seed(SLICE_DIR / f"{args.slice_id}.json"))
    body = frames[window.start : window.end]
    run_dir = next(
        (d for fr in body if (d := _run_dir(fr))),
        None,
    )

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(read_state_bytes(args.state_path))
        start = parse_env_state(env, frame=0, mode="nav")
        start_pin = _pin_brief(start)
        print(f"pin {args.state_path.name} {start_pin}", flush=True)
        print(
            f"window {window.skill} {window.start}:{window.end} "
            f"({window.end - window.start}f) dir={run_dir}",
            flush=True,
        )

        # Claim before the first act (predict-path).
        claim = (
            f"room stays {start_pin['room']}; "
            f"x does not move against {run_dir}"
        )
        print(f"claim: after 1f of {frame_button_names(body[0])}: {claim}", flush=True)

        grades: list[dict[str, object]] = []
        miss: dict[str, object] | None = None
        prev_x = int(start.samus_x)
        prev_room = int(start.room_id)
        played = 0
        last = start
        for i, fr in enumerate(body):
            env.step(action_array(fr))
            played = i + 1
            state = parse_env_state(env, frame=played, mode="nav")
            last = state
            x = int(state.samus_x)
            room = int(state.room_id)
            against = (
                (run_dir == "LEFT" and x > prev_x)
                or (run_dir == "RIGHT" and x < prev_x)
            )
            room_ok = room == prev_room
            hit = room_ok and not against
            row = {
                "i": i,
                "movie_frame": window.start + i,
                "buttons": frame_button_names(fr),
                "room": f"0x{room:04X}",
                "x": x,
                "y": int(state.samus_y),
                "pose": int(state.pose),
                "hit": hit,
            }
            grades.append(row)
            if not hit:
                why = []
                if not room_ok:
                    why.append(f"room 0x{prev_room:04X}→0x{room:04X}")
                if against:
                    why.append(f"x {prev_x}→{x} against {run_dir}")
                miss = {**row, "why": why}
                print(f"MISS f{played} {miss['why']} {row}", flush=True)
                break
            prev_x = x
            prev_room = room
        else:
            print(
                f"HIT window {played}f  start_x={start_pin['x']} end_x={last.samus_x} "
                f"room=0x{last.room_id:04X}",
                flush=True,
            )
    finally:
        env.close()

    timing = format_segment_time(played)
    report = {
        "slice": args.slice_id,
        "skill": window.skill,
        "window": {"start": window.start, "end": window.end},
        "state": str(args.state_path),
        "start": start_pin,
        "claim": claim,
        "assist": "off",
        "played": played,
        "timing": timing,
        "miss": miss,
        "final": _pin_brief(last),
        "grades": grades,
        "status": "miss" if miss else "hit",
    }
    out = args.out or (
        SCRATCH / f"tas_skill_{args.slice_id}_{args.skill}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"played={played} frames={timing['frames']} seconds={timing['seconds']} "
        f"clock={timing['clock']} status={report['status']} → {out}"
    )
    return 0 if miss is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
