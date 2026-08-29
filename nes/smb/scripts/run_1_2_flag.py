#!/usr/bin/env python3
"""Record / verify the 1-2 flag body (HL UG prefix + lift/pipe tail → 1-3).

Does not touch the warp any% line. Predecessor is HappyLee 1-1; the 1-2 body
is FM2 until the end-of-UG lift, then ``FlagTailController``.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_2_flag --record
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_2_flag --trials 2
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.flag_12 import (
    DEFAULT_1_2_FLAG_SEED,
    DEFAULT_MAX_FRAMES,
    Flag12Policy,
    FlagTailController,
    is_lift_pose,
    play_flag_12,
)
from smb.paths import RECORDINGS_DIR
from smb.flag_12 import is_ceiling
from smb.ram import player_on_ground, read_snapshot
from smb.tas.chain import reach_surface_after_hl_1_1
from smb.tas.fm2 import frames_to_nes9_rle_payload, parse_fm2
from smb.tas.replay import get_state, make_level1_env, set_state, to_action9
from smb.tas.stages import DEFAULT_FM2, HL_1_2_FM2_START, is_1_3_control, snap_fingerprint

OUT_DIR = RECORDINGS_DIR / "segments_all_exits"
REPORT_PATH = OUT_DIR / "run_1_2_flag_report.json"
SURFACE_CACHE = OUT_DIR / "hl_1_2_surface_control.state"


def log(*parts: object) -> None:
    print(*parts, flush=True)


def _buttons(action: Any) -> list[int]:
    return [int(b) for b in list(action)[:9]]


def record_flag_body(env: Any, *, fm2_path: Path = DEFAULT_FM2) -> dict[str, Any]:
    """From 1-2 surface control: HL FM2 to the lift, then the flag tail."""
    t0 = time.time()
    fm2 = parse_fm2(fm2_path).frames
    body = fm2[HL_1_2_FM2_START:]
    recorded: list[list[int]] = []
    lift_state: Any | None = None
    lift_n = 0
    lift_snap: dict[str, Any] | None = None
    snap = read_snapshot(env.get_ram())
    for i, frame in enumerate(body):
        buttons = list(frame[:9])
        env.step(to_action9(buttons))
        recorded.append(buttons)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if player_on_ground(ram) and is_lift_pose(snap):
            lift_state = get_state(env)
            lift_n = len(recorded)
            lift_snap = {
                "fm2_i": i + 1,
                "x": int(snap.player_x),
                "y": int(snap.player_y),
                "xs": int(snap.x_speed),
                "ps": int(snap.player_state),
                "motion": int(ram[0x001D]),
            }
        if is_ceiling(snap) and int(snap.player_x) >= 2200:
            log(f"HL left floor f{i + 1} x={snap.player_x} y={snap.player_y}")
            break
        if i + 1 >= 1800:
            break
    if lift_state is None or lift_n <= 0:
        raise RuntimeError("HL 1-2 never stood on the end-of-UG lift")
    set_state(env, lift_state)
    recorded = recorded[:lift_n]
    log(f"lift cut @{lift_n} {lift_snap}")

    tail = FlagTailController()
    tail_frames = 0
    last_phase = tail.phase.name
    for _ in range(DEFAULT_MAX_FRAMES):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if is_1_3_control(snap):
            break
        tick = tail.step(snap, on_ground=player_on_ground(ram))
        if tail.phase.name != last_phase:
            log(
                f"  tail {tail.phase.name} f{tail_frames} "
                f"x={snap.player_x} y={snap.player_y} ps={snap.player_state} "
                f"area={snap.area_pointer}"
            )
            last_phase = tail.phase.name
        env.step(tick.action)
        recorded.append(_buttons(tick.action))
        tail_frames += 1
        snap = read_snapshot(env.get_ram())
        if is_1_3_control(snap) or tail.done:
            break
    ok = is_1_3_control(snap)
    log(
        f"tail {tail_frames}f phase={tail.phase.name} "
        f"dash={snap.dash_level} x={snap.player_x} ps={snap.player_state} ok={ok}"
    )
    payload = frames_to_nes9_rle_payload(
        recorded,
        route_id="smb_1_2_flag",
        source="HappyLee 1-2 UG prefix + lift A19 / pipe walk / outdoor flag",
        extra={
            "level_id": "smb_1_2_flag",
            "start_state": "1-2_surface_control_after_happylee_1_1",
            "target": "1-3_control",
            "lift": lift_snap,
            "prefix_frames": lift_n,
            "tail_frames": tail_frames,
            "verified_completed": ok,
            "note": (
                "Do not sanitize L+R. Flag pipe is the short DOWN pipe after "
                "the lifts, not plant pipes B/C and not the warp room."
            ),
        },
    )
    return {
        "ok": ok,
        "num_frames": len(recorded),
        "prefix_frames": lift_n,
        "tail_frames": tail_frames,
        "lift": lift_snap,
        "final": snap_fingerprint(snap),
        "elapsed_s": round(time.time() - t0, 1),
        "payload": payload,
        "frames": recorded,
    }


def write_seed(payload: dict[str, Any], path: Path = DEFAULT_1_2_FLAG_SEED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path} frames={payload.get('num_frames')}")
    return path


def boot_to_surface(env: Any) -> dict[str, Any]:
    leave, wait, ctrl = reach_surface_after_hl_1_1(env)
    SURFACE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SURFACE_CACHE.write_bytes(get_state(env))
    log(f"surface leave_1_1={leave} wait={wait} {snap_fingerprint(ctrl)}")
    return {"leave_1_1": leave, "ctrl_wait": wait, "snap": snap_fingerprint(ctrl)}


def verify_from_surface(
    env: Any,
    *,
    seed_path: Path = DEFAULT_1_2_FLAG_SEED,
    trials: int = 1,
) -> dict[str, Any]:
    surface = get_state(env)
    rows: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        set_state(env, surface)
        policy = Flag12Policy(seed_path=seed_path)
        report = play_flag_12(env, policy=policy)
        log(
            f"trial {trial}/{trials} ok={report['ok']} "
            f"outcome={report['outcome']} f{report['frames']} {report['snap']}"
        )
        rows.append({k: report[k] for k in ("ok", "outcome", "frames", "max_x", "snap")})
        if not report["ok"]:
            break
    return {
        "ok": bool(rows) and all(row["ok"] for row in rows),
        "trials": rows,
        "seed": str(seed_path),
        "seed_frames": len(policy.frames) if rows else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="capture HL prefix + tail and write models/smb_1_2_flag.json",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=Path, default=DEFAULT_1_2_FLAG_SEED)
    args = parser.parse_args(argv)

    t0 = time.time()
    env = make_level1_env()
    report: dict[str, Any] = {"ok": False}
    try:
        report["boot"] = boot_to_surface(env)
        surface = get_state(env)
        if args.record or not Path(args.seed).is_file():
            captured = record_flag_body(env)
            write_seed(captured["payload"], args.seed)
            report["record"] = {k: captured[k] for k in captured if k not in {"payload", "frames"}}
            if not captured["ok"]:
                report["elapsed_s"] = round(time.time() - t0, 1)
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                log(f"record failed {REPORT_PATH}")
                return 1
            set_state(env, surface)
        report["verify"] = verify_from_surface(env, seed_path=args.seed, trials=args.trials)
        report["ok"] = bool(report["verify"].get("ok"))
    finally:
        env.close()
    report["elapsed_s"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"report {REPORT_PATH} ok={report['ok']}")
    print(json.dumps({k: report[k] for k in ("ok", "elapsed_s") if k in report}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
