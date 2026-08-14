"""Autobot: 1-1 → 1-2 secret warp → World 4.

Two verified modes:

1. **``segment-12``** — mid-1-2 warp segment only (``Level1_2_WarpMid`` via
   ``em.set_state`` + ``smb_1_2_warp_w4`` seed). 3/3 → World 4.
2. **``chain``** (default) — power-on natural 1-1 clear, then load the mid-1-2
   warp state and finish the secret exit to World 4.

True continuous 1-1→1-2 without the mid-level reload is still in progress
(frame-perfect 1-2 seeds desync from natural post-1-1 entry).

```bash
# Power-on → 1-1 → mid-1-2 warp → World 4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python smb/scripts/run_warp_chain.py --trials 3

# Mid-1-2 segment only
uv run python smb/scripts/run_warp_chain.py --mode segment-12 --trials 3
```

Trap: ``make_env(..., state='Level1_2_WarpMid')`` does **not** replay cleanly
for this mid-transition state; always use ``em.set_state`` after reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from smb.full_run import read_state_bytes
from smb.menus import boot_to_level1_script
from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.policy import (
    DEFAULT_1_1_SEED,
    DEFAULT_1_2_WARP_SEED,
    Level11ReplayPolicy,
    Nes9ReplayPolicy,
)
from smb.ram import (
    is_dying,
    is_level1_ready,
    parse_game_state,
    read_snapshot,
    reached_world_4,
    segment_1_1_success,
    segment_1_2_warp_success,
)

WARP_MID_STATE = INTEGRATION_V0_DIR / "Level1_2_WarpMid.state"
NATURAL_SETTLE = 1
BOOT_STABLE = 20
MIN_BOOT_FRAME = 200
DEFAULT_MAX_FRAMES_11 = 4000
DEFAULT_MAX_FRAMES_12 = 4000

def _boot_to_ready(env) -> tuple[object, int]:
    frame = 0
    obs = None
    stable = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        mean = float(obs.mean())
        if frame >= MIN_BOOT_FRAME and is_level1_ready(env.get_ram(), obs_mean=mean):
            stable += 1
        else:
            stable = 0
        if stable >= BOOT_STABLE:
            return obs, frame
    return obs, frame

def _idle(env, n: int) -> object:
    obs = None
    action = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
    for _ in range(n):
        obs, *_ = env.step(action)
    return obs

def _load_warp_mid(env) -> None:
    if not WARP_MID_STATE.exists():
        raise SystemExit(
            f"missing {WARP_MID_STATE}; copy playthrough mid-1-2 state first"
        )
    env.em.set_state(read_state_bytes(WARP_MID_STATE))

def run_1_2_warp_segment(
    env,
    *,
    seed_path: Path = DEFAULT_1_2_WARP_SEED,
    max_frames: int = DEFAULT_MAX_FRAMES_12,
) -> dict[str, Any]:
    """Replay 1-2 warp seed from current env state until World 4 / death / timeout."""
    policy = Nes9ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    snap0 = read_snapshot(env.get_ram())
    start_lives = snap0.lives
    max_x = snap0.player_x
    outcome = "timeout"
    end_frame = 0
    obs = None

    for frame in range(1, max_frames + 1):
        tick = policy.step()
        obs, *_ = env.step(tick.action)
        end_frame = frame
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=frame)
        max_x = max(max_x, snap.player_x)

        if snap.lives < start_lives or snap.dying:
            outcome = "death"
            break
        if segment_1_2_warp_success(ram, start_lives=start_lives):
            outcome = "success"
            break

    ram = env.get_ram()
    snap = read_snapshot(ram, frame=end_frame)
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "frames": end_frame,
        "max_player_x": max_x,
        "final": {
            "player_x": snap.player_x,
            "world": snap.world,
            "level": snap.level,
            "level_id": snap.level_id,
            "lives": snap.lives,
            "player_state": snap.player_state,
        },
        "policy": policy.report(),
        "last_obs": obs,
    }

def run_warp_chain(
    *,
    mode: str = "chain",
    max_frames_11: int = DEFAULT_MAX_FRAMES_11,
    max_frames_12: int = DEFAULT_MAX_FRAMES_12,
    seed_11: Path = DEFAULT_1_1_SEED,
    seed_12: Path = DEFAULT_1_2_WARP_SEED,
    out_dir: Path | None = None,
    save_clear: bool = True,
    tag: str = "warp_w4",
) -> dict[str, Any]:
    """Run segment-12 or natural-1-1 + mid-1-2 warp chain to World 4."""
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "warp_chain")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    report: dict[str, Any] = {
        "mode": mode,
        "success": False,
        "stages": {},
    }
    try:
        obs, _ = reset_obs(env)
        boot_frames = 0
        frames_11 = 0

        if mode == "chain":
            obs, boot_frames = _boot_to_ready(env)
            if not is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
                report["outcome"] = "boot_fail"
                report["boot_frames"] = boot_frames
                return report
            obs = _idle(env, NATURAL_SETTLE)

            policy11 = Level11ReplayPolicy(
                seed_path=seed_11,
                action_size=int(env.action_space.shape[0]),
            )
            snap0 = read_snapshot(env.get_ram())
            lives0 = snap0.lives
            max_x = snap0.player_x
            outcome_11 = "timeout"
            for frame in range(1, max_frames_11 + 1):
                tick = policy11.step()
                obs, *_ = env.step(tick.action)
                frames_11 = frame
                ram = env.get_ram()
                snap = read_snapshot(ram, frame=frame)
                max_x = max(max_x, snap.player_x)
                if snap.lives < lives0 or snap.dying:
                    outcome_11 = "death"
                    break
                if segment_1_1_success(
                    ram, start_lives=lives0, max_player_x=max_x
                ):
                    outcome_11 = "success"
                    break
            report["stages"]["1-1"] = {
                "success": outcome_11 == "success",
                "outcome": outcome_11,
                "frames": frames_11,
                "max_player_x": max_x,
                "boot_frames": boot_frames,
                "settle": NATURAL_SETTLE,
            }
            if outcome_11 != "success":
                png = save_rgb_png(obs, out / f"{tag}_1_1_fail.png")
                report["outcome"] = f"1-1_{outcome_11}"
                report["screenshot"] = str(png)
                write_json_report(out / f"{tag}_report.json", report)
                print(f"chain fail at 1-1: {outcome_11}")
                return report
            png = save_rgb_png(obs, out / f"{tag}_1_1_clear.png")
            report["stages"]["1-1"]["screenshot"] = png.name

            # Mid-1-2 warp segment (disclosed reload — not pure continuous M5).
            _load_warp_mid(env)
            report["stages"]["1-2_entry"] = {
                "method": "set_state",
                "state": str(WARP_MID_STATE),
                "note": "mid-1-2 warp approach; continuous full 1-2 still WIP",
            }
        elif mode == "segment-12":
            _load_warp_mid(env)
            report["stages"]["1-2_entry"] = {
                "method": "set_state",
                "state": str(WARP_MID_STATE),
            }
        else:
            raise SystemExit(f"unknown mode {mode!r}")

        seg12 = run_1_2_warp_segment(
            env, seed_path=seed_12, max_frames=max_frames_12
        )
        obs = seg12.pop("last_obs", obs)
        report["stages"]["1-2_warp"] = {
            k: v for k, v in seg12.items() if k != "policy"
        }
        report["stages"]["1-2_warp"]["policy"] = seg12.get("policy")

        if seg12["success"]:
            report["success"] = True
            report["outcome"] = "world_4"
            if obs is not None:
                png = save_rgb_png(obs, out / f"{tag}_w4.png")
                report["screenshot"] = str(png)
            if save_clear:
                path = save_state(env, GAME_DIR, GAME_V0, "World4_WarpEntry")
                report["saved_state"] = path.name
        else:
            report["outcome"] = f"1-2_{seg12['outcome']}"
            if obs is not None:
                png = save_rgb_png(obs, out / f"{tag}_1_2_fail.png")
                report["screenshot"] = str(png)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        report["final"] = {
            "world": snap.world,
            "level": snap.level,
            "level_id": snap.level_id,
            "player_x": snap.player_x,
            "reached_world_4": reached_world_4(ram),
            "game_mode": parse_game_state(ram).mode.name,
        }
        write_json_report(out / f"{tag}_report.json", report)
        print(
            f"warp_chain mode={mode} outcome={report['outcome']} "
            f"world={snap.world} level={snap.level} "
            f"1-2_frames={seg12['frames']} max_x={seg12['max_player_x']}"
        )
        return report
    finally:
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("chain", "segment-12"),
        default="chain",
        help="chain=natural 1-1 then mid-1-2 warp; segment-12=warp only",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed-11", type=Path, default=DEFAULT_1_1_SEED)
    parser.add_argument("--seed-12", type=Path, default=DEFAULT_1_2_WARP_SEED)
    args = parser.parse_args()

    ok = 0
    for trial in range(1, args.trials + 1):
        tag = f"warp_w4_t{trial}" if args.trials > 1 else "warp_w4"
        report = run_warp_chain(
            mode=args.mode,
            seed_11=args.seed_11,
            seed_12=args.seed_12,
            out_dir=args.out_dir,
            save_clear=not args.no_save,
            tag=tag,
        )
        if report.get("success"):
            ok += 1
    if args.trials > 1:
        print(f"trials {ok}/{args.trials} success")
    raise SystemExit(0 if ok == args.trials else 1)

if __name__ == "__main__":
    main()
