"""Autobot: clear Super Mario Bros. 1-1 (flagpole).

M3 isolated — frame-perfect replay from ``Level1_1``.
M4 natural-entry — power-on boot, 1-frame settle, same seed (3/3 verified).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python smb/scripts/run_1_1.py

# Natural-entry (power-on → 1-1 clear, no checkpoint load)
uv run python smb/scripts/run_1_1.py --natural-entry --trials 3

# Plan-only (load seed, print metadata)
uv run python smb/scripts/run_1_1.py --plan-only
```

Trap: after boot readiness, **exactly 1 idle frame** is required before the
seed (settle=0 or 2 desyncs into the first pit; settle=1 and 3 clear).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from smb.menus import boot_to_level1_script
from smb.paths import GAME_DIR, GAME_V0, RECORDINGS_DIR
from smb.policy import DEFAULT_1_1_SEED, Level11ReplayPolicy
from smb.ram import (
    is_level1_ready,
    parse_game_state,
    read_snapshot,
    segment_1_1_success,
)

DEFAULT_STATE = "Level1_1"
DEFAULT_MAX_FRAMES = 4000
MIN_PROGRESS_X = 2500
STABLE_BOOT_FRAMES = 20
MIN_BOOT_FRAME = 200
# Phase-align seed with continuous boot. Verified: settle=1 and 3 clear;
# settle=0 and 2 die at first pit (~x=676).
NATURAL_SETTLE_FRAMES = 1

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
        if stable >= STABLE_BOOT_FRAMES:
            return obs, frame
    return obs, frame

def _settle(env, n: int) -> object:
    """Hold idle for ``n`` frames after boot ready (natural-entry phase align)."""
    obs = None
    idle = np.asarray(nes_idle_action(), dtype=np.int8)
    action_size = int(env.action_space.shape[0])
    if idle.shape[0] != action_size:
        idle = np.zeros(action_size, dtype=np.int8)
    for _ in range(n):
        obs, *_ = env.step(idle)
    return obs

def run_1_1(
    *,
    state_name: str = DEFAULT_STATE,
    natural_entry: bool = False,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed_path: Path = DEFAULT_1_1_SEED,
    out_dir: Path | None = None,
    save_clear: bool = True,
    tag: str = "1_1",
    natural_settle_frames: int = NATURAL_SETTLE_FRAMES,
) -> dict[str, Any]:
    """Load checkpoint (or boot), replay 1-1 seed until clear / death / timeout."""
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "segment_1_1")
    out.mkdir(parents=True, exist_ok=True)

    start_state = "NONE" if natural_entry else state_name
    if not natural_entry:
        available = get_available_states(GAME_V0, GAME_DIR)
        if state_name not in available:
            raise SystemExit(
                f"missing state {state_name!r} in {GAME_V0}; have {available[:12]}"
            )

    env = make_env(GAME_V0, start_state, GAME_DIR, render_mode="rgb_array")
    policy = Level11ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    try:
        obs, _ = reset_obs(env)
        boot_frames = 0
        settle_frames = 0
        if natural_entry:
            obs, boot_frames = _boot_to_ready(env)
            if not is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
                png = save_rgb_png(obs, out / f"{tag}_boot_fail.png")
                return {
                    "success": False,
                    "outcome": "boot_fail",
                    "boot_frames": boot_frames,
                    "screenshot": str(png),
                }
            if natural_settle_frames > 0:
                obs = _settle(env, natural_settle_frames)
                settle_frames = natural_settle_frames

        snap0 = read_snapshot(env.get_ram(), frame=0)
        start_lives = snap0.lives
        start_level_id = snap0.level_id
        max_x = snap0.player_x
        screenshots: list[str] = []
        png = save_rgb_png(obs, out / f"{tag}_0000_start.png")
        screenshots.append(png.name)

        outcome = "timeout"
        end_frame = 0
        saved: list[str] = []

        for frame in range(1, max_frames + 1):
            tick = policy.step()
            obs, *_ = env.step(tick.action)
            end_frame = frame
            ram = env.get_ram()
            snap = read_snapshot(ram, frame=frame)
            max_x = max(max_x, snap.player_x)

            if snap.lives < start_lives or snap.dying:
                outcome = "death"
                png = save_rgb_png(obs, out / f"{tag}_{frame:04d}_death.png")
                screenshots.append(png.name)
                break

            if segment_1_1_success(
                ram,
                start_lives=start_lives,
                max_player_x=max_x,
                start_level_id=start_level_id,
                min_progress_x=MIN_PROGRESS_X,
            ):
                outcome = "success"
                png = save_rgb_png(obs, out / f"{tag}_{frame:04d}_clear.png")
                screenshots.append(png.name)
                if save_clear:
                    clear_name = (
                        "Level1_1_NaturalClear" if natural_entry else "Level1_1_Clear"
                    )
                    path = save_state(env, GAME_DIR, GAME_V0, clear_name)
                    saved.append(path.name)
                break

            if frame % 500 == 0:
                png = save_rgb_png(obs, out / f"{tag}_{frame:04d}.png")
                screenshots.append(png.name)
                print(
                    f"f={frame} x={snap.player_x} max_x={max_x} "
                    f"lives={snap.lives} lid={snap.level_id} "
                    f"seed={policy.index}/{len(policy.frames)}"
                )

        ram = env.get_ram()
        snap = read_snapshot(ram, frame=end_frame)
        state = parse_game_state(ram, frame=end_frame)
        report: dict[str, Any] = {
            "success": outcome == "success",
            "outcome": outcome,
            "natural_entry": natural_entry,
            "start_state": start_state,
            "game": GAME_V0,
            "boot_frames": boot_frames,
            "settle_frames": settle_frames,
            "frames": end_frame,
            "max_player_x": max_x,
            "start_lives": start_lives,
            "start_level_id": start_level_id,
            "final": {
                "player_x": snap.player_x,
                "player_y": snap.player_y,
                "lives": snap.lives,
                "world": snap.world,
                "level": snap.level,
                "level_id": snap.level_id,
                "oper_mode": snap.oper_mode,
                "player_state": snap.player_state,
                "game_mode": state.mode.name,
            },
            "policy": policy.report(),
            "screenshots": screenshots,
            "saved_states": saved,
            "min_progress_x": MIN_PROGRESS_X,
        }
        write_json_report(out / f"{tag}_report.json", report)
        print(
            f"1-1 outcome={outcome} frames={end_frame} max_x={max_x} "
            f"final_lid={snap.level_id} natural={natural_entry} "
            f"settle={settle_frames}"
        )
        return report
    finally:
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE, help="Start state name")
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Power-on boot then clear (M4); applies 1-frame settle",
    )
    parser.add_argument(
        "--settle",
        type=int,
        default=NATURAL_SETTLE_FRAMES,
        help=f"Idle frames after natural boot ready (default {NATURAL_SETTLE_FRAMES})",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--seed", type=Path, default=DEFAULT_1_1_SEED)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.plan_only:
        policy = Level11ReplayPolicy(seed_path=args.seed)
        print(json.dumps(policy.report(), indent=2, sort_keys=True))
        raise SystemExit(0)

    results = []
    ok = 0
    for trial in range(1, args.trials + 1):
        tag = f"1_1_t{trial}" if args.trials > 1 else "1_1"
        if args.natural_entry:
            tag = f"1_1_natural_t{trial}" if args.trials > 1 else "1_1_natural"
        report = run_1_1(
            state_name=args.state,
            natural_entry=args.natural_entry,
            max_frames=args.max_frames,
            seed_path=args.seed,
            out_dir=args.out_dir,
            save_clear=not args.no_save,
            tag=tag,
            natural_settle_frames=args.settle,
        )
        results.append(report)
        if report.get("success"):
            ok += 1

    if args.trials > 1:
        print(f"trials {ok}/{args.trials} success")
    raise SystemExit(0 if ok == args.trials else 1)

if __name__ == "__main__":
    main()
