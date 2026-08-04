"""Autobot: reactive 1-2 secret warp → World 4.

Natural-entry only — no mid-1-2 state splice. Starts from a 1-1 predecessor
(stairs-improved continuous prefix or baseline), waits for control gates,
then runs the state-gated :class:`smb.reactive_12.Reactive12Policy`.

```bash
# After stairs 1-1 fragment (Level1_1 settle=14)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3

# Baseline continuous 1-1 prefix
uv run python -m smb.scripts.run_1_2 --predecessor baseline --trials 3
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from retro_harness.env import make_env
from smb.full_run import read_state_bytes
from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_CONTINUOUS_SEED,
    expand_nes9_rle,
    load_nes9_rle_seed,
    frames_to_actions,
)
from smb.ram import read_snapshot
from smb.reactive_12 import Reactive12Policy, play_reactive_12
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report

LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
STAIRS_1_1 = GAME_DIR / "models" / "smb_1_1_stairs_best_frames.json"
Predecessor = Literal["stairs", "baseline"]


def _play_1_1_until_clear(env, seed_frames: list[list[int]]) -> dict[str, Any]:
    """Replay a 1-1 seed until level becomes 1-2 (or death)."""
    idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
    start = read_snapshot(env.get_ram())
    lives0 = start.lives
    recorded: list[list[int]] = []
    for i, act in enumerate(frames_to_actions(seed_frames), start=1):
        env.step(act)
        recorded.append([int(b) for b in act[:9]])
        snap = read_snapshot(env.get_ram(), frame=i)
        if snap.lives < lives0 or snap.dying:
            return {
                "success": False,
                "outcome": "death",
                "frames": i,
                "recorded": recorded,
                "final": snap,
            }
        if snap.world == 0 and snap.level == 1:
            return {
                "success": True,
                "outcome": "clear",
                "frames": i,
                "recorded": recorded,
                "final": snap,
            }
    snap = read_snapshot(env.get_ram())
    return {
        "success": False,
        "outcome": "timeout",
        "frames": len(recorded),
        "recorded": recorded,
        "final": snap,
    }


def run_1_2_natural(
    *,
    predecessor: Predecessor = "stairs",
    settle: int = CONTINUOUS_SETTLE_FRAMES,
    max_frames_12: int = 4000,
    out_dir: Path | None = None,
    tag: str = "1_2_reactive",
    save_recording: bool = True,
) -> dict[str, Any]:
    """Level1_1 → 1-1 seed → reactive 1-2 → World 4 (controller only)."""
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "segment_1_2")
    out.mkdir(parents=True, exist_ok=True)

    if predecessor == "stairs":
        seed_path = STAIRS_1_1
        if not seed_path.exists():
            raise SystemExit(f"missing stairs 1-1 seed: {seed_path}")
        seed_11 = expand_nes9_rle(load_nes9_rle_seed(seed_path))
    else:
        seed_11 = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_CONTINUOUS_SEED))

    if not LEVEL1_1_STATE.exists():
        raise SystemExit(f"missing {LEVEL1_1_STATE}")

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    report: dict[str, Any] = {
        "mode": "natural_1_2",
        "predecessor": predecessor,
        "settle_frames": settle,
        "success": False,
    }
    try:
        env.reset()
        env.em.set_state(read_state_bytes(LEVEL1_1_STATE))
        idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
        for _ in range(settle):
            env.step(idle)

        stage_11 = _play_1_1_until_clear(env, seed_11)
        report["stages"] = {
            "1-1": {
                "success": stage_11["success"],
                "outcome": stage_11["outcome"],
                "frames": stage_11["frames"],
            }
        }
        if not stage_11["success"]:
            report["outcome"] = f"1-1_{stage_11['outcome']}"
            return report

        policy = Reactive12Policy(action_size=int(env.action_space.shape[0]))
        stage_12 = play_reactive_12(env, policy=policy, max_frames=max_frames_12)
        report["stages"]["1-2"] = {
            k: v for k, v in stage_12.items() if k not in ("recorded", "last_obs")
        }
        report["success"] = bool(stage_12["success"])
        report["outcome"] = stage_12["outcome"]
        report["total_frames"] = stage_11["frames"] + stage_12["frames"]
        report["world4"] = stage_12.get("final")

        if save_recording and stage_12.get("recorded"):
            rec_path = out / f"{tag}_recorded.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "format": "nes9_frames",
                        "predecessor": predecessor,
                        "frames_1_1": stage_11["frames"],
                        "frames_1_2": stage_12["frames"],
                        "success": stage_12["success"],
                        "1_2_buttons": stage_12["recorded"],
                        "policy": stage_12["policy"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            report["recording"] = str(rec_path)

        obs = stage_12.get("last_obs")
        if obs is not None:
            suffix = "w4" if stage_12["success"] else "fail"
            png = save_rgb_png(obs, out / f"{tag}_{suffix}.png")
            report["screenshot"] = str(png)
        return report
    finally:
        write_json_report(out / f"{tag}_report.json", report)
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predecessor",
        choices=("stairs", "baseline"),
        default="stairs",
        help="1-1 seed: stairs-improved or continuous baseline prefix",
    )
    p.add_argument("--settle", type=int, default=CONTINUOUS_SETTLE_FRAMES)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-frames-12", type=int, default=4000)
    p.add_argument("--tag", default="1_2_reactive")
    args = p.parse_args()

    results = []
    for t in range(1, args.trials + 1):
        tag = f"{args.tag}_t{t}" if args.trials > 1 else args.tag
        report = run_1_2_natural(
            predecessor=args.predecessor,
            settle=args.settle,
            max_frames_12=args.max_frames_12,
            tag=tag,
        )
        results.append(report)
        print(
            f"trial={t} success={report['success']} outcome={report.get('outcome')} "
            f"total_f={report.get('total_frames')} "
            f"1-1={report.get('stages', {}).get('1-1', {}).get('frames')} "
            f"1-2={report.get('stages', {}).get('1-2', {}).get('frames')}"
        )

    if args.trials > 1:
        ok = sum(1 for r in results if r.get("success"))
        summary = {
            "trials": args.trials,
            "successes": ok,
            "predecessor": args.predecessor,
            "results": [
                {
                    "success": r.get("success"),
                    "outcome": r.get("outcome"),
                    "total_frames": r.get("total_frames"),
                }
                for r in results
            ],
        }
        out = RECORDINGS_DIR / "segment_1_2"
        write_json_report(out / f"{args.tag}_trials_report.json", summary)
        print(f"trials {ok}/{args.trials} → {out / (args.tag + '_trials_report.json')}")


if __name__ == "__main__":
    main()
