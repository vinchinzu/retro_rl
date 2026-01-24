#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hierarchical_ppo import GamePhase, HierarchicalModelSelector, make_hierarchical_env


def record_run(
    run_id: int,
    output_dir: str,
    max_steps: int,
    nav_model_path: str,
    device: str,
) -> str:
    print(f"Starting run {run_id} (max_steps={max_steps}, device={device})", flush=True)
    selector = HierarchicalModelSelector(device=device)
    if nav_model_path:
        selector.load_nav_model(nav_model_path)

    temp_dir = tempfile.mkdtemp(prefix=f"tmp_run_{run_id:02d}_", dir=output_dir)
    print(f"Recording to temp dir: {temp_dir}", flush=True)
    env = make_hierarchical_env(
        state="ZebesStart",
        phase=GamePhase.DESCENT,
        max_steps=max_steps,
        render_mode="rgb_array",
        record_dir=temp_dir,
    )

    obs, info = env.reset()
    frame = 0
    success = False

    while True:
        action = selector.get_action(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        frame += 1

        if frame % 500 == 0:
            room_id = info.get("room_id", 0)
            phase = info.get("phase", "UNKNOWN")
            print(f"  Step {frame}: room=0x{room_id:04X} phase={phase}", flush=True)

        if terminated or truncated:
            success = info.get("phase") == "COMPLETE"
            break

    env.close()
    time.sleep(2.0)

    bk2_files = sorted(glob.glob(os.path.join(temp_dir, "*.bk2")), key=os.path.getmtime)
    if not bk2_files:
        raise RuntimeError(f"No .bk2 recorded in {temp_dir}")

    src = bk2_files[-1]
    status = "success" if success else "fail"
    dst = os.path.join(output_dir, f"bot_run_{run_id:02d}_{status}.bk2")
    shutil.move(src, dst)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Run {run_id}: {status} in {frame} steps -> {dst}")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Record hierarchical bot runs to .bk2")
    parser.add_argument("--runs", type=int, default=3, help="Number of successful runs to keep")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--out-dir", default=os.path.join("recordings", "bot_runs"))
    parser.add_argument("--nav-model", default=os.path.join("models", "nav_ppo.zip"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    os.makedirs(args.out_dir, exist_ok=True)

    successes = 0
    attempts = 0

    while successes < args.runs and attempts < args.max_attempts:
        attempts += 1
        bk2_path = record_run(
            run_id=attempts,
            output_dir=args.out_dir,
            max_steps=args.max_steps,
            nav_model_path=args.nav_model,
            device=args.device,
        )
        if bk2_path.endswith("_success.bk2"):
            successes += 1

    print(f"Completed: {successes} successes in {attempts} attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
