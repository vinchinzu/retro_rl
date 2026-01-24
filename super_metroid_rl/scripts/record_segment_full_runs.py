#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

import stable_retro as retro
import torch
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hierarchical_ppo import ForceMissiles, BOSS_ROOMS
from train_curriculum import (
    ActionHoldRepeat,
    DISCRETE_ACTIONS,
    DiscreteAction,
    FrameStack,
    ROUTE_SEGMENTS,
    SanitizeAction,
    TRAINING_ORDER,
)


def _load_models(device: str):
    models = {}
    for segment_name in TRAINING_ORDER:
        model_path = os.path.join(PROJECT_ROOT, "models", f"segment_{segment_name}.zip")
        if os.path.exists(model_path):
            models[segment_name] = PPO.load(model_path, device=device)

    boss_path = os.path.join(PROJECT_ROOT, "models", "boss_ppo.zip")
    boss_model = PPO.load(boss_path, device=device) if os.path.exists(boss_path) else None
    return models, boss_model


def _make_env(record_dir: str):
    env = retro.make(
        game="SuperMetroid-Snes",
        state="ZebesStart",
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        record=record_dir,
    )
    env = SanitizeAction(env)
    env = ForceMissiles(env)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = ActionHoldRepeat(env, min_hold=2, max_hold=5)
    env = FrameStack(env, n_frames=4)
    return env


def record_run(run_id: int, output_dir: str, max_steps: int, device: str) -> str:
    print(f"Starting run {run_id} (max_steps={max_steps}, device={device})", flush=True)
    models, boss_model = _load_models(device)
    if not models:
        raise RuntimeError("No segment models found. Train curriculum segments first.")
    if boss_model is None:
        raise RuntimeError("Boss model not found at models/boss_ppo.zip.")

    temp_dir = tempfile.mkdtemp(prefix=f"tmp_run_{run_id:02d}_", dir=output_dir)
    print(f"Recording to temp dir: {temp_dir}", flush=True)
    env = _make_env(temp_dir)
    obs, info = env.reset()

    phase = "descent"
    room_to_segment = {
        "descent": {
            0x91F8: "landing_site",
            0x92FD: "parlor_descent",
            0x96BA: "climb_descent",
            0x975C: "pit_room_descent",
            0x97B5: "elevator_descent",
            0x9E9F: "morph_ball_collect",
        },
        "return": {
            0x9E9F: "morph_ball_return",
            0x97B5: "elevator_return",
            0x975C: "pit_room_return",
            0x96BA: "climb_return",
            0x92FD: "parlor_to_flyway",
            0x9879: "flyway_to_torizo",
        },
    }
    frame = 0
    success = False
    boss_prev_hp = None
    in_boss = False

    while frame < max_steps:
        room_id = info.get("room_id", 0)
        items = info.get("collected_items", 0) or info.get("items", 0)
        if phase == "descent" and (items & 0x1):
            phase = "return"

        if room_id in BOSS_ROOMS:
            in_boss = True

        if in_boss:
            segment_name = "boss"
            model = boss_model
        else:
            segment_name = room_to_segment.get(phase, {}).get(room_id)
            model = models.get(segment_name) if segment_name else None

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=False)

        obs, _, terminated, truncated, info = env.step(action)
        frame += 1

        room_id = info.get("room_id", 0)
        items = info.get("collected_items", 0) or info.get("items", 0)
        if frame % 500 == 0:
            print(
                f"  Step {frame}: room=0x{room_id:04X} phase={phase} segment={segment_name}",
                flush=True,
            )

        if in_boss:
            boss_hp = info.get("boss_hp", 0) or info.get("enemy0_hp", 0)
            if boss_prev_hp is None:
                boss_prev_hp = boss_hp
            if boss_prev_hp > 0 and boss_hp == 0:
                success = True
                break
            boss_prev_hp = boss_hp

        if terminated or truncated:
            break

    env.close()
    time.sleep(2.0)

    bk2_files = sorted(glob.glob(os.path.join(temp_dir, "*.bk2")), key=os.path.getmtime)
    if not bk2_files:
        raise RuntimeError(f"No .bk2 recorded in {temp_dir}")

    src = bk2_files[-1]
    status = "success" if success else "fail"
    dst = os.path.join(output_dir, f"segment_run_{run_id:02d}_{status}.bk2")
    shutil.move(src, dst)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Run {run_id}: {status} in {frame} steps -> {dst}")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Record full-route runs using segment models")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--out-dir", default=os.path.join("recordings", "bot_runs"))
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
            device=args.device,
        )
        if bk2_path.endswith("_success.bk2"):
            successes += 1

    print(f"Completed: {successes} successes in {attempts} attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
