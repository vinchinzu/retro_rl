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

from train_curriculum import (
    ActionHoldRepeat,
    DISCRETE_ACTIONS,
    DiscreteAction,
    FrameStack,
    ROUTE_SEGMENTS,
    SanitizeAction,
)
from gymnasium.wrappers import TimeLimit


def make_env(segment_name: str, record_dir: str):
    segment = ROUTE_SEGMENTS[segment_name]
    env = retro.make(
        game="SuperMetroid-Snes",
        state=segment.start_state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        record=record_dir,
    )
    env = SanitizeAction(env)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = ActionHoldRepeat(env, min_hold=2, max_hold=5)
    env = FrameStack(env, n_frames=4)
    env = TimeLimit(env, max_episode_steps=segment.max_steps)
    return env


def record_episode(segment_name: str, out_dir: str, device: str, max_steps: int, idx: int) -> str:
    model_path = os.path.join(PROJECT_ROOT, "models", f"segment_{segment_name}.zip")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing model: {model_path}")

    model = PPO.load(model_path, device=torch.device(device))

    temp_dir = tempfile.mkdtemp(prefix=f"tmp_{segment_name}_{idx:02d}_", dir=out_dir)
    env = make_env(segment_name, temp_dir)
    obs, info = env.reset()
    steps = 0

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    # Wait for recorder to finalize file
    bk2_files = []
    for _ in range(40):
        time.sleep(1.0)
        bk2_files = sorted(glob.glob(os.path.join(temp_dir, "*.bk2")), key=os.path.getmtime)
        if bk2_files:
            break

    if not bk2_files:
        print(f"Warning: no .bk2 found yet in {temp_dir}, leaving for recovery.", flush=True)
        return None
    src = bk2_files[-1]
    dst = os.path.join(out_dir, f"{segment_name}_episode_{idx:02d}.bk2")
    shutil.move(src, dst)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Saved {dst} ({steps} steps)")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Record segment episodes to bk2")
    parser.add_argument("--segment", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", default=os.path.join("recordings", "segment_debug"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(1, args.episodes + 1):
        record_episode(args.segment, args.out_dir, args.device, args.max_steps, i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
