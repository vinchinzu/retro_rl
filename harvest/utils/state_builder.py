#!/usr/bin/env python3
import os
import json
import gzip
import argparse
from typing import List, Optional, Tuple

import numpy as np
import stable_retro as retro

import harvest_bot as hb
from task_recorder import Task


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")


def make_env(state: Optional[str] = None):
    kwargs = {
        "game": "HarvestMoon-Snes",
        "inttype": retro.data.Integrations.ALL,
        "use_restricted_actions": retro.Actions.ALL,
        "render_mode": "rgb_array",
    }
    if state:
        kwargs["state"] = state
    return retro.make(**kwargs)


def save_state_bytes(env, name: str):
    data = env.em.get_state()
    path = os.path.join(hb.STATES_DIR, f"{name}.state")
    with gzip.open(path, "wb") as f:
        f.write(data)
    print(f"[STATE] Saved {name} -> {path}")


def run_task(env, task: Task, stop_frame: Optional[int] = None):
    for i, frame in enumerate(task.frames):
        if stop_frame is not None and i >= stop_frame:
            break
        action = np.array(frame, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def find_liftable_nearby(ram: np.ndarray, center: hb.Point, radius_tiles: int) -> Optional[Tuple[int, int]]:
    tf = hb.TargetFinder()
    cx = center.x // tf.TILE_SIZE
    cy = center.y // tf.TILE_SIZE
    for dy in range(-radius_tiles, radius_tiles + 1):
        for dx in range(-radius_tiles, radius_tiles + 1):
            tx = cx + dx
            ty = cy + dy
            if tx < 0 or ty < 0 or tx >= tf.MAP_WIDTH or ty >= tf.MAP_WIDTH:
                continue
            tile_id = tf.get_tile_at(ram, tx * tf.TILE_SIZE + 8, ty * tf.TILE_SIZE + 8)
            if tile_id in hb.LIFTABLE_TILE_IDS:
                return (tx, ty)
    return None


def load_task(name: str) -> Task:
    path = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Task.load(path)


def main():
    parser = argparse.ArgumentParser(description="Build save states via tasks")
    parser.add_argument("--base", type=str, default="Y1_Spring_Day01_06h", help="Base state name")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated tasks to run")
    parser.add_argument("--save", type=str, required=True, help="Save state name")
    parser.add_argument("--stop-frame", type=int, default=None, help="Stop task at frame index")
    parser.add_argument("--find-liftable", action="store_true", help="Save when a liftable tile is near the player")
    parser.add_argument(
        "--find-liftable-after",
        action="store_true",
        help="Save after tasks if a liftable tile is near the player",
    )
    parser.add_argument("--radius", type=int, default=2, help="Tile radius for liftable scan")
    args = parser.parse_args()

    env = make_env(args.base)
    obs, info = env.reset()

    if args.tasks:
        for task_name in args.tasks.split(","):
            task_name = task_name.strip()
            if not task_name:
                continue
            task = load_task(task_name)
            if args.find_liftable:
                for i, frame in enumerate(task.frames):
                    action = np.array(frame, dtype=np.int32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    ram = env.get_ram()
                    pos = hb.Navigator()
                    pos.update(env)
                    found = find_liftable_nearby(ram, pos.current_pos, args.radius)
                    if found:
                        save_state_bytes(env, args.save)
                        env.close()
                        return
                continue
            run_task(env, task, stop_frame=args.stop_frame)

    if args.find_liftable_after:
        ram = env.get_ram()
        pos = hb.Navigator()
        pos.update(env)
        found = find_liftable_nearby(ram, pos.current_pos, args.radius)
        if found:
            save_state_bytes(env, args.save)
            env.close()
            return
        print("[STATE] No liftable found after tasks; saving anyway.")

    save_state_bytes(env, args.save)
    env.close()


if __name__ == "__main__":
    main()
