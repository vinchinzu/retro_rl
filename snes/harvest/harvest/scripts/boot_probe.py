#!/usr/bin/env python3
"""Probe Harvest Moon morning readiness from a pinned save state or power-on.

Harvest Moon's title/new-game intro is long.  A named morning state remains a
useful M1/M2 fixture probe, while ``--power-on`` drives title → new diary →
opening with ordinary input and confirms the true Spring day-1 handoff without
loading any state.

Examples:

    uv run python -m harvest.scripts.boot_probe
    uv run python -m harvest.scripts.boot_probe --state Y1_After_Sleep
    uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House --steps 30
    HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

from harvest.paths import GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from retro_harness import TaskStatus, WorldState

from harvest.runtime.power_on import PowerOnStartTask
from harvest.runtime.retro_setup import make_harvest_env


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _save_rgb_png(obs, path: Path) -> Path | None:
    """Best-effort screenshot; skip if Pillow is not installed."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(obs)).save(path)
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state",
        default="Y1_Inside_House",
        help="Save state name under custom_integrations/HarvestMoon-Snes/",
    )
    p.add_argument(
        "--power-on",
        action="store_true",
        help="Use a clean emulator boot instead of loading --state.",
    )
    p.add_argument("--steps", type=int, default=20, help="Idle frames after load")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "boot_probe.json",
        help="JSON report path",
    )
    p.add_argument(
        "--screenshot",
        type=Path,
        default=PROJECT_DIR / "recordings" / "boot_probe.png",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_headless()

    state_path = GAME_DIR / f"{args.state}.state"
    if not args.power_on and not state_path.is_file():
        print(f"STATE missing: {state_path}")
        return 2

    env = make_harvest_env(None if args.power_on else args.state, render_mode="rgb_array")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        power_on: dict[str, object] | None = None
        if args.power_on:
            task = PowerOnStartTask()
            frame = 0
            task.reset(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            while frame < task.timeout:
                world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
                result = task.step(world)
                if result.status == TaskStatus.SUCCESS:
                    power_on = task.summary(world)
                    break
                if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    power_on = task.summary(world)
                    power_on["failure"] = result.reason or result.status.value
                    break
                action = result.action.action if result.action is not None else env.action_space.sample() * 0
                obs = env.step(action)[0]
                frame += 1
            else:
                world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
                power_on = task.summary(world)
                power_on["failure"] = "power-on frame budget exhausted"
        else:
            idle = env.action_space.sample() * 0
            for _ in range(max(0, args.steps)):
                step = env.step(idle)
                obs = step[0]

        ram = env.get_ram()
        scene = classify_scene_from_ram(ram)
        fields = {
            key: int(read_ram_value(ram, key))
            for key in (
                "year",
                "season",
                "day",
                "weekday",
                "hour",
                "minute",
                "tilemap",
                "money",
                "stamina",
                "input_lock",
            )
        }
        hour = fields["hour"]
        ready = morning_scene_ready(scene, hour) or (
            scene.is_normal_map and fields["input_lock"] == 1
        )
        png = _save_rgb_png(obs, args.screenshot)
        report = {
            "state": None if args.power_on else args.state,
            "power_on": power_on,
            "ready": bool(ready),
            "scene": scene.to_dict(),
            "fields": fields,
            "screenshot": str(png) if png else None,
            "clean_run": {
                "initial_state_loads": 0 if args.power_on else 1,
                "mid_run_state_loads": 0,
                "ram_writes": 0,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"BOOT state={'power_on' if args.power_on else args.state} ready={ready} "
            f"day={fields['day']} hour={fields['hour']:02d}:{fields['minute']:02d} "
            f"tilemap=0x{fields['tilemap']:02X} scene={scene.summary()} "
            f"report={args.out}"
        )
        power_on_ok = not args.power_on or bool(power_on and "failure" not in power_on)
        return 0 if ready and power_on_ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
