#!/usr/bin/env python3
"""Clear weeds/stones inside the west plant pocket (no plant tape).

    HEADLESS=1 uv run python -m harvest.scripts.pocket_clear_probe
    HEADLESS=1 uv run python -m harvest.scripts.pocket_clear_probe \\
      --out recordings/pocket_clear_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP, CLEARABLE_DEBRIS_TYPES
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram, make_action


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument("--timeout", type=int, default=9_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "pocket_clear_probe.json",
    )
    return p.parse_args()


def _debris(ram) -> list[tuple[int, int, str]]:
    return [
        (t.tile[0], t.tile[1], t.debris_type.name)
        for t in TileScanner().scan(
            ram, WEST_PLANT_POCKET_BOUNDS, types=set(CLEARABLE_DEBRIS_TYPES)
        )
    ]


def _run_task(env, task, *, timeout: int, start_frame: int) -> tuple[int, object, object]:
    obs = None
    result = None
    frame = start_frame
    for frame in range(start_frame, start_frame + timeout + 1):
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = task.step(world)
        if result.status != TaskStatus.RUNNING:
            break
        action = result.action.action if result.action is not None else make_action()
        obs, _reward, _term, _trunc, _info = env.step(action)
    return frame, result, env.get_ram()


def main() -> int:
    _configure_headless()
    args = _parse_args()
    env = make_harvest_env(state=args.state)
    try:
        env.reset()
        ram = env.get_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        frame = 0
        exit_reason = None
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            exit_task = ExitToFarmTask()
            exit_task.reset(world)
            frame, exit_result, ram = _run_task(
                env, exit_task, timeout=2_000, start_frame=0
            )
            exit_reason = exit_result.reason if exit_result is not None else None
        before = _debris(ram)
        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        task = FarmClearTask(
            timeout=args.timeout,
            fetch_tools=False,
            prefer_lift_for_weeds=True,
            prefer_lift_for_stones=True,
            farm_bounds=WEST_PLANT_POCKET_BOUNDS,
        )
        task.reset(world)
        end_frame, result, ram = _run_task(
            env, task, timeout=args.timeout, start_frame=frame
        )
        after = _debris(ram)
        pos = get_pos_from_ram(ram)
        payload = {
            "status": result.status.value if result is not None else "none",
            "reason": result.reason if result is not None else "",
            "frames": end_frame,
            "exit_reason": exit_reason,
            "bounds": list(WEST_PLANT_POCKET_BOUNDS),
            "debris_before": before,
            "debris_after": after,
            "cleared": max(0, len(before) - len(after)),
            "remaining": len(after),
            "end_tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
            "end_pos": [pos.x, pos.y],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0 if result is not None and result.status == TaskStatus.SUCCESS else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
