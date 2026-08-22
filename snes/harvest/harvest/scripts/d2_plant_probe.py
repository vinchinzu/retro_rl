#!/usr/bin/env python3
"""Post-shop D2 collect + hoe + plant probe (no grape, no water).

Default pin is ``Y1_After_Buy_Potato`` (stock=1, carry often empty).

    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe
    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \\
      --state Y1_After_Buy_Potato --out recordings/d2_plant_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.carry import backpack_tool, selected_tool
from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP, CLEARABLE_DEBRIS_TYPES, DebrisType
from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap, ram_seed_count
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.planner.tasks.inventory_shed import EnsureCropSeedsTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action
from harvest.tasks.skills import farm_pocket_plant_skill


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_After_Buy_Potato")
    p.add_argument("--timeout", type=int, default=28_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "d2_plant_probe.json",
    )
    p.add_argument("--skip-clear", action="store_true")
    return p.parse_args()


def _run_task(env, task, *, timeout: int, start_frame: int):
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


def _carry(ram) -> dict:
    return {
        "selected": int(selected_tool(ram)),
        "backpack": int(backpack_tool(ram)),
        "potato_stock": int(ram_seed_count(ram, "potato")),
    }


def _pocket_tiles(ram) -> dict:
    cx, cy = WEST_POCKET_PLANT_CENTER
    grid = []
    for dy in range(-1, 2):
        row = []
        for dx in range(-1, 2):
            row.append(int(get_tile_at(ram, cx + dx, cy + dy)))
        grid.append(row)
    return {
        "center": [cx, cy],
        "center_tid": int(get_tile_at(ram, cx, cy)),
        "grid": grid,
    }


def _debris(ram):
    return [
        (t.tile[0], t.tile[1], t.debris_type.name)
        for t in TileScanner().scan(
            ram, WEST_PLANT_POCKET_BOUNDS, types=set(CLEARABLE_DEBRIS_TYPES)
        )
    ]


def main() -> int:
    _configure_headless()
    args = _parse_args()
    env = make_harvest_env(state=args.state)
    journal = []
    try:
        env.reset()
        ram = env.get_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        frame = 0
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        start = {
            "tilemap": hex(tilemap),
            "carry": _carry(ram),
            "pocket": _pocket_tiles(ram),
        }

        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            exit_task = ExitToFarmTask()
            exit_task.reset(world)
            frame, result, ram = _run_task(env, exit_task, timeout=2_000, start_frame=0)
            journal.append(
                {
                    "phase": "exit_to_farm",
                    "status": result.status.value if result is not None else "none",
                    "reason": result.reason if result is not None else "",
                    "frames": frame,
                }
            )
            if result is None or result.status != TaskStatus.SUCCESS:
                payload = {"start": start, "journal": journal, "ok": False}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps(payload, indent=2))
                return 1

        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        ensure = EnsureCropSeedsTask(seed_type="potato")
        ensure.reset(world)
        frame, result, ram = _run_task(env, ensure, timeout=8_000, start_frame=frame)
        journal.append(
            {
                "phase": "ensure_crop_seeds",
                "status": result.status.value if result is not None else "none",
                "reason": result.reason if result is not None else "",
                "frames": frame,
                "carry": _carry(ram),
            }
        )
        if result is None or result.status != TaskStatus.SUCCESS:
            payload = {
                "start": start,
                "journal": journal,
                "end": {"carry": _carry(ram), "pocket": _pocket_tiles(ram)},
                "ok": False,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2))
            return 1

        if not args.skip_clear:
            world = WorldState(frame=frame, ram=ram, info={}, obs=None)
            clear = FarmClearTask(
                timeout=7_000,
                fetch_tools=False,
                prefer_lift_for_weeds=True,
                prefer_lift_for_stones=True,
                farm_bounds=WEST_PLANT_POCKET_BOUNDS,
                priority=[DebrisType.WEED, DebrisType.STONE],
            )
            clear.reset(world)
            before = _debris(ram)
            frame, result, ram = _run_task(
                env, clear, timeout=7_000, start_frame=frame
            )
            journal.append(
                {
                    "phase": "clear_plot",
                    "status": result.status.value if result is not None else "none",
                    "reason": result.reason if result is not None else "",
                    "frames": frame,
                    "cleared": max(0, len(before) - len(_debris(ram))),
                    "remaining": len(_debris(ram)),
                }
            )

        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        plant = farm_pocket_plant_skill(seed_type="potato", include_water=False)
        plant.reset(world)
        remaining = max(200, args.timeout - frame)
        frame, result, ram = _run_task(env, plant, timeout=remaining, start_frame=frame)
        pos = get_pos_from_ram(ram)
        pocket = _pocket_tiles(ram)
        journal.append(
            {
                "phase": "pocket_plant",
                "status": result.status.value if result is not None else "none",
                "reason": result.reason if result is not None else "",
                "frames": frame,
                "carry": _carry(ram),
                "pocket": pocket,
            }
        )
        planted = pocket["center_tid"] in {0x54, 0x55} or any(
            tid in {0x54, 0x55} for row in pocket["grid"] for tid in row
        )
        bag_spent = _carry(ram)["selected"] != 0x07 and _carry(ram)["backpack"] != 0x07
        # One-cell D2 leaves the bag equipped when neighbors are untilled;
        # 3x3 tape spends it. Planted dry 0x54 is the close.
        ok = (
            result is not None
            and result.status == TaskStatus.SUCCESS
            and planted
        )
        payload = {
            "start": start,
            "journal": journal,
            "end": {
                "tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
                "pos": [pos.x, pos.y],
                "carry": _carry(ram),
                "pocket": pocket,
                "planted": planted,
                "bag_spent": bag_spent,
            },
            "ok": ok,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
