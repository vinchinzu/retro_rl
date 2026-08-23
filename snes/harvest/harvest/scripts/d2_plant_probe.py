#!/usr/bin/env python3
"""Post-shop D2 collect + 3x3 hoe + plant probe (no grape).

Default pin is ``Y1_After_Buy_Potato`` (stock=1, carry often empty).
Hoe the 8-tile ring around (13,28), then plant from the untilled notch.
``--hoe-only`` tills until 5pm-tuning without spending the bag.
``--water`` waters the 8-ring after plant (0x54 → 0x55).

    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe
    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \\
      --state Y1_After_Buy_Potato --out recordings/d2_plant_probe.json
    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \\
      --state Y1_After_Buy_Potato --hoe-only --out recordings/d2_hoe_ring.json
    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \\
      --state Y1_After_Buy_Potato --water --out recordings/d2_plant_water.json
    uv run python -m harvest.scripts.d2_plant_probe --watch
"""

from __future__ import annotations

import argparse
import json
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
from harvest.runtime.watch_display import (
    WatchDisplay,
    configure_headed,
    configure_headless,
    fast_env_step,
)
from harvest.tasks.crop_skills import (
    PLOT_RING_SIZE,
    count_ring_planted,
    count_ring_tilled,
    count_ring_wet,
)
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action
from harvest.tasks.skills import farm_pocket_plant_skill


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_After_Buy_Potato")
    p.add_argument("--timeout", type=int, default=36_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "d2_plant_probe.json",
    )
    p.add_argument("--skip-clear", action="store_true")
    p.add_argument(
        "--hoe-only",
        action="store_true",
        help="Till the 8-tile ring only; do not plant. Tune hoe until 5pm.",
    )
    p.add_argument(
        "--water",
        action="store_true",
        help="Water the 8-ring after plant (notch stays untilled).",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Open a pygame window ([ ] speed, TAB turbo). No HEADLESS.",
    )
    p.add_argument("--watch-scale", type=int, default=3, help="Watch window integer scale")
    return p.parse_args()


def _run_task(env, task, *, timeout: int, start_frame: int, watch: WatchDisplay | None = None):
    obs = None
    result = None
    frame = start_frame
    closed = False
    while frame <= start_frame + timeout:
        if watch is not None:
            if not watch.pump():
                closed = True
                break
            budget = watch.emu_repeat()
        else:
            budget = 1
        stopped = False
        for _ in range(budget):
            ram = env.get_ram()
            world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                stopped = True
                break
            action = result.action.action if result.action is not None else make_action()
            if watch is not None:
                last = _ == budget - 1
                obs = fast_env_step(env, action, update_obs=last)
            else:
                obs, _reward, _term, _trunc, _info = env.step(action)
            frame += 1
            if frame > start_frame + timeout:
                stopped = True
                break
        if watch is not None and not watch.present(obs, emu_frame=frame):
            closed = True
            break
        if stopped:
            break
    return frame, result, env.get_ram(), closed


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


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> int:
    args = _parse_args()
    if args.watch:
        configure_headed()
    else:
        configure_headless()
    env = make_harvest_env(state=args.state)
    journal = []
    watch = None
    try:
        boot = env.reset()
        obs = boot[0] if isinstance(boot, tuple) else boot
        if args.watch:
            watch = WatchDisplay(
                scale=args.watch_scale,
                title="Harvest D2 plant probe",
            )
            if not watch.start(obs):
                payload = {"ok": False, "reason": "watch window failed"}
                _write_payload(args.out, payload)
                return 1
        ram = env.get_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        frame = 0
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        start = {
            "tilemap": hex(tilemap),
            "carry": _carry(ram),
            "pocket": _pocket_tiles(ram),
        }

        def _phase(task, *, timeout: int, start_frame: int):
            return _run_task(
                env, task, timeout=timeout, start_frame=start_frame, watch=watch
            )

        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            exit_task = ExitToFarmTask()
            exit_task.reset(world)
            frame, result, ram, closed = _phase(exit_task, timeout=2_000, start_frame=0)
            journal.append(
                {
                    "phase": "exit_to_farm",
                    "status": result.status.value if result is not None else "none",
                    "reason": "watch window closed" if closed else (
                        result.reason if result is not None else ""
                    ),
                    "frames": frame,
                }
            )
            if closed or result is None or result.status != TaskStatus.SUCCESS:
                payload = {"start": start, "journal": journal, "ok": False}
                _write_payload(args.out, payload)
                return 1

        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        ensure = EnsureCropSeedsTask(seed_type="potato")
        ensure.reset(world)
        frame, result, ram, closed = _phase(ensure, timeout=8_000, start_frame=frame)
        journal.append(
            {
                "phase": "ensure_crop_seeds",
                "status": result.status.value if result is not None else "none",
                "reason": "watch window closed" if closed else (
                    result.reason if result is not None else ""
                ),
                "frames": frame,
                "carry": _carry(ram),
            }
        )
        if closed or result is None or result.status != TaskStatus.SUCCESS:
            payload = {
                "start": start,
                "journal": journal,
                "end": {"carry": _carry(ram), "pocket": _pocket_tiles(ram)},
                "ok": False,
            }
            _write_payload(args.out, payload)
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
            frame, result, ram, closed = _phase(
                clear, timeout=7_000, start_frame=frame
            )
            journal.append(
                {
                    "phase": "clear_plot",
                    "status": result.status.value if result is not None else "none",
                    "reason": "watch window closed" if closed else (
                        result.reason if result is not None else ""
                    ),
                    "frames": frame,
                    "cleared": max(0, len(before) - len(_debris(ram))),
                    "remaining": len(_debris(ram)),
                }
            )
            if closed or result is None or result.status != TaskStatus.SUCCESS:
                payload = {
                    "start": start,
                    "journal": journal,
                    "end": {"carry": _carry(ram), "pocket": _pocket_tiles(ram)},
                    "ok": False,
                }
                _write_payload(args.out, payload)
                return 1

        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        plant = farm_pocket_plant_skill(
            seed_type="potato",
            include_water=bool(args.water) and not args.hoe_only,
            include_plant=not args.hoe_only,
        )
        plant.reset(world)
        remaining = max(200, args.timeout - frame)
        frame, result, ram, closed = _phase(
            plant, timeout=remaining, start_frame=frame
        )
        pos = get_pos_from_ram(ram)
        pocket = _pocket_tiles(ram)
        planted_n = count_ring_planted(ram, WEST_POCKET_PLANT_CENTER)
        tilled_n = count_ring_tilled(ram, WEST_POCKET_PLANT_CENTER)
        wet_n = count_ring_wet(ram, WEST_POCKET_PLANT_CENTER)
        hour = int(read_ram_value(ram, "hour") or 0)
        minute = int(read_ram_value(ram, "minute") or 0)
        plant_reason = "watch window closed" if closed else (
            result.reason if result is not None else ""
        )
        journal.append(
            {
                "phase": "hoe_only" if args.hoe_only else "pocket_plant",
                "status": result.status.value if result is not None else "none",
                "reason": plant_reason,
                "frames": frame,
                "carry": _carry(ram),
                "pocket": pocket,
                "planted_ring": planted_n,
                "tilled_ring": tilled_n,
                "wet_ring": wet_n,
                "hour": hour,
                "minute": minute,
            }
        )
        bag_spent = _carry(ram)["selected"] != 0x07 and _carry(ram)["backpack"] != 0x07
        if closed:
            ok = False
        elif args.hoe_only:
            ok = (
                result is not None
                and result.status == TaskStatus.SUCCESS
                and tilled_n >= PLOT_RING_SIZE
            )
        elif args.water:
            ok = (
                result is not None
                and result.status == TaskStatus.SUCCESS
                and planted_n >= PLOT_RING_SIZE
                and wet_n >= PLOT_RING_SIZE
                and bag_spent
            )
        else:
            ok = (
                result is not None
                and result.status == TaskStatus.SUCCESS
                and planted_n >= PLOT_RING_SIZE
                and bag_spent
            )
        payload = {
            "start": start,
            "journal": journal,
            "end": {
                "tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
                "pos": [pos.x, pos.y],
                "carry": _carry(ram),
                "pocket": pocket,
                "planted_ring": planted_n,
                "tilled_ring": tilled_n,
                "wet_ring": wet_n,
                "bag_spent": bag_spent,
                "hour": hour,
                "minute": minute,
            },
            "ok": ok,
        }
        _write_payload(args.out, payload)
        return 0 if ok else 1
    finally:
        if watch is not None:
            watch.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
