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
    HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \\
      --center 13,28 --water --out recordings/d2_plant_center.json
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
from harvest.core.tile_catalog import ADDR_TILEMAP, CLEARABLE_DEBRIS_TYPES, DebrisType, Tool
from harvest.maps.farm_pond import (
    POCKET_PLANT_CENTERS,
    SECOND_POCKET_PLANT_CENTER,
    WEST_POCKET_PLANT_CENTER,
)
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap, ram_seed_count
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask, EnsureCropSeedsTask
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
from harvest.tasks.skills import farm_pocket_plant_skill, farm_pocket_water_skill
from harvest.core.shipping_credit import shipping_scene_needs_dismiss
from harvest.tasks.primitives import dismiss_dialogue_action


def _parse_center(raw: str) -> tuple[int, int]:
    x_str, y_str = raw.split(",")
    return int(x_str), int(y_str)


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
        "--second-plot",
        action="store_true",
        help=(
            "Establish the D3 second potato ring at SECOND_POCKET_PLANT_CENTER "
            "(19,28) beside the D2 rows. Implies --skip-clear (open soil)."
        ),
    )
    p.add_argument(
        "--center",
        type=_parse_center,
        default=None,
        help="Override pocket center as X,Y (implies --skip-clear).",
    )
    p.add_argument(
        "--save-end-state",
        default=None,
        help="Save an emulator pin after a successful establish (+water).",
    )
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
            # 5pm ShippingScene interrupts the hoe/plant/water sequence with a
            # dialogue box. Pulse A to clear it without stepping the task so
            # nav does not time out mid-ring (handoff: dismiss, do not wander).
            if shipping_scene_needs_dismiss(ram):
                action = dismiss_dialogue_action(frame)
                if watch is not None:
                    last = _ == budget - 1
                    obs = fast_env_step(env, action, update_obs=last)
                else:
                    obs, _reward, _term, _trunc, _info = env.step(action)
                frame += 1
                if frame > start_frame + timeout:
                    stopped = True
                    break
                continue
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
        "can_level": int(read_ram_value(ram, "watering_can") or 0),
    }


def _rings_planted(ram) -> dict:
    return {
        f"{cx},{cy}": count_ring_planted(ram, (cx, cy))
        for cx, cy in POCKET_PLANT_CENTERS
    }


def _pocket_tiles(ram, center=WEST_POCKET_PLANT_CENTER) -> dict:
    cx, cy = center
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
    center = args.center or (
        SECOND_POCKET_PLANT_CENTER if args.second_plot else WEST_POCKET_PLANT_CENTER
    )
    skill_center = args.center or (SECOND_POCKET_PLANT_CENTER if args.second_plot else None)
    if args.center is not None or args.second_plot:
        args.skip_clear = True
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
        # Town-return pins (Y1_D3_PostShop) load with a stale farm tilemap;
        # the crop overlay repopulates ~60f in. Warm up before snapshotting so
        # the hoe-stand remap sees real tile IDs (the second ring is walled
        # east by the x21 bank).
        for _ in range(90):
            step_out = env.step(make_action())
            obs = step_out[0] if isinstance(step_out, tuple) else obs
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
            "pocket": _pocket_tiles(ram, center),
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
                "end": {"carry": _carry(ram), "pocket": _pocket_tiles(ram, center)},
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
                    "end": {"carry": _carry(ram), "pocket": _pocket_tiles(ram, center)},
                    "ok": False,
                }
                _write_payload(args.out, payload)
                return 1

        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        # Day plan: CROP_ESTABLISH (hoe+seeds) then ENSURE_WATERING_CAN then
        # CROP_WATER. Do not ask include_water to select a can that is not
        # in the pair — bag spend frees the seed slot for the fetch.
        plant = farm_pocket_plant_skill(
            seed_type="potato",
            center=skill_center,
            ram=ram,
            include_water=False,
            include_plant=not args.hoe_only,
        )
        plant.reset(world)
        remaining = max(200, args.timeout - frame)
        frame, result, ram, closed = _phase(
            plant, timeout=remaining, start_frame=frame
        )
        pos = get_pos_from_ram(ram)
        pocket = _pocket_tiles(ram, center)
        planted_n = count_ring_planted(ram, center)
        tilled_n = count_ring_tilled(ram, center)
        wet_n = count_ring_wet(ram, center)
        hour = int(read_ram_value(ram, "hour") or 0)
        minute = int(read_ram_value(ram, "minute") or 0)
        plant_reason = "watch window closed" if closed else (
            result.reason if result is not None else ""
        )
        plant_ok = (
            result is not None
            and result.status == TaskStatus.SUCCESS
            and not closed
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
        if plant_ok and args.water and not args.hoe_only:
            world = WorldState(frame=frame, ram=ram, info={}, obs=None)
            ensure_can = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))
            ensure_can.reset(world)
            frame, result, ram, closed = _phase(
                ensure_can, timeout=8_000, start_frame=frame
            )
            journal.append(
                {
                    "phase": "ensure_watering_can",
                    "status": result.status.value if result is not None else "none",
                    "reason": "watch window closed" if closed else (
                        result.reason if result is not None else ""
                    ),
                    "frames": frame,
                    "carry": _carry(ram),
                }
            )
            can_ok = (
                result is not None
                and result.status == TaskStatus.SUCCESS
                and not closed
            )
            if can_ok:
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                water = farm_pocket_water_skill(center=skill_center)
                water.reset(world)
                remaining = max(200, args.timeout - frame)
                frame, result, ram, closed = _phase(
                    water, timeout=remaining, start_frame=frame
                )
                pos = get_pos_from_ram(ram)
                pocket = _pocket_tiles(ram, center)
                planted_n = count_ring_planted(ram, center)
                tilled_n = count_ring_tilled(ram, center)
                wet_n = count_ring_wet(ram, center)
                hour = int(read_ram_value(ram, "hour") or 0)
                minute = int(read_ram_value(ram, "minute") or 0)
                journal.append(
                    {
                        "phase": "pocket_water",
                        "status": result.status.value if result is not None else "none",
                        "reason": "watch window closed" if closed else (
                            result.reason if result is not None else ""
                        ),
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
                plant_ok = (
                    plant_ok
                    and result is not None
                    and result.status == TaskStatus.SUCCESS
                    and not closed
                )
            else:
                plant_ok = False
        if closed:
            ok = False
        elif args.hoe_only:
            ok = plant_ok and tilled_n >= PLOT_RING_SIZE
        elif args.water:
            ok = (
                plant_ok
                and planted_n >= PLOT_RING_SIZE
                and wet_n >= PLOT_RING_SIZE
                and bag_spent
            )
        else:
            ok = plant_ok and planted_n >= PLOT_RING_SIZE and bag_spent
        rings_planted = _rings_planted(ram)
        payload = {
            "start": start,
            "second_plot": bool(args.second_plot),
            "center": list(center),
            "journal": journal,
            "end": {
                "tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
                "pos": [pos.x, pos.y],
                "carry": _carry(ram),
                "pocket": pocket,
                "planted_ring": planted_n,
                "tilled_ring": tilled_n,
                "wet_ring": wet_n,
                "rings_planted": rings_planted,
                "total_planted": sum(rings_planted.values()),
                "bag_spent": bag_spent,
                "hour": hour,
                "minute": minute,
            },
            "ok": ok,
        }
        if args.save_end_state and ok:
            from harvest.scripts.leftover_exec import save_emulator_state

            saved = save_emulator_state(env, args.save_end_state)
            payload["end_state"] = str(saved)
            print(f"[PLANT] saved {saved}")
        _write_payload(args.out, payload)
        return 0 if ok else 1
    finally:
        if watch is not None:
            watch.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
