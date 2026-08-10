#!/usr/bin/env python3
"""Clean harvest → ship → farm 5pm scene → overnight wallet credit (rr-53g).

Flow:

1. Load mature-crop morning (default ``Y1_Test_Crops_DayPlus6``).
2. Exit house → ``HarvestTask`` (pick + bin drop; ``shipped_count``).
3. Save **pre-5pm** checkpoint (wallet flat; ``shipping_money`` may be up).
4. Stay on farm until hour ≥ 17; mash through ``ShippingScene`` dialogue.
5. Save **post-5pm** checkpoint.
6. Return home + sleep; assert **wallet money rose**.
7. Write journal JSON under ``recordings/``.

ROM note: wallet ``money`` does not rise at the 5pm cutscene itself — decomp
``AddMoney(shipping_money)`` runs on overnight/morning settle. Acceptance is
"money rises after 5pm on harvest day" = after the shipping window completes
(next morning). Bin drop alone is not enough.

Examples::

    HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe
    HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe \\
      --state Y1_Day09_Harvest_Mode_Start
    # Skip harvest when bin already filled (shipping_money>0 fixture):
    HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe \\
      --state Y1_Day09_Harvest_Mode_Harvest_End --skip-harvest
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from harvest.paths import GAME_DIR, PROJECT_DIR, TASKS_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.shipping_credit import (
    SHIPPING_SCENE_HOUR,
    acceptance_ok,
    shipping_credit_journal_row,
)
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.tasks.home import GoToSleepTask, ReturnHomeTask
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.farm_clearer import get_pos_from_ram, make_action
from harvest.tasks.harvest_task import HarvestTask, read_shipping_money

# Day09 harvest fixture: dense mature potatoes near the shipping bin (Clean
# HarvestTask path verified). Keep-alive DayPlus6 still needs nav hardening.
DEFAULT_STATE = "Y1_Day09_Harvest_Mode_Start"
DEFAULT_OUT = PROJECT_DIR / "recordings" / "harvest_ship_5pm_money.json"
PRE_5PM_STATE = "Y1_Harvest_Ship_Pre5pm"
POST_5PM_STATE = "Y1_Harvest_Ship_Post5pm"
POST_SLEEP_STATE = "Y1_Harvest_Ship_PostSleep"


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default=DEFAULT_STATE, help="Start save state name")
    p.add_argument(
        "--skip-harvest",
        action="store_true",
        help="Skip HarvestTask (fixture already has bin filled / shipping_money)",
    )
    p.add_argument(
        "--assume-shipped",
        type=int,
        default=0,
        help="When --skip-harvest, treat this as shipped_count (default: use shipped_potatoes)",
    )
    p.add_argument("--harvest-timeout", type=int, default=25000)
    p.add_argument("--wait-5pm-timeout", type=int, default=20000)
    p.add_argument("--nav-timeout", type=int, default=20000)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Evidence journal JSON path",
    )
    p.add_argument(
        "--pre-5pm-state",
        default=PRE_5PM_STATE,
        help="Checkpoint name written before hour 17",
    )
    p.add_argument(
        "--post-5pm-state",
        default=POST_5PM_STATE,
        help="Checkpoint name written after shipping scene",
    )
    p.add_argument(
        "--post-sleep-state",
        default=POST_SLEEP_STATE,
        help="Checkpoint name written after overnight money settle",
    )
    return p.parse_args()


def _world(env, frame: int) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)


def _snap(env, frame: int, phase: str = "") -> dict[str, Any]:
    ram = env.get_ram()
    pos = get_pos_from_ram(ram)
    return {
        "frame": int(frame),
        "phase": phase,
        "season": int(read_ram_value(ram, "season")),
        "day": int(read_ram_value(ram, "day")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
        "money": int(read_ram_value(ram, "money")),
        "shipping_money": int(read_shipping_money(ram)),
        "shipped_potatoes": int(read_ram_value(ram, "shipped_potatoes")),
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "tilemap_hex": f"0x{int(read_ram_value(ram, 'tilemap')):02X}",
        "input_lock": int(ram[0x019A]) if len(ram) > 0x019A else -1,
        "x": int(pos.x),
        "y": int(pos.y),
    }


def _save_checkpoint(env, name: str) -> str:
    """Write gzip snes9x state under the game integration dir. Returns path."""
    GAME_DIR.mkdir(parents=True, exist_ok=True)
    path = GAME_DIR / f"{name}.state"
    state_bytes = env.em.get_state()
    with gzip.open(path, "wb", compresslevel=9) as handle:
        handle.write(state_bytes)
    return str(path)


def _run_task(
    env,
    task,
    *,
    frame: int,
    max_frames: int,
    label: str,
) -> tuple[int, TaskStatus, str, list[dict[str, Any]]]:
    """Step a Task to terminal status. Returns (frame, status, reason, snaps)."""
    snaps: list[dict[str, Any]] = []
    world = _world(env, frame)
    task.reset(world)
    snaps.append(_snap(env, frame, f"{label}:start"))
    status = TaskStatus.RUNNING
    reason = ""
    for _ in range(max_frames):
        world = _world(env, frame)
        result = task.step(world)
        status = result.status
        reason = result.reason or ""
        if status == TaskStatus.RUNNING:
            action = result.action.action if result.action is not None else make_action()
            env.step(action)
            frame += 1
            continue
        snaps.append(_snap(env, frame, f"{label}:{status.value}"))
        return frame, status, reason, snaps
    snaps.append(_snap(env, frame, f"{label}:timeout"))
    return frame, TaskStatus.FAILURE, f"{label} timeout", snaps


def _keep_on_farm_action(ram, frame: int) -> object:
    """Idle on farm; if drifted to path, walk east/right back toward farm."""
    tilemap = int(read_ram_value(ram, "tilemap"))
    lock = int(ram[0x019A]) if len(ram) > 0x019A else 1
    hour = int(read_ram_value(ram, "hour"))
    if not is_farm_tilemap(tilemap) and not is_house_tilemap(tilemap):
        # Path / other outdoor: push toward farm (east from west path).
        return make_action(right=True, b=True)
    if lock != 1 or hour >= SHIPPING_SCENE_HOUR:
        # Mash A through shipping dialogue / locks.
        return make_action(a=True) if frame % 2 == 0 else make_action()
    # Nudge slightly inland if hugging west edge (y high / x low risks path).
    pos = get_pos_from_ram(ram)
    if int(pos.x) < 80:
        return make_action(right=True)
    return make_action()


def _wait_farm_shipping_scene(
    env,
    *,
    frame: int,
    timeout: int,
) -> tuple[int, bool, list[dict[str, Any]]]:
    """Stay on farm until hour≥17 and input unlocks after shipping dialog."""
    snaps: list[dict[str, Any]] = []
    snaps.append(_snap(env, frame, "wait_5pm:start"))
    saw_lock = False
    start = frame
    while frame - start < timeout:
        ram = env.get_ram()
        hour = int(read_ram_value(ram, "hour"))
        lock = int(ram[0x019A]) if len(ram) > 0x019A else 1
        if hour >= SHIPPING_SCENE_HOUR and lock != 1:
            saw_lock = True
        env.step(_keep_on_farm_action(ram, frame))
        frame += 1
        if hour >= SHIPPING_SCENE_HOUR and lock == 1 and (saw_lock or frame - start > 500):
            # Either we saw dialog lock, or clock passed 17 with free input.
            if frame % 30 == 0 or saw_lock:
                snaps.append(_snap(env, frame, "wait_5pm:settle"))
            if saw_lock and lock == 1:
                snaps.append(_snap(env, frame, "wait_5pm:done"))
                return frame, True, snaps
            if hour > SHIPPING_SCENE_HOUR or (
                hour == SHIPPING_SCENE_HOUR and int(read_ram_value(ram, "minute")) >= 5
            ):
                snaps.append(_snap(env, frame, "wait_5pm:done_no_dialog"))
                return frame, True, snaps
        if frame % 900 == 0:
            snaps.append(_snap(env, frame, "wait_5pm:tick"))
    snaps.append(_snap(env, frame, "wait_5pm:timeout"))
    hour = int(read_ram_value(env.get_ram(), "hour"))
    return frame, hour >= SHIPPING_SCENE_HOUR, snaps


def main() -> int:
    _configure_headless()
    args = _parse_args()
    t0 = time.time()
    journal: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "state": args.state,
        "skip_harvest": bool(args.skip_harvest),
        "intervention": "Clean",
        "ram_writes": 0,
        "mid_run_state_loads": 0,
        "journal": journal,
        "success": False,
    }

    env = make_harvest_env(args.state, render_mode="rgb_array")
    try:
        env.reset()
        frame = 0
        journal.append(_snap(env, frame, "boot"))
        money_start = int(read_ram_value(env.get_ram(), "money"))
        report["money_start"] = money_start

        # --- Exit house if needed ---
        tilemap = int(read_ram_value(env.get_ram(), "tilemap"))
        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            frame, status, reason, snaps = _run_task(
                env,
                ExitToFarmTask(tasks_dir=str(TASKS_DIR)),
                frame=frame,
                max_frames=args.nav_timeout,
                label="exit_to_farm",
            )
            journal.extend(snaps)
            if status != TaskStatus.SUCCESS:
                report["failure"] = f"exit_to_farm: {reason}"
                return _finish(args, report, t0, exit_code=1)

        # Inland nudge so west-edge fixtures do not auto-transition to path.
        for _ in range(180):
            ram = env.get_ram()
            if is_farm_tilemap(int(read_ram_value(ram, "tilemap"))):
                pos = get_pos_from_ram(ram)
                if int(pos.x) >= 120:
                    break
            env.step(make_action(right=True, b=True))
            frame += 1
        journal.append(_snap(env, frame, "inland"))

        shipped_count = 0
        harvested_count = 0
        harvest_reason = ""

        if not args.skip_harvest:
            harvest = HarvestTask(
                name="harvest_rr53g",
                state_name=args.state,
                timeout=args.harvest_timeout,
            )
            frame, status, harvest_reason, snaps = _run_task(
                env,
                harvest,
                frame=frame,
                max_frames=args.harvest_timeout,
                label="harvest",
            )
            journal.extend(snaps)
            shipped_count = int(harvest.shipped_count)
            harvested_count = int(harvest.harvested_count)
            report["harvest_status"] = status.value
            report["harvest_reason"] = harvest_reason
            report["shipped_count"] = shipped_count
            report["harvested_count"] = harvested_count
            if status != TaskStatus.SUCCESS or shipped_count <= 0:
                # Partial ship still useful if any bin drops succeeded.
                if shipped_count <= 0:
                    report["failure"] = (
                        f"harvest incomplete: status={status.value} "
                        f"reason={harvest_reason} shipped={shipped_count}"
                    )
                    return _finish(args, report, t0, exit_code=1)
        else:
            # Fixture already filled the bin (e.g. Harvest_Mode_Harvest_End).
            ram = env.get_ram()
            shipped_count = int(args.assume_shipped) or max(
                1, int(read_ram_value(ram, "shipped_potatoes"))
            )
            report["shipped_count"] = shipped_count
            report["harvested_count"] = shipped_count
            report["harvest_status"] = "skipped"
            journal.append(_snap(env, frame, "harvest:skipped"))

        # --- Pre-5pm checkpoint ---
        pre_snap = _snap(env, frame, "pre_5pm")
        money_pre = int(pre_snap["money"])
        ship_pre = int(pre_snap["shipping_money"])
        pre_path = _save_checkpoint(env, args.pre_5pm_state)
        report["pre_5pm"] = {**pre_snap, "state": args.pre_5pm_state, "path": pre_path}
        journal.append({**pre_snap, "checkpoint": args.pre_5pm_state})
        print(
            f"[PROBE] pre-5pm money={money_pre} shipping_money={ship_pre} "
            f"shipped={shipped_count} -> {pre_path}",
            flush=True,
        )

        # --- Farm wait through 5pm ShippingScene ---
        frame, ok_5pm, snaps = _wait_farm_shipping_scene(
            env, frame=frame, timeout=args.wait_5pm_timeout
        )
        journal.extend(snaps)
        post5_snap = _snap(env, frame, "post_5pm")
        post5_path = _save_checkpoint(env, args.post_5pm_state)
        report["post_5pm"] = {
            **post5_snap,
            "state": args.post_5pm_state,
            "path": post5_path,
            "shipping_scene_ok": ok_5pm,
        }
        journal.append({**post5_snap, "checkpoint": args.post_5pm_state})
        print(
            f"[PROBE] post-5pm hour={post5_snap['hour']} money={post5_snap['money']} "
            f"(wallet still pre-sleep) -> {post5_path}",
            flush=True,
        )

        # --- Return home + sleep (wallet AddMoney) ---
        frame, status, reason, snaps = _run_task(
            env,
            ReturnHomeTask(tasks_dir=str(TASKS_DIR)),
            frame=frame,
            max_frames=args.nav_timeout,
            label="return_home",
        )
        journal.extend(snaps)
        if status != TaskStatus.SUCCESS:
            report["failure"] = f"return_home: {reason}"
            return _finish(args, report, t0, exit_code=1)

        frame, status, reason, snaps = _run_task(
            env,
            GoToSleepTask(tasks_dir=str(TASKS_DIR)),
            frame=frame,
            max_frames=args.nav_timeout,
            label="sleep",
        )
        journal.extend(snaps)
        if status != TaskStatus.SUCCESS:
            report["failure"] = f"sleep: {reason}"
            return _finish(args, report, t0, exit_code=1)

        post_sleep = _snap(env, frame, "post_sleep")
        post_sleep_path = _save_checkpoint(env, args.post_sleep_state)
        report["post_sleep"] = {
            **post_sleep,
            "state": args.post_sleep_state,
            "path": post_sleep_path,
        }
        journal.append({**post_sleep, "checkpoint": args.post_sleep_state})

        credit_row = shipping_credit_journal_row(
            shipped_count=shipped_count,
            harvested_count=harvested_count,
            money_pre_5pm=money_pre,
            money_post_5pm=int(post5_snap["money"]),
            money_post_sleep=int(post_sleep["money"]),
            shipping_money_pre_5pm=ship_pre,
            shipping_money_post_5pm=int(post5_snap["shipping_money"]),
            shipping_money_post_sleep=int(post_sleep["shipping_money"]),
            hour_pre_5pm=int(pre_snap["hour"]),
            hour_post_5pm=int(post5_snap["hour"]),
            day_pre=int(pre_snap["day"]),
            day_post_sleep=int(post_sleep["day"]),
            pre_5pm_state=args.pre_5pm_state,
            post_5pm_state=args.post_5pm_state,
            post_sleep_state=args.post_sleep_state,
        )
        report["shipping_credit"] = credit_row
        journal.append(credit_row)

        ok = acceptance_ok(credit_row)
        report["success"] = ok
        report["frames"] = frame
        print(
            f"[PROBE] money {money_pre} -> {post_sleep['money']} "
            f"(delta={credit_row['money_delta']}) shipped={shipped_count} "
            f"ok={ok}",
            flush=True,
        )
        if not ok:
            report["failure"] = (
                f"acceptance failed: shipped={shipped_count} "
                f"money_pre={money_pre} money_post_sleep={post_sleep['money']}"
            )
            return _finish(args, report, t0, exit_code=1)
        return _finish(args, report, t0, exit_code=0)
    finally:
        env.close()


def _finish(args: argparse.Namespace, report: dict[str, Any], t0: float, *, exit_code: int) -> int:
    report["wall_seconds"] = round(time.time() - t0, 2)
    report["exit_code"] = exit_code
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PROBE] Wrote {out} success={report.get('success')}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
