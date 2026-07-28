#!/usr/bin/env python3
"""Run Harvest Moon from a morning state through one or more overnights.

Default path: multi-day planner with ``target_days=1`` (return-home + sleep that
always finds the house). Optional ``--day-plan boot_to_day2`` runs the explicit
macro-chained phase sequence inside a single-day plan that ends with sleep.

Success: calendar goal reached, morning scene stable, no mid-run state load.

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --state Y1_Inside_House
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --days 2 --until-day 4
    # Full spring from pinned morning (sleeps through Spring 30 → Summer 1):
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --end-of-spring \\
      --out recordings/run_spring_month.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.planner.day_plan import DayPlanTask, MultiDayPlannerTask, PHASE_SEQUENCES
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.farm_clearer import make_action


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument(
        "--day-plan",
        default=None,
        choices=list(PHASE_SEQUENCES.keys()),
        help="Named phase sequence; default uses multi-day auto plan",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of successful overnights to complete (multi-day mode)",
    )
    p.add_argument(
        "--until-day",
        type=int,
        default=None,
        help="Stop once calendar day is >= this (inclusive morning target)",
    )
    p.add_argument(
        "--until-season",
        type=int,
        default=None,
        help="Season index for --until-day (default: start season)",
    )
    p.add_argument(
        "--end-of-spring",
        action="store_true",
        help="Run until Summer morning (MultiDay until Spring 30 exclusive end)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Hard frame budget (default scales with overnight count)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N frames (0 disables)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "run_to_day2.json",
    )
    p.add_argument(
        "--save-end-state",
        type=str,
        default=None,
        help="Optional state name to write under custom_integrations when done",
    )
    return p.parse_args()


def _world(env, frame: int = 0) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)


def _date_fields(ram) -> dict[str, int]:
    return {
        "season": int(read_ram_value(ram, "season")),
        "day": int(read_ram_value(ram, "day")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "money": int(read_ram_value(ram, "money")),
        "stamina": int(read_ram_value(ram, "stamina")),
    }


def _build_task(args: argparse.Namespace, start_season: int) -> object:
    if args.day_plan:
        if (
            args.days is not None
            or args.until_day is not None
            or args.end_of_spring
        ):
            raise SystemExit(
                "--day-plan cannot be combined with --days/--until-day/--end-of-spring"
            )
        return DayPlanTask(
            phase_sequence=list(PHASE_SEQUENCES[args.day_plan]),
            state_name=args.state,
        )

    # MultiDayPlannerTask: target_days OR exclusive until_(season, day).
    # harvest_bot --end-of-spring uses until_season=0 until_day=30 so success
    # when date > Spring 30 (i.e. Summer 1 morning).
    if args.end_of_spring:
        return MultiDayPlannerTask(
            until_season=0,
            until_day=30,
            max_days=40,
        )

    target_days = args.days
    until_day = args.until_day
    until_season = args.until_season if args.until_season is not None else start_season

    if target_days is None and until_day is None:
        target_days = 1

    kwargs: dict = {"max_days": 40}
    if target_days is not None:
        kwargs["target_days"] = target_days
        kwargs["max_days"] = max(target_days + 1, 2)

    if until_day is not None:
        # Inclusive morning target → exclusive MultiDay bound (day-1).
        # Example: --until-day 4 → stop when date > Spring 3 → D4 morning.
        # Cross-season: --until-season 1 --until-day 1 → exclusive (1, 0).
        if until_day <= 1 and until_season > 0:
            kwargs["until_season"] = until_season - 1
            kwargs["until_day"] = 30
        else:
            kwargs["until_season"] = until_season
            kwargs["until_day"] = max(0, until_day - 1)
        if target_days is None:
            kwargs.pop("target_days", None)
            kwargs["max_days"] = max(kwargs.get("until_day", 30) + 5, 4)

    return MultiDayPlannerTask(**kwargs)


def _goal_reached(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    days_completed: int | None,
    args: argparse.Namespace,
) -> bool:
    if end <= start and not args.end_of_spring:
        # Still allow end-of-spring only if we somehow wrap? No.
        if end == start:
            return False
    if args.end_of_spring:
        # Summer (or later) morning after finishing Spring 30 overnight.
        return end > (0, 30)
    if end <= start:
        return False
    if args.until_day is not None:
        until_season = (
            args.until_season if args.until_season is not None else start[0]
        )
        return end >= (until_season, args.until_day)
    if args.days is not None:
        if days_completed is not None:
            return days_completed >= args.days
        return end[0] > start[0] or end[1] >= start[1] + args.days
    return end > start


def _summarize_journal(journal: list[dict]) -> dict:
    phase_success: dict[str, int] = {}
    phase_skip: dict[str, int] = {}
    for row in journal:
        for result in row.get("phase_results") or []:
            name = str(result.get("phase", "?"))
            status = str(result.get("status", ""))
            if status == "success":
                phase_success[name] = phase_success.get(name, 0) + 1
            elif status == "skipped":
                phase_skip[name] = phase_skip.get(name, 0) + 1
    return {
        "overnights": len(journal),
        "phase_success_counts": phase_success,
        "phase_skip_counts": phase_skip,
        "final_money": journal[-1].get("money") if journal else None,
    }


def main() -> int:
    args = _parse_args()
    _configure_headless()

    if args.end_of_spring:
        overnights_budget = 32
    elif args.days is not None:
        overnights_budget = args.days
    elif args.until_day is not None:
        overnights_budget = max(2, args.until_day + 2)
    else:
        overnights_budget = 1

    if args.max_frames is None:
        # ~25k frames/day upper bound with crop/clear work.
        args.max_frames = 30_000 * max(1, overnights_budget)

    env = make_harvest_env(state=args.state, render_mode="rgb_array")
    t0 = time.monotonic()
    try:
        env.reset()
        frames = 0
        world = _world(env, frames)
        start_fields = _date_fields(world.ram)
        start_season = start_fields["season"]
        start_day = start_fields["day"]
        start_key = (start_season, start_day)

        task = _build_task(args, start_season)
        task.reset(world)

        plan_label = args.day_plan or (
            "end_of_spring" if args.end_of_spring else "auto_multi_day"
        )
        print(
            f"[RUN] state={args.state} plan={plan_label} "
            f"start=S{start_season}D{start_day} "
            f"days={args.days} until=({args.until_season},{args.until_day}) "
            f"end_of_spring={args.end_of_spring} max_frames={args.max_frames}",
            flush=True,
        )

        reason = "budget"
        terminal = False
        last_logged_day = start_key
        while frames < args.max_frames:
            world = _world(env, frames)
            result = task.step(world)
            frames += 1

            season = int(read_ram_value(world.ram, "season"))
            day = int(read_ram_value(world.ram, "day"))
            current = (season, day)

            if current != last_logged_day:
                print(
                    f"[RUN] day change S{last_logged_day[0]}D{last_logged_day[1]} "
                    f"-> S{season}D{day} at frame={frames}",
                    flush=True,
                )
                last_logged_day = current

            if result.status == TaskStatus.SUCCESS:
                reason = result.reason or "success"
                terminal = True
                break
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or result.status.value
                terminal = True
                break

            if args.day_plan and _goal_reached(
                start=start_key,
                end=current,
                days_completed=getattr(task, "_days_completed", None),
                args=args,
            ):
                reason = "goal reached"
                terminal = True
                break

            if args.progress_every and frames % args.progress_every == 0:
                phase = getattr(task, "phase_text", "?")
                progress = getattr(task, "progress_text", "")
                hour = int(read_ram_value(world.ram, "hour"))
                minute = int(read_ram_value(world.ram, "minute"))
                money = int(read_ram_value(world.ram, "money"))
                print(
                    f"[RUN] f={frames} date=S{season}D{day} "
                    f"{hour:02d}:{minute:02d} ${money} phase={phase} {progress}",
                    flush=True,
                )

            action = (
                result.action.action
                if result.action is not None
                else make_action()
            )
            env.step(action)

        world = _world(env, frames)
        scene = classify_scene_from_ram(world.ram)
        end_fields = _date_fields(world.ram)
        end_key = (end_fields["season"], end_fields["day"])
        days_completed = getattr(task, "_days_completed", None)
        journal = list(getattr(task, "day_journal", ()) or ())
        advanced = end_key > start_key
        goal = _goal_reached(
            start=start_key,
            end=end_key,
            days_completed=days_completed,
            args=args,
        )
        morning_ok = morning_scene_ready(scene, end_fields["hour"]) or (
            scene.is_normal_map and int(read_ram_value(world.ram, "input_lock")) == 1
        )
        success = bool(goal and advanced and morning_ok)

        if args.save_end_state and success:
            try:
                from harvest.paths import GAME_DIR

                state_bytes = env.em.get_state()
                out_state = GAME_DIR / f"{args.save_end_state}.state"
                out_state.write_bytes(state_bytes)
                print(f"[RUN] Saved end state -> {out_state}", flush=True)
            except Exception as exc:
                print(f"[RUN] Could not save end state: {exc}", flush=True)

        report = {
            "state": args.state,
            "day_plan": plan_label,
            "days": args.days,
            "until_day": args.until_day,
            "until_season": args.until_season,
            "end_of_spring": bool(args.end_of_spring),
            "frames": frames,
            "wall_seconds": round(time.monotonic() - t0, 1),
            "start": start_fields,
            "end": end_fields,
            "scene": scene.summary(),
            "morning_ready": bool(morning_ok),
            "days_completed": days_completed,
            "day_failures": list(getattr(task, "day_failures", ()) or ()),
            "day_journal": journal,
            "journal_summary": _summarize_journal(journal),
            "success": success,
            "advanced": advanced,
            "goal_reached": goal,
            "reason": ("goal reached" if success else reason),
            "terminal": terminal,
            "mid_run_state_load": False,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if success else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
