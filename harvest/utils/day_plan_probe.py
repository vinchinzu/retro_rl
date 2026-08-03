#!/usr/bin/env python3
"""Headless day-plan probe.

Runs the bot without opening a Pygame window and emits JSONL events for phase
changes, watched RAM changes, tilemap changes, and high-stasis navigation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import stable_retro as retro

from harvest.planner.day_phase_types import preflight_phase_contract
from harvest.runtime.harvest_bot import AutoClearBot, GameState, resolve_day_plan_name
from harvest.runtime.probe_utils import (
    event_row,
    frame_in_ranges,
    parse_field_list,
    parse_frame_ranges,
    snapshot_from_ram,
    watch_changes,
    watch_values,
)
from harvest.runtime.retro_setup import register_harvest_integration


def _json_write(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def _current_phase_key(bot: AutoClearBot) -> tuple[object, ...]:
    if not bot.day_plan_started:
        return ("not_started",)
    task = getattr(bot.day_plan_task, "_current_task", None)
    return (
        getattr(bot.day_plan_task, "phase_text", ""),
        getattr(bot.day_plan_task, "_phase_index", None),
        task.__class__.__name__ if task is not None else None,
        getattr(task, "_phase", None),
    )


def _planned_contract_summary(bot: AutoClearBot, ram) -> list[dict[str, object]]:
    """Soft-evaluate contracts for the planned (not runtime-spliced) phase list."""
    day_plan = getattr(bot, "day_plan_task", None)
    if day_plan is None:
        return []
    phases = getattr(day_plan, "phases", None) or ()
    summary: list[dict[str, object]] = []
    for phase in phases:
        if not hasattr(phase, "contract"):
            continue
        result = preflight_phase_contract(phase, ram=ram)
        # Compact: skip fully empty contracts in the start summary.
        if result.get("empty"):
            continue
        summary.append(
            {
                "phase": result["phase"],
                "ok": result["ok"],
                "reasons": result["reasons"],
                "tools": result["tools"],
                "tilemap_hex": result["tilemap_hex"],
            }
        )
    return summary


def run_probe(args: argparse.Namespace) -> int:
    fields = parse_field_list(args.watch)
    human_ranges = parse_frame_ranges(args.human_range)
    day_plan_name = resolve_day_plan_name(args.day_plan, args.resume_water, args.state)

    register_harvest_integration(retro, require_rom=True)
    env = retro.make(
        game="HarvestMoon-Snes",
        state=args.state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
    )

    out_path = Path(args.out) if args.out else None
    out_handle = out_path.open("w", encoding="utf-8") if out_path else sys.stdout
    try:
        obs, info = env.reset()
        bot = AutoClearBot(
            day_plan_enabled=True,
            day_plan_sequence=day_plan_name,
            auto_day_plan_state_name=args.state,
        )
        bot.set_env(env)
        bot.enabled = True
        bot.prepare_for_enable()

        game_state = GameState(info, env.get_ram())
        previous_watch = None
        previous_tilemap = None
        previous_phase = None
        last_stasis_report = -10_000
        was_human = False

        for frame in range(args.max_frames + 1):
            ram_before = env.get_ram()
            if frame_in_ranges(frame, human_ranges):
                if not was_human:
                    bot.enabled = False
                    was_human = True
                    snap = snapshot_from_ram(ram_before, frame=frame, action=np.zeros(12, dtype=np.int32))
                    _json_write(
                        out_handle,
                        event_row(
                            "hotswap_human",
                            snap,
                            watches=watch_values(ram_before, fields),
                            ram=ram_before,
                        ),
                    )
                action = np.zeros(12, dtype=np.int32)
            else:
                if was_human:
                    bot.enabled = True
                    bot.prepare_for_enable()
                    was_human = False
                    snap = snapshot_from_ram(ram_before, frame=frame, action=np.zeros(12, dtype=np.int32))
                    _json_write(
                        out_handle,
                        event_row(
                            "hotswap_bot",
                            snap,
                            watches=watch_values(ram_before, fields),
                            day_plan=bot.day_plan_task,
                            ram=ram_before,
                        ),
                    )
                action = bot.get_action(game_state, obs)

            obs, _reward, _terminated, _truncated, info = env.step(action)
            ram = env.get_ram()
            game_state = GameState(info, ram)

            snap = snapshot_from_ram(ram, frame=frame, action=action)
            current_watch = watch_values(ram, fields)
            changes = watch_changes(previous_watch, current_watch)
            phase_key = _current_phase_key(bot)

            if frame == 0:
                _json_write(
                    out_handle,
                    event_row(
                        "start",
                        snap,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        ram=ram,
                        extras={
                            "contract_preflight_planned": _planned_contract_summary(
                                bot, ram
                            ),
                        },
                    ),
                )
            if previous_tilemap is not None and snap.tilemap != previous_tilemap:
                _json_write(
                    out_handle,
                    event_row(
                        "tilemap",
                        snap,
                        changes=changes,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        ram=ram,
                    ),
                )
            if previous_phase is not None and phase_key != previous_phase:
                # Soft contract preflight on every phase boundary (A5).
                current_phase = getattr(bot.day_plan_task, "current_phase", None)
                preflight = (
                    preflight_phase_contract(current_phase, ram=ram)
                    if current_phase is not None and hasattr(current_phase, "contract")
                    else None
                )
                _json_write(
                    out_handle,
                    event_row(
                        "phase",
                        snap,
                        changes=changes,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        ram=ram,
                        extras={"contract_preflight": preflight} if preflight else None,
                    ),
                )
                if preflight is not None and not preflight.get("empty") and not preflight.get("ok"):
                    _json_write(
                        out_handle,
                        event_row(
                            "contract_preflight",
                            snap,
                            watches=current_watch,
                            day_plan=bot.day_plan_task,
                            ram=ram,
                            note=(
                                f"soft fail {preflight.get('phase')}: "
                                f"{','.join(preflight.get('reasons') or ())}"
                            ),
                            extras={"contract_preflight": preflight},
                        ),
                    )
            if changes:
                _json_write(
                    out_handle,
                    event_row(
                        "watch",
                        snap,
                        changes=changes,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        ram=ram,
                    ),
                )

            current_task = getattr(bot.day_plan_task, "_current_task", None) if bot.day_plan_started else None
            navigator = getattr(current_task, "_navigator", None)
            stasis = int(getattr(navigator, "stasis", 0)) if navigator is not None else 0
            if stasis >= args.stasis_threshold and frame - last_stasis_report >= args.stasis_report_interval:
                last_stasis_report = frame
                _json_write(
                    out_handle,
                    event_row(
                        "stasis",
                        snap,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        task=current_task,
                        ram=ram,
                    ),
                )

            if bot.disable_reason:
                _json_write(
                    out_handle,
                    event_row(
                        "disabled",
                        snap,
                        watches=current_watch,
                        day_plan=bot.day_plan_task,
                        note=bot.disable_reason,
                        ram=ram,
                    ),
                )
                return 0

            previous_watch = current_watch
            previous_tilemap = snap.tilemap
            previous_phase = phase_key

        end_ram = env.get_ram()
        snap = snapshot_from_ram(end_ram, frame=args.max_frames, action=np.zeros(12, dtype=np.int32))
        _json_write(
            out_handle,
            event_row(
                "max_frames",
                snap,
                watches=watch_values(end_ram, fields),
                day_plan=bot.day_plan_task,
                ram=end_ram,
            ),
        )
        return 0
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a headless autonomous day-plan probe and emit JSONL diagnostics.")
    parser.add_argument("--state", default="latest", help="Start save state")
    parser.add_argument("--day-plan", default=None, help="Named day plan; omit for auto")
    parser.add_argument("--resume-water", action="store_true", help="Use the resume-water day plan")
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--watch", action="append", help="Comma-separated RAM fields to watch; repeatable")
    parser.add_argument("--human-range", action="append", default=[], help="Simulate human control for frame range START:END; repeatable")
    parser.add_argument("--stasis-threshold", type=int, default=90)
    parser.add_argument("--stasis-report-interval", type=int, default=60)
    parser.add_argument("--out", help="Write JSONL to path instead of stdout")
    args = parser.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
