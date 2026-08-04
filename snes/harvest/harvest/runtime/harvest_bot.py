#!/usr/bin/env python3
"""Harvest Moon SNES Bot CLI entry point.

Usage:
    python -m harvest.runtime.harvest_bot play --autoplay --state latest
    python -m harvest.runtime.harvest_bot list
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import List, Optional

import stable_retro as retro

from harvest.core.npc_catalog import dialogue_catalog, npc_snapshot_dict
from harvest.core.ram_catalog import LiveRamEditor, RamPatch, parse_ram_patches
from harvest.core.world_snapshot import load_state_ram, parse_bounds as parse_world_bounds, world_snapshot_dict
from harvest.paths import CUSTOM_INTEGRATIONS_DIR, PROJECT_DIR, SAVES_DIR as PROJECT_SAVES_DIR
from harvest.planner.day_plan import PHASE_SEQUENCES
from harvest.runtime.autoplay_bot import AutoClearBot
from harvest.runtime.game_state import GameState
from harvest.runtime.play_session import PlaySession
from harvest.runtime.retro_setup import register_harvest_integration
from harvest.tasks.farm_clearer import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    parse_priority_list,
)

SCRIPT_DIR = os.fspath(PROJECT_DIR)
STATES_DIR = os.path.join(os.fspath(CUSTOM_INTEGRATIONS_DIR), "HarvestMoon-Snes")
SAVES_DIR = os.fspath(PROJECT_SAVES_DIR)

os.makedirs(SAVES_DIR, exist_ok=True)
register_harvest_integration(retro, require_rom=False)


def list_states() -> None:
    states = sorted(glob.glob(os.path.join(STATES_DIR, "*.state")))
    print("\nSAVE STATES:")
    for path in states:
        name = os.path.basename(path).replace(".state", "")
        dt = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        print(f"  {name} ({dt})")
    print(f"Total: {len(states)}")


def list_ram_fields() -> None:
    print("\nRAM FIELDS:")
    for spec in SCALAR_FIELDS:
        aliases = f" aliases={','.join(spec.aliases)}" if spec.aliases else ""
        scale = f" x{spec.display_multiplier}" if spec.display_multiplier != 1 else ""
        print(
            f"  {spec.key:20s} 0x{spec.address:05X} {spec.kind:3s} "
            f"{spec.section:16s} {spec.source}{scale}{aliases}"
        )
    print(f"Total: {len(SCALAR_FIELDS)}")


def resolve_day_plan_name(
    day_plan: Optional[str],
    resume_water: bool,
    state_name: Optional[str],
) -> Optional[str]:
    del state_name  # reserved for future auto overrides
    if resume_water:
        if day_plan and day_plan != "resume_water":
            raise ValueError("--resume-water cannot be combined with a different --day-plan")
        return "resume_water"
    return day_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Moon Bot")
    subparsers = parser.add_subparsers(dest="command")

    play = subparsers.add_parser("play")
    play.add_argument("--state", type=str)
    play.add_argument("--scale", type=int, default=2)
    play.add_argument("--hud-width", type=int, default=176)
    play.add_argument("--autoplay", action="store_true")
    play.add_argument("--priority", type=str)
    play.add_argument("--priority-only", action="store_true")
    play.add_argument("--fence-only", action="store_true", help="Only clear fences then stop")
    play.add_argument("--grass", action="store_true", help="Enable grass planting (till + plant)")
    play.add_argument("--till-only", action="store_true", help="Only till, skip planting")
    play.add_argument(
        "--grass-bounds",
        type=str,
        default=None,
        help="Custom bounds x1,y1,x2,y2 (default: right half)",
    )
    play.add_argument(
        "--grass-no-go",
        type=str,
        default=None,
        help="No-go rects: x1,y1,x2,y2;x1,y1,x2,y2 (areas to skip)",
    )
    play.add_argument("--crop", action="store_true", help="Crop mode: detect plots, plant + water")
    play.add_argument("--seed", type=str, default="potato", help="Seed type (potato, turnip, corn, tomato)")
    play.add_argument(
        "--no-day-plan",
        action="store_true",
        help="Disable day plan (default: on unless --crop/--grass/--fence-only)",
    )
    play.add_argument(
        "--day-plan",
        type=str,
        default=None,
        choices=list(PHASE_SEQUENCES.keys()),
        help="Select day plan sequence (default: auto from live/save state)",
    )
    play.add_argument(
        "--resume-water",
        action="store_true",
        help="Exit farm building, fetch watering can, route to crop field",
    )
    play.add_argument(
        "--until-day",
        type=int,
        default=None,
        help="Run day plans until the morning after this in-season day",
    )
    play.add_argument(
        "--until-season",
        type=int,
        default=None,
        help="Season index for multi-day runs (0=spring, 1=summer, 2=fall, 3=winter)",
    )
    play.add_argument("--days", type=int, default=None, help="Run this many full in-game days")
    play.add_argument(
        "--end-of-spring",
        action="store_true",
        help="Shortcut for --until-season 0 --until-day 30",
    )
    play.add_argument(
        "--eve-loop",
        action="store_true",
        help='Run the recorded Eve "No" dialogue loop from the bar exterior',
    )
    play.add_argument(
        "--eve-target-hearts",
        type=int,
        default=10,
        help="Target Eve heart tier for --eve-loop (1-10, default: 10)",
    )
    play.add_argument("--money", type=int, default=None, help="Set money each frame (e.g. --money 7000)")
    play.add_argument(
        "--ram-set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Hot-edit a named RAM field every frame",
    )
    play.add_argument("--save-end", action="store_true", help="Save state when task completes")
    play.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (testing)")
    play.add_argument(
        "--diagnostics-dir",
        type=str,
        default=None,
        help="Directory for autoplay watchdog screenshots, states, and JSON logs",
    )
    play.add_argument(
        "--watchdog-frames",
        type=int,
        default=None,
        help="Force end-of-day after N frames without progress (0 disables)",
    )
    play.add_argument(
        "--keep-running-after-disable",
        action="store_true",
        help="Keep the emulator loop alive after autoplay disables the bot",
    )
    play.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="NAME",
        help="Record inputs as a task (F5 to save)",
    )

    subparsers.add_parser("list")
    subparsers.add_parser("ram-fields", help="List named RAM fields usable with --ram-set")
    world = subparsers.add_parser(
        "world",
        aliases=["world-snapshot"],
        help="Export a RAM-backed world snapshot for a save state",
    )
    world.add_argument("--state", default="latest", help="Save state name")
    world.add_argument("--bounds", help="x_min,y_min,x_max,y_max; default full 64x64 map")
    world.add_argument("--compact", action="store_true", help="Omit the full scalar table")
    world.add_argument("--grid", action="store_true", help="Include raw tile grid")
    world.add_argument("--out", help="Write JSON to this path")
    npc = subparsers.add_parser(
        "npc",
        aliases=["npc-snapshot"],
        help="Export dynamic game-object/NPC positions and decoded NPC status flags",
    )
    npc.add_argument("--state", default="latest", help="Save state name")
    npc.add_argument("--dialogue-text", action="store_true", help="Include decoded ROM dialogue text")
    npc.add_argument("--npc", dest="npc_name", help="Filter dialogue text to one NPC/group")
    npc.add_argument("--compact", action="store_true", help="Omit dialogue body text")
    npc.add_argument("--out", help="Write JSON to this path")
    dialogue = subparsers.add_parser(
        "dialogue",
        aliases=["npc-dialogue"],
        help="Export decoded ROM dialogue groups",
    )
    dialogue.add_argument("--npc", dest="npc_name", help="Filter to one NPC/group, e.g. maria or eve")
    dialogue.add_argument("--compact", action="store_true", help="Omit dialogue body text")
    dialogue.add_argument("--out", help="Write JSON to this path")

    args = parser.parse_args()

    if args.command == "play":
        try:
            day_plan_name = resolve_day_plan_name(args.day_plan, args.resume_water, args.state)
        except ValueError as exc:
            parser.error(str(exc))
        if args.eve_loop:
            day_plan_name = "eve_loop"

        priority = parse_priority_list(
            getattr(args, "priority", None),
            getattr(args, "priority_only", False),
        )
        grass_bounds = None
        if args.grass_bounds:
            parts = [int(x.strip()) for x in args.grass_bounds.split(",")]
            if len(parts) == 4:
                grass_bounds = tuple(parts)
        grass_no_go = None
        if args.grass_no_go:
            grass_no_go = []
            for rect_str in args.grass_no_go.split(";"):
                rect_str = rect_str.strip()
                if not rect_str:
                    continue
                parts = [int(x.strip()) for x in rect_str.split(",")]
                if len(parts) == 4:
                    grass_no_go.append(tuple(parts))
        until_day = args.until_day
        until_season = args.until_season
        if args.end_of_spring:
            until_day = 30
            until_season = 0
        if args.days is not None and (until_day is not None or until_season is not None):
            parser.error("--days cannot be combined with --until-day/--until-season/--end-of-spring")
        if args.days is not None and args.days < 1:
            parser.error("--days must be at least 1")
        try:
            ram_patches = parse_ram_patches(args.ram_set)
        except ValueError as exc:
            parser.error(str(exc))
        bot = AutoClearBot(
            priority=priority,
            clear_fences_only=bool(args.fence_only),
            grass_enabled=bool(args.grass) or bool(args.till_only),
            till_only=bool(args.till_only),
            grass_bounds=grass_bounds,
            grass_no_go=grass_no_go,
            crop_enabled=bool(args.crop),
            crop_seed_type=args.seed,
            day_plan_enabled=bool(day_plan_name)
            or not (
                args.no_day_plan
                or args.crop
                or args.grass
                or args.till_only
                or args.fence_only
            ),
            day_plan_sequence=day_plan_name,
            auto_day_plan_state_name=args.state,
            multi_day_until_day=until_day,
            multi_day_until_season=until_season,
            multi_day_count=args.days,
            eve_target_hearts=args.eve_target_hearts,
        )
        PlaySession(
            state=args.state,
            scale=args.scale,
            bot=bot,
            autoplay=args.autoplay,
            max_frames=args.max_frames,
            record_name=args.record,
            save_end=bool(getattr(args, "save_end", False)),
            money_hack=args.money,
            ram_patches=ram_patches,
            hud_width=args.hud_width,
            diagnostics_dir=args.diagnostics_dir,
            watchdog_frames=args.watchdog_frames,
            exit_on_bot_disable=not args.keep_running_after_disable,
        ).run()
    elif args.command == "list":
        list_states()
    elif args.command == "ram-fields":
        list_ram_fields()
    elif args.command in ("world", "world-snapshot"):
        try:
            bounds = parse_world_bounds(args.bounds)
        except ValueError as exc:
            parser.error(str(exc))
        data = world_snapshot_dict(
            load_state_ram(args.state),
            bounds=bounds,
            include_grid=bool(args.grid),
            compact=bool(args.compact),
        )
        text = json.dumps(data, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            print(f"Wrote world snapshot to {args.out}")
        else:
            print(text)
    elif args.command in ("npc", "npc-snapshot"):
        data = npc_snapshot_dict(
            load_state_ram(args.state),
            include_dialogue_text=bool(args.dialogue_text),
            npc=args.npc_name,
            compact=bool(args.compact),
        )
        text = json.dumps(data, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            print(f"Wrote NPC snapshot to {args.out}")
        else:
            print(text)
    elif args.command in ("dialogue", "npc-dialogue"):
        data = dialogue_catalog(npc=args.npc_name, compact=bool(args.compact))
        text = json.dumps(data, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            print(f"Wrote dialogue catalog to {args.out}")
        else:
            print(text)
    else:
        parser.print_help()


__all__ = [
    "ADDR_INPUT_LOCK",
    "ADDR_TILEMAP",
    "AutoClearBot",
    "GameState",
    "PlaySession",
    "main",
]


if __name__ == "__main__":
    main()
