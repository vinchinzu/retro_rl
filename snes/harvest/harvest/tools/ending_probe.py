#!/usr/bin/env python3
"""Drive the Harvest Moon ending and capture evaluation RAM fields."""

from __future__ import annotations

import argparse
import gzip
import json
from typing import Any

import numpy as np

from retro_harness import TaskStatus, WorldState

from harvest.planner.day_plan_tasks import GoToSleepTask, ReturnHomeTask
from harvest.tasks.nav import make_action
from harvest.core.ram_catalog import LiveRamEditor, parse_ram_patches, read_ram_value
from harvest.runtime.retro_setup import STATES_DIR, make_harvest_env


CAPTURE_FIELDS = (
    "year",
    "season",
    "weekday",
    "day",
    "hour",
    "minute",
    "tilemap",
    "input_lock",
    "dialog_text_id",
    "ending_scene_index",
    "ending_aux_scene_index",
    "happiness",
    "development_rate",
    "dog_hugs",
    "ranch_mastery",
    "ranch_development",
    "money_raw",
    "num_cows",
    "num_chickens",
    "maria_hearts",
    "ann_hearts",
    "nina_hearts",
    "ellen_hearts",
    "eve_hearts",
    "shipped_corn",
    "shipped_tomatoes",
    "shipped_turnips",
    "shipped_potatoes",
    "power_berries",
    "power_berry_count",
    "max_stamina",
    "upgrade_flags",
    "marriage_flags",
    "incubator_flags",
    "family_event_flags",
)


def _snapshot(ram: np.ndarray, frame: int) -> dict[str, int]:
    row = {"frame": frame}
    for field in CAPTURE_FIELDS:
        row[field] = read_ram_value(ram, field, raw=True)
    return row


def _interesting_key(snapshot: dict[str, int]) -> tuple[int, ...]:
    return (
        snapshot["year"],
        snapshot["season"],
        snapshot["day"],
        snapshot["hour"],
        snapshot["minute"],
        snapshot["tilemap"],
        snapshot["input_lock"],
        snapshot["dialog_text_id"],
        snapshot["ending_scene_index"],
        snapshot["ending_aux_scene_index"],
        snapshot["ranch_mastery"],
        snapshot["ranch_development"],
    )


def _compact(snapshot: dict[str, int]) -> dict[str, int]:
    keys = (
        "frame",
        "year",
        "season",
        "day",
        "hour",
        "minute",
        "tilemap",
        "dialog_text_id",
        "ending_scene_index",
        "happiness",
        "ranch_mastery",
        "ranch_development",
    )
    return {key: snapshot[key] for key in keys}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    patches = parse_ram_patches(args.ram_set)
    env = make_harvest_env(args.state)
    editor = LiveRamEditor(env)

    obs, info = env.reset()
    initial_ram = np.asarray(env.get_ram(), dtype=np.uint8)
    initial = _snapshot(initial_ram, 0)
    snapshots = [initial]
    seen_indices: list[int] = []
    seen_set: set[int] = set()

    phase = "return_home"
    task = ReturnHomeTask()
    task.reset(WorldState(0, initial_ram, info, obs))
    sleeps_completed = 0
    last_key = _interesting_key(initial)
    final = initial

    print(f"[ENDING] state={args.state} initial={_compact(initial)}")

    for frame in range(1, args.max_frames + 1):
        if patches:
            editor.apply(patches)

        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        world = WorldState(frame, ram, info, obs)
        if phase in {"return_home", "sleep"}:
            result = task.step(world)
            action = result.action.action if result.action is not None else make_action()
            if result.status == TaskStatus.SUCCESS:
                snap = _snapshot(ram, frame)
                print(f"[ENDING] {phase} -> success frame={frame} {result.reason or ''} {_compact(snap)}")
                if phase == "return_home":
                    phase = "sleep"
                    task = GoToSleepTask()
                    task.reset(world)
                else:
                    sleeps_completed += 1
                    if sleeps_completed >= args.sleep_count:
                        phase = "credits"
                        action = make_action(a=True)
                    else:
                        task = GoToSleepTask()
                        task.reset(world)
            elif result.status == TaskStatus.FAILURE:
                snap = _snapshot(ram, frame)
                print(f"[ENDING] {phase} -> credits frame={frame} reason={result.reason or ''} {_compact(snap)}")
                phase = "credits"
                action = make_action(a=True)
        else:
            action = make_action(a=(frame % args.a_pulse != 0))

        obs, _reward, _terminated, _truncated, info = env.step(action)

        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        snap = _snapshot(ram, frame)
        final = snap
        ending_index = snap["ending_scene_index"]
        if ending_index and ending_index not in seen_set:
            seen_set.add(ending_index)
            seen_indices.append(ending_index)
            snapshots.append(snap)
            print(f"[ENDING] scene=0x{ending_index:02x} {_compact(snap)}")
        key = _interesting_key(snap)
        if key != last_key and (
            frame < args.startup_frames
            or ending_index
            or frame % args.capture_interval == 0
        ):
            snapshots.append(snap)
            last_key = key

        if (
            phase == "credits"
            and frame >= args.min_credits_frames
            and ending_index >= args.stop_ending_index
        ):
            break

    if args.save_end:
        save_path = STATES_DIR / f"{args.state}_ending_probe_end.state"
        with gzip.open(save_path, "wb") as handle:
            handle.write(env.em.get_state())
        print(f"[ENDING] saved_end={save_path}")

    env.close()
    return {
        "state": args.state,
        "max_frames": args.max_frames,
        "sleep_count": args.sleep_count,
        "ram_set": args.ram_set,
        "seen_ending_indices": seen_indices,
        "initial": initial,
        "final": final,
        "snapshots": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="latest", help="Save state name to load")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--sleep-count", type=int, default=1, help="Number of successful sleeps before A-advance mode")
    parser.add_argument("--a-pulse", type=int, default=3, help="Release A every Nth frame while advancing credits")
    parser.add_argument("--startup-frames", type=int, default=250)
    parser.add_argument("--capture-interval", type=int, default=600)
    parser.add_argument("--min-credits-frames", type=int, default=5000)
    parser.add_argument("--stop-ending-index", type=lambda text: int(text, 0), default=0x25)
    parser.add_argument("--ram-set", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--save-end", action="store_true")
    parser.add_argument("--out", help="Write capture JSON; default debug_alignment/ending_probe_<state>.json")
    args = parser.parse_args()

    result = run_probe(args)
    out_path = Path(args.out) if args.out else Path("debug_alignment") / f"ending_probe_{args.state}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[ENDING] wrote={out_path}")
    print(f"[ENDING] seen={','.join(f'0x{value:02x}' for value in result['seen_ending_indices'])}")
    print(f"[ENDING] final={_compact(result['final'])}")


if __name__ == "__main__":
    main()
