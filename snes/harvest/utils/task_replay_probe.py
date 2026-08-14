#!/usr/bin/env python3
"""Headless recorded-task replay probe.

Replays a task JSON and reports watched RAM changes with nearby input context.
This is meant for discovering stand tiles, facing directions, and verification
RAM fields from a human recording before writing an autonomous task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPT_DIR.parent

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import stable_retro as retro

from harvest.runtime.probe_utils import event_row, parse_field_list, snapshot_from_ram, watch_changes, watch_values
from harvest.runtime.recording_trace import pressed_buttons
from harvest.runtime.retro_setup import register_harvest_integration

TASKS_DIR = SCRIPT_DIR / "tasks"

def _json_write(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()

def _load_task(task_name: str) -> dict[str, object]:
    path = Path(task_name)
    if path.suffix != ".json":
        path = TASKS_DIR / f"{task_name}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def _nearby_input_context(history: Sequence[dict[str, object]]) -> dict[str, object]:
    last_a = None
    last_move = None
    for row in reversed(history):
        buttons = set(row.get("buttons", []))
        if last_a is None and "A" in buttons:
            last_a = row
        if last_move is None and buttons & {"Up", "Down", "Left", "Right"}:
            last_move = row
        if last_a is not None and last_move is not None:
            break
    return {
        "last_a": last_a,
        "last_move": last_move,
    }

def run_probe(args: argparse.Namespace) -> int:
    fields = parse_field_list(args.watch)
    data = _load_task(args.task)
    frames = data["frames"]
    start_state = args.state or data.get("start_state") or "latest"

    if start_state == "latest" and not args.allow_mutable_latest:
        raise RuntimeError(
            "Refusing to replay from mutable start_state=latest. "
            "Pass --state <stable_state> or --allow-mutable-latest."
        )

    register_harvest_integration(retro, require_rom=True)
    env = retro.make(
        game="HarvestMoon-Snes",
        state=start_state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
    )

    out_path = Path(args.out) if args.out else None
    out_handle = out_path.open("w", encoding="utf-8") if out_path else sys.stdout
    history: deque[dict[str, object]] = deque(maxlen=max(1, args.lookback))
    try:
        env.reset()
        previous_watch = None
        previous_tilemap = None
        last_report = -10_000

        for frame, action_list in enumerate(frames):
            action = np.asarray(action_list, dtype=np.int32)
            _obs, _reward, _terminated, _truncated, _info = env.step(action)
            ram = env.get_ram()
            snap = snapshot_from_ram(ram, frame=frame, action=action)
            current_watch = watch_values(ram, fields)
            changes = watch_changes(previous_watch, current_watch)
            context_row = {
                **snap.as_event(),
                "buttons": pressed_buttons(action),
            }
            history.append(context_row)

            if frame == 0:
                _json_write(out_handle, event_row("start", snap, watches=current_watch, note=f"state={start_state}"))
            if previous_tilemap is not None and snap.tilemap != previous_tilemap:
                _json_write(out_handle, event_row("tilemap", snap, watches=current_watch, changes=changes))
            if changes:
                row = event_row("watch", snap, watches=current_watch, changes=changes)
                row["context"] = _nearby_input_context(list(history))
                _json_write(out_handle, row)

            moving = any(button in snap.buttons for button in ("Up", "Down", "Left", "Right"))
            if args.report_inputs and moving and frame - last_report >= args.input_report_interval:
                last_report = frame
                _json_write(out_handle, event_row("input", snap, watches=current_watch))

            previous_watch = current_watch
            previous_tilemap = snap.tilemap

        snap = snapshot_from_ram(env.get_ram(), frame=len(frames), action=np.zeros(12, dtype=np.int32))
        _json_write(out_handle, event_row("end", snap, watches=watch_values(env.get_ram(), fields)))
        return 0
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()
        env.close()

def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded task headlessly and emit RAM/action diagnostics.")
    parser.add_argument("task", help="Task name or JSON path")
    parser.add_argument("--state", help="Override start state")
    parser.add_argument("--allow-mutable-latest", action="store_true", help="Allow replaying from current latest.state")
    parser.add_argument("--watch", action="append", help="Comma-separated RAM fields to watch; repeatable")
    parser.add_argument("--lookback", type=int, default=90, help="Frames of input context attached to watch changes")
    parser.add_argument("--report-inputs", action="store_true", help="Also emit periodic movement/input rows")
    parser.add_argument("--input-report-interval", type=int, default=120)
    parser.add_argument("--out", help="Write JSONL to path instead of stdout")
    args = parser.parse_args()
    return run_probe(args)

if __name__ == "__main__":
    raise SystemExit(main())
