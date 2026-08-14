#!/usr/bin/env python3
"""Replay a recorded task and save a focused RAM trace.

This is intended for reverse-engineering short interactions such as TV weather
checks. It records a compact per-frame CSV plus a JSON summary of the watched
addresses and their value windows, instead of dumping the full emulator RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import stable_retro as retro

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_TOOL,
)
from harvest.tasks.nav import get_pos_from_ram
from harvest.core.harvest_state import SCALAR_FIELDS, SCALAR_FIELDS_BY_KEY, WEATHER_CODES

INTEGRATION_PATH = ROOT_DIR / "custom_integrations"
TASKS_DIR = ROOT_DIR / "tasks"

BUTTON_NAMES = {
    0: "B",
    1: "Y",
    2: "Select",
    3: "Start",
    4: "Up",
    5: "Down",
    6: "Left",
    7: "Right",
    8: "A",
    9: "X",
    10: "L",
    11: "R",
}

# Low WRAM bytes below are still inferred. The replay summary makes that clear.
DEFAULT_WATCHES = {
    0x0022: "tilemap",
    0x019A: "input_lock",
    0x019B: "inferred_dialog_substate",
    0x019E: "inferred_dialog_flag_019E",
    0x01A6: "inferred_player_y_mirror",
    0x01A8: "inferred_tv_state_01A8",
    0x01A9: "inferred_tv_state_01A9",
    0x01AA: "inferred_tv_state_01AA",
    0x01AB: "inferred_tv_state_01AB",
    0x01AC: "inferred_tv_state_01AC",
    0x01AD: "inferred_tv_state_01AD",
    0x01AE: "inferred_tv_state_01AE",
    0x01B0: "inferred_tv_state_01B0",
    0x01B1: "inferred_tv_state_01B1",
    0x0921: "tool",
    0x098C: "weather_tomorrow",
    0x15F04: "money_lo_raw",
    0x15F05: "money_mid_raw",
    0x15F06: "money_hi_raw",
    0x15F18: "year",
    0x15F19: "season",
    0x15F1A: "weekday",
    0x15F1B: "day",
    0x15F1C: "hour",
    0x15F1D: "minute",
}

DEFAULT_SCAN_RANGES = (
    (0x0000, 0x01FF),
    (0x0900, 0x099F),
    (0x15F00, 0x15F1F),
)

STATE_TO_ENV_ADDR_OFFSET = 0x4000

def parse_address(text: str) -> int:
    return int(text, 0)

def parse_range(text: str) -> tuple[int, int]:
    start_text, end_text = text.split(":", 1)
    start = parse_address(start_text)
    end = parse_address(end_text)
    if end < start:
        raise ValueError(f"Invalid range {text!r}: end before start")
    return start, end

def state_addr_to_env_addr(addr: int) -> int:
    """Translate save-state/editor addresses to env.get_ram() addresses."""
    return addr + STATE_TO_ENV_ADDR_OFFSET if addr >= 0x10000 else addr

def parse_watch_args(values: list[str]) -> dict[int, str]:
    watches = dict(DEFAULT_WATCHES)
    for item in values:
        if "=" in item:
            addr_text, label = item.split("=", 1)
            watches[parse_address(addr_text)] = label.strip()
        else:
            addr = parse_address(item)
            watches[addr] = f"addr_{addr:04X}"
    return dict(sorted(watches.items()))

def parse_watch_field_args(values: list[str], watches: dict[int, str] | None = None) -> dict[int, str]:
    resolved = dict(DEFAULT_WATCHES if watches is None else watches)
    for key in values:
        spec = SCALAR_FIELDS_BY_KEY.get(key)
        if spec is None:
            raise ValueError(f"Unknown scalar field {key!r}")
        resolved[state_addr_to_env_addr(spec.address)] = spec.key
    return dict(sorted(resolved.items()))

def parse_watch_section_args(values: list[str], watches: dict[int, str] | None = None) -> dict[int, str]:
    resolved = dict(DEFAULT_WATCHES if watches is None else watches)
    known_sections = {spec.section.lower(): spec.section for spec in SCALAR_FIELDS}
    for section_name in values:
        section_key = section_name.strip().lower()
        if section_key not in known_sections:
            raise ValueError(f"Unknown scalar section {section_name!r}")
        for spec in SCALAR_FIELDS:
            if spec.section.lower() != section_key:
                continue
            resolved[state_addr_to_env_addr(spec.address)] = spec.key
    return dict(sorted(resolved.items()))

def coalesce_frame_windows(frames: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(frames)
    if not ordered:
        return []

    windows: list[dict[str, int]] = []
    start = ordered[0]
    end = start
    for frame in ordered[1:]:
        if frame == end + 1:
            end = frame
            continue
        windows.append({"start": start, "end": end, "length": end - start + 1})
        start = end = frame
    windows.append({"start": start, "end": end, "length": end - start + 1})
    return windows

def pressed_buttons(frame: list[int]) -> list[str]:
    return [name for idx, name in BUTTON_NAMES.items() if idx < len(frame) and frame[idx]]

def coalesce_action_runs(frames: list[list[int]], frame_offset: int = 0) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    start = None
    last_buttons: list[str] | None = None

    for idx, frame in enumerate(frames):
        frame_idx = idx + frame_offset
        buttons = pressed_buttons(frame)
        if not buttons:
            if start is not None and last_buttons is not None:
                runs.append(
                    {
                        "start": start,
                        "end": frame_idx - 1,
                        "length": frame_idx - start,
                        "buttons": last_buttons,
                    }
                )
                start = None
                last_buttons = None
            continue

        if start is None:
            start = frame_idx
            last_buttons = buttons
            continue

        if buttons != last_buttons:
            runs.append(
                {
                    "start": start,
                    "end": frame_idx - 1,
                    "length": frame_idx - start,
                    "buttons": last_buttons,
                }
            )
            start = frame_idx
            last_buttons = buttons

    if start is not None and last_buttons is not None:
        runs.append(
            {
                "start": start,
                "end": frame_offset + len(frames) - 1,
                "length": frame_offset + len(frames) - start,
                "buttons": last_buttons,
            }
        )
    return runs

def read_u24(ram: np.ndarray, addr: int) -> int:
    if addr + 2 >= len(ram):
        return 0
    return int(ram[addr]) | (int(ram[addr + 1]) << 8) | (int(ram[addr + 2]) << 16)

def read_u16(ram: np.ndarray, addr: int) -> int:
    if addr + 1 >= len(ram):
        return 0
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)

def value_at(ram: np.ndarray, addr: int) -> int | None:
    if addr >= len(ram):
        return None
    return int(ram[addr])

def watch_value(ram: np.ndarray, addr: int, label: str) -> int | None:
    """Read a watched value, decoding named scalar fields with their real width."""
    spec = SCALAR_FIELDS_BY_KEY.get(label)
    if spec is None or state_addr_to_env_addr(spec.address) != addr:
        return value_at(ram, addr)
    if spec.kind == "u8":
        return value_at(ram, addr)
    if spec.kind == "u16":
        return read_u16(ram, addr)
    if spec.kind == "u24":
        value = read_u24(ram, addr)
        # Harvest money fields are stored in value/10 units; report gold for planning.
        if spec.key in {"money", "shipping_money"}:
            return value * 10
        return value
    return value_at(ram, addr)

def weather_label(code: int | None) -> str | None:
    if code is None:
        return None
    return WEATHER_CODES.get(code, f"{code} unknown")

def snapshot_from_ram(ram: np.ndarray) -> dict[str, object]:
    pos = get_pos_from_ram(ram)
    weather = value_at(ram, 0x098C)
    tilemap = value_at(ram, ADDR_TILEMAP)
    hour = value_at(ram, 0x15F1C)
    minute = value_at(ram, 0x15F1D)
    tool = value_at(ram, ADDR_TOOL)
    return {
        "tilemap": tilemap,
        "player_x": pos.x,
        "player_y": pos.y,
        "player_tile_x": pos.x // 16,
        "player_tile_y": pos.y // 16,
        "tool": tool,
        "weather_tomorrow": weather,
        "weather_tomorrow_label": weather_label(weather),
        "hour": hour,
        "minute": minute,
        "money": read_u24(ram, 0x15F04) * 10,
    }

def make_env(state: str):
    retro.data.Integrations.add_custom_path(str(INTEGRATION_PATH.resolve()))
    return retro.make(
        game="HarvestMoon-Snes",
        state=state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
    )

def load_task(task_name: str) -> dict[str, object]:
    path = TASKS_DIR / f"{task_name}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def capture_frames(
    env,
    frames: list[list[int]],
    watches: dict[int, str],
    scan_ranges: list[tuple[int, int]],
    dialog_addr: int,
    dialog_active_value: int,
    top_n: int,
    frame_offset: int = 0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    base_ram = np.array(env.get_ram(), dtype=np.uint8)
    start_snapshot = snapshot_from_ram(base_ram)

    watch_history: dict[int, list[int | None]] = {addr: [] for addr in watches}
    dialog_frames: list[int] = []
    trace_rows: list[dict[str, object]] = []

    scan_addresses = [
        addr
        for start, end in scan_ranges
        for addr in range(start, end + 1)
        if addr < len(base_ram)
    ]
    scan_counts = {addr: 0 for addr in scan_addresses}
    first_last: dict[int, tuple[int, int]] = {}

    for frame_idx, frame in enumerate(frames):
        task_frame = frame_offset + frame_idx
        action = np.array(frame, dtype=np.int32)
        env.step(action)
        ram = np.array(env.get_ram(), dtype=np.uint8)

        pos = get_pos_from_ram(ram)
        row = {
            "frame": frame_idx,
            "task_frame": task_frame,
            "player_x": pos.x,
            "player_y": pos.y,
            "player_tile_x": pos.x // 16,
            "player_tile_y": pos.y // 16,
            "money": read_u24(ram, 0x15F04) * 10,
        }
        for addr, label in watches.items():
            value = watch_value(ram, addr, label)
            watch_history[addr].append(value)
            row[label] = value

        if row.get(watches.get(dialog_addr, "")) == dialog_active_value:
            dialog_frames.append(task_frame)
        elif value_at(ram, dialog_addr) == dialog_active_value:
            dialog_frames.append(task_frame)

        for addr in scan_addresses:
            current = int(ram[addr])
            if current == int(base_ram[addr]):
                continue
            scan_counts[addr] += 1
            first, _ = first_last.get(addr, (task_frame, task_frame))
            first_last[addr] = (first, task_frame)

        trace_rows.append(row)

    final_ram = np.array(env.get_ram(), dtype=np.uint8)
    end_snapshot = snapshot_from_ram(final_ram)

    summary = {
        "frame_count": len(frames),
        "capture_start_frame": frame_offset,
        "capture_end_frame": frame_offset + max(len(frames) - 1, 0),
        "recorded_input_runs": coalesce_action_runs(frames, frame_offset=frame_offset),
        "dialog_addr": f"0x{dialog_addr:04X}",
        "dialog_active_value": dialog_active_value,
        "dialog_windows": coalesce_frame_windows(dialog_frames),
        "start": start_snapshot,
        "end": end_snapshot,
        "watch_summary": [
            summarize_watch(
                addr=addr,
                label=label,
                base_value=watch_value(base_ram, addr, label),
                final_value=watch_value(final_ram, addr, label),
                values=watch_history[addr],
                frame_offset=frame_offset,
            )
            for addr, label in watches.items()
        ],
        "top_scan_changes": summarize_scan(
            base_ram=base_ram,
            final_ram=final_ram,
            scan_counts=scan_counts,
            first_last=first_last,
            watches=watches,
            top_n=top_n,
        ),
        "notes": [
            "Labels prefixed with inferred_ are replay-based observations, not confirmed decomp names.",
            "weather_tomorrow is the documented save/RAM byte used by the editor and planning tools.",
        ],
    }
    return summary, trace_rows

def write_capture_output(
    out_dir: Path,
    summary: dict[str, object],
    trace_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = out_dir / "trace.csv"
    fieldnames = list(trace_rows[0].keys()) if trace_rows else ["frame", "task_frame"]
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path, trace_path

def summarize_watch(
    addr: int,
    label: str,
    base_value: int | None,
    final_value: int | None,
    values: list[int | None],
    frame_offset: int = 0,
) -> dict[str, object]:
    changed_frames = [frame_offset + idx for idx, value in enumerate(values) if value != base_value]
    unique_values = sorted({value for value in values if value is not None})
    return {
        "address": f"0x{addr:04X}",
        "label": label,
        "base": base_value,
        "final": final_value,
        "unique_values": unique_values,
        "changed_frame_count": len(changed_frames),
        "change_windows": coalesce_frame_windows(changed_frames),
    }

def summarize_scan(
    base_ram: np.ndarray,
    final_ram: np.ndarray,
    scan_counts: dict[int, int],
    first_last: dict[int, tuple[int, int]],
    watches: dict[int, str],
    top_n: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for addr, count in scan_counts.items():
        if count <= 0:
            continue
        first, last = first_last[addr]
        rows.append(
            {
                "address": f"0x{addr:04X}",
                "label": watches.get(addr),
                "changed_frame_count": count,
                "first_changed_frame": first,
                "last_changed_frame": last,
                "base": int(base_ram[addr]),
                "final": int(final_ram[addr]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["changed_frame_count"],
            row["last_changed_frame"],
            row["address"],
        ),
        reverse=True,
    )
    return rows[:top_n]

def capture_task(
    task_name: str,
    state_name: str,
    out_dir: Path,
    watches: dict[int, str],
    scan_ranges: list[tuple[int, int]],
    dialog_addr: int,
    dialog_active_value: int,
    top_n: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[Path, Path]:
    task = load_task(task_name)
    all_frames: list[list[int]] = task["frames"]
    capture_start = max(0, start_frame)
    capture_end = len(all_frames) if end_frame is None else min(max(capture_start, end_frame), len(all_frames))
    frames = all_frames[capture_start:capture_end]

    env = make_env(state_name)
    try:
        env.reset()
        for frame in all_frames[:capture_start]:
            env.step(np.array(frame, dtype=np.int32))
        summary, trace_rows = capture_frames(
            env=env,
            frames=frames,
            watches=watches,
            scan_ranges=scan_ranges,
            dialog_addr=dialog_addr,
            dialog_active_value=dialog_active_value,
            top_n=top_n,
            frame_offset=capture_start,
        )
    finally:
        env.close()

    summary["task_name"] = task_name
    summary["state_name"] = state_name
    return write_capture_output(out_dir=out_dir, summary=summary, trace_rows=trace_rows)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a task and save a focused RAM trace")
    parser.add_argument("--task", required=True, help="Recorded task name, without .json")
    parser.add_argument(
        "--state",
        default=None,
        help="Starting save state. Defaults to the task's recorded start_state.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to debug_alignment/ram_capture/<task>_<state>",
    )
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        metavar="ADDR[=LABEL]",
        help="Add a watched address, e.g. 0x019B=dialog_substate",
    )
    parser.add_argument(
        "--watch-field",
        action="append",
        default=[],
        metavar="FIELD_KEY",
        help="Watch a named scalar field from harvest_state.py, e.g. eve_hearts",
    )
    parser.add_argument(
        "--watch-section",
        action="append",
        default=[],
        metavar="SECTION",
        help="Watch every scalar field in a harvest_state.py section, e.g. Romance",
    )
    parser.add_argument(
        "--scan-range",
        action="append",
        default=[],
        metavar="START:END",
        help="Additional scan range for top-change summary, e.g. 0x0100:0x01FF",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Replay task up to this frame before starting capture",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Stop capture before this task frame index",
    )
    parser.add_argument(
        "--dialog-addr",
        default="0x019A",
        help="Address used to infer dialog-active windows",
    )
    parser.add_argument(
        "--dialog-active-value",
        type=int,
        default=2,
        help="Dialog-active byte value at --dialog-addr",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=24,
        help="How many scan-summary addresses to keep",
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    task = load_task(args.task)
    state_name = args.state or task.get("start_state")
    if not state_name:
        parser.error("No state provided and task has no recorded start_state")

    watches = parse_watch_args(args.watch)
    watches = parse_watch_field_args(args.watch_field, watches=watches)
    watches = parse_watch_section_args(args.watch_section, watches=watches)
    scan_ranges = list(DEFAULT_SCAN_RANGES)
    scan_ranges.extend(parse_range(item) for item in args.scan_range)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT_DIR / "debug_alignment" / "ram_capture" / f"{args.task}_{state_name}"
    )

    summary_path, trace_path = capture_task(
        task_name=args.task,
        state_name=state_name,
        out_dir=out_dir,
        watches=watches,
        scan_ranges=scan_ranges,
        dialog_addr=parse_address(args.dialog_addr),
        dialog_active_value=args.dialog_active_value,
        top_n=args.top_n,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    print(f"[RAM_CAPTURE] Summary -> {summary_path}")
    print(f"[RAM_CAPTURE] Trace   -> {trace_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
