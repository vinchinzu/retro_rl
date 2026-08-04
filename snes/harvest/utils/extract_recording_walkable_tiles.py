#!/usr/bin/env python3
"""Extract observed walkable player tiles from a recorded task trace."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harvest.maps.map_config import get_map_name


TASKS_DIR = SCRIPT_DIR / "tasks"
MOVEMENT_BUTTONS = ("Up", "Down", "Left", "Right")


def parse_tilemap(value: str) -> int:
    return int(value, 0)


def load_task(name_or_path: str) -> dict[str, object]:
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = TASKS_DIR / f"{name_or_path}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def button_windows(rows: list[dict[str, object]], button: str) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    start_idx: int | None = None
    end_idx: int | None = None

    def has_button(row: dict[str, object]) -> bool:
        buttons = row.get("buttons", [])
        return isinstance(buttons, list) and button in buttons

    for idx, row in enumerate(rows):
        if has_button(row):
            if start_idx is None:
                start_idx = idx
            end_idx = idx
            continue
        if start_idx is not None and end_idx is not None:
            windows.append(window_summary(rows, start_idx, end_idx))
            start_idx = None
            end_idx = None

    if start_idx is not None and end_idx is not None:
        windows.append(window_summary(rows, start_idx, end_idx))
    return windows


def last_movement_before(rows: list[dict[str, object]], idx: int, lookback: int = 90) -> str | None:
    start = max(0, idx - lookback)
    for row in reversed(rows[start:idx]):
        buttons = row.get("buttons", [])
        if not isinstance(buttons, list):
            continue
        for button in MOVEMENT_BUTTONS:
            if button in buttons:
                return button
    return None


def window_summary(rows: list[dict[str, object]], start_idx: int, end_idx: int) -> dict[str, object]:
    row = rows[start_idx]
    return {
        "start": int(row.get("frame", start_idx)),
        "end": int(rows[end_idx].get("frame", end_idx)),
        "length": end_idx - start_idx + 1,
        "tile": [int(row.get("tx", 0)), int(row.get("ty", 0))],
        "pixel": [int(row.get("x", 0)), int(row.get("y", 0))],
        "tile_id": f"0x{int(row.get('tile_id', 0)):02X}",
        "last_move": last_movement_before(rows, start_idx),
        "input_lock": int(row.get("input_lock", 0)),
    }


def summarize(data: dict[str, object], tilemap: int) -> dict[str, object]:
    trace = data.get("trace", [])
    if not isinstance(trace, list):
        trace = []
    rows = [row for row in trace if isinstance(row, dict) and int(row.get("tm", -1)) == tilemap]

    tiles: dict[tuple[int, int], dict[str, object]] = {}
    tile_ids: dict[int, int] = defaultdict(int)
    for row in rows:
        tile = (int(row.get("tx", 0)), int(row.get("ty", 0)))
        tile_id = int(row.get("tile_id", 0))
        tile_ids[tile_id] += 1
        entry = tiles.setdefault(
            tile,
            {
                "tile": [tile[0], tile[1]],
                "tile_ids": set(),
                "labels": set(),
                "frames": 0,
                "first_frame": int(row.get("frame", 0)),
                "last_frame": int(row.get("frame", 0)),
            },
        )
        entry["tile_ids"].add(tile_id)
        label = row.get("tile_label")
        if label:
            entry["labels"].add(str(label))
        entry["frames"] = int(entry["frames"]) + 1
        entry["last_frame"] = int(row.get("frame", 0))

    observed_tiles = []
    for entry in tiles.values():
        observed_tiles.append(
            {
                "tile": entry["tile"],
                "tile_ids": [f"0x{tile_id:02X}" for tile_id in sorted(entry["tile_ids"])],
                "labels": sorted(entry["labels"]),
                "frames": entry["frames"],
                "first_frame": entry["first_frame"],
                "last_frame": entry["last_frame"],
            }
        )
    observed_tiles.sort(key=lambda item: (item["tile"][1], item["tile"][0]))
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "task": data.get("name"),
        "start_state": data.get("start_state"),
        "tilemap": tilemap,
        "tilemap_hex": f"0x{tilemap:02X}",
        "map": get_map_name(tilemap),
        "frames_on_map": len(rows),
        "observed_player_tiles": observed_tiles,
        "tile_ids": [
            {"tile_id": f"0x{tile_id:02X}", "frames": count}
            for tile_id, count in sorted(tile_ids.items())
        ],
        "a_press_windows": button_windows(rows, "A"),
        "transitions": metadata.get("transitions", []),
    }


def render_text(summary: dict[str, object]) -> str:
    tiles = summary["observed_player_tiles"]
    tile_lines = ", ".join(f"({tile['tile'][0]},{tile['tile'][1]})" for tile in tiles)
    lines = [
        f"Task: {summary.get('task')}",
        f"Start state: {summary.get('start_state')}",
        f"Map: {summary['tilemap_hex']} {summary['map']}",
        f"Frames on map: {summary['frames_on_map']}",
        f"Observed player tiles ({len(tiles)}):",
        tile_lines or "  (none)",
        "",
        "Tile IDs walked:",
    ]
    for item in summary["tile_ids"]:
        lines.append(f"  {item['tile_id']}: {item['frames']} frames")

    lines.append("")
    lines.append("A-press windows:")
    windows = summary["a_press_windows"]
    if windows:
        for window in windows:
            lines.append(
                "  "
                f"f={window['start']}-{window['end']} len={window['length']} "
                f"tile={tuple(window['tile'])} px={tuple(window['pixel'])} "
                f"tile_id={window['tile_id']} last_move={window['last_move']}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def write_text(path: str | None, text: str) -> None:
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract observed walkable tiles from a task trace.")
    parser.add_argument("task", help="Task name or JSON path")
    parser.add_argument("--tilemap", type=parse_tilemap, default=0x17, help="Tilemap to extract, default 0x17")
    parser.add_argument("--out", help="Write JSON summary to this path")
    parser.add_argument("--text-out", help="Write text summary to this path")
    args = parser.parse_args(argv)

    summary = summarize(load_task(args.task), args.tilemap)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_text(args.text_out, render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
