#!/usr/bin/env python3
"""Capture RAM changes for the trimmed Day 9 harvest-mode route.

This replays the farm-only harvest slice, writes the usual RAM trace output,
and adds derived shipping metrics for planning/budget work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.capture_task_ram import capture_task, parse_watch_args, parse_watch_field_args


DEFAULT_TASK = "harvest"
DEFAULT_STATE = "Y1_Day09_Harvest_Mode_Start"
DEFAULT_START_FRAME = 279
DEFAULT_END_FRAME = 5721
DEFAULT_OUT_DIR = ROOT_DIR / "debug_alignment" / "ram_capture" / "harvest_mode_day9"
DEFAULT_FIELDS = [
    "money",
    "shipping_money",
    "day",
    "hour",
    "minute",
    "potato_seeds",
    "turnip_seeds",
    "corn_seeds",
    "tomato_seeds",
]
DEFAULT_RAW_WATCHES = [
    "0x0921=tool_selected",
    "0x0923=tool_backpack",
]
DEFAULT_SCAN_RANGES = [
    (0x0900, 0x099F),
    (0x15F00, 0x15F1F),
]


def derive_harvest_metrics(summary: dict[str, object]) -> dict[str, object]:
    row = next(
        (entry for entry in summary.get("watch_summary", []) if entry.get("label") == "shipping_money"),
        None,
    )
    if row is None:
        return {}

    base = row.get("base")
    final = row.get("final")
    unique_values = [value for value in row.get("unique_values", []) if isinstance(value, int)]
    unique_values.sort()
    increments = sorted({b - a for a, b in zip(unique_values, unique_values[1:]) if b > a})

    delta = None
    if isinstance(base, int) and isinstance(final, int):
        delta = final - base

    unit = increments[0] if increments else None
    estimated_deposit_count = None
    if isinstance(delta, int) and isinstance(unit, int) and unit > 0 and delta >= 0 and delta % unit == 0:
        estimated_deposit_count = delta // unit

    return {
        "shipping_money_start": base,
        "shipping_money_end": final,
        "shipping_money_delta": delta,
        "shipping_money_step_candidates": increments,
        "estimated_shipping_unit": unit,
        "estimated_deposit_count": estimated_deposit_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture RAM for the trimmed harvest-mode route")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--start-frame", type=int, default=DEFAULT_START_FRAME)
    parser.add_argument("--end-frame", type=int, default=DEFAULT_END_FRAME)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    watches = parse_watch_args(DEFAULT_RAW_WATCHES)
    watches = parse_watch_field_args(DEFAULT_FIELDS, watches=watches)

    summary_path, trace_path = capture_task(
        task_name=args.task,
        state_name=args.state,
        out_dir=Path(args.out_dir),
        watches=watches,
        scan_ranges=list(DEFAULT_SCAN_RANGES),
        dialog_addr=0x019A,
        dialog_active_value=0,
        top_n=40,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = derive_harvest_metrics(summary)
    summary["harvest_metrics"] = metrics
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[HARVEST_RAM] Summary -> {summary_path}")
    print(f"[HARVEST_RAM] Trace   -> {trace_path}")
    if metrics:
        print(f"[HARVEST_RAM] Metrics -> {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
