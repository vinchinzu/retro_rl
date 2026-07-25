#!/usr/bin/env python3
"""Replay a dialogue segment from a task anchor and compare choice branches."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.capture_task_ram import (
    BUTTON_NAMES,
    DEFAULT_SCAN_RANGES,
    capture_frames,
    load_task,
    make_env,
    parse_address,
    parse_range,
    parse_watch_args,
    parse_watch_field_args,
    parse_watch_section_args,
    snapshot_from_ram,
    write_capture_output,
)


BUTTON_IDS = {name.lower(): idx for idx, name in BUTTON_NAMES.items()}


@dataclass(frozen=True)
class OverrideWindow:
    branch: str
    start_frame: int
    end_frame: int
    buttons: tuple[str, ...]


def parse_override_spec(text: str) -> OverrideWindow:
    try:
        branch_part, buttons_part = text.split("=", 1)
        branch_name, frame_part = branch_part.split("@", 1)
        start_text, end_text = frame_part.split("-", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid override {text!r}; expected BRANCH@START-END=BUTTON[,BUTTON...]"
        ) from exc

    branch = branch_name.strip()
    if not branch:
        raise ValueError(f"Invalid override {text!r}: missing branch name")

    start_frame = int(start_text)
    end_frame = int(end_text)
    if end_frame < start_frame:
        raise ValueError(f"Invalid override {text!r}: end before start")

    buttons_raw = buttons_part.strip()
    if not buttons_raw or buttons_raw.lower() == "none":
        buttons: tuple[str, ...] = ()
    else:
        parsed = []
        for item in buttons_raw.split(","):
            name = item.strip()
            if not name:
                continue
            if name.lower() not in BUTTON_IDS:
                raise ValueError(f"Unknown button {name!r} in override {text!r}")
            parsed.append(name)
        buttons = tuple(parsed)

    return OverrideWindow(
        branch=branch,
        start_frame=start_frame,
        end_frame=end_frame,
        buttons=buttons,
    )


def build_branch_frames(
    all_frames: list[list[int]],
    anchor_frame: int,
    end_frame: int,
    overrides: list[OverrideWindow],
) -> list[list[int]]:
    frames = [list(frame) for frame in all_frames[anchor_frame:end_frame]]
    for override in overrides:
        for task_frame in range(override.start_frame, override.end_frame + 1):
            idx = task_frame - anchor_frame
            if idx < 0 or idx >= len(frames):
                raise ValueError(
                    f"Override {override.branch!r} frame {task_frame} is outside "
                    f"capture range {anchor_frame}:{end_frame}"
                )
            new_frame = [0] * len(frames[idx])
            for button in override.buttons:
                new_frame[BUTTON_IDS[button.lower()]] = 1
            frames[idx] = new_frame
    return frames


def final_value(summary: dict[str, object], label: str) -> int | None:
    for row in summary.get("watch_summary", []):
        if row["label"] == label:
            return row["final"]
    return None


def compare_branch_rams(
    baseline_name: str,
    baseline_ram: np.ndarray,
    branch_name: str,
    branch_ram: np.ndarray,
    scan_ranges: list[tuple[int, int]],
    watches: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scan_addresses = {
        addr
        for start, end in scan_ranges
        for addr in range(start, end + 1)
        if addr < len(baseline_ram) and addr < len(branch_ram)
    }
    scan_addresses.update(watches.keys())
    for addr in sorted(scan_addresses):
        base_value = int(baseline_ram[addr])
        branch_value = int(branch_ram[addr])
        if base_value == branch_value:
            continue
        rows.append(
            {
                "address": f"0x{addr:04X}",
                "label": watches.get(addr),
                "baseline_branch": baseline_name,
                "baseline_value": base_value,
                "branch": branch_name,
                "branch_value": branch_value,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe dialogue branches from a task anchor")
    parser.add_argument("--task", required=True, help="Recorded task name, without .json")
    parser.add_argument(
        "--state",
        default=None,
        help="Starting save state. Defaults to the task's recorded start_state.",
    )
    parser.add_argument(
        "--anchor-frame",
        type=int,
        required=True,
        help="Replay task up to this frame before branching",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Stop replay before this task frame index",
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="NAME",
        help="Branch name. Defaults to a single 'recorded' branch.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="BRANCH@START-END=BUTTON[,BUTTON...]",
        help="Override a frame range for one branch; use '=none' to clear inputs",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to debug_alignment/dialogue_probe/<task>_<anchor>_<end>",
    )
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        metavar="ADDR[=LABEL]",
        help="Add a watched env RAM address",
    )
    parser.add_argument(
        "--watch-field",
        action="append",
        default=[],
        metavar="FIELD_KEY",
        help="Watch a named scalar field from harvest_state.py",
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
        help="Additional scan range for candidate address diffs",
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
        default=64,
        help="How many scan-summary addresses to keep per branch",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    task = load_task(args.task)
    all_frames: list[list[int]] = task["frames"]
    state_name = args.state or task.get("start_state")
    if not state_name:
        parser.error("No state provided and task has no recorded start_state")
    if args.anchor_frame < 0 or args.anchor_frame > len(all_frames):
        parser.error("anchor-frame must be within task bounds")
    if args.end_frame <= args.anchor_frame or args.end_frame > len(all_frames):
        parser.error("end-frame must be within task bounds and after anchor-frame")

    branch_names = args.branch or ["recorded"]
    overrides = [parse_override_spec(item) for item in args.override]
    unknown_branches = sorted({override.branch for override in overrides if override.branch not in branch_names})
    if unknown_branches:
        parser.error(f"Overrides reference unknown branches: {', '.join(unknown_branches)}")

    watches = parse_watch_args(args.watch)
    watches = parse_watch_field_args(args.watch_field, watches=watches)
    watches = parse_watch_section_args(args.watch_section, watches=watches)
    scan_ranges = list(DEFAULT_SCAN_RANGES)
    scan_ranges.extend(parse_range(item) for item in args.scan_range)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT_DIR
        / "debug_alignment"
        / "dialogue_probe"
        / f"{args.task}_{state_name}_{args.anchor_frame}_{args.end_frame}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(state_name)
    try:
        env.reset()
        for frame in all_frames[: args.anchor_frame]:
            env.step(np.array(frame, dtype=np.int32))
        anchor_state = env.em.get_state()
        anchor_ram = np.array(env.get_ram(), dtype=np.uint8)
        anchor_snapshot = snapshot_from_ram(anchor_ram)

        branch_records: list[dict[str, object]] = []
        branch_final_rams: dict[str, np.ndarray] = {}

        for branch_name in branch_names:
            branch_overrides = [override for override in overrides if override.branch == branch_name]
            branch_frames = build_branch_frames(
                all_frames=all_frames,
                anchor_frame=args.anchor_frame,
                end_frame=args.end_frame,
                overrides=branch_overrides,
            )
            env.em.set_state(anchor_state)
            summary, trace_rows = capture_frames(
                env=env,
                frames=branch_frames,
                watches=watches,
                scan_ranges=scan_ranges,
                dialog_addr=parse_address(args.dialog_addr),
                dialog_active_value=args.dialog_active_value,
                top_n=args.top_n,
                frame_offset=args.anchor_frame,
            )
            summary["task_name"] = args.task
            summary["state_name"] = state_name
            summary["branch_name"] = branch_name
            summary["anchor_frame"] = args.anchor_frame
            summary["override_windows"] = [
                {
                    "start": override.start_frame,
                    "end": override.end_frame,
                    "buttons": list(override.buttons),
                }
                for override in branch_overrides
            ]
            branch_out_dir = out_dir / branch_name
            summary_path, trace_path = write_capture_output(
                out_dir=branch_out_dir,
                summary=summary,
                trace_rows=trace_rows,
            )
            final_ram = np.array(env.get_ram(), dtype=np.uint8)
            branch_final_rams[branch_name] = final_ram
            branch_records.append(
                {
                    "branch_name": branch_name,
                    "summary_path": str(summary_path),
                    "trace_path": str(trace_path),
                    "final_watch_values": {
                        row["label"]: row["final"] for row in summary["watch_summary"]
                    },
                    "dialog_windows": summary["dialog_windows"],
                }
            )
    finally:
        env.close()

    baseline_name = branch_names[0]
    baseline_ram = branch_final_rams[baseline_name]
    branch_diffs = []
    for branch_name in branch_names[1:]:
        branch_diffs.extend(
            compare_branch_rams(
                baseline_name=baseline_name,
                baseline_ram=baseline_ram,
                branch_name=branch_name,
                branch_ram=branch_final_rams[branch_name],
                scan_ranges=scan_ranges,
                watches=watches,
            )
        )

    root_summary = {
        "task_name": args.task,
        "state_name": state_name,
        "anchor_frame": args.anchor_frame,
        "end_frame": args.end_frame,
        "anchor_snapshot": anchor_snapshot,
        "branches": branch_records,
        "branch_differences_vs_baseline": branch_diffs,
    }
    root_path = out_dir / "summary.json"
    root_path.write_text(json.dumps(root_summary, indent=2), encoding="utf-8")
    print(f"[DIALOGUE_PROBE] Summary -> {root_path}")
    for branch in branch_records:
        print(
            "[DIALOGUE_PROBE] "
            f"{branch['branch_name']}: "
            f"eve_hearts={branch['final_watch_values'].get('eve_hearts')} "
            f"dialog_windows={branch['dialog_windows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
