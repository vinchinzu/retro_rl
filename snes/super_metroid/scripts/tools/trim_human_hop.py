#!/usr/bin/env python3
"""Offline idle + retry trim for one guided_human room hop.

```bash
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \\
  snes/super_metroid/tasks/g4_tourian_human_mb.json \\
  --hop 1 --mode traversal \\
  -o snes/super_metroid/tasks/g4_tourian_human_mb_seeds/escape1_trim.json

# List hops
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \\
  snes/super_metroid/tasks/g4_tourian_human_mb.json --list
```

No emulator. Uses task ``frames`` + ``trace`` only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.human_tape import load_room_hops, load_room_names  # noqa: E402
from super_metroid.human_tape.trim import (  # noqa: E402
    export_trimmed_seed,
    infer_mode,
    trim_task_hop,
)


def _print_hops(hops: list[dict]) -> None:
    print(f"room hops: {len(hops)}")
    for h in hops:
        mode = infer_mode(int(h.get("room_id") or 0))
        print(
            f"  [{h['index']:02d}] idx {h['start_index']:5d}-{h['end_index']:5d} "
            f"({h['dwell']:4d}f) {h['room']} {h.get('name', '?')}  "
            f"default_mode={mode}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", type=Path, help="Path to guided_human task JSON")
    parser.add_argument(
        "--hop",
        type=int,
        default=None,
        help="Room hop index (from extract / --list)",
    )
    parser.add_argument(
        "--mode",
        choices=("traversal", "combat", "safe", "auto"),
        default="safe",
        help=(
            "Trim mode: safe=leading+trailing only (open-loop default); "
            "traversal=also mid-idle/retry heuristics; combat=bosses; "
            "auto=combat rooms → combat else traversal"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write trimmed seed JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List room hops and exit",
    )
    parser.add_argument("--drop-px", type=float, default=48.0)
    parser.add_argument("--min-loop-frames", type=int, default=45)
    parser.add_argument("--min-idle", type=int, default=40)
    parser.add_argument("--pad-after", type=int, default=8)
    parser.add_argument(
        "--keep-leading-idle",
        type=int,
        default=0,
        help="Keep this many leading idle frames (0–2 typical)",
    )
    args = parser.parse_args()

    task_path = args.task
    if not task_path.is_file():
        print(f"ERROR: missing task {task_path}", file=sys.stderr)
        return 1

    data = json.loads(task_path.read_text(encoding="utf-8"))
    names = load_room_names()
    hops = load_room_hops(task_data=data, room_names=names, settle=True)

    if args.list or args.hop is None:
        _print_hops(hops)
        if args.list or args.hop is None:
            if args.hop is None and not args.list:
                print("ERROR: pass --hop N (or --list)", file=sys.stderr)
                return 1
            if args.list:
                return 0

    assert args.hop is not None
    mode = None if args.mode == "auto" else args.mode
    try:
        trimmed, report, hop = trim_task_hop(
            data,
            args.hop,
            hops=hops,
            mode=mode,
            drop_px=args.drop_px,
            min_loop_frames=args.min_loop_frames,
            min_idle=args.min_idle,
            pad_after=args.pad_after,
            keep_leading_idle=args.keep_leading_idle,
        )
    except IndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"hop [{hop.get('index')}] {hop.get('room')} {hop.get('name', '?')}  "
        f"mode={report.mode}"
    )
    print(f"  frames: {report.frames_before} → {report.frames_after}")
    print(
        f"  cuts: leading_idle={report.leading_idle_cut} "
        f"trailing={report.trailing_cut} "
        f"mid_idle={report.mid_idle_cut} "
        f"retry_loops={report.retry_loops_cut} "
        f"retry_frames={report.retry_frames_cut}"
    )
    print(f"  kept_ranges: {report.kept_ranges}")
    for note in report.notes:
        print(f"  note: {note}")

    out = args.out
    if out is None:
        seed_dir = task_path.with_name(task_path.stem + "_seeds")
        room = hop.get("room") or f"hop{args.hop}"
        out = seed_dir / f"hop{args.hop:02d}_{room}_trim.json"

    meta = {
        "source_task": str(task_path),
        "source_name": data.get("name") or task_path.stem,
        "hop_index": int(hop.get("index", args.hop)),
        "hop": {
            "room": hop.get("room"),
            "room_id": hop.get("room_id"),
            "name": hop.get("name"),
            "start_index": hop.get("start_index"),
            "end_index": hop.get("end_index"),
            "xy": hop.get("xy"),
            "end_xy": hop.get("end_xy"),
            "dwell": hop.get("dwell"),
        },
        "trim": report.to_dict(),
    }
    export_trimmed_seed(out, trimmed, meta)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
