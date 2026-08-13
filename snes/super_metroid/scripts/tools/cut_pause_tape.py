#!/usr/bin/env python3
"""Cut pause-menu freeze (+ trailing idle) from a guided_human take.

Pause freezes SM world/RNG — safe to drop for RTA and open-loop reseam.

```bash
# Dry-run: show spans only
uv run python snes/super_metroid/scripts/tools/cut_pause_tape.py \\
  snes/super_metroid/tasks/full_start_v1.json --dry-run

# Apply: archive pre-cut → rewrite tape → materialize → durable supers pin
uv run python snes/super_metroid/scripts/tools/cut_pause_tape.py \\
  snes/super_metroid/tasks/full_start_v1.json --pin supers

# Or via play wrapper
./snes/super_metroid/play --cut-pause full_start_v1 --pin supers
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.human_tape.cut_pause import (  # noqa: E402
    cut_pause_tape,
    find_phase_runs,
    find_trailing_idle,
    promote_end_durable_pin,
)
from super_metroid.human_tape.hops import load_task_json  # noqa: E402
from super_metroid.human_tape.rta_clock import fmt_time  # noqa: E402
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR  # noqa: E402

TASKS = GAME_DIR / "tasks"
SCRATCH = INTEGRATION_DIR / "scratch"

# Short pin name → durable stem under scratch/tasks
_PIN_STEMS = {
    "supers": "full_start_v1_supers",
    "super": "full_start_v1_supers",
    "super-missile": "full_start_v1_supers",
    "spore-super": "full_start_v1_supers",
    "bomb": "full_start_v1_bomb",
    "bombs": "full_start_v1_bomb",
    "morph": "full_start_v1_morph",
    "hj": "full_start_v1_hj",
    "hijump": "full_start_v1_hj",
    "hi-jump": "full_start_v1_hj",
    "varia": "full_start_v1_varia",
    "main-street": "full_start_v1_main_street",
    "maridia": "full_start_v1_main_street",
    "plasma-beam": "full_start_v1_plasma",
    "plasma": "full_start_v1_plasma",
    "post-plasma": "full_start_v1_plasma",
    "golden-torizo": "full_start_v1_golden_torizo",
    "gt": "full_start_v1_golden_torizo",
    "metal-pirates": "full_start_v1_metal_pirates",
    "metal-pirate": "full_start_v1_metal_pirates",
    "pirates": "full_start_v1_metal_pirates",
    "mp": "full_start_v1_metal_pirates",
    "post-ridley": "full_start_v1_ridley",
    "ridley-tank": "full_start_v1_ridley",
    "post-ridley-tank": "full_start_v1_ridley",
}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "task",
        nargs="?",
        default="full_start_v1",
        help="Task stem under tasks/ or path to .json (default: full_start_v1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List cut spans only; do not rewrite",
    )
    p.add_argument(
        "--no-trailing",
        action="store_true",
        help="Keep trailing stand-still after last input",
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not archive pre-cut take under *_segments/",
    )
    p.add_argument(
        "--no-materialize",
        action="store_true",
        help="Skip hop bodies / run_timing rewrite",
    )
    p.add_argument(
        "--bank",
        action="store_true",
        help="Merge rematerialized hops into skill bank",
    )
    p.add_argument(
        "--pin",
        default=None,
        help=(
            "After cut, copy end.state to durable pin "
            f"(names: {', '.join(sorted(_PIN_STEMS))})"
        ),
    )
    p.add_argument("--min-pause", type=int, default=30)
    p.add_argument("--min-trailing", type=int, default=30)
    args = p.parse_args()

    raw = Path(args.task)
    if raw.is_file():
        task_path = raw
    else:
        stem = raw.stem if raw.suffix == ".json" else str(args.task)
        task_path = TASKS / f"{stem}.json"
    if not task_path.is_file():
        print(f"ERROR: missing {task_path}", file=sys.stderr)
        return 1

    if args.dry_run:
        data = load_task_json(task_path)
        frames = list(data.get("frames") or [])
        trace = list(data.get("trace") or [])
        spans = find_phase_runs(trace, min_frames=args.min_pause)
        if not args.no_trailing:
            t = find_trailing_idle(
                frames, trace, min_frames=args.min_trailing, keep_tail=0
            )
            if t:
                spans.append(t)
        total = sum(s.frames for s in spans)
        print(f"task={task_path}  frames={len(frames)}")
        if not spans:
            print("  (no cuttable pause / trailing idle)")
            return 0
        for s in spans:
            print(
                f"  cut f{s.start}-{s.end}  {s.frames}f ({fmt_time(s.frames)})  "
                f"{s.reason}  room={s.room}"
            )
        print(
            f"  total cut {total}f ({fmt_time(total)}) → "
            f"kept {len(frames) - total}f ({fmt_time(len(frames) - total)})"
        )
        return 0

    report = cut_pause_tape(
        task_path,
        write=True,
        in_place=True,
        archive_first=not args.no_archive,
        cut_pause_phase=True,
        cut_trailing_idle=not args.no_trailing,
        min_pause_frames=args.min_pause,
        min_trailing_idle=args.min_trailing,
        materialize=not args.no_materialize,
        merge_bank=bool(args.bank),
    )
    print(
        f"[CUT] {report.frames_before}f → {report.frames_after}f  "
        f"removed {report.cut_frames}f ({report.cut_time})"
    )
    for s in report.spans:
        print(
            f"  − f{s.start}-{s.end}  {s.frames}f ({fmt_time(s.frames)})  "
            f"{s.reason}  room={s.room}"
        )
    for note in report.notes:
        print(f"  · {note}")

    if args.pin:
        key = str(args.pin).lower().strip()
        stem = _PIN_STEMS.get(key, key if key.startswith("full_start") else f"full_start_v1_{key}")
        written = promote_end_durable_pin(
            task_path,
            stem=stem,
            integration_scratch=SCRATCH,
            tasks_dir=TASKS,
        )
        for w in written:
            print(f"[PIN] durable → {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
