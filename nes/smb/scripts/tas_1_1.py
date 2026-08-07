"""TAS toolkit CLI for Super Mario Bros. 1-1.

Analyze seeds, auto-discover bottleneck windows, multi-window hill-climb,
hold compression, and idle trims — aiming for the earliest level exit
from ``Level1_1``.

```bash
# Baseline metrics (flag / leave / wall-slams)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.tas_1_1 analyze

# Full polish (pick fastest completing seed, hill-climb, save)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.tas_1_1 optimize --iters 400

# Focus stairs only
uv run python -m smb.scripts.tas_1_1 optimize --window stairs --iters 600

# Verify a seed clears and print timings
uv run python -m smb.scripts.tas_1_1 verify --seed nes/smb/models/smb_1_1_tas_best.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless by default when launched as a module
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.platformer.frame_tools import analyze_seed_static, load_raw_frames
from smb.paths import MODELS_DIR
from smb.policy import DEFAULT_1_1_SEED
from smb.tas.pipeline import (
    ensure_completing_seed,
    optimize_1_1,
    pick_best_seed,
    save_nes9_seed,
)
from smb.tas.trace import trace_seed
from smb.tas.windows import discover_windows, windows_from_labels


def _cmd_analyze(args: argparse.Namespace) -> int:
    seed = Path(args.seed) if args.seed else pick_best_seed()
    frames = load_raw_frames(seed)
    padded, tr = ensure_completing_seed(frames)
    static = analyze_seed_static(padded)
    wins = discover_windows(tr, seed_len=len(padded), max_windows=args.max_windows)

    report = {
        "seed": str(seed),
        "seed_frames": len(frames),
        "padded_frames": len(padded),
        "trace": tr.summary(),
        "static": static,
        "windows": [w.to_dict() for w in wins],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0 if tr.completed else 1


def _cmd_optimize(args: argparse.Namespace) -> int:
    labels = None
    if args.window:
        labels = [w.strip() for w in args.window.split(",") if w.strip()]
    seed = Path(args.seed) if args.seed else None
    out = Path(args.out) if args.out else (MODELS_DIR / "smb_1_1_tas_best.json")
    _best, report = optimize_1_1(
        seed_path=seed,
        out_path=out,
        window_labels=labels,
        iters_per_window=args.iters,
        hold_compress=not args.no_holds,
        trim_leading=not args.no_trim,
        systematic=not args.no_systematic,
        delete_stride=args.delete_stride,
        max_windows=args.max_windows,
        verbose=not args.quiet,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.completed else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    seed = Path(args.seed) if args.seed else (MODELS_DIR / "smb_1_1_tas_best.json")
    if not seed.exists():
        seed = Path(DEFAULT_1_1_SEED)
    frames = load_raw_frames(seed)
    padded, tr = ensure_completing_seed(frames)
    print(
        f"seed={seed}\n"
        f"  frames={len(frames)} padded={len(padded)}\n"
        f"  completed={tr.completed} died={tr.died}\n"
        f"  flag={tr.flag_frame} castle={tr.castle_frame} leave={tr.leave_frame}\n"
        f"  max_x={tr.max_player_x} wall_slams={len(tr.wall_slams)}"
    )
    if args.save_padded and tr.completed:
        out = Path(args.save_padded)
        save_nes9_seed(
            out,
            padded,
            metadata={
                "verified_completed": True,
                "verified_clear_frames": tr.leave_frame,
                "verified_flag_frames": tr.flag_frame,
                "source": str(seed),
                "notes": "padded/verified by tas_1_1 verify",
            },
        )
        print(f"  wrote padded seed → {out}")
    return 0 if tr.completed else 1


def _cmd_windows(args: argparse.Namespace) -> int:
    seed = Path(args.seed) if args.seed else pick_best_seed()
    frames = load_raw_frames(seed)
    padded, tr = ensure_completing_seed(frames)
    if args.labels:
        wins = windows_from_labels(
            args.labels.split(","),
            seed_len=len(padded),
            flag_frame=tr.flag_frame,
        )
    else:
        wins = discover_windows(tr, seed_len=len(padded), max_windows=20)
    for w in wins:
        print(f"{w.label:20s} [{w.start:5d}:{w.end:5d}] ({w.length:4d}f) prio={w.priority:3d}  {w.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Trace seed: flag/leave/stalls/windows")
    pa.add_argument("--seed", type=str, default=None)
    pa.add_argument("--out", type=str, default=None, help="write JSON report")
    pa.add_argument("--max-windows", type=int, default=8)
    pa.set_defaults(func=_cmd_analyze)

    po = sub.add_parser("optimize", help="Multi-window hill-climb + trims")
    po.add_argument("--seed", type=str, default=None, help="default: fastest known seed")
    po.add_argument("--out", type=str, default=None, help="output nes9_rle path")
    po.add_argument(
        "--window",
        type=str,
        default=None,
        help="comma labels or start:end (default: auto-discover)",
    )
    po.add_argument("--iters", type=int, default=300, help="iters per window")
    po.add_argument("--max-windows", type=int, default=5)
    po.add_argument("--no-holds", action="store_true")
    po.add_argument("--no-trim", action="store_true")
    po.add_argument(
        "--no-systematic",
        action="store_true",
        help="skip exhaustive delete/edge sweep",
    )
    po.add_argument(
        "--delete-stride",
        type=int,
        default=2,
        help="frame stride for systematic delete (1=every frame)",
    )
    po.add_argument("--quiet", action="store_true")
    po.set_defaults(func=_cmd_optimize)

    pv = sub.add_parser("verify", help="Confirm seed leaves 1-1")
    pv.add_argument("--seed", type=str, default=None)
    pv.add_argument(
        "--save-padded",
        type=str,
        default=None,
        help="if set, write a completing padded nes9_rle seed",
    )
    pv.set_defaults(func=_cmd_verify)

    pw = sub.add_parser("windows", help="List polish windows for a seed")
    pw.add_argument("--seed", type=str, default=None)
    pw.add_argument("--labels", type=str, default=None)
    pw.set_defaults(func=_cmd_windows)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
