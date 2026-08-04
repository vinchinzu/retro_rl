"""Hierarchical RLE polish on SMB bottleneck windows.

Examples::

    # 1-1 stairs window on continuous seed (default)
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m smb.scripts.rle_polish --window 1-1-stairs --iters 200

    # 4-2 natural entry
    uv run python -m smb.scripts.rle_polish --window 4-2-entry --mode ga --gens 30

    # List known windows
    uv run python -m smb.scripts.rle_polish --list-windows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import retro_harness.platformer.levels.smb  # noqa: F401 — register LevelConfigs

from retro_harness.platformer.evaluator import Evaluator
from retro_harness.platformer.frame_tools import load_raw_frames
from retro_harness.platformer.level_config import get_level_config
from retro_harness.platformer.rle_optimize import (
    SMB_BOTTLENECK_WINDOWS,
    RleWindow,
    phase_shift_transitions,
    rle_ga_window,
    rle_hillclimb_window,
)
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_CONTINUOUS_SEED,
    compress_nes9_rle,
)


def _window_by_label(label: str) -> RleWindow:
    for w in SMB_BOTTLENECK_WINDOWS:
        if w.label == label:
            return w
    # parse start:end
    if ":" in label:
        a, b = label.split(":", 1)
        return RleWindow(int(a), int(b), label=label)
    known = ", ".join(w.label for w in SMB_BOTTLENECK_WINDOWS)
    raise SystemExit(f"unknown window {label!r}; known: {known} or start:end")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_CONTINUOUS_SEED,
        help="nes9_rle or raw_buttons seed path",
    )
    p.add_argument(
        "--level",
        default="smb_1_1",
        help="LevelConfig id for evaluator (default smb_1_1 for early windows)",
    )
    p.add_argument("--window", default="1-1-stairs", help="label or start:end")
    p.add_argument("--list-windows", action="store_true")
    p.add_argument("--mode", choices=("hill", "ga"), default="hill")
    p.add_argument("--iters", type=int, default=300, help="hillclimb iterations")
    p.add_argument("--gens", type=int, default=40, help="GA generations")
    p.add_argument("--pop", type=int, default=24, help="GA population")
    p.add_argument(
        "--phase-shifts",
        type=str,
        default="",
        help="comma-separated transition frame indices for idle phase polish",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output nes9_rle path (default: models/ under cwd with suffix)",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.list_windows:
        for w in SMB_BOTTLENECK_WINDOWS:
            print(f"{w.label:16s}  frames {w.start:5d}-{w.end:5d}  ({w.length}f)")
        return 0

    if not args.seed.exists():
        print(f"seed not found: {args.seed}", file=sys.stderr)
        return 1

    frames = load_raw_frames(args.seed)
    # nes loader pads to 12; fine for evaluator
    window = _window_by_label(args.window)

    # Pick level config: for 4-2 windows use smb_4_2 if registered
    level_id = args.level
    if window.label.startswith("4-2") and args.level == "smb_1_1":
        level_id = "smb_4_2"
    # Continuous seed 4-2 frames are mid-route inputs; isolated Level4_2
    # eval is only valid for smb_4_2_fast_w8-style natural-entry fragments.
    if window.label.startswith("4-2") and "to_ending" in args.seed.name:
        print(
            "[RLE-POLISH] warning: 4-2 windows on continuous seed need full-route "
            "eval (not isolated smb_4_2). Prefer --seed smb_4_2_fast_w8.json "
            "or a future continuous ending evaluator.",
            file=sys.stderr,
        )

    config = get_level_config(level_id)
    evaluator = Evaluator(config)
    out_dir = Path(config.runs_dir) / "rle_polish" / (window.label or "window")
    out_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    if args.mode == "hill":
        best, result = rle_hillclimb_window(
            frames,
            window,
            evaluator,
            max_iters=args.iters,
            button_mode=True,
            verbose=verbose,
            output_dir=out_dir,
        )
    else:
        best, result = rle_ga_window(
            frames,
            window,
            evaluator,
            population_size=args.pop,
            num_generations=args.gens,
            button_mode=True,
            verbose=verbose,
            output_dir=out_dir,
        )

    if args.phase_shifts.strip():
        transitions = [int(x) for x in args.phase_shifts.split(",") if x.strip()]
        best, result = phase_shift_transitions(
            best,
            transitions,
            evaluator,
            button_mode=True,
            verbose=verbose,
        )

    # Save as nes9_rle (trim to 9 buttons)
    nes9 = [[int(b) for b in f[:9]] for f in best]
    while any(len(f) < 9 for f in nes9):
        for f in nes9:
            if len(f) < 9:
                f.append(0)
    segments = compress_nes9_rle(nes9)
    out_path = args.out or (
        Path(__file__).resolve().parents[1]
        / "models"
        / f"smb_rle_polish_{window.label or 'win'}.json"
    )
    payload = {
        "format": "nes9_rle",
        "segments": segments,
        "num_frames": len(nes9),
        "source": str(args.seed),
        "window": {"start": window.start, "end": window.end, "label": window.label},
        "fitness": result.fitness,
        "completed": result.completed,
        "total_frames": result.total_frames,
        "max_progress": result.max_progress,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"[RLE-POLISH] wrote {out_path} ({len(nes9)} frames)")
        print(
            f"[RLE-POLISH] completed={result.completed} "
            f"frames={result.total_frames} fit={result.fitness:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
