#!/usr/bin/env python3
"""Train one RAM v3 Liu Kang specialist (used by train_overnight)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from mortal_kombat.train_v3 import train_stage  # noqa: E402
from mortal_kombat.v3_run import V3Run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Stage prefix, e.g. Fight or Goro")
    parser.add_argument("--state", default=None, help="Override state name")
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--load", default=None, help="Optional v3 checkpoint (same obs dim)")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Distinct output prefix for safe candidates (never overwrite the incumbent)",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef-start", type=float, default=None)
    parser.add_argument("--ent-coef-end", type=float, default=None)
    parser.add_argument(
        "--randomize-state",
        action="store_true",
        help="Curriculum: randomize starting health/timer on some episodes",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0,
        help="Wall-clock cutoff; writes *_ppo_{timesteps}_steps.zip and exits 1",
    )
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    state = args.state or f"{args.prefix}_LiuKang"
    run_kw: dict = dict(
        state=state,
        stage=args.prefix,
        steps=args.steps,
        n_envs=args.n_envs,
        load=args.load,
        candidate=args.output_prefix,
        max_seconds=args.max_hours * 3600 if args.max_hours else 0,
        randomize_state=args.randomize_state,
    )
    if args.learning_rate is not None:
        run_kw["learning_rate"] = args.learning_rate
    if args.ent_coef_start is not None:
        run_kw["ent_coef_start"] = args.ent_coef_start
    if args.ent_coef_end is not None:
        run_kw["ent_coef_end"] = args.ent_coef_end
    try:
        run = V3Run(**run_kw)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    result = train_stage(run)
    return 1 if result.wall_stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
