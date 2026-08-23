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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Stage prefix, e.g. Fight or Goro")
    parser.add_argument("--state", default=None, help="Override state name")
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--load", default=None, help="Optional v3 checkpoint (same obs dim)")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    state = args.state or f"{args.prefix}_LiuKang"
    train_stage(
        state=state,
        stage_prefix=args.prefix,
        steps=args.steps,
        n_envs=args.n_envs,
        load=args.load,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
