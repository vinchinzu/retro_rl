#!/usr/bin/env python3
"""One continuous Liu Kang attempt: power-on, roster swap, credits detect."""

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

from mortal_kombat.tournament import TournamentRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=200_000)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    result = TournamentRunner(deterministic=args.deterministic).run(max_frames=args.max_frames)
    print(
        f"cleared={result.cleared} credits={result.credits} "
        f"furthest={result.furthest} wins={result.wins} losses={result.losses} "
        f"frames={result.frames}"
    )
    if result.swaps:
        print("swaps:")
        for item in result.swaps:
            print(f"  {item}")
    return 0 if result.cleared else 1


if __name__ == "__main__":
    raise SystemExit(main())
