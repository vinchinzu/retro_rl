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
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use RAM-scripted Liu Kang for every fight (no zip required)",
    )
    parser.add_argument(
        "--ladder-model",
        default="",
        help="Zip filename for M1-M7 (e.g. mk1_v3_Match5_ppo_final.zip)",
    )
    args = parser.parse_args()
    if args.scripted and args.ladder_model:
        raise SystemExit("--scripted and --ladder-model are mutually exclusive")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    result = TournamentRunner(
        deterministic=args.deterministic,
        force_scripted=args.scripted,
        ladder_model=args.ladder_model or None,
    ).run(max_frames=args.max_frames)
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
