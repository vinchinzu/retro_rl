"""Run one clean power-on attempt through ALTTP's verified room-0x50 tip.

This is not a Sanctuary full run. It writes evidence for the current clean
continuous prefix only and stops before planned Zelda/Sanctuary work.

    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_to_verified_tip.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from alttp.opening_route.full_tip import run_to_verified_tip
from alttp.paths import RECORDINGS_DIR
from alttp.startup import build_boot_env

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=RECORDINGS_DIR / "verified_tip_run.json",
        help="JSON evidence path",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=RECORDINGS_DIR / "verified_tip_run.png",
        help="final-frame screenshot path",
    )
    args = parser.parse_args(argv)
    _configure_headless()

    env = build_boot_env()
    try:
        result = run_to_verified_tip(env, close=False)
        frame = env.render()  # type: ignore[attr-defined]
        if frame is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(frame)).save(args.screenshot)
            print(f"Wrote {args.screenshot}")
    finally:
        env.close()  # type: ignore[attr-defined]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result.to_report(), indent=2) + "\n")
    print(f"Wrote {args.report}")
    print(
        f"ok={result.ok} phase={result.phase} frames={result.frames} "
        f"tip={result.tip_node} blocker={result.blocker!r}"
    )
    return 0 if result.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
