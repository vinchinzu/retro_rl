"""Exercise the measured first-Hyrule-Castle dungeon prefix from room 0x61.

Runs ``room 0x61 → 0x60 → 0x50`` from a development state. The clean
power-on proof belongs to ``run_to_verified_tip.py``; this command remains a
fast diagnostic and does not claim Zelda or Sanctuary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from alttp.opening_route.castle_dungeon import run_from_main_hall
from alttp.paths import RECORDINGS_DIR
from alttp.startup import build_boot_env

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="CastleMain", help="main-hall state")
    parser.add_argument(
        "--report",
        type=Path,
        default=RECORDINGS_DIR / "castle_dungeon_prefix.json",
        help="JSON evidence path",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=RECORDINGS_DIR / "castle_dungeon_prefix.png",
        help="final-frame screenshot path",
    )
    args = parser.parse_args(argv)
    _configure_headless()

    env = build_boot_env(args.state)
    try:
        env.reset()  # type: ignore[attr-defined]
        result = run_from_main_hall(env, source="state_load_dev")
        frame = env.render()  # type: ignore[attr-defined]
        if frame is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(frame)).save(args.screenshot)
            print(f"Wrote {args.screenshot}")
    finally:
        env.close()  # type: ignore[attr-defined]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result.to_report("castle_dungeon_prefix"), indent=2) + "\n"
    )
    print(f"Wrote {args.report}")
    print(
        f"ok={result.ok} phase={result.phase} frames={result.frames} "
        f"blocker={result.blocker!r}"
    )
    return 0 if result.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
