"""Headless fighter-sword → secret-entrance clear (stairs outdoor exit).

Usage:
    # Dev diagnostic from FighterSword state:
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/secret_entrance_clear.py

    # Compose after castle_to_sword natural success is not automatic here;
    # pass --state FighterSword (default) until the segment is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, globals().get("_SNES_IMPORT_ROOT", _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp import primitives  # noqa: E402
from alttp.opening_route.secret_entrance_clear import run_from_sword  # noqa: E402
from alttp.paths import FIGHTER_SWORD_STATE, RECORDINGS_DIR  # noqa: E402
from alttp.startup import build_boot_env  # noqa: E402

DEFAULT_REPORT = RECORDINGS_DIR / "secret_entrance_clear.json"
DEFAULT_SCREENSHOT = RECORDINGS_DIR / "secret_entrance_clear.png"


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        default=FIGHTER_SWORD_STATE,
        help=f"Dev state name (default: {FIGHTER_SWORD_STATE})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=DEFAULT_SCREENSHOT,
        help=f"Final-frame PNG (default: {DEFAULT_SCREENSHOT})",
    )
    parser.add_argument(
        "--no-south",
        action="store_true",
        help="Skip south-chamber approach (exit only)",
    )
    parser.add_argument(
        "--no-exit",
        action="store_true",
        help="Skip stairs exit (south only)",
    )
    args = parser.parse_args()
    _configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env: Any = None
    obs: np.ndarray | None = None
    try:
        env = build_boot_env(args.state)
        env.reset()
        primitives.settle_control(env)
        result = run_from_sword(
            env,
            source="state_load_dev",
            try_south=not args.no_south,
            try_exit=not args.no_exit,
        )
        report = result.to_report()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        try:
            frame = env.render()
            if frame is not None:
                obs = np.asarray(frame)
        except Exception:
            obs = None
        if obs is not None:
            Image.fromarray(obs).save(args.screenshot)
        return 0 if result.ok else 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
