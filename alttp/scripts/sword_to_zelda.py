"""Headless fighter-sword → Zelda rescue progress probe.

Usage:
    # Dev diagnostic from FighterSword state:
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/sword_to_zelda.py

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttp import primitives  # noqa: E402
from alttp.paths import FIGHTER_SWORD_STATE, RECORDINGS_DIR  # noqa: E402
from alttp.startup import build_boot_env  # noqa: E402
from alttp.sword_to_zelda import run_from_sword  # noqa: E402

DEFAULT_REPORT = RECORDINGS_DIR / "sword_to_zelda.json"
DEFAULT_SCREENSHOT = RECORDINGS_DIR / "sword_to_zelda.png"


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
    args = parser.parse_args()
    _configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env: Any = None
    obs: np.ndarray | None = None
    try:
        env = build_boot_env(args.state)
        env.reset()
        primitives.settle_control(env)
        result = run_from_sword(env, source="state_load_dev")
        try:
            rendered = env.render()
            if rendered is not None:
                obs = np.asarray(rendered)
        except Exception:
            obs = None
    finally:
        if env is not None:
            env.close()

    report = result.to_report()
    report["cli"] = {"state": args.state}
    report["classification"] = (
        "clean_natural_chain"
        if result.source == "natural_boot" and result.ok
        else (
            "natural_chain_partial"
            if result.source == "natural_boot"
            else "development_diagnostic"
        )
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")

    if obs is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(obs)).save(args.screenshot)
        print(f"Wrote {args.screenshot}")

    if result.ok:
        return 0
    if result.acceptance.get("fighter_sword_ram") and result.phases:
        # Partial measured progress (south chamber, etc.)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
