"""Headless courtyard pocket → main castle door (room 0x61).

Usage:
  # From FighterSword (composes stairs exit then pocket→hall):
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pocket_to_main_hall.py

  # Pocket-only (already outdoors after stairs; still boots via FighterSword
  # then runs sword_to_zelda first unless --pocket-only with a pocket state):
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pocket_to_main_hall.py --from-sword
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
for _p in (_REPO_ROOT, globals().get('_SNES_IMPORT_ROOT', _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp import primitives  # noqa: E402
from alttp.opening_route.pocket_to_main_hall import (  # noqa: E402
    run_from_pocket,
    run_from_sword_through_pocket,
)
from alttp.paths import FIGHTER_SWORD_STATE, RECORDINGS_DIR  # noqa: E402
from alttp.startup import build_boot_env  # noqa: E402

DEFAULT_REPORT = RECORDINGS_DIR / "pocket_to_main_hall.json"
DEFAULT_SCREENSHOT = RECORDINGS_DIR / "pocket_to_main_hall.png"


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
        "--from-sword",
        action="store_true",
        default=True,
        help="Compose sword_to_zelda then pocket→hall (default)",
    )
    parser.add_argument(
        "--pocket-only",
        action="store_true",
        help="Assume already at outdoor pocket (skip stairs segment)",
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
        if args.pocket_only:
            # Still need stairs exit if loading FighterSword.
            from alttp.opening_route.sword_to_zelda import run_from_sword

            pre = run_from_sword(env, source="state_load_dev")
            if not pre.ok:
                result = pre
            else:
                result = run_from_pocket(env, source="state_load_dev")
                result = type(result)(
                    ok=result.ok,
                    phase=result.phase,
                    frames=pre.frames + result.frames,
                    snapshot=result.snapshot,
                    phases=list(pre.phases) + list(result.phases),
                    source=result.source,
                    acceptance=result.acceptance,
                    blocker=result.blocker,
                    notes=list(pre.notes) + list(result.notes),
                )
        else:
            result = run_from_sword_through_pocket(env, source="state_load_dev")
        try:
            rendered = env.render()
            if rendered is not None:
                obs = np.asarray(rendered)
        except Exception:
            obs = None
    finally:
        if env is not None:
            env.close()

    report = result.to_report(kind="alttp_pocket_to_main_hall_report")
    report["cli"] = {"state": args.state, "pocketOnly": args.pocket_only}
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
    if result.acceptance.get("open_courtyard") or result.acceptance.get("near_main_door"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
