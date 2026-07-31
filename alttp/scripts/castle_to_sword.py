"""Headless castle-grounds → secret entrance / uncle / fighter sword.

Usage:
    # Development diagnostic from saved castle-grounds state:
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py

    # Natural chain: title → castle grounds → segment:
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --natural

    # Approach only (skip entry search / uncle):
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --approach-only
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
from alttp.opening_route.castle_to_sword import (  # noqa: E402
    run_from_castle_grounds,
    run_from_state,
    run_natural_chain,
)
from alttp.paths import (  # noqa: E402
    HYRULE_CASTLE_GROUNDS_STATE,
    RECORDINGS_DIR,
)
from alttp.ram import snapshot_to_diag  # noqa: E402
from alttp.route_report import RoutePhaseResult  # noqa: E402
from alttp.startup import build_boot_env  # noqa: E402

DEFAULT_REPORT = RECORDINGS_DIR / "castle_to_sword.json"
DEFAULT_SCREENSHOT = RECORDINGS_DIR / "castle_to_sword.png"


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Compose with boot_past_title_to_castle (clean natural chain)",
    )
    parser.add_argument(
        "--state",
        default=HYRULE_CASTLE_GROUNDS_STATE,
        help="Dev state name when not using --natural "
        f"(default: {HYRULE_CASTLE_GROUNDS_STATE})",
    )
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help="Stop after secret-hole approach (no bush-lift search / uncle)",
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
        if args.natural:
            env = build_boot_env()
            # Manual natural so we can grab a screenshot after.
            from alttp.startup import boot_past_title_to_castle

            boot = boot_past_title_to_castle(env, close=False)
            boot_phase = RoutePhaseResult(
                phase="boot_to_castle",
                ok=bool(boot.snapshot.on_castle_grounds),
                frames=boot.frames,
                snapshot=boot.snapshot,
                detail=(
                    "verified castle-grounds predecessor"
                    if boot.snapshot.on_castle_grounds
                    else "boot_past_title_to_castle missed castle grounds"
                ),
                diag=snapshot_to_diag(boot.snapshot),
            )
            if not boot.snapshot.on_castle_grounds:
                result = run_from_castle_grounds(
                    env, source="natural_boot", try_entry=False, try_uncle=False
                )
                result.ok = False
                result.blocker = "natural boot missed castle grounds"
                result.phase = "boot_to_castle"
                result.frames += boot.frames
                result.phases.insert(0, boot_phase)
            else:
                result = run_from_castle_grounds(
                    env,
                    source="natural_boot",
                    try_entry=not args.approach_only,
                    try_uncle=not args.approach_only,
                )
                result.frames += boot.frames
                result.phases.insert(0, boot_phase)
            try:
                rendered = env.render()
                if rendered is not None:
                    obs = np.asarray(rendered)
            except Exception:
                obs = None
        else:
            env = build_boot_env(args.state)
            env.reset()
            primitives.settle_control(env)
            result = run_from_castle_grounds(
                env,
                source="state_load_dev",
                try_entry=not args.approach_only,
                try_uncle=not args.approach_only,
            )
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
    report["cli"] = {
        "natural": bool(args.natural),
        "state": None if args.natural else args.state,
        "approach_only": bool(args.approach_only),
    }
    # Explicit classification for artifacts.
    if result.source == "natural_boot" and result.ok:
        report["classification"] = "clean_natural_chain"
    elif result.source == "natural_boot":
        report["classification"] = "natural_chain_partial"
    else:
        report["classification"] = "development_diagnostic"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")

    if obs is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(obs)).save(args.screenshot)
        print(f"Wrote {args.screenshot}")

    # Exit 0 only on full sword success; 2 for partial measured progress.
    if result.ok:
        return 0
    if result.acceptance.get("near_secret_hole") or result.acceptance.get(
        "castle_entry"
    ):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
