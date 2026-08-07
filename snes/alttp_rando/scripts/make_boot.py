#!/usr/bin/env python3
"""Headless power-on → FirstPlay.state for ALTTP Rando (JP 1.0).

Usage::

    SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot
    SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SNES = _REPO / "snes"
for _p in (_REPO, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild FirstPlay.state even if it already exists",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=12_000,
        help="Abort boot after this many frames",
    )
    parser.add_argument(
        "--mash-only",
        action="store_true",
        help="Skip alttp.startup path; mash START/A only",
    )
    args = parser.parse_args(argv)
    _configure_headless()

    from alttp_rando.boot import (
        boot_to_controllable,
        create_first_play_state,
        make_boot_env,
    )
    from alttp_rando.paths import FIRST_PLAY_STATE, INTEGRATION_DIR, RECORDINGS_DIR
    from alttp_rando.scripts.setup_rom import main as setup_main

    # Always ensure JP ROM is wired.
    rc = setup_main()
    if rc != 0:
        return rc

    state_path = INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state"
    if state_path.is_file() and not args.force:
        print(f"Already exists: {state_path}")
        print("Pass --force to rebuild.")
        return 0

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mash_only:
        env = make_boot_env()
        try:
            env.reset()
            result = boot_to_controllable(
                env,
                max_frames=args.max_frames,
                prefer_alttp_startup=False,
                close=False,
            )
            if result.ok:
                from retro_harness.env import write_state_bytes
                import numpy as np
                from PIL import Image

                write_state_bytes(state_path, env.em.get_state())
                obs = env.render()
                png = RECORDINGS_DIR / f"{FIRST_PLAY_STATE}.png"
                if obs is not None:
                    Image.fromarray(np.asarray(obs)).save(png)
                    Image.fromarray(np.asarray(obs)).save(
                        INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.png"
                    )
                result = type(result)(
                    ok=True,
                    frames=result.frames,
                    snapshot=result.snapshot,
                    method=result.method,
                    detail=result.detail,
                    state_path=str(state_path),
                    png_path=str(png),
                )
        finally:
            env.close()
    else:
        result = create_first_play_state(max_frames=args.max_frames)

    report = result.to_dict()
    report_path = RECORDINGS_DIR / "make_boot.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    if result.state_path:
        print(f"State: {result.state_path}")
    if result.png_path:
        print(f"PNG: {result.png_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
