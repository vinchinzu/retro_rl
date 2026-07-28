#!/usr/bin/env python3
"""Probe Harvest Moon morning readiness from a pinned save state.

Harvest Moon's title/new-game intro is long; the maturity M1/M2 contract here
is a **named morning state** that is RAM-verified controllable (not power-on
yet). Use this probe to confirm a fixture is a valid day-plan start point.

Examples:

    uv run python -m harvest.scripts.boot_probe
    uv run python -m harvest.scripts.boot_probe --state Y1_After_Sleep
    uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House --steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

from harvest.paths import GAME, GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.runtime.retro_setup import register_harvest_integration


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _save_rgb_png(obs, path: Path) -> Path | None:
    """Best-effort screenshot; skip if Pillow is not installed."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(obs)).save(path)
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state",
        default="Y1_Inside_House",
        help="Save state name under custom_integrations/HarvestMoon-Snes/",
    )
    p.add_argument("--steps", type=int, default=20, help="Idle frames after load")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "boot_probe.json",
        help="JSON report path",
    )
    p.add_argument(
        "--screenshot",
        type=Path,
        default=PROJECT_DIR / "recordings" / "boot_probe.png",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_headless()

    import retro

    register_harvest_integration(retro)
    state_path = GAME_DIR / f"{args.state}.state"
    if not state_path.is_file():
        print(f"STATE missing: {state_path}")
        return 2

    env = retro.make(
        game=GAME,
        state=args.state,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="rgb_array",
    )
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        idle = env.action_space.sample() * 0
        for _ in range(max(0, args.steps)):
            step = env.step(idle)
            obs = step[0]

        ram = env.get_ram()
        scene = classify_scene_from_ram(ram)
        fields = {
            key: int(read_ram_value(ram, key))
            for key in (
                "year",
                "season",
                "day",
                "weekday",
                "hour",
                "minute",
                "tilemap",
                "money",
                "stamina",
                "input_lock",
            )
        }
        hour = fields["hour"]
        ready = morning_scene_ready(scene, hour) or (
            scene.is_normal_map and fields["input_lock"] == 1
        )
        png = _save_rgb_png(obs, args.screenshot)
        report = {
            "state": args.state,
            "ready": bool(ready),
            "scene": scene.to_dict(),
            "fields": fields,
            "screenshot": str(png) if png else None,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"BOOT state={args.state} ready={ready} "
            f"day={fields['day']} hour={fields['hour']:02d}:{fields['minute']:02d} "
            f"tilemap=0x{fields['tilemap']:02X} scene={scene.summary()} "
            f"report={args.out}"
        )
        return 0 if ready else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
