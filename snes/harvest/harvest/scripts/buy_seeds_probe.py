#!/usr/bin/env python3
"""Headless Spring D2 seed-shop probe: nav shop_door, RAM-close the buy.

Does not open a play window. Default pin is ``Y1_Inside_House``.

    HEADLESS=1 uv run python -m harvest.scripts.buy_seeds_probe
    HEADLESS=1 uv run python -m harvest.scripts.buy_seeds_probe \\
      --out recordings/buy_seeds_d2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.buy_seeds import BuySeedsTask
from harvest.tasks.nav import get_pos_from_ram, make_action


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument("--timeout", type=int, default=18_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "buy_seeds_d2_probe.json",
    )
    return p.parse_args()


def main() -> int:
    _configure_headless()
    args = _parse_args()
    env = make_harvest_env(state=args.state)
    try:
        obs, _info = env.reset()
        ram = env.get_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=obs)
        task = BuySeedsTask(timeout=args.timeout)
        task.reset(world)
        start_money = int(read_ram_value(ram, "money") or 0)
        start_stock = int(read_ram_value(ram, "potato_seeds") or 0)
        seen_maps: list[int] = []
        last_status = TaskStatus.RUNNING
        reason = "start"
        for frame in range(args.timeout + 1):
            ram = env.get_ram()
            world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
            tm = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
            if not seen_maps or seen_maps[-1] != tm:
                seen_maps.append(tm)
            result = task.step(world)
            last_status = result.status
            reason = result.reason or reason
            if result.status != TaskStatus.RUNNING:
                break
            action = result.action.action if result.action is not None else make_action()
            obs, _reward, _term, _trunc, _info = env.step(action)
        ram = env.get_ram()
        pos = get_pos_from_ram(ram)
        payload = {
            "status": last_status.value if hasattr(last_status, "value") else str(last_status),
            "reason": reason,
            "frames": frame,
            "seen_tilemaps": [f"0x{tm:02X}" for tm in seen_maps],
            "saw_shop": 0x1C in seen_maps,
            "money": [start_money, int(read_ram_value(ram, "money") or 0)],
            "potato_seeds": [start_stock, int(read_ram_value(ram, "potato_seeds") or 0)],
            "end_tilemap": f"0x{int(ram[ADDR_TILEMAP]):02X}",
            "end_pos": [int(pos.x), int(pos.y)],
            "phase": task.phase_text,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0 if last_status == TaskStatus.SUCCESS else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
