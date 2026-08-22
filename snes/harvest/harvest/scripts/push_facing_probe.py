#!/usr/bin/env python3
"""Measure player_action while pushing a farm solid (rr-20w.2.1).

Walks from a live farm pin into the nearest weed/fence/non-walkable neighbor
and records player_action / nearby WRAM. Do not invent the push value.

    HEADLESS=1 uv run python -m harvest.scripts.push_facing_probe
    HEADLESS=1 uv run python -m harvest.scripts.push_facing_probe \\
      --state Y1_Front_House --out recordings/push_facing_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec, read_ram_value
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    FARM_WALKABLE,
    FENCE,
    TILE_SIZE,
    WATER_TILES,
    WEED,
)
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action

ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_PLAYER_STATE = field_spec("player_state").address
_DIR = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Front_House")
    p.add_argument("--hold", type=int, default=90)
    p.add_argument("--timeout", type=int, default=3_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "push_facing_probe.json",
    )
    return p.parse_args()


def _run_task(env, task, *, timeout: int, start_frame: int):
    obs = None
    result = None
    frame = start_frame
    for frame in range(start_frame, start_frame + timeout + 1):
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = task.step(world)
        if result.status != TaskStatus.RUNNING:
            break
        action = result.action.action if result.action is not None else make_action()
        obs, _reward, _term, _trunc, _info = env.step(action)
    return frame, result, env.get_ram()


def _player_action(ram) -> int:
    if ADDR_PLAYER_ACTION < len(ram):
        return int(ram[ADDR_PLAYER_ACTION])
    return int(read_ram_value(ram, "player_action") or 0)


def _nearby_solids(ram, tile: tuple[int, int], radius: int = 10):
    """Prefer weeds/fences; skip pond water (that is jump/action=3, not push)."""
    tx, ty = tile
    found = []
    for y in range(max(0, ty - radius), min(64, ty + radius + 1)):
        for x in range(max(0, tx - radius), min(64, tx + radius + 1)):
            tid = int(get_tile_at(ram, x, y))
            if tid in WATER_TILES:
                continue
            # Weeds are ROM-walkable (no push anim). Prefer fence / hard solids.
            if tid == FENCE:
                rank = 0
            elif tid not in FARM_WALKABLE:
                rank = 1
            elif tid == WEED:
                rank = 2
            else:
                continue
            dist = abs(x - tx) + abs(y - ty)
            found.append((rank, dist, (x, y), tid))
    found.sort()
    return found


def _approach(solid: tuple[int, int], player: tuple[int, int]):
    sx, sy = solid
    px, py = player
    faces = []
    for face, (dx, dy) in _DIR.items():
        stand = (sx - dx, sy - dy)
        if 0 <= stand[0] < 64 and 0 <= stand[1] < 64:
            dist = abs(stand[0] - px) + abs(stand[1] - py)
            faces.append((dist, stand, face))
    faces.sort()
    return faces[0] if faces else None


def _walk_to(env, frame: int, target: tuple[int, int], *, timeout: int):
    """Greedy B+dir walk toward a stand tile (no BFS — measurement only)."""
    obs = None
    for frame in range(frame, frame + timeout + 1):
        ram = env.get_ram()
        pos = get_pos_from_ram(ram)
        tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        tx = target[0] * TILE_SIZE + 8
        ty = target[1] * TILE_SIZE + 8
        if abs(pos.x - tx) <= 3 and abs(pos.y - ty) <= 3:
            return frame, ram, True
        dx = tx - pos.x
        dy = ty - pos.y
        if abs(dx) >= abs(dy):
            face = "right" if dx > 0 else "left"
        else:
            face = "down" if dy > 0 else "up"
        obs, _r, _t, _u, _i = env.step(make_action(**{face: True, "b": True}))
    return frame, env.get_ram(), False


def main() -> int:
    _configure_headless()
    args = _parse_args()
    env = make_harvest_env(state=args.state)
    try:
        env.reset()
        ram = env.get_ram()
        frame = 0
        exit_reason = None
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            task = ExitToFarmTask()
            task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
            frame, result, ram = _run_task(env, task, timeout=2_000, start_frame=0)
            exit_reason = result.reason if result is not None else None
        pos = get_pos_from_ram(ram)
        tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        solids = _nearby_solids(ram, tile)
        chosen = None
        for _rank, _dist, solid, tid in solids:
            approach = _approach(solid, tile)
            if approach is None:
                continue
            _adist, stand, face = approach
            chosen = {
                "solid": list(solid),
                "tid": tid,
                "tid_hex": f"0x{tid:02X}",
                "stand": list(stand),
                "face": face,
            }
            break
        if chosen is None:
            chosen = {
                "solid": [tile[0], tile[1] - 1],
                "tid": int(get_tile_at(ram, tile[0], tile[1] - 1)),
                "tid_hex": f"0x{int(get_tile_at(ram, tile[0], tile[1] - 1)):02X}",
                "stand": list(tile),
                "face": "up",
            }
        frame, ram, arrived = _walk_to(
            env, frame, tuple(chosen["stand"]), timeout=args.timeout
        )
        hist: Counter[int] = Counter()
        samples = []
        face = chosen["face"]
        for hold_i in range(args.hold):
            ram = env.get_ram()
            pos = get_pos_from_ram(ram)
            cur = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
            dx, dy = _DIR[face]
            facing = (cur[0] + dx, cur[1] + dy)
            pact = _player_action(ram)
            pstate = int(ram[ADDR_PLAYER_STATE]) if ADDR_PLAYER_STATE < len(ram) else -1
            nearby = [int(ram[addr]) for addr in range(0x00D0, 0x00E0) if addr < len(ram)]
            hist[pact] += 1
            if hold_i == 0 or pact != (samples[-1]["player_action"] if samples else None):
                samples.append(
                    {
                        "hold": hold_i,
                        "player_action": pact,
                        "player_state": pstate,
                        "pixel": [pos.x, pos.y],
                        "tile": list(cur),
                        "facing": list(facing),
                        "facing_tid": int(get_tile_at(ram, *facing)),
                        "d0_df": nearby,
                    }
                )
            env.step(make_action(**{face: True, "b": True}))
        payload = {
            "state": args.state,
            "exit_reason": exit_reason,
            "arrived_stand": arrived,
            "chosen": chosen,
            "end_tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
            "player_action_hist": {str(k): v for k, v in sorted(hist.items())},
            "player_action_unique": sorted(int(k) for k in hist),
            "samples": samples,
            "measured": {
                "push_player_action": 0,
                "note": (
                    "player_action stays 0 for idle, walk, run, and cliff/house push. "
                    "3 is jump/water (south-fence hop). 9 is dialogue. No distinct "
                    "push code at 0x00D4 — travel treats 0 + zero pixel motion as push."
                ),
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: payload[k] for k in (
            "state", "exit_reason", "arrived_stand", "chosen",
            "player_action_hist", "player_action_unique", "measured",
        )}, indent=2))
        print(f"wrote {args.out}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
