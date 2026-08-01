#!/usr/bin/env python3
"""Validate mountain spa corridor walkability (paths clear of bushes/debris).

Does **not** replay raw input traces. Loads a mountain (or farm) save state,
reads the live 64×64 metatile grid, and checks the outdoor-spa corridor tiles
against ``MOUNTAIN_WALKABLE`` / debris sets.

Facts this locks for seasonal durability:
  - Spa path is on tilemap ``0x10`` (mountain stays 0x10 all seasons).
  - Corridor stand tiles are path IDs (0xA0/0xA8/…), not weeds/stumps/rocks.
  - Debris exists off-path on mountain but must not sit on the spa corridor.

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.mountain_spa_validate
    HEADLESS=1 uv run python -m harvest.scripts.mountain_spa_validate \\
      --state mountain_fish_power_berry_end --out recordings/mountain_spa_validate.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import (
    MAP_HEIGHT,
    MAP_WIDTH,
    MOUNTAIN_DEBRIS_TILES,
    MOUNTAIN_WALKABLE,
    TILE_LABEL,
    WATER_TILES,
    metatile_grid_base,
)
from harvest.maps.map_config import ROUTES
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.hot_spring import (
    MOUNTAIN_TILEMAP,
    SPA_OUTDOOR_STAND_PX,
    SPA_OUTDOOR_STAND_TILE,
    SPA_WATER_TILE,
    SPA_WATER_TILE_ID,
)

DEFAULT_STATE = "hot_spring_bath_end"

# Human bath corridor tiles (tx, ty) — stand positions from hot_spring_bath.
# Used as ground truth for "clear path" validation (not raw frame replay).
SPA_CORRIDOR_TILES: frozenset[tuple[int, int]] = frozenset(
    {
        (4, 22),
        (4, 23),
        (4, 24),
        (4, 25),
        (4, 26),
        (4, 27),
        (5, 22),
        (5, 27),
        (6, 22),
        (6, 27),
        (7, 22),
        (7, 27),
        (8, 22),
        (8, 27),
        (9, 22),
        (9, 27),
        (10, 22),
        (10, 27),
        (11, 22),
        (11, 27),
        (11, 28),
        (11, 29),
        (12, 22),
        (12, 29),
        (13, 22),
        (13, 29),
        (14, 22),
        (14, 29),
        (15, 22),
        (15, 29),
        (16, 22),
        (16, 29),
        (17, 22),
        (17, 29),
        (18, 22),
        (18, 29),
        (19, 22),
        (19, 29),
        (20, 22),
        (20, 29),
        (21, 22),
        (21, 29),
        (22, 22),
        (22, 29),
        (23, 21),
        (23, 22),
        (23, 29),
        (24, 21),
        (24, 29),
        (25, 21),
        (25, 29),
        (26, 21),
        (26, 29),
        (27, 15),
        (27, 16),
        (27, 17),
        (27, 18),
        (27, 19),
        (27, 20),
        (27, 21),
        (27, 29),
        (28, 15),
        (28, 29),
        (29, 15),
        (29, 28),
        (29, 29),
        (30, 15),
        (30, 28),
        (31, 15),
        (31, 28),
        (32, 15),
        (32, 28),
        (33, 15),
        (33, 28),
        (34, 15),
        (34, 28),
        (35, 12),
        (35, 13),
        (35, 14),
        (35, 15),
        (35, 28),
        (36, 12),
        (36, 28),
        (37, 12),
        (37, 26),
        (37, 27),
        (37, 28),
        (38, 12),
        (38, 26),
        (39, 12),
        (39, 26),
        (40, 12),
        (40, 26),
        (41, 26),
        (42, 25),
        (42, 26),
    }
)


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "mountain_spa_validate.json",
    )
    return p.parse_args()


def _tile_at(ram, base: int, tx: int, ty: int) -> int | None:
    if not (0 <= tx < MAP_WIDTH and 0 <= ty < MAP_HEIGHT):
        return None
    idx = base + ty * MAP_WIDTH + tx
    if idx >= len(ram):
        return None
    return int(ram[idx])


def main() -> int:
    _configure_headless()
    args = _parse_args()

    env = make_harvest_env(args.state)
    try:
        env.reset()
        for _ in range(40):
            env.step([0] * 12)
        ram = env.get_ram()
        tilemap = int(read_ram_value(ram, "tilemap"))
        px = int(read_ram_value(ram, "player_x"))
        py = int(read_ram_value(ram, "player_y"))
        base = metatile_grid_base(ram)

        if tilemap != MOUNTAIN_TILEMAP:
            report = {
                "ok": False,
                "reason": f"expected mountain 0x10, got 0x{tilemap:02X}",
                "state": args.state,
                "tilemap": tilemap,
                "pos": [px, py],
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"FAIL not on mountain: {report['reason']}")
            return 1

        hist: Counter[int] = Counter()
        debris_off_path: list[dict] = []
        corridor_rows: list[dict] = []
        blocked: list[dict] = []
        water_ok = False

        for ty in range(MAP_HEIGHT):
            for tx in range(MAP_WIDTH):
                tid = _tile_at(ram, base, tx, ty)
                if tid is None:
                    continue
                hist[tid] += 1
                on_corr = (tx, ty) in SPA_CORRIDOR_TILES
                if tid in MOUNTAIN_DEBRIS_TILES:
                    entry = {
                        "tile": [tx, ty],
                        "id": tid,
                        "hex": f"0x{tid:02X}",
                        "label": TILE_LABEL.get(tid, f"0x{tid:02X}"),
                        "on_spa_corridor": on_corr,
                    }
                    if on_corr:
                        blocked.append(entry)
                    else:
                        debris_off_path.append(entry)

        for tx, ty in sorted(SPA_CORRIDOR_TILES):
            tid = _tile_at(ram, base, tx, ty)
            if tid is None:
                continue
            walk = tid in MOUNTAIN_WALKABLE or tid in WATER_TILES
            row = {
                "tile": [tx, ty],
                "id": tid,
                "hex": f"0x{tid:02X}",
                "label": TILE_LABEL.get(tid, f"0x{tid:02X}"),
                "walkable": walk,
                "debris": tid in MOUNTAIN_DEBRIS_TILES,
            }
            corridor_rows.append(row)
            if not walk and tid != 0xFF:
                blocked.append(row)

        stand_id = _tile_at(ram, base, *SPA_OUTDOOR_STAND_TILE)
        water_id = _tile_at(ram, base, *SPA_WATER_TILE)
        water_ok = water_id == SPA_WATER_TILE_ID

        route_len = len(ROUTES.get("mountain_entry_to_outdoor_spa", []))
        farm_route_len = len(ROUTES.get("farm_to_spa", []))
        return_len = len(ROUTES.get("mountain_to_farm", []))

        corridor_ids = Counter(r["id"] for r in corridor_rows)
        ok = (
            not blocked
            and water_ok
            and stand_id in MOUNTAIN_WALKABLE
            and route_len >= 10
            and farm_route_len >= 10
            and return_len >= 10
        )

        report = {
            "ok": ok,
            "state": args.state,
            "tilemap": tilemap,
            "tilemap_hex": f"0x{tilemap:02X}",
            "pos": [px, py],
            "spa_stand": {
                "tile": list(SPA_OUTDOOR_STAND_TILE),
                "px": list(SPA_OUTDOOR_STAND_PX),
                "id": stand_id,
                "hex": f"0x{stand_id:02X}" if stand_id is not None else None,
                "walkable": stand_id in MOUNTAIN_WALKABLE if stand_id is not None else False,
            },
            "spa_water": {
                "tile": list(SPA_WATER_TILE),
                "id": water_id,
                "expected": SPA_WATER_TILE_ID,
                "ok": water_ok,
            },
            "corridor_tile_count": len(SPA_CORRIDOR_TILES),
            "corridor_id_histogram": {
                f"0x{k:02X}": v for k, v in sorted(corridor_ids.items())
            },
            "blocked_on_corridor": blocked,
            "debris_off_path_count": len(debris_off_path),
            "debris_off_path_sample": debris_off_path[:30],
            "mountain_walkable": sorted(f"0x{t:02X}" for t in MOUNTAIN_WALKABLE),
            "map_top_tiles": [
                {"id": tid, "hex": f"0x{tid:02X}", "count": n}
                for tid, n in hist.most_common(15)
            ],
            "routes": {
                "mountain_entry_to_outdoor_spa": route_len,
                "farm_to_spa": farm_route_len,
                "mountain_to_farm": return_len,
                "fish_spot_to_outdoor_spa": len(
                    ROUTES.get("fish_spot_to_outdoor_spa", [])
                ),
            },
            "notes": [
                "Mountain tilemap 0x10 is season-stable (palette changes only).",
                "Spa corridor must stay free of weeds/stumps/rocks for BFS.",
                "Off-path mountain debris is expected and ignored by spa nav.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"RESULT ok={ok} blocked={len(blocked)} "
            f"debris_off_path={len(debris_off_path)} "
            f"water={'0xF7' if water_ok else water_id} "
            f"report={args.out}"
        )
        if blocked:
            print("BLOCKED:", blocked[:12])
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
