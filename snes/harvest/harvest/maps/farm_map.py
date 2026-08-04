"""
Farm tile map snapshot utility.

Reads the 64x64 tile map from RAM and outputs structured data for
visualization, LLM reasoning, and analytics.

Usage from agent:
    from harvest.maps.farm_map import farm_snapshot, farm_ascii
    snap = farm_snapshot(ram)
    print(farm_ascii(ram))

Usage as CLI:
    uv run python farm_map.py --state Y1_Spring_Day01_06h00m
    uv run python farm_map.py --state Y1_Spring_Day01_06h00m --json
    uv run python farm_map.py --state Y1_Spring_Day01_06h00m --out farm_dump.txt
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from harvest.runtime.retro_setup import make_harvest_env
from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import (
    MAP_WIDTH,
    TILE_SIZE,
    Tool,
    tile_category_counts as catalog_tile_category_counts,
    read_tile_grid as catalog_read_tile_grid,
    tile_glyph,
    tile_histogram as catalog_tile_histogram,
)


# ── core functions ────────────────────────────────────────────────────

def read_tile_grid(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> List[List[int]]:
    """Read tile IDs from RAM as a 2D list [y][x].

    bounds: (x_min, y_min, x_max, y_max) inclusive. None = full 64x64.
    """
    return catalog_read_tile_grid(ram, bounds)


def player_info(ram: np.ndarray) -> Dict:
    """Extract player state from RAM."""
    px = read_ram_value(ram, "player_x", raw=True)
    py = read_ram_value(ram, "player_y", raw=True)
    tool_id = read_ram_value(ram, "tool_selected", raw=True)
    stamina = read_ram_value(ram, "stamina", raw=True)
    tool_name = Tool(tool_id).name if tool_id in Tool._value2member_map_ else f"0x{tool_id:02X}"
    return {
        "tile": (px // TILE_SIZE, py // TILE_SIZE),
        "pixel": (px, py),
        "tool": tool_name,
        "tool_id": tool_id,
        "stamina": stamina,
    }


def tile_histogram(
    grid: List[List[int]],
) -> Dict[str, int]:
    """Count tiles by label. Returns {label: count} sorted descending."""
    return catalog_tile_histogram(grid)


def tile_category_counts(
    grid: List[List[int]],
) -> Dict[str, int]:
    """Count tiles by category (walkable, debris, tillable, etc.)."""
    return catalog_tile_category_counts(grid)


def farm_snapshot(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Dict:
    """Full structured snapshot of the farm state.

    Returns a dict suitable for JSON serialization and LLM consumption.
    """
    grid = read_tile_grid(ram, bounds)
    player = player_info(ram)
    b = bounds or (0, 0, MAP_WIDTH - 1, MAP_WIDTH - 1)
    return {
        "bounds": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]},
        "player": player,
        "histogram": tile_histogram(grid),
        "categories": tile_category_counts(grid),
        "grid": grid,
    }


def farm_ascii(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    show_player: bool = True,
    header: bool = True,
) -> str:
    """Render farm tiles as an ASCII grid.

    Each tile is one character. Player position marked with @.
    Includes axis labels and a legend.
    """
    grid = read_tile_grid(ram, bounds)
    b = bounds or (0, 0, MAP_WIDTH - 1, MAP_WIDTH - 1)
    x0, y0 = b[0], b[1]

    player = player_info(ram) if show_player else None
    ptx = player["tile"][0] if player else -1
    pty = player["tile"][1] if player else -1

    lines = []

    if header:
        lines.append(f"Farm Map ({b[0]},{b[1]})-({b[2]},{b[3]})")
        if player:
            lines.append(
                f"Player @ tile ({ptx},{pty}) px=({player['pixel'][0]},{player['pixel'][1]}) "
                f"tool={player['tool']} stamina={player['stamina']}"
            )
        lines.append("")

    # x-axis labels (tens digit)
    width = b[2] - b[0] + 1
    if width <= 80:
        tens_row = "    "
        ones_row = "    "
        for x in range(x0, b[2] + 1):
            if x % 5 == 0:
                tens_row += str(x // 10) if x >= 10 else " "
                ones_row += str(x % 10)
            else:
                tens_row += " "
                ones_row += " "
        lines.append(tens_row)
        lines.append(ones_row)

    for yi, row in enumerate(grid):
        y = y0 + yi
        label = f"{y:3d} "
        chars = []
        for xi, tid in enumerate(row):
            x = x0 + xi
            if show_player and x == ptx and y == pty:
                chars.append("@")
            else:
                chars.append(tile_glyph(tid))
        lines.append(label + "".join(chars))

    # legend
    lines.append("")
    lines.append("Legend: . empty  - untilled  = tilled  ~ hoed  * watered  C crop  G planted  g mature_grass")
    lines.append("        w weed  o stone  O rock  S stump  R large_rock  | fence")
    lines.append("        P pond/water  # border/structure  B building  @ player")

    return "\n".join(lines)


def farm_json(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    compact: bool = False,
) -> str:
    """Snapshot as JSON string. Omits raw grid if compact=True."""
    snap = farm_snapshot(ram, bounds)
    if compact:
        snap.pop("grid", None)
    return json.dumps(snap, indent=2)


def write_snapshot(
    ram: np.ndarray,
    path: str,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    fmt: str = "ascii",
):
    """Write snapshot to file.

    fmt: "ascii" | "json" | "both"
    """
    parts = []
    if fmt in ("ascii", "both"):
        parts.append(farm_ascii(ram, bounds))
    if fmt in ("json", "both"):
        if parts:
            parts.append("\n--- JSON ---\n")
        parts.append(farm_json(ram, bounds))
    text = "\n".join(parts)
    with open(path, "w") as f:
        f.write(text + "\n")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dump farm tile map")
    parser.add_argument("--state", default="Y1_Spring_Day01_06h00m", help="Save state name")
    parser.add_argument("--bounds", help="x_min,y_min,x_max,y_max (default: full map)")
    parser.add_argument("--farm", action="store_true", help="Use default farm bounds (3,1,62,60)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of ASCII")
    parser.add_argument("--compact", action="store_true", help="Compact JSON (no grid)")
    parser.add_argument("--out", help="Write to file instead of stdout")
    args = parser.parse_args()

    env = make_harvest_env(args.state)
    env.reset()
    ram = env.get_ram()

    bounds = None
    if args.bounds:
        bounds = tuple(int(v) for v in args.bounds.split(","))
    elif args.farm:
        bounds = (3, 1, 62, 60)

    if args.out:
        fmt = "json" if args.json else "ascii"
        write_snapshot(ram, args.out, bounds, fmt)
        print(f"Wrote {fmt} snapshot to {args.out}")
    elif args.json:
        print(farm_json(ram, bounds, compact=args.compact))
    else:
        print(farm_ascii(ram, bounds))
        print()
        snap = farm_snapshot(ram, bounds)
        print("Histogram:", json.dumps(snap["histogram"], indent=2))
        print("Categories:", json.dumps(snap["categories"], indent=2))

    env.close()


if __name__ == "__main__":
    main()
