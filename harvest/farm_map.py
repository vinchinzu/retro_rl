"""
Farm tile map snapshot utility.

Reads the 64x64 tile map from RAM and outputs structured data for
visualization, LLM reasoning, and analytics.

Usage from agent:
    from farm_map import farm_snapshot, farm_ascii
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
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from farm_clearer import (
    ADDR_MAP,
    ADDR_X,
    ADDR_Y,
    ADDR_TOOL,
    ADDR_STAMINA,
    ADDR_INPUT_LOCK,
    MAP_WIDTH,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    WALKABLE_TILES,
    LIFTABLE_TILES,
    Tool,
    get_tile_at,
)
from grass_planter import (
    TILLABLE_TILES,
    PLANTABLE_TILES,
    PLANTED_GRASS_TILE,
    ADDR_POWER_BERRIES,
)

# ── tile classification ──────────────────────────────────────────────

TILE_LABEL: Dict[int, str] = {
    # ground
    0x00: "empty",
    0x01: "untilled",
    0x02: "tilled",
    0x03: "weed",
    0x04: "stone",
    0x05: "fence",
    0x06: "rock",
    0x07: "hoed",
    0x08: "watered",
    # rocks/stumps 0x09-0x14
    0x09: "rock2", 0x0A: "rock3", 0x0B: "rock4", 0x0C: "rock5",
    0x0D: "rock6", 0x0E: "rock7", 0x0F: "rock8", 0x10: "rock9",
    0x11: "rockA", 0x12: "rockB", 0x13: "rockC", 0x14: "rockD",
    # planted
    0x70: "grass_planted",
    # mature grass
    0x80: "grass1", 0x81: "grass2", 0x82: "grass3",
    0x83: "grass4", 0x84: "grass5", 0x85: "grass6",
    # paths / borders
    0xA0: "path", 0xA1: "structure", 0xA2: "path2", 0xA3: "path3",
    0xA5: "structure2", 0xA6: "pond", 0xA8: "border",
    # buildings
    0xC1: "bldg", 0xC4: "bldg", 0xC5: "bldg", 0xC6: "bldg",
    0xD0: "bldg", 0xD1: "bldg", 0xD2: "bldg", 0xD3: "bldg",
    0xD4: "bldg", 0xD6: "bldg", 0xD7: "bldg", 0xD8: "bldg",
    0xE0: "bldg", 0xE1: "bldg",
    # water / walls
    0xF0: "water", 0xF1: "water", 0xF2: "water",
    0xF7: "water", 0xF8: "water", 0xF9: "water",
    0xFA: "water", 0xFB: "water", 0xFC: "water", 0xFD: "water",
    0xFF: "wall",
}

# single-char glyph for ASCII grid
TILE_GLYPH: Dict[int, str] = {
    0x00: ".",      # empty
    0x01: "-",      # untilled (tillable)
    0x02: "=",      # tilled (plantable)
    0x03: "w",      # weed
    0x04: "o",      # stone
    0x05: "|",      # fence
    0x06: "O",      # big rock
    0x07: "~",      # freshly hoed (plantable)
    0x08: "*",      # watered soil
    0x70: "G",      # planted grass
    0xA0: " ",      # path
    0xA1: "#",      # structure
    0xA2: " ",      # path
    0xA3: " ",      # path
    0xA5: "#",      # structure
    0xA6: "P",      # pond
    0xA8: "#",      # border
    0xF0: "P",      # water
    0xFF: "#",      # wall
}
# rocks / stumps 0x09-0x14 all get "R"
for _tid in range(0x09, 0x15):
    TILE_GLYPH[_tid] = "R"
# crop growth stages 0x1E-0x6F all get "C"
for _tid in range(0x1E, 0x70):
    TILE_GLYPH[_tid] = "C"
# mature grass 0x80-0x85 all get "g"
for _tid in range(0x80, 0x86):
    TILE_GLYPH[_tid] = "g"
# buildings 0xC0-0xEF
for _tid in (0xC1, 0xC4, 0xC5, 0xC6,
             0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD6, 0xD7, 0xD8,
             0xE0, 0xE1):
    TILE_GLYPH[_tid] = "B"
# water variants
for _tid in (0xF1, 0xF2, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD):
    TILE_GLYPH[_tid] = "P"

TILE_CATEGORY: Dict[int, str] = {}
for _tid in WALKABLE_TILES:
    TILE_CATEGORY[_tid] = "walkable"
for _tid in TILE_TO_DEBRIS:
    TILE_CATEGORY[_tid] = "debris"
for _tid in TILLABLE_TILES:
    TILE_CATEGORY[_tid] = "tillable"
for _tid in PLANTABLE_TILES:
    TILE_CATEGORY[_tid] = "plantable"
TILE_CATEGORY[PLANTED_GRASS_TILE] = "planted"
for _tid in range(0x80, 0x86):
    TILE_CATEGORY[_tid] = "grass"


# ── core functions ────────────────────────────────────────────────────

def read_tile_grid(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> List[List[int]]:
    """Read tile IDs from RAM as a 2D list [y][x].

    bounds: (x_min, y_min, x_max, y_max) inclusive. None = full 64x64.
    """
    if bounds:
        x0, y0, x1, y1 = bounds
    else:
        x0, y0, x1, y1 = 0, 0, MAP_WIDTH - 1, MAP_WIDTH - 1
    grid = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            row.append(get_tile_at(ram, x, y))
        grid.append(row)
    return grid


def player_info(ram: np.ndarray) -> Dict:
    """Extract player state from RAM."""
    px = int(ram[ADDR_X]) | (int(ram[ADDR_X + 1]) << 8)
    py = int(ram[ADDR_Y]) | (int(ram[ADDR_Y + 1]) << 8)
    tool_id = int(ram[ADDR_TOOL])
    stamina = int(ram[ADDR_STAMINA])
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
    counts: Counter = Counter()
    for row in grid:
        for tid in row:
            label = TILE_LABEL.get(tid, f"0x{tid:02X}")
            counts[label] += 1
    return dict(counts.most_common())


def tile_category_counts(
    grid: List[List[int]],
) -> Dict[str, int]:
    """Count tiles by category (walkable, debris, tillable, etc.)."""
    counts: Counter = Counter()
    for row in grid:
        for tid in row:
            cat = TILE_CATEGORY.get(tid, "other")
            counts[cat] += 1
    return dict(counts.most_common())


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
                chars.append(TILE_GLYPH.get(tid, "?"))
        lines.append(label + "".join(chars))

    # legend
    lines.append("")
    lines.append("Legend: . empty  - untilled  = tilled  ~ hoed  * watered  C crop  G planted  g mature_grass")
    lines.append("        w weed  o stone  O rock  R rock/stump  | fence")
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

    import retro

    INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
    env = retro.make("HarvestMoon-Snes", state=args.state, inttype=retro.data.Integrations.CUSTOM_ONLY)
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
