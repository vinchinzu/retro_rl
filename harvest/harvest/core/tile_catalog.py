#!/usr/bin/env python3
"""Shared map-tile metadata for Harvest Moon SNES.

This module is intentionally pure data plus small RAM readers.  Navigation,
farm scanning, world export, and editor views should use these definitions
instead of carrying local copies of tile IDs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, WRAM_SNAPSHOT_SIZE, field_spec


class DebrisType(IntEnum):
    WEED = 1
    STONE = 2
    ROCK = 3
    STUMP = 4
    FENCE = 5


class Tool(IntEnum):
    NONE = 0x00
    SICKLE = 0x01
    HOE = 0x02
    HAMMER = 0x03
    AXE = 0x04
    MILKER = 0x0E
    BRUSH = 0x0F
    WATERING_CAN = 0x10


ADDR_X = field_spec("player_x").address
ADDR_Y = field_spec("player_y").address
ADDR_TOOL = field_spec("tool_selected").address
ADDR_STAMINA = field_spec("stamina").address
ADDR_TILEMAP = field_spec("tilemap").address
ADDR_INPUT_LOCK = field_spec("input_lock").address
ADDR_MAP = 0x09B6

TILE_SIZE = 16
MAP_WIDTH = 64
MAP_HEIGHT = 64
MAP_TILE_COUNT = MAP_WIDTH * MAP_HEIGHT


# ── Crop / soil tile IDs ──────────────────────────────────────────────

UNTILLED = 0x01
DRIED_TILLED = 0x02
WEED = 0x03
STONE = 0x04
FENCE = 0x05
ROCK = 0x06
FRESH_TILLED = 0x07
WATERED_TILLED = 0x08
PLANTED_GRASS_TILE = 0x70

TILLABLE_TILES = frozenset({UNTILLED, DRIED_TILLED})
PLANTABLE_TILES = frozenset({FRESH_TILLED})
CROP_TILE_RANGE = range(0x1E, 0x70)
DRY_CROP_TILES = frozenset(tile_id for tile_id in CROP_TILE_RANGE if tile_id % 2 == 0)
WET_CROP_TILES = frozenset(CROP_TILE_RANGE) - DRY_CROP_TILES
MATURE_CROP_TILES = frozenset({
    0x38, 0x39,  # tomato
    0x52, 0x53,  # corn
    0x60, 0x61,  # potato
    0x6E, 0x6F,  # turnip
})
PLOT_TILES = frozenset({FRESH_TILLED, WATERED_TILLED}) | frozenset(CROP_TILE_RANGE)
UNRIPE_DRY_CROP_TILES = frozenset(tile_id for tile_id in DRY_CROP_TILES if tile_id not in MATURE_CROP_TILES)
WATERABLE_TILES = UNRIPE_DRY_CROP_TILES
MATURE_GRASS_TILES = frozenset(range(0x80, 0x86))
STALE_TILE_IDS = frozenset({0x72, 0x75, 0x76, 0xFF})

# Pond/water tiles.  0xA6 is a pond border/decor tile that is useful for
# detection; REFILL_WATER_TILES are the actual water IDs for can refills.
WATER_TILES = frozenset({
    0xA6,
    0xF0, 0xF1, 0xF2,
    0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
})
REFILL_WATER_TILES = WATER_TILES - frozenset({0xA6})
POND_CHARACTERISTIC_TILES = frozenset({0xA6, 0xF0})


# ── Debris and tools ──────────────────────────────────────────────────

# Farm 2x2 debris families (row-major TL, TR, BL, BR). Verified from day1 /
# Y1_After_First_Rock_Smash: hammer clears 0x0D-0x10 quads; 0x09-0x0C remain
# for the axe.
STUMP_TL = 0x09
STUMP_TILES = frozenset({0x09, 0x0A, 0x0B, 0x0C})
LARGE_ROCK_TL = 0x0D
LARGE_ROCK_TILES = frozenset({0x0D, 0x0E, 0x0F, 0x10})
# Mid-hit visual variants observed on large rocks while swinging.
LARGE_ROCK_DAMAGE_TILES = frozenset({0x11, 0x12, 0x13, 0x14})

TILE_TO_DEBRIS = {
    WEED: DebrisType.WEED,
    STONE: DebrisType.STONE,
    FENCE: DebrisType.FENCE,
    ROCK: DebrisType.ROCK,  # single-tile small boulder (rare on day1)
    **{tid: DebrisType.STUMP for tid in STUMP_TILES},
    **{tid: DebrisType.ROCK for tid in LARGE_ROCK_TILES},
    **{tid: DebrisType.ROCK for tid in LARGE_ROCK_DAMAGE_TILES},
}

DEBRIS_TOOL = {
    DebrisType.WEED: Tool.SICKLE,
    DebrisType.STONE: Tool.HAMMER,
    DebrisType.ROCK: Tool.HAMMER,
    DebrisType.STUMP: Tool.AXE,
    DebrisType.FENCE: None,
}

LIFTABLE_TILES = frozenset({WEED, STONE, FENCE})
CLEARABLE_DEBRIS_TYPES = frozenset(
    {
        DebrisType.WEED,
        DebrisType.STONE,
        DebrisType.ROCK,
        DebrisType.STUMP,
    }
)


def debris_footprint(
    tile: Tuple[int, int],
    tile_id: int,
) -> Tuple[Tuple[int, int], ...]:
    """Tiles occupied by a debris object (2x2 when ``tile`` is a known TL)."""
    tx, ty = tile
    if tile_id in (STUMP_TL, LARGE_ROCK_TL) or tile_id in LARGE_ROCK_DAMAGE_TILES:
        return ((tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1))
    return ((tx, ty),)


def is_multitile_debris_anchor(tile_id: int) -> bool:
    """True when this tile ID is the top-left of a 2x2 stump/rock."""
    return tile_id in (STUMP_TL, LARGE_ROCK_TL)


# ── Walkability by map ────────────────────────────────────────────────

FARM_WALKABLE = frozenset({
    0x00, UNTILLED, DRIED_TILLED, WEED, FRESH_TILLED, WATERED_TILLED,
    PLANTED_GRASS_TILE,
    0x79,  # Soil/ground variant after viewport scrolls over farm plots.
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85,
    0xA0, 0xA1, 0xA2, 0xA3, 0xA8,
    0xC5,
})

PATH_WALKABLE = frozenset({
    0xA0, 0xA1, 0xA2, 0xC0,
})

TOWN_WALKABLE = frozenset({
    0xA0, 0xA1, 0xA2, 0xA4, 0xC0, 0xC3, 0xD6,
})

SHOP_WALKABLE = frozenset({
    0xA0, 0xA1, 0xC3, 0xD4, 0xD6,
})

COOP_WALKABLE = frozenset({
    0xA0, 0xA1, 0xA4, 0xC5,
})

CHURCH_WALKABLE = frozenset({
    # Provisional from sunday_go_to_church live replay.  These include aisle,
    # doorway, and pew-adjacent tiles the player actually stood on.
    0xA0, 0xA1, 0xC2, 0xD5, 0xD6, 0xDA, 0xDB, 0xDC,
})

# Seeded from multi-recording stand tiles (spa bath, sunday mountain, chop wood,
# fish/berry). Primary path: 0xA0 / 0xA8. Entry uses 0xC6; 0xA7 is a common
# path-edge stand tile on the south corridor. Spa bath corridor only needs
# 0xA0/0xA3/0xA8 (ROM-validated clear of debris).
# Do NOT add 0xFF (viewport unload garbage) or water (0xF7 etc).
# Mountain keeps tilemap 0x10 all seasons (palette/season overlay only).
MOUNTAIN_WALKABLE = frozenset({
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA7, 0xA8,
    0xC0, 0xC2, 0xC3, 0xC6,
    0x35,  # transitional load tile during mountain entry scroll
    0xD6,
})

# Farm-style debris that can appear on mountain (off spa corridor in spring
# ROM dump: stumps west of climb, large rocks near mid-ridge). BFS must not
# treat them as path; clear only if a future mountain-clear task is added.
# Do NOT include LARGE_ROCK_DAMAGE_TILES (0x11-0x14): those IDs collide with
# common mountain terrain fills in the metatile grid (false-positive debris).
MOUNTAIN_DEBRIS_TILES = frozenset(
    {WEED, STONE, ROCK, FENCE} | STUMP_TILES | LARGE_ROCK_TILES
)

WALKABLE_BY_TILEMAP = {
    0x00: FARM_WALKABLE,
    0x01: FARM_WALKABLE,
    0x02: FARM_WALKABLE,
    0x03: FARM_WALKABLE,
    0x0C: PATH_WALKABLE,
    0x04: TOWN_WALKABLE,
    0x05: TOWN_WALKABLE,
    0x10: MOUNTAIN_WALKABLE,
    0x15: SHOP_WALKABLE | COOP_WALKABLE,
    0x16: SHOP_WALKABLE | COOP_WALKABLE,
    0x17: SHOP_WALKABLE | COOP_WALKABLE,
    0x1B: CHURCH_WALKABLE,
    0x1C: SHOP_WALKABLE,
    0x24: SHOP_WALKABLE,
    0x28: COOP_WALKABLE,
    0x29: SHOP_WALKABLE | COOP_WALKABLE,  # MapMountainCave interior (not outdoor spa)
}


# ── Labels / categories for exports ──────────────────────────────────

TILE_LABEL = {
    0x00: "empty",
    UNTILLED: "untilled",
    DRIED_TILLED: "tilled",
    WEED: "weed",
    STONE: "stone",
    FENCE: "fence",
    ROCK: "rock",
    FRESH_TILLED: "hoed",
    WATERED_TILLED: "watered_soil",
    PLANTED_GRASS_TILE: "grass_planted",
    0x79: "grass_stage_3",
    0x80: "grass_mature_1",
    0x81: "grass_mature_2",
    0x82: "grass_mature_3",
    0x83: "grass_mature_4",
    0x84: "grass_mature_5",
    0x85: "grass_mature_6",
    0xA0: "path",
    0xA1: "structure",
    0xA2: "path2",
    0xA3: "path3",
    0xA4: "floor",
    0xA5: "structure2",
    0xA6: "pond_edge",
    0xA7: "path_edge",
    0xA8: "border",
    0xC0: "path",
    0xC1: "building",
    0xC3: "floor",
    0xC4: "building",
    0xC5: "building_walkable",
    0xC6: "building",
    0xD0: "building",
    0xD1: "building",
    0xD2: "building",
    0xD3: "building",
    0xD4: "floor",
    0xD5: "church_floor",
    0xD6: "floor",
    0xD7: "building",
    0xD8: "building",
    0xDA: "church_floor",
    0xDB: "church_floor",
    0xDC: "church_floor",
    0xE0: "building",
    0xE1: "building",
    0xF0: "water",
    0xF1: "water",
    0xF2: "water",
    0xF7: "water",
    0xF8: "water",
    0xF9: "water",
    0xFA: "water",
    0xFB: "water",
    0xFC: "water",
    0xFD: "water",
    0xFF: "unloaded",
}
for _tid in STUMP_TILES:
    TILE_LABEL.setdefault(_tid, "stump")
for _tid in LARGE_ROCK_TILES | LARGE_ROCK_DAMAGE_TILES:
    TILE_LABEL.setdefault(_tid, "large_rock")

TILE_GLYPH = {
    0x00: ".",
    UNTILLED: "-",
    DRIED_TILLED: "=",
    WEED: "w",
    STONE: "o",
    FENCE: "|",
    ROCK: "O",
    FRESH_TILLED: "~",
    WATERED_TILLED: "*",
    PLANTED_GRASS_TILE: "G",
    0xA0: " ",
    0xA1: "#",
    0xA2: " ",
    0xA3: " ",
    0xA5: "#",
    0xA6: "P",
    0xA7: " ",
    0xA8: "#",
    0xF0: "P",
    0xFF: "#",
}
for _tid in STUMP_TILES:
    TILE_GLYPH[_tid] = "S"
for _tid in LARGE_ROCK_TILES | LARGE_ROCK_DAMAGE_TILES:
    TILE_GLYPH[_tid] = "R"
for _tid in CROP_TILE_RANGE:
    TILE_GLYPH[_tid] = "C"
for _tid in MATURE_GRASS_TILES:
    TILE_GLYPH[_tid] = "g"
for _tid in (0xC1, 0xC4, 0xC5, 0xC6, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD6, 0xD7, 0xD8, 0xE0, 0xE1):
    TILE_GLYPH[_tid] = "B"
for _tid in REFILL_WATER_TILES:
    TILE_GLYPH[_tid] = "P"


def metatile_grid_base(ram: np.ndarray) -> int:
    """Return the most likely base address for the 64x64 metatile grid."""
    live_base = ADDR_MAP + LIVE_RAM_WRAM_OFFSET
    if len(ram) > WRAM_SNAPSHOT_SIZE and live_base + MAP_TILE_COUNT <= len(ram):
        live_nonzero = int(np.count_nonzero(ram[live_base:live_base + MAP_TILE_COUNT]))
        if live_nonzero > 0:
            return live_base
    return ADDR_MAP


def get_tile_at(ram: np.ndarray, tx: int, ty: int) -> int:
    if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_HEIGHT:
        return 0
    addr = metatile_grid_base(ram) + ty * MAP_WIDTH + tx
    return int(ram[addr]) if addr < len(ram) else 0


def set_tile_at(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_HEIGHT:
        raise IndexError(f"tile out of bounds: ({tx}, {ty})")
    addr = metatile_grid_base(ram) + ty * MAP_WIDTH + tx
    if addr >= len(ram):
        raise IndexError(f"tile address outside RAM: 0x{addr:X}")
    ram[addr] = tile_id & 0xFF


def read_tile_grid(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> list[list[int]]:
    x0, y0, x1, y1 = normalize_bounds(bounds)
    return [
        [get_tile_at(ram, x, y) for x in range(x0, x1 + 1)]
        for y in range(y0, y1 + 1)
    ]


def normalize_bounds(bounds: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if bounds is None:
        return (0, 0, MAP_WIDTH - 1, MAP_HEIGHT - 1)
    x0, y0, x1, y1 = bounds
    return (
        max(0, min(MAP_WIDTH - 1, x0)),
        max(0, min(MAP_HEIGHT - 1, y0)),
        max(0, min(MAP_WIDTH - 1, x1)),
        max(0, min(MAP_HEIGHT - 1, y1)),
    )


def tile_label(tile_id: int) -> str:
    if tile_id in CROP_TILE_RANGE:
        return "crop_watered" if is_watered_crop_tile(tile_id) else "crop_dry"
    return TILE_LABEL.get(tile_id, f"0x{tile_id:02X}")


def tile_glyph(tile_id: int) -> str:
    return TILE_GLYPH.get(tile_id, "?")


def is_crop_tile(tile_id: int) -> bool:
    return tile_id in CROP_TILE_RANGE


def is_dry_crop_tile(tile_id: int) -> bool:
    return tile_id in DRY_CROP_TILES


def is_watered_crop_tile(tile_id: int) -> bool:
    return tile_id in WET_CROP_TILES


def crop_pickup_stage(tile_id: int) -> int:
    return tile_id if tile_id % 2 == 0 else tile_id - 1


def is_mature_crop_tile(tile_id: int) -> bool:
    return tile_id in MATURE_CROP_TILES


def tile_is_watered(tile_id: int) -> bool:
    return tile_id == WATERED_TILLED or is_watered_crop_tile(tile_id)


def tile_needs_watering(tile_id: int, *, include_fresh_tilled: bool = False) -> bool:
    if is_mature_crop_tile(tile_id):
        return False
    if include_fresh_tilled and tile_id == FRESH_TILLED:
        return True
    return tile_id in UNRIPE_DRY_CROP_TILES


def crop_stage(tile_id: int) -> int | None:
    if not is_crop_tile(tile_id):
        return None
    return (tile_id - CROP_TILE_RANGE.start) // 2


def tile_category(tile_id: int) -> str:
    if tile_id in STALE_TILE_IDS:
        return "unloaded"
    if tile_id in TILE_TO_DEBRIS:
        return "debris"
    if tile_id == UNTILLED:
        return "tillable"
    if tile_id in PLANTABLE_TILES:
        return "plantable"
    if tile_id == WATERED_TILLED:
        return "watered_soil"
    if tile_id == PLANTED_GRASS_TILE:
        return "planted_grass"
    if tile_id in MATURE_GRASS_TILES or tile_id == 0x79:
        return "grass"
    if is_crop_tile(tile_id):
        return "crop_watered" if is_watered_crop_tile(tile_id) else "crop_dry"
    if tile_id in WATER_TILES:
        return "water"
    return "walkable" if tile_id in FARM_WALKABLE else "other"


def walkable_tiles_for_tilemap(tilemap_id: int) -> frozenset[int]:
    return WALKABLE_BY_TILEMAP.get(tilemap_id, FARM_WALKABLE)


def tile_histogram(grid: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in grid:
        for tile_id in row:
            counts[tile_label(int(tile_id))] += 1
    return dict(counts.most_common())


def tile_category_counts(grid: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in grid:
        for tile_id in row:
            counts[tile_category(int(tile_id))] += 1
    return dict(counts.most_common())


@dataclass(frozen=True)
class TileFact:
    x: int
    y: int
    tile_id: int
    label: str
    category: str
    walkable: bool
    crop_stage: int | None = None
    watered: bool | None = None

    def to_dict(self) -> dict:
        data = {
            "x": self.x,
            "y": self.y,
            "tile": [self.x, self.y],
            "tile_id": self.tile_id,
            "tile_hex": f"0x{self.tile_id:02X}",
            "label": self.label,
            "category": self.category,
            "walkable": self.walkable,
        }
        if self.crop_stage is not None:
            data["crop_stage"] = self.crop_stage
            data["watered"] = bool(self.watered)
        return data


def interesting_tile_fact(tilemap_id: int, x: int, y: int, tile_id: int) -> TileFact | None:
    category = tile_category(tile_id)
    if category in {"other", "walkable"}:
        return None
    watered = tile_is_watered(tile_id) if is_crop_tile(tile_id) or tile_id == WATERED_TILLED else None
    return TileFact(
        x=x,
        y=y,
        tile_id=tile_id,
        label=tile_label(tile_id),
        category=category,
        walkable=tile_id in walkable_tiles_for_tilemap(tilemap_id),
        crop_stage=crop_stage(tile_id),
        watered=watered,
    )


def scan_interesting_tiles(
    ram: np.ndarray,
    tilemap_id: int,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> list[TileFact]:
    x0, y0, x1, y1 = normalize_bounds(bounds)
    facts: list[TileFact] = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            fact = interesting_tile_fact(tilemap_id, x, y, get_tile_at(ram, x, y))
            if fact is not None:
                facts.append(fact)
    return facts
