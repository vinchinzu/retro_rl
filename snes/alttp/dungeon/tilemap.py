"""Indoor room attribute grids in bank ``$7F``.

``get_ram()`` is 147456 bytes. Offsets below ``0x2000`` are mirrored at
index 0; the full 128KiB WRAM (``$7E0000``–``$7FFFFF``) starts at
``ram.WRAM_IDX``. That span includes bank ``$7F``. Measured on CastleMain:
the low mirror matches, and both attribute tables are populated.

Layer 0 (``$EE`` = 0, upper floor) is ``dung_bg2_attr_table`` at ``$7F2000``.
Layer 1 (``$EE`` = 1, lower floor) is ``dung_bg1_attr_table`` at ``$7F3000``.
Each table is 64×64 bytes, row-major, one byte per 8×8 pixels. snesrev
``TileDetection_Execute`` indexes it as
``(y & ~7) * 8 + (x & 63) + (layer ? 0x1000 : 0)`` after ``$EC`` masks the
coordinate into the room. CastleMain's mask was ``$01F8``.

CastleMain spawn ``(760, 3520)`` is tile ``(31, 56)`` = ``0x8E`` (south
door). Holding LEFT stops at ``(688, 3496)``, tile ``(22, 53)`` = ``0x00``,
with ``0x04`` on the tile to the west. This module never writes.

``LEGEND`` is the attribute bytes sampled on a named pin. Spike (``0x44``),
conveyor (``0x68``–``0x6B``) and thick grass (``0x40``) are not on any indoor
pin's grid. Overworld collision does not use this table: on
HyruleCastleGrounds the mask was ``$FFF8`` and ``$114`` stayed ``0x8E``
while Link walked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alttp.ram import WRAM_IDX

__all__ = [
    "ATTR_G_RAM",
    "LAYER_BYTES",
    "LEGEND",
    "TILE_COLS",
    "TILE_PX",
    "TILE_ROWS",
    "UNSAMPLED_KINDS",
    "read_room_tiles",
    "tile_kind",
]

# g_ram offset of dung_bg2_attr_table. $7E0000 + 0x12000 = $7F2000.
ATTR_G_RAM = 0x12000
LAYER_BYTES = 0x1000
TILE_COLS = 64
TILE_ROWS = 64
TILE_PX = 8


def read_room_tiles(ram: object, layer: int) -> np.ndarray:
    """Return one layer's 64×64 attribute grid. ``layer`` is 0 (BG2) or 1 (BG1)."""
    which = int(layer)
    if which not in (0, 1):
        raise ValueError(f"layer must be 0 or 1, got {layer}")
    buf = np.asarray(ram, dtype=np.uint8)
    start = WRAM_IDX + ATTR_G_RAM + which * LAYER_BYTES
    raw = np.ascontiguousarray(buf[start : start + LAYER_BYTES])
    if raw.shape[0] != LAYER_BYTES:
        raise ValueError(
            f"get_ram() does not cover bank $7F layer {which} "
            f"(need index {start + LAYER_BYTES}, have {buf.shape[0]})"
        )
    return raw.reshape(TILE_ROWS, TILE_COLS)


@dataclass(frozen=True)
class TileLegend:
    """One attribute byte, the kind it behaved as, and the pin that showed it."""

    value: int
    kind: str
    pin: str
    detail: str
    where: str


# Sampled 2026-09-29. ``where`` is the live pose or the grid cell, not a guess.
LEGEND: tuple[TileLegend, ...] = (
    TileLegend(0x00, "floor", "CastleMain", "", "LEFT stops on tile (22,53)=(688,3496)"),
    TileLegend(0x01, "wall", "CastleMain", "", "room border, tile (0,0), 1734 cells"),
    TileLegend(0x02, "wall", "CastleMain", "", "south-door flank, tile (30,56)"),
    TileLegend(0x03, "wall", "SwampPalaceEntry", "", "plugs the row-48 ledge gap, tile (30,48)"),
    TileLegend(0x04, "wall", "CastleMain", "", "west of the LEFT stop, tile (21,53)"),
    TileLegend(
        0x08,
        "deep_water",
        "SwampPalaceEntry",
        "",
        "14 cells in the room 0x28 channel; a one-cell overlap left $345 clear",
    ),
    TileLegend(
        0x09,
        "shallow_water",
        "FighterSword",
        "",
        "tile (48,15) room 0x55; $114=0x09 $359=1; 45px/36f vs 54px on floor",
    ),
    TileLegend(
        0x0F,
        "ice",
        "IcePalaceEntry",
        "",
        "168-cell block anchored at (8,43), room 0x0E; door does not reach it",
    ),
    TileLegend(0x1D, "stairs", "CastleMain", "north", "2x2 at (31,12), (11,21) and (51,21)"),
    TileLegend(
        0x20,
        "pit",
        "CastleB1Pit",
        "",
        "RIGHT onto (34,56) sets $59=0x0F, then layer 1 to 0 at (1272,3872)",
    ),
    TileLegend(0x28, "ledge", "CastleMain", "north", "platform rim, 72 cells from (16,11)"),
    TileLegend(0x2A, "ledge", "SwampPalaceEntry", "east_west", "vertical rim beside the channel"),
    TileLegend(0x8E, "door", "CastleMain", "entrance", "south door under spawn tile (31,56)"),
    TileLegend(0xF0, "door", "CastleB1Guard", "closed", "2x2 at (31,37), room 0x72"),
    TileLegend(
        0x23,
        "floor",
        "EasternPalaceEntry",
        "switch",
        "center pot 0x71 lift reveals 2x2 at (30,44); Link stood on it and the shutter opened",
    ),
    TileLegend(
        0x82,
        "door",
        "EasternPalaceEntry",
        "open",
        "shutter mouth after the 0x23 switch; Link walked tile (31,37) into the hall",
    ),
    TileLegend(
        0x1C,
        "stairs",
        "Scan_eastern_0xb9",
        "",
        "Link walked tile (25,50) in room 0xB9 while submodule 0x10",
    ),
    TileLegend(
        0x3E,
        "stairs",
        "Scan_eastern_0xb9",
        "",
        "Link walked tile (25,52) in room 0xB9 before the 0x1C conveyor",
    ),
    TileLegend(
        0x66,
        "floor",
        "HeraEntry",
        "down peg",
        "layer 1 cols 24-25 row 49; $C172 00->01 let the west alcove reach (3704,3978)",
    ),
    TileLegend(
        0x67,
        "wall",
        "HeraEntry",
        "raised peg",
        "same cells were 0x67 at $C172=00 and the alcove stopped at (3792,3979)",
    ),
    TileLegend(
        0x5E,
        "stairs",
        "HeraEntry",
        "west down",
        "room 0x77 tile (17,43); holding it at $C172=01 crossed to room 0x87 (3721,4424)",
    ),
)

UNSAMPLED_KINDS = ("spike", "conveyor", "grass")

_BY_VALUE = {row.value: row for row in LEGEND}


def tile_kind(value: int) -> str:
    """Legend kind for an attribute byte, or ``unknown``."""
    row = _BY_VALUE.get(int(value) & 0xFF)
    return row.kind if row is not None else "unknown"
