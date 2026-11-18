"""Tile attribute reader on a synthetic get_ram() fixture. No ROM."""

from __future__ import annotations

import numpy as np
import pytest

from alttp.dungeon.tilemap import (
    ATTR_G_RAM,
    LAYER_BYTES,
    LEGEND,
    TILE_COLS,
    TILE_ROWS,
    UNSAMPLED_KINDS,
    read_room_tiles,
    tile_kind,
)
from alttp.ram import WRAM_IDX

# CastleMain layer 0, scanned 2026-09-29. Spawn (760, 3520) sits on the south
# door. LEFT until the pose stops (688, 3496) has floor under Link and an
# indoor wall on the tile to the west.
_CASTLE_MAIN = (
    (0, 56, 31, 0x8E),
    (0, 56, 30, 0x02),
    (0, 53, 22, 0x00),
    (0, 53, 21, 0x04),
    (1, 56, 31, 0x20),
)


def _fixture(paints: tuple[tuple[int, int, int, int], ...] = _CASTLE_MAIN) -> np.ndarray:
    ram = np.zeros(WRAM_IDX + ATTR_G_RAM + 2 * LAYER_BYTES, dtype=np.uint8)
    for layer, row, col, value in paints:
        ram[WRAM_IDX + 0x12000 + layer * 0x1000 + row * TILE_COLS + col] = value
    return ram


def test_read_room_tiles_returns_64x64_from_bank_7f() -> None:
    assert ATTR_G_RAM == 0x12000
    assert LAYER_BYTES == TILE_ROWS * TILE_COLS == 0x1000
    ram = _fixture()
    upper = read_room_tiles(ram, 0)
    lower = read_room_tiles(ram, 1)
    assert upper.shape == (64, 64)
    assert lower.shape == (64, 64)
    assert int(upper[56, 31]) == 0x8E
    assert int(upper[56, 30]) == 0x02
    assert int(upper[53, 22]) == 0x00
    assert int(upper[53, 21]) == 0x04
    assert int(lower[56, 31]) == 0x20
    assert int(upper[56, 31]) != int(lower[56, 31])
    assert int(upper.sum()) == 0x8E + 0x02 + 0x04


def test_legend_names_a_pin_for_each_sampled_kind() -> None:
    kinds = {row.kind for row in LEGEND}
    assert {
        "floor",
        "wall",
        "pit",
        "shallow_water",
        "deep_water",
        "stairs",
        "door",
        "ledge",
        "ice",
    } <= kinds
    assert len({row.value for row in LEGEND}) == len(LEGEND)
    assert all(row.pin and row.where for row in LEGEND)
    assert tile_kind(0x00) == "floor"
    assert tile_kind(0x04) == "wall"
    assert tile_kind(0x09) == "shallow_water"
    assert tile_kind(0x20) == "pit"
    assert tile_kind(0x8E) == "door"
    assert tile_kind(0x0F) == "ice"
    assert tile_kind(0xFF) == "unknown"
    by_value = {row.value: row for row in LEGEND}
    assert by_value[0x09].pin == "FighterSword"
    assert by_value[0x20].pin == "CastleB1Pit"
    assert by_value[0x28].detail == "north"
    assert by_value[0x2A].detail == "east_west"
    assert set(UNSAMPLED_KINDS) == {"spike", "conveyor", "grass"}
    assert set(UNSAMPLED_KINDS).isdisjoint(kinds)


def test_read_room_tiles_rejects_a_short_buffer_and_a_bad_layer() -> None:
    short = np.zeros(WRAM_IDX + 0x12000 + 10, dtype=np.uint8)
    with pytest.raises(ValueError, match="bank \\$7F"):
        read_room_tiles(short, 0)
    with pytest.raises(ValueError, match="layer"):
        read_room_tiles(_fixture(), 2)
