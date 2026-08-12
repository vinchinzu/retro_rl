"""Animal RAM probes shared by recordings and autonomous tasks."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.nav import TILE_SIZE
from harvest.core.ram_catalog import CHICKEN_SLOT_COUNT, COW_SLOT_COUNT, read_animal_slot_field


COOP_TILEMAP = 0x28
BARN_TILEMAP = 0x27


def chicken_life_stage(status_raw: int) -> str:
    age = (int(status_raw) >> 1) & 0x07
    if age >= 4:
        return "adult"
    if age >= 2:
        return "chick"
    return "egg"


def chicken_slot_snapshots(ram: np.ndarray, *, require_coop: bool = False) -> list[dict[str, object]]:
    """Return persistent chicken slot positions decoded from WRAM.

    ``raw_1`` is the home/current map byte observed as ``0x28`` in the coop.
    Empty coordinates are ignored because older tests and some states leave
    inactive animal-slot position fields zeroed.
    """
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if require_coop and tilemap != COOP_TILEMAP:
        return []

    rows: list[dict[str, object]] = []
    for slot in range(CHICKEN_SLOT_COUNT):
        status = read_animal_slot_field(ram, "chicken", slot, "status_raw")
        if not (status & 0x01):
            continue
        x = read_animal_slot_field(ram, "chicken", slot, "position_x")
        y = read_animal_slot_field(ram, "chicken", slot, "position_y")
        if x == 0 and y == 0:
            continue
        map_raw = read_animal_slot_field(ram, "chicken", slot, "raw_1")
        if require_coop and map_raw not in (0, COOP_TILEMAP):
            continue
        age = (int(status) >> 1) & 0x07
        rows.append(
            {
                "slot": int(slot),
                "status_raw": int(status),
                "status_hex": f"0x{int(status):02X}",
                "age": int(age),
                "stage": chicken_life_stage(status),
                "map_raw": int(map_raw),
                "map_hex": f"0x{int(map_raw):02X}",
                "pixel": [int(x), int(y)],
                "tile": [int(x // TILE_SIZE), int(y // TILE_SIZE)],
                "source": "animal_slot",
            }
        )
    return rows


def adult_chicken_tiles_from_slots(ram: np.ndarray, *, require_coop: bool = False) -> set[tuple[int, int]]:
    """Return only adult chicken tiles that should block navigation.

    Baby chicks can be walked over in-game, so they are useful trace facts but
    should not be temporary path blockers.
    """
    tiles: set[tuple[int, int]] = set()
    for row in chicken_slot_snapshots(ram, require_coop=require_coop):
        if row["stage"] != "adult":
            continue
        tile = row["tile"]
        if isinstance(tile, list) and len(tile) == 2:
            tiles.add((int(tile[0]), int(tile[1])))
    return tiles


def cow_slot_snapshots(ram: np.ndarray, *, require_barn: bool = False) -> list[dict[str, object]]:
    """Return persistent cow slot positions decoded from WRAM."""
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if require_barn and tilemap != BARN_TILEMAP:
        return []

    rows: list[dict[str, object]] = []
    for slot in range(COW_SLOT_COUNT):
        status = read_animal_slot_field(ram, "cow", slot, "status_raw")
        if not (status & 0x01):
            continue
        x = read_animal_slot_field(ram, "cow", slot, "position_x")
        y = read_animal_slot_field(ram, "cow", slot, "position_y")
        if x == 0 and y == 0:
            continue
        home_map = read_animal_slot_field(ram, "cow", slot, "home_map_raw")
        if require_barn and home_map not in (0, BARN_TILEMAP):
            continue
        pregnancy = read_animal_slot_field(ram, "cow", slot, "pregnancy_raw")
        happiness = read_animal_slot_field(ram, "cow", slot, "happiness")
        rows.append(
            {
                "slot": int(slot),
                "status_raw": int(status),
                "status_hex": f"0x{int(status):02X}",
                "home_map_raw": int(home_map),
                "home_map_hex": f"0x{int(home_map):02X}",
                "pregnancy_raw": int(pregnancy),
                "happiness": int(happiness),
                "pixel": [int(x), int(y)],
                "tile": [int(x // TILE_SIZE), int(y // TILE_SIZE)],
                "source": "animal_slot",
            }
        )
    return rows


def cow_tiles_from_slots(ram: np.ndarray, *, require_barn: bool = False) -> set[tuple[int, int]]:
    """Return cow tiles that should be treated as dynamic blockers."""
    tiles: set[tuple[int, int]] = set()
    for row in cow_slot_snapshots(ram, require_barn=require_barn):
        tile = row["tile"]
        if isinstance(tile, list) and len(tile) == 2:
            tiles.add((int(tile[0]), int(tile[1])))
    return tiles


def animal_blocker_tiles_from_slots(ram: np.ndarray, *, tilemap_id: int | None = None) -> set[tuple[int, int]]:
    """Return dynamic animal blockers for the current or requested map."""
    if tilemap_id is None:
        tilemap_id = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap_id == COOP_TILEMAP:
        return adult_chicken_tiles_from_slots(ram, require_coop=True)
    if tilemap_id == BARN_TILEMAP:
        return cow_tiles_from_slots(ram, require_barn=True)
    return set()


def chicken_tiles_from_entities(entities: Iterable[dict[str, object]]) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for entity in entities:
        if entity.get("label") != "chicken":
            continue
        tile = entity.get("tile")
        if isinstance(tile, list) and len(tile) == 2:
            tiles.add((int(tile[0]), int(tile[1])))
    return tiles
