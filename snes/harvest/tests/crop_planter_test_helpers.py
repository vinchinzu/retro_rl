"""Shared fixtures for crop planter unit tests (split from monofile)."""
from __future__ import annotations

import numpy as np

from harvest.core.tile_catalog import ADDR_MAP, ADDR_X, ADDR_Y
from harvest.tasks.nav import MAP_WIDTH


def blank_ram() -> np.ndarray:
    return np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)


def set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def set_player_tile(ram: np.ndarray, tile: tuple[int, int]) -> None:
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF
