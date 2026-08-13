"""Farm reference / twin map helpers for the Harvest Moon editor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from harvest.core.harvest_state import HarvestStateDocument
from harvest.maps.extract_tiles import load_rgb_image, save_rgb_image
from harvest.maps.map_config import FARM_TILEMAP_IDS
from harvest.runtime.rom_tools import read_metatile_grid
from harvest.tools import editor_canvas as _canvas
from harvest.tools.editor_canvas import (
    FARM_REFERENCE_BASELINE_STATE,
    FARM_REFERENCE_MAP_PATH,
    FARM_REFERENCE_TWIN_TILES,
    FARM_REFERENCE_WORLD_Y,
    MAP_PX_H,
    MAP_PX_W,
    MAP_WIDTH,
    TILE_PX,
    TWIN_CACHE_VERSION,
    _slug_label,
    map_name,
)

_FARM_REFERENCE_CACHE: np.ndarray | None = None
_FARM_BASELINE_GRID_CACHE: np.ndarray | None = None


def _farm_reference_map() -> np.ndarray | None:
    global _FARM_REFERENCE_CACHE
    if _FARM_REFERENCE_CACHE is not None:
        return _FARM_REFERENCE_CACHE
    if not FARM_REFERENCE_MAP_PATH.is_file():
        return None
    try:
        image = load_rgb_image(str(FARM_REFERENCE_MAP_PATH))
    except Exception:
        return None
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[1] <= 0:
        return None
    _FARM_REFERENCE_CACHE = np.ascontiguousarray(image)
    return _FARM_REFERENCE_CACHE


def _farm_reference_baseline_grid() -> np.ndarray | None:
    global _FARM_BASELINE_GRID_CACHE
    if _FARM_BASELINE_GRID_CACHE is not None:
        return _FARM_BASELINE_GRID_CACHE
    try:
        document = HarvestStateDocument.load(FARM_REFERENCE_BASELINE_STATE)
        grid = read_metatile_grid(document.to_data().ram)
    except Exception:
        return None
    if grid.shape[0] < MAP_WIDTH or grid.shape[1] < MAP_WIDTH:
        return None
    _FARM_BASELINE_GRID_CACHE = grid[:MAP_WIDTH, :MAP_WIDTH].copy()
    return _FARM_BASELINE_GRID_CACHE


def _farm_tile_uses_reference(grid: np.ndarray, tx: int, ty: int) -> bool:
    tile_id = int(grid[ty, tx])
    baseline = _farm_reference_baseline_grid()
    if baseline is not None and ty < baseline.shape[0] and tx < baseline.shape[1]:
        return tile_id == int(baseline[ty, tx])
    return tile_id in FARM_REFERENCE_TWIN_TILES


def _copy_reference_tile_patch(base: np.ndarray, reference: np.ndarray, tx: int, ty: int) -> None:
    x0 = tx * TILE_PX
    y0 = ty * TILE_PX
    src_y0 = y0 - FARM_REFERENCE_WORLD_Y
    src_y1 = src_y0 + TILE_PX
    dst_y0 = y0
    dst_y1 = y0 + TILE_PX
    if src_y1 <= 0 or src_y0 >= reference.shape[0] or x0 >= reference.shape[1]:
        return
    if src_y0 < 0:
        dst_y0 -= src_y0
        src_y0 = 0
    if src_y1 > reference.shape[0]:
        dst_y1 -= src_y1 - reference.shape[0]
        src_y1 = reference.shape[0]
    width = min(TILE_PX, reference.shape[1] - x0, base.shape[1] - x0)
    if width <= 0 or dst_y1 <= dst_y0:
        return
    base[dst_y0:dst_y1, x0 : x0 + width] = reference[src_y0:src_y1, x0 : x0 + width]


def _apply_farm_reference_state_overlay(
    rendered: np.ndarray,
    grid: np.ndarray,
    tilemap_id: int,
) -> np.ndarray:
    if tilemap_id not in FARM_TILEMAP_IDS:
        return rendered
    reference = _farm_reference_map()
    if reference is None:
        return rendered

    out = np.ascontiguousarray(rendered.copy())
    for ty in range(min(MAP_WIDTH, grid.shape[0])):
        for tx in range(min(MAP_WIDTH, grid.shape[1])):
            if _farm_tile_uses_reference(grid, tx, ty):
                _copy_reference_tile_patch(out, reference, tx, ty)
    return out


def _farm_twin_grid_digest(grid: np.ndarray) -> str:
    view = np.ascontiguousarray(grid[:MAP_WIDTH, :MAP_WIDTH].astype(np.uint8, copy=False))
    return hashlib.blake2b(view.data, digest_size=8).hexdigest()


def _farm_twin_cache_paths(
    *,
    state_name: str,
    state_path: Path,
    tilemap_id: int,
    grid: np.ndarray,
) -> tuple[Path, Path]:
    try:
        stat = state_path.stat()
        stamp = f"{stat.st_mtime_ns}_{stat.st_size}"
    except OSError:
        stamp = "unknown"
    stem = _slug_label(f"{state_name}_{map_name(tilemap_id)}")
    digest = _farm_twin_grid_digest(grid)
    # Read cache dir via module attr so tests can rebind editor_canvas.TWIN_CACHE_DIR.
    cache_dir = _canvas.TWIN_CACHE_DIR
    cache_path = cache_dir / f"{stem}_v{TWIN_CACHE_VERSION}_{stamp}_{digest}.png"
    latest_path = cache_dir / f"{stem}_latest.png"
    return cache_path, latest_path


def _load_cached_twin_map(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        image = load_rgb_image(str(path))
    except Exception:
        return None
    if image.shape != (MAP_PX_H, MAP_PX_W, 3):
        return None
    return np.ascontiguousarray(image)


def _load_or_build_farm_twin_map(
    *,
    rendered: np.ndarray,
    grid: np.ndarray,
    tilemap_id: int,
    state_name: str,
    state_path: Path,
) -> tuple[np.ndarray, Path | None, bool]:
    if tilemap_id not in FARM_TILEMAP_IDS:
        return rendered, None, False
    cache_path, latest_path = _farm_twin_cache_paths(
        state_name=state_name,
        state_path=state_path,
        tilemap_id=tilemap_id,
        grid=grid,
    )
    cached = _load_cached_twin_map(cache_path)
    if cached is not None:
        return cached, cache_path, True

    image = _apply_farm_reference_state_overlay(rendered, grid, tilemap_id)
    try:
        save_rgb_image(cache_path, image)
        save_rgb_image(latest_path, image)
    except Exception:
        return image, None, False
    return image, cache_path, False

