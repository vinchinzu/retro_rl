#!/usr/bin/env python3
"""Lightweight PySide6 tile map editor for Harvest Moon (SNES).

Map rendering is ROM-first: loading a save state renders the complete 1024x1024
map from ROM tile data + save-state VRAM/palette, with no emulator required.
The emulator panel is still available for interactive play and live overlay.

Supports gamepad controller via pygame + keyboard input.

Launch:
    ./startup.sh --state Y1_Spring_D1_Farm
    PYTHONPATH=.. uv run --project .. python -m retro_harness.editor_launcher harvest -- --state Y1_Spring_D1_Farm
    uv run python -m harvest.tools.editor_app --state Y1_Spring_D1_Farm
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from harvest.paths import CUSTOM_INTEGRATIONS_DIR, DEBUG_ALIGNMENT_DIR, PROJECT_DIR
from harvest.maps.extract_tiles import load_rgb_image, save_rgb_image
from harvest.core.harvest_state import HarvestStateDocument, WEATHER_CODES
from harvest.core.npc_catalog import game_objects
from harvest.maps.map_config import FARM_TILEMAP_IDS, MAP_REGISTRY, ROUTES, MapExit, Waypoint, get_walkable_tiles
from harvest.planner.day_plan_decision import auto_day_plan_decision
from harvest.runtime.rom_tools import (
    HarvestMoonRom,
    build_metatile_atlas,
    read_metatile_grid,
    read_tilemap_id,
    render_full_map,
)
from harvest.runtime.retro_setup import make_harvest_env

# -- Harvest Moon constants --

SCRIPT_DIR = PROJECT_DIR
ROOT_DIR = PROJECT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
INTEGRATION_PATH = CUSTOM_INTEGRATIONS_DIR
STATES_DIR = INTEGRATION_PATH / "HarvestMoon-Snes"
GAME = "HarvestMoon-Snes"
EXPORTS_DIR = SCRIPT_DIR / "debug_alignment" / "editor_exports"

from harvest.tools.emulator_panel import HarvestEmulatorPanel, KEY_TO_BUTTON
from harvest.tools.cursor_agent import attach_harvest_agent_dock
ADDR_X = 0x00D6
ADDR_Y = 0x00D8
ADDR_TOOL = 0x0921
ADDR_STAMINA = 0x0918
ADDR_MAP = 0x09B6
ADDR_TILEMAP = 0x0022
MAP_WIDTH = 64
TILE_PX = 16  # each tile is 16x16 pixels in-game

# SNES frame dimensions
SCREEN_W = 256
SCREEN_H = 224

# Map pixel dimensions
MAP_PX_W = MAP_WIDTH * TILE_PX  # 1024
MAP_PX_H = MAP_WIDTH * TILE_PX  # 1024
TWIN_CACHE_VERSION = 2
TWIN_CACHE_DIR = DEBUG_ALIGNMENT_DIR / "editor_twin_cache"
FARM_REFERENCE_MAP_PATH = DEBUG_ALIGNMENT_DIR / "reference_ranch_map.png"
FARM_REFERENCE_BASELINE_STATE = "Y1_Spring_D1_Farm"
FARM_REFERENCE_WORLD_Y = 16

# RAM addresses
WALKABLE_TILES = {
    0x00, 0x01, 0x02, 0x03, 0x07, 0x08, 0x70,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85,
    0xA0, 0xA2, 0xA3, 0xA8,
}
DEBRIS_TILES = {0x03, 0x04, 0x05, 0x06, 0x09, 0x0A, 0x0B, 0x0C,
                0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14}
WATER_TILES = {0xA6, 0xF0, 0xF1, 0xF2, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD}
CROP_TILES = set(range(0x1E, 0x70))
GRASS_TILES = {0x70, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85}
BUILDING_TILES = {0xC1, 0xC4, 0xC5, 0xC6, 0xD0, 0xD1, 0xD2, 0xD3,
                  0xD4, 0xD6, 0xD7, 0xD8, 0xE0, 0xE1}
STRUCTURE_TILES = {0xA1, 0xA5}

TILE_NAMES = {
    0x00: "empty", 0x01: "untilled", 0x02: "tilled", 0x03: "weed",
    0x04: "stone", 0x05: "fence", 0x06: "rock", 0x07: "hoed",
    0x08: "watered",
    0x09: "stump_tl", 0x0A: "stump_tr", 0x0B: "stump_bl", 0x0C: "stump_br",
    0x0D: "large_rock_tl", 0x0E: "large_rock_tr",
    0x0F: "large_rock_bl", 0x10: "large_rock_br",
    0x70: "grass_planted", 0xA0: "path",
    0xA1: "structure", 0xA2: "path2", 0xA3: "path3", 0xA5: "structure2",
    0xA6: "pond", 0xA8: "border", 0xFF: "wall",
}

MAP_NAMES = {
    0x00: "Farm", 0x01: "Farm", 0x02: "Farm", 0x03: "Farm",
    0x0C: "Path", 0x04: "Town", 0x1C: "Shop",
    0x24: "Animal Shop",
    0x15: "House", 0x16: "House L1", 0x17: "House L2",
    0x18: "Shed", 0x19: "Barn", 0x1A: "Coop",
    0x26: "Shed", 0x27: "Barn", 0x28: "Coop",
}

DOOR_CANDIDATE_TILES = {
    0xA4,  # Town storefront approach
    0xC0,  # Cross-map path transition
    0xC3,  # Verified shop doorway tile
    0xD6,  # Common interior doorway / threshold
}

TOOL_NAMES = {
    0x00: "None", 0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer",
    0x04: "Axe", 0x0F: "Brush", 0x10: "Watering Can",
}

RENDER_MODE_EXACT = "exact"
RENDER_MODE_ATLAS = "atlas"
FARM_REFERENCE_STATIC_TILES = frozenset(BUILDING_TILES | STRUCTURE_TILES)
FARM_REFERENCE_TWIN_TILES = frozenset(range(0xA0, 0x100))
FARM_STATE_TWIN_TILES = frozenset(range(0x00, 0xA0))
_FARM_REFERENCE_CACHE: np.ndarray | None = None
_FARM_BASELINE_GRID_CACHE: np.ndarray | None = None


def _tile_color_rgb(tile_id: int) -> tuple[int, int, int]:
    """Map tile ID to an RGB fallback color tuple."""
    if tile_id == 0xFF:              return (40, 40, 40)
    if tile_id in WATER_TILES:       return (30, 100, 200)
    if tile_id in BUILDING_TILES:    return (120, 80, 50)
    if tile_id in STRUCTURE_TILES:   return (100, 100, 100)
    if tile_id == 0xA8:              return (80, 80, 80)
    if tile_id in (0xA0, 0xA2, 0xA3): return (180, 160, 120)
    if tile_id == 0x03:              return (60, 140, 40)
    if tile_id == 0x04:              return (160, 160, 170)
    if tile_id == 0x05:              return (140, 100, 60)
    if 0x09 <= tile_id <= 0x0C: return (120, 90, 40)  # stump
    if tile_id == 0x06 or (0x0D <= tile_id <= 0x14): return (100, 90, 80)
    if tile_id in CROP_TILES:        return (200, 180, 50)
    if tile_id == 0x70:              return (80, 180, 60)
    if tile_id in GRASS_TILES:       return (50, 160, 50)
    if tile_id == 0x01:              return (90, 70, 50)
    if tile_id in (0x02, 0x07):      return (70, 55, 40)
    if tile_id == 0x08:              return (50, 45, 55)
    if tile_id == 0x00:              return (110, 90, 65)
    if tile_id == 0xA6:              return (30, 80, 180)
    return (200, 50, 200)


def _build_color_patch_lut() -> np.ndarray:
    patches = np.zeros((256, TILE_PX, TILE_PX, 3), dtype=np.uint8)
    for tile_id in range(256):
        patches[tile_id, :, :] = _tile_color_rgb(tile_id)
    return patches


def _build_unknown_map_background() -> np.ndarray:
    """Build a neutral background for map regions we have not observed yet."""
    base = np.zeros((MAP_PX_H, MAP_PX_W, 3), dtype=np.uint8)
    dark = np.array((20, 22, 28), dtype=np.uint8)
    light = np.array((32, 35, 44), dtype=np.uint8)
    grid = np.array((54, 58, 70), dtype=np.uint8)
    for ty in range(MAP_WIDTH):
        for tx in range(MAP_WIDTH):
            y0 = ty * TILE_PX
            x0 = tx * TILE_PX
            base[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = light if (tx + ty) & 1 else dark
            base[y0, x0 : x0 + TILE_PX] = grid
            base[y0 : y0 + TILE_PX, x0] = grid
            base[y0 + TILE_PX - 1, x0 : x0 + TILE_PX] = grid
            base[y0 : y0 + TILE_PX, x0 + TILE_PX - 1] = grid
    return base


def _decompress_snapshot_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _read_env_wram(env) -> np.ndarray:
    if hasattr(env, "data") and hasattr(env.data, "memory"):
        blocks = getattr(env.data.memory, "blocks", {})
        if 0x7E0000 in blocks:
            return np.frombuffer(bytes(blocks[0x7E0000]), dtype=np.uint8).copy()
    if hasattr(env, "get_ram"):
        return np.asarray(env.get_ram(), dtype=np.uint8).copy()
    raise RuntimeError("Could not read emulator WRAM")


def _get_pos(ram: np.ndarray) -> tuple[int, int]:
    if ADDR_X + 1 < len(ram) and ADDR_Y + 1 < len(ram):
        x = int(ram[ADDR_X]) | (int(ram[ADDR_X + 1]) << 8)
        y = int(ram[ADDR_Y]) | (int(ram[ADDR_Y + 1]) << 8)
        return x, y
    return 0, 0


def _get_tilemap_id(ram: np.ndarray) -> int:
    if ADDR_TILEMAP < len(ram):
        return int(ram[ADDR_TILEMAP])
    return 0


def _camera_offset(px: int, py: int) -> tuple[int, int]:
    """Calculate camera top-left pixel offset (camera centers on player)."""
    cx = max(0, min(px - SCREEN_W // 2, MAP_PX_W - SCREEN_W))
    cy = max(0, min(py - SCREEN_H // 2, MAP_PX_H - SCREEN_H))
    return cx, cy


def map_name(tilemap_id: int) -> str:
    return MAP_NAMES.get(tilemap_id, f"0x{tilemap_id:02X}")


def _slug_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "export"


def _walkable_tiles_for_map(tilemap_id: int) -> set[int]:
    if tilemap_id in MAP_REGISTRY:
        return set(get_walkable_tiles(tilemap_id))
    return set(WALKABLE_TILES)


def _is_walkable(tilemap_id: int, tile_id: int) -> bool:
    return tile_id in _walkable_tiles_for_map(tilemap_id)


def _clamp_rect(scene) -> tuple[int, int, int, int] | None:
    entry = scene.map_entry
    if entry.object_clamp_right is None or entry.object_clamp_down is None:
        return None
    left = entry.object_clamp_left or 0
    right = entry.object_clamp_right
    top = entry.object_clamp_up or 0
    bottom = entry.object_clamp_down
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


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
    cache_path = TWIN_CACHE_DIR / f"{stem}_v{TWIN_CACHE_VERSION}_{stamp}_{digest}.png"
    latest_path = TWIN_CACHE_DIR / f"{stem}_latest.png"
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


def _document_from_state_path(state_name: str, state_path: Path) -> HarvestStateDocument:
    from harvest.runtime.rom_tools import MutableSaveState

    return HarvestStateDocument(state_name, state_path, MutableSaveState.load(state_path))


# ---------------------------------------------------------------------------
# Map Canvas - single QImage approach for performance
# ---------------------------------------------------------------------------

class TileMapCanvas(QGraphicsView):
    """Renders the 64x64 tile grid as a single pixmap.

    Loading from disk populates the full observed buffer from ROM-rendered
    map data (no emulator needed). Live emulator sessions only add viewport
    observations when no complete ROM render is available, keeping sprites and
    dialogue boxes out of the full-map base image.

    Performance strategy:
    - _observed_rgb/_observed_mask: pixels placed in world coords (ROM or emu)
    - _tile_atlas: per-tile ROM atlas for atlas render mode
    - _base_img: pre-rendered map (rebuilt when observed pixels or mode change)
    - Composited each frame: base + overlays + player marker
    - Only 1 QGraphicsPixmapItem, no per-tile items
    """

    tile_hovered = Signal(int, int, int)  # tx, ty, tile_id
    tile_clicked = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(15, 15, 20)))

        self._map_item = QGraphicsPixmapItem()
        self._map_item.setZValue(0)
        self._scene.addItem(self._map_item)

        # Pre-built base image (only rebuilt when tiles change)
        self._base_img: QImage | None = None
        self._base_buf: np.ndarray | None = None

        # Current tile ID grid for hover/click lookups and dirty detection
        self._tile_grid = np.zeros((MAP_WIDTH, MAP_WIDTH), dtype=np.uint8)
        self._prev_tile_grid = np.zeros((MAP_WIDTH, MAP_WIDTH), dtype=np.uint8)
        self._tilemap_id = 0
        self._prev_tilemap_id = -1

        # ROM-rendered metatile atlas (debug mode only)
        self._tile_atlas = np.zeros((256, TILE_PX, TILE_PX, 3), dtype=np.uint8)
        self._fallback_tiles = _build_color_patch_lut()
        self._unknown_base = _build_unknown_map_background()
        self._observed_rgb = np.zeros((MAP_PX_H, MAP_PX_W, 3), dtype=np.uint8)
        self._observed_mask = np.zeros((MAP_PX_H, MAP_PX_W), dtype=bool)
        self._render_mode = RENDER_MODE_EXACT

        self._initialized = False
        self._show_live_overlay = False
        self._show_collision_overlay = False
        self._show_doors_overlay = True
        self._show_clamp_overlay = True
        self._show_sprite_delta = False
        self._show_player_marker = True
        self._show_entities_overlay = True
        self._show_route_overlay = False
        self._scene_exits: list[MapExit] = []
        self._object_clamp_rect: tuple[int, int, int, int] | None = None
        self._entity_markers: list[object] = []
        self._route_name = ""
        self._route_waypoints: list[Waypoint] = []
        self._door_tile_coords: list[tuple[int, int]] = []
        self._blocked_tile_coords: list[tuple[int, int]] = []
        self._rom_render_locked = False
        self._last_ram: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None

    def set_atlas(self, atlas: np.ndarray) -> None:
        """Set the metatile atlas (256, 16, 16, 3) for rendering."""
        self._tile_atlas = atlas
        if self._render_mode == RENDER_MODE_ATLAS:
            self._base_img = None  # force rebuild

    def clear_observed_pixels(self) -> None:
        self._observed_rgb.fill(0)
        self._observed_mask.fill(False)
        self._base_img = None
        self._base_buf = None

    def lock_rom_render(self, locked: bool = True) -> None:
        self._rom_render_locked = locked

    def rom_render_locked(self) -> bool:
        return self._rom_render_locked

    def _tile_patch_from_atlas(self, tile_id: int) -> np.ndarray:
        patch = self._tile_atlas[tile_id]
        if patch.any():
            return patch
        return self._fallback_tiles[tile_id]

    def _patch_tile_at(self, tx: int, ty: int) -> None:
        tile_id = int(self._tile_grid[ty, tx])
        y0 = ty * TILE_PX
        x0 = tx * TILE_PX
        self._observed_rgb[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = self._tile_patch_from_atlas(
            tile_id
        )
        self._observed_mask[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = True
        if self._tilemap_id in FARM_TILEMAP_IDS and _farm_tile_uses_reference(self._tile_grid, tx, ty):
            reference = _farm_reference_map()
            if reference is not None:
                _copy_reference_tile_patch(self._observed_rgb, reference, tx, ty)

    def _patch_changed_tiles_from_atlas(self) -> None:
        if not self._rom_render_locked or not self._tile_atlas.any():
            return
        changed = np.argwhere(self._tile_grid != self._prev_tile_grid)
        for ty, tx in changed:
            self._patch_tile_at(int(tx), int(ty))

    def set_scene_metadata(
        self,
        *,
        exits: list[MapExit] | None = None,
        object_clamp_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._scene_exits = list(exits or [])
        self._object_clamp_rect = object_clamp_rect
        self._refresh_frame()

    def set_collision_overlay_enabled(self, enabled: bool) -> None:
        self._show_collision_overlay = enabled
        self._refresh_frame()

    def collision_overlay_enabled(self) -> bool:
        return self._show_collision_overlay

    def set_doors_overlay_enabled(self, enabled: bool) -> None:
        self._show_doors_overlay = enabled
        self._refresh_frame()

    def doors_overlay_enabled(self) -> bool:
        return self._show_doors_overlay

    def set_clamp_overlay_enabled(self, enabled: bool) -> None:
        self._show_clamp_overlay = enabled
        self._refresh_frame()

    def clamp_overlay_enabled(self) -> bool:
        return self._show_clamp_overlay

    def set_sprite_delta_enabled(self, enabled: bool) -> None:
        self._show_sprite_delta = enabled
        self._refresh_frame()

    def sprite_delta_enabled(self) -> bool:
        return self._show_sprite_delta

    def set_player_marker_enabled(self, enabled: bool) -> None:
        self._show_player_marker = enabled
        self._refresh_frame()

    def player_marker_enabled(self) -> bool:
        return self._show_player_marker

    def set_entities_overlay_enabled(self, enabled: bool) -> None:
        self._show_entities_overlay = enabled
        self._refresh_frame()

    def entities_overlay_enabled(self) -> bool:
        return self._show_entities_overlay

    def set_route_overlay_enabled(self, enabled: bool) -> None:
        self._show_route_overlay = enabled
        self._refresh_frame()

    def route_overlay_enabled(self) -> bool:
        return self._show_route_overlay

    def set_route_overlay(self, route_name: str | None) -> None:
        clean_name = str(route_name or "")
        if clean_name and clean_name not in ROUTES:
            clean_name = ""
        self._route_name = clean_name
        self._route_waypoints = list(ROUTES.get(clean_name, ()))
        self._refresh_frame()

    def route_overlay_name(self) -> str:
        return self._route_name

    def set_render_mode(self, mode: str) -> None:
        if mode not in {RENDER_MODE_EXACT, RENDER_MODE_ATLAS}:
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._render_mode == mode:
            return
        self._render_mode = mode
        self._base_img = None
        self._refresh_frame()

    def render_mode(self) -> str:
        return self._render_mode

    def _tile_patch(self, tid: int) -> np.ndarray:
        """Get the 16x16 RGB patch for a tile ID."""
        if self._render_mode == RENDER_MODE_ATLAS:
            patch = self._tile_atlas[tid]
            if patch.any():
                return patch
        return self._fallback_tiles[tid]

    def _build_base_buffer(self) -> np.ndarray:
        if self._render_mode == RENDER_MODE_EXACT:
            base = self._unknown_base.copy()
            if np.any(self._observed_mask):
                base[self._observed_mask] = self._observed_rgb[self._observed_mask]
            return np.ascontiguousarray(base)

        base = np.zeros((MAP_PX_H, MAP_PX_W, 3), dtype=np.uint8)

        for ty in range(MAP_WIDTH):
            for tx in range(MAP_WIDTH):
                tid = int(self._tile_grid[ty, tx])
                y0 = ty * TILE_PX
                x0 = tx * TILE_PX
                base[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = self._tile_patch(tid)
        return np.ascontiguousarray(base)

    def _rebuild_base(self):
        """Rebuild the displayed base image."""
        self._base_buf = self._build_base_buffer()
        self._base_img = QImage(
            self._base_buf.data, MAP_PX_W, MAP_PX_H,
            MAP_PX_W * 3, QImage.Format.Format_RGB888,
        )

    def render_map_rgb(self) -> np.ndarray:
        if self._base_buf is not None:
            return np.ascontiguousarray(self._base_buf.copy())
        return self._build_base_buffer()

    def render_viewport_rgb(self, ram: np.ndarray) -> np.ndarray:
        px, py = _get_pos(ram)
        cam_x, cam_y = _camera_offset(px, py)
        base = self.render_map_rgb()
        return np.ascontiguousarray(base[cam_y : cam_y + SCREEN_H, cam_x : cam_x + SCREEN_W].copy())

    def _capture_observation(self, px: int, py: int, obs: np.ndarray | None) -> bool:
        if obs is None or obs.shape[0] != SCREEN_H or obs.shape[1] != SCREEN_W:
            return False
        cam_x, cam_y = _camera_offset(px, py)
        x1 = min(MAP_PX_W, cam_x + obs.shape[1])
        y1 = min(MAP_PX_H, cam_y + obs.shape[0])
        observed = np.ascontiguousarray(obs[: y1 - cam_y, : x1 - cam_x].copy())
        current = self._observed_rgb[cam_y:y1, cam_x:x1]
        current_mask = self._observed_mask[cam_y:y1, cam_x:x1]
        changed = not current_mask.all() or not np.array_equal(current, observed)
        self._observed_rgb[cam_y:y1, cam_x:x1] = observed
        self._observed_mask[cam_y:y1, cam_x:x1] = True
        return changed

    def _compose_frame_image(
        self,
        px: int,
        py: int,
        *,
        obs: np.ndarray | None,
        include_player_marker: bool,
        include_live_overlay: bool,
    ) -> QImage:
        frame_img = self._base_img.copy()
        painter = QPainter(frame_img)

        if (
            include_live_overlay
            and obs is not None
            and obs.shape[0] == SCREEN_H
            and obs.shape[1] == SCREEN_W
        ):
            cam_x, cam_y = _camera_offset(px, py)
            obs_bytes = obs.tobytes()
            emu_img = QImage(
                obs_bytes,
                SCREEN_W,
                SCREEN_H,
                SCREEN_W * 3,
                QImage.Format.Format_RGB888,
            )
            painter.drawImage(cam_x, cam_y, emu_img)

        if self._show_collision_overlay:
            self._draw_collision_overlay(painter)

        if self._show_doors_overlay:
            self._draw_door_overlay(painter)

        if self._show_clamp_overlay:
            self._draw_clamp_overlay(painter)

        if self._show_sprite_delta and obs is not None:
            self._draw_sprite_delta_overlay(painter, px, py, obs)

        if self._show_route_overlay:
            self._draw_route_overlay(painter)

        if self._show_entities_overlay:
            self._draw_entity_overlay(painter)

        if include_player_marker:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.setBrush(QBrush(QColor(255, 50, 50, 220)))
            painter.drawEllipse(px - 5, py - 5, 10, 10)

        painter.end()
        return frame_img

    def _refresh_frame(self) -> None:
        if self._last_ram is None or self._base_img is None:
            return
        px, py = _get_pos(self._last_ram)
        frame_img = self._compose_frame_image(
            px,
            py,
            obs=self._last_obs,
            include_player_marker=self._show_player_marker,
            include_live_overlay=self._show_live_overlay,
        )
        self._map_item.setPixmap(QPixmap.fromImage(frame_img))

    def _rebuild_overlay_tile_cache(self) -> None:
        self._door_tile_coords = [
            (tx, ty)
            for ty in range(MAP_WIDTH)
            for tx in range(MAP_WIDTH)
            if int(self._tile_grid[ty, tx]) in DOOR_CANDIDATE_TILES
        ]
        walkable_tiles = _walkable_tiles_for_map(self._tilemap_id)
        self._blocked_tile_coords = [
            (tx, ty)
            for ty in range(MAP_WIDTH)
            for tx in range(MAP_WIDTH)
            if int(self._tile_grid[ty, tx]) not in walkable_tiles
        ]

    def _draw_collision_overlay(self, painter: QPainter) -> None:
        for tx, ty in self._blocked_tile_coords:
            painter.fillRect(
                tx * TILE_PX,
                ty * TILE_PX,
                TILE_PX,
                TILE_PX,
                QColor(205, 55, 55, 72),
            )

    def _draw_door_overlay(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(255, 210, 90, 220), 2))
        for exit_region in self._scene_exits:
            x1, y1, x2, y2 = exit_region.region
            painter.drawRect(
                x1 * TILE_PX,
                y1 * TILE_PX,
                max(1, (x2 - x1 + 1) * TILE_PX),
                max(1, (y2 - y1 + 1) * TILE_PX),
            )
            painter.drawText(
                x1 * TILE_PX + 4,
                y1 * TILE_PX + 14,
                f"{exit_region.direction}->{map_name(exit_region.dest_tilemap)}",
            )

        painter.setPen(QPen(QColor(80, 235, 255, 230), 2))
        painter.setBrush(QBrush(QColor(80, 235, 255, 70)))
        for tx, ty in self._door_tile_coords:
            tid = int(self._tile_grid[ty, tx])
            painter.drawRect(tx * TILE_PX + 2, ty * TILE_PX + 2, TILE_PX - 4, TILE_PX - 4)
            painter.drawText(tx * TILE_PX + 3, ty * TILE_PX + 13, f"{tid:02X}")

    def _draw_clamp_overlay(self, painter: QPainter) -> None:
        if self._object_clamp_rect is None:
            return
        left, top, right, bottom = self._object_clamp_rect
        pen = QPen(QColor(170, 120, 255, 235), 3)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(left, top, max(1, right - left), max(1, bottom - top))
        painter.drawText(left + 6, top + 18, "Sprite Clamp")

    def _draw_route_overlay(self, painter: QPainter) -> None:
        if not self._route_waypoints:
            return
        current = [
            (index, waypoint)
            for index, waypoint in enumerate(self._route_waypoints, start=1)
            if int(waypoint.tilemap) == int(self._tilemap_id)
        ]
        if not current:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        route_pen = QPen(QColor(90, 220, 255, 235), 3)
        painter.setPen(route_pen)
        previous: Waypoint | None = None
        for _index, waypoint in current:
            if previous is not None:
                painter.drawLine(
                    int(previous.target_px[0]),
                    int(previous.target_px[1]),
                    int(waypoint.target_px[0]),
                    int(waypoint.target_px[1]),
                )
            previous = waypoint

        for index, waypoint in current:
            x, y = int(waypoint.target_px[0]), int(waypoint.target_px[1])
            radius = max(4, min(12, int(waypoint.radius)))
            fill = QColor(90, 220, 255, 75)
            if waypoint.is_exit:
                fill = QColor(255, 210, 90, 95)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
            painter.drawText(x + radius + 3, y + 4, f"{index}")
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _draw_entity_overlay(self, painter: QPainter) -> None:
        if not self._entity_markers:
            return
        color_by_kind = {
            "animal": QColor(255, 190, 80, 220),
            "npc_candidate": QColor(120, 230, 255, 220),
            "game_object": QColor(190, 160, 255, 210),
        }
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for obj in self._entity_markers:
            if getattr(obj, "is_player", False):
                continue
            x, y = getattr(obj, "pixel", (None, None))
            if x is None or y is None:
                continue
            x = int(x)
            y = int(y)
            if not (0 <= x < MAP_PX_W and 0 <= y < MAP_PX_H):
                continue
            kind = str(getattr(obj, "kind", "game_object"))
            color = color_by_kind.get(kind, QColor(180, 220, 120, 220))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 85)))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            label = str(getattr(obj, "label", "") or f"slot_{getattr(obj, 'slot', '?')}")
            if label:
                painter.drawText(x + 7, y - 7, label)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _draw_sprite_delta_overlay(self, painter: QPainter, px: int, py: int, obs: np.ndarray) -> None:
        if self._base_buf is None:
            return
        cam_x, cam_y = _camera_offset(px, py)
        expected = self._base_buf[cam_y : cam_y + SCREEN_H, cam_x : cam_x + SCREEN_W]
        if expected.shape != obs.shape:
            return
        diff = np.abs(obs.astype(np.int16) - expected.astype(np.int16)).mean(axis=2)
        mask = diff > 48
        if not np.any(mask):
            return
        overlay = np.zeros((SCREEN_H, SCREEN_W, 4), dtype=np.uint8)
        overlay[mask] = (255, 40, 220, 120)
        overlay = np.ascontiguousarray(overlay)
        sprite_img = QImage(
            overlay.data,
            SCREEN_W,
            SCREEN_H,
            SCREEN_W * 4,
            QImage.Format.Format_RGBA8888,
        )
        painter.drawImage(cam_x, cam_y, sprite_img)

    def update_from_ram(self, ram: np.ndarray, obs: np.ndarray | None = None):
        """Update the logical grid plus any exact observed pixels from the emulator."""
        self._last_ram = ram.copy()
        self._last_obs = None if obs is None else obs.copy()
        px, py = _get_pos(ram)
        self._tilemap_id = _get_tilemap_id(ram)
        try:
            self._entity_markers = list(game_objects(ram))
        except Exception:
            self._entity_markers = []

        # Read tile grid
        tile_data = ram[ADDR_MAP:ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        self._tile_grid = tile_data.reshape((MAP_WIDTH, MAP_WIDTH)).copy()

        if self._tilemap_id != self._prev_tilemap_id:
            self.clear_observed_pixels()

        observed_changed = False
        if (
            self._render_mode == RENDER_MODE_EXACT
            and not self._rom_render_locked
            and not self._observed_mask.all()
        ):
            observed_changed = self._capture_observation(px, py, obs)

        # Rebuild base only if tiles changed
        grid_changed = not np.array_equal(self._tile_grid, self._prev_tile_grid)
        if grid_changed and self._rom_render_locked:
            self._patch_changed_tiles_from_atlas()

        if (
            self._base_img is None
            or self._tilemap_id != self._prev_tilemap_id
            or grid_changed
            or (self._render_mode == RENDER_MODE_EXACT and observed_changed)
        ):
            self._rebuild_overlay_tile_cache()
            self._rebuild_base()
            self._prev_tile_grid = self._tile_grid.copy()
            self._prev_tilemap_id = self._tilemap_id

        self._refresh_frame()

        if not self._initialized:
            self._initialized = True
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_logical_from_ram(self, ram: np.ndarray) -> None:
        """Lightweight twin sync: markers/grid only, no emulator pixel capture."""

        self._last_ram = ram.copy()
        px, py = _get_pos(ram)
        self._tilemap_id = _get_tilemap_id(ram)
        try:
            self._entity_markers = list(game_objects(ram))
        except Exception:
            self._entity_markers = []

        tile_data = ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        self._tile_grid = tile_data.reshape((MAP_WIDTH, MAP_WIDTH)).copy()

        tilemap_changed = self._tilemap_id != self._prev_tilemap_id
        if tilemap_changed:
            self.clear_observed_pixels()
            self._rom_render_locked = False

        grid_changed = not np.array_equal(self._tile_grid, self._prev_tile_grid)
        if grid_changed and self._rom_render_locked:
            self._patch_changed_tiles_from_atlas()

        if (
            self._base_img is None
            or tilemap_changed
            or grid_changed
        ):
            self._rebuild_overlay_tile_cache()
            self._rebuild_base()
            self._prev_tile_grid = self._tile_grid.copy()
            self._prev_tilemap_id = self._tilemap_id

        self._refresh_frame()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            tx = int(pos.x()) // TILE_PX
            ty = int(pos.y()) // TILE_PX
            if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
                tid = int(self._tile_grid[ty, tx])
                self.tile_clicked.emit(tx, ty, tid)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        tx = int(pos.x()) // TILE_PX
        ty = int(pos.y()) // TILE_PX
        if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
            tid = int(self._tile_grid[ty, tx])
            self.tile_hovered.emit(tx, ty, tid)
        super().mouseMoveEvent(event)

    def center_on_tile(self, tx: int, ty: int):
        self.centerOn(tx * TILE_PX + TILE_PX / 2, ty * TILE_PX + TILE_PX / 2)

    def set_live_overlay_enabled(self, enabled: bool):
        self._show_live_overlay = enabled
        self._refresh_frame()

    def live_overlay_enabled(self) -> bool:
        return self._show_live_overlay


# ---------------------------------------------------------------------------
# Tile Info Panel
# ---------------------------------------------------------------------------

class TileInfoPanel(QWidget):
    """Shows selected tile info and color legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._info_label = QLabel("Click a tile for details")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px;")
        layout.addWidget(self._info_label)

        legend_label = QLabel("Legend:")
        legend_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(legend_label)

        legend_items = [
            (0x00, "Empty ground"), (0x01, "Untilled soil"), (0x02, "Tilled"),
            (0x08, "Watered"), (0x03, "Weed"), (0x04, "Stone"), (0x05, "Fence"),
            (0x06, "Rock"), (0x70, "Planted grass"), (0x80, "Mature grass"),
            (0x1E, "Crop"), (0xA0, "Path"), (0xA1, "Structure"), (0xA6, "Pond"),
            (0xC1, "Building"), (0xF0, "Water"), (0xFF, "Wall"),
        ]
        for tile_id, name in legend_items:
            r, g, b = _tile_color_rgb(tile_id)
            item = QLabel(f"  {name}")
            item.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            item.setStyleSheet(
                f"color: #ccc; font-size: 10px; padding: 1px 4px; "
                f"border-left: 8px solid rgb({r},{g},{b});"
            )
            layout.addWidget(item)
        layout.addStretch()

    def show_tile_info(self, tx: int, ty: int, tile_id: int, tilemap_id: int | None = None):
        name = TILE_NAMES.get(tile_id, "unknown")
        if tilemap_id is None:
            walkable = "Yes" if tile_id in WALKABLE_TILES else "No"
        else:
            walkable = "Yes" if _is_walkable(tilemap_id, tile_id) else "No"
        debris = "Yes" if tile_id in DEBRIS_TILES else "No"
        doorish = "Yes" if tile_id in DOOR_CANDIDATE_TILES else "No"
        self._info_label.setText(
            f"Tile ({tx}, {ty})\n"
            f"ID: 0x{tile_id:02X} ({name})\n"
            f"Walkable: {walkable}\n"
            f"Debris: {debris}\n"
            f"Door/Exit Candidate: {doorish}"
        )


class LayerControlsPanel(QWidget):
    def __init__(self, canvas: TileMapCanvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        hint = QLabel("Useful map layers")
        hint.setStyleSheet("color: #ddd; font-size: 12px; font-weight: bold;")
        layout.addWidget(hint)

        render_label = QLabel("Base render")
        render_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(render_label)

        self._render_mode = QComboBox()
        self._render_mode.addItem("Exact observed pixels", RENDER_MODE_EXACT)
        self._render_mode.addItem("ROM atlas (debug only)", RENDER_MODE_ATLAS)
        current = canvas.render_mode()
        self._render_mode.setCurrentIndex(0 if current == RENDER_MODE_EXACT else 1)
        self._render_mode.currentIndexChanged.connect(self._on_render_mode_changed)
        layout.addWidget(self._render_mode)

        self._doors = QCheckBox("Doors / transitions")
        self._doors.setChecked(canvas.doors_overlay_enabled())
        self._doors.toggled.connect(canvas.set_doors_overlay_enabled)
        layout.addWidget(self._doors)

        self._collision = QCheckBox("Collision / blocked tiles")
        self._collision.setChecked(canvas.collision_overlay_enabled())
        self._collision.toggled.connect(canvas.set_collision_overlay_enabled)
        layout.addWidget(self._collision)

        self._clamp = QCheckBox("Sprite clamp bounds")
        self._clamp.setChecked(canvas.clamp_overlay_enabled())
        self._clamp.toggled.connect(canvas.set_clamp_overlay_enabled)
        layout.addWidget(self._clamp)

        self._sprites = QCheckBox("Sprite delta (live only)")
        self._sprites.setChecked(canvas.sprite_delta_enabled())
        self._sprites.toggled.connect(canvas.set_sprite_delta_enabled)
        layout.addWidget(self._sprites)

        self._player = QCheckBox("Player marker")
        self._player.setChecked(canvas.player_marker_enabled())
        self._player.toggled.connect(canvas.set_player_marker_enabled)
        layout.addWidget(self._player)

        self._entities = QCheckBox("Game objects / NPCs")
        self._entities.setChecked(canvas.entities_overlay_enabled())
        self._entities.toggled.connect(canvas.set_entities_overlay_enabled)
        layout.addWidget(self._entities)

        self._live = QCheckBox("Live viewport overlay")
        self._live.setChecked(canvas.live_overlay_enabled())
        self._live.toggled.connect(canvas.set_live_overlay_enabled)
        layout.addWidget(self._live)

        self._route = QCheckBox("Route waypoints")
        self._route.setChecked(canvas.route_overlay_enabled())
        self._route.toggled.connect(self._on_route_overlay_toggled)
        layout.addWidget(self._route)

        self._route_combo = QComboBox()
        self._route_combo.addItem("None", "")
        for route_name in sorted(ROUTES):
            self._route_combo.addItem(route_name, route_name)
        self._route_combo.setEnabled(self._route.isChecked())
        self._route_combo.currentIndexChanged.connect(self._on_route_changed)
        layout.addWidget(self._route_combo)

        note = QLabel(
            "Exact mode draws observed pixels (from ROM render or emulator). "
            "ROM atlas mode renders each tile individually from the atlas. "
            "Doors show known cross-map exits plus door-like tiles. "
            "Routes use map_config waypoints; game objects come from WRAM."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()

    def _on_render_mode_changed(self, _index: int) -> None:
        mode = self._render_mode.currentData()
        self._canvas.set_render_mode(mode)

    def _on_route_overlay_toggled(self, enabled: bool) -> None:
        self._route_combo.setEnabled(enabled)
        self._canvas.set_route_overlay_enabled(enabled)
        if enabled:
            self._canvas.set_route_overlay(str(self._route_combo.currentData() or ""))

    def _on_route_changed(self, _index: int) -> None:
        if self._route.isChecked():
            self._canvas.set_route_overlay(str(self._route_combo.currentData() or ""))

    def set_live_overlay_checked(self, enabled: bool) -> None:
        if self._live.isChecked() == enabled:
            self._canvas.set_live_overlay_enabled(enabled)
            return
        self._live.blockSignals(True)
        self._live.setChecked(enabled)
        self._live.blockSignals(False)
        self._canvas.set_live_overlay_enabled(enabled)


# ---------------------------------------------------------------------------
# Tile Stats Panel
# ---------------------------------------------------------------------------

class TileStatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Category", "Count"])
        self._tree.setColumnWidth(0, 120)
        layout.addWidget(self._tree)

    def update_from_ram(self, ram: np.ndarray):
        self._tree.clear()
        tile_data = ram[ADDR_MAP:ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        categories: dict[str, int] = {}
        for tid in tile_data:
            cat = self._categorize(int(tid))
            categories[cat] = categories.get(cat, 0) + 1
        for cat in sorted(categories, key=lambda c: -categories[c]):
            QTreeWidgetItem(self._tree, [cat, str(categories[cat])])

    @staticmethod
    def _categorize(tile_id: int) -> str:
        if tile_id in DEBRIS_TILES:    return "Debris"
        if tile_id in CROP_TILES:      return "Crop"
        if tile_id in GRASS_TILES:     return "Grass"
        if tile_id in WATER_TILES:     return "Water"
        if tile_id in BUILDING_TILES:  return "Building"
        if tile_id in STRUCTURE_TILES: return "Structure"
        if tile_id in (0xA0, 0xA2, 0xA3): return "Path"
        if tile_id == 0xFF:            return "Wall"
        if tile_id in (0x01, 0x02, 0x07, 0x08): return "Farmland"
        if tile_id == 0x00:            return "Empty"
        return "Other"


# ---------------------------------------------------------------------------
# Day Plan Preview Panel
# ---------------------------------------------------------------------------

class PlanPreviewPanel(QWidget):
    """Read-only RAM-backed day-plan preview for editor validation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._summary = QLabel("Load a snapshot or start the emulator to preview the day plan.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._summary)

        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Item", "Kind", "Detail"])
        self._tree.setColumnWidth(0, 170)
        self._tree.setColumnWidth(1, 90)
        layout.addWidget(self._tree, 1)

    def update_from_ram(self, ram: np.ndarray | None, *, state_name: str | None = None) -> None:
        self._tree.clear()
        if ram is None:
            self._summary.setText("No RAM snapshot available.")
            return
        try:
            decision = auto_day_plan_decision(state_name=state_name, ram=ram)
        except Exception as exc:
            self._summary.setText(f"Plan preview failed: {exc}")
            return

        facts = decision.facts.to_jsonable()
        hour = int(facts.get("hour") or 0)
        minute = int(facts.get("minute") or 0)
        time_text = f"{hour:02}:{minute:02}"
        map_text = facts.get("tilemap")
        map_label = f"0x{int(map_text):02X}" if map_text is not None else "--"
        self._summary.setText(
            f"{len(decision.phases)} phases from {facts.get('source')} facts | "
            f"time {time_text} | map {map_label}"
        )

        phases_root = QTreeWidgetItem(self._tree, ["Phases", "", ""])
        for index, phase in enumerate(decision.phases, start=1):
            detail = self._phase_detail(phase.params)
            item = QTreeWidgetItem(phases_root, [f"{index:02d}. {phase.phase}", phase.kind, detail])
            if phase.failure_policy != "required":
                item.setText(2, f"{detail} | {phase.failure_policy}" if detail else phase.failure_policy)

        if decision.deferred:
            deferred_root = QTreeWidgetItem(self._tree, ["Deferred", "", ""])
            for item in decision.deferred:
                QTreeWidgetItem(
                    deferred_root,
                    [item.phase, item.kind, f"{item.reason} -> {item.retry}"],
                )

        if decision.notes:
            notes_root = QTreeWidgetItem(self._tree, ["Notes", "", ""])
            for note in decision.notes:
                QTreeWidgetItem(notes_root, [str(note), "", ""])

        facts_root = QTreeWidgetItem(self._tree, ["Facts", "", ""])
        for key in sorted(facts):
            QTreeWidgetItem(facts_root, [key, "", str(facts[key])])
        self._tree.expandToDepth(1)

    @staticmethod
    def _phase_detail(params: dict) -> str:
        if not params:
            return ""
        if route := params.get("route"):
            return f"route={route}"
        if target := params.get("target_px"):
            return f"target={tuple(target)}"
        if task := params.get("task_name"):
            return f"task={task}"
        if recording := params.get("recording_name"):
            return f"recording={recording}"
        if direction := params.get("direction"):
            return f"direction={direction}"
        return ", ".join(f"{key}={value}" for key, value in sorted(params.items())[:3])


# ---------------------------------------------------------------------------
# State Editor Panel
# ---------------------------------------------------------------------------

class StateEditorPanel(QWidget):
    """Editable snapshot fields, source-tagged by confidence."""

    state_changed = Signal()
    load_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._document: HarvestStateDocument | None = None
        self._selected_tile: tuple[int, int] | None = None
        self._loading_tree = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._hint = QLabel("Load a snapshot to edit persistent state.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._hint)

        actions = QHBoxLayout()
        self._load_button = QPushButton("Load Snapshot")
        self._load_button.clicked.connect(self.load_requested.emit)
        actions.addWidget(self._load_button)
        self._save_button = QPushButton("Save Patched State")
        self._save_button.clicked.connect(self.save_requested.emit)
        self._save_button.setEnabled(False)
        actions.addWidget(self._save_button)
        layout.addLayout(actions)

        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Field", "Value", "Source"])
        self._tree.setColumnWidth(0, 180)
        self._tree.setColumnWidth(1, 90)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree, 1)

    def set_document(self, document: HarvestStateDocument | None) -> None:
        self._document = document
        if document is None:
            self._hint.setText(
                "Live emulator sessions are view-only here. Select a snapshot in the "
                "emulator panel, click Load Snapshot, edit values, then save a patched state."
            )
            self._selected_tile = None
            self._save_button.setEnabled(False)
        else:
            self._hint.setText(
                "Sources: state=validated from local state diffs, retro=stable-retro metadata, "
                "decomp=provisional. Edit values directly, then save a patched state "
                "to a new *_edited.state file."
            )
            self._save_button.setEnabled(True)
        self._rebuild_tree()

    def select_tile(self, tx: int, ty: int) -> None:
        self._selected_tile = (tx, ty)
        if self._document is not None:
            self._rebuild_tree()

    @property
    def selected_tile(self) -> tuple[int, int] | None:
        return self._selected_tile

    def _format_scalar_value(self, key: str, value: int) -> str:
        if key == "weather_tomorrow":
            return str(value)
        return str(value)

    def _add_item(
        self,
        parent: QTreeWidgetItem,
        label: str,
        value: str,
        source: str,
        payload: tuple[object, ...] | None = None,
        *,
        editable: bool = False,
        tooltip: str | None = None,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label, value, source])
        if editable:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        if payload is not None:
            item.setData(0, Qt.ItemDataRole.UserRole, payload)
        if tooltip:
            item.setToolTip(0, tooltip)
            item.setToolTip(1, tooltip)
            item.setToolTip(2, tooltip)
        return item

    def _rebuild_tree(self) -> None:
        self._loading_tree = True
        self._tree.clear()
        if self._document is None:
            self._loading_tree = False
            return

        sections: dict[str, QTreeWidgetItem] = {}
        for spec in self._document.scalar_fields():
            section = sections.get(spec.section)
            if section is None:
                section = QTreeWidgetItem(self._tree, [spec.section, "", ""])
                sections[spec.section] = section
            value = self._document.scalar_value(spec.key)
            tooltip = spec.note
            if spec.key == "weather_tomorrow":
                tooltip = f"{spec.note} Known codes: {', '.join(WEATHER_CODES.values())}"
            self._add_item(
                section,
                spec.label,
                self._format_scalar_value(spec.key, value),
                spec.source,
                ("scalar", spec.key),
                editable=True,
                tooltip=tooltip or None,
            )

        tile_root = QTreeWidgetItem(self._tree, ["Selected Tile", "", ""])
        if self._selected_tile is None:
            self._add_item(tile_root, "Status", "Click a map tile", "state")
        else:
            tx, ty = self._selected_tile
            tile = self._document.farm_tile(tx, ty)
            self._add_item(tile_root, "Coords", f"({tx}, {ty})", tile.source)
            self._add_item(
                tile_root,
                "Persistent Tile",
                f"0x{tile.persistent_value:02X}",
                tile.source,
                ("farm_tile", tx, ty),
                editable=True,
                tooltip="Persistent farm-state byte. This is the value saved back into the snapshot.",
            )
            self._add_item(
                tile_root,
                "Visible Tile",
                f"0x{tile.visible_value:02X}",
                "state",
                tooltip="Current rendered tile in the active map buffer.",
            )

        cows_root = QTreeWidgetItem(self._tree, ["Cow Slots", "", ""])
        for cow in self._document.cows():
            cow_item = QTreeWidgetItem(cows_root, [f"Cow {cow.slot + 1:02d}", "", cow.source])
            self._add_item(cow_item, "Status Raw", f"0x{cow.status_raw:02X}", cow.source, ("cow", cow.slot, "status_raw"), editable=True)
            self._add_item(cow_item, "Raw 1", f"0x{cow.raw_1:02X}", cow.source, ("cow", cow.slot, "raw_1"), editable=True)
            self._add_item(cow_item, "Home Map Raw", f"0x{cow.home_map_raw:02X}", cow.source, ("cow", cow.slot, "home_map_raw"), editable=True)
            self._add_item(cow_item, "Pregnancy Raw", f"0x{cow.pregnancy_raw:02X}", cow.source, ("cow", cow.slot, "pregnancy_raw"), editable=True)
            self._add_item(cow_item, "Happiness", str(cow.happiness), cow.source, ("cow", cow.slot, "happiness"), editable=True)
            self._add_item(cow_item, "Raw 5", f"0x{cow.raw_5:02X}", cow.source, ("cow", cow.slot, "raw_5"), editable=True)
            self._add_item(cow_item, "Pos X", str(cow.position_x), cow.source, ("cow", cow.slot, "position_x"), editable=True)
            self._add_item(cow_item, "Pos Y", str(cow.position_y), cow.source, ("cow", cow.slot, "position_y"), editable=True)

        chickens_root = QTreeWidgetItem(self._tree, ["Chicken Slots", "", ""])
        for chicken in self._document.chickens():
            chicken_item = QTreeWidgetItem(chickens_root, [f"Chicken {chicken.slot + 1:02d}", "", chicken.source])
            self._add_item(chicken_item, "Status Raw", f"0x{chicken.status_raw:02X}", chicken.source, ("chicken", chicken.slot, "status_raw"), editable=True)
            self._add_item(chicken_item, "Raw 1", f"0x{chicken.raw_1:02X}", chicken.source, ("chicken", chicken.slot, "raw_1"), editable=True)
            self._add_item(chicken_item, "Raw 2", f"0x{chicken.raw_2:02X}", chicken.source, ("chicken", chicken.slot, "raw_2"), editable=True)
            self._add_item(chicken_item, "Raw 3", f"0x{chicken.raw_3:02X}", chicken.source, ("chicken", chicken.slot, "raw_3"), editable=True)
            self._add_item(chicken_item, "Pos X", str(chicken.position_x), chicken.source, ("chicken", chicken.slot, "position_x"), editable=True)
            self._add_item(chicken_item, "Pos Y", str(chicken.position_y), chicken.source, ("chicken", chicken.slot, "position_y"), editable=True)

        self._tree.expandToDepth(0)
        self._loading_tree = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_tree or column != 1 or self._document is None:
            return

        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if payload is None:
            return

        text = item.text(1).strip()
        try:
            value = int(text, 0)
        except ValueError:
            self._rebuild_tree()
            return

        kind = payload[0]
        try:
            if kind == "scalar":
                self._document.set_scalar_value(payload[1], value)
            elif kind == "farm_tile":
                self._document.set_farm_tile_value(payload[1], payload[2], value)
            elif kind == "cow":
                self._document.set_cow_field(payload[1], payload[2], value)
            elif kind == "chicken":
                self._document.set_chicken_field(payload[1], payload[2], value)
            else:
                return
        except (KeyError, IndexError, ValueError):
            self._rebuild_tree()
            return

        self.state_changed.emit()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class EditorWindow(QMainWindow):
    def __init__(self, initial_state: str | None = None):
        super().__init__()
        self.setWindowTitle("Harvest Moon Map Editor")
        self.resize(1200, 850)
        self._rom: HarvestMoonRom | None = None
        self._state_doc: HarvestStateDocument | None = None
        self._current_state_name = initial_state
        self._last_ram: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None
        self._last_twin_cache_path: Path | None = None
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QDockWidget { color: #ccc; }
            QDockWidget::title { background: #16213e; padding: 4px; }
            QTreeWidget { background: #0f0f1a; color: #ccc; border: none; }
            QTreeWidget::item:selected { background: #2a4a7f; }
            QComboBox { background: #16213e; color: #ccc; border: 1px solid #334; padding: 4px; }
            QComboBox QAbstractItemView { background: #16213e; color: #ccc; }
            QPushButton { background: #16213e; color: #ccc; border: 1px solid #334; padding: 4px 12px; }
            QPushButton:hover { background: #1a3a5f; }
            QPushButton:disabled { color: #555; }
            QLabel { color: #ccc; }
            QStatusBar { background: #0f0f1a; color: #888; }
        """)

        # Central widget: tile map canvas
        self._canvas = TileMapCanvas()
        self._canvas.tile_clicked.connect(self._on_tile_clicked)
        self._canvas.tile_hovered.connect(self._on_tile_hovered)
        self.setCentralWidget(self._canvas)

        # Tile info dock (left)
        self._tile_info = TileInfoPanel()
        info_dock = QDockWidget("Tile Info", self)
        info_dock.setWidget(self._tile_info)
        info_dock.setMinimumWidth(180)
        info_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, info_dock)

        # Tile stats dock (left)
        self._stats = TileStatsPanel()
        stats_dock = QDockWidget("Tile Stats", self)
        stats_dock.setWidget(self._stats)
        stats_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, stats_dock)

        # Layer controls dock (left)
        self._layers = LayerControlsPanel(self._canvas)
        layers_dock = QDockWidget("Layers", self)
        layers_dock.setWidget(self._layers)
        layers_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, layers_dock)

        # State editor dock (right)
        self._state_editor = StateEditorPanel()
        self._state_editor.state_changed.connect(self._on_state_document_changed)
        self._state_editor.load_requested.connect(self._load_selected_snapshot)
        self._state_editor.save_requested.connect(self._save_patched_state)
        state_dock = QDockWidget("State Editor", self)
        state_dock.setWidget(self._state_editor)
        state_dock.setMinimumWidth(340)
        state_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, state_dock)

        # Plan preview dock (right)
        self._plan_preview = PlanPreviewPanel()
        plan_dock = QDockWidget("Day Plan Preview", self)
        plan_dock.setWidget(self._plan_preview)
        plan_dock.setMinimumWidth(340)
        plan_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, plan_dock)

        # Emulator dock (right)
        self._emu_panel = HarvestEmulatorPanel(initial_state=initial_state)
        self._emu_panel.snapshot_received.connect(self._on_emulator_snapshot)
        self._emu_panel.running_changed.connect(self._on_emulator_running_changed)
        emu_dock = QDockWidget("Emulator", self)
        emu_dock.setWidget(self._emu_panel)
        emu_dock.setMinimumWidth(300)
        emu_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, emu_dock)

        attach_harvest_agent_dock(self)

        # Status bar
        self._status_map = QLabel("Map: --")
        self._status_pos = QLabel("Hover: --")
        self._status_tile = QLabel("Tile: --")
        sb = QStatusBar()
        sb.addWidget(self._status_map, 1)
        sb.addWidget(self._status_pos, 1)
        sb.addWidget(self._status_tile, 0)
        self.setStatusBar(sb)

        self._setup_menus()

        # Load static snapshot if initial state given
        if initial_state:
            self._load_static_snapshot(initial_state)

    def _setup_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        load_snapshot_action = QAction("Load Selected Snapshot", self)
        load_snapshot_action.setShortcut("Ctrl+L")
        load_snapshot_action.triggered.connect(self._load_selected_snapshot)
        file_menu.addAction(load_snapshot_action)

        save_snapshot_action = QAction("Save Patched State", self)
        save_snapshot_action.setShortcuts(["Ctrl+S", "Ctrl+Shift+S"])
        save_snapshot_action.triggered.connect(self._save_patched_state)
        file_menu.addAction(save_snapshot_action)

        file_menu.addSeparator()
        export_map_action = QAction("Export Map PNG", self)
        export_map_action.setShortcut("Ctrl+Shift+M")
        export_map_action.triggered.connect(self._export_map_png)
        file_menu.addAction(export_map_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("View")
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_view)
        view_menu.addAction(fit_action)

        self._overlay_action = QAction("Show Live Overlay", self)
        self._overlay_action.setCheckable(True)
        self._overlay_action.setChecked(self._canvas.live_overlay_enabled())
        self._overlay_action.toggled.connect(self._layers.set_live_overlay_checked)
        self._layers._live.toggled.connect(self._overlay_action.setChecked)
        view_menu.addAction(self._overlay_action)

    def _fit_view(self):
        self._canvas.fitInView(
            self._canvas.scene().sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def _rom_instance(self) -> HarvestMoonRom:
        if self._rom is None:
            self._rom = HarvestMoonRom.load()
        return self._rom

    def _capture_snapshot_preview(self, state_path: Path) -> tuple[np.ndarray, np.ndarray]:
        raw_state = _decompress_snapshot_bytes(state_path)
        env = make_harvest_env(None)
        try:
            env.reset()
            env.em.set_state(raw_state)
            obs, _reward, _terminated, _truncated, _info = env.step(np.zeros(12, dtype=np.int32))
            ram = _read_env_wram(env)
            return ram, obs.copy()
        finally:
            env.close()

    def _capture_document_preview(self, document: HarvestStateDocument) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="hm_editor_preview_") as tmpdir:
            path = Path(tmpdir) / f"{document.state_name}.state"
            document.save_as(path)
            return self._capture_snapshot_preview(path)

    def _update_canvas_scene_metadata(self, tilemap_id: int) -> None:
        exits = MAP_REGISTRY.get(tilemap_id).exits if tilemap_id in MAP_REGISTRY else []
        try:
            scene = self._rom_instance().read_map_scene(tilemap_id)
            clamp_rect = _clamp_rect(scene)
        except Exception:
            clamp_rect = None
        self._canvas.set_scene_metadata(exits=exits, object_clamp_rect=clamp_rect)

    def _apply_state_document(self) -> None:
        if self._state_doc is None:
            return

        state = self._state_doc.to_data()
        tilemap_id = read_tilemap_id(state.ram)
        scene = self._rom_instance().read_map_scene(tilemap_id)
        atlas = build_metatile_atlas(
            state,
            scene.graphic_preset.bg12nba,
            rom=self._rom_instance(),
            tilemap_id=tilemap_id,
            bg1sc=scene.graphic_preset.bg1sc,
        )
        grid = read_metatile_grid(state.ram)
        rendered_map = render_full_map(atlas, grid)
        full_map, cache_path, cache_hit = _load_or_build_farm_twin_map(
            rendered=rendered_map,
            grid=grid,
            tilemap_id=tilemap_id,
            state_name=self._state_doc.state_name,
            state_path=self._state_doc.state_path,
        )

        ram = np.frombuffer(state.ram, dtype=np.uint8).copy()

        self._last_ram = ram
        self._last_obs = None
        self._canvas.set_atlas(atlas)
        self._canvas.set_scene_metadata(
            exits=MAP_REGISTRY.get(tilemap_id).exits if tilemap_id in MAP_REGISTRY else [],
            object_clamp_rect=_clamp_rect(scene),
        )
        # update_from_ram first (clears observed pixels on tilemap change),
        # then populate the full ROM render so the map is 100% complete
        self._canvas.update_from_ram(ram, None)
        self._canvas._observed_rgb[:] = full_map
        self._canvas._observed_mask[:] = True
        self._canvas._base_img = None
        self._canvas._rebuild_base()
        self._canvas.lock_rom_render(True)
        self._canvas._refresh_frame()
        self._last_twin_cache_path = cache_path
        self._stats.update_from_ram(ram)
        cache_text = "cache" if cache_hit else "precalc"
        self._status_map.setText(
            f"Map: {map_name(tilemap_id)} (state twin {cache_text}: {self._state_doc.state_name})"
        )
        self._state_editor.set_document(self._state_doc)
        self._plan_preview.update_from_ram(ram, state_name=self._state_doc.state_name)

    def _load_selected_snapshot(self) -> None:
        state_name = self._emu_panel.selected_state()
        if not state_name:
            self.statusBar().showMessage("Select a snapshot first", 4000)
            return
        self._load_static_snapshot(state_name)

    def _load_static_snapshot(self, state_name: str):
        """Load a save state and seed the canvas with an exact emulator preview."""
        try:
            self._state_doc = HarvestStateDocument.load(state_name)
            self._current_state_name = state_name
            self._apply_state_document()
        except Exception as e:
            self._state_doc = None
            self._state_editor.set_document(None)
            self._plan_preview.update_from_ram(None)
            self._status_map.setText(f"Could not load: {e}")

    def start_emulator_session(self, state_name: str | None = None) -> bool:
        if state_name is not None:
            self._current_state_name = state_name
        return self._emu_panel.start_session(state_name)

    def set_autoplay_enabled(self, enabled: bool) -> bool:
        return self._emu_panel.set_autoplay_enabled(enabled)

    def autoplay_enabled(self) -> bool:
        return self._emu_panel.autoplay_enabled()

    def _on_emulator_running_changed(self, running: bool) -> None:
        if running:
            self._state_doc = None
            self._state_editor.set_document(None)
            self._status_map.setText(
                f"Map: planning view ({self._current_state_name or 'snapshot'}) — "
                "live twin sync off during playback"
            )
            return
        self._refresh_planning_from_session_end()

    def _refresh_planning_from_session_end(self) -> None:
        """Rebuild the planning map once after a session ends (e.g. end of day)."""
        from harvest.runtime.editor_snapshot import HOT_SAVE_PATH

        state_name = self._current_state_name or self._emu_panel.selected_state()
        if HOT_SAVE_PATH.is_file():
            try:
                label = f"{state_name}_hot" if state_name else "editor_hot"
                self._state_doc = _document_from_state_path(label, HOT_SAVE_PATH)
                self._apply_state_document()
                self._plan_preview.update_from_ram(
                    self._last_ram,
                    state_name=state_name,
                )
                self.statusBar().showMessage(
                    "Refreshed planning map from editor hot save",
                    4000,
                )
                return
            except Exception:
                pass
        if self._last_ram is not None:
            self._plan_preview.update_from_ram(self._last_ram, state_name=state_name)

    def _on_emulator_snapshot(self, snapshot: dict[str, object]) -> None:
        try:
            tilemap_id = int(snapshot.get("tilemapId") or 0)
        except (TypeError, ValueError):
            tilemap_id = 0
        try:
            tx = int(snapshot.get("playerTileX") or 0)
            ty = int(snapshot.get("playerTileY") or 0)
        except (TypeError, ValueError):
            tx = ty = 0
        self._status_map.setText(f"Map: {map_name(tilemap_id)} | Player: ({tx},{ty})")

    def _on_tile_clicked(self, tx: int, ty: int, tile_id: int):
        tilemap_id = _get_tilemap_id(self._last_ram) if self._last_ram is not None else None
        self._tile_info.show_tile_info(tx, ty, tile_id, tilemap_id)
        self._state_editor.select_tile(tx, ty)

    def _on_tile_hovered(self, tx: int, ty: int, tile_id: int):
        name = TILE_NAMES.get(tile_id, f"0x{tile_id:02X}")
        tilemap_id = _get_tilemap_id(self._last_ram) if self._last_ram is not None else 0
        w = "W" if _is_walkable(tilemap_id, tile_id) else "X"
        self._status_pos.setText(f"({tx},{ty})")
        self._status_tile.setText(f"0x{tile_id:02X} {name} [{w}]")

    def _on_state_document_changed(self) -> None:
        self._apply_state_document()

    def _save_patched_state(self) -> None:
        if self._state_doc is None:
            self.statusBar().showMessage("Load a snapshot before saving edits", 4000)
            return
        try:
            path = self._state_doc.save_as()
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}", 5000)
            return
        self.statusBar().showMessage(f"Saved patched state: {path}", 5000)

    def _export_frame_label(self) -> str:
        if self._last_ram is None:
            return _slug_label(self._current_state_name or self._emu_panel.selected_state() or "export")
        state_label = self._current_state_name or self._emu_panel.selected_state() or "export"
        map_label = map_name(_get_tilemap_id(self._last_ram))
        frame_label = f"f{self._emu_panel.frame_count():05d}"
        return _slug_label(f"{state_label}_{map_label}_{frame_label}")

    def build_export_map_image(self) -> np.ndarray:
        if self._last_ram is None:
            raise RuntimeError("Load a snapshot or start a session before exporting")
        return self._canvas.render_map_rgb()

    def export_current_map_png(
        self,
        output_dir: Path = EXPORTS_DIR,
        *,
        prefix: str | None = None,
    ) -> Path:
        image = self.build_export_map_image()
        stem = _slug_label(prefix or self._export_frame_label())
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{stem}_map.png"
        save_rgb_image(path, image)
        self.statusBar().showMessage(f"Exported map PNG to {path}", 5000)
        return path

    def _export_map_png(self) -> None:
        try:
            path = self.export_current_map_png()
        except Exception as exc:
            self.statusBar().showMessage(f"Export failed: {exc}", 5000)
            return
        self.statusBar().showMessage(f"Saved map PNG: {path}", 5000)

    def _focus_accepts_text_input(self) -> bool:
        focus = QApplication.focusWidget()
        return isinstance(focus, (QAbstractSpinBox, QLineEdit, QPlainTextEdit, QTextEdit))

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        text_focus = self._focus_accepts_text_input()
        if self._emu_panel.is_running() and not text_focus and not event.isAutoRepeat():
            if key == Qt.Key.Key_BracketLeft:
                self._emu_panel.decrease_speed()
                event.accept()
                return
            if key == Qt.Key.Key_BracketRight:
                self._emu_panel.increase_speed()
                event.accept()
                return
            if key == Qt.Key.Key_F5:
                self._emu_panel.hot_save()
                event.accept()
                return
            if key == Qt.Key.Key_F1:
                self._emu_panel.load_hot_save()
                event.accept()
                return
            if key == Qt.Key.Key_F6:
                self._emu_panel.toggle_ram_recording()
                event.accept()
                return
            if key == Qt.Key.Key_F8:
                self._emu_panel.toggle_autoplay()
                event.accept()
                return
        if self._emu_panel.is_running() and not text_focus and key in KEY_TO_BUTTON:
            self._emu_panel.handle_key_press(key)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if key in KEY_TO_BUTTON:
            self._emu_panel.handle_key_release(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        self._emu_panel.close_session()
        if hasattr(self, "cursor_agent_panel"):
            self.cursor_agent_panel.shutdown()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Harvest Moon Map Editor")
    parser.add_argument("--state", default=None, help="Initial save state to load")
    parser.add_argument("--autostart", action="store_true", help="Start the embedded emulator after loading")
    parser.add_argument("--autoplay", action="store_true", help="Enable embedded autoplay after starting")
    parser.add_argument("--export-dir", default=None, help="Export a full-map PNG, then exit")
    parser.add_argument("--export-prefix", default=None, help="Optional filename prefix for the exported map PNG")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EditorWindow(initial_state=args.state)
    window.show()
    if args.export_dir:
        def _export_and_quit() -> None:
            try:
                if window._last_ram is None:
                    window._load_static_snapshot(window._emu_panel.selected_state())
                window.export_current_map_png(Path(args.export_dir), prefix=args.export_prefix)
            finally:
                window.close()
                app.quit()

        QTimer.singleShot(0, _export_and_quit)
    elif args.autostart or args.autoplay:
        def _start_session() -> None:
            if window.start_emulator_session(args.state) and args.autoplay:
                window.set_autoplay_enabled(True)

        QTimer.singleShot(0, _start_session)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
