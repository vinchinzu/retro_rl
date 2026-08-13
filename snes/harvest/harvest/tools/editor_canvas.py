"""Tile map canvas and shared map/editor constants for Harvest Moon editor."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from harvest.paths import CUSTOM_INTEGRATIONS_DIR, DEBUG_ALIGNMENT_DIR, PROJECT_DIR
from harvest.core.harvest_state import HarvestStateDocument
from harvest.core.npc_catalog import game_objects
from harvest.maps.map_config import FARM_TILEMAP_IDS, MAP_REGISTRY, ROUTES, MapExit, Waypoint, get_walkable_tiles

# -- Harvest Moon constants --

SCRIPT_DIR = PROJECT_DIR
ROOT_DIR = PROJECT_DIR
INTEGRATION_PATH = CUSTOM_INTEGRATIONS_DIR
STATES_DIR = INTEGRATION_PATH / "HarvestMoon-Snes"
GAME = "HarvestMoon-Snes"
EXPORTS_DIR = SCRIPT_DIR / "debug_alignment" / "editor_exports"

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



# Farm twin helpers used when patching atlas tiles (imported late to avoid cycles).
from harvest.tools.editor_farm_twin import (  # noqa: E402
    _copy_reference_tile_patch,
    _farm_reference_map,
    _farm_tile_uses_reference,
)
