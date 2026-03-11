#!/usr/bin/env python3
"""Lightweight PySide6 tile map editor for Harvest Moon (SNES).

Renders the 64x64 tile grid using pixel-perfect tile extraction from
the emulator frame. Tiles in the current viewport are captured live;
tiles outside use color-coded fallback until seen.

Supports gamepad controller via pygame + keyboard input.

Launch:
    uv run python harvest/editor_app.py
    uv run python harvest/editor_app.py --state Y1_Spring_D1_Farm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PySide6.QtCore import (
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# -- Harvest Moon constants --

SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_PATH = SCRIPT_DIR / "custom_integrations"
STATES_DIR = INTEGRATION_PATH / "HarvestMoon-Snes"
GAME = "HarvestMoon-Snes"

# RAM addresses
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

# Key mapping: Qt key -> SNES button index
# Matches harvest_bot.py's keyboard layout
KEY_TO_BUTTON = {
    Qt.Key.Key_Z: 0,       # B (cancel)
    Qt.Key.Key_C: 8,       # A (confirm)
    Qt.Key.Key_V: 9,       # X (menu)
    Qt.Key.Key_X: 1,       # Y (use item)
    Qt.Key.Key_Up: 4,
    Qt.Key.Key_Down: 5,
    Qt.Key.Key_Left: 6,
    Qt.Key.Key_Right: 7,
    Qt.Key.Key_A: 10,      # L
    Qt.Key.Key_S: 11,      # R
    Qt.Key.Key_Return: 3,  # Start
    Qt.Key.Key_Shift: 2,   # Select
}

# Tile classifications
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
    0x08: "watered", 0x70: "grass_planted", 0xA0: "path",
    0xA1: "structure", 0xA2: "path2", 0xA3: "path3", 0xA5: "structure2",
    0xA6: "pond", 0xA8: "border", 0xFF: "wall",
}

MAP_NAMES = {
    0x00: "Farm", 0x0C: "Path", 0x04: "Town", 0x1C: "Shop",
    0x15: "House", 0x19: "Barn", 0x1A: "Coop", 0x18: "Shed",
}

TOOL_NAMES = {
    0x00: "None", 0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer",
    0x04: "Axe", 0x10: "Watering Can",
}


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
    if tile_id == 0x06 or (0x09 <= tile_id <= 0x14): return (100, 90, 80)
    if tile_id in CROP_TILES:        return (200, 180, 50)
    if tile_id == 0x70:              return (80, 180, 60)
    if tile_id in GRASS_TILES:       return (50, 160, 50)
    if tile_id == 0x01:              return (90, 70, 50)
    if tile_id in (0x02, 0x07):      return (70, 55, 40)
    if tile_id == 0x08:              return (50, 45, 55)
    if tile_id == 0x00:              return (110, 90, 65)
    if tile_id == 0xA6:              return (30, 80, 180)
    return (200, 50, 200)


def _get_tile_at(ram: np.ndarray, tx: int, ty: int) -> int:
    if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_WIDTH:
        return 0
    addr = ADDR_MAP + ty * MAP_WIDTH + tx
    return int(ram[addr]) if addr < len(ram) else 0


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


# ---------------------------------------------------------------------------
# Map Canvas - single QImage approach for performance
# ---------------------------------------------------------------------------

class TileMapCanvas(QGraphicsView):
    """Renders the 64x64 tile grid as a single pixmap using extracted tile graphics.

    Uses a pre-extracted tile atlas (maps/tile_atlas.npy) for pixel-perfect
    rendering. Falls back to color-coded squares for uncaptured tiles.

    Performance strategy:
    - _tile_atlas: 256x16x16x3 numpy array of tile pixel data
    - _base_img: pre-rendered map (rebuilt only when tiles change)
    - Composited each frame: base + emu viewport overlay + player marker
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

        # Load tile atlas (pixel-perfect tile graphics)
        self._tile_atlas = np.zeros((256, TILE_PX, TILE_PX, 3), dtype=np.uint8)
        self._atlas_ids: set[int] = set()
        atlas_path = SCRIPT_DIR / "maps" / "tile_atlas.npy"
        ids_path = SCRIPT_DIR / "maps" / "tile_ids.npy"
        if atlas_path.exists() and ids_path.exists():
            self._tile_atlas = np.load(atlas_path)
            self._atlas_ids = set(int(i) for i in np.load(ids_path))

        # Fallback color LUT for uncaptured tiles
        self._color_lut = np.zeros((256, 3), dtype=np.uint8)
        for tid in range(256):
            self._color_lut[tid] = _tile_color_rgb(tid)

        self._initialized = False

    def _rebuild_base(self):
        """Rebuild the base image from tile grid using tile atlas (fast)."""
        base = np.zeros((MAP_PX_H, MAP_PX_W, 3), dtype=np.uint8)

        for ty in range(MAP_WIDTH):
            for tx in range(MAP_WIDTH):
                tid = int(self._tile_grid[ty, tx])
                y0 = ty * TILE_PX
                x0 = tx * TILE_PX
                if tid in self._atlas_ids:
                    base[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = self._tile_atlas[tid]
                else:
                    base[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX] = self._color_lut[tid]

        self._base_buf = np.ascontiguousarray(base)
        self._base_img = QImage(
            self._base_buf.data, MAP_PX_W, MAP_PX_H,
            MAP_PX_W * 3, QImage.Format.Format_RGB888,
        )

    def update_from_ram(self, ram: np.ndarray, obs: np.ndarray | None = None):
        """Fast per-frame update: blit emu frame + player marker on pre-built base."""
        px, py = _get_pos(ram)

        # Read tile grid
        tile_data = ram[ADDR_MAP:ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        self._tile_grid = tile_data.reshape((MAP_WIDTH, MAP_WIDTH)).copy()

        # Rebuild base only if tiles changed
        if self._base_img is None or not np.array_equal(self._tile_grid, self._prev_tile_grid):
            self._rebuild_base()
            self._prev_tile_grid = self._tile_grid.copy()

        # Start from base image copy
        frame_img = self._base_img.copy()
        painter = QPainter(frame_img)

        # Blit emulator frame at camera position (pixel-perfect viewport)
        if obs is not None and obs.shape[0] == SCREEN_H and obs.shape[1] == SCREEN_W:
            cam_x, cam_y = _camera_offset(px, py)
            obs_bytes = obs.tobytes()
            emu_img = QImage(obs_bytes, SCREEN_W, SCREEN_H, SCREEN_W * 3,
                             QImage.Format.Format_RGB888)
            painter.drawImage(cam_x, cam_y, emu_img)

        # Player marker
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        painter.setBrush(QBrush(QColor(255, 50, 50, 220)))
        painter.drawEllipse(px - 5, py - 5, 10, 10)

        painter.end()
        self._map_item.setPixmap(QPixmap.fromImage(frame_img))

        if not self._initialized:
            self._initialized = True
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

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

    def show_tile_info(self, tx: int, ty: int, tile_id: int):
        name = TILE_NAMES.get(tile_id, "unknown")
        walkable = "Yes" if tile_id in WALKABLE_TILES else "No"
        debris = "Yes" if tile_id in DEBRIS_TILES else "No"
        self._info_label.setText(
            f"Tile ({tx}, {ty})\n"
            f"ID: 0x{tile_id:02X} ({name})\n"
            f"Walkable: {walkable}\n"
            f"Debris: {debris}"
        )


# ---------------------------------------------------------------------------
# Emulator Panel
# ---------------------------------------------------------------------------

class EmulatorPanel(QWidget):
    """Manages emulator + controller. Emits (ram, obs) each frame."""

    frame_ready = Signal(np.ndarray, np.ndarray)  # ram, obs

    def __init__(self, initial_state: str | None = None, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._env = None
        self._running = False
        self._keys_pressed: set[int] = set()
        self._initial_state = initial_state
        self._frame_count = 0

        # Controller via pygame
        self._pygame = None
        self._controller = None
        self._controller_name: str | None = None
        self._init_pygame()

        # Step at 60fps (one step per tick), render at 30fps
        self._step_timer = QTimer(self)
        self._step_timer.setInterval(16)  # ~60fps stepping
        self._step_timer.timeout.connect(self._step_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Frame display
        self._frame_label = QLabel("No emulator session")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(256, 224)
        self._frame_label.setStyleSheet("background-color: #111; color: #888;")
        self._frame_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._frame_label, 1)

        # State selector - NoFocus so keys aren't stolen
        state_row = QHBoxLayout()
        self._state_combo = QComboBox()
        self._state_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._state_combo.setMinimumWidth(150)
        self._populate_states()
        state_row.addWidget(self._state_combo, 1)

        self._start_btn = QPushButton("Start")
        self._start_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._start_btn.clicked.connect(self._on_start)
        state_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        state_row.addWidget(self._stop_btn)
        layout.addLayout(state_row)

        # Info line
        self._info_label = QLabel("Disconnected")
        self._info_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

    def _init_pygame(self):
        """Initialize pygame for controller support (headless - no display)."""
        try:
            import pygame
            pygame.init()
            self._pygame = pygame
            self._refresh_controller()
        except Exception:
            pass

    def _refresh_controller(self):
        if not self._pygame:
            return
        try:
            self._pygame.joystick.quit()
            self._pygame.joystick.init()
            if self._pygame.joystick.get_count() > 0:
                joy = self._pygame.joystick.Joystick(0)
                joy.init()
                self._controller = joy
                self._controller_name = joy.get_name()
            else:
                self._controller = None
                self._controller_name = None
        except Exception:
            self._controller = None

    def _populate_states(self):
        self._state_combo.clear()
        if STATES_DIR.exists():
            states = sorted(p.stem for p in STATES_DIR.glob("*.state"))
            for s in states:
                self._state_combo.addItem(s)
        if self._initial_state:
            idx = self._state_combo.findText(self._initial_state)
            if idx >= 0:
                self._state_combo.setCurrentIndex(idx)

    def _on_start(self):
        state_name = self._state_combo.currentText()
        if not state_name:
            return
        try:
            import stable_retro as retro
            retro.data.Integrations.add_custom_path(str(INTEGRATION_PATH))
            if self._env:
                self._env.close()
                self._env = None
            self._env = retro.make(
                GAME, state=state_name,
                inttype=retro.data.Integrations.ALL,
                use_restricted_actions=retro.Actions.ALL,
                render_mode="rgb_array",
            )
            obs, info = self._env.reset()
            self._frame_count = 0
            self._running = True
            self._step_timer.start()
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._info_label.setText(f"Running: {state_name}")

            ram = self._env.get_ram()
            self.frame_ready.emit(ram, obs)
            self._render_frame(obs)
        except Exception as e:
            self._info_label.setText(f"Error: {e}")

    def _on_stop(self):
        self._step_timer.stop()
        self._running = False
        if self._env:
            self._env.close()
            self._env = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._info_label.setText("Session closed")
        self._frame_label.setText("No emulator session")
        self._frame_label.setPixmap(QPixmap())

    def _poll_controller(self, action: np.ndarray):
        """Read controller state into action array."""
        if not self._pygame or not self._controller:
            return
        try:
            self._pygame.event.pump()
        except Exception:
            return
        joy = self._controller
        try:
            # D-pad via hat
            if joy.get_numhats() > 0:
                hat = joy.get_hat(0)
                if hat[0] < 0: action[6] = 1   # Left
                if hat[0] > 0: action[7] = 1   # Right
                if hat[1] > 0: action[4] = 1   # Up
                if hat[1] < 0: action[5] = 1   # Down
            # Analog stick
            if joy.get_numaxes() >= 2:
                ax = joy.get_axis(0)
                ay = joy.get_axis(1)
                if ax < -0.5: action[6] = 1
                if ax > 0.5:  action[7] = 1
                if ay < -0.5: action[4] = 1
                if ay > 0.5:  action[5] = 1
            # Buttons: Xbox layout -> SNES
            btn_map = {0: 0, 1: 8, 2: 1, 3: 9, 4: 10, 5: 11, 6: 2, 7: 3}
            for joy_btn, snes_btn in btn_map.items():
                if joy_btn < joy.get_numbuttons() and joy.get_button(joy_btn):
                    action[snes_btn] = 1
        except Exception:
            self._controller = None

    def _step_tick(self):
        if not self._running or not self._env:
            return
        action = np.zeros(12, dtype=np.int32)
        # Keyboard
        for key, btn_idx in KEY_TO_BUTTON.items():
            if key in self._keys_pressed:
                action[btn_idx] = 1
        # Controller
        self._poll_controller(action)

        try:
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._frame_count += 1

            ram = self._env.get_ram()
            # Emit every frame for map update, render video every other frame
            self.frame_ready.emit(ram, obs)
            if self._frame_count % 2 == 0:
                self._render_frame(obs)
                self._update_info(ram)

            if terminated or truncated:
                self._on_stop()
        except Exception as e:
            self._info_label.setText(f"Error: {e}")
            self._on_stop()

    def _render_frame(self, obs: np.ndarray):
        h, w = obs.shape[0], obs.shape[1]
        # obs.data may not be contiguous after slicing; ensure bytes copy
        img = QImage(obs.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        scaled = img.scaled(w * 2, h * 2, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.FastTransformation)
        self._frame_label.setPixmap(QPixmap.fromImage(scaled))

    def _update_info(self, ram: np.ndarray):
        px, py = _get_pos(ram)
        tx, ty = px // TILE_PX, py // TILE_PX
        tilemap_id = _get_tilemap_id(ram)
        map_name = MAP_NAMES.get(tilemap_id, f"0x{tilemap_id:02X}")
        tool_id = int(ram[ADDR_TOOL]) if ADDR_TOOL < len(ram) else 0
        tool_name = TOOL_NAMES.get(tool_id, f"0x{tool_id:02X}")
        ctrl = f" | Ctrl: {self._controller_name}" if self._controller_name else ""
        self._info_label.setText(
            f"{map_name} ({tx},{ty}) | {tool_name} | F:{self._frame_count}{ctrl}"
        )

    def handle_key_press(self, key: int):
        self._keys_pressed.add(key)

    def handle_key_release(self, key: int):
        self._keys_pressed.discard(key)

    def close_session(self):
        self._on_stop()


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
# Main Window
# ---------------------------------------------------------------------------

class EditorWindow(QMainWindow):
    def __init__(self, initial_state: str | None = None):
        super().__init__()
        self.setWindowTitle("Harvest Moon Map Editor")
        self.resize(1200, 850)
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

        # Emulator dock (right)
        self._emu_panel = EmulatorPanel(initial_state=initial_state)
        self._emu_panel.frame_ready.connect(self._on_frame_ready)
        emu_dock = QDockWidget("Emulator", self)
        emu_dock.setWidget(self._emu_panel)
        emu_dock.setMinimumWidth(300)
        emu_dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, emu_dock)

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
        self._stats_counter = 0

        # Load static snapshot if initial state given
        if initial_state:
            self._load_static_snapshot(initial_state)

    def _setup_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("View")
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_view)
        view_menu.addAction(fit_action)

    def _fit_view(self):
        self._canvas.fitInView(
            self._canvas.scene().sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def _load_static_snapshot(self, state_name: str):
        try:
            import stable_retro as retro
            retro.data.Integrations.add_custom_path(str(INTEGRATION_PATH))
            env = retro.make(GAME, state=state_name,
                           inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
            obs, _ = env.reset()
            ram = env.get_ram()
            self._canvas.update_from_ram(ram, obs)
            self._stats.update_from_ram(ram)
            tilemap_id = _get_tilemap_id(ram)
            map_name = MAP_NAMES.get(tilemap_id, f"0x{tilemap_id:02X}")
            self._status_map.setText(f"Map: {map_name} (snapshot: {state_name})")
            env.close()
        except Exception as e:
            self._status_map.setText(f"Could not load: {e}")

    def _on_frame_ready(self, ram: np.ndarray, obs: np.ndarray):
        self._canvas.update_from_ram(ram, obs)
        tilemap_id = _get_tilemap_id(ram)
        map_name = MAP_NAMES.get(tilemap_id, f"0x{tilemap_id:02X}")
        px, py = _get_pos(ram)
        self._status_map.setText(f"Map: {map_name} | Player: ({px // TILE_PX},{py // TILE_PX})")

        self._stats_counter += 1
        if self._stats_counter % 120 == 0:
            self._stats.update_from_ram(ram)

    def _on_tile_clicked(self, tx: int, ty: int, tile_id: int):
        self._tile_info.show_tile_info(tx, ty, tile_id)

    def _on_tile_hovered(self, tx: int, ty: int, tile_id: int):
        name = TILE_NAMES.get(tile_id, f"0x{tile_id:02X}")
        w = "W" if tile_id in WALKABLE_TILES else "X"
        self._status_pos.setText(f"({tx},{ty})")
        self._status_tile.setText(f"0x{tile_id:02X} {name} [{w}]")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in KEY_TO_BUTTON:
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
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Harvest Moon Map Editor")
    parser.add_argument("--state", default=None, help="Initial save state to load")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EditorWindow(initial_state=args.state)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
