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
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QStatusBar,
    QTextEdit,
)

from harvest.paths import PROJECT_DIR
from harvest.maps.extract_tiles import save_rgb_image
from harvest.core.harvest_state import HarvestStateDocument
from harvest.maps.map_config import MAP_REGISTRY
from harvest.runtime.rom_tools import (
    HarvestMoonRom,
    build_metatile_atlas,
    read_metatile_grid,
    read_tilemap_id,
    render_full_map,
)
from harvest.runtime.retro_setup import make_harvest_env

# Ensure package root on path (same as pre-split editor_app).
ROOT_DIR = PROJECT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harvest.tools.emulator_panel import HarvestEmulatorPanel, KEY_TO_BUTTON
from harvest.tools.cursor_agent import attach_harvest_agent_dock
from harvest.tools.editor_canvas import (
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
    ADDR_TOOL,
    ADDR_STAMINA,
    BUILDING_TILES,
    CROP_TILES,
    DEBRIS_TILES,
    DOOR_CANDIDATE_TILES,
    EXPORTS_DIR,
    FARM_REFERENCE_BASELINE_STATE,
    FARM_REFERENCE_MAP_PATH,
    FARM_REFERENCE_STATIC_TILES,
    FARM_REFERENCE_TWIN_TILES,
    FARM_REFERENCE_WORLD_Y,
    FARM_STATE_TWIN_TILES,
    GAME,
    GRASS_TILES,
    INTEGRATION_PATH,
    MAP_NAMES,
    MAP_PX_H,
    MAP_PX_W,
    MAP_WIDTH,
    RENDER_MODE_ATLAS,
    RENDER_MODE_EXACT,
    SCREEN_H,
    SCREEN_W,
    SCRIPT_DIR,
    STATES_DIR,
    STRUCTURE_TILES,
    TILE_NAMES,
    TILE_PX,
    TOOL_NAMES,
    TWIN_CACHE_DIR,
    TWIN_CACHE_VERSION,
    TileMapCanvas,
    WALKABLE_TILES,
    WATER_TILES,
    _build_color_patch_lut,
    _build_unknown_map_background,
    _camera_offset,
    _clamp_rect,
    _document_from_state_path,
    _get_pos,
    _get_tilemap_id,
    _is_walkable,
    _slug_label,
    _tile_color_rgb,
    _walkable_tiles_for_map,
    map_name,
)
from harvest.tools.editor_farm_twin import (
    _apply_farm_reference_state_overlay,
    _copy_reference_tile_patch,
    _farm_reference_baseline_grid,
    _farm_reference_map,
    _farm_tile_uses_reference,
    _farm_twin_cache_paths,
    _farm_twin_grid_digest,
    _load_cached_twin_map,
    _load_or_build_farm_twin_map,
)
from harvest.tools.editor_panels import (
    LayerControlsPanel,
    PlanPreviewPanel,
    StateEditorPanel,
    TileInfoPanel,
    TileStatsPanel,
)


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
