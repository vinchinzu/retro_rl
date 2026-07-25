from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.pop("QT_STYLE_OVERRIDE", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication

import harvest.tools.editor_app as editor_app
from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from harvest.tools.editor_app import (
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    FARM_REFERENCE_MAP_PATH,
    FARM_REFERENCE_STATIC_TILES,
    FARM_REFERENCE_TWIN_TILES,
    FARM_STATE_TWIN_TILES,
    FARM_REFERENCE_WORLD_Y,
    MAP_WIDTH,
    PlanPreviewPanel,
    SCREEN_H,
    SCREEN_W,
    TileMapCanvas,
    EditorWindow,
    RENDER_MODE_ATLAS,
    RENDER_MODE_EXACT,
    map_name,
)
from harvest.tools.emulator_panel import HarvestEmulatorPanel
from harvest.maps.extract_tiles import load_rgb_image
from harvest.runtime.rom_tools import parse_save_state


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ROM_PATH = SCRIPT_DIR / "roms" / "Harvest Moon.sfc"
STATES_DIR = SCRIPT_DIR / "custom_integrations" / "HarvestMoon-Snes"


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        app.setStyle("Fusion")
    return app


class EditorMapNameTests(unittest.TestCase):
    def test_farm_season_tilemap_aliases(self) -> None:
        for tilemap_id in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap_id=tilemap_id):
                self.assertEqual(map_name(tilemap_id), "Farm")

    def test_interior_tilemap_aliases(self) -> None:
        self.assertEqual(map_name(0x15), "House")
        self.assertEqual(map_name(0x16), "House L1")
        self.assertEqual(map_name(0x17), "House L2")
        self.assertEqual(map_name(0x24), "Animal Shop")
        self.assertEqual(map_name(0x26), "Shed")
        self.assertEqual(map_name(0x27), "Barn")
        self.assertEqual(map_name(0x28), "Coop")


class EditorTwinOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def tearDown(self) -> None:
        if hasattr(self, "_widget"):
            self._widget.close()
            self.app.processEvents()

    @staticmethod
    def _blank_ram() -> np.ndarray:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_MAP:ADDR_MAP + MAP_WIDTH * MAP_WIDTH] = 0x01
        return ram

    def test_canvas_decodes_live_game_object_markers_from_ram(self) -> None:
        canvas = TileMapCanvas()
        self._widget = canvas
        ram = self._blank_ram()
        offset = GOBJ_STRUCT_BASE + GOBJ_STRUCT_STRIDE
        ram[offset] = GOBJ_INITIALIZED & 0xFF
        ram[offset + 1] = GOBJ_INITIALIZED >> 8
        ram[offset + 2] = 0x01
        ram[offset + 3] = 0x02
        ram[offset + 8] = 160
        ram[offset + 9] = 0
        ram[offset + 10] = 176
        ram[offset + 11] = 0

        canvas.update_from_ram(ram, None)

        labels = [getattr(obj, "label", "") for obj in canvas._entity_markers]
        self.assertIn("candidate_npc_0201", labels)

    def test_full_rom_render_is_not_replaced_by_live_viewport(self) -> None:
        canvas = TileMapCanvas()
        self._widget = canvas
        ram = self._blank_ram()
        canvas.update_from_ram(ram, None)
        canvas._observed_rgb[:] = 11
        canvas._observed_mask[:] = True
        canvas._base_img = None
        canvas._rebuild_base()

        obs = np.full((SCREEN_H, SCREEN_W, 3), 250, dtype=np.uint8)
        canvas.update_from_ram(ram, obs)

        self.assertEqual(int(canvas._observed_rgb[0, 0, 0]), 11)

    def test_canvas_can_select_named_route_overlay(self) -> None:
        canvas = TileMapCanvas()
        self._widget = canvas
        canvas.set_route_overlay("farm_to_coop")
        canvas.set_route_overlay_enabled(True)

        self.assertTrue(canvas.route_overlay_enabled())
        self.assertEqual(canvas.route_overlay_name(), "farm_to_coop")
        self.assertGreater(len(canvas._route_waypoints), 0)

    def test_plan_preview_builds_from_ram(self) -> None:
        panel = PlanPreviewPanel()
        self._widget = panel
        panel.update_from_ram(self._blank_ram(), state_name=None)

        self.assertIn("phases", panel._summary.text())
        self.assertGreater(panel._tree.topLevelItemCount(), 0)

    def test_farm_reference_overlay_keeps_static_tiles_but_leaves_state_tiles_live(self) -> None:
        self.assertIn(0xC1, FARM_REFERENCE_STATIC_TILES)
        self.assertNotIn(0xA6, FARM_REFERENCE_STATIC_TILES)
        self.assertNotIn(0x70, FARM_REFERENCE_STATIC_TILES)
        self.assertIn(0xA6, FARM_REFERENCE_TWIN_TILES)
        self.assertNotIn(0x70, FARM_REFERENCE_TWIN_TILES)
        self.assertIn(0x70, FARM_STATE_TWIN_TILES)

    def test_rom_locked_canvas_skips_emulator_pixel_capture(self) -> None:
        canvas = TileMapCanvas()
        self._widget = canvas
        ram = self._blank_ram()
        canvas.update_from_ram(ram, None)
        canvas._observed_rgb[:] = 9
        canvas._observed_mask[:] = True
        canvas.lock_rom_render(True)

        obs = np.full((SCREEN_H, SCREEN_W, 3), 250, dtype=np.uint8)
        canvas.update_from_ram(ram, obs)

        self.assertEqual(int(canvas._observed_rgb[0, 0, 0]), 9)

    def test_logical_sync_moves_player_without_obs(self) -> None:
        canvas = TileMapCanvas()
        self._widget = canvas
        ram = self._blank_ram()
        canvas.update_from_ram(ram, None)
        canvas._observed_rgb[:] = 4
        canvas._observed_mask[:] = True
        canvas._rebuild_base()
        canvas.lock_rom_render(True)

        ram_moved = ram.copy()
        ram_moved[ADDR_X] = 32
        canvas.update_logical_from_ram(ram_moved)

        self.assertEqual(int(canvas._observed_rgb[0, 0, 0]), 4)

    def test_autoplay_api_sends_bridge_command_and_updates_state(self) -> None:
        panel = HarvestEmulatorPanel()
        self._widget = panel
        panel.start_bridge = lambda: None  # type: ignore[method-assign]
        panel._set_running(True)
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_send(command: str, **kwargs: object) -> dict[str, object]:
            calls.append((command, dict(kwargs)))
            return {"ok": True, "autoplayEnabled": bool(kwargs.get("enabled"))}

        panel._send_command = fake_send  # type: ignore[method-assign]

        self.assertTrue(panel.set_autoplay_enabled(True))
        self.assertTrue(panel.autoplay_enabled())
        self.assertTrue(panel.autoplay_button.isChecked())
        self.assertEqual(calls[-1][0], "set_autoplay")
        self.assertTrue(calls[-1][1]["enabled"])
        self.assertIn("stateName", calls[-1][1])

        self.assertTrue(panel.set_autoplay_enabled(False))
        self.assertFalse(panel.autoplay_enabled())
        self.assertFalse(panel.autoplay_button.isChecked())

    def test_emulator_frame_scaling_preserves_snes_aspect_ratio(self) -> None:
        panel = HarvestEmulatorPanel()
        self._widget = panel
        panel.start_bridge = lambda: None  # type: ignore[method-assign]
        panel.frame_label.resize(800, 300)
        pixmap = QPixmap(256, 224)
        pixmap.fill(QColor(20, 30, 40))

        panel._set_frame_pixmap(pixmap)

        displayed = panel.frame_label.pixmap()
        self.assertIsNotNone(displayed)
        self.assertLessEqual(displayed.width(), 800)
        self.assertLessEqual(displayed.height(), 300)
        self.assertAlmostEqual(displayed.width() / displayed.height(), 256 / 224, places=2)


def _has_rom_and_state(state_name: str) -> bool:
    return ROM_PATH.exists() and (STATES_DIR / f"{state_name}.state").exists()


@unittest.skipUnless(
    _has_rom_and_state("Y1_After_Buy_Potato"),
    "Harvest Moon ROM and save states not available locally",
)
class EditorWindowSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_twin_cache_dir = editor_app.TWIN_CACHE_DIR
        editor_app.TWIN_CACHE_DIR = Path(self._cache_tmp.name)

    def tearDown(self) -> None:
        if hasattr(self, "_window"):
            if hasattr(self._window, "cursor_agent_panel"):
                self._window.cursor_agent_panel.shutdown()
            self._window._emu_panel.close_session()
            self._window.close()
            self.app.processEvents()
        editor_app.TWIN_CACHE_DIR = self._old_twin_cache_dir
        self._cache_tmp.cleanup()

    def _make_window(self, initial_state: str | None = None) -> EditorWindow:
        self._window = EditorWindow(initial_state=initial_state)
        self.app.processEvents()
        return self._window

    def test_snapshot_loads_map_from_rom_state(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        self.assertIn("Farm", window._status_map.text())
        self.assertIsNotNone(window._last_ram)
        self.assertIsNotNone(window._state_doc)
        self.assertIsNotNone(window._state_editor._document)
        self.assertGreaterEqual(len(window._canvas._scene_exits), 1)
        self.assertIsNotNone(window._canvas._object_clamp_rect)

    def test_snapshot_exports_map_png_only(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        image = window.build_export_map_image()
        self.assertEqual(image.shape, (1024, 1024, 3))
        self.assertIsNotNone(window._last_twin_cache_path)
        self.assertTrue(window._last_twin_cache_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = window.export_current_map_png(Path(tmpdir), prefix="buy_potato")
            self.assertEqual(path.name, "buy_potato_map.png")
            self.assertEqual(sorted(p.name for p in Path(tmpdir).iterdir()), ["buy_potato_map.png"])
            loaded = QImage(str(path))
            self.assertFalse(loaded.isNull())
            self.assertEqual((loaded.width(), loaded.height()), (1024, 1024))

    def test_live_camera_overlay_defaults_off_and_can_be_toggled(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        self.assertFalse(window._canvas.live_overlay_enabled())
        self.assertFalse(window._overlay_action.isChecked())

        window._overlay_action.setChecked(True)
        self.app.processEvents()
        self.assertTrue(window._canvas.live_overlay_enabled())

        window._overlay_action.setChecked(False)
        self.app.processEvents()
        self.assertFalse(window._canvas.live_overlay_enabled())

    def test_layer_panel_live_overlay_stays_in_sync_with_menu(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        self.assertFalse(window._overlay_action.isChecked())

        window._layers._live.setChecked(True)
        self.app.processEvents()
        self.assertTrue(window._canvas.live_overlay_enabled())
        self.assertTrue(window._overlay_action.isChecked())

        window._layers._live.setChecked(False)
        self.app.processEvents()
        self.assertFalse(window._canvas.live_overlay_enabled())
        self.assertFalse(window._overlay_action.isChecked())

    def test_exact_render_is_default_and_atlas_mode_can_be_toggled(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        self.assertEqual(window._canvas.render_mode(), RENDER_MODE_EXACT)
        self.assertEqual(window._layers._render_mode.currentData(), RENDER_MODE_EXACT)

        window._layers._render_mode.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(window._canvas.render_mode(), RENDER_MODE_ATLAS)

        window._layers._render_mode.setCurrentIndex(0)
        self.app.processEvents()
        self.assertEqual(window._canvas.render_mode(), RENDER_MODE_EXACT)

    def test_snapshot_rom_render_fills_observed_mask(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        # ROM render should fill the entire observed mask (no emulator needed)
        self.assertTrue(window._canvas._observed_mask.all())
        rendered = window._canvas.render_viewport_rgb(window._last_ram)
        # Viewport should have non-zero pixels (actual map content)
        self.assertGreater(rendered.sum(), 0)

    @unittest.skipUnless(FARM_REFERENCE_MAP_PATH.exists(), "farm reference map not available")
    def test_farm_static_tiles_use_reference_building_art(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        rendered = window.build_export_map_image()
        reference = load_rgb_image(str(FARM_REFERENCE_MAP_PATH))

        tx, ty = 7, 18
        x0 = tx * 16
        y0 = ty * 16
        src_y0 = y0 - FARM_REFERENCE_WORLD_Y
        np.testing.assert_array_equal(
                rendered[y0 : y0 + 16, x0 : x0 + 16],
                reference[src_y0 : src_y0 + 16, x0 : x0 + 16],
        )

    def test_farm_state_changed_tiles_use_current_state_art(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        rendered = window.build_export_map_image()

        tx, ty = 12, 25
        x0 = tx * 16
        y0 = ty * 16
        tile_id = int(window._canvas._tile_grid[ty, tx])
        self.assertEqual(tile_id, 0x03)
        np.testing.assert_array_equal(
            rendered[y0 : y0 + 16, x0 : x0 + 16],
            window._canvas._tile_atlas[tile_id],
        )

    def test_state_editor_save_button_writes_patched_money_value(self) -> None:
        window = self._make_window(initial_state="Y1_After_Buy_Potato")
        self.assertTrue(window._state_editor._save_button.isEnabled())

        with tempfile.TemporaryDirectory() as tmpdir:
            window._state_doc.state_path = Path(tmpdir) / "Y1_After_Buy_Potato.state"
            window._state_doc.set_scalar_value("money", 43210)
            window._state_editor._save_button.click()
            self.app.processEvents()

            saved_path = Path(tmpdir) / "Y1_After_Buy_Potato_edited.state"
            self.assertTrue(saved_path.exists())
            parsed = parse_save_state(saved_path)
            saved_money = parsed.ram[0x11F04] | (parsed.ram[0x11F05] << 8) | (parsed.ram[0x11F06] << 16)
            self.assertEqual(saved_money, 43210)


@unittest.skipUnless(
    _has_rom_and_state("Y1_Front_House"),
    "Harvest Moon ROM and save states not available locally",
)
class EditorWindowEmulatorTests(unittest.TestCase):
    """Tests that require the emulator (stable_retro)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_twin_cache_dir = editor_app.TWIN_CACHE_DIR
        editor_app.TWIN_CACHE_DIR = Path(self._cache_tmp.name)

    def tearDown(self) -> None:
        if hasattr(self, "_window"):
            if hasattr(self._window, "cursor_agent_panel"):
                self._window.cursor_agent_panel.shutdown()
            self._window._emu_panel.close_session()
            self._window.close()
            self.app.processEvents()
        editor_app.TWIN_CACHE_DIR = self._old_twin_cache_dir
        self._cache_tmp.cleanup()

    def _make_window(self, initial_state: str | None = None) -> EditorWindow:
        self._window = EditorWindow(initial_state=initial_state)
        self.app.processEvents()
        return self._window

    def _start_session(self, state_name: str) -> EditorWindow:
        window = self._make_window()
        self.assertTrue(window.start_emulator_session(state_name))
        self.app.processEvents()
        return window

    def _drive_until_map(self, state_name: str, key: Qt.Key, max_frames: int, expected_map: str) -> None:
        window = self._start_session(state_name)
        window._emu_panel.handle_key_press(key)
        try:
            for _ in range(max_frames):
                window._emu_panel.step_once()
                self.app.processEvents()
                if window._emu_panel.current_map_name() == expected_map:
                    break
        finally:
            window._emu_panel.handle_key_release(key)
            self.app.processEvents()

        self.assertEqual(window._emu_panel.current_map_name(), expected_map)
        self.assertIn(expected_map, window._status_map.text())

    def test_live_editor_enters_house_and_shed(self) -> None:
        self._drive_until_map("Y1_Front_House", Qt.Key.Key_Up, 120, "House")
        self._window._emu_panel.close_session()
        self._window.close()
        self.app.processEvents()
        del self._window

        self._drive_until_map("Y1_Front_Shed", Qt.Key.Key_Up, 10, "Shed")

    def test_live_editor_exits_buildings_to_farm(self) -> None:
        cases = [
            ("Y1_Inside_House", 30),
            ("Y1_Near_Barn", 30),
            ("Y1_Near_Coop", 30),
            ("Y1_Inside_Shed", 240),
        ]
        for state_name, max_frames in cases:
            with self.subTest(state=state_name):
                self._drive_until_map(state_name, Qt.Key.Key_Down, max_frames, "Farm")
                self._window._emu_panel.close_session()
                self._window.close()
                self.app.processEvents()
                del self._window


if __name__ == "__main__":
    unittest.main()
