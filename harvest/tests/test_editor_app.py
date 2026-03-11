from __future__ import annotations

import os
import unittest

os.environ.pop("QT_STYLE_OVERRIDE", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from editor_app import EditorWindow, map_name


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        app.setStyle("Fusion")
    return app


class EditorMapNameTests(unittest.TestCase):
    def test_interior_tilemap_aliases(self) -> None:
        self.assertEqual(map_name(0x15), "House")
        self.assertEqual(map_name(0x26), "Shed")
        self.assertEqual(map_name(0x27), "Barn")
        self.assertEqual(map_name(0x28), "Coop")


class EditorWindowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def tearDown(self) -> None:
        if hasattr(self, "_window"):
            self._window._emu_panel.close_session()
            self._window.close()
            self.app.processEvents()

    def _make_window(self, initial_state: str | None = None) -> EditorWindow:
        self._window = EditorWindow(initial_state=initial_state)
        self.app.processEvents()
        return self._window

    def _start_session(self, state_name: str) -> EditorWindow:
        window = self._make_window()
        self.assertTrue(window._emu_panel.start_session(state_name))
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

    def test_snapshot_labels_building_states(self) -> None:
        cases = [
            ("Y1_Inside_House", "House"),
            ("Y1_Inside_Shed", "Shed"),
            ("Y1_Near_Barn", "Barn"),
            ("Y1_Near_Coop", "Coop"),
        ]
        for state_name, expected_map in cases:
            with self.subTest(state=state_name):
                window = self._make_window(initial_state=state_name)
                self.assertIn(expected_map, window._status_map.text())
                window.close()
                self.app.processEvents()
                del self._window

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
