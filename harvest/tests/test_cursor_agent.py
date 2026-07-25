from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from harvest.tools.cursor_agent import (
    HARVEST_AGENT_INSTRUCTIONS,
    build_harvest_agent_context,
    harvest_agent_context_from_window,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        app.setStyle("Fusion")
    return app


class HarvestAgentContextTests(unittest.TestCase):
    def test_build_context_includes_live_snapshot_and_selection(self) -> None:
        context = build_harvest_agent_context(
            state_name="Y1_Spring_D1_Farm",
            selected_tile=(12, 34),
            emulator_snapshot={
                "mapName": "Farm",
                "playerTileX": 12,
                "playerTileY": 34,
                "frameRgb24Base64": "drop-me",
            },
            tilemap_id=0x00,
            player_tile=(12, 34),
            live_overlay_enabled=True,
            state_document_name="Y1_Spring_D1_Farm_edited",
        )
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context.title, "Harvest Moon Editor")
        self.assertIn("Farm", context.summary)
        self.assertEqual(context.details["game"], "harvest")
        self.assertNotIn("frameRgb24Base64", context.details["emulator_snapshot"])
        self.assertEqual(context.details["selected_tile"], {"x": 12, "y": 34})

    def test_build_context_returns_none_without_useful_state(self) -> None:
        self.assertIsNone(
            build_harvest_agent_context(
                state_name=None,
                selected_tile=None,
                emulator_snapshot=None,
                tilemap_id=None,
                player_tile=None,
                live_overlay_enabled=False,
                state_document_name=None,
            )
        )

    def test_harvest_agent_instructions_are_game_specific(self) -> None:
        joined = "\n".join(HARVEST_AGENT_INSTRUCTIONS)
        self.assertIn("Harvest Moon", joined)
        self.assertIn("harvest/", joined)


class HarvestAgentDockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def tearDown(self) -> None:
        if hasattr(self, "_window"):
            if hasattr(self._window, "cursor_agent_panel"):
                self._window.cursor_agent_panel.shutdown()
            self._window.close()
            self.app.processEvents()

    def test_editor_window_exposes_agent_dock(self) -> None:
        from harvest.tools.editor_app import EditorWindow

        self._window = EditorWindow()
        self.app.processEvents()
        self.assertTrue(hasattr(self._window, "cursor_agent_panel"))
        self.assertTrue(hasattr(self._window, "agent_dock"))
        context = harvest_agent_context_from_window(self._window)
        self.assertIsNotNone(context)
        self.assertEqual(context.details["game"], "harvest")


if __name__ == "__main__":
    unittest.main()
