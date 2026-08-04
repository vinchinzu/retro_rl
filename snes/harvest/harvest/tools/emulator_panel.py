"""Harvest Moon wrapper around the shared embedded emulator panel."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from harvest.paths import PROJECT_DIR, STATES_DIR
from harvest.runtime.editor_snapshot import (
    CAPTURE_DIR,
    DEFAULT_STATE_PATH,
    HOT_SAVE_PATH,
)
from retro_harness.controls import SNES_BUTTON_NAMES
from retro_harness.editor import (
    EmbeddedEmulatorPanelBase,
    EmulatorPanelConfig,
    script_segments_from_file,
)

KEY_TO_BUTTON = {
    Qt.Key.Key_Z: 0,
    Qt.Key.Key_C: 8,
    Qt.Key.Key_V: 9,
    Qt.Key.Key_X: 1,
    Qt.Key.Key_Up: 4,
    Qt.Key.Key_Down: 5,
    Qt.Key.Key_Left: 6,
    Qt.Key.Key_Right: 7,
    Qt.Key.Key_A: 10,
    Qt.Key.Key_S: 11,
    Qt.Key.Key_Return: 3,
    Qt.Key.Key_Shift: 2,
}
RECORDING_DIR = PROJECT_DIR / "tasks"
COMPACT_SNAPSHOT_KEYS = (
    "frameCounter",
    "tilemapId",
    "mapName",
    "playerX",
    "playerY",
    "playerTileX",
    "playerTileY",
    "toolName",
)


def _hud_lines(snapshot: dict[str, object], room_label: str | None) -> list[str]:
    title = str(room_label or snapshot.get("mapName") or "Unknown")
    tx = snapshot.get("playerTileX")
    ty = snapshot.get("playerTileY")
    tool = snapshot.get("toolName") or "?"
    pos = f"({tx},{ty})" if tx is not None and ty is not None else "(?,?)"
    return [title, f"{title} {pos} | {tool}"]


class HarvestEmulatorPanel(EmbeddedEmulatorPanelBase):
    """Bridge-backed emulator panel tuned for fast autoplay and static planning maps."""

    def __init__(self, initial_state: str | None = None, parent=None) -> None:
        self._initial_state = initial_state
        config = EmulatorPanelConfig(
            project_root=PROJECT_DIR,
            bridge_module="harvest.runtime.editor_bridge",
            button_order=SNES_BUTTON_NAMES,
            key_to_button=KEY_TO_BUTTON,
            default_editor_state_file=DEFAULT_STATE_PATH,
            editor_capture_dir=CAPTURE_DIR,
            editor_hot_save_state=HOT_SAVE_PATH,
            default_recording_dir=RECORDING_DIR,
            game_states_dir=STATES_DIR,
            recording_format="harvest-editor-script-recording",
            recording_tool="harvest-editor",
            compact_snapshot_keys=COMPACT_SNAPSHOT_KEYS,
            base_frame_ms=16,
            speed_levels=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
            speed_uses_frame_repeat=True,
            include_wram_when_stepping=False,
            skip_frame_when_turbo=True,
            turbo_speed_threshold=4.0,
            turbo_frame_preview_interval=8,
            unthrottled_speed_threshold=8.0,
        )
        super().__init__(
            config,
            hud_lines=_hud_lines,
            script_segments_from_file=script_segments_from_file,
        )
        self.start_requested.connect(self.start_default_session)
        self.snapshot_received.connect(self._sync_autoplay_button_from_snapshot)
        if initial_state:
            self._pending_state_name = initial_state
        else:
            self._pending_state_name = None
        self._autoplay_enabled = False
        self.autoplay_button = QPushButton("Autoplay")
        self.autoplay_button.setCheckable(True)
        self.autoplay_button.setEnabled(False)
        self.autoplay_button.toggled.connect(self.set_autoplay_enabled)
        layout = self.layout()
        if layout is not None:
            layout.addWidget(self.autoplay_button)
        self.running_changed.connect(self._on_running_changed_for_autoplay)

    def _apply_pending_state(self) -> None:
        if not self._pending_state_name:
            return
        for index in range(self.state_combo.count()):
            if self.state_combo.itemText(index) == self._pending_state_name:
                self.state_combo.setCurrentIndex(index)
                break

    def start_bridge(self) -> None:
        super().start_bridge()
        self._apply_pending_state()

    def selected_state(self) -> str:
        state_file = self.selected_state_file()
        if state_file and state_file != "NONE":
            return Path(state_file).stem
        return self.state_combo.currentText()

    def set_selected_state(self, state_name: str) -> bool:
        index = self.state_combo.findText(state_name)
        if index < 0:
            for candidate in range(self.state_combo.count()):
                data = self.state_combo.itemData(candidate)
                if data and Path(str(data)).stem == state_name:
                    index = candidate
                    break
        if index < 0:
            return False
        self.state_combo.setCurrentIndex(index)
        return True

    def start_session(self, state_name: str | None = None) -> bool:
        self.start_bridge()
        if state_name is not None and not self.set_selected_state(state_name):
            self.status_label.setText(f"Unknown state: {state_name}")
            return False
        self.start_default_session()
        return self.is_running()

    def autoplay_enabled(self) -> bool:
        return self._autoplay_enabled

    def set_autoplay_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if not self.is_running():
            self._set_autoplay_button_checked(False)
            self._autoplay_enabled = False
            return False
        response = self._send_command(
            "set_autoplay",
            enabled=enabled,
            stateName=self.selected_state(),
            includeFrame=True,
        )
        if not isinstance(response, dict) or not response.get("ok"):
            self._set_autoplay_button_checked(self._autoplay_enabled)
            return False
        self._set_autoplay_enabled_from_response(response)
        return self._autoplay_enabled == enabled

    def toggle_autoplay(self) -> bool:
        return self.set_autoplay_enabled(not self._autoplay_enabled)

    def _set_autoplay_enabled_from_response(self, response: dict[str, object]) -> None:
        value = response.get("autoplayEnabled")
        if value is None:
            snapshot = response.get("snapshot")
            if isinstance(snapshot, dict):
                value = snapshot.get("autoplayEnabled")
        self._autoplay_enabled = bool(value)
        self._set_autoplay_button_checked(self._autoplay_enabled)

    def _set_autoplay_button_checked(self, enabled: bool) -> None:
        if self.autoplay_button.isChecked() == enabled:
            return
        self.autoplay_button.blockSignals(True)
        self.autoplay_button.setChecked(enabled)
        self.autoplay_button.blockSignals(False)

    def _on_running_changed_for_autoplay(self, running: bool) -> None:
        self.autoplay_button.setEnabled(running)
        if not running:
            self._autoplay_enabled = False
            self._set_autoplay_button_checked(False)

    def _sync_autoplay_button_from_snapshot(self, snapshot: dict[str, object]) -> None:
        if "autoplayEnabled" not in snapshot:
            return
        self._autoplay_enabled = bool(snapshot.get("autoplayEnabled"))
        self._set_autoplay_button_checked(self._autoplay_enabled)

    def frame_count(self) -> int:
        if self._last_snapshot is None:
            return 0
        try:
            return int(self._last_snapshot.get("frameCounter") or 0)
        except (TypeError, ValueError):
            return 0

    def current_tilemap_id(self) -> int | None:
        if self._last_snapshot is None:
            return None
        value = self._last_snapshot.get("tilemapId")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def current_map_name(self) -> str | None:
        if self._last_snapshot is None:
            return None
        label = self._last_snapshot.get("mapName")
        return str(label) if label else None

    def step_once(self) -> None:
        if not self.is_running():
            return
        action = [0] * len(self._cfg.button_order)
        for key, button_index in self._cfg.key_to_button.items():
            if key in self._keys_pressed:
                action[button_index] = 1
        self._send_command(
            "step",
            action=action,
            repeat=1,
            includeFrame=True,
            includeWram=False,
        )

    def close_session(self) -> None:
        self.stop_session()
        self.stop_bridge()
        self._autoplay_enabled = False
        self._set_autoplay_button_checked(False)

    def last_snapshot(self) -> dict[str, object] | None:
        if self._last_snapshot is None:
            return None
        return dict(self._last_snapshot)
