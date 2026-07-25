"""Generic Qt embedded emulator panel for editor ↔ bridge workflows."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import wave

from PySide6.QtCore import QBuffer, QIODevice, Qt, QTimer, Signal, QCoreApplication
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
except Exception:  # pragma: no cover - depends on host Qt multimedia install
    QAudioFormat = None  # type: ignore[assignment]
    QAudioSource = None  # type: ignore[assignment]
    QMediaDevices = None  # type: ignore[assignment]

from retro_harness.editor.bridge_worker import BridgeController, BridgeReply
from retro_harness.editor.recording import (
    append_recording_marker as _core_append_recording_marker,
    append_recording_segment as _core_append_recording_segment,
    safe_recording_slug,
)
from retro_harness.editor.snapshot import snapshot_frame_counter, snapshot_int, snapshot_without_frame
from retro_harness.editor.util import frame_budget_ms_for_speed

HudLinesFn = Callable[[dict[str, object], str | None], list[str]]
RamTagsSummaryFn = Callable[[object | None], str]
ScriptSegmentsFn = Callable[[Path], tuple[dict[str, object], list[dict[str, object]]]]


@dataclass(frozen=True)
class EmulatorPanelConfig:
    """Static paths and bridge identity for an embedded editor emulator panel."""

    project_root: Path
    bridge_module: str
    button_order: tuple[str, ...]
    key_to_button: dict[int, int]
    default_editor_state_file: Path
    editor_capture_dir: Path
    editor_hot_save_state: Path
    default_recording_dir: Path
    game_states_dir: Path | None = None
    idle_label: str = "IDLE"
    follow_checkbox_label: str = "Follow"
    recording_format: str = "editor-script-recording"
    recording_tool: str = "editor"
    recording_version: int = 1
    headless_bridge_module: str = ""
    env_prefix: str = "EDITOR"
    compact_snapshot_keys: tuple[str, ...] = ()
    base_frame_ms: int = 16
    speed_levels: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    default_speed_index: int = 0
    speed_uses_frame_repeat: bool = False
    wram_sync_interval_frames: int = 0
    include_wram_when_stepping: bool = True
    skip_frame_when_turbo: bool = True
    turbo_speed_threshold: float = 4.0
    turbo_frame_preview_interval: int = 8
    unthrottled_speed_threshold: float = 8.0


class EmbeddedEmulatorPanelBase(QWidget):
    """Bridge-driven emulator frame, controls, recording, and HUD overlay."""

    snapshot_received = Signal(dict)
    running_changed = Signal(bool)
    start_requested = Signal()

    def __init__(
        self,
        config: EmulatorPanelConfig,
        *,
        hud_lines: HudLinesFn,
        script_segments_from_file: ScriptSegmentsFn,
        format_ram_tags_summary: RamTagsSummaryFn | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config
        self._hud_lines = hud_lines
        self._script_segments_from_file = script_segments_from_file
        self._format_ram_tags = format_ram_tags_summary
        self._bridge = BridgeController(
            project_root=self._cfg.project_root,
            bridge_module=self._cfg.bridge_module,
            on_disconnect=self._on_bridge_disconnect,
        )
        self._bridge.reply.connect(self._on_bridge_reply)
        self._running = False
        self._step_generation = 0
        self._step_in_flight = False
        self._pending_step_request_id: str | None = None
        self._step_started = 0.0
        self._last_step_repeat = 1
        self._last_step_action: list[int] = []
        self._turbo_step_counter = 0
        self._speed_index = self._cfg.default_speed_index
        self._target_frame_ms = frame_budget_ms_for_speed(
            self._cfg.speed_levels[self._speed_index],
            base_frame_ms=self._cfg.base_frame_ms,
        )
        self._fps_samples: deque[float] = deque(maxlen=30)
        self._last_frame_ms = 0.0
        self._last_bridge_step_ms = 0.0
        self._last_frame_bytes: bytes | None = None
        self._keys_pressed: set[int] = set()
        self._room_label: str | None = None
        self._last_snapshot: dict[str, object] | None = None
        self._last_frame_pixmap: QPixmap | None = None
        self._recording = False
        self._recording_name = ""
        self._recording_segments: list[dict[str, object]] = []
        self._recording_markers: list[dict[str, object]] = []
        self._recording_frames = 0
        self._recording_start_capture: dict[str, object] | None = None
        self._recording_selected_state_file: str | None = None
        self._last_recording_path: Path | None = None
        self._ram_recording = False
        self._mic_audio = None
        self._mic_buffer: QBuffer | None = None
        self._mic_format = None
        self._mic_started_at_frame = 0
        self._last_wram_sync_frame = -1
        self._force_wram_next_step = False
        self._step_timer = QTimer(self)
        self._step_timer.setSingleShot(True)
        self._step_timer.timeout.connect(self._step_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.frame_label = QLabel("No emulator session")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(256, 224)
        self.frame_label.setStyleSheet("background-color: #111; color: #888;")
        self.frame_label.setScaledContents(False)
        self.frame_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.frame_label, 1)

        state_row = QHBoxLayout()
        self.state_combo = QComboBox()
        state_row.addWidget(self.state_combo, 1)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_requested.emit)
        state_row.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_session)
        self.stop_button.setEnabled(False)
        state_row.addWidget(self.stop_button)
        layout.addLayout(state_row)

        save_row = QHBoxLayout()
        self.hot_save_button = QPushButton("Hot Save")
        self.hot_save_button.clicked.connect(self.hot_save)
        self.hot_save_button.setEnabled(False)
        save_row.addWidget(self.hot_save_button)
        self.load_save_button = QPushButton("Load Save")
        self.load_save_button.clicked.connect(self.load_hot_save)
        self.load_save_button.setEnabled(False)
        save_row.addWidget(self.load_save_button)
        layout.addLayout(save_row)

        automation_row = QHBoxLayout()
        self.ram_record_button = QPushButton("RAM Rec")
        self.ram_record_button.clicked.connect(self.toggle_ram_recording)
        self.ram_record_button.setEnabled(False)
        automation_row.addWidget(self.ram_record_button)
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_session)
        self.capture_button.setEnabled(False)
        automation_row.addWidget(self.capture_button)
        self.run_script_button = QPushButton("Run Script")
        self.run_script_button.clicked.connect(self.run_script_dialog)
        self.run_script_button.setEnabled(False)
        automation_row.addWidget(self.run_script_button)
        layout.addLayout(automation_row)

        record_name_row = QHBoxLayout()
        self.recording_name_edit = QLineEdit()
        self.recording_name_edit.setPlaceholderText("recording / marker label")
        record_name_row.addWidget(self.recording_name_edit, 1)
        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        record_name_row.addWidget(self.record_button)
        layout.addLayout(record_name_row)

        record_action_row = QHBoxLayout()
        self.mark_button = QPushButton("Mark")
        self.mark_button.clicked.connect(self.mark_recording)
        self.mark_button.setEnabled(False)
        record_action_row.addWidget(self.mark_button)
        self.mic_button = QPushButton("Mic Note")
        self.mic_button.clicked.connect(self.toggle_mic_annotation)
        self.mic_button.setEnabled(False)
        record_action_row.addWidget(self.mic_button)
        self.save_recording_button = QPushButton("Save")
        self.save_recording_button.clicked.connect(self.save_recording)
        self.save_recording_button.setEnabled(False)
        record_action_row.addWidget(self.save_recording_button)
        self.run_last_recording_button = QPushButton("Run Last")
        self.run_last_recording_button.clicked.connect(self.run_last_recording)
        self.run_last_recording_button.setEnabled(False)
        record_action_row.addWidget(self.run_last_recording_button)
        layout.addLayout(record_action_row)

        self.follow_check = QCheckBox(self._cfg.follow_checkbox_label)
        self.follow_check.setChecked(True)
        layout.addWidget(self.follow_check)

        self.overlay_check = QCheckBox("HUD overlay")
        self.overlay_check.setChecked(False)
        self.overlay_check.toggled.connect(self._rerender_last_snapshot)
        layout.addWidget(self.overlay_check)

        self.fps_label = QLabel("FPS —")
        self.fps_label.setStyleSheet("color: #6cf; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.fps_label)

        self.status_label = QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)
        self._populate_state_combo()
        QTimer.singleShot(0, self._warm_bridge)

    def _env(self, suffix: str) -> str:
        return f"{self._cfg.env_prefix}_{suffix}"

    def _append_recording_segment(
        self,
        segments: list[dict[str, object]],
        action: list[int],
        frames: int,
    ) -> None:
        _core_append_recording_segment(
            segments,
            action,
            frames,
            button_order=self._cfg.button_order,
            idle_label=self._cfg.idle_label,
        )

    def _append_recording_marker(
        self,
        segments: list[dict[str, object]],
        label: str,
    ) -> dict[str, object]:
        return _core_append_recording_marker(
            segments,
            label,
            idle_label=self._cfg.idle_label,
        )

    def _on_bridge_disconnect(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText(message)

    def _warm_bridge(self) -> None:
        if self._bridge.is_connected():
            return
        try:
            self.start_bridge()
        except Exception:
            self.status_label.setText("Disconnected")

    def _populate_state_combo(self, states: list[dict[str, object]] | None = None) -> None:
        self.state_combo.clear()
        if states is None:
            default_state = self._cfg.default_editor_state_file
            states = [
                {
                    "name": "Reset",
                    "path": "NONE",
                    "default": not default_state.is_file(),
                }
            ]
            if default_state.is_file():
                states.append(
                    {
                        "name": "Latest original",
                        "path": str(default_state),
                        "default": True,
                    }
                )
            if self._cfg.editor_hot_save_state.is_file():
                states.append(
                    {
                        "name": "Editor hot save",
                        "path": str(self._cfg.editor_hot_save_state),
                        "default": False,
                    }
                )
            game_dir = self._cfg.game_states_dir
            if game_dir is not None and game_dir.exists():
                for path in sorted(game_dir.glob("*.state"), key=lambda item: item.stem.casefold()):
                    states.append({"name": path.stem, "path": str(path), "default": False})
        default_index = 0
        for index, entry in enumerate(states):
            name = str(entry.get("name") or "State")
            path = entry.get("path")
            self.state_combo.addItem(name, str(path) if path else None)
            if entry.get("default"):
                default_index = index
        self.state_combo.setCurrentIndex(default_index)

    def selected_state_file(self) -> str | None:
        data = self.state_combo.currentData()
        return str(data) if data else None

    def _select_state_path(self, state_path: str) -> None:
        for index in range(self.state_combo.count()):
            if self.state_combo.itemData(index) == state_path:
                self.state_combo.setCurrentIndex(index)
                return

    def is_running(self) -> bool:
        return self._running

    def should_follow_target(self) -> bool:
        return self.follow_check.isChecked()

    def _synced_tilemap_id(self) -> int | None:
        """Return the last tilemap id that received a full WRAM sync, if any."""

        return None

    def _step_include_wram(self) -> bool:
        if not self._cfg.include_wram_when_stepping:
            return False
        if self._force_wram_next_step:
            return True
        interval = self._cfg.wram_sync_interval_frames
        if interval <= 0:
            return True
        snapshot = self._last_snapshot
        if snapshot is None:
            return True
        frame = snapshot_frame_counter(snapshot)
        if frame <= 0 or self._last_wram_sync_frame < 0:
            return True
        synced_tilemap = self._synced_tilemap_id()
        tilemap_id = snapshot_int(snapshot, "tilemapId")
        if (
            synced_tilemap is not None
            and tilemap_id is not None
            and tilemap_id != synced_tilemap
        ):
            return True
        return frame - self._last_wram_sync_frame >= interval

    def _step_include_frame(self) -> bool:
        speed = self.current_speed_multiplier()
        if speed < self._cfg.turbo_speed_threshold:
            return True
        if not self._cfg.skip_frame_when_turbo:
            return True
        interval = max(1, int(self._cfg.turbo_frame_preview_interval))
        self._turbo_step_counter += 1
        return self._turbo_step_counter % interval == 0

    def _step_delay_ms(self, *, repeat: int, frame_ms: float) -> int:
        speed = self.current_speed_multiplier()
        if speed >= self._cfg.unthrottled_speed_threshold:
            return 0
        target_tick_ms = self._cfg.base_frame_ms if repeat > 1 else self._target_frame_ms
        return max(0, target_tick_ms - int(frame_ms))

    def _after_step_snapshot(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("wramBase64") or snapshot.get("wramRaw"):
            self._last_wram_sync_frame = snapshot_frame_counter(snapshot)
            self._force_wram_next_step = False
            return
        if self._cfg.wram_sync_interval_frames <= 0:
            return
        synced_tilemap = self._synced_tilemap_id()
        tilemap_id = snapshot_int(snapshot, "tilemapId")
        if (
            synced_tilemap is not None
            and tilemap_id is not None
            and tilemap_id != synced_tilemap
        ):
            self._force_wram_next_step = True

    def set_room_label(self, room_label: str | None) -> None:
        label = str(room_label).strip() if room_label else None
        if label == self._room_label:
            return
        self._room_label = label
        self._rerender_last_snapshot()

    def _rerender_last_snapshot(self) -> None:
        if self._last_frame_pixmap is None or self._last_snapshot is None:
            return
        pixmap = self._last_frame_pixmap
        if self.overlay_check.isChecked():
            pixmap = self._with_hud_overlay(self._last_frame_pixmap, self._last_snapshot)
        self._set_frame_pixmap(pixmap)

    def _set_frame_pixmap(self, pixmap: QPixmap) -> None:
        target = self.frame_label.contentsRect().size()
        if target.width() > 0 and target.height() > 0:
            pixmap = pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self.frame_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rerender_last_snapshot()

    def start_bridge(self) -> None:
        if self._bridge.is_connected():
            return
        self._bridge.start()
        QCoreApplication.processEvents()
        self._send_command("hello", includeFrame=False)
        self._send_command("discover", includeFrame=False)
        self.status_label.setText("Bridge connected")

    def stop_bridge(self) -> None:
        self._cancel_step_loop()
        self._set_running(False)
        if self._bridge.is_connected():
            self._send_command("close_session", includeFrame=False)
        self._bridge.stop()
        self._last_snapshot = None
        self._last_frame_pixmap = None
        self._room_label = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.hot_save_button.setEnabled(False)
        self.load_save_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.ram_record_button.setEnabled(False)
        self.run_script_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.mark_button.setEnabled(False)
        self.mic_button.setEnabled(False)
        self.save_recording_button.setEnabled(False)
        self.run_last_recording_button.setEnabled(self._last_recording_path is not None)
        self.status_label.setText("Disconnected")

    def start_rom(self, rom_path: Path) -> None:
        self.start_bridge()
        state_file = self.selected_state_file()
        response = self._send_command(
            "start_session",
            romPath=str(rom_path),
            stateFile=state_file,
            includeFrame=True,
        )
        if response and response.get("ok"):
            self._set_running(True)
            self._schedule_next_step(0)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.hot_save_button.setEnabled(True)
            self.load_save_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.ram_record_button.setEnabled(True)
            self.run_script_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.mic_button.setEnabled(True)
            self.run_last_recording_button.setEnabled(self._last_recording_path is not None)
            self.frame_label.setFocus()
            self.status_label.setText(str(response.get("message") or "Running"))

    def start_default_session(self) -> None:
        self.start_bridge()
        state_file = self.selected_state_file()
        response = self._send_command(
            "start_session",
            stateFile=state_file,
            includeFrame=True,
        )
        if response and response.get("ok"):
            self._set_running(True)
            self._schedule_next_step(0)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.hot_save_button.setEnabled(True)
            self.load_save_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.ram_record_button.setEnabled(True)
            self.run_script_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.mic_button.setEnabled(True)
            self.run_last_recording_button.setEnabled(self._last_recording_path is not None)
            self.frame_label.setFocus()
            self.status_label.setText(str(response.get("message") or "Running"))

    def stop_session(self) -> None:
        self._cancel_step_loop()
        self._send_command("close_session", includeFrame=False)
        self._set_running(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.hot_save_button.setEnabled(False)
        self.load_save_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.ram_record_button.setEnabled(False)
        self.ram_record_button.setText("RAM Rec")
        self._ram_recording = False
        self.run_script_button.setEnabled(False)
        self._recording = False
        self._stop_mic_without_marker()
        self.record_button.setText("Record")
        self.record_button.setEnabled(False)
        self.mark_button.setEnabled(False)
        self.mic_button.setText("Mic Note")
        self.mic_button.setEnabled(False)
        self.save_recording_button.setEnabled(bool(self._recording_segments))
        self.run_last_recording_button.setEnabled(self._last_recording_path is not None)
        self._last_snapshot = None
        self._last_frame_pixmap = None
        self._last_frame_bytes = None
        self._room_label = None
        self._last_wram_sync_frame = -1
        self._force_wram_next_step = False
        self.frame_label.clear()
        self.frame_label.setText("No emulator session")
        self.fps_label.setText("FPS —")
        self._reset_speed()
        self.status_label.setText("Session closed")

    def current_speed_multiplier(self) -> float:
        return float(self._cfg.speed_levels[self._speed_index])

    def _reset_speed(self) -> None:
        self._speed_index = self._cfg.default_speed_index
        self._apply_speed_index()

    def _apply_speed_index(self) -> None:
        self._target_frame_ms = frame_budget_ms_for_speed(
            self._cfg.speed_levels[self._speed_index],
            base_frame_ms=self._cfg.base_frame_ms,
        )

    def decrease_speed(self) -> bool:
        if self._speed_index <= 0:
            return False
        self._speed_index -= 1
        self._apply_speed_index()
        self._notify_speed_change()
        return True

    def increase_speed(self) -> bool:
        if self._speed_index >= len(self._cfg.speed_levels) - 1:
            return False
        self._speed_index += 1
        self._apply_speed_index()
        self._notify_speed_change()
        return True

    def _notify_speed_change(self) -> None:
        self._turbo_step_counter = 0
        speed = self.current_speed_multiplier()
        label = f"{speed:g}x" if speed != int(speed) else f"{int(speed)}x"
        self.status_label.setText(f"Speed {label}  ([ slower  ] faster)")
        if self._last_frame_ms > 0:
            self._record_frame_timing(
                self._last_frame_ms,
                bridge_step_ms=self._last_bridge_step_ms or None,
            )

    def _cancel_step_loop(self) -> None:
        self._step_generation += 1
        self._step_in_flight = False
        self._pending_step_request_id = None
        self._step_timer.stop()

    def _send_command(self, command: str, **kwargs) -> dict[str, object] | None:
        if not self._bridge.is_connected():
            self._bridge.start()
            QCoreApplication.processEvents()
        response = self._bridge.call(command, **kwargs)
        if response is None:
            if not self._bridge.is_connected():
                self.status_label.setText("Bridge returned invalid response")
            return None
        self._handle_response(command, response)
        return response

    def _on_bridge_reply(self, payload: object) -> None:
        if not isinstance(payload, BridgeReply):
            return
        if payload.request_id != self._pending_step_request_id:
            return
        self._pending_step_request_id = None
        self._step_in_flight = False
        response = payload.response
        frame_ms = (time.perf_counter() - self._step_started) * 1000.0
        bridge_step_ms = None
        if isinstance(response, dict):
            try:
                bridge_step_ms = float(response.get("stepMs") or 0.0)
            except (TypeError, ValueError):
                bridge_step_ms = None
        repeat = self._last_step_repeat
        display_frame_ms = frame_ms / repeat if repeat > 1 else frame_ms
        self._record_frame_timing(display_frame_ms, bridge_step_ms=bridge_step_ms)
        self._handle_response("step", response or {})
        snapshot = response.get("snapshot") if isinstance(response, dict) else None
        if self._recording and isinstance(snapshot, dict):
            recorded_action = list(self._last_step_action)
            if isinstance(snapshot.get("logicalAction"), list):
                recorded_action = [int(value) for value in snapshot["logicalAction"]]
            self._append_recording_segment(self._recording_segments, recorded_action, repeat)
            self._recording_frames += repeat
            self.save_recording_button.setEnabled(True)
        if isinstance(snapshot, dict) and (snapshot.get("terminated") or snapshot.get("truncated")):
            self.stop_session()
            return
        if not self._running:
            return
        delay_ms = self._step_delay_ms(repeat=repeat, frame_ms=frame_ms)
        self._schedule_next_step(delay_ms)

    def _schedule_next_step(self, delay_ms: int) -> None:
        if not self._running:
            return
        self._step_timer.stop()
        self._step_timer.setInterval(max(0, int(delay_ms)))
        self._step_timer.start()

    def _record_frame_timing(
        self,
        frame_ms: float,
        *,
        bridge_step_ms: float | None = None,
    ) -> None:
        self._last_frame_ms = frame_ms
        if bridge_step_ms is not None:
            self._last_bridge_step_ms = bridge_step_ms
        if frame_ms > 0:
            self._fps_samples.append(1000.0 / frame_ms)
        avg_fps = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0.0
        bridge_text = (
            f"  bridge {self._last_bridge_step_ms:.0f}ms"
            if self._last_bridge_step_ms > 0
            else ""
        )
        speed = self.current_speed_multiplier()
        speed_label = f"{speed:g}x" if speed != int(speed) else f"{int(speed)}x"
        mode_bits: list[str] = [speed_label]
        if self._ram_recording:
            mode_bits.append("RAM rec")
        if self._recording:
            mode_bits.append("script rec")
        mode_text = f"  [{' | '.join(mode_bits)}]"
        self.fps_label.setText(
            f"FPS {avg_fps:4.1f}  frame {frame_ms:4.0f}ms{bridge_text}{mode_text}"
        )

    def _set_running(self, running: bool) -> None:
        if running == self._running:
            return
        self._running = running
        self.running_changed.emit(running)

    def _handle_response(self, command: str, response: dict[str, object]) -> None:
        if not response.get("ok", False):
            self.status_label.setText(f"Error: {response.get('error', '?')}")
            return
        if command == "discover":
            states = response.get("states")
            if isinstance(states, list):
                self._populate_state_combo([entry for entry in states if isinstance(entry, dict)])
        snapshot = response.get("snapshot")
        if isinstance(snapshot, dict):
            if snapshot.get("frameRgb24Raw") or snapshot.get("frameRgb24Base64"):
                self._render_frame(snapshot)
            else:
                self._update_snapshot_metadata(snapshot)
            self._after_step_snapshot(snapshot)
            self.snapshot_received.emit(snapshot)
            status_text = " | ".join(self._hud_lines(snapshot, self._room_label)[1:])
            controller_name = str(snapshot.get("controllerName") or "").strip()
            if controller_name:
                status_text = f"{status_text} | Pad {controller_name}"
            if command in {"hot_save", "load_hot_save"} and response.get("message"):
                status_text = f"{response.get('message')} | {status_text}"
            if command in {"capture", "run_script", "toggle_ram_recording"} and response.get("message"):
                status_text = f"{response.get('message')} | {status_text}"
            self.status_label.setText(status_text)
        elif response.get("message"):
            self.status_label.setText(str(response.get("message")))

    def _update_snapshot_metadata(self, snapshot: dict[str, object]) -> None:
        if self._last_snapshot is None:
            self._last_snapshot = snapshot_without_frame(snapshot)
            return
        for key in (
            "frameCounter",
            "tilemapId",
            "mapName",
            "playerX",
            "playerY",
            "playerTileX",
            "playerTileY",
            "toolName",
            "logicalAction",
            "autoplayEnabled",
            "autoplayMode",
            "autoplayGoal",
            "terminated",
            "truncated",
        ):
            if key in snapshot:
                self._last_snapshot[key] = snapshot[key]

    def hot_save(self) -> None:
        if not self._running:
            return
        response = self._send_command("hot_save", includeFrame=False)
        capture = response.get("capture") if isinstance(response, dict) else None
        state_path = None
        if isinstance(capture, dict) and isinstance(capture.get("paths"), dict):
            state_path = capture["paths"].get("state")
        if isinstance(state_path, str):
            self._populate_state_combo()
            self._select_state_path(state_path)

    def load_hot_save(self) -> None:
        if not self._running:
            return
        self._send_command("load_hot_save", includeFrame=True)

    def capture_session(self) -> None:
        if not self._running:
            return
        self._send_command("capture", prefix="editor_capture", includeFrame=True)

    def toggle_ram_recording(self) -> None:
        if not self._running:
            return
        label = safe_recording_slug(self.recording_name_edit.text() or "editor_ram")
        response = self._send_command(
            "toggle_ram_recording",
            label=label,
            includeFrame=False,
        )
        if not isinstance(response, dict) or not response.get("ok"):
            return
        self._ram_recording = bool(response.get("ramRecording"))
        self.ram_record_button.setText("Stop RAM" if self._ram_recording else "RAM Rec")
        message = str(response.get("message") or "")
        summary = response.get("ramRecordingSummary")
        if isinstance(summary, dict) and self._format_ram_tags is not None:
            tagged = self._format_ram_tags(summary.get("taggedChanges"))
            if tagged:
                message = f"{message} | {tagged}"
        if message:
            self.status_label.setText(message)

    def run_script_dialog(self) -> None:
        if not self._running:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Run Emulator Script",
            "",
            "JSON scripts (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            script, segments = self._script_segments_from_file(Path(path))
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
            return
        prefix = str(script.get("name") or Path(path).stem)
        state_file = str(script.get("stateFile")) if script.get("stateFile") else None
        self._run_script_segments(
            segments,
            prefix,
            bool(script.get("captureEachSegment", False)),
            state_file=state_file,
        )

    def _run_script_segments(
        self,
        segments: list[dict[str, object]],
        prefix: str,
        capture_each_segment: bool,
        *,
        state_file: str | None = None,
    ) -> None:
        if state_file:
            response = self._send_command("start_session", stateFile=state_file, includeFrame=True)
            if not isinstance(response, dict) or not response.get("ok"):
                return
        self._send_command(
            "run_script",
            segments=segments,
            prefix=prefix,
            captureEachSegment=capture_each_segment,
            includeFrame=True,
        )

    def toggle_recording(self) -> None:
        if not self._running:
            return
        if self._recording:
            self._recording = False
            self.record_button.setText("Record")
            self.mark_button.setEnabled(False)
            self.save_recording_button.setEnabled(bool(self._recording_segments))
            self.status_label.setText(f"Recording stopped: {self._recording_frames} frames")
            return
        self._recording_name = safe_recording_slug(
            self.recording_name_edit.text() or "editor_recording"
        )
        self._recording_segments = []
        self._recording_markers = []
        self._recording_frames = 0
        self._recording_start_capture = None
        self._recording_selected_state_file = self.selected_state_file()
        response = self._send_command(
            "capture",
            prefix=f"{self._recording_name}_record_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            includeFrame=False,
        )
        if not isinstance(response, dict) or not response.get("ok") or not isinstance(response.get("capture"), dict):
            self.status_label.setText("Recording start checkpoint failed")
            return
        self._recording_start_capture = response["capture"]
        self._recording = True
        self.record_button.setText("Stop Rec")
        self.mark_button.setEnabled(True)
        self.save_recording_button.setEnabled(False)
        self.status_label.setText(f"Recording {self._recording_name} from checkpoint")

    def mark_recording(self) -> None:
        if not self._running:
            return
        label = safe_recording_slug(
            self.recording_name_edit.text() or f"mark_{len(self._recording_markers) + 1:02d}"
        )
        marker_segment = self._append_recording_marker(self._recording_segments, label)
        marker: dict[str, object] = {
            "frame": self._recording_frames,
            "label": label,
            "segmentIndex": len(self._recording_segments) - 1,
        }
        if self._last_snapshot is not None:
            marker["snapshot"] = self._compact_snapshot(self._last_snapshot)
        prefix = f"{self._recording_name or 'editor_recording'}_{label}_{self._recording_frames:05d}"
        response = self._send_command("capture", prefix=prefix, includeFrame=False)
        if isinstance(response, dict) and isinstance(response.get("capture"), dict):
            marker["capture"] = response["capture"]
            marker_segment["capturePrefix"] = response["capture"].get("prefix")
        self._recording_markers.append(marker)
        self.save_recording_button.setEnabled(True)
        self.status_label.setText(f"Marked {label} at frame {self._recording_frames}")

    def toggle_mic_annotation(self) -> None:
        if not self._running:
            return
        if self._mic_audio is not None:
            self.stop_mic_annotation()
            return
        if not self._recording:
            self.toggle_recording()
            if not self._recording:
                return
        if QAudioFormat is None or QAudioSource is None or QMediaDevices is None:
            self.status_label.setText("Qt multimedia audio input is unavailable")
            return
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.status_label.setText("No microphone input device found")
            return
        audio_format = QAudioFormat()
        audio_format.setSampleRate(16000)
        audio_format.setChannelCount(1)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(audio_format):
            audio_format = device.preferredFormat()
        self._mic_buffer = QBuffer(self)
        self._mic_buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self._mic_audio = QAudioSource(device, audio_format, self)
        self._mic_audio.start(self._mic_buffer)
        self._mic_format = audio_format
        self._mic_started_at_frame = self._recording_frames
        self.mic_button.setText("Stop Mic")
        self.status_label.setText("Mic annotation recording")

    def stop_mic_annotation(self) -> None:
        if self._mic_audio is None or self._mic_buffer is None or self._mic_format is None:
            self._stop_mic_without_marker()
            return
        self._mic_audio.stop()
        self._mic_buffer.close()
        raw = bytes(self._mic_buffer.data())
        audio_format = self._mic_format
        self._stop_mic_without_marker()
        if not raw:
            self.status_label.setText("Mic annotation was empty")
            return
        label = safe_recording_slug(
            self.recording_name_edit.text() or f"voice_{len(self._recording_markers) + 1:02d}"
        )
        audio_dir = self._cfg.default_recording_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.wav"
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(int(audio_format.channelCount()))
            handle.setsampwidth(max(1, int(audio_format.bytesPerSample())))
            handle.setframerate(int(audio_format.sampleRate()))
            handle.writeframes(raw)
        transcript = self._transcribe_audio(audio_path)
        marker_segment = self._append_recording_marker(self._recording_segments, label)
        marker: dict[str, object] = {
            "frame": self._recording_frames,
            "startFrame": self._mic_started_at_frame,
            "label": label,
            "segmentIndex": len(self._recording_segments) - 1,
            "audio": str(audio_path),
            "transcription": transcript,
        }
        if transcript.get("text"):
            marker_segment["voiceText"] = transcript["text"]
        if self._last_snapshot is not None:
            marker["snapshot"] = self._compact_snapshot(self._last_snapshot)
        response = self._send_command(
            "capture",
            prefix=f"{self._recording_name or 'editor_recording'}_{label}_voice",
            includeFrame=False,
        )
        if isinstance(response, dict) and isinstance(response.get("capture"), dict):
            marker["capture"] = response["capture"]
            marker_segment["capturePrefix"] = response["capture"].get("prefix")
        self._recording_markers.append(marker)
        self.save_recording_button.setEnabled(True)
        summary = transcript.get("text") or transcript.get("status") or "saved"
        self.status_label.setText(f"Voice note {label}: {summary}")

    def _stop_mic_without_marker(self) -> None:
        if self._mic_audio is not None:
            try:
                self._mic_audio.stop()
            except Exception:
                pass
        if self._mic_buffer is not None and self._mic_buffer.isOpen():
            self._mic_buffer.close()
        self._mic_audio = None
        self._mic_buffer = None
        self._mic_format = None
        self.mic_button.setText("Mic Note")

    def _transcribe_audio(self, audio_path: Path) -> dict[str, object]:
        command = os.environ.get(self._env("TRANSCRIBE_CMD"), "").strip()
        if command:
            try:
                result = subprocess.run(
                    [*shlex.split(command), str(audio_path)],
                    cwd=str(self._cfg.project_root),
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
            except Exception as exc:
                return {"status": "error", "text": "", "error": str(exc), "command": command}
            text = result.stdout.strip()
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "text": text,
                "command": command,
                "returnCode": result.returncode,
                "stderr": result.stderr.strip(),
            }

        whisper = shutil.which("whisper")
        if whisper is None:
            prefix = self._cfg.env_prefix
            return {
                "status": "not_configured",
                "text": "",
                "hint": (
                    f"Install the whisper CLI or set {prefix}_TRANSCRIBE_CMD to a command "
                    "that accepts the WAV path."
                ),
            }
        output_dir = self._cfg.default_recording_dir / "transcripts"
        output_dir.mkdir(parents=True, exist_ok=True)
        device = os.environ.get(self._env("WHISPER_DEVICE")) or (
            "cuda" if shutil.which("nvidia-smi") else "cpu"
        )
        command_args = [
            whisper,
            str(audio_path),
            "--model",
            os.environ.get(self._env("WHISPER_MODEL"), "turbo"),
            "--device",
            device,
            "--language",
            os.environ.get(self._env("WHISPER_LANGUAGE"), "en"),
            "--output_format",
            "json",
            "--output_dir",
            str(output_dir),
            "--verbose",
            "False",
        ]
        try:
            result = subprocess.run(
                command_args,
                cwd=str(self._cfg.project_root),
                text=True,
                capture_output=True,
                timeout=int(os.environ.get(self._env("WHISPER_TIMEOUT"), "300")),
                check=False,
            )
        except Exception as exc:
            return {"status": "error", "text": "", "error": str(exc), "engine": "whisper"}
        transcript_path = output_dir / f"{audio_path.stem}.json"
        transcript: dict[str, object] = {}
        if transcript_path.is_file():
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            except Exception:
                transcript = {}
        text = str(transcript.get("text") or "").strip()
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "text": text,
            "segments": transcript.get("segments", []),
            "engine": "whisper",
            "device": device,
            "command": " ".join(shlex.quote(arg) for arg in command_args),
            "returnCode": result.returncode,
            "stderr": result.stderr.strip(),
            "transcriptPath": str(transcript_path) if transcript_path.is_file() else None,
        }

    def save_recording(self) -> None:
        if self._recording:
            self.toggle_recording()
        if not self._recording_segments:
            self.status_label.setText("No recording segments to save")
            return
        self._cfg.default_recording_dir.mkdir(parents=True, exist_ok=True)
        name = self._recording_name or safe_recording_slug(
            self.recording_name_edit.text() or "editor_recording"
        )
        path = self._cfg.default_recording_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.json"
        start_state_file = None
        if isinstance(self._recording_start_capture, dict):
            paths = self._recording_start_capture.get("paths")
            if isinstance(paths, dict) and isinstance(paths.get("state"), str):
                start_state_file = paths["state"]
        bridge_module = self._cfg.headless_bridge_module or self._cfg.bridge_module
        data = {
            "format": self._cfg.recording_format,
            "version": self._cfg.recording_version,
            "tool": self._cfg.recording_tool,
            "name": name,
            "recordedAt": datetime.now().isoformat(timespec="seconds"),
            "buttonOrder": list(self._cfg.button_order),
            "stateFile": start_state_file or self._recording_selected_state_file or self.selected_state_file(),
            "selectedStateFile": self._recording_selected_state_file,
            "startCapture": self._recording_start_capture,
            "roomLabel": self._room_label,
            "totalFrames": self._recording_frames,
            "segments": self._recording_segments,
            "markers": self._recording_markers,
            "lastSnapshot": self._compact_snapshot(self._last_snapshot) if self._last_snapshot else None,
            "captureEachSegment": False,
            "aiUse": {
                "description": f"Replay with {bridge_module} run_script to reproduce labeled emulator states.",
                "headlessCommand": (
                    f"PYTHONPATH=. uv run --project .. python -m {bridge_module} --script {path}"
                ),
            },
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._last_recording_path = path
        self.run_last_recording_button.setEnabled(True)
        self.status_label.setText(f"Saved recording {path.name}")

    def run_last_recording(self) -> None:
        if not self._running or self._last_recording_path is None:
            return
        try:
            script, segments = self._script_segments_from_file(self._last_recording_path)
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
            return
        prefix = str(script.get("name") or self._last_recording_path.stem)
        state_file = str(script.get("stateFile")) if script.get("stateFile") else None
        self._run_script_segments(segments, prefix, False, state_file=state_file)

    def _compact_snapshot(self, snapshot: dict[str, object]) -> dict[str, object]:
        keys = self._cfg.compact_snapshot_keys
        if not keys:
            return dict(snapshot)
        return {key: snapshot.get(key) for key in keys if key in snapshot}

    def _render_frame(self, snapshot: dict[str, object]) -> None:
        raw = snapshot.get("frameRgb24Raw")
        if raw is None:
            frame_data = snapshot.get("frameRgb24Base64")
            if not frame_data:
                return
            raw = base64.b64decode(str(frame_data))
        else:
            raw = bytes(raw)
        self._last_snapshot = snapshot_without_frame(snapshot)
        width = int(snapshot.get("frameWidth") or 256)
        height = int(snapshot.get("frameHeight") or 224)
        expected = width * height * 3
        if len(raw) != expected:
            return
        self._last_frame_bytes = raw
        image = QImage(
            self._last_frame_bytes,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        base_pixmap = QPixmap.fromImage(image)
        self._last_frame_pixmap = base_pixmap
        pixmap = base_pixmap
        if self.overlay_check.isChecked():
            pixmap = self._with_hud_overlay(base_pixmap, self._last_snapshot)
        self._set_frame_pixmap(pixmap)

    def _with_hud_overlay(self, pixmap: QPixmap, snapshot: dict[str, object]) -> QPixmap:
        lines = self._hud_lines(snapshot, self._room_label)
        if not lines:
            return pixmap
        canvas = QPixmap(pixmap)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        font = QFont("monospace", max(8, min(11, canvas.height() // 24)))
        font.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding = 6
        line_height = metrics.height() + 1
        overlay_width = min(
            max(1, canvas.width() - 8),
            max(metrics.horizontalAdvance(line) for line in lines) + padding * 2,
        )
        overlay_height = line_height * len(lines) + padding * 2
        painter.fillRect(4, 4, overlay_width, overlay_height, QColor(0, 0, 0, 172))
        painter.setPen(QPen(QColor(255, 255, 255, 235), 1))
        y = 4 + padding + metrics.ascent()
        text_width = max(1, overlay_width - padding * 2)
        for line in lines:
            painter.drawText(
                4 + padding,
                y,
                metrics.elidedText(line, Qt.TextElideMode.ElideRight, text_width),
            )
            y += line_height
        painter.end()
        return canvas

    def _step_tick(self) -> None:
        if not self._running or self._step_in_flight:
            return
        action = [0] * len(self._cfg.button_order)
        for key, button_index in self._cfg.key_to_button.items():
            if key in self._keys_pressed:
                action[button_index] = 1
        speed = self.current_speed_multiplier()
        repeat = 1
        if self._cfg.speed_uses_frame_repeat and speed >= 2.0:
            repeat = max(1, int(round(speed)))
        self._last_step_repeat = repeat
        self._last_step_action = action
        self._step_started = time.perf_counter()
        self._step_in_flight = True
        self._pending_step_request_id = self._bridge.post(
            "step",
            action=action,
            repeat=repeat,
            includeFrame=self._step_include_frame(),
            includeWram=self._step_include_wram(),
        )

    def handle_key_press(self, key: int) -> bool:
        if key not in self._cfg.key_to_button:
            return False
        self._keys_pressed.add(key)
        return True

    def handle_key_release(self, key: int) -> bool:
        if key not in self._cfg.key_to_button:
            return False
        self._keys_pressed.discard(key)
        return True


__all__ = ["EmbeddedEmulatorPanelBase", "EmulatorPanelConfig"]
