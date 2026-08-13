"""Generic Qt embedded emulator panel for editor ↔ bridge workflows."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer, Signal, QCoreApplication
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from retro_harness.editor.bridge_worker import BridgeController, BridgeReply
from retro_harness.editor.emulator_loop import (
    EmulatorSpeedController,
    FrameTimingTracker,
    after_step_wram_flags,
    should_include_wram,
)
from retro_harness.editor.gui_emulator_recording import EmulatorRecordingMixin
from retro_harness.editor.snapshot import snapshot_frame_counter, snapshot_int, snapshot_without_frame

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


class EmbeddedEmulatorPanelBase(EmulatorRecordingMixin, QWidget):
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
        self._speed = EmulatorSpeedController(
            levels=self._cfg.speed_levels,
            default_index=self._cfg.default_speed_index,
            base_frame_ms=self._cfg.base_frame_ms,
            speed_uses_frame_repeat=self._cfg.speed_uses_frame_repeat,
            skip_frame_when_turbo=self._cfg.skip_frame_when_turbo,
            turbo_speed_threshold=self._cfg.turbo_speed_threshold,
            turbo_frame_preview_interval=self._cfg.turbo_frame_preview_interval,
            unthrottled_speed_threshold=self._cfg.unthrottled_speed_threshold,
        )
        self._timing = FrameTimingTracker()
        self._keys_pressed: set[int] = set()
        self._room_label: str | None = None
        self._last_snapshot: dict[str, object] | None = None
        self._last_frame_pixmap: QPixmap | None = None
        self._last_frame_bytes: bytes | None = None
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
        self._mic_buffer = None
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
        self.autoplay_button = QPushButton("Autoplay")
        self.autoplay_button.setEnabled(False)
        automation_row.addWidget(self.autoplay_button)
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

        follow_row = QHBoxLayout()
        self.follow_check = QCheckBox(self._cfg.follow_checkbox_label)
        self.follow_check.setChecked(True)
        follow_row.addWidget(self.follow_check)
        self.overlay_check = QCheckBox("HUD overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(lambda _checked: self._rerender_last_snapshot())
        follow_row.addWidget(self.overlay_check)
        follow_row.addStretch(1)
        layout.addLayout(follow_row)

        self.fps_label = QLabel("FPS —")
        layout.addWidget(self.fps_label)
        self.status_label = QLabel("Disconnected")
        layout.addWidget(self.status_label)

        self._populate_state_combo()
        QTimer.singleShot(0, self._warm_bridge)

    def _env(self, suffix: str) -> str:
        return f"{self._cfg.env_prefix}_{suffix}"

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
        snapshot = self._last_snapshot
        frame = snapshot_frame_counter(snapshot) if snapshot is not None else 0
        tilemap_id = snapshot_int(snapshot, "tilemapId") if snapshot is not None else None
        return should_include_wram(
            include_wram_when_stepping=self._cfg.include_wram_when_stepping,
            force_wram_next_step=self._force_wram_next_step,
            wram_sync_interval_frames=self._cfg.wram_sync_interval_frames,
            frame=frame,
            last_wram_sync_frame=self._last_wram_sync_frame,
            synced_tilemap=self._synced_tilemap_id(),
            tilemap_id=tilemap_id,
        )

    def _step_include_frame(self) -> bool:
        return self._speed.should_include_frame()

    def _step_delay_ms(self, *, repeat: int, frame_ms: float) -> int:
        return self._speed.delay_ms(repeat=repeat, frame_ms=frame_ms)

    def _after_step_snapshot(self, snapshot: dict[str, object]) -> None:
        new_sync, force_next = after_step_wram_flags(
            snapshot,
            wram_sync_interval_frames=self._cfg.wram_sync_interval_frames,
            synced_tilemap=self._synced_tilemap_id(),
            tilemap_id=snapshot_int(snapshot, "tilemapId"),
            frame=snapshot_frame_counter(snapshot),
        )
        if new_sync is not None:
            self._last_wram_sync_frame = new_sync
            self._force_wram_next_step = force_next
            return
        if force_next:
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

    def _enable_session_controls(self) -> None:
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
            self._enable_session_controls()
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
            self._enable_session_controls()
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
        return self._speed.multiplier

    def _reset_speed(self) -> None:
        self._speed.reset()

    def decrease_speed(self) -> bool:
        if not self._speed.decrease():
            return False
        self._notify_speed_change()
        return True

    def increase_speed(self) -> bool:
        if not self._speed.increase():
            return False
        self._notify_speed_change()
        return True

    def _notify_speed_change(self) -> None:
        self.status_label.setText(
            f"Speed {self._speed.label()}  ([ slower  ] faster)"
        )
        if self._timing.last_frame_ms > 0:
            self._record_frame_timing(
                self._timing.last_frame_ms,
                bridge_step_ms=self._timing.last_bridge_step_ms or None,
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
        if bridge_step_ms is not None:
            self._timing.last_bridge_step_ms = bridge_step_ms
        self.fps_label.setText(
            self._timing.status_text(
                frame_ms=frame_ms,
                speed=self.current_speed_multiplier(),
                ram_recording=self._ram_recording,
                script_recording=self._recording,
            )
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
        repeat = self._speed.step_repeat()
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
