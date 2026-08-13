"""Recording / mic-annotation mixin for the embedded Qt emulator panel."""

from __future__ import annotations

import json
import wave
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtWidgets import QFileDialog

try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
except Exception:  # pragma: no cover - depends on host Qt multimedia install
    QAudioFormat = None  # type: ignore[assignment]
    QAudioSource = None  # type: ignore[assignment]
    QMediaDevices = None  # type: ignore[assignment]

from retro_harness.editor.emulator_loop import (
    build_script_recording_document,
    compact_snapshot,
)
from retro_harness.editor.recording import (
    append_recording_marker as _core_append_recording_marker,
    append_recording_segment as _core_append_recording_segment,
    safe_recording_slug,
)
from retro_harness.editor.transcribe import transcribe_audio_file


class EmulatorRecordingMixin:
    """Script recording, markers, mic notes, and save/replay helpers.

    Expects the host widget to provide: ``_cfg``, ``_running``, ``_send_command``,
    ``status_label``, recording UI widgets, and ``_script_segments_from_file``.
    """

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
        if (
            not isinstance(response, dict)
            or not response.get("ok")
            or not isinstance(response.get("capture"), dict)
        ):
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
        return transcribe_audio_file(
            audio_path,
            project_root=self._cfg.project_root,
            env_prefix=self._cfg.env_prefix,
            default_recording_dir=self._cfg.default_recording_dir,
        )

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
        path = (
            self._cfg.default_recording_dir
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.json"
        )
        start_state_file = None
        if isinstance(self._recording_start_capture, dict):
            paths = self._recording_start_capture.get("paths")
            if isinstance(paths, dict) and isinstance(paths.get("state"), str):
                start_state_file = paths["state"]
        bridge_module = self._cfg.headless_bridge_module or self._cfg.bridge_module
        data = build_script_recording_document(
            name=name,
            button_order=self._cfg.button_order,
            state_file=start_state_file
            or self._recording_selected_state_file
            or self.selected_state_file(),
            selected_state_file=self._recording_selected_state_file,
            start_capture=self._recording_start_capture,
            room_label=self._room_label,
            total_frames=self._recording_frames,
            segments=self._recording_segments,
            markers=self._recording_markers,
            last_snapshot=(
                self._compact_snapshot(self._last_snapshot) if self._last_snapshot else None
            ),
            recording_format=self._cfg.recording_format,
            recording_version=self._cfg.recording_version,
            recording_tool=self._cfg.recording_tool,
            bridge_module=bridge_module,
            recording_path=path,
        )
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
        return compact_snapshot(snapshot, self._cfg.compact_snapshot_keys)


__all__ = ["EmulatorRecordingMixin"]
