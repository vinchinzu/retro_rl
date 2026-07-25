"""Game-agnostic emulator session runtime for editor bridge subprocesses."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from retro_harness.controls import action_from_snes_button_names, sanitize_action
from retro_harness.env import read_state_bytes
from retro_harness.editor.script_segments import script_segments_from_payload
from retro_harness.editor.util import safe_recording_slug
from retro_harness.runtime import reset_env, step_env


ReadWramFn = Callable[[object], bytes]
BuildSnapshotFn = Callable[[object, object, dict[str, object], int, list[int]], dict[str, object]]
MakeEnvFn = Callable[[str | None], object]


def decompress_state_bytes(path: Path) -> bytes:
    """Compatibility alias for the shared state reader."""

    return read_state_bytes(path)


@dataclass
class EditorBridgeRuntime:
    """Manage a single stable-retro session for editor bridge commands."""

    project_root: Path
    states_dir: Path
    capture_dir: Path
    hot_save_path: Path
    button_order: tuple[str, ...]
    make_env: MakeEnvFn
    read_wram: ReadWramFn
    build_snapshot: BuildSnapshotFn
    idle_label: str = "IDLE"
    _env: object | None = field(default=None, init=False)
    _frame_counter: int = field(default=0, init=False)
    _last_action: list[int] = field(default_factory=lambda: [0] * 12, init=False)
    _controller: object | None = field(default=None, init=False)
    _controller_name: str | None = field(default=None, init=False)
    _controller_scan_interval_frames: int = field(default=120, init=False)
    _last_controller_scan_frame: int = field(default=-120, init=False)
    _ram_recording: bool = field(default=False, init=False)
    _ram_baseline: bytes | None = field(default=None, init=False)
    _ram_changes: dict[int, list[int]] = field(default_factory=dict, init=False)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None
        self._frame_counter = 0
        self._ram_recording = False
        self._ram_baseline = None
        self._ram_changes = {}

    @property
    def env(self) -> object | None:
        return self._env

    def set_controller(self, controller: object | None) -> None:
        self._controller = controller

    def set_controller_name(self, name: str | None) -> None:
        self._controller_name = name

    def _scan_controller(self) -> object | None:
        try:
            import pygame

            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                self._controller = None
                self._controller_name = None
                return None
            controller = pygame.joystick.Joystick(0)
            controller.init()
            self._controller = controller
            try:
                self._controller_name = controller.get_name()
            except Exception:
                self._controller_name = None
            return controller
        except Exception:
            self._controller = None
            self._controller_name = None
            return None

    def _merge_controller_input(self, action: list[int]) -> None:
        controller = self._controller
        should_scan = (
            self._frame_counter - self._last_controller_scan_frame
            >= self._controller_scan_interval_frames
        )
        if should_scan:
            self._last_controller_scan_frame = self._frame_counter
            controller = self._scan_controller()
        if controller is None:
            return
        try:
            import pygame
            from retro_harness.controls import controller_action

            pygame.event.pump()
            controller_action(controller, action)
            sanitize_action(action)
        except Exception:
            self._controller = None
            self._controller_name = None

    def discover_states(self, *, default_state: Path | None = None) -> list[dict[str, object]]:
        states: list[dict[str, object]] = [
            {"name": "Reset", "path": "NONE", "default": default_state is None},
        ]
        if default_state is not None and default_state.is_file():
            states.append(
                {
                    "name": "Latest original",
                    "path": str(default_state),
                    "default": True,
                }
            )
        if self.hot_save_path.is_file():
            states.append(
                {
                    "name": "Editor hot save",
                    "path": str(self.hot_save_path),
                    "default": False,
                }
            )
        if self.states_dir.is_dir():
            for path in sorted(self.states_dir.glob("*.state"), key=lambda item: item.stem.casefold()):
                states.append({"name": path.stem, "path": str(path), "default": False})
        return states

    def _resolve_state_path(self, state_file: str | None) -> Path | None:
        if not state_file or state_file == "NONE":
            return None
        path = Path(state_file)
        if path.is_file():
            return path
        candidate = self.states_dir / f"{state_file}.state"
        if candidate.is_file():
            return candidate
        return None

    def start_session(
        self,
        *,
        state_file: str | None = None,
        rom_path: str | None = None,
    ) -> tuple[dict[str, object], bytes | None]:
        del rom_path  # reserved for ROM-first editors
        self.close()
        state_path = self._resolve_state_path(state_file)
        if state_path is None and state_file not in (None, "", "NONE"):
            stem = Path(state_file).stem
            self._env = self.make_env(stem if stem else None)
        elif state_path is None:
            self._env = self.make_env(None)
        else:
            self._env = self.make_env(None)
        _obs, _info = reset_env(self._env)
        if state_path is not None:
            raw_state = decompress_state_bytes(state_path)
            self._env.em.set_state(raw_state)
        self._frame_counter = 0
        self._last_action = [0] * len(self.button_order)
        snapshot, frame_rgb = self._snapshot(include_frame=True)
        label = state_path.stem if state_path is not None else "Reset"
        snapshot["message"] = f"Started session: {label}"
        return snapshot, frame_rgb

    def step(
        self,
        *,
        action: list[int] | None = None,
        repeat: int = 1,
        include_frame: bool = True,
        include_wram: bool = True,
    ) -> tuple[dict[str, object], bytes | None, float]:
        if self._env is None:
            raise RuntimeError("No active session")
        started = time.perf_counter()
        logical_action = [int(value) for value in (action or self._last_action)]
        self._merge_controller_input(logical_action)
        sanitize_action(logical_action)
        self._last_action = logical_action
        terminated = False
        truncated = False
        obs: object = None
        info: dict[str, object] = {}
        for _ in range(max(1, int(repeat))):
            obs, _reward, terminated, truncated, info = step_env(
                self._env,
                np.asarray(logical_action, dtype=np.int32),
            )
            self._frame_counter += 1
            if self._ram_recording:
                self._track_ram_changes()
            if terminated or truncated:
                break
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        snapshot, frame_rgb = self._snapshot(
            obs=obs,
            info=info,
            include_frame=include_frame,
            include_wram=include_wram,
            logical_action=logical_action,
            terminated=terminated,
            truncated=truncated,
        )
        snapshot["stepMs"] = elapsed_ms
        return snapshot, frame_rgb if include_frame else None, elapsed_ms

    def hot_save(self) -> dict[str, object]:
        if self._env is None:
            raise RuntimeError("No active session")
        self.hot_save_path.parent.mkdir(parents=True, exist_ok=True)
        self._env.em.save_state(str(self.hot_save_path))
        capture = self.capture(prefix="editor_hot_save", include_frame=False)
        capture["message"] = f"Hot saved to {self.hot_save_path.name}"
        return capture

    def load_hot_save(self) -> tuple[dict[str, object], bytes | None]:
        if self._env is None:
            raise RuntimeError("No active session")
        if not self.hot_save_path.is_file():
            raise FileNotFoundError(f"Hot save not found: {self.hot_save_path}")
        raw_state = decompress_state_bytes(self.hot_save_path)
        self._env.em.set_state(raw_state)
        self._frame_counter = 0
        snapshot, frame_rgb = self._snapshot(include_frame=True)
        snapshot["message"] = f"Loaded hot save {self.hot_save_path.name}"
        return snapshot, frame_rgb

    def capture(self, *, prefix: str, include_frame: bool) -> dict[str, object]:
        if self._env is None:
            raise RuntimeError("No active session")
        safe_prefix = safe_recording_slug(prefix, default="editor_capture")
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.capture_dir / f"{safe_prefix}.state"
        self._env.em.save_state(str(state_path))
        frame_rgb: bytes | None = None
        if include_frame:
            obs = self._env.render()
            if isinstance(obs, np.ndarray):
                frame_rgb = obs.astype(np.uint8, copy=False).tobytes()
        return {
            "prefix": safe_prefix,
            "paths": {
                "state": str(state_path),
                "png": None,
            },
            "frameCounter": self._frame_counter,
        }

    def toggle_ram_recording(self, *, label: str) -> dict[str, object]:
        if self._env is None:
            raise RuntimeError("No active session")
        if not self._ram_recording:
            self._ram_recording = True
            self._ram_baseline = self.read_wram(self._env)
            self._ram_changes = {}
            return {
                "ramRecording": True,
                "message": f"RAM recording started ({label})",
            }
        summary = self._finish_ram_recording(label=label)
        return {
            "ramRecording": False,
            "message": f"RAM recording saved ({label})",
            "ramRecordingSummary": summary,
        }

    def run_script(
        self,
        *,
        segments: list[dict[str, object]] | object,
        prefix: str,
        capture_each_segment: bool = False,
        include_frame: bool = True,
    ) -> tuple[dict[str, object], bytes | None]:
        if self._env is None:
            raise RuntimeError("No active session")
        normalized = script_segments_from_payload(segments, idle_label=self.idle_label)
        safe_prefix = safe_recording_slug(prefix, default="editor_script")
        captures: list[dict[str, object]] = []
        for index, segment in enumerate(normalized):
            labels = segment.get("buttons", [self.idle_label])
            if not isinstance(labels, list):
                labels = [str(labels)]
            action = action_from_snes_button_names(
                [str(item) for item in labels],
                action_size=len(self._last_action),
            )
            _snapshot, _frame, _elapsed = self.step(
                action=action,
                repeat=int(segment.get("frames", 1)),
                include_frame=False,
            )
            if capture_each_segment or segment.get("label"):
                marker = str(segment.get("label") or f"segment_{index:03d}")
                captures.append(
                    self.capture(prefix=f"{safe_prefix}_{marker}", include_frame=False)
                )
        snapshot, frame_rgb = self._snapshot(include_frame=include_frame)
        snapshot["message"] = f"Ran script {safe_prefix} ({len(normalized)} segments)"
        snapshot["scriptCaptures"] = captures
        return snapshot, frame_rgb if include_frame else None

    def _snapshot(
        self,
        *,
        obs: object | None = None,
        info: dict[str, object] | None = None,
        include_frame: bool = True,
        include_wram: bool = True,
        logical_action: list[int] | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> tuple[dict[str, object], bytes | None]:
        if self._env is None:
            raise RuntimeError("No active session")
        if obs is None:
            obs = self._env.render()
        frame_rgb: bytes | None = None
        if include_frame and isinstance(obs, np.ndarray):
            frame_rgb = obs.astype(np.uint8, copy=False).tobytes()
        snapshot = self.build_snapshot(
            self._env,
            obs,
            info or {},
            self._frame_counter,
            logical_action or self._last_action,
        )
        snapshot["frameCounter"] = self._frame_counter
        snapshot["logicalAction"] = list(logical_action or self._last_action)
        snapshot["terminated"] = terminated
        snapshot["truncated"] = truncated
        if self._controller_name:
            snapshot["controllerName"] = self._controller_name
        if isinstance(obs, np.ndarray):
            snapshot["frameWidth"] = int(obs.shape[1])
            snapshot["frameHeight"] = int(obs.shape[0])
        if include_wram:
            snapshot["wramRaw"] = self.read_wram(self._env)
        return snapshot, frame_rgb

    def _track_ram_changes(self) -> None:
        if self._env is None or self._ram_baseline is None:
            return
        current = self.read_wram(self._env)
        limit = min(len(current), len(self._ram_baseline))
        for offset in range(limit):
            if current[offset] == self._ram_baseline[offset]:
                continue
            values = self._ram_changes.setdefault(offset, [])
            if not values or values[-1] != current[offset]:
                values.append(current[offset])

    def _finish_ram_recording(self, *, label: str) -> dict[str, object]:
        self._ram_recording = False
        tagged = [
            {"offset": offset, "values": values}
            for offset, values in sorted(self._ram_changes.items())
        ]
        summary = {
            "label": label,
            "changedCount": len(tagged),
            "taggedChanges": tagged,
        }
        self._ram_baseline = None
        self._ram_changes = {}
        return summary
