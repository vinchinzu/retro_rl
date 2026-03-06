#!/usr/bin/env python3
"""Small stdio bridge between SMEDIT and the Python stable-retro runtime.

This is intentionally narrow:
- discover published save states and PPO checkpoints
- start/close a single live emulator session
- snapshot or step the session while returning raw RGB frames
- save/load states inside custom_integrations
- record a light JSONL trace for editor sessions

The bridge speaks one JSON object per line on stdin/stdout.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

GAME = "SuperMetroid-Snes"
TRACE_LIMIT = 512
NOOP = [0] * 12
DEFAULT_NAV_EXPORT_DIR = Path("/tmp/sm_export")


def _json_response(
    *,
    request_id: str | None,
    ok: bool = True,
    error: str | None = None,
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": request_id, "ok": ok}
    if error:
        payload["error"] = error
    if message:
        payload["message"] = message
    payload.update(extra)
    return payload


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _published_states() -> list[dict[str, str]]:
    states_dir = PROJECT_DIR / "custom_integrations" / GAME
    return sorted(
        [
            {"name": path.stem, "path": str(path)}
            for path in states_dir.glob("*.state")
        ],
        key=lambda entry: entry["name"].lower(),
    )


def _published_models() -> list[dict[str, str]]:
    models_dir = PROJECT_DIR / "models"
    models: list[dict[str, str]] = []
    if not models_dir.exists():
        return models
    for suffix, fmt in (("*.zip", "sb3_zip"), ("*.pth", "torch_pth")):
        for path in models_dir.glob(suffix):
            models.append({"name": path.name, "path": str(path), "format": fmt})
    return sorted(models, key=lambda entry: entry["name"].lower())


def _room_lookup(nav_export_dir: Path | None) -> dict[int, tuple[str, str]]:
    if nav_export_dir is None:
        return {}
    nav_graph = nav_export_dir / "nav_graph.json"
    if not nav_graph.exists():
        return {}
    try:
        data = json.loads(nav_graph.read_text())
    except Exception:
        return {}
    result: dict[int, tuple[str, str]] = {}
    for node in data.get("nodes", []):
        room_id = int(node.get("roomId", 0))
        result[room_id] = (str(node.get("name", f"0x{room_id:04X}")), str(node.get("areaName", "?")))
    return result


def _room_name(room_lookup: dict[int, tuple[str, str]], room_id: int | None) -> str | None:
    if room_id is None:
        return None
    return room_lookup.get(room_id, (f"0x{room_id:04X}", "?"))[0]


def _area_name(room_lookup: dict[int, tuple[str, str]], room_id: int | None) -> str | None:
    if room_id is None:
        return None
    return room_lookup.get(room_id, (f"0x{room_id:04X}", "?"))[1]


def _runtime_imports():
    import numpy as np
    from retro_harness.env import make_env, save_state

    return np, make_env, save_state


@dataclass
class Session:
    env: Any
    current_state: str
    frame_counter: int = 0
    recording_path: Path | None = None
    control_mode: str = "manual"
    selected_model: str | None = None

    def __post_init__(self) -> None:
        self.trace: deque[dict[str, int]] = deque(maxlen=TRACE_LIMIT)
        self.last_obs = None
        self.last_info: dict[str, Any] = {}
        self.record_file = None

    @property
    def recording(self) -> bool:
        return self.record_file is not None and self.recording_path is not None

    def close(self) -> None:
        if self.record_file is not None:
            self.record_file.close()
            self.record_file = None
        self.env.close()

    def append_trace(self, info: dict[str, Any]) -> None:
        room_id = int(info.get("room_id", 0) or 0)
        x = int(info.get("samus_x", 0) or 0)
        y = int(info.get("samus_y", 0) or 0)
        self.trace.append({"frame": self.frame_counter, "roomId": room_id, "x": x, "y": y})

    def write_record(self, action: list[int], terminated: bool, truncated: bool) -> None:
        if self.record_file is None:
            return
        entry = {
            "frame": self.frame_counter,
            "action": action,
            "room_id": self.last_info.get("room_id"),
            "samus_x": self.last_info.get("samus_x"),
            "samus_y": self.last_info.get("samus_y"),
            "health": self.last_info.get("health"),
            "game_state": self.last_info.get("game_state"),
            "terminated": terminated,
            "truncated": truncated,
        }
        self.record_file.write(json.dumps(entry))
        self.record_file.write("\n")
        self.record_file.flush()

    def build_snapshot(
        self,
        room_lookup: dict[int, tuple[str, str]],
        *,
        include_frame: bool = True,
        terminated: bool = False,
        truncated: bool = False,
    ) -> dict[str, Any]:
        if self.last_obs is None:
            return {
                "frameCounter": self.frame_counter,
                "trace": list(self.trace),
            }
        obs = self.last_obs
        height, width = int(obs.shape[0]), int(obs.shape[1])
        room_id = self.last_info.get("room_id")
        snapshot = {
            "frameCounter": self.frame_counter,
            "roomId": room_id,
            "roomName": _room_name(room_lookup, room_id),
            "areaName": _area_name(room_lookup, room_id),
            "gameState": self.last_info.get("game_state"),
            "health": self.last_info.get("health"),
            "samusX": self.last_info.get("samus_x"),
            "samusY": self.last_info.get("samus_y"),
            "terminated": terminated,
            "truncated": truncated,
            "frameWidth": width,
            "frameHeight": height,
            "trace": list(self.trace),
        }
        if include_frame:
            snapshot["frameRgb24Base64"] = base64.b64encode(obs.tobytes()).decode("ascii")
        return snapshot


class BridgeRuntime:
    def __init__(self) -> None:
        self.session: Session | None = None
        self.nav_export_dir = DEFAULT_NAV_EXPORT_DIR
        self.room_lookup = _room_lookup(self.nav_export_dir)
        self.control_mode = "manual"
        self.selected_model: str | None = None

    def _apply_config(
        self,
        *,
        nav_export_dir: str | None = None,
        control_mode: str | None = None,
        selected_model: str | None = None,
    ) -> None:
        if nav_export_dir is not None:
            nav_dir = Path(nav_export_dir).expanduser()
            self.nav_export_dir = nav_dir
            self.room_lookup = _room_lookup(nav_dir)
        if control_mode:
            self.control_mode = control_mode
        if selected_model is not None:
            self.selected_model = selected_model or None
        if self.session is not None:
            self.session.control_mode = self.control_mode
            self.session.selected_model = self.selected_model

    def hello(self) -> dict[str, Any]:
        return {
            "capabilities": {
                "game": GAME,
                "gameDir": str(PROJECT_DIR),
                "bridgeVersion": "0.1.0",
                "supportsFrames": True,
                "supportsRecording": True,
                "supportsKeyboardInput": True,
                "supportsAgentControl": True,
                "supportsHotConfig": True,
            },
            "message": "SM bridge ready",
        }

    def configure(
        self,
        *,
        nav_export_dir: str | None = None,
        control_mode: str | None = None,
        selected_model: str | None = None,
    ) -> dict[str, Any]:
        self._apply_config(
            nav_export_dir=nav_export_dir,
            control_mode=control_mode,
            selected_model=selected_model,
        )
        return {
            "session": self.session_state(),
            "message": "Bridge configuration updated",
        }

    def discover(self) -> dict[str, Any]:
        return {
            "states": _published_states(),
            "models": _published_models(),
            "session": self.session_state(),
            "message": "Discovered save states and models",
        }

    def session_state(self) -> dict[str, Any]:
        if self.session is None:
            return {
                "active": False,
                "paused": True,
                "currentState": None,
                "frameCounter": 0,
                "recording": False,
                "controlMode": self.control_mode,
                "selectedModel": self.selected_model,
            }
        return {
            "active": True,
            "paused": True,
            "currentState": self.session.current_state,
            "frameCounter": self.session.frame_counter,
            "recording": self.session.recording,
            "controlMode": self.session.control_mode,
            "selectedModel": self.session.selected_model,
        }

    def start_session(
        self,
        state_name: str,
        *,
        nav_export_dir: str | None = None,
        control_mode: str | None = None,
        selected_model: str | None = None,
        include_frame: bool = True,
    ) -> dict[str, Any]:
        np, make_env, _ = _runtime_imports()
        self._apply_config(
            nav_export_dir=nav_export_dir,
            control_mode=control_mode,
            selected_model=selected_model,
        )
        if self.session is not None:
            self.session.close()
            self.session = None
        env = make_env(GAME, state_name, PROJECT_DIR, render_mode="rgb_array")
        env.reset()
        obs, _, terminated, truncated, info = env.step(np.array(NOOP, dtype=np.int8))
        self.session = Session(
            env=env,
            current_state=state_name,
            frame_counter=1,
            control_mode=self.control_mode,
            selected_model=self.selected_model,
        )
        self.session.last_obs = obs
        self.session.last_info = info
        self.session.append_trace(info)
        return {
            "session": self.session_state(),
            "snapshot": self.session.build_snapshot(
                self.room_lookup,
                include_frame=include_frame,
                terminated=terminated,
                truncated=truncated,
            ),
            "message": f"Session started from {state_name}",
        }

    def require_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("No active session")
        return self.session

    def snapshot(self, *, include_frame: bool = True) -> dict[str, Any]:
        session = self.require_session()
        return {
            "session": self.session_state(),
            "snapshot": session.build_snapshot(self.room_lookup, include_frame=include_frame),
            "message": "Snapshot refreshed",
        }

    def step(self, action: list[int], repeat: int, *, include_frame: bool = True) -> dict[str, Any]:
        np, _, _ = _runtime_imports()
        session = self.require_session()
        repeat = max(1, min(int(repeat or 1), 8))
        action = [int(v) for v in action[:12]]
        if len(action) < 12:
            action = action + [0] * (12 - len(action))
        terminated = False
        truncated = False
        for _ in range(repeat):
            obs, _, terminated, truncated, info = session.env.step(np.array(action, dtype=np.int8))
            session.frame_counter += 1
            session.last_obs = obs
            session.last_info = info
            session.append_trace(info)
            session.write_record(action, terminated, truncated)
            if terminated or truncated:
                break
        return {
            "session": self.session_state(),
            "snapshot": session.build_snapshot(
                self.room_lookup,
                include_frame=include_frame,
                terminated=terminated,
                truncated=truncated,
            ),
        }

    def save_state(self, save_name: str) -> dict[str, Any]:
        _, _, save_state = _runtime_imports()
        session = self.require_session()
        if not save_name:
            raise RuntimeError("Missing save name")
        path = save_state(session.env, PROJECT_DIR, GAME, save_name)
        return {
            "session": self.session_state(),
            "message": f"Saved {save_name} -> {path.name}",
        }

    def load_state(self, state_name: str, *, include_frame: bool = True) -> dict[str, Any]:
        np, _, _ = _runtime_imports()
        state_path = PROJECT_DIR / "custom_integrations" / GAME / f"{state_name}.state"
        if not state_path.exists():
            raise RuntimeError(f"State not found: {state_path.name}")
        session = self.require_session()
        with gzip.open(state_path, "rb") as fh:
            data = fh.read()
        session.env.em.set_state(data)
        obs, _, terminated, truncated, info = session.env.step(np.array(NOOP, dtype=np.int8))
        session.current_state = state_name
        session.frame_counter += 1
        session.last_obs = obs
        session.last_info = info
        session.append_trace(info)
        return {
            "session": self.session_state(),
            "snapshot": session.build_snapshot(
                self.room_lookup,
                include_frame=include_frame,
                terminated=terminated,
                truncated=truncated,
            ),
            "message": f"Loaded {state_name}",
        }

    def start_recording(self) -> dict[str, Any]:
        session = self.require_session()
        if session.recording:
            return {
                "session": self.session_state(),
                "recordingPath": str(session.recording_path),
                "message": "Recording already active",
            }
        recordings_dir = PROJECT_DIR / "editor_recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        session.recording_path = recordings_dir / f"editor_session_{int(time.time())}.jsonl"
        session.record_file = session.recording_path.open("w", encoding="utf-8")
        return {
            "session": self.session_state(),
            "recordingPath": str(session.recording_path),
            "message": "Recording started",
        }

    def stop_recording(self) -> dict[str, Any]:
        session = self.require_session()
        if session.record_file is not None:
            session.record_file.close()
            session.record_file = None
        return {
            "session": self.session_state(),
            "recordingPath": str(session.recording_path) if session.recording_path else None,
            "message": "Recording stopped",
        }

    def close_session(self) -> dict[str, Any]:
        if self.session is not None:
            self.session.close()
            self.session = None
        return {
            "session": self.session_state(),
            "message": "Session closed",
        }


def _run_stdio() -> int:
    runtime = BridgeRuntime()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id: str | None = None
        try:
            data = json.loads(line)
            request_id = data.get("id")
            command = data.get("command")
            nav_export_dir = data.get("navExportDir")
            control_mode = data.get("controlMode")
            selected_model = data.get("selectedModel")
            include_frame = bool(data.get("includeFrame", True))
            if command == "hello":
                _write(_json_response(request_id=request_id, **runtime.hello()))
            elif command == "configure":
                _write(_json_response(
                    request_id=request_id,
                    **runtime.configure(
                        nav_export_dir=nav_export_dir,
                        control_mode=control_mode,
                        selected_model=selected_model,
                    ),
                ))
            elif command == "discover":
                runtime._apply_config(
                    nav_export_dir=nav_export_dir,
                    control_mode=control_mode,
                    selected_model=selected_model,
                )
                _write(_json_response(request_id=request_id, **runtime.discover()))
            elif command == "start_session":
                _write(_json_response(
                    request_id=request_id,
                    **runtime.start_session(
                        str(data.get("state") or ""),
                        nav_export_dir=nav_export_dir,
                        control_mode=control_mode,
                        selected_model=selected_model,
                        include_frame=include_frame,
                    ),
                ))
            elif command == "snapshot":
                _write(_json_response(
                    request_id=request_id,
                    **runtime.snapshot(include_frame=include_frame),
                ))
            elif command == "step":
                _write(_json_response(
                    request_id=request_id,
                    **runtime.step(
                        list(data.get("action") or []),
                        int(data.get("repeat", 1)),
                        include_frame=include_frame,
                    ),
                ))
            elif command == "save_state":
                _write(_json_response(request_id=request_id, **runtime.save_state(str(data.get("saveName") or ""))))
            elif command == "load_state":
                _write(_json_response(
                    request_id=request_id,
                    **runtime.load_state(str(data.get("state") or ""), include_frame=include_frame),
                ))
            elif command == "start_recording":
                _write(_json_response(request_id=request_id, **runtime.start_recording()))
            elif command == "stop_recording":
                _write(_json_response(request_id=request_id, **runtime.stop_recording()))
            elif command == "close_session":
                _write(_json_response(request_id=request_id, **runtime.close_session()))
            else:
                _write(_json_response(request_id=request_id, ok=False, error=f"Unknown command: {command}"))
        except Exception as exc:  # pragma: no cover - exercised in live runtime
            _write(_json_response(request_id=request_id, ok=False, error=str(exc)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--stdio" not in args:
        sys.stderr.write("editor_bridge.py only supports --stdio\n")
        return 2
    return _run_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
