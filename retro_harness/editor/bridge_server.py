"""Stdio JSON command loop for editor bridge subprocesses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from retro_harness.editor.bridge_protocol import emit_stdio_response
from retro_harness.editor.bridge_runtime import EditorBridgeRuntime


def read_stdio_request() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    try:
        payload = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return {"id": None, "command": "__invalid__"}
    if not isinstance(payload, dict):
        return {"id": None, "command": "__invalid__"}
    return payload


def handle_bridge_command(
    runtime: EditorBridgeRuntime,
    *,
    request_id: str | None,
    command: str,
    payload: dict[str, Any],
    default_state: object | None = None,
) -> dict[str, Any]:
    include_frame = bool(payload.get("includeFrame", True))

    if command in {"hello", "ping"}:
        return {"ok": True, "message": "ready"}

    if command == "discover":
        default_path = default_state if isinstance(default_state, Path) else None
        return {
            "ok": True,
            "states": runtime.discover_states(default_state=default_path),
        }

    if command == "close_session":
        runtime.close()
        return {"ok": True, "message": "Session closed"}

    if command == "start_session":
        try:
            snapshot, frame_rgb = runtime.start_session(
                state_file=payload.get("stateFile"),
                rom_path=payload.get("romPath"),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "snapshot": snapshot, "message": snapshot.get("message"), "_frame_rgb": frame_rgb}

    if command == "step":
        try:
            action = payload.get("action")
            if isinstance(action, list):
                action_values = [int(value) for value in action]
            else:
                action_values = None
            snapshot, frame_rgb, step_ms = runtime.step(
                action=action_values,
                repeat=int(payload.get("repeat", 1)),
                include_frame=include_frame,
                include_wram=bool(payload.get("includeWram", True)),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "snapshot": snapshot,
            "stepMs": step_ms,
            "_frame_rgb": frame_rgb,
        }

    if command == "set_autoplay":
        setter = getattr(runtime, "set_autoplay", None)
        if not callable(setter):
            return {"ok": False, "error": "Autoplay is not supported by this editor runtime"}
        try:
            status = setter(
                enabled=bool(payload.get("enabled", False)),
                state_name=payload.get("stateName"),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        result: dict[str, Any] = {"ok": True, **status}
        if include_frame and runtime.env is not None:
            snapshot, frame_rgb = runtime._snapshot(include_frame=True)
            result["snapshot"] = snapshot
            result["_frame_rgb"] = frame_rgb
        return result

    if command == "hot_save":
        try:
            capture = runtime.hot_save()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "capture": capture, "message": capture.get("message")}

    if command == "load_hot_save":
        try:
            snapshot, frame_rgb = runtime.load_hot_save()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "snapshot": snapshot,
            "message": snapshot.get("message"),
            "_frame_rgb": frame_rgb,
        }

    if command == "capture":
        try:
            capture = runtime.capture(
                prefix=str(payload.get("prefix") or "editor_capture"),
                include_frame=include_frame,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        frame_rgb = None
        if include_frame and runtime.env is not None:
            obs = runtime.env.render()
            if hasattr(obs, "tobytes"):
                frame_rgb = obs.tobytes()
        result: dict[str, Any] = {
            "ok": True,
            "capture": capture,
            "message": f"Captured {capture.get('prefix')}",
        }
        if include_frame and runtime.env is not None:
            snapshot, frame_rgb = runtime._snapshot(include_frame=True)
            result["snapshot"] = snapshot
            result["_frame_rgb"] = frame_rgb
        return result

    if command == "toggle_ram_recording":
        try:
            payload_result = runtime.toggle_ram_recording(
                label=str(payload.get("label") or "editor_ram"),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **payload_result}

    if command == "run_script":
        try:
            snapshot, frame_rgb = runtime.run_script(
                segments=payload.get("segments", []),
                prefix=str(payload.get("prefix") or "editor_script"),
                capture_each_segment=bool(payload.get("captureEachSegment", False)),
                include_frame=include_frame,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "snapshot": snapshot,
            "message": snapshot.get("message"),
            "_frame_rgb": frame_rgb,
        }

    if command == "__invalid__":
        return {"ok": False, "error": "Invalid JSON request"}

    return {"ok": False, "error": f"Unknown command: {command}"}


def run_stdio_bridge(
    runtime: EditorBridgeRuntime,
    *,
    default_state: object | None = None,
) -> None:
    """Process JSON-line commands from stdin until EOF."""

    while True:
        request = read_stdio_request()
        if request is None:
            runtime.close()
            return
        request_id = request.get("id")
        command = str(request.get("command") or "")
        result = handle_bridge_command(
            runtime,
            request_id=str(request_id) if request_id is not None else None,
            command=command,
            payload=request,
            default_state=default_state,
        )
        emit_stdio_response(request_id=str(request_id) if request_id is not None else None, result=result)


def bridge_main(
    *,
    build_runtime: object,
    description: str,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stdio", action="store_true", help="Run JSON-line bridge on stdio")
    args = parser.parse_args()
    if not args.stdio:
        parser.error("Only --stdio mode is supported")
    runtime = build_runtime()
    run_stdio_bridge(runtime)
