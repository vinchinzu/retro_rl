"""JSON-line stdio protocol helpers for editor bridge subprocesses."""

from __future__ import annotations

import json
import sys
from typing import Any

from retro_harness.runtime import reset_env as reset_env
from retro_harness.runtime import step_env as step_env


def json_response(
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


def write_stdio_payload(
    payload: dict[str, Any],
    *,
    frame_rgb: bytes | None = None,
    wram: bytes | None = None,
) -> None:
    if frame_rgb is not None:
        payload = {**payload, "frameBinaryLength": len(frame_rgb)}
    if wram is not None:
        payload = {**payload, "wramBinaryLength": len(wram)}
    line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    sys.stdout.buffer.write(line)
    if frame_rgb is not None:
        sys.stdout.buffer.write(frame_rgb)
    if wram is not None:
        sys.stdout.buffer.write(wram)
    sys.stdout.buffer.flush()


def emit_stdio_response(
    *,
    request_id: str | None,
    result: dict[str, Any],
) -> None:
    frame_rgb = result.pop("_frame_rgb", None)
    wram = None
    snapshot = result.get("snapshot")
    if isinstance(snapshot, dict):
        raw = snapshot.pop("wramRaw", None)
        if isinstance(raw, (bytes, bytearray, memoryview)):
            wram = bytes(raw)
    payload = json_response(request_id=request_id, **result)
    write_stdio_payload(
        payload,
        frame_rgb=frame_rgb if isinstance(frame_rgb, (bytes, bytearray)) else None,
        wram=wram,
    )
