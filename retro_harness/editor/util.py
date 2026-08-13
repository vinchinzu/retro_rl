"""Small helpers shared by editor emulator panels."""

from __future__ import annotations

from retro_harness.emulator_session import frame_budget_ms_for_speed


def int_value(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_recording_slug(value: str, *, default: str = "editor_recording") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe.strip("_") or default


__all__ = ["frame_budget_ms_for_speed", "int_value", "safe_recording_slug"]
