"""Small helpers shared by editor emulator panels."""

from __future__ import annotations


def int_value(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def frame_budget_ms_for_speed(speed: float, *, base_frame_ms: int = 16) -> int:
    """Return wall-clock delay between emulator steps at ``speed``."""

    if speed <= 0:
        return base_frame_ms
    return max(1, int(round(base_frame_ms / speed)))


def safe_recording_slug(value: str, *, default: str = "editor_recording") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe.strip("_") or default
