"""Generic bridge snapshot helpers (game fields live in each title)."""

from __future__ import annotations


def snapshot_int(snapshot: dict[str, object], key: str) -> int | None:
    value = snapshot.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def snapshot_frame_counter(snapshot: dict[str, object]) -> int:
    try:
        return int(snapshot.get("frameCounter") or 0)
    except (TypeError, ValueError):
        return 0


def snapshot_flag(snapshot: dict[str, object], key: str) -> bool:
    return bool(snapshot.get(key))


def snapshot_without_frame(snapshot: dict[str, object]) -> dict[str, object]:
    """Drop bulky pixel/WRAM bytes while keeping HUD / recording metadata."""

    return {
        key: value
        for key, value in snapshot.items()
        if key not in {"frameRgb24Base64", "frameRgb24Raw", "wramBase64", "wramRaw"}
    }
