"""Button script recording segment helpers."""

from __future__ import annotations

from retro_harness.editor.util import safe_recording_slug


def recording_buttons_for_action(
    action: list[int],
    button_order: tuple[str, ...],
    *,
    idle_label: str = "IDLE",
) -> list[str]:
    buttons = [
        name
        for index, name in enumerate(button_order)
        if index < len(action) and int(action[index])
    ]
    return buttons or [idle_label]


def append_recording_segment(
    segments: list[dict[str, object]],
    action: list[int],
    frames: int,
    *,
    button_order: tuple[str, ...],
    idle_label: str = "IDLE",
) -> None:
    frames = int(frames)
    if frames <= 0:
        return
    buttons = recording_buttons_for_action(action, button_order, idle_label=idle_label)
    if segments and "label" not in segments[-1] and segments[-1].get("buttons") == buttons:
        segments[-1]["frames"] = int(segments[-1].get("frames", 0)) + frames
        return
    segments.append({"buttons": buttons, "frames": frames})


def append_recording_marker(
    segments: list[dict[str, object]],
    label: str,
    *,
    idle_label: str = "IDLE",
) -> dict[str, object]:
    marker = {"buttons": [idle_label], "frames": 0, "label": label}
    segments.append(marker)
    return marker


__all__ = [
    "append_recording_marker",
    "append_recording_segment",
    "recording_buttons_for_action",
    "safe_recording_slug",
]
