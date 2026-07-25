"""Normalize editor emulator script JSON segments."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_IDLE_LABEL = "IDLE"


def normalize_script_segment(
    raw: object,
    *,
    idle_label: str = DEFAULT_IDLE_LABEL,
) -> dict[str, object]:
    if isinstance(raw, str):
        raw = {"buttons": raw, "frames": 1}
    if not isinstance(raw, dict):
        raise ValueError(f"Script segment must be an object or string, got {type(raw).__name__}")
    segment = dict(raw)
    buttons = segment.get("buttons", segment.get("button", []))
    if isinstance(buttons, str):
        compact = buttons.replace("+", " ").replace(",", " ")
        segment["buttons"] = [part.strip().upper() for part in compact.split() if part.strip()]
    elif buttons is None:
        segment["buttons"] = [idle_label]
    else:
        segment["buttons"] = [str(item).upper() for item in buttons]
    if not segment["buttons"]:
        segment["buttons"] = [idle_label]
    segment["frames"] = int(segment.get("frames", segment.get("repeat", 1)) or 1)
    return segment


def script_segments_from_payload(
    payload: object,
    *,
    idle_label: str = DEFAULT_IDLE_LABEL,
) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        payload = payload.get("segments", [])
    if not isinstance(payload, list):
        raise ValueError("Script payload must be a list or an object with a segments list")
    return [normalize_script_segment(item, idle_label=idle_label) for item in payload]


def script_segments_from_file(
    path: Path,
    *,
    idle_label: str = DEFAULT_IDLE_LABEL,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {}, script_segments_from_payload(data, idle_label=idle_label)
    if not isinstance(data, dict):
        raise ValueError("Script file must contain a list or object")
    segments: list[dict[str, object]] = []
    chain = data.get("chain")
    if isinstance(chain, list):
        for item in chain:
            child = Path(str(item)).expanduser()
            if not child.is_absolute():
                child = path.parent / child
            _child_data, child_segments = script_segments_from_file(child, idle_label=idle_label)
            segments.append({"buttons": [idle_label], "frames": 0, "label": child.stem})
            segments.extend(child_segments)
    own_segments = data.get("segments")
    if isinstance(own_segments, list):
        segments.extend(script_segments_from_payload(own_segments, idle_label=idle_label))
    return data, segments
