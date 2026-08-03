"""Generic human task recording: frames, position traces, save/load, analysis.

Game packages should keep RAM-specific trace builders (tilemaps, entities) local.
Shared pieces live here so Harvest, Super Metroid, and others record the same way.

Harvest's recorder is the reference consumer: JSON ``frames`` + optional
per-frame ``trace`` rows + ``metadata`` summary + optional end-state gzip.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from retro_harness.controls import SNES_BUTTON_NAMES, pressed_snes_buttons

MOVEMENT_BUTTONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT", "Up", "Down", "Left", "Right"})


def pressed_buttons(
    action: Sequence[int],
    names: Sequence[str] = SNES_BUTTON_NAMES,
) -> list[str]:
    """Return pressed button names for an action vector.

    Defaults to SNES layout (uppercase names). Pass a custom ``names`` map for
    game-local casing (e.g. Harvest title-case labels).
    """
    if names is SNES_BUTTON_NAMES or tuple(names) == SNES_BUTTON_NAMES:
        return pressed_snes_buttons(list(action))
    return [name for idx, name in enumerate(names) if idx < len(action) and int(action[idx]) != 0]


def coalesce_windows(frames: Iterable[int]) -> list[dict[str, int]]:
    """Merge sorted frame indices into contiguous {start, end, length} windows."""
    ordered = sorted(frames)
    if not ordered:
        return []
    windows: list[dict[str, int]] = []
    start = end = ordered[0]
    for frame in ordered[1:]:
        if frame == end + 1:
            end = frame
            continue
        windows.append({"start": start, "end": end, "length": end - start + 1})
        start = end = frame
    windows.append({"start": start, "end": end, "length": end - start + 1})
    return windows


def coalesce_action_runs(
    frames: Sequence[Sequence[int]],
    *,
    names: Sequence[str] = SNES_BUTTON_NAMES,
) -> list[dict[str, object]]:
    """Collapse consecutive identical non-empty button holds into runs."""
    runs: list[dict[str, object]] = []
    start: int | None = None
    last_buttons: list[str] | None = None

    for idx, frame in enumerate(frames):
        buttons = pressed_buttons(frame, names=names)
        if not buttons:
            if start is not None and last_buttons is not None:
                runs.append(
                    {
                        "start": start,
                        "end": idx - 1,
                        "length": idx - start,
                        "buttons": last_buttons,
                    }
                )
                start = None
                last_buttons = None
            continue
        if start is None:
            start = idx
            last_buttons = buttons
            continue
        if buttons != last_buttons:
            runs.append(
                {
                    "start": start,
                    "end": idx - 1,
                    "length": idx - start,
                    "buttons": last_buttons,
                }
            )
            start = idx
            last_buttons = buttons

    if start is not None and last_buttons is not None:
        runs.append(
            {
                "start": start,
                "end": len(frames) - 1,
                "length": len(frames) - start,
                "buttons": last_buttons,
            }
        )
    return runs


def stasis_windows(
    trace: Sequence[dict[str, object]],
    *,
    x_key: str = "x",
    y_key: str = "y",
    buttons_key: str = "buttons",
    movement_buttons: frozenset[str] = MOVEMENT_BUTTONS,
    min_length: int = 45,
    frame_key: str = "frame",
) -> list[dict[str, object]]:
    """Find stretches where movement is held but position does not change."""
    windows: list[dict[str, object]] = []
    start_idx: int | None = None
    last_pos: tuple[int, int] | None = None

    for idx, row in enumerate(trace):
        pos = (int(row.get(x_key, 0)), int(row.get(y_key, 0)))
        buttons = set(row.get(buttons_key, []) or [])
        moving = bool(buttons & movement_buttons)
        if moving and pos == last_pos:
            if start_idx is None:
                start_idx = idx - 1 if idx > 0 else idx
        else:
            if start_idx is not None and idx - start_idx >= min_length:
                windows.append(_stasis_window(trace, start_idx, idx - 1, x_key, y_key, buttons_key, frame_key))
            start_idx = None
        last_pos = pos

    if start_idx is not None and len(trace) - start_idx >= min_length:
        windows.append(
            _stasis_window(trace, start_idx, len(trace) - 1, x_key, y_key, buttons_key, frame_key)
        )
    return windows


def _stasis_window(
    trace: Sequence[dict[str, object]],
    start_idx: int,
    end_idx: int,
    x_key: str,
    y_key: str,
    buttons_key: str,
    frame_key: str,
) -> dict[str, object]:
    first = trace[start_idx]
    last = trace[end_idx]
    buttons = sorted(
        {
            button
            for row in trace[start_idx : end_idx + 1]
            for button in (row.get(buttons_key, []) or [])
        }
    )
    start_frame = int(first.get(frame_key, start_idx))
    end_frame = int(last.get(frame_key, end_idx))
    return {
        "start": start_frame,
        "end": end_frame,
        "length": end_frame - start_frame + 1,
        "pixel_start": [int(first.get(x_key, 0)), int(first.get(y_key, 0))],
        "pixel_end": [int(last.get(x_key, 0)), int(last.get(y_key, 0))],
        "buttons": buttons,
    }


def value_change_windows(
    trace: Sequence[dict[str, object]],
    key: str,
    *,
    frame_key: str = "frame",
) -> list[dict[str, int]]:
    """Frame windows where ``key`` changes between consecutive trace rows."""
    if not trace:
        return []
    prev = trace[0].get(key)
    changed: list[int] = []
    for row in trace[1:]:
        value = row.get(key)
        if value != prev:
            changed.append(int(row.get(frame_key, 0)))
        prev = value
    return coalesce_windows(changed)


def summarize_position_trace(
    *,
    frames: Sequence[Sequence[int]],
    trace: Sequence[dict[str, object]],
    room_key: str = "room",
    x_key: str = "x",
    y_key: str = "y",
    names: Sequence[str] = SNES_BUTTON_NAMES,
) -> dict[str, object]:
    """Game-agnostic summary: duration, room transitions, stasis, input runs."""
    transitions: list[dict[str, object]] = []
    for idx, row in enumerate(trace):
        room = row.get(room_key)
        prev_room = trace[idx - 1].get(room_key) if idx > 0 else object()
        if idx == 0 or room != prev_room:
            transitions.append(
                {
                    "frame": int(row.get("frame", idx)),
                    "room": room,
                    "x": int(row.get(x_key, 0)),
                    "y": int(row.get(y_key, 0)),
                }
            )
    return {
        "frame_count": len(frames),
        "duration_seconds": len(frames) / 60.0,
        "transitions": transitions,
        "recorded_input_runs": coalesce_action_runs(frames, names=names),
        "stasis_windows": stasis_windows(trace, x_key=x_key, y_key=y_key),
    }


@dataclass
class RecordedTask:
    """A human input capture: action frames + optional position/RAM trace."""

    name: str
    frames: list[list[int]] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    start_state: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    end_state_data: bytes | None = None
    recorded_at: str | None = None

    def save(
        self,
        path: str | Path,
        *,
        end_state_paths: Sequence[str | Path] | None = None,
        indent: int = 2,
    ) -> Path:
        """Write task JSON and optional gzip end-state mirrors."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "frames": self.frames,
            "trace": self.trace,
            "start_state": self.start_state,
            "metadata": self.metadata,
            "recorded_at": self.recorded_at or datetime.now().isoformat(),
            "frame_count": len(self.frames),
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

        if self.end_state_data is not None:
            paths = list(end_state_paths) if end_state_paths else [out.with_name(out.stem + "_end.state")]
            for state_path in paths:
                sp = Path(state_path)
                sp.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(sp, "wb") as gz:
                    gz.write(self.end_state_data)
        return out

    @classmethod
    def load(cls, path: str | Path) -> RecordedTask:
        """Load task JSON; attach end-state gzip if present next to the file."""
        src = Path(path)
        with src.open(encoding="utf-8") as f:
            data = json.load(f)
        task = cls(
            name=str(data.get("name") or src.stem),
            frames=list(data.get("frames") or []),
            trace=list(data.get("trace") or []),
            start_state=data.get("start_state"),
            metadata=dict(data.get("metadata") or {}),
            recorded_at=data.get("recorded_at"),
        )
        end_path = src.with_name(src.stem + "_end.state")
        if end_path.exists():
            with gzip.open(end_path, "rb") as gz:
                task.end_state_data = gz.read()
        return task

    def append_frame(
        self,
        action: Sequence[int],
        *,
        trace_row: dict[str, object] | None = None,
    ) -> None:
        self.frames.append([int(v) for v in action])
        if trace_row is not None:
            self.trace.append(trace_row)
