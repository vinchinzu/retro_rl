"""Headless probe helpers for autonomous task and recording diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import task_progress_snapshot
from harvest.tasks.farm_clearer import ADDR_TILEMAP, TILE_SIZE, get_pos_from_ram, get_tile_at
from harvest.runtime.recording_trace import pressed_buttons


DEFAULT_WATCH_FIELDS = (
    "tilemap",
    "input_lock",
    "player_state",
    "player_action",
    "held_item",
    "tool_selected",
    "hour",
    "minute",
    "stored_grass",
    "cow_feed",
    "chicken_feed",
    "num_cows",
    "num_chickens",
    "fed_cows_n",
    "fed_chickens_n",
    "egg_available",
    "incubator_flags",
    "shipping_money",
)


@dataclass(frozen=True)
class ProbeSnapshot:
    frame: int
    tilemap: int
    x: int
    y: int
    tx: int
    ty: int
    tile_id: int
    buttons: tuple[str, ...]

    def as_event(self) -> dict[str, object]:
        row = asdict(self)
        row["tilemap_hex"] = f"0x{self.tilemap:02X}"
        row["tile_id_hex"] = f"0x{self.tile_id:02X}"
        row["tile"] = [self.tx, self.ty]
        row["pixel"] = [self.x, self.y]
        return row


def parse_field_list(values: Iterable[str] | None, *, default: Sequence[str] = DEFAULT_WATCH_FIELDS) -> list[str]:
    """Parse comma-separated and repeated field names."""
    fields: list[str] = []
    for value in values or ():
        fields.extend(part.strip() for part in value.split(",") if part.strip())
    return fields or list(default)


def parse_frame_ranges(values: Iterable[str] | None) -> list[tuple[int, int]]:
    """Parse ranges like ``600:900`` or a single frame ``1200``."""
    ranges: list[tuple[int, int]] = []
    for value in values or ():
        for part in value.split(","):
            text = part.strip()
            if not text:
                continue
            if ":" in text:
                start_text, end_text = text.split(":", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
            else:
                start = end = int(text)
            if end < start:
                raise ValueError(f"Frame range end before start: {text}")
            ranges.append((start, end))
    return ranges


def frame_in_ranges(frame: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in ranges)


def snapshot_from_ram(ram: np.ndarray, *, frame: int, action: Sequence[int]) -> ProbeSnapshot:
    pos = get_pos_from_ram(ram)
    tx = pos.x // TILE_SIZE
    ty = pos.y // TILE_SIZE
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    return ProbeSnapshot(
        frame=int(frame),
        tilemap=tilemap,
        x=int(pos.x),
        y=int(pos.y),
        tx=int(tx),
        ty=int(ty),
        tile_id=int(get_tile_at(ram, tx, ty)),
        buttons=tuple(pressed_buttons(action)),
    )


def watch_values(ram: np.ndarray, fields: Sequence[str]) -> dict[str, int]:
    return {field: int(read_ram_value(ram, field, raw=True)) for field in fields}


def watch_changes(before: dict[str, int] | None, after: dict[str, int]) -> dict[str, dict[str, int]]:
    if before is None:
        return {}
    return {
        key: {"from": int(before[key]), "to": int(value)}
        for key, value in after.items()
        if key in before and before[key] != value
    }


def task_debug_snapshot(task, *, depth: int = 0, max_depth: int = 3) -> dict[str, object]:
    """Return best-effort debug state for an autonomous task object."""
    if task is None:
        return {}

    row: dict[str, object] = {
        "class": task.__class__.__name__,
    }
    phase = getattr(task, "_phase", None)
    if phase is not None:
        row["phase"] = str(phase)

    navigator = getattr(task, "_navigator", None)
    if navigator is not None:
        row["nav_tile"] = list(navigator.current_tile)
        row["nav_pos"] = [int(navigator.current_pos.x), int(navigator.current_pos.y)]
        row["stasis"] = int(getattr(navigator, "stasis", 0))
        path = list(getattr(navigator, "path", []) or [])
        row["path_len"] = len(path)
        row["path_head"] = [list(item) for item in path[:6]]

    current = getattr(task, "_current", None)
    if current is not None:
        row["current"] = repr(current)

    action_queue = getattr(task, "_action_queue", None)
    if action_queue is not None:
        row["action_queue_len"] = len(action_queue)

    if depth < max_depth:
        nested = getattr(task, "current_task", None)
        if nested is None:
            nested = getattr(task, "_current_task", None)
        if nested is None:
            nested = getattr(task, "_task", None)
        if nested is not None and nested is not task:
            row["current_task"] = task_debug_snapshot(nested, depth=depth + 1, max_depth=max_depth)

    return row


def day_plan_debug_snapshot(day_plan_task) -> dict[str, object]:
    """Return current day-plan state plus current subtask debug state."""
    if day_plan_task is None:
        return {}

    row: dict[str, object] = {
        "class": day_plan_task.__class__.__name__,
    }
    for attr, key in (
        ("phase_text", "phase_text"),
        ("progress_text", "progress_text"),
        ("phase_index", "phase_index"),
        ("step_count", "step_count"),
    ):
        try:
            value = getattr(day_plan_task, attr)
        except Exception:
            continue
        row[key] = value

    progress = task_progress_snapshot(day_plan_task)
    if progress is not None:
        row["progress_signature"] = progress.signature()

    current = getattr(day_plan_task, "current_task", None)
    if current is not None:
        row["current_task"] = task_debug_snapshot(current)
    return row


def event_row(
    event: str,
    snapshot: ProbeSnapshot,
    *,
    watches: dict[str, int] | None = None,
    changes: dict[str, dict[str, int]] | None = None,
    day_plan=None,
    task=None,
    note: str | None = None,
) -> dict[str, object]:
    row = {"event": event, **snapshot.as_event()}
    if watches:
        row["watch"] = dict(watches)
    if changes:
        row["changes"] = changes
    if day_plan is not None:
        row["day_plan"] = day_plan_debug_snapshot(day_plan)
    if task is not None:
        row["task"] = task_debug_snapshot(task)
    if note:
        row["note"] = note
    return row
