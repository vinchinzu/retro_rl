"""Trace, RAM-delta, failure-tail, and state-provenance helpers."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from retro_harness.adventure.hashutil import sha256_file
from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.ram_state import diff_changed
from zelda_i.dungeon_ids import (
    mode_name,
    object_name,
    ram_symbol,
    room_item_name,
)
from zelda_i.ram import ZeldaSnapshot


def action_button_names(action: Iterable[int]) -> list[str]:
    """Decode a stable-retro NES action vector into pressed button names."""
    values = list(action)
    return [
        name
        for name, index in NES_BUTTON_NAME_TO_INDEX.items()
        if index is not None and index < len(values) and values[index]
    ]


def compact_snapshot(snap: ZeldaSnapshot) -> dict[str, Any]:
    """JSON-safe per-frame state used by trace and divergence reports."""
    return {
        "mode": snap.mode,
        "mode_name": mode_name(snap.mode),
        "submode": snap.submode,
        "is_updating_mode": snap.is_updating_mode,
        "level": snap.level,
        "room": snap.screen,
        "next_room": snap.next_screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "facing": snap.facing,
        "health": snap.health,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "rupees": snap.rupees,
        "bombs": snap.bombs,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "objects": [
            {
                "slot": obj.slot,
                "type_id": obj.type_id,
                "type_name": object_name(obj.type_id),
                "x": obj.x,
                "y": obj.y,
                "hp": obj.hp,
                "state": obj.state,
            }
            for obj in snap.objects
            if obj.type_id or obj.hp
        ],
    }


@dataclass
class TraceRecorder:
    """Accumulate a full trace plus an always-available failure tail."""

    tail_frames: int = 120
    frames: list[dict[str, Any]] = field(default_factory=list)
    _tail: deque[dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tail = deque(maxlen=max(1, self.tail_frames))

    def record(
        self,
        *,
        frame: int,
        phase: str,
        reason: str,
        action: Iterable[int],
        snap: ZeldaSnapshot,
    ) -> None:
        entry = {
            "frame": int(frame),
            "phase": phase,
            "reason": reason,
            "action": list(action),
            "buttons": action_button_names(action),
            "state": compact_snapshot(snap),
        }
        self.frames.append(entry)
        self._tail.append(entry)

    @property
    def tail(self) -> list[dict[str, Any]]:
        return list(self._tail)

    def write(self, path: Path, *, tail_only: bool = False) -> Path:
        entries = self.tail if tail_only else self.frames
        return write_jsonl(path, entries)


def write_jsonl(path: Path, entries: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    entries.append(value)
    return entries


def first_trace_divergence(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first differing action/policy/state frame."""
    left_items = list(left)
    right_items = list(right)
    common = min(len(left_items), len(right_items))
    fields = ("phase", "reason", "action", "state")
    for index in range(common):
        lhs = left_items[index]
        rhs = right_items[index]
        changed = [field for field in fields if lhs.get(field) != rhs.get(field)]
        if changed:
            return {
                "index": index,
                "frame": min(int(lhs.get("frame", index)), int(rhs.get("frame", index))),
                "changed_fields": changed,
                "left": lhs,
                "right": rhs,
            }
    if len(left_items) != len(right_items):
        return {
            "index": common,
            "frame": common,
            "changed_fields": ["trace_length"],
            "left": left_items[common] if common < len(left_items) else None,
            "right": right_items[common] if common < len(right_items) else None,
        }
    return None


def ram_delta_report(
    before: np.ndarray,
    after: np.ndarray,
    *,
    unknown_limit: int = 128,
) -> dict[str, Any]:
    """Describe all known changes and a bounded list of unknown changes."""
    known: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    all_deltas = diff_changed(before, after, limit=None)
    for delta in all_deltas:
        row = {
            "address": delta.address,
            "address_hex": f"0x{delta.address:04X}",
            "before": delta.before,
            "after": delta.after,
            "delta": delta.delta,
        }
        symbol = ram_symbol(delta.address)
        if symbol is None:
            if len(unknown) < unknown_limit:
                unknown.append(row)
        else:
            known.append({**row, "symbol": symbol})
    return {
        "changed_count": len(all_deltas),
        "known": known,
        "unknown": unknown,
        "unknown_truncated": max(0, len(all_deltas) - len(known) - len(unknown)),
    }


def write_state_provenance(
    state_path: Path,
    *,
    source_state_path: Path | None,
    request: dict[str, Any],
    selected_trial: dict[str, Any],
    natural_entry: bool = False,
) -> Path:
    """Write an auditable sidecar for a generated development checkpoint."""
    sidecar = state_path.with_suffix(".provenance.json")
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "state_path": str(state_path.resolve()),
        "state_sha256": sha256_file(state_path),
        "source_state_path": (
            str(source_state_path.resolve()) if source_state_path else None
        ),
        "source_state_sha256": (
            sha256_file(source_state_path) if source_state_path else None
        ),
        "request": request,
        "selected_trial": selected_trial,
        "natural_entry": natural_entry,
        "development_only": not natural_entry,
        "acceptance_warning": (
            (
                "Captured from power-on without a state load; retain the matching "
                "evidence manifest for acceptance."
            )
            if natural_entry
            else (
                "This checkpoint is for isolated development. Route readiness "
                "still requires a successful natural-entry run from the real "
                "predecessor."
            )
        ),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar
