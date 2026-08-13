"""32-exit human-tape helpers for ``./play smb``.

Mirrors Super Metroid ``./play`` seams at SMB scale: archive prior takes,
clock declared route exits (deaths do not abort), and write durable stage pins.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from smb.ram import PLAYER_STATE_DYING, SmbSnapshot
from smb.reactive_route import level_control_gate, snapshot_fingerprint
from smb.routes import ExitRoute, ExitSegment
from smb.start_presets import pin_meta_path, pin_state_path, pins_dir

_SEG_DIR_RE = re.compile(r"^s(\d+)$")

NES9_LAYOUT = ("B", "_", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")


def stage_label(world: int, level: int) -> str:
    return f"{int(world) + 1}-{int(level) + 1}"


def stage_label_of(snap: SmbSnapshot) -> str:
    return stage_label(snap.world, snap.dash_level)


def fmt_time(frames: int) -> str:
    """60fps wall-time label: m:ss.mmm"""
    total_ms = max(0, int(frames)) * 1000 // 60
    minutes, rest = divmod(total_ms, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def pressed_nes9(action: list[int] | tuple[int, ...]) -> list[str]:
    pressed: list[str] = []
    for idx, name in enumerate(NES9_LAYOUT):
        if name == "_":
            continue
        if idx < len(action) and int(action[idx]) != 0:
            pressed.append(name)
    return pressed


def trace_row(snap: SmbSnapshot, action: list[int], *, rec_frame: int) -> dict[str, Any]:
    return {
        "frame": rec_frame,
        "world": snap.world,
        "level": snap.level,
        "stage": stage_label_of(snap),
        "x": snap.player_x,
        "y": snap.player_y,
        "xs": snap.x_speed,
        "ys": snap.y_speed,
        "player_state": snap.player_state,
        "oper_mode": snap.oper_mode,
        "timer": snap.timer,
        "lives": snap.lives,
        "area_pointer": snap.area_pointer,
        "in_air": bool(snap.in_air),
        "buttons": pressed_nes9(action),
        "source": "human",
    }


@dataclass
class ExitClock:
    """Forgiving 32-exit (or warp) clock for a live human take.

    Unlike :class:`RouteProgressTracker`, a death is logged and the clock
    keeps going — humans continue from the last life / checkpoint.
    """

    route: ExitRoute
    start_index: int = 0
    completed: list[dict[str, Any]] = field(default_factory=list)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    deaths: list[dict[str, Any]] = field(default_factory=list)
    off_route: list[dict[str, Any]] = field(default_factory=list)
    _was_dying: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.start_index < len(self.route.exits):
            raise ValueError(
                f"start_index {self.start_index} outside route of {len(self.route.exits)} exits"
            )

    @property
    def next_index(self) -> int:
        return self.start_index + len(self.completed)

    @property
    def next_exit(self) -> ExitSegment | None:
        if self.next_index >= len(self.route.exits):
            return None
        return self.route.exits[self.next_index]

    @property
    def complete(self) -> bool:
        return self.next_index == len(self.route.exits)

    def rewind(self, frame: int) -> None:
        """Drop events at or after *frame* (checkpoint load)."""
        cut = int(frame)
        self.completed = [row for row in self.completed if int(row["frame"]) < cut]
        self.entries = {
            key: row for key, row in self.entries.items() if int(row["frame"]) < cut
        }
        self.deaths = [row for row in self.deaths if int(row["frame"]) < cut]
        self.off_route = [row for row in self.off_route if int(row["frame"]) < cut]
        self._was_dying = False

    def observe(self, snap: SmbSnapshot, *, frame: int) -> str | None:
        """Observe one post-action snapshot. Return event kind or None."""
        event: str | None = None
        if snap.player_state == PLAYER_STATE_DYING and not self._was_dying:
            self.deaths.append(
                {
                    "frame": frame,
                    "stage": stage_label_of(snap),
                    "x": snap.player_x,
                    "y": snap.player_y,
                    "lives": snap.lives,
                }
            )
            event = "death"
        self._was_dying = snap.player_state == PLAYER_STATE_DYING

        if self.complete:
            return event

        exit_seg = self.next_exit
        if exit_seg is None:
            return event

        if exit_seg.exit_id not in self.entries and level_control_gate(exit_seg).matches(snap):
            self.entries[exit_seg.exit_id] = {
                "frame": frame,
                "fingerprint": snapshot_fingerprint(snap),
            }
            event = "entry"

        if exit_seg.accepts_successor(snap):
            successor = next(
                destination.label or f"{destination.world}-{destination.level}"
                for destination in exit_seg.successors
                if destination.matches(snap)
            )
            self.completed.append(
                {
                    "exit_id": exit_seg.exit_id,
                    "frame": frame,
                    "successor": successor,
                    "fingerprint": snapshot_fingerprint(snap),
                }
            )
            return "exit"

        skipped = self._skipped_stage(snap)
        if skipped is not None:
            row = {
                "frame": frame,
                "stage": skipped,
                "expected": exit_seg.exit_id,
            }
            if not self.off_route or self.off_route[-1].get("stage") != skipped:
                self.off_route.append(row)
                event = event or "off_route"
        return event

    def _skipped_stage(self, snap: SmbSnapshot) -> str | None:
        if not snap.playing:
            return None
        current = stage_label_of(snap)
        ids = [exit_seg.exit_id for exit_seg in self.route.exits]
        if current not in ids:
            return current
        idx = ids.index(current)
        if idx > self.next_index:
            return current
        return None

    def report(self) -> dict[str, Any]:
        expected = [exit_seg.exit_id for exit_seg in self.route.exits[self.start_index :]]
        return {
            "route_id": self.route.route_id,
            "start_exit": self.route.exits[self.start_index].exit_id,
            "expected_exits": expected,
            "completed_exits": [row["exit_id"] for row in self.completed],
            "complete": self.complete,
            "deaths": list(self.deaths),
            "off_route": list(self.off_route),
            "entries": dict(self.entries),
            "splits": list(self.completed),
        }


def segments_dir_for(task_path: Path | str) -> Path:
    path = Path(task_path)
    return path.with_name(path.stem + "_segments")


def _next_segment_id(segments_dir: Path) -> int:
    if not segments_dir.is_dir():
        return 0
    ids = []
    for child in segments_dir.iterdir():
        if not child.is_dir():
            continue
        match = _SEG_DIR_RE.match(child.name)
        if match:
            ids.append(int(match.group(1)))
    return (max(ids) + 1) if ids else 0


def archive_existing_take(task_path: Path | str) -> Path | None:
    """Copy a previous tape into ``<name>_segments/sN/`` (pins stay)."""
    path = Path(task_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    frames = data.get("frames") or []
    if not frames:
        return None

    dest = segments_dir_for(path) / f"s{_next_segment_id(segments_dir_for(path))}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest / "tape.json")
    end_state = path.with_name(path.stem + "_end.state")
    if end_state.is_file():
        shutil.copy2(end_state, dest / "end.state")
    join = {
        "schemaVersion": 1,
        "kind": "smb_segment_join",
        "task_name": str(data.get("name") or path.stem),
        "archived_from": str(path.resolve()),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "start_state": data.get("start_state"),
        "frame_count": int(data.get("num_frames") or len(frames)),
        "route": (data.get("route") or (data.get("metadata") or {}).get("route")),
        "note": "Immutable button tape. Pins under <name>_pins/ stay live.",
    }
    (dest / "join.json").write_text(json.dumps(join, indent=2) + "\n", encoding="utf-8")
    return dest


def write_stage_pin(
    *,
    task_name: str,
    stage_id: str,
    state_bytes: bytes,
    snap: SmbSnapshot,
    frame: int,
    rta_frames: int,
    kind: str = "control",
    out_dir: Path | None = None,
) -> Path:
    """Write a durable stage pin + sidecar JSON (SM-style item seam)."""
    state_path = pin_state_path(task_name, stage_id, out_dir=out_dir)
    meta_path = pin_meta_path(task_name, stage_id, out_dir=out_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(state_bytes)
    payload = {
        "stage": stage_id,
        "kind": kind,
        "frame": frame,
        "rta_frames": int(rta_frames),
        "fingerprint": snapshot_fingerprint(snap),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    resume = pins_dir(task_name, out_dir=out_dir) / "resume.state"
    resume.write_bytes(state_bytes)
    return state_path


def load_pin_rta_offset(task_name: str, stage_id: str, *, out_dir: Path | None = None) -> int:
    meta = pin_meta_path(task_name, stage_id, out_dir=out_dir)
    if not meta.is_file():
        return 0
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, Mapping):
        return 0
    return int(data.get("rta_frames") or 0)
