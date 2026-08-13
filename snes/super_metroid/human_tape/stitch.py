"""Stitch multi-session human takes via live anchors + print PB tables.

When a long free-record is cancelled and resumed from a live pin, the second
take's frame clock restarts at 0 while the first session's gzip anchors remain
on disk as **orphans** (not in the current ``*_anchors.json`` index).

This module:

1. Discovers orphan prefix anchors in the anchors dir
2. Joins them to the current take at the shared pin room (e.g. Big Pink enter)
3. Emits a stitched timeline with full-run frames + a printable PB table

Timing only (RTA clock). Button replay across seams is hop-compose
(``human_tape.compose``): pin → body → leave pin — not frame-append.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import (
    load_anchors_index,
    parse_items_value,
    parse_room_id,
)
from super_metroid.human_tape.hops import load_task_json
from super_metroid.rooms.canonical_names import load_canonical_names, room_name

# Major collected-items bits for milestone labels (see ram.py masks).
_ITEM_BIT_NAMES: tuple[tuple[int, str], ...] = (
    (0x0001, "Varia"),
    (0x0002, "SpringBall"),
    (0x0004, "Morph"),
    (0x0008, "ScrewAttack"),
    (0x0020, "Gravity"),
    (0x0100, "HiJump"),
    (0x0200, "SpaceJump"),
    (0x1000, "Bombs"),
    (0x2000, "SpeedBooster"),
    (0x4000, "Grapple"),
    (0x8000, "XRay"),
)

# Named route milestones (first time room_id is entered / item bit set).
_ROOM_MILESTONES: tuple[tuple[int, str], ...] = (
    (0xDF45, "Ceres Elevator"),
    (0x91F8, "Landing Site"),
    (0x9E9F, "Morph Ball Room"),
    (0x9804, "Bomb Torizo"),
    (0x9D19, "Big Pink"),
    (0x9B5B, "Spore Super"),
    (0xA253, "Red Tower"),
    (0xA447, "Spazer Room"),
    (0xA7DE, "Business Center"),
    (0xA9E5, "Hi Jump Boots Room"),
    (0xA59F, "Kraid"),
    (0xA6E2, "Varia Suit Room"),
    (0xAD1B, "Speed Booster Room"),
    (0xADDE, "Wave Beam Room"),
    (0xA890, "Ice Beam Room"),
    (0xA3AE, "Alpha Power Bomb Room"),
    (0x95FF, "The Moat"),
    (0xCA08, "Wrecked Ship Entrance"),
    (0xCD13, "Phantoon's Room"),
    (0xCE40, "Gravity Suit Room"),
    (0xAC2B, "Grapple Beam Room"),
    (0xCFC9, "Main Street"),
    (0xD95E, "Botwoon's Room"),
    (0xDA60, "Draygon's Room"),
    (0xD9AA, "Space Jump Room"),
    (0xD2AA, "Plasma Beam Room"),
    (0xB283, "Golden Torizo's Room"),
    (0xB6C1, "Screw Attack Room"),
    (0xB62B, "Metal Pirates Room"),
    (0xB32E, "Ridley's Room"),
    (0xA66A, "Statues Room"),
    (0xDAAE, "Tourian First Room"),
    (0xDD58, "Mother Brain Room"),
)


@dataclass
class StitchedEvent:
    """One event on the stitched full-run clock."""

    frame: int
    kind: str
    room_id: int
    name: str
    items: int | None = None
    label: str = ""
    segment: str = ""
    local_frame: int = 0
    xy: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "kind": self.kind,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "name": self.name,
            "items": f"0x{self.items:04X}" if self.items is not None else None,
            "label": self.label,
            "segment": self.segment,
            "local_frame": self.local_frame,
            "xy": list(self.xy) if self.xy else None,
        }


@dataclass
class StitchReport:
    """Stitched full-run timing from one or more anchor sessions."""

    task: str
    join_room_id: int | None
    join_frame: int
    prefix_events: int
    take_events: int
    events: list[StitchedEvent] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    room_splits: list[dict[str, Any]] = field(default_factory=list)
    total_frames: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "super_metroid_stitched_run",
            "schemaVersion": 1,
            "task": self.task,
            "join_room_id": self.join_room_id,
            "join_room_hex": (
                f"0x{self.join_room_id:04X}" if self.join_room_id is not None else None
            ),
            "join_frame": self.join_frame,
            "prefix_events": self.prefix_events,
            "take_events": self.take_events,
            "total_frames": self.total_frames,
            "total_time": fmt_time(self.total_frames),
            "milestones": list(self.milestones),
            "room_splits": list(self.room_splits),
            "events": [e.to_dict() for e in self.events],
            "notes": list(self.notes),
        }


def fmt_time(frames: int) -> str:
    """60fps clock: ``m:ss.mmm``."""
    frames = max(0, int(frames))
    total = frames / 60.0
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def item_names(mask: int | None) -> list[str]:
    if mask is None:
        return []
    return [name for bit, name in _ITEM_BIT_NAMES if int(mask) & bit]


def item_delta_label(prev: int | None, cur: int | None) -> str:
    if cur is None:
        return "item"
    if prev is None:
        names = item_names(cur)
        return "+".join(names) if names else f"items=0x{cur:04X}"
    gained = int(cur) & ~int(prev)
    names = item_names(gained)
    if names:
        return "+".join(names)
    return f"items=0x{prev:04X}→0x{cur:04X}"


def _parse_state_filename(name: str) -> dict[str, Any] | None:
    m = re.match(r"f(\d+)_(.+)\.state$", name)
    if not m:
        return None
    frame = int(m.group(1))
    rest = m.group(2)
    kind_raw = rest.split("_", 1)[0]
    kind_map = {
        "boot": "boot",
        "enter": "room_enter",
        "items": "item_delta",
        "end": "end",
        "manual": "manual",
    }
    kind = kind_map.get(kind_raw, kind_raw)
    hexes = re.findall(r"0x([0-9A-Fa-f]{4})", rest)
    room_id = int(hexes[-1], 16) if hexes else 0
    items = None
    if kind == "item_delta" and len(hexes) >= 2:
        items = int(hexes[0], 16)
        room_id = int(hexes[1], 16)
    return {
        "kind": kind,
        "frame": frame,
        "room_id": room_id,
        "room": f"0x{room_id:04X}",
        "items": f"0x{items:04X}" if items is not None else None,
        "path": name,
    }


def list_anchor_state_files(anchors_dir: Path) -> list[dict[str, Any]]:
    """All gzip anchors under dir. Each row has ``mtime`` + parsed fields."""
    rows: list[dict[str, Any]] = []
    if not anchors_dir.is_dir():
        return rows
    for path in anchors_dir.glob("f*.state"):
        parsed = _parse_state_filename(path.name)
        if parsed is None:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        parsed["path"] = str(path.resolve())
        parsed["mtime"] = float(mtime)
        rows.append(parsed)
    return rows


def orphan_prefix_anchors(
    anchors_dir: Path,
    index_anchors: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """State files on disk not listed in the current take's anchors index."""
    indexed_names: set[str] = set()
    for row in index_anchors or []:
        p = row.get("path")
        if p:
            indexed_names.add(Path(str(p)).name)
    orphans: list[dict[str, Any]] = []
    for row in list_anchor_state_files(anchors_dir):
        if Path(str(row["path"])).name in indexed_names:
            continue
        orphans.append(row)
    orphans.sort(key=lambda r: (float(r.get("mtime") or 0), int(r.get("frame") or 0)))
    return orphans


def _normalize_index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    room_id = parse_room_id(row.get("room_id") if row.get("room_id") is not None else row.get("room")) or 0
    items = parse_items_value(row.get("items"))
    kind = str(row.get("kind") or "pin")
    if kind == "room_enter":
        kind = "room_enter"
    return {
        "kind": kind,
        "frame": int(row.get("frame") or 0),
        "room_id": int(room_id),
        "room": f"0x{int(room_id):04X}",
        "items": f"0x{items:04X}" if items is not None else None,
        "xy": row.get("xy"),
        "path": row.get("path"),
        "label": row.get("label"),
        "mtime": float(row.get("mtime") or 0),
    }


def _to_stitched(
    row: Mapping[str, Any],
    *,
    offset: int,
    segment: str,
    names: Mapping[int, str],
) -> StitchedEvent:
    room_id = int(row.get("room_id") or 0)
    items = parse_items_value(row.get("items"))
    local = int(row.get("frame") or 0)
    xy = row.get("xy")
    xy_list = [int(xy[0]), int(xy[1])] if isinstance(xy, (list, tuple)) and len(xy) >= 2 else None
    return StitchedEvent(
        frame=offset + local,
        kind=str(row.get("kind") or "pin"),
        room_id=room_id,
        name=room_name(room_id, names=names),
        items=items,
        label=str(row.get("label") or ""),
        segment=segment,
        local_frame=local,
        xy=xy_list,
    )


def sessions_from_boots(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split anchor files into record sessions by boot + mtime windows.

    Never sorts the whole dir by frame alone — that interleaves cancelled
    multi-hour free-records that each restarted their local frame clock.
    """
    files = [dict(r) for r in rows]
    if not files:
        return []
    boots = sorted(
        [r for r in files if str(r.get("kind") or "") == "boot"],
        key=lambda r: (float(r.get("mtime") or 0), int(r.get("frame") or 0)),
    )
    if not boots:
        # Single blob — sort by frame only.
        files.sort(key=lambda r: (int(r.get("frame") or 0), str(r.get("kind") or "")))
        return [files]

    sessions: list[list[dict[str, Any]]] = []
    for i, boot in enumerate(boots):
        t0 = float(boot.get("mtime") or 0)
        t1 = (
            float(boots[i + 1].get("mtime") or 0)
            if i + 1 < len(boots)
            else float("inf")
        )
        members = [
            r
            for r in files
            if t0 <= float(r.get("mtime") or 0) < t1
        ]
        # Within a session the local frame clock is meaningful.
        members.sort(
            key=lambda r: (
                int(r.get("frame") or 0),
                {"boot": 0, "room_enter": 1, "enter": 1, "item_delta": 2, "end": 3}.get(
                    str(r.get("kind") or ""), 9
                ),
            )
        )
        if members:
            sessions.append(members)
    return sessions


def session_join_frame(
    session: Sequence[Mapping[str, Any]],
    next_boot_room: int,
) -> int:
    """Local frame in ``session`` where the next take resumes.

    Prefer last item_delta in the join room, then end, then room_enter/boot.
    """
    best_item: int | None = None
    best_end: int | None = None
    best_enter: int | None = None
    for row in session:
        if int(row.get("room_id") or 0) != int(next_boot_room):
            continue
        kind = str(row.get("kind") or "")
        fr = int(row.get("frame") or 0)
        if kind == "item_delta":
            best_item = fr
        elif kind == "end":
            best_end = fr
        elif kind in ("room_enter", "boot", "enter"):
            best_enter = fr
    if best_item is not None:
        return best_item
    if best_end is not None:
        return best_end
    if best_enter is not None:
        return best_enter
    if session:
        return max(int(r.get("frame") or 0) for r in session)
    return 0


def trim_session_at_join(
    session: Sequence[Mapping[str, Any]],
    join_frame: int,
) -> list[dict[str, Any]]:
    """Keep events with frame <= join_frame (drop post-handoff / rewind noise after pin)."""
    return [dict(r) for r in session if int(r.get("frame") or 0) <= int(join_frame)]


def find_join(
    prefix: Sequence[Mapping[str, Any]],
    take: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int, int]:
    """Return (join_room_id, prefix_join_frame, take_join_local_frame).

    Prefers the last prefix room_enter whose room matches the take boot/first
    room_enter (resume pin).
    """
    if not take:
        last = prefix[-1] if prefix else None
        if last is None:
            return None, 0, 0
        return int(last.get("room_id") or 0), int(last.get("frame") or 0), 0

    take0 = take[0]
    take_room = int(take0.get("room_id") or 0)
    join_frame = session_join_frame(prefix, take_room) if prefix else 0
    if not prefix:
        join_frame = 0
    return take_room or None, int(join_frame), 0


def chain_sessions(
    sessions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    names: Mapping[int, str],
) -> tuple[list[StitchedEvent], int, int | None, list[str]]:
    """Chain multi-session records onto one full-run clock.

    Returns (events, total_join_offset_to_last, first_join_room, notes).
    """
    notes: list[str] = []
    events: list[StitchedEvent] = []
    offset = 0
    first_join_room: int | None = None
    kind_rank = {"boot": 0, "room_enter": 1, "enter": 1, "item_delta": 2, "manual": 3, "end": 4}

    for i, sess in enumerate(sessions):
        if not sess:
            continue
        seg_name = f"s{i}"
        nxt = sessions[i + 1] if i + 1 < len(sessions) else None
        if nxt:
            next_room = int(nxt[0].get("room_id") or 0)
            join_fr = session_join_frame(sess, next_room)
            if first_join_room is None:
                first_join_room = next_room
            trimmed = trim_session_at_join(sess, join_fr)
            notes.append(
                f"{seg_name}: {len(trimmed)} events → join "
                f"0x{next_room:04X} @ local f{join_fr} ({fmt_time(join_fr)}) "
                f"full={fmt_time(offset + join_fr)}"
            )
            for row in trimmed:
                # Skip trailing boot-only if identical to join enter
                events.append(
                    _to_stitched(row, offset=offset, segment=seg_name, names=names)
                )
            offset += join_fr
        else:
            # Final take
            notes.append(
                f"{seg_name}: final take {len(sess)} events offset={fmt_time(offset)}"
            )
            for j, row in enumerate(sess):
                if (
                    j == 0
                    and str(row.get("kind") or "") == "boot"
                    and int(row.get("frame") or 0) == 0
                    and events
                    and int(row.get("room_id") or 0) == events[-1].room_id
                ):
                    continue
                events.append(
                    _to_stitched(row, offset=offset, segment=seg_name, names=names)
                )

    events.sort(key=lambda e: (e.frame, kind_rank.get(e.kind, 9), e.local_frame))
    # Dedup seam boots / double enters
    deduped: list[StitchedEvent] = []
    seen: set[tuple[int, str, int]] = set()
    for ev in events:
        key = (ev.frame, ev.kind, ev.room_id)
        if key in seen and ev.kind in ("boot", "room_enter", "enter"):
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped, offset, first_join_room, notes


def build_milestones(events: Sequence[StitchedEvent]) -> list[dict[str, Any]]:
    """First-hit route milestones + every item_delta."""
    seen_rooms: set[int] = set()
    rows: list[dict[str, Any]] = []
    prev_items: int | None = None
    room_want = {rid: label for rid, label in _ROOM_MILESTONES}

    # Power-on / first event
    if events:
        e0 = events[0]
        rows.append(
            {
                "frame": e0.frame,
                "time": fmt_time(e0.frame),
                "split": fmt_time(0),
                "kind": "start",
                "label": f"start ({e0.segment or 'take'})",
                "room": e0.name,
                "room_id_hex": f"0x{e0.room_id:04X}",
                "items": f"0x{e0.items:04X}" if e0.items is not None else None,
            }
        )
        prev_items = e0.items

    last_frame = events[0].frame if events else 0
    for ev in events:
        if ev.kind in ("room_enter", "boot") and ev.room_id in room_want and ev.room_id not in seen_rooms:
            seen_rooms.add(ev.room_id)
            rows.append(
                {
                    "frame": ev.frame,
                    "time": fmt_time(ev.frame),
                    "split": fmt_time(ev.frame - last_frame),
                    "kind": "room",
                    "label": room_want[ev.room_id],
                    "room": ev.name,
                    "room_id_hex": f"0x{ev.room_id:04X}",
                    "items": f"0x{ev.items:04X}" if ev.items is not None else None,
                }
            )
            last_frame = ev.frame
        if ev.kind == "item_delta":
            label = item_delta_label(prev_items, ev.items)
            rows.append(
                {
                    "frame": ev.frame,
                    "time": fmt_time(ev.frame),
                    "split": fmt_time(ev.frame - last_frame),
                    "kind": "item",
                    "label": label,
                    "room": ev.name,
                    "room_id_hex": f"0x{ev.room_id:04X}",
                    "items": f"0x{ev.items:04X}" if ev.items is not None else None,
                }
            )
            prev_items = ev.items if ev.items is not None else prev_items
            last_frame = ev.frame
        if ev.kind == "end":
            rows.append(
                {
                    "frame": ev.frame,
                    "time": fmt_time(ev.frame),
                    "split": fmt_time(ev.frame - last_frame),
                    "kind": "end",
                    "label": "end",
                    "room": ev.name,
                    "room_id_hex": f"0x{ev.room_id:04X}",
                    "items": f"0x{ev.items:04X}" if ev.items is not None else None,
                }
            )
            last_frame = ev.frame
    return rows


def build_room_splits(events: Sequence[StitchedEvent]) -> list[dict[str, Any]]:
    """Consecutive room_enter/boot → dwell until next enter."""
    enters = [e for e in events if e.kind in ("room_enter", "boot")]
    rows: list[dict[str, Any]] = []
    for i, ev in enumerate(enters):
        nxt = enters[i + 1] if i + 1 < len(enters) else None
        leave = nxt.frame if nxt is not None else (events[-1].frame if events else ev.frame)
        dwell = max(0, leave - ev.frame)
        rows.append(
            {
                "index": i,
                "room_id": ev.room_id,
                "room_id_hex": f"0x{ev.room_id:04X}",
                "name": ev.name,
                "entry_frame": ev.frame,
                "leave_frame": leave,
                "dwell_frames": dwell,
                "dwell_time": fmt_time(dwell),
                "time": fmt_time(ev.frame),
                "segment": ev.segment,
                "dest": nxt.name if nxt is not None else None,
            }
        )
    return rows


def stitch_task_anchors(
    task_path: Path | str,
    *,
    anchors_dir: Path | str | None = None,
    names: Mapping[int, str] | None = None,
) -> StitchReport:
    """Stitch multi-session anchors onto one full-run clock.

    Sessions are split by **boot file mtime** (each ``./play`` resume starts a
    new boot). Within a session, local frames are ordered. Sessions are chained
    at the resume pin room so cancelled+resumed free-records accumulate RTA
    from the beginning instead of interleaving by raw frame number.
    """
    path = Path(task_path)
    name_table = names if names is not None else load_canonical_names()
    data = load_task_json(path) if path.is_file() else {}
    task_name = str(data.get("name") or path.stem)

    index = load_anchors_index(path)
    index_rows_raw: list[Mapping[str, Any]] = []
    if isinstance(index, Mapping):
        raw = index.get("anchors")
        if isinstance(raw, list):
            index_rows_raw = [r for r in raw if isinstance(r, Mapping)]
    take_rows = [_normalize_index_row(r) for r in index_rows_raw]

    if anchors_dir is not None:
        adir = Path(anchors_dir)
    else:
        adir = path.with_name(path.stem + "_anchors")
        if isinstance(index, Mapping) and index.get("anchors_dir"):
            adir = Path(str(index["anchors_dir"]))

    disk_rows = list_anchor_state_files(adir)
    # Prefer index rows for the latest take (has items/xy); map by basename.
    index_by_name: dict[str, dict[str, Any]] = {}
    for row in take_rows:
        p = row.get("path")
        if p:
            index_by_name[Path(str(p)).name] = dict(row)

    merged: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row in disk_rows:
        name = Path(str(row["path"])).name
        seen_names.add(name)
        if name in index_by_name:
            # Keep mtime from disk for session windows; content from index.
            rich = dict(index_by_name[name])
            rich["mtime"] = row.get("mtime", 0)
            rich["path"] = row.get("path")
            merged.append(rich)
        else:
            merged.append(dict(row))
    # Index-only rows (path missing on disk) — attach high mtime so they land last.
    for name, row in index_by_name.items():
        if name not in seen_names:
            rich = dict(row)
            rich["mtime"] = rich.get("mtime") or 1e18
            merged.append(rich)

    sessions = sessions_from_boots(merged)
    notes: list[str] = [f"sessions={len(sessions)} (boot+mtime windows)"]
    if not sessions and take_rows:
        sessions = [take_rows]
        notes.append("fallback: index-only single session")

    events, chain_offset, first_join_room, chain_notes = chain_sessions(
        sessions, names=name_table
    )
    notes.extend(chain_notes)

    # Forward-fill items so post-Varia rooms show 0x1105 etc.
    last_items: int | None = None
    filled: list[StitchedEvent] = []
    for ev in events:
        if ev.items is not None:
            last_items = ev.items
        elif last_items is not None and ev.kind in ("room_enter", "boot", "end"):
            ev = StitchedEvent(
                frame=ev.frame,
                kind=ev.kind,
                room_id=ev.room_id,
                name=ev.name,
                items=last_items,
                label=ev.label,
                segment=ev.segment,
                local_frame=ev.local_frame,
                xy=ev.xy,
            )
        filled.append(ev)
    events = filled

    milestones = build_milestones(events)
    room_splits = build_room_splits(events)
    total = events[-1].frame if events else 0
    for ev in reversed(events):
        if ev.kind in ("end", "item_delta"):
            total = ev.frame
            break

    # Report join as first seam (for header); full chain in notes.
    join_frame = 0
    join_room = first_join_room
    if len(sessions) >= 2:
        join_room = int(sessions[1][0].get("room_id") or 0) if sessions[1] else join_room
        join_frame = session_join_frame(sessions[0], int(join_room or 0))

    prefix_n = sum(len(s) for s in sessions[:-1]) if len(sessions) > 1 else 0
    take_n = len(sessions[-1]) if sessions else 0
    notes.append(f"full-run clock total_frames={total} ({fmt_time(total)})")
    if last_items is not None:
        notes.append(f"inventory fill last_items=0x{last_items:04X}")

    return StitchReport(
        task=task_name,
        join_room_id=int(join_room) if join_room is not None else None,
        join_frame=int(join_frame),
        prefix_events=prefix_n,
        take_events=take_n,
        events=events,
        milestones=milestones,
        room_splits=room_splits,
        total_frames=int(total),
        notes=notes,
    )


def format_pb_table(report: StitchReport, *, max_rooms: int | None = None) -> str:
    """Human-readable PB / full-run table for terminal dump."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"FULL RUN PB TABLE  ·  {report.task}")
    lines.append(
        f"total {fmt_time(report.total_frames)}  ({report.total_frames}f)  "
        f"join=0x{report.join_room_id:04X}@{fmt_time(report.join_frame)}"
        if report.join_room_id is not None
        else f"total {fmt_time(report.total_frames)}  ({report.total_frames}f)"
    )
    for note in report.notes:
        lines.append(f"  · {note}")
    lines.append("-" * 72)
    lines.append(f"{'TIME':>10}  {'SPLIT':>10}  {'KIND':<6}  MILESTONE")
    lines.append("-" * 72)
    for m in report.milestones:
        items = m.get("items") or ""
        item_s = f"  {items}" if items else ""
        lines.append(
            f"{m['time']:>10}  {m['split']:>10}  {m['kind']:<6}  "
            f"{m['label']:<22} {m.get('room') or ''}{item_s}"
        )
    lines.append("-" * 72)
    lines.append("ROOM SPLITS (entry → next entry)")
    lines.append(f"{'TIME':>10}  {'DWELL':>10}  ROOM → NEXT")
    lines.append("-" * 72)
    splits = report.room_splits
    if max_rooms is not None:
        splits = splits[: max_rooms]
    for r in splits:
        dest = r.get("dest") or "—"
        lines.append(
            f"{r['time']:>10}  {r['dwell_time']:>10}  "
            f"{r['name']} → {dest}"
        )
    if max_rooms is not None and len(report.room_splits) > max_rooms:
        lines.append(f"  … {len(report.room_splits) - max_rooms} more rooms")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_stitch_report(report: StitchReport, out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def rezero_report_to_ceres(report: StitchReport) -> StitchReport:
    """Shift stitched frames so t=0 is first Ceres Elevator control (any% RTA)."""
    ceres_frame: int | None = None
    for ev in report.events:
        if ev.room_id == 0xDF45 and ev.kind in ("boot", "room_enter", "enter", "start"):
            ceres_frame = int(ev.frame)
            break
    if ceres_frame is None:
        for m in report.milestones:
            if str(m.get("room_id_hex") or "") == "0xDF45" or "Ceres" in str(
                m.get("label") or ""
            ):
                ceres_frame = int(m.get("frame") or 0)
                break
    if ceres_frame is None or ceres_frame <= 0:
        report.notes.append("ceres rezero skipped (no Ceres Elevator event)")
        return report

    z = int(ceres_frame)

    def _shift_ev(ev: StitchedEvent) -> StitchedEvent:
        return StitchedEvent(
            frame=max(0, int(ev.frame) - z),
            kind=ev.kind,
            room_id=ev.room_id,
            name=ev.name,
            items=ev.items,
            label=ev.label,
            segment=ev.segment,
            local_frame=ev.local_frame,
            xy=ev.xy,
        )

    report.events = [_shift_ev(e) for e in report.events]
    report.total_frames = max(0, int(report.total_frames) - z)
    report.join_frame = max(0, int(report.join_frame) - z)
    # Rebuild milestone/split times from shifted events.
    report.milestones = build_milestones(report.events)
    report.room_splits = build_room_splits(report.events)
    report.notes.append(
        f"ceres rezero: subtracted f{z} ({fmt_time(z)}) — any% from first Ceres control"
    )
    return report


def materialize_stitch(
    task_path: Path | str,
    *,
    write: bool = True,
    print_table: bool = True,
    max_rooms: int | None = None,
    ceres_zero: bool = True,
) -> StitchReport:
    """Stitch + optional write ``*_stitched.json`` + print PB table.

    When *ceres_zero* is True (default), shift the full-run clock so t=0 is
    first Ceres Elevator control (any% KPDR RTA), not title/menu wall time.
    """
    path = Path(task_path)
    report = stitch_task_anchors(path)
    if ceres_zero:
        report = rezero_report_to_ceres(report)
    if write:
        out = path.with_name(path.stem + "_stitched.json")
        write_stitch_report(report, out)
        report.notes.append(f"wrote {out}")
    if print_table:
        print(format_pb_table(report, max_rooms=max_rooms), flush=True)
    return report
