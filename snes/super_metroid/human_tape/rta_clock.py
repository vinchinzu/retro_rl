"""Any% KPDR RTA clock: t=0 at first Ceres control (ordinary gameplay).

Segment free-records restart the local frame clock at each ``./play`` seam.
This module folds archived segment joins + the live tape into a single
full-run timer so HUD / [ROOM] / [ITEM] lines match speedrun any% timing
(first movement / control on Ceres Elevator), not title-menu wall time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import load_anchors_index, parse_room_id
from super_metroid.human_tape.hops import load_task_json
from super_metroid.human_tape.segment_archive import list_segment_ids, segments_dir_for

# Ceres Elevator Room — first ordinary control room after intro.
CERES_ELEVATOR_ROOM = 0xDF45


def fmt_time(frames: int) -> str:
    """60fps wall-time label: m:ss.mmm"""
    frames = max(0, int(frames))
    total = frames / 60.0
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


@dataclass
class RtaClockInfo:
    """How to map a live session frame → full-run RTA from Ceres."""

    # Frames to add to the *current session* local frame so t=0 is Ceres control.
    offset_frames: int = 0
    # Local frame in the power-on segment where Ceres control began (for notes).
    ceres_zero_local: int | None = None
    # Absolute full-run frame of the current segment's start pin (Ceres-zeroed).
    start_rta_frames: int = 0
    power_on_segment: str | None = None
    notes: list[str] = field(default_factory=list)

    def full_frames(self, local_frame: int) -> int:
        return max(0, int(self.offset_frames) + max(0, int(local_frame)))

    def full_time(self, local_frame: int) -> str:
        return fmt_time(self.full_frames(local_frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "super_metroid_rta_clock",
            "schemaVersion": 1,
            "offset_frames": int(self.offset_frames),
            "ceres_zero_local": self.ceres_zero_local,
            "start_rta_frames": int(self.start_rta_frames),
            "start_rta_time": fmt_time(self.start_rta_frames),
            "power_on_segment": self.power_on_segment,
            "notes": list(self.notes),
        }


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _anchor_rows(index: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(index, Mapping):
        return []
    raw = index.get("anchors")
    if not isinstance(raw, list):
        return []
    return [dict(r) for r in raw if isinstance(r, Mapping)]


def find_ceres_zero_frame(
    anchors: Sequence[Mapping[str, Any]] | None = None,
    *,
    rooms: Sequence[Mapping[str, Any]] | None = None,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> int | None:
    """First frame of Ceres Elevator ordinary control.

    Prefers live ``boot`` / ``room_enter`` on 0xDF45, then run_timing room
    entry, then first trace row in that room.
    """
    if anchors:
        best: int | None = None
        for row in anchors:
            rid = parse_room_id(row.get("room_id") if row.get("room_id") is not None else row.get("room"))
            if rid != CERES_ELEVATOR_ROOM:
                continue
            kind = str(row.get("kind") or "")
            fr = int(row.get("frame") or 0)
            if kind in ("boot", "room_enter", "enter"):
                if best is None or fr < best:
                    best = fr
        if best is not None:
            return best

    if rooms:
        for row in rooms:
            rid = parse_room_id(
                row.get("room_id") if row.get("room_id") is not None else row.get("room_id_hex")
            )
            if rid != CERES_ELEVATOR_ROOM:
                continue
            if row.get("entry_frame") is not None:
                return int(row["entry_frame"])
            if row.get("frame") is not None:
                return int(row["frame"])

    if trace:
        for row in trace:
            rid = parse_room_id(row.get("room") if row.get("room") is not None else row.get("room_id"))
            if rid == CERES_ELEVATOR_ROOM:
                return int(row.get("frame") or 0)
    return None


def _segment_end_frame(join: Mapping[str, Any] | None, tape: Mapping[str, Any] | None) -> int:
    if join:
        end_fp = join.get("end_fingerprint")
        if isinstance(end_fp, Mapping) and end_fp.get("frame") is not None:
            return int(end_fp["frame"])
        if join.get("frame_count") is not None:
            return max(0, int(join["frame_count"]) - 1)
    if tape:
        if tape.get("frame_count") is not None:
            return max(0, int(tape["frame_count"]) - 1)
        frames = tape.get("frames") or []
        if isinstance(frames, list) and frames:
            return len(frames) - 1
    return 0


def _seam_key(join: Mapping[str, Any], *, power_on: bool) -> tuple[Any, ...]:
    """Identity for a free-record seam so retakes of the same pin are deduped.

    Reusing ``./play supers`` archives the prior supers-end tape under a new
    sN each time. Without dedupe, RTA would sum those retakes (e.g. 3× bomb→
    supers) and the HUD clock would look ~11 min high.

    Key is start pin + end room/items (not frame length), so a faster retake
    of the same seam still replaces the older archive in the RTA chain.
    """
    end_fp = join.get("end_fingerprint") if isinstance(join.get("end_fingerprint"), Mapping) else {}
    end_fp = end_fp or {}
    end_room = end_fp.get("room_id")
    if end_room is None:
        end_room = end_fp.get("room")
    end_items = end_fp.get("items")
    start = str(join.get("start_state") or "")
    return (start, str(end_room), str(end_items), bool(power_on))


def load_archive_segments(
    task_path: Path | str,
    *,
    include_excluded: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load ``tasks/<name>_segments/sN/`` rows for RTA / PB board.

    Each row: sid, join, tape, anchors, rooms, trace, power_on, end_fr, seam,
    rta_exclude, timing path.

    *include_excluded*: when True, keep ``rta_exclude`` archives (for historical
    hop samples / avg). Product RTA chain leaves them out.
    """
    path = Path(task_path)
    notes: list[str] = []
    seg_root = segments_dir_for(path)
    rows: list[dict[str, Any]] = []
    for sid in list_segment_ids(seg_root):
        sdir = seg_root / f"s{sid}"
        join = _safe_json(sdir / "join.json") or {}
        tape = _safe_json(sdir / "tape.json")
        anchors_idx = _safe_json(sdir / "anchors.json")
        timing = _safe_json(sdir / "run_timing.json")
        power_on = bool(join.get("power_on")) or str(join.get("start_state") or "") in (
            "power_on",
            "start",
            "power-on",
            "beginning",
            "full",
            "poweron",
        )
        excluded = bool(join.get("rta_exclude"))
        if excluded and not include_excluded:
            notes.append(f"s{sid}: skipped (rta_exclude={join.get('reason') or True})")
            continue
        end_fr = _segment_end_frame(join, tape)
        rows.append(
            {
                "sid": sid,
                "join": join,
                "tape": tape,
                "anchors": _anchor_rows(anchors_idx),
                "rooms": timing.get("rooms")
                if timing and isinstance(timing.get("rooms"), list)
                else None,
                "trace": tape.get("trace")
                if tape and isinstance(tape.get("trace"), list)
                else None,
                "timing": timing,
                "power_on": power_on,
                "end_fr": end_fr,
                "seam": _seam_key(join, power_on=power_on),
                "rta_exclude": excluded,
                "source": f"s{sid}",
            }
        )
    return rows, notes


def product_chain_segments(
    task_path: Path | str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Latest power-on → … product chain with seam retakes deduped.

    Does not include the live task tape; caller folds that separately.
    """
    rows, notes = load_archive_segments(task_path, include_excluded=False)
    start_i = 0
    for i, row in enumerate(rows):
        if row["power_on"]:
            start_i = i
    if rows and not any(r["power_on"] for r in rows):
        notes.append("no power_on segment in archive — RTA may be incomplete")

    chain_rows: list[dict[str, Any]] = []
    seen_seams: set[tuple[Any, ...]] = set()
    for row in reversed(rows[start_i:]):
        seam = row.get("seam")
        if seam in seen_seams:
            notes.append(
                f"s{row['sid']}: skipped (retake of seam start={row['join'].get('start_state')!s})"
            )
            continue
        seen_seams.add(seam)  # type: ignore[arg-type]
        chain_rows.append(row)
    return list(reversed(chain_rows)), notes


def resolve_rta_clock(
    task_path: Path | str,
    *,
    include_live_tape: bool = False,
) -> RtaClockInfo:
    """Compute RTA offset for the next / current free-record segment.

    Uses ``tasks/<name>_segments/sN/`` archives (power-on first) plus optional
    live ``tasks/<name>.json`` when *include_live_tape* (e.g. post-hoc stitch).

    ``offset_frames`` is what a **new** session should add to its local frame
    so the HUD shows full-run time from Ceres. It equals prior segments'
    Ceres→end span (joined at each seam end pin).

    When the same seam is archived multiple times (retakes / cut-pause re-
    materialize), only the **latest** segment for that seam key is counted.
    """
    path = Path(task_path)
    info = RtaClockInfo()

    chain_rows, notes = product_chain_segments(path)

    ceres_zero: int | None = None
    power_seg: str | None = None
    chain_rta = 0

    for row in chain_rows:
        sid = int(row["sid"])
        end_fr = int(row["end_fr"])
        if row["power_on"] and ceres_zero is None:
            cz = find_ceres_zero_frame(
                row["anchors"], rooms=row["rooms"], trace=row["trace"]
            )
            if cz is not None:
                ceres_zero = cz
                power_seg = f"s{sid}"
                span = max(0, end_fr - cz)
                chain_rta += span
                notes.append(
                    f"s{sid} power_on: ceres_zero=f{cz} end=f{end_fr} "
                    f"span={fmt_time(span)} ({span}f)"
                )
            else:
                chain_rta += end_fr
                power_seg = f"s{sid}"
                notes.append(
                    f"s{sid} power_on: no Ceres zero; using full end=f{end_fr} "
                    f"(includes title/menu)"
                )
            continue

        # Later archived segments: full local length (already post-Ceres).
        chain_rta += end_fr
        notes.append(f"s{sid}: +{fmt_time(end_fr)} (f{end_fr}) end pin")

    if ceres_zero is None and path.is_file():
        # Live power-on tape only (no archives yet).
        live = load_task_json(path)
        meta = live.get("metadata") if isinstance(live.get("metadata"), Mapping) else {}
        meta = meta or {}
        if meta.get("power_on") or str(live.get("start_state") or "") == "power_on":
            anchors = _anchor_rows(load_anchors_index(path))
            timing = _safe_json(path.with_name(path.stem + "_run_timing.json"))
            rooms = timing.get("rooms") if timing else None
            cz = find_ceres_zero_frame(
                anchors,
                rooms=rooms if isinstance(rooms, list) else None,
                trace=live.get("trace") if isinstance(live.get("trace"), list) else None,
            )
            if cz is not None:
                ceres_zero = cz
                power_seg = "live"
                notes.append(f"live power_on ceres_zero=f{cz}")

    if include_live_tape and path.is_file():
        live = load_task_json(path)
        meta = live.get("metadata") if isinstance(live.get("metadata"), Mapping) else {}
        meta = meta or {}
        end_fr = _segment_end_frame(
            {
                "end_fingerprint": meta.get("end_fingerprint"),
                "frame_count": live.get("frame_count"),
            },
            live,
        )
        if meta.get("power_on") or str(live.get("start_state") or "") == "power_on":
            # Live is still the power-on take — offset for *continuing* is end-ceres.
            if ceres_zero is not None:
                span = max(0, end_fr - int(ceres_zero))
                chain_rta = span  # replace; live is the only power-on body
                notes.append(f"live power_on span to end: {fmt_time(span)} (f{span})")
            else:
                chain_rta = end_fr
                notes.append(f"live power_on full end=f{end_fr}")
        else:
            chain_rta += end_fr
            notes.append(f"live tape: +{fmt_time(end_fr)} (f{end_fr})")

    info.ceres_zero_local = ceres_zero
    info.power_on_segment = power_seg
    info.offset_frames = int(chain_rta)
    info.start_rta_frames = int(chain_rta)
    info.notes = notes
    if ceres_zero is None:
        info.notes.append(
            "no Ceres Elevator pin found — RTA offset may include title/menu "
            "or be 0 until a power-on segment is archived"
        )
    return info
