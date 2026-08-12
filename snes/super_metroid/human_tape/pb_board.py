"""Persistent room PB board: all-run samples with PB / avg / sd / Δ.

``./play --pb`` folds every archived segment + the live take into
``tasks/<name>_pb_board.json``. Re-runs **merge** (sample ids are stable per
source segment + hop_key + dwell) so history survives retakes.

Product-line table (absolute RTA) uses the same seam-deduped chain as
``rta_clock`` (latest supers pin, not triple-counted retakes). Stats columns
use **all** historical samples for that hop_key (including rta_exclude and
older retakes).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.hops import load_task_json
from super_metroid.human_tape.rta_clock import (
    find_ceres_zero_frame,
    fmt_time,
    load_archive_segments,
    product_chain_segments,
    resolve_rta_clock,
)
from super_metroid.rooms.canonical_names import room_name
from super_metroid.hop_id import make_hop_key
from super_metroid.human_tape.anchors import parse_items_value as parse_items

BOARD_SCHEMA = 1
BOARD_KIND = "super_metroid_pb_board"

# First-entry milestone labels for the product-line summary.
_MILESTONES: tuple[tuple[int, str], ...] = (
    (0xDF45, "Ceres Elevator"),
    (0x91F8, "Landing Site"),
    (0x9E9F, "Morph Ball Room"),
    (0x9804, "Bomb Torizo"),
    (0x9D19, "Big Pink"),
    (0x9B5B, "Spore Super"),
    (0xA253, "Red Tower"),
    (0xA447, "Spazer Room"),
    (0xA7DE, "Business Center"),
    (0xA9E5, "Hi Jump"),
    (0xA59F, "Kraid"),
    (0xA6E2, "Varia"),
)


def pb_board_path(task_path: Path | str) -> Path:
    path = Path(task_path)
    return path.with_name(path.stem + "_pb_board.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n-1); 0 when n < 2."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _dest_name(dest_id: int | None) -> str:
    if dest_id is None:
        return "—"
    try:
        return room_name(int(dest_id))
    except Exception:
        return f"0x{int(dest_id):04X}"


@dataclass
class HopStats:
    hop_key: str
    name: str
    room_id: int
    n: int
    pb: int
    avg: float
    sd: float
    last: int
    samples: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_key": self.hop_key,
            "name": self.name,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "n": self.n,
            "pb": self.pb,
            "pb_time": fmt_time(self.pb),
            "avg": round(self.avg, 2),
            "avg_time": fmt_time(int(round(self.avg))),
            "sd": round(self.sd, 2),
            "sd_time": fmt_time(int(round(self.sd))),
            "last": self.last,
            "last_time": fmt_time(self.last),
        }


@dataclass
class PbBoard:
    """Persistent multi-run hop sample store."""

    task: str
    hops: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BOARD_SCHEMA,
            "kind": BOARD_KIND,
            "task": self.task,
            "updated_at": self.updated_at or _utc_now(),
            "hop_count": len(self.hops),
            "sample_count": sum(
                len((h.get("samples") or [])) for h in self.hops.values()
            ),
            "run_count": len(self.runs),
            "hops": self.hops,
            "runs": self.runs,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PbBoard:
        hops_raw = data.get("hops") if isinstance(data.get("hops"), Mapping) else {}
        hops = {str(k): dict(v) for k, v in hops_raw.items() if isinstance(v, Mapping)}
        runs = [dict(r) for r in (data.get("runs") or []) if isinstance(r, Mapping)]
        return cls(
            task=str(data.get("task") or ""),
            hops=hops,
            runs=runs,
            notes=[str(n) for n in (data.get("notes") or [])],
            updated_at=str(data.get("updated_at") or ""),
        )

    @classmethod
    def load(cls, path: Path | str) -> PbBoard:
        p = Path(path)
        data = _safe_json(p)
        if not data:
            return cls(task=p.stem.replace("_pb_board", "") if p.stem else "run")
        return cls.from_dict(data)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    def add_sample(
        self,
        *,
        hop_key: str,
        room_id: int,
        name: str,
        dwell: int,
        source: str,
        room_frames: int | None = None,
        dest_room_id: int | None = None,
        sample_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append one hop sample. Returns True if new (False if id already present)."""
        dwell = int(dwell)
        if dwell < 0:
            return False
        sid = sample_id or f"{source}:{hop_key}:{dwell}"
        bucket = self.hops.setdefault(
            hop_key,
            {
                "hop_key": hop_key,
                "room_id": int(room_id),
                "room_id_hex": f"0x{int(room_id):04X}",
                "name": name,
                "samples": [],
            },
        )
        samples = bucket.setdefault("samples", [])
        for row in samples:
            if str(row.get("id")) == sid:
                return False
        samples.append(
            {
                "id": sid,
                "source": source,
                "dwell": dwell,
                "room_frames": int(room_frames) if room_frames is not None else dwell,
                "dest_room_id": int(dest_room_id) if dest_room_id is not None else None,
                "ingested_at": _utc_now(),
                **(dict(meta) if meta else {}),
            }
        )
        # Keep name fresh from latest ingest
        bucket["name"] = name
        bucket["room_id"] = int(room_id)
        return True

    def add_run(
        self,
        *,
        run_id: str,
        total_frames: int,
        segments: Sequence[str],
        milestones: Mapping[str, int] | None = None,
    ) -> bool:
        for row in self.runs:
            if str(row.get("id")) == run_id:
                return False
        self.runs.append(
            {
                "id": run_id,
                "total_frames": int(total_frames),
                "total_time": fmt_time(int(total_frames)),
                "segments": list(segments),
                "milestones": dict(milestones or {}),
                "ingested_at": _utc_now(),
            }
        )
        return True

    def hop_stats(self, hop_key: str) -> HopStats | None:
        bucket = self.hops.get(hop_key)
        if not bucket:
            return None
        samples = [int(s["dwell"]) for s in (bucket.get("samples") or []) if "dwell" in s]
        if not samples:
            return None
        return HopStats(
            hop_key=hop_key,
            name=str(bucket.get("name") or hop_key),
            room_id=int(bucket.get("room_id") or 0),
            n=len(samples),
            pb=min(samples),
            avg=_mean(samples),
            sd=_stdev(samples),
            last=samples[-1],
            samples=samples,
        )

    def stats_by_room(self, room_id: int) -> HopStats | None:
        """Best-effort: pick hop_key with most samples for room (or lowest PB)."""
        best: HopStats | None = None
        for key, bucket in self.hops.items():
            if int(bucket.get("room_id") or 0) != int(room_id):
                continue
            st = self.hop_stats(key)
            if st is None:
                continue
            if best is None or st.n > best.n or (st.n == best.n and st.pb < best.pb):
                best = st
        return best


def rooms_to_hop_samples(
    rooms: Sequence[Mapping[str, Any]],
    *,
    source: str,
    items: int | None = None,
) -> list[dict[str, Any]]:
    """Turn run_timing room leaves into hop sample dicts."""
    out: list[dict[str, Any]] = []
    prev: int | None = None
    for i, room in enumerate(rooms):
        rid = int(room.get("room_id") or 0)
        dest = room.get("dest_room_id")
        dest_i = int(dest) if dest is not None else None
        dwell = int(room.get("dwell_frames") or room.get("room_frames") or 0)
        rframes = int(room.get("room_frames") or dwell)
        leaf_items = parse_items(room.get("items"))
        if leaf_items is None:
            leaf_items = items
        key = make_hop_key(
            rid,
            from_room_id=prev,
            to_room_id=dest_i,
            items=leaf_items,
        )
        name = str(room.get("name") or room_name(rid))
        out.append(
            {
                "hop_key": key,
                "room_id": rid,
                "name": name,
                "dwell": dwell,
                "room_frames": rframes,
                "dest_room_id": dest_i,
                "source": source,
                "sample_id": f"{source}:{i}:{key}:{dwell}",
                "index": i,
                "entry_frame": int(room.get("entry_frame") or 0),
            }
        )
        prev = rid
    return out


def ingest_rooms(
    board: PbBoard,
    rooms: Sequence[Mapping[str, Any]],
    *,
    source: str,
    items: int | None = None,
) -> int:
    """Ingest one take's room leaves. Returns number of new samples."""
    added = 0
    for sample in rooms_to_hop_samples(rooms, source=source, items=items):
        if board.add_sample(
            hop_key=sample["hop_key"],
            room_id=sample["room_id"],
            name=sample["name"],
            dwell=sample["dwell"],
            source=source,
            room_frames=sample["room_frames"],
            dest_room_id=sample["dest_room_id"],
            sample_id=sample["sample_id"],
        ):
            added += 1
    return added


def _end_items_from_join(join: Mapping[str, Any] | None) -> int | None:
    if not join:
        return None
    end_fp = join.get("end_fingerprint")
    if isinstance(end_fp, Mapping):
        return parse_items(end_fp.get("items"))
    return None


def ingest_task_archives(board: PbBoard, task_path: Path | str) -> dict[str, int]:
    """Ingest every archived segment (including rta_exclude) + live run_timing."""
    path = Path(task_path)
    stats = {"segments": 0, "samples": 0, "live_samples": 0, "skipped_no_timing": 0}

    rows, _notes = load_archive_segments(path, include_excluded=True)
    for row in rows:
        rooms = row.get("rooms")
        if not rooms:
            stats["skipped_no_timing"] += 1
            continue
        items = _end_items_from_join(row.get("join"))
        # Prefer items at start of segment when end is only known inventory.
        n = ingest_rooms(board, rooms, source=str(row["source"]), items=items)
        stats["segments"] += 1
        stats["samples"] += n

    live_timing = _safe_json(path.with_name(path.stem + "_run_timing.json"))
    if live_timing and isinstance(live_timing.get("rooms"), list):
        live_items = None
        if path.is_file():
            task = load_task_json(path)
            meta = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
            end_fp = (meta or {}).get("end_fingerprint")
            if isinstance(end_fp, Mapping):
                live_items = parse_items(end_fp.get("items"))
        n = ingest_rooms(
            board,
            live_timing["rooms"],
            source="live",
            items=live_items,
        )
        stats["live_samples"] += n

    return stats


def build_product_room_timeline(
    task_path: Path | str,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Absolute Ceres-zero room visits for the product chain + live take.

    Returns (rows, total_frames, notes). Each row has abs_entry, dwell, hop_key, …
    """
    path = Path(task_path)
    chain, notes = product_chain_segments(path)
    timeline: list[dict[str, Any]] = []
    rta = 0
    ceres: int | None = None

    for row in chain:
        rooms = list(row.get("rooms") or [])
        source = str(row["source"])
        power_on = bool(row.get("power_on"))
        end_fr = int(row["end_fr"])
        items = _end_items_from_join(row.get("join"))
        local0 = 0
        if power_on:
            cz = find_ceres_zero_frame(
                row.get("anchors"),
                rooms=rooms,
                trace=row.get("trace"),
            )
            if cz is not None:
                ceres = cz
                local0 = cz
        samples = rooms_to_hop_samples(rooms, source=source, items=items)
        for sample, room in zip(samples, rooms):
            ef = int(room.get("entry_frame") or 0)
            if power_on and ceres is not None and ef < ceres:
                continue
            abs_e = rta + (ef - local0 if power_on else ef)
            timeline.append(
                {
                    **sample,
                    "abs_entry": abs_e,
                    "source": source,
                }
            )
        if power_on and ceres is not None:
            rta += max(0, end_fr - ceres)
        else:
            rta += end_fr
        notes.append(
            f"{source}: product chain +{fmt_time(end_fr if not power_on or ceres is None else end_fr - (ceres or 0))}"
        )

    live_timing = _safe_json(path.with_name(path.stem + "_run_timing.json"))
    live_total = 0
    if live_timing and isinstance(live_timing.get("rooms"), list):
        live_items = None
        live_end = int(live_timing.get("total_frames") or 0)
        if path.is_file():
            task = load_task_json(path)
            meta = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
            end_fp = (meta or {}).get("end_fingerprint")
            if isinstance(end_fp, Mapping):
                live_items = parse_items(end_fp.get("items"))
                if end_fp.get("frame") is not None:
                    live_end = max(live_end, int(end_fp["frame"]))
            if live_end <= 0:
                live_end = max(0, int(task.get("frame_count") or 0) - 1)
        samples = rooms_to_hop_samples(
            live_timing["rooms"], source="live", items=live_items
        )
        for sample, room in zip(samples, live_timing["rooms"]):
            ef = int(room.get("entry_frame") or 0)
            timeline.append({**sample, "abs_entry": rta + ef, "source": "live"})
        live_total = int(live_timing.get("total_frames") or live_end or 0)
        if live_total <= 0 and samples:
            live_total = max(int(s.get("entry_frame") or 0) for s in samples) + 1
        rta += live_total
        notes.append(f"live: +{fmt_time(live_total)} (f{live_total})")

    return timeline, rta, notes


def _pace_mark(dwell: int, st: HopStats | None) -> str:
    """★ PB · ✓ ≤avg+0.5σ or ≤1.1×PB · ~ ≤avg+1.5σ · ✗ slower."""
    if st is None:
        if dwell / 60.0 <= 15:
            return "✓"
        if dwell / 60.0 <= 30:
            return "~"
        return "✗"
    if dwell <= st.pb:
        return "★"
    soft = max(st.pb * 1.1, st.avg + 0.5 * st.sd)
    hard = max(st.pb * 1.5, st.avg + 1.5 * st.sd) if st.n >= 2 else st.pb * 1.5
    if dwell <= soft:
        return "✓"
    if dwell <= hard:
        return "~"
    return "✗"


def format_pb_board_table(
    board: PbBoard,
    timeline: Sequence[Mapping[str, Any]],
    *,
    total_frames: int,
    notes: Sequence[str] | None = None,
    max_rooms: int | None = None,
) -> str:
    """Human-readable PB table with PB / avg / sd / Δ."""
    lines: list[str] = []
    lines.append("=" * 108)
    lines.append(f"PB BOARD  ·  {board.task}  ·  product RTA {fmt_time(total_frames)} ({total_frames}f)")
    n_hops = len(board.hops)
    n_samp = sum(len(h.get("samples") or []) for h in board.hops.values())
    lines.append(
        f"history: {n_hops} hop_keys · {n_samp} samples · {len(board.runs)} product runs"
    )
    for note in notes or []:
        lines.append(f"  · {note}")
    lines.append("-" * 108)
    lines.append(
        f"{'RTA':>10} {'DWELL':>9} {'PB':>9} {'AVG':>9} {'SD':>8} {'n':>3} "
        f"{'ΔPB':>7} {'ΔAVG':>7} {'ok':>3}  ROOM → next"
    )
    lines.append("-" * 108)

    rows = list(timeline)
    if max_rooms is not None:
        rows = rows[:max_rooms]

    for row in rows:
        hop_key = str(row.get("hop_key") or "")
        dwell = int(row.get("dwell") or 0)
        st = board.hop_stats(hop_key) if hop_key else None
        # Fall back to any samples for this room if hop_key is new/rare
        if st is None and row.get("room_id") is not None:
            st = board.stats_by_room(int(row["room_id"]))
        mark = _pace_mark(dwell, st)
        if st is not None:
            pb_s = fmt_time(st.pb)
            avg_s = fmt_time(int(round(st.avg)))
            sd_s = fmt_time(int(round(st.sd))) if st.n >= 2 else "—"
            n_s = str(st.n)
            dpb = f"{dwell - st.pb:+d}f"
            davg = f"{dwell - int(round(st.avg)):+d}f"
        else:
            pb_s = avg_s = sd_s = "—"
            n_s = "0"
            dpb = davg = ""
        dest = _dest_name(row.get("dest_room_id"))
        name = str(row.get("name") or hop_key)
        src = str(row.get("source") or "")
        lines.append(
            f"{fmt_time(int(row.get('abs_entry') or 0)):>10} "
            f"{fmt_time(dwell):>9} "
            f"{pb_s:>9} {avg_s:>9} {sd_s:>8} {n_s:>3} "
            f"{dpb:>7} {davg:>7} {mark:>3}  "
            f"{name} → {dest}  [{src}]"
        )

    if max_rooms is not None and len(timeline) > max_rooms:
        lines.append(f"  … {len(timeline) - max_rooms} more rooms")

    # Milestones
    lines.append("-" * 108)
    lines.append("MILESTONES (first entry on product line)")
    lines.append(f"{'RTA':>10}  MILESTONE")
    seen: set[int] = set()
    want = {rid: label for rid, label in _MILESTONES}
    for row in timeline:
        rid = int(row.get("room_id") or 0)
        if rid in want and rid not in seen:
            seen.add(rid)
            lines.append(f"{fmt_time(int(row.get('abs_entry') or 0)):>10}  {want[rid]}")
    lines.append(f"{fmt_time(total_frames):>10}  END (product total)")
    lines.append("-" * 108)
    lines.append("ok: ★=PB  ✓=near avg/PB  ~=soft  ✗=slow  ·  Δ vs board history (all runs)")
    lines.append("=" * 108)
    return "\n".join(lines)


def materialize_pb_board(
    task_path: Path | str,
    *,
    write: bool = True,
    print_table: bool = True,
    max_rooms: int | None = None,
) -> tuple[PbBoard, list[dict[str, Any]], int, str]:
    """Ingest archives + live, save board, build product timeline, print table.

    Returns (board, timeline, total_frames, table_text).
    """
    path = Path(task_path)
    board_path = pb_board_path(path)
    board = PbBoard.load(board_path) if board_path.is_file() else PbBoard(task=path.stem)
    board.task = path.stem

    ingest = ingest_task_archives(board, path)
    board.notes = [
        f"ingest segments={ingest['segments']} new_samples={ingest['samples']} "
        f"live_new={ingest['live_samples']} no_timing={ingest['skipped_no_timing']}"
    ]

    timeline, total, chain_notes = build_product_room_timeline(path)
    board.notes.extend(chain_notes[:12])

    # Stable product-run id from chain sources + total (re-ingest same = no dup)
    chain_segs = [str(r.get("source")) for r in timeline]
    # unique order-preserving sources
    seen_src: list[str] = []
    for s in chain_segs:
        if s not in seen_src:
            seen_src.append(s)
    run_id = f"product:{'+'.join(seen_src)}:{total}"
    milestones: dict[str, int] = {}
    seen_m: set[int] = set()
    for row in timeline:
        rid = int(row.get("room_id") or 0)
        for mid, label in _MILESTONES:
            if rid == mid and mid not in seen_m:
                seen_m.add(mid)
                milestones[label] = int(row.get("abs_entry") or 0)
    board.add_run(
        run_id=run_id,
        total_frames=total,
        segments=seen_src,
        milestones=milestones,
    )

    if write:
        board.save(board_path)
        board.notes.append(f"wrote {board_path}")

    # Also keep a light stitched-compatible summary for tools that read it
    if write:
        summary = {
            "schemaVersion": 1,
            "kind": "super_metroid_stitched_run",
            "task": path.stem,
            "total_frames": total,
            "total_time": fmt_time(total),
            "source": "pb_board",
            "segments": seen_src,
            "milestones": [
                {"label": k, "frame": v, "time": fmt_time(v)}
                for k, v in milestones.items()
            ],
            "room_splits": [
                {
                    "time": fmt_time(int(r.get("abs_entry") or 0)),
                    "dwell_time": fmt_time(int(r.get("dwell") or 0)),
                    "dwell": int(r.get("dwell") or 0),
                    "name": r.get("name"),
                    "hop_key": r.get("hop_key"),
                    "source": r.get("source"),
                    "room_id": r.get("room_id"),
                    "dest_room_id": r.get("dest_room_id"),
                }
                for r in timeline
            ],
            "notes": list(board.notes),
            "pb_board": str(board_path),
        }
        stitch_path = path.with_name(path.stem + "_stitched.json")
        stitch_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        board.notes.append(f"wrote {stitch_path}")

    table = format_pb_board_table(
        board,
        timeline,
        total_frames=total,
        notes=board.notes,
        max_rooms=max_rooms,
    )
    if print_table:
        print(table, flush=True)
    return board, list(timeline), total, table
