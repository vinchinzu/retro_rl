"""Hop skill bank: combine runs, track PBs, optimize room-by-room.

Product path
------------

1. **Record** full button tapes (guided_human / ``./play``) with live anchors.
2. **Materialize** → settled hops + hop bodies + bank candidates.
3. **Unit of optimization = one hop** from its live ``entry_anchor``.
4. **Compose** multi-room runs as pin→body→leave pin (see ``human_tape.compose``),
   not multi-minute power-on open-loop (desync risk for pin recovery).
5. Full-run "PB" is verified continuous compose **or** a theoretical
   Frankenstein sum of hop PBs (labeled until dual-green hop-compose).
6. When a faster previous hop changes leave kinematics, re-pin the next hop's
   entry via natural entry — do not assume frame-append across seams is sound.

Identity
--------

``hop_key`` is stable and direction-aware::

    {room_hex}:{from_room_hex|start}->{to_room_hex|goal}:{items_hex}

Example: ``0x9E9F:0x9F11->0x9F64:0x0004`` Morph Ball Room traversal.

See ``docs/RUN_TIMING_AND_SKILL_BANK.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from super_metroid.hop_id import (  # re-export for back-compat
    make_hop_key,
    parse_items,
    parse_items_value,
    parse_room_id,
)
from super_metroid.human_tape.anchors import match_anchor
from super_metroid.rooms.canonical_names import load_canonical_names, room_name

if TYPE_CHECKING:
    from super_metroid.run_splits import RoomSplit

try:
    from super_metroid.paths import RECORDINGS_DIR as _RECORDINGS_DIR
except ImportError:  # pragma: no cover - fallback for partial installs
    _RECORDINGS_DIR = Path("recordings")

DEFAULT_BANK_DIR = Path(_RECORDINGS_DIR) / "skill_bank"
DEFAULT_BANK_PATH = DEFAULT_BANK_DIR / "bank.json"


@dataclass
class HopSkillRecord:
    """One known solution for a hop key (human, pure, GA, hill-climb, …)."""

    hop_key: str
    room_id: int
    name: str
    frames: int
    source: str
    entry_anchor: str | None = None
    entry_fingerprint: Mapping[str, Any] | None = None
    leave_fingerprint: Mapping[str, Any] | None = None
    dual_green: bool = False
    assist: bool | None = None
    body_path: str | None = None
    run_id: str | None = None
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room_id_hex"] = f"0x{self.room_id:04X}"
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HopSkillRecord:
        return cls(
            hop_key=str(data["hop_key"]),
            room_id=int(data["room_id"]),
            name=str(data.get("name") or ""),
            frames=int(data["frames"]),
            source=str(data.get("source") or ""),
            entry_anchor=data.get("entry_anchor"),
            entry_fingerprint=data.get("entry_fingerprint"),
            leave_fingerprint=data.get("leave_fingerprint"),
            dual_green=bool(data.get("dual_green", False)),
            assist=data.get("assist"),
            body_path=data.get("body_path"),
            run_id=data.get("run_id"),
            notes=str(data.get("notes") or ""),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class SkillBank:
    """In-memory PB bank keyed by hop_key (best dual-green preferred)."""

    records: dict[str, list[HopSkillRecord]] = field(default_factory=dict)

    def add(self, record: HopSkillRecord) -> None:
        self.records.setdefault(record.hop_key, []).append(record)

    def best(self, hop_key: str, *, require_dual_green: bool = False) -> HopSkillRecord | None:
        cands = list(self.records.get(hop_key) or [])
        if require_dual_green:
            cands = [c for c in cands if c.dual_green]
        if not cands:
            return None
        return min(cands, key=lambda r: (not r.dual_green, r.frames, r.source))

    def pb_map(self, *, require_dual_green: bool = False) -> dict[str, int]:
        out: dict[str, int] = {}
        for key in self.records:
            rec = self.best(key, require_dual_green=require_dual_green)
            if rec is not None:
                out[key] = rec.frames
        return out

    def theoretical_route_pb(
        self,
        route_keys: Sequence[str],
        *,
        require_dual_green: bool = False,
    ) -> dict[str, Any]:
        from super_metroid.run_splits import frankenstein_pb

        return frankenstein_pb(self.pb_map(require_dual_green=require_dual_green), route_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "super_metroid_skill_bank",
            "hop_count": len(self.records),
            "record_count": sum(len(v) for v in self.records.values()),
            "hops": {
                key: [r.to_dict() for r in recs]
                for key, recs in sorted(self.records.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SkillBank:
        bank = cls()
        hops = data.get("hops") or {}
        if isinstance(hops, Mapping):
            for key, recs in hops.items():
                if not isinstance(recs, list):
                    continue
                for row in recs:
                    if isinstance(row, Mapping):
                        bank.add(HopSkillRecord.from_dict(row))
        return bank

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> SkillBank:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _match_entry_anchor(
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    room_id: int,
    target_frame: int,
    *,
    prefer_kinds: Sequence[str] = ("room_enter", "boot"),
    task_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Match hop entry pin via canonical ``match_anchor`` (needs state on disk)."""
    if anchors is None:
        return None
    return match_anchor(
        anchors,
        int(target_frame),
        int(room_id),
        task_path=task_path,
        prefer_kinds=prefer_kinds,
    )


def _match_leave_fingerprint(
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    next_room_id: int | None,
    hop_end_frame: int | None,
    hop: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Leave fingerprint: next room_enter if known, else hop end geometry."""
    if next_room_id is not None:
        target = int(hop_end_frame) if hop_end_frame is not None else 0
        nxt = _match_entry_anchor(
            anchors,
            int(next_room_id),
            target,
            prefer_kinds=("room_enter",),
        )
        if nxt is not None:
            return nxt
    end_xy = hop.get("end_xy")
    if end_xy is None and hop.get("end_x") is not None:
        end_xy = [hop.get("end_x"), hop.get("end_y")]
    if hop_end_frame is None and end_xy is None and hop.get("end_pose") is None:
        return None
    leave: dict[str, Any] = {"kind": "hop_end"}
    if hop_end_frame is not None:
        leave["frame"] = int(hop_end_frame)
    if end_xy is not None:
        leave["xy"] = list(end_xy)
    if hop.get("end_pose") is not None:
        leave["pose"] = int(hop["end_pose"])
    end_items = parse_items(hop.get("end_items"))
    if end_items is not None:
        leave["items"] = f"0x{end_items:04X}"
    return leave


def records_from_room_splits(
    rooms: Sequence[RoomSplit],
    *,
    source: str,
    run_id: str | None = None,
    items: int | None = None,
    items_per_leaf: Sequence[int | None] | None = None,
    dual_green: bool = False,
    assist: bool | None = None,
    names: Mapping[int, str] | None = None,
) -> list[HopSkillRecord]:
    """Promote each room leaf to a bank record.

    *items* applies to every leaf when per-leaf inventory is unknown.
    *items_per_leaf* overrides per visit (same length as *rooms*; missing
    slots fall back to *items*).
    """
    name_table = names if names is not None else load_canonical_names()
    out: list[HopSkillRecord] = []
    prev_room: int | None = None
    for i, r in enumerate(rooms):
        leaf_items = items
        if items_per_leaf is not None and i < len(items_per_leaf):
            leaf_items = items_per_leaf[i]
            if leaf_items is None:
                leaf_items = items
        key = make_hop_key(
            r.room_id,
            from_room_id=prev_room,
            to_room_id=r.dest_room_id,
            items=leaf_items,
        )
        out.append(
            HopSkillRecord(
                hop_key=key,
                room_id=r.room_id,
                name=r.name or room_name(r.room_id, names=name_table),
                frames=int(r.dwell_frames or r.room_frames),
                source=source,
                dual_green=dual_green,
                assist=assist,
                run_id=run_id,
                meta={"entry_frame": r.entry_frame, "leave_frame": r.leave_frame},
            )
        )
        prev_room = r.room_id
    return out


def records_from_hops_and_anchors(
    hops: Sequence[Mapping[str, Any]],
    *,
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    source: str,
    run_id: str | None = None,
    dual_green: bool = False,
    assist: bool | None = None,
    names: Mapping[int, str] | None = None,
) -> list[HopSkillRecord]:
    """Build bank records from human-tape hops + optional anchors index.

    Fills inventory into ``hop_key``, attaches entry anchor path / fingerprint
    when anchors are provided, and leaves ``dual_green`` false until hop-replay
    verifies. Prefer this over :func:`records_from_room_splits` for tape ingest.
    """
    name_table = names if names is not None else load_canonical_names()
    out: list[HopSkillRecord] = []
    prev_end_items: int | None = None
    for i, hop in enumerate(hops):
        room_raw = hop.get("room_id", hop.get("room"))
        room_id = parse_room_id(room_raw)
        if room_id is None:
            continue
        from_room: int | None = None
        if i > 0:
            prev = hops[i - 1]
            from_room = parse_room_id(prev.get("room_id", prev.get("room")))
        to_room: int | None = None
        if i + 1 < len(hops):
            nxt = hops[i + 1]
            to_room = parse_room_id(nxt.get("room_id", nxt.get("room")))

        items = parse_items(hop.get("items"))
        if items is None:
            items = prev_end_items

        if hop.get("dwell") is not None:
            frames = int(hop["dwell"])
        else:
            start_i = hop.get("start_index")
            end_i = hop.get("end_index")
            if start_i is not None and end_i is not None:
                frames = int(end_i) - int(start_i) + 1
            else:
                frames = int(hop.get("frames") or hop.get("room_frames") or 0)

        target_frame = hop.get("frame")
        if target_frame is None:
            target_frame = hop.get("start_index", 0)
        entry = _match_entry_anchor(anchors, int(room_id), int(target_frame))
        entry_path: str | None = None
        entry_fp: dict[str, Any] | None = None
        if entry is not None:
            entry_fp = dict(entry)
            raw_path = entry.get("path") or entry.get("resolved_path")
            if raw_path:
                entry_path = str(raw_path)

        end_frame = hop.get("end_frame")
        if end_frame is None and hop.get("end_index") is not None:
            end_frame = hop.get("end_index")
        leave_fp = _match_leave_fingerprint(
            anchors,
            next_room_id=to_room,
            hop_end_frame=int(end_frame) if end_frame is not None else None,
            hop=hop,
        )

        hop_name = str(hop.get("name") or "").strip()
        if not hop_name or hop_name == "?":
            hop_name = room_name(int(room_id), names=name_table)

        key = make_hop_key(
            int(room_id),
            from_room_id=from_room,
            to_room_id=to_room,
            items=items,
        )
        out.append(
            HopSkillRecord(
                hop_key=key,
                room_id=int(room_id),
                name=hop_name,
                frames=frames,
                source=source,
                entry_anchor=entry_path,
                entry_fingerprint=entry_fp,
                leave_fingerprint=leave_fp,
                dual_green=dual_green,
                assist=assist,
                run_id=run_id,
                meta={
                    "hop_index": int(hop.get("index", i)),
                    "start_index": hop.get("start_index"),
                    "end_index": hop.get("end_index"),
                    "frame": hop.get("frame"),
                    "end_frame": hop.get("end_frame"),
                    "items": f"0x{items:04X}" if items is not None else None,
                },
            )
        )
        end_items = parse_items(hop.get("end_items"))
        if end_items is not None:
            prev_end_items = end_items
        elif items is not None:
            prev_end_items = items
    return out


# Alias preferred by materialize / tape pipelines.
records_from_tape = records_from_hops_and_anchors


def merge_runs_into_bank(
    bank: SkillBank,
    runs: Sequence[tuple[str, Sequence[RoomSplit]]],
    *,
    dual_green: bool = False,
    items: int | None = None,
    items_per_leaf: Sequence[int | None] | None = None,
) -> SkillBank:
    """Ingest room-split runs; bank keeps all records, ``best()`` picks PB.

    For human tapes with inventory + anchors, prefer::

        for rec in records_from_hops_and_anchors(hops, anchors=..., source=...):
            bank.add(rec)
    """
    for run_id, rooms in runs:
        for rec in records_from_room_splits(
            rooms,
            source=run_id,
            run_id=run_id,
            dual_green=dual_green,
            items=items,
            items_per_leaf=items_per_leaf,
        ):
            bank.add(rec)
    return bank


def merge_hops_into_bank(
    bank: SkillBank,
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    dual_green: bool = False,
    assist: bool | None = None,
    names: Mapping[int, str] | None = None,
) -> SkillBank:
    """Ingest hop-list runs (optional shared anchors index) into *bank*."""
    for run_id, hops in runs:
        for rec in records_from_hops_and_anchors(
            hops,
            anchors=anchors,
            source=run_id,
            run_id=run_id,
            dual_green=dual_green,
            assist=assist,
            names=names,
        ):
            bank.add(rec)
    return bank


def compose_plan(
    bank: SkillBank,
    route_keys: Sequence[str],
    *,
    require_dual_green: bool = True,
) -> dict[str, Any]:
    """Resolve each hop to a body/anchor for compose (no emulator).

    Returns missing hops and the ordered skill plan. Execution still must
    verify natural entry — this only selects bank entries.
    """
    steps: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in route_keys:
        rec = bank.best(key, require_dual_green=require_dual_green)
        if rec is None and require_dual_green:
            rec = bank.best(key, require_dual_green=False)
        if rec is None:
            missing.append(key)
            steps.append({"hop_key": key, "status": "missing"})
            continue
        meta = dict(rec.meta or {})
        steps.append(
            {
                "hop_key": key,
                "status": "ready" if rec.dual_green else "candidate",
                "frames": rec.frames,
                "source": rec.source,
                "entry_anchor": rec.entry_anchor,
                "body_path": rec.body_path,
                "dual_green": rec.dual_green,
                "task": meta.get("source_task") or meta.get("task"),
                "hop_index": meta.get("hop_index"),
                "meta": meta,
            }
        )
    theory = bank.theoretical_route_pb(route_keys, require_dual_green=False)
    return {
        "steps": steps,
        "missing": missing,
        "ready": all(s.get("status") == "ready" for s in steps),
        "theoretical": theory,
        "note": (
            "Compose plan only (select). Execute with human_tape.compose / "
            "compose_human_hops.py: boot each entry_anchor → replay body → "
            "verify leave. Optimization stays per-hop from a live pin."
        ),
    }


stitch_route_plan = compose_plan  # back-compat


# --- Optimizer boundary (room-local) -----------------------------------------


@dataclass(frozen=True)
class HopOptimizeJob:
    """Spec for hill-climb / GA on a single hop (emulator boots entry only)."""

    hop_key: str
    entry_anchor: str
    room_id: int
    exit_predicate: str  # "leave_room" | "end_xy" | "item" | "boss_flag"
    exit_detail: Mapping[str, Any] = field(default_factory=dict)
    seed_body: str | None = None
    max_frames: int = 10_000
    assist: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_key": self.hop_key,
            "entry_anchor": self.entry_anchor,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "exit_predicate": self.exit_predicate,
            "exit_detail": dict(self.exit_detail),
            "seed_body": self.seed_body,
            "max_frames": self.max_frames,
            "assist": self.assist,
            "isolation": "single_hop_from_anchor",
            "optimize_unit": "one_hop",
            "compose": "pin_then_body_chain",
        }


def hop_job_from_record(
    record: HopSkillRecord,
    *,
    exit_predicate: str = "leave_room",
    exit_detail: Mapping[str, Any] | None = None,
    max_frames: int | None = None,
) -> HopOptimizeJob:
    if not record.entry_anchor:
        raise ValueError(f"record {record.hop_key} has no entry_anchor")
    return HopOptimizeJob(
        hop_key=record.hop_key,
        entry_anchor=record.entry_anchor,
        room_id=record.room_id,
        exit_predicate=exit_predicate,
        exit_detail=dict(exit_detail or {}),
        seed_body=record.body_path,
        max_frames=max_frames or max(1000, record.frames * 3),
        assist=bool(record.assist) if record.assist is not None else True,
    )


def promote_dual_green(
    hop_key: str,
    *,
    bank_path: Path | str | None = None,
    source: str | None = None,
    entry_anchor: str | None = None,
    body_path: str | None = None,
    frames: int | None = None,
    room_id: int | None = None,
    name: str = "",
    assist: bool | None = None,
    run_id: str | None = None,
    meta: Mapping[str, Any] | None = None,
    notes: str = "promoted after hop-replay dual-green",
) -> HopSkillRecord:
    """Mark a hop dual-green in the bank (create or update matching record).

    Prefer updating an existing non-green record with the same hop_key + source
    (or body_path). Otherwise append a new dual-green record.
    """
    bp = Path(bank_path) if bank_path is not None else DEFAULT_BANK_PATH
    bank = SkillBank.load(bp) if bp.is_file() else SkillBank()
    cands = list(bank.records.get(hop_key) or [])
    updated: HopSkillRecord | None = None
    for rec in cands:
        if source is not None and rec.source != source:
            continue
        if body_path is not None and rec.body_path and rec.body_path != body_path:
            continue
        rec.dual_green = True
        if entry_anchor:
            rec.entry_anchor = entry_anchor
        if body_path:
            rec.body_path = body_path
        if frames is not None:
            rec.frames = int(frames)
        if assist is not None:
            rec.assist = assist
        if notes:
            rec.notes = notes
        if meta:
            rec.meta.update(dict(meta))
        updated = rec
        break
    if updated is None:
        if room_id is None:
            # Parse room from hop_key prefix "0xRRRR:..."
            try:
                room_id = int(str(hop_key).split(":", 1)[0], 0)
            except ValueError as exc:
                raise ValueError(
                    f"promote_dual_green needs room_id or parseable hop_key={hop_key!r}"
                ) from exc
        updated = HopSkillRecord(
            hop_key=hop_key,
            room_id=int(room_id),
            name=name or hop_key,
            frames=int(frames or 0),
            source=source or "hop_replay",
            entry_anchor=entry_anchor,
            dual_green=True,
            assist=assist,
            body_path=body_path,
            run_id=run_id or source,
            notes=notes,
            meta=dict(meta or {}),
        )
        bank.add(updated)
    bank.save(bp)
    return updated


def _find_bank_record_for_hop(
    bank: SkillBank,
    *,
    source: str | None,
    hop_index: int | None,
    entry_anchor: str | None,
    body_path: str | None,
    room_id: int | None,
) -> HopSkillRecord | None:
    """Locate the materialize-ingested record for a hop-replay promote."""
    cands: list[HopSkillRecord] = []
    for recs in bank.records.values():
        cands.extend(recs)
    if hop_index is not None and source is not None:
        for rec in cands:
            if rec.source != source and rec.run_id != source:
                continue
            if int(rec.meta.get("hop_index", -1)) == int(hop_index):
                return rec
    if body_path:
        for rec in cands:
            if rec.body_path and Path(rec.body_path) == Path(body_path):
                return rec
    if entry_anchor:
        for rec in cands:
            if rec.entry_anchor and Path(str(rec.entry_anchor)) == Path(str(entry_anchor)):
                if room_id is None or int(rec.room_id) == int(room_id):
                    return rec
    return None


def promote_from_hop_replay(
    report: Mapping[str, Any],
    *,
    bank_path: Path | str | None = None,
    source: str | None = None,
    body_path: str | None = None,
) -> HopSkillRecord | None:
    """If hop-replay report is green, promote dual_green into the skill bank.

    Prefer matching an existing materialize record by ``source`` + ``hop_index``.
    Returns the promoted record, or None when report is not green.
    """
    if not report.get("green") and not report.get("ok"):
        return None
    sl = report.get("slice") or {}
    hop = sl.get("hop") or {}
    room_id = sl.get("start_room")
    if room_id is None:
        room_id = hop.get("room_id")
    leave = sl.get("leave_room")
    items = parse_items(hop.get("items")) if hop else None
    hop_index = hop.get("index") if hop else sl.get("hop_index")
    if hop_index is None:
        hop_index = sl.get("hop_index")
    entry = report.get("anchor_path") or sl.get("anchor_path")
    task_name = source or sl.get("name") or "hop_replay"
    steps = sl.get("steps")
    frames = int(steps) if steps is not None else hop.get("dwell")
    meta = {
        "hop_index": hop_index,
        "start_index": sl.get("start_index") or hop.get("start_index"),
        "end_index": sl.get("end_index") or hop.get("end_index"),
        "promoted_from": "hop_replay",
        "dual": report.get("dual"),
        "anchor_path": entry,
        "task": sl.get("task"),
        "source_task": sl.get("task"),
    }

    bp = Path(bank_path) if bank_path is not None else DEFAULT_BANK_PATH
    bank = SkillBank.load(bp) if bp.is_file() else SkillBank()
    existing = _find_bank_record_for_hop(
        bank,
        source=str(task_name),
        hop_index=int(hop_index) if hop_index is not None else None,
        entry_anchor=str(entry) if entry else None,
        body_path=body_path,
        room_id=int(room_id) if room_id is not None else None,
    )
    if existing is not None:
        return promote_dual_green(
            existing.hop_key,
            bank_path=bp,
            source=existing.source,
            entry_anchor=str(entry) if entry else existing.entry_anchor,
            body_path=body_path or existing.body_path,
            frames=int(frames) if frames is not None else existing.frames,
            room_id=existing.room_id,
            name=existing.name,
            assist=report.get("assist") if report.get("assist") is not None else existing.assist,
            run_id=existing.run_id or existing.source,
            meta=meta,
        )

    # No prior bank row — invent hop_key from slice (from_room unknown → start).
    hop_key = make_hop_key(
        int(room_id) if room_id is not None else 0,
        from_room_id=None,
        to_room_id=int(leave) if leave is not None else None,
        items=items,
    )
    return promote_dual_green(
        hop_key,
        bank_path=bp,
        source=str(task_name),
        entry_anchor=str(entry) if entry else None,
        body_path=body_path,
        frames=int(frames) if frames is not None else None,
        room_id=int(room_id) if room_id is not None else None,
        name=str(hop.get("name") or sl.get("start_room_hex") or hop_key),
        assist=report.get("assist"),
        run_id=str(task_name),
        meta=meta,
    )
