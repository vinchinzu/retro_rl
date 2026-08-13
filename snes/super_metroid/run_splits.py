"""Hierarchical run timing: room hops fold into items, bosses, segments.

Leaf unit is a **room visit** (or hop) with frame bounds. Higher-level
splits are pure sums / spans over those leaves — no second clock.

Milestone kinds (from a live or offline event stream):

- ``room_enter`` / ``room_leave`` — settled ordinary transitions
- ``item_delta`` — collected_items / beams / expansions change
- ``boss_start`` / ``boss_finish`` — enter boss room / boss flag
- ``segment`` — named route anchors (optional)

PB tracking keys leave kinematics out of the *time* number but the skill
bank stores entry fingerprints so hill-climb stays single-hop (see
``docs/RUN_TIMING_AND_SKILL_BANK.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import anchor_rows, parse_room_id
from super_metroid.rooms.canonical_names import load_canonical_names, room_name


# Boss rooms by SNES room_id (Map Rando / vanilla).
BOSS_ROOMS: dict[int, str] = {
    0x9804: "Bomb Torizo",
    0x9DC7: "Spore Spawn",
    0xA59F: "Kraid",
    0xA98D: "Crocomire",
    0xCD13: "Phantoon",
    0xD95E: "Botwoon",
    0xDA60: "Draygon",
    0xB283: "Golden Torizo",
    0xB32E: "Ridley",
    0xDD58: "Mother Brain",
    0xE0B5: "Ceres Ridley",
}

# Upgrade-oriented milestone labels used when only a bit / name is known.
DEFAULT_ITEM_MILESTONE_NAMES: frozenset[str] = frozenset(
    {
        "Morph",
        "Bombs",
        "Charge",
        "HiJump",
        "Spazer",
        "Varia",
        "SpeedBooster",
        "Ice",
        "Grapple",
        "Wave",
        "XRayScope",
        "Gravity",
        "SpaceJump",
        "SpringBall",
        "Plasma",
        "ScrewAttack",
    }
)


@dataclass(frozen=True)
class TimingEvent:
    """One ordered event on a run timeline (emulator frames)."""

    frame: int
    kind: str
    room_id: int = 0
    label: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "kind": self.kind,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}" if self.room_id else None,
            "label": self.label,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class RoomSplit:
    """Leaf timing: one visit in a room until leave/exit."""

    index: int
    room_id: int
    name: str
    entry_frame: int
    leave_frame: int
    exit_frame: int | None = None
    dest_room_id: int | None = None
    dwell_frames: int = 0
    room_frames: int = 0
    transition_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "name": self.name,
            "entry_frame": self.entry_frame,
            "leave_frame": self.leave_frame,
            "exit_frame": self.exit_frame,
            "dest_room_id": self.dest_room_id,
            "dest_room_id_hex": (
                f"0x{self.dest_room_id:04X}" if self.dest_room_id else None
            ),
            "dwell_frames": self.dwell_frames,
            "room_frames": self.room_frames,
            "transition_frames": self.transition_frames,
        }


@dataclass(frozen=True)
class FoldedSplit:
    """Higher-level split spanning one or more room leaves."""

    kind: str
    id: str
    label: str
    start_frame: int
    end_frame: int
    frames: int
    room_ids: tuple[int, ...] = ()
    room_names: tuple[str, ...] = ()
    leaf_indices: tuple[int, ...] = ()
    start_event: str | None = None
    end_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
            "room_ids": list(self.room_ids),
            "room_id_hex": [f"0x{r:04X}" for r in self.room_ids],
            "room_names": list(self.room_names),
            "leaf_indices": list(self.leaf_indices),
            "start_event": self.start_event,
            "end_event": self.end_event,
        }


@dataclass
class RunTimingReport:
    """Room leaves + folded item/boss/segment PBs for one run."""

    source: str
    rooms: list[RoomSplit]
    items: list[FoldedSplit]
    bosses: list[FoldedSplit]
    segments: list[FoldedSplit]
    events: list[TimingEvent] = field(default_factory=list)
    total_frames: int | None = None
    pb_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "super_metroid_run_timing",
            "source": self.source,
            "total_frames": self.total_frames,
            "summary": {
                "room_visits": len(self.rooms),
                "item_splits": len(self.items),
                "boss_splits": len(self.bosses),
                "segment_splits": len(self.segments),
                "total_room_frames": sum(r.room_frames for r in self.rooms),
                "total_dwell_frames": sum(r.dwell_frames for r in self.rooms),
            },
            "rooms": [r.to_dict() for r in self.rooms],
            "items": [s.to_dict() for s in self.items],
            "bosses": [s.to_dict() for s in self.bosses],
            "segments": [s.to_dict() for s in self.segments],
            "events": [e.to_dict() for e in self.events],
            "pb_notes": list(self.pb_notes),
        }


def room_splits_from_timer_visits(
    visits: Sequence[Mapping[str, Any] | Any],
    *,
    names: Mapping[int, str] | None = None,
) -> list[RoomSplit]:
    """Convert ``RoomTimer`` visit dicts / objects into ``RoomSplit`` leaves."""
    name_table = names if names is not None else load_canonical_names()
    out: list[RoomSplit] = []
    for i, visit in enumerate(visits):
        if hasattr(visit, "to_dict") and not isinstance(visit, Mapping):
            data = visit.to_dict()
        else:
            data = dict(visit)  # type: ignore[arg-type]
        room_id = int(data.get("room_id") or 0)
        entry = int(data.get("entry_frame") or 0)
        leave = int(data.get("leave_frame") or entry)
        exit_f = data.get("exit_frame")
        exit_frame = int(exit_f) if exit_f is not None else None
        dest = data.get("dest_room_id")
        dest_id = int(dest) if dest is not None else None
        dwell = int(data.get("dwell_frames") or max(0, leave - entry))
        room_frames = int(
            data.get("room_frames")
            or (max(0, (exit_frame or leave) - entry))
        )
        transition = int(
            data.get("transition_frames")
            or (
                max(0, exit_frame - leave)
                if exit_frame is not None
                else 0
            )
        )
        out.append(
            RoomSplit(
                index=i,
                room_id=room_id,
                name=room_name(room_id, names=name_table),
                entry_frame=entry,
                leave_frame=leave,
                exit_frame=exit_frame,
                dest_room_id=dest_id,
                dwell_frames=dwell,
                room_frames=room_frames,
                transition_frames=transition,
            )
        )
    return out


def room_splits_from_hops(
    hops: Sequence[Mapping[str, Any]],
    *,
    names: Mapping[int, str] | None = None,
    timeline: str = "index",
) -> list[RoomSplit]:
    """Convert human_tape hop inventory into room leaves.

    ``timeline`` selects which hop fields become entry/leave bounds:

    - ``"index"`` (default): ``start_index`` / ``end_index``. Renumber-safe when
      checkpoint reloads make ``frame`` non-monotonic while the trace list
      keeps growing (matches hop ``dwell``).
    - ``"frame"``: ``frame`` / ``end_frame`` absolute emulator timeline
      (compat with older callers).

    When a hop carries ``transition_frames`` (and optional settled fields),
    dwell still comes from hop ``dwell`` (or end−start+1); transition is
    stored on the leaf for kinematics accounting.
    """
    if timeline not in ("index", "frame"):
        raise ValueError(f"timeline must be 'index' or 'frame', got {timeline!r}")
    name_table = names if names is not None else load_canonical_names()
    out: list[RoomSplit] = []
    for i, hop in enumerate(hops):
        room_id = int(hop.get("room_id") or 0)
        start_index = int(hop.get("start_index", hop.get("frame", 0)))
        end_index = int(hop.get("end_index", hop.get("end_frame", start_index)))
        transition = int(hop.get("transition_frames") or 0)
        # Provisional dwell for frame-mode leave fallback when end_frame missing.
        provisional_dwell = int(
            hop.get("dwell") or max(0, end_index - start_index + 1)
        )

        if timeline == "frame":
            entry_frame = int(hop.get("frame", start_index))
            leave_frame = int(
                hop.get(
                    "end_frame",
                    entry_frame + max(0, provisional_dwell - 1),
                )
            )
        else:
            entry_frame = start_index
            leave_frame = end_index

        # Prefer hop dwell (stable under renumber / settled bounds).
        dwell = int(hop.get("dwell") or max(0, leave_frame - entry_frame + 1))

        exit_f = hop.get("exit_frame")
        exit_frame = int(exit_f) if exit_f is not None else None
        if exit_frame is None and transition:
            exit_frame = leave_frame + transition

        if hop.get("room_frames") is not None:
            room_frames = int(hop["room_frames"])
        elif exit_frame is not None:
            room_frames = max(0, exit_frame - entry_frame)
        else:
            room_frames = dwell + transition

        dest = None
        if i + 1 < len(hops):
            dest = int(hops[i + 1].get("room_id") or 0)
        elif hop.get("dest_room_id") is not None:
            dest = int(hop["dest_room_id"])

        out.append(
            RoomSplit(
                index=i,
                room_id=room_id,
                name=str(hop.get("name") or room_name(room_id, names=name_table)),
                entry_frame=entry_frame,
                leave_frame=leave_frame,
                exit_frame=exit_frame,
                dest_room_id=dest,
                dwell_frames=dwell,
                room_frames=room_frames,
                transition_frames=transition,
            )
        )
    return out


def _leaves_between(
    rooms: Sequence[RoomSplit],
    start_frame: int,
    end_frame: int,
) -> list[RoomSplit]:
    """Room leaves whose entry falls in [start, end)."""
    return [
        r
        for r in rooms
        if r.entry_frame >= start_frame and r.entry_frame < end_frame
    ]


def fold_item_to_item(
    events: Sequence[TimingEvent],
    rooms: Sequence[RoomSplit],
    *,
    item_kinds: frozenset[str] | None = None,
) -> list[FoldedSplit]:
    """Adjacent item_delta events → splits; rooms that start inside the span."""
    kinds = item_kinds or frozenset({"item_delta"})
    items = [e for e in events if e.kind in kinds]
    items = sorted(items, key=lambda e: e.frame)
    if len(items) < 2:
        # Single item from run start
        if len(items) == 1:
            e1 = items[0]
            start = 0
            end = e1.frame
            leaves = _leaves_between(rooms, start, end + 1)
            label = e1.label or "item"
            return [
                FoldedSplit(
                    kind="item",
                    id=f"start_to_{label}",
                    label=f"start → {label}",
                    start_frame=start,
                    end_frame=end,
                    frames=max(0, end - start),
                    room_ids=tuple(r.room_id for r in leaves),
                    room_names=tuple(r.name for r in leaves),
                    leaf_indices=tuple(r.index for r in leaves),
                    start_event="run_start",
                    end_event=label,
                )
            ]
        return []

    out: list[FoldedSplit] = []
    for a, b in zip(items, items[1:]):
        start, end = a.frame, b.frame
        leaves = _leaves_between(rooms, start, end)
        la = a.label or "item_a"
        lb = b.label or "item_b"
        out.append(
            FoldedSplit(
                kind="item",
                id=f"{la}_to_{lb}",
                label=f"{la} → {lb}",
                start_frame=start,
                end_frame=end,
                frames=max(0, end - start),
                room_ids=tuple(r.room_id for r in leaves),
                room_names=tuple(r.name for r in leaves),
                leaf_indices=tuple(r.index for r in leaves),
                start_event=la,
                end_event=lb,
            )
        )
    return out


def fold_boss_fights(
    events: Sequence[TimingEvent],
    rooms: Sequence[RoomSplit],
) -> list[FoldedSplit]:
    """Pair boss_start → boss_finish; fallback to boss-room leaf span."""
    starts = [e for e in events if e.kind == "boss_start"]
    finishes = [e for e in events if e.kind == "boss_finish"]
    out: list[FoldedSplit] = []
    used_finish: set[int] = set()

    for start in sorted(starts, key=lambda e: e.frame):
        label = start.label or BOSS_ROOMS.get(start.room_id, "boss")
        finish = None
        for i, f in enumerate(sorted(finishes, key=lambda e: e.frame)):
            if i in used_finish:
                continue
            if f.frame < start.frame:
                continue
            # Same boss label or same room preferred
            if f.label and start.label and f.label != start.label:
                if f.room_id and start.room_id and f.room_id != start.room_id:
                    continue
            finish = f
            used_finish.add(i)
            break
        if finish is None:
            continue
        leaves = _leaves_between(rooms, start.frame, finish.frame + 1)
        out.append(
            FoldedSplit(
                kind="boss",
                id=f"boss_{label}".replace(" ", "_").lower(),
                label=label,
                start_frame=start.frame,
                end_frame=finish.frame,
                frames=max(0, finish.frame - start.frame),
                room_ids=tuple(r.room_id for r in leaves),
                room_names=tuple(r.name for r in leaves),
                leaf_indices=tuple(r.index for r in leaves),
                start_event="boss_start",
                end_event="boss_finish",
            )
        )

    # Infer unfinished boss rooms that appear in leaves without events
    if not out:
        for r in rooms:
            if r.room_id in BOSS_ROOMS:
                out.append(
                    FoldedSplit(
                        kind="boss",
                        id=f"boss_room_{r.room_id:04X}",
                        label=BOSS_ROOMS[r.room_id],
                        start_frame=r.entry_frame,
                        end_frame=r.leave_frame,
                        frames=r.dwell_frames,
                        room_ids=(r.room_id,),
                        room_names=(r.name,),
                        leaf_indices=(r.index,),
                        start_event="room_enter",
                        end_event="room_leave",
                    )
                )
    return out


def fold_named_segments(
    events: Sequence[TimingEvent],
    rooms: Sequence[RoomSplit],
) -> list[FoldedSplit]:
    """Adjacent ``segment`` markers (route anchors) → folded spans."""
    segs = sorted(
        [e for e in events if e.kind == "segment"],
        key=lambda e: e.frame,
    )
    if len(segs) < 2:
        return []
    out: list[FoldedSplit] = []
    for a, b in zip(segs, segs[1:]):
        leaves = _leaves_between(rooms, a.frame, b.frame)
        la = a.label or "seg_a"
        lb = b.label or "seg_b"
        out.append(
            FoldedSplit(
                kind="segment",
                id=f"{la}_to_{lb}",
                label=f"{la} → {lb}",
                start_frame=a.frame,
                end_frame=b.frame,
                frames=max(0, b.frame - a.frame),
                room_ids=tuple(r.room_id for r in leaves),
                room_names=tuple(r.name for r in leaves),
                leaf_indices=tuple(r.index for r in leaves),
                start_event=la,
                end_event=lb,
            )
        )
    return out


def _parse_room_field(row: Mapping[str, Any]) -> int:
    """Room id from ``room_id`` int or ``room`` / ``room_hex`` hex string."""
    if row.get("room_id") is not None:
        rid = parse_room_id(row["room_id"])
        if rid is not None:
            return rid
    for key in ("room", "room_hex"):
        rid = parse_room_id(row.get(key))
        if rid is not None:
            return rid
    return 0


def _parse_items_label(raw: Any) -> str | None:
    """Normalize items field to a display hex label (e.g. ``0x3105``)."""
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            val = int(raw, 0)
            # Preserve 0x prefix style when input was hex-ish
            if raw.strip().lower().startswith("0x"):
                return f"0x{val:04X}"
            return f"0x{val:04X}"
        return f"0x{int(raw):04X}"
    except (TypeError, ValueError):
        return str(raw) if raw else None


_anchor_rows = anchor_rows


def events_from_anchors(
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    include_end: bool = True,
) -> list[TimingEvent]:
    """Map guided_human anchor fingerprints to ``TimingEvent`` rows.

    Kind mapping:

    - ``room_enter`` → ``room_enter``
    - ``boot`` → ``room_enter`` with label ``\"boot\"``
    - ``item_delta`` → ``item_delta`` (label from items hex when present)
    - ``manual`` / ``mid_lockstep`` → ``segment`` (label from anchor label/kind)
    - ``end`` → optional ``segment`` label ``\"end\"`` (``include_end``)
    """
    events: list[TimingEvent] = []
    for row in _anchor_rows(anchors):
        kind = str(row.get("kind") or "pin")
        try:
            frame = int(row.get("frame", 0))
        except (TypeError, ValueError):
            continue
        room_id = _parse_room_field(row)
        label = row.get("label")
        label_s = str(label) if label is not None else None
        detail = {
            k: row[k]
            for k in ("path", "items", "beams", "xy", "pose", "energy")
            if k in row
        }

        if kind == "room_enter":
            events.append(
                TimingEvent(
                    frame=frame,
                    kind="room_enter",
                    room_id=room_id,
                    label=label_s or f"enter_0x{room_id:04X}",
                    detail=detail,
                )
            )
        elif kind == "boot":
            events.append(
                TimingEvent(
                    frame=frame,
                    kind="room_enter",
                    room_id=room_id,
                    label=label_s or "boot",
                    detail=detail,
                )
            )
        elif kind == "item_delta":
            items_label = _parse_items_label(row.get("items"))
            events.append(
                TimingEvent(
                    frame=frame,
                    kind="item_delta",
                    room_id=room_id,
                    label=items_label or label_s or "item_delta",
                    detail=detail,
                )
            )
        elif kind in ("manual", "mid_lockstep"):
            events.append(
                TimingEvent(
                    frame=frame,
                    kind="segment",
                    room_id=room_id,
                    label=label_s or kind,
                    detail=detail,
                )
            )
        elif kind == "end":
            if include_end:
                events.append(
                    TimingEvent(
                        frame=frame,
                        kind="segment",
                        room_id=room_id,
                        label=label_s or "end",
                        detail=detail,
                    )
                )
        # Unknown kinds (pin, etc.) skipped — not timing milestones
    events.sort(key=lambda e: (e.frame, e.kind, e.room_id))
    return events


def events_from_boss_room_splits(
    rooms: Sequence[RoomSplit],
) -> list[TimingEvent]:
    """Infer boss_start / boss_finish from boss-room leaf spans.

    Honest **room dwell as fight proxy**: when live fight markers are absent,
    enter/leave of a known boss room stand in for fight bounds. This is not a
    verified boss-flag timing — fold_boss_fights prefers real events when
    present.
    """
    events: list[TimingEvent] = []
    for r in rooms:
        if r.room_id not in BOSS_ROOMS:
            continue
        label = BOSS_ROOMS[r.room_id]
        events.append(
            TimingEvent(
                frame=r.entry_frame,
                kind="boss_start",
                room_id=r.room_id,
                label=label,
                detail={"proxy": "room_dwell", "source": "room_split"},
            )
        )
        events.append(
            TimingEvent(
                frame=r.leave_frame,
                kind="boss_finish",
                room_id=r.room_id,
                label=label,
                detail={"proxy": "room_dwell", "source": "room_split"},
            )
        )
    events.sort(key=lambda e: (e.frame, e.kind, e.room_id))
    return events


def events_from_task_payload(
    *,
    trace: Sequence[Mapping[str, Any]] | None = None,
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    rooms: Sequence[RoomSplit] | None = None,
) -> list[TimingEvent]:
    """Unify trace item deltas + anchors + optional boss-room proxies.

    Prefers ``item_delta`` from the trace when both trace and anchors fire on
    the same ``(frame, room_id)``. Deduplicates near-identical
    ``(kind, frame, room_id)`` events (first wins after preference order).
    """
    out: list[TimingEvent] = []
    if trace is not None:
        out.extend(events_from_trace_item_deltas(trace))

    trace_item_keys = {
        (e.frame, e.room_id) for e in out if e.kind == "item_delta"
    }

    if anchors is not None:
        for e in events_from_anchors(anchors):
            if e.kind == "item_delta" and (e.frame, e.room_id) in trace_item_keys:
                continue
            out.append(e)

    if rooms is not None:
        out.extend(events_from_boss_room_splits(rooms))

    out.sort(key=lambda e: (e.frame, e.kind, e.room_id))
    deduped: list[TimingEvent] = []
    seen: set[tuple[str, int, int]] = set()
    for e in out:
        key = (e.kind, e.frame, e.room_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def events_from_trace_item_deltas(
    trace: Sequence[Mapping[str, Any]],
    *,
    item_key: str = "items",
) -> list[TimingEvent]:
    """Scan a human/continuous-style trace for collected_items changes."""
    events: list[TimingEvent] = []
    prev: int | None = None
    for i, row in enumerate(trace):
        raw = row.get(item_key)
        if raw is None:
            continue
        try:
            if isinstance(raw, str):
                cur = int(raw, 0)
            else:
                cur = int(raw)
        except (TypeError, ValueError):
            continue
        if prev is not None and cur != prev:
            frame = int(row.get("frame", i))
            room = _parse_room_field(row)
            events.append(
                TimingEvent(
                    frame=frame,
                    kind="item_delta",
                    room_id=room,
                    label=f"items_{cur:04X}",
                    detail={"from": prev, "to": cur},
                )
            )
        prev = cur
    return events


def build_run_timing(
    rooms: Sequence[RoomSplit],
    events: Sequence[TimingEvent] | None = None,
    *,
    source: str = "run",
    total_frames: int | None = None,
) -> RunTimingReport:
    """Assemble room leaves + folded item/boss/segment layers."""
    ev = list(events or [])
    items = fold_item_to_item(ev, rooms)
    bosses = fold_boss_fights(ev, rooms)
    segments = fold_named_segments(ev, rooms)
    if total_frames is None and rooms:
        last = rooms[-1]
        total_frames = last.exit_frame or last.leave_frame
    return RunTimingReport(
        source=source,
        rooms=list(rooms),
        items=items,
        bosses=bosses,
        segments=segments,
        events=ev,
        total_frames=total_frames,
    )


def compare_room_pbs(
    baseline: Sequence[RoomSplit],
    candidate: Sequence[RoomSplit],
    *,
    key: str = "dwell_frames",
) -> list[dict[str, Any]]:
    """Per-index room delta (candidate - baseline). Negative = faster."""
    n = min(len(baseline), len(candidate))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        b, c = baseline[i], candidate[i]
        b_val = int(getattr(b, key))
        c_val = int(getattr(c, key))
        rows.append(
            {
                "index": i,
                "room_id": c.room_id,
                "room_id_hex": f"0x{c.room_id:04X}",
                "name": c.name,
                "baseline": b_val,
                "candidate": c_val,
                "delta": c_val - b_val,
                "same_room": b.room_id == c.room_id,
            }
        )
    return rows


def frankenstein_pb(
    room_pb_by_key: Mapping[str, int],
    route_keys: Sequence[str],
) -> dict[str, Any]:
    """Sum best known hop times along a route (theoretical until compose green).

    ``route_keys`` are hop skill ids (not bare room names — same room can be
    traversed in different directions / inventories).
    """
    missing = [k for k in route_keys if k not in room_pb_by_key]
    total = sum(int(room_pb_by_key[k]) for k in route_keys if k in room_pb_by_key)
    return {
        "route_keys": list(route_keys),
        "frames": total,
        "covered": len(route_keys) - len(missing),
        "missing": missing,
        "complete": not missing,
        "note": (
            "Theoretical sum of hop PBs. Not a verified continuous run until "
            "natural-entry compose is dual-green end-to-end."
        ),
    }


def best_room_deltas(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = 15,
) -> list[dict[str, Any]]:
    """Rooms where candidate is slowest vs baseline (tightening targets)."""
    rows = [dict(r) for r in comparisons if r.get("same_room", True)]
    rows.sort(key=lambda r: int(r.get("delta") or 0), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows
