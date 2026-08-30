"""TapeCandidate adapters for Attic and Bowling from s23.

Settled-room tape slices with bounded live projection and
``search_live_adapter`` recovery. Scaffold-only. Does not boot an emulator
or emit Main Shaft hops.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.hop_id import make_hop_key, parse_room_id
from super_metroid.human_tape.anchors import (
    load_anchors_index,
    match_anchor,
    resolve_anchor_path,
)
from super_metroid.human_tape.hops import hop_items_int
from super_metroid.leave_specs import LeaveSpec
from super_metroid.paths import GAME_DIR
from super_metroid.room_adapter import AdapterSearchConfig, search_live_adapter
from super_metroid.routes.kpdr.room_ids import ROOM_WEST_OCEAN, ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.splice.cards import artifact_dir
from super_metroid.splice.errors import PreflightError
from super_metroid.splice.manifest import dest_leave_spec
from super_metroid.splice.preflight import file_digest
from super_metroid.splice.schema import (
    CandidateArtifact,
    EntryContract,
    EntryFingerprint,
    LeaveSpecRef,
    RouteEdge,
    rel_path,
)

SEGMENT = "s23"
ATTIC_TASK_ID = "attic"
BOWLING_TASK_ID = "bowling"
ATTIC_ROOM = ROOM_WS_ATTIC  # 0xCA52
WEST_OCEAN_ROOM = ROOM_WEST_OCEAN  # 0x93FE
BOWLING_ROOM = 0xC98E
GRAVITY_ROOM = 0xCE40
MAIN_SHAFT_ROOM = ROOM_WS_MAIN  # serial; never a tape hop here
DEFAULT_S23_DIR = GAME_DIR / "tasks" / "full_start_v1_segments" / SEGMENT
OWNER_PACKAGE = "snes/super_metroid/splice"
RECOVERY = "search_live_adapter"
BOWLING_PLANNED_DWELL = 5015  # s23 bowling hop; split internally, one Gravity contract
BOWLING_INTERNAL_MAX_FRAMES = 1800
BOUNDED_LIVE_ADAPTER = AdapterSearchConfig(beam_width=8, max_depth=4, frame_penalty=0.35)
PROJECTION_XY_TOL = 24
PROJECTION_FRAME_WINDOW = 16
_LEAVE_BAND = 80
_SOURCE_NOTES = (
    "Scaffold tape candidate; not Survival/Finish",
    "Main Shaft / rr-kw8t remains serial",
)

__all__ = [
    "ATTIC_ROOM",
    "ATTIC_TASK_ID",
    "BOUNDED_LIVE_ADAPTER",
    "BOWLING_INTERNAL_MAX_FRAMES",
    "BOWLING_PLANNED_DWELL",
    "BOWLING_ROOM",
    "BOWLING_TASK_ID",
    "DEFAULT_S23_DIR",
    "ExternalContract",
    "GRAVITY_ROOM",
    "LiveProjection",
    "MAIN_SHAFT_ROOM",
    "RECOVERY",
    "SEGMENT",
    "TapeCandidate",
    "TapeSlice",
    "WEST_OCEAN_ROOM",
    "load_s23_tape_candidates",
    "project_live",
    "recover_live",
    "resolve_segment_dir",
    "search_live_adapter",
]


def resolve_segment_dir(path: Path | str | None = None) -> Path:
    """s23 directory. ``None`` → gitignored default under tasks/."""
    if path is None:
        return DEFAULT_S23_DIR
    p = Path(path)
    if p.name == SEGMENT or (p / "tape.json").is_file() or (p / "extract.json").is_file():
        return p
    return p / SEGMENT


@dataclass(frozen=True)
class TapeSlice:
    """Internal hop window. Not an external task."""

    slice_id: str
    frame_start: int
    frame_end: int
    room_id: int
    pin_path: str | None = None
    pin_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room"] = f"0x{self.room_id:04X}"
        return d


@dataclass(frozen=True)
class ExternalContract:
    """One natural-entry room contract (internal slices are not tasks)."""

    task_id: str
    room_id: int
    next_room_id: int
    natural_entry: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "room_id": self.room_id,
            "room": f"0x{self.room_id:04X}",
            "next_room_id": self.next_room_id,
            "next_room": f"0x{self.next_room_id:04X}",
            "natural_entry": self.natural_entry,
        }


@dataclass(frozen=True)
class LiveProjection:
    """Bounded projection of a live fingerprint onto a tape slice."""

    within_bound: bool
    score: float
    sample_index: int | None
    recovery: str | None
    room_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TapeCandidate:
    """One settled room slice. Recovery is ``search_live_adapter``."""

    task_id: str
    room_id: int
    next_room_id: int
    artifact: CandidateArtifact
    edge: RouteEdge
    contract: ExternalContract
    internal_slices: tuple[TapeSlice, ...]
    recovery: str = RECOVERY
    adapter_config: AdapterSearchConfig = BOUNDED_LIVE_ADAPTER
    source_notes: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return self.artifact.kind

    @property
    def candidate_id(self) -> str:
        return self.artifact.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "room_id": self.room_id,
            "room": f"0x{self.room_id:04X}",
            "next_room_id": self.next_room_id,
            "next_room": f"0x{self.next_room_id:04X}",
            "recovery": self.recovery,
            "adapter_config": asdict(self.adapter_config),
            "contract": self.contract.to_dict(),
            "internal_slices": [s.to_dict() for s in self.internal_slices],
            "artifact": self.artifact.to_dict(),
            "edge": self.edge.to_dict(),
            "source_notes": list(self.source_notes),
        }


def recover_live(
    env: Any,
    runner: Any,
    *,
    config: AdapterSearchConfig | None = None,
) -> Any:
    """Bounded ``search_live_adapter`` recovery. Callers supply a live env."""
    return search_live_adapter(env, runner, config=config or BOUNDED_LIVE_ADAPTER)


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _hops_from_extract(extract: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not extract:
        return []
    hops = extract.get("room_hops") or extract.get("hops_settled") or extract.get("hops")
    if not isinstance(hops, list):
        return []
    return [dict(h) for h in hops if isinstance(h, Mapping)]


def _room_of(hop: Mapping[str, Any]) -> int | None:
    return parse_room_id(hop.get("room_id", hop.get("room")))


def _find_hop(hops: Sequence[Mapping[str, Any]], room_id: int) -> tuple[int, dict[str, Any]] | None:
    want = int(room_id)
    for i, hop in enumerate(hops):
        rid = _room_of(hop)
        if rid is not None and int(rid) == want:
            return i, dict(hop)
    return None


def _int_field(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hop_span(hop: Mapping[str, Any]) -> tuple[int, int] | None:
    """Inclusive [start, end] from extract fields. Missing bounds → None."""
    start = _int_field(hop.get("start_index"))
    if start is None:
        start = _int_field(hop.get("frame"))
    if start is None:
        return None
    end = _int_field(hop.get("end_index"))
    if end is None:
        end = _int_field(hop.get("end_frame"))
    if end is None:
        dwell = _int_field(hop.get("dwell"))
        if dwell is None:
            return None
        end = start + int(dwell) - 1
    if end < start:
        return None
    return start, end


def _split_span(start: int, end: int, max_frames: int) -> tuple[tuple[int, int], ...]:
    dwell = int(end) - int(start) + 1
    width = max(1, int(max_frames))
    if dwell <= width:
        return ((int(start), int(end)),)
    spans: list[tuple[int, int]] = []
    cur = int(start)
    last = int(end)
    while cur <= last:
        chunk_end = min(last, cur + width - 1)
        spans.append((cur, chunk_end))
        cur = chunk_end + 1
    return tuple(spans)


def _xy(hop: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not hop:
        return None
    raw = hop.get("xy")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return None


def _fingerprint(hop: Mapping[str, Any], *, prior: int | None) -> EntryFingerprint:
    xy = _xy(hop)
    room = _room_of(hop)
    assert room is not None
    pose = hop.get("pose")
    return EntryFingerprint(
        room_id=int(room),
        x=None if xy is None else xy[0],
        y=None if xy is None else xy[1],
        pose=None if pose is None else int(pose),
        items=hop_items_int(hop),
        prior_room_id=prior,
    )


def _leave(hop_key: str, next_room: int, next_hop: Mapping[str, Any] | None) -> LeaveSpec:
    xy = _xy(next_hop)
    if xy is None:
        return dest_leave_spec(hop=hop_key, room_id=int(next_room))
    x, y = xy
    band = _LEAVE_BAND
    return LeaveSpec(
        hop=hop_key,
        room=int(next_room),
        x=(x - band, x + band),
        y=(y - band, y + band),
        pose_class="any",
    )


def _rel(path: Path | str | None) -> str | None:
    return rel_path(path)


def _anchor_room(row: Mapping[str, Any]) -> int | None:
    return parse_room_id(row.get("room_id", row.get("room")))


def _pin(
    anchors: Mapping[str, Any] | None,
    frame: int,
    room_id: int,
    tape: Path,
) -> tuple[str | None, str | None]:
    """Same-room enter pin only. Other-room / Main Shaft hits do not cover."""
    if not anchors:
        return None, None
    hit = match_anchor(anchors, frame, room_id, task_path=tape)
    if hit is None:
        return None, None
    hit_room = _anchor_room(hit)
    if hit_room is None or int(hit_room) != int(room_id):
        return None, None
    resolved = resolve_anchor_path(hit, anchors_index=anchors, task_path=tape)
    if resolved is None:
        return None, None
    digest = file_digest(resolved)
    if digest is None:
        return _rel(resolved), None
    return _rel(resolved), digest


def _artifact_status(kind: str, path: Path) -> dict[str, Any]:
    exists = path.is_file()
    digest = file_digest(path) if exists else None
    missing: list[str] = []
    if not exists:
        missing.append("file")
    elif digest is None:
        missing.append("empty")
    elif kind in {"tape", "anchors", "extract"} and _safe_json(path) is None:
        missing.append("corrupt")
    return {
        "kind": kind,
        "path": _rel(path),
        "exists": exists and not missing,
        "digest": digest,
        "missing": missing,
    }


def _fail_missing(segment_dir: Path, artifacts: Sequence[Mapping[str, Any]], extra: Sequence[str] = ()) -> None:
    labels: list[str] = []
    for art in artifacts:
        miss = tuple(art.get("missing") or ())
        if not miss:
            continue
        labels.append(f"{SEGMENT}:{art['kind']}:{','.join(miss)}")
    labels.extend(extra)
    if not labels:
        return
    raise PreflightError(
        "s23 tape artifacts missing or unresolved",
        code="preflight.missing",
        details={
            "missing": labels,
            "segment": SEGMENT,
            "path": _rel(segment_dir),
            "artifacts": [dict(a) for a in artifacts],
        },
    )


def _slice_ids(task_id: str, n: int) -> tuple[str, ...]:
    if n <= 1:
        return (task_id,)
    if task_id == BOWLING_TASK_ID and n == 2:
        return (f"{task_id}:entry", f"{task_id}:leave")
    if task_id == BOWLING_TASK_ID and n == 3:
        return (f"{task_id}:entry", f"{task_id}:mid", f"{task_id}:leave")
    return tuple(f"{task_id}:{i}" for i in range(n))


def _internal_slices(
    task_id: str,
    room_id: int,
    start: int,
    end: int,
    *,
    split: bool,
    enter_pin: str | None,
    enter_digest: str | None,
    anchors: Mapping[str, Any] | None,
    tape: Path,
) -> tuple[TapeSlice, ...]:
    spans = _split_span(start, end, BOWLING_INTERNAL_MAX_FRAMES) if split else ((start, end),)
    ids = _slice_ids(task_id, len(spans))
    slices: list[TapeSlice] = []
    for i, (lo, hi) in enumerate(spans):
        pin, digest = enter_pin, enter_digest
        if i:
            pin, digest = _pin(anchors, lo, room_id, tape)
        slices.append(
            TapeSlice(
                slice_id=ids[i],
                frame_start=lo,
                frame_end=hi,
                room_id=int(room_id),
                pin_path=pin,
                pin_digest=digest,
            )
        )
    return tuple(slices)


def _build_candidate(
    *,
    task_id: str,
    hop: Mapping[str, Any],
    hops: Sequence[Mapping[str, Any]],
    index: int,
    next_room: int,
    tape: Path,
    tape_digest: str,
    anchors: Mapping[str, Any] | None,
    split: bool,
    notes: Sequence[str],
    order: int,
) -> TapeCandidate:
    room = _room_of(hop)
    assert room is not None
    pred = _room_of(hops[index - 1]) if index else None
    nxt_hop = hops[index + 1] if index + 1 < len(hops) else None
    items = hop_items_int(hop)
    span = _hop_span(hop)
    if span is None:
        raise PreflightError(
            f"s23 {task_id} hop span missing",
            code="preflight.missing",
            details={
                "missing": [f"{SEGMENT}:hop:0x{int(room):04X}:span"],
                "segment": SEGMENT,
                "task_id": task_id,
                "room": f"0x{int(room):04X}",
            },
        )
    start, end = span
    hop_key = make_hop_key(
        int(room),
        from_room_id=pred,
        to_room_id=int(next_room),
        items=items,
    )
    pin_path, pin_digest = _pin(anchors, start, int(room), tape)
    if pin_path is None or pin_digest is None:
        raise PreflightError(
            f"s23 {task_id} enter pin missing or unresolved",
            code="preflight.missing",
            details={
                "missing": [f"{SEGMENT}:hop:0x{int(room):04X}:enter_pin"],
                "segment": SEGMENT,
                "task_id": task_id,
                "room": f"0x{int(room):04X}",
                "path": _rel(tape),
            },
        )
    slices = _internal_slices(
        task_id,
        int(room),
        start,
        end,
        split=split,
        enter_pin=pin_path,
        enter_digest=pin_digest,
        anchors=anchors,
        tape=tape,
    )
    dwell = end - start + 1
    max_frames = max(int(dwell), 1)
    tape_rel = _rel(tape)
    notes_t = tuple(dict.fromkeys((*_SOURCE_NOTES, *notes)))
    entry = EntryContract(
        fingerprint=_fingerprint(hop, prior=pred),
        state_path=pin_path,
        state_digest=pin_digest,
    )
    leave = _leave(hop_key, int(next_room), nxt_hop if _room_of(nxt_hop or {}) == int(next_room) else None)
    cid = f"tape:s23_{task_id}"
    edge = RouteEdge.from_dict(
        {
            "task_id": task_id,
            "hop_key": hop_key,
            "room_id": int(room),
            "predecessor_room_id": pred,
            "next_room_id": int(next_room),
            "required_items": items,
            "entry": entry.to_dict(),
            "successor_leave": LeaveSpecRef.from_leave_spec(leave).to_dict(),
            "allowed_kinds": ["tape"],
            "selected": {"scaffold": cid},
            "owner_package": OWNER_PACKAGE,
            "integration_order": int(order),
            "max_frames": max_frames,
            "max_no_progress": max(1, min(600, max_frames)),
            "segment": SEGMENT,
            "hop_index": int(hop.get("index", hop.get("hop_index", index)) or index),
            "frame_start": start,
            "frame_end": end,
            "tape_path": tape_rel,
            "tape_digest": tape_digest,
            "source_notes": list(notes_t),
        }
    )
    art = artifact_dir(task_id)
    artifact = CandidateArtifact(
        candidate_id=cid,
        kind="tape",
        implementation_id=f"full_start_v1_segments/{SEGMENT}",
        task_id=task_id,
        entry_fingerprint=entry.fingerprint,
        source_digest=tape_digest,
        start_state_digest=pin_digest,
        tape_digest=tape_digest,
        frame_count=max_frames,
        max_no_progress=edge.max_no_progress,
        action_reasons=("tape_slice", "bounded_live_projection", RECOVERY),
        leftover_state_path=f"{art}leftover.state",
        screenshot_path=f"{art}red.png",
        trace_path=f"{art}trace.json",
    )
    return TapeCandidate(
        task_id=task_id,
        room_id=int(room),
        next_room_id=int(next_room),
        artifact=artifact,
        edge=edge,
        contract=ExternalContract(
            task_id=task_id,
            room_id=int(room),
            next_room_id=int(next_room),
            natural_entry=True,
        ),
        internal_slices=slices,
        recovery=RECOVERY,
        adapter_config=BOUNDED_LIVE_ADAPTER,
        source_notes=notes_t,
    )


def _intended_next(
    hops: Sequence[Mapping[str, Any]],
    index: int,
    expected: int,
    *,
    task_id: str,
    room_id: int,
) -> int:
    nxt = _room_of(hops[index + 1]) if index + 1 < len(hops) else None
    if nxt is None or int(nxt) != int(expected):
        got = None if nxt is None else f"0x{int(nxt):04X}"
        raise PreflightError(
            f"s23 {task_id} successor hop 0x{int(expected):04X} missing",
            code="preflight.missing",
            details={
                "missing": [f"{SEGMENT}:hop:0x{int(expected):04X}"],
                "segment": SEGMENT,
                "task_id": task_id,
                "room": f"0x{int(room_id):04X}",
                "next_room": got,
                "expected_next": f"0x{int(expected):04X}",
            },
        )
    return int(nxt)


def load_s23_tape_candidates(
    segment_dir: Path | str | None = None,
) -> tuple[TapeCandidate, TapeCandidate]:
    """Attic then Bowling from s23. Fail closed when artifacts are missing."""
    sdir = resolve_segment_dir(segment_dir)
    tape_path = sdir / "tape.json"
    anchors_path = sdir / "anchors.json"
    extract_path = sdir / "extract.json"
    artifacts = (
        _artifact_status("tape", tape_path),
        _artifact_status("anchors", anchors_path),
        _artifact_status("extract", extract_path),
    )
    _fail_missing(sdir, artifacts)

    extract = _safe_json(extract_path) or {}
    hops = _hops_from_extract(extract)
    extra: list[str] = []
    attic_hit = _find_hop(hops, ATTIC_ROOM)
    bowl_hit = _find_hop(hops, BOWLING_ROOM)
    if attic_hit is None:
        extra.append(f"{SEGMENT}:hop:0x{ATTIC_ROOM:04X}")
    if bowl_hit is None:
        extra.append(f"{SEGMENT}:hop:0x{BOWLING_ROOM:04X}")
    if extra:
        _fail_missing(sdir, artifacts, extra)

    assert attic_hit is not None and bowl_hit is not None
    attic_i, attic_hop = attic_hit
    bowl_i, bowl_hop = bowl_hit
    tape_digest = artifacts[0]["digest"]
    assert tape_digest
    anchors = load_anchors_index(tape_path) or _safe_json(anchors_path)

    attic_next = _intended_next(
        hops, attic_i, WEST_OCEAN_ROOM, task_id=ATTIC_TASK_ID, room_id=ATTIC_ROOM
    )
    bowl_next = _intended_next(
        hops, bowl_i, GRAVITY_ROOM, task_id=BOWLING_TASK_ID, room_id=BOWLING_ROOM
    )
    attic = _build_candidate(
        task_id=ATTIC_TASK_ID,
        hop=attic_hop,
        hops=hops,
        index=attic_i,
        next_room=attic_next,
        tape=tape_path,
        tape_digest=tape_digest,
        anchors=anchors,
        split=False,
        notes=(
            "Attic 0xCA52 → West Ocean 0x93FE (kill-all gray door)",
            "Scaffold HP clamp allowed later; not in this adapter",
        ),
        order=0,
    )
    bowling = _build_candidate(
        task_id=BOWLING_TASK_ID,
        hop=bowl_hop,
        hops=hops,
        index=bowl_i,
        next_room=bowl_next,
        tape=tape_path,
        tape_digest=tape_digest,
        anchors=anchors,
        split=True,
        notes=(
            "Bowling 0xC98E internal split; one external natural-entry→Gravity contract",
        ),
        order=1,
    )
    return attic, bowling


def _live_fields(live: EntryFingerprint | Mapping[str, Any]) -> tuple[int, int | None, int | None, int | None]:
    if isinstance(live, EntryFingerprint):
        return int(live.room_id), live.x, live.y, None
    raw = dict(live)
    room = parse_room_id(raw.get("room_id", raw.get("room")))
    if room is None:
        raise PreflightError(
            "live projection needs a room id",
            code="schema.room",
            details={"missing": ["live:room"]},
        )
    x = raw.get("x")
    y = raw.get("y")
    if (x is None or y is None) and isinstance(raw.get("xy"), (list, tuple)) and len(raw["xy"]) >= 2:
        x, y = raw["xy"][0], raw["xy"][1]
    frame = raw.get("frame", raw.get("sample_index"))
    return (
        int(room),
        None if x is None else int(x),
        None if y is None else int(y),
        None if frame is None else int(frame),
    )


def project_live(
    candidate: TapeCandidate,
    live: EntryFingerprint | Mapping[str, Any],
    *,
    xy_tol: int = PROJECTION_XY_TOL,
    frame_window: int = PROJECTION_FRAME_WINDOW,
) -> LiveProjection:
    """Project live RAM onto the tape slice. No emulator.

    Misses recover through ``search_live_adapter`` (not invoked here).
    """
    room, x, y, frame = _live_fields(live)
    entry = candidate.edge.entry.fingerprint
    score = 0.0
    if int(room) != int(candidate.room_id):
        return LiveProjection(
            within_bound=False,
            score=1_000_000.0,
            sample_index=None,
            recovery=RECOVERY,
            room_id=int(room),
        )
    if entry.x is not None and entry.y is not None:
        if x is None or y is None:
            return LiveProjection(
                within_bound=False,
                score=1_000_000.0,
                sample_index=candidate.edge.frame_start,
                recovery=RECOVERY,
                room_id=int(room),
            )
        score = float(abs(int(x) - int(entry.x)) + abs(int(y) - int(entry.y)))
        if score > int(xy_tol):
            return LiveProjection(
                within_bound=False,
                score=score,
                sample_index=candidate.edge.frame_start,
                recovery=RECOVERY,
                room_id=int(room),
            )
    sample = candidate.edge.frame_start
    if frame is not None:
        lo = int(candidate.edge.frame_start or 0) - int(frame_window)
        hi = int(candidate.edge.frame_end or candidate.edge.frame_start or 0) + int(frame_window)
        if frame < lo or frame > hi:
            return LiveProjection(
                within_bound=False,
                score=float(abs(frame - int(candidate.edge.frame_start or 0))),
                sample_index=sample,
                recovery=RECOVERY,
                room_id=int(room),
            )
        sample = int(frame)
    return LiveProjection(
        within_bound=True,
        score=score,
        sample_index=sample,
        recovery=None,
        room_id=int(room),
    )
