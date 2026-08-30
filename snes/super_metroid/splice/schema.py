"""Immutable route / task / candidate records for splice planning.

Cards and assembly tables are generated from one route manifest. Public path
fields are repo-relative. Does not boot an emulator or write bank.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from super_metroid.hop_id import make_hop_key, parse_room_id
from super_metroid.leave_specs import LeaveSpec
from super_metroid.splice.errors import SchemaError
from super_metroid.splice.preflight import INVALID_ROOMS, repo_relative

CANDIDATE_KINDS = ("tape", "controller", "reactive_policy", "boss")
INTERVENTION_PROFILES = ("clean", "survival", "scaffold")
MANIFEST_KIND = "super_metroid_route_manifest"
SCHEMA_VERSION = 1

FORBIDDEN_HOT_FILES: tuple[str, ...] = (
    "snes/super_metroid/routes/tips.py",
    "snes/super_metroid/routes/kpdr/spine_hops.py",
    "snes/super_metroid/routes/kpdr/tip_segments.py",
    "snes/super_metroid/routes/catalog.py",
    "snes/super_metroid/progression/",
    "snes/super_metroid/assist.py",
    "snes/super_metroid/docs/STATUS.md",
)
NON_CLAIMS: tuple[str, ...] = (
    "STATUS.md",
    "DEFAULT_CONTINUOUS_TIP",
    "Survival/Finish from Scaffold evidence",
    "skill_bank bank.json writes",
    "second runner beside tips.play_hops",
    "Main Shaft / rr-kw8t dirty files",
)
REPLAY_GREEN = "clears twice from the development anchor"
SYNC_GREEN = (
    "actual predecessor leave starts it AND exact leave passes successor "
    "LeaveSpec twice; only sync_green is route-ready"
)

def _canon(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canon(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def _digest_payload(value: Any) -> str:
    blob = json.dumps(_canon(value), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def rel_path(path: Path | str | None) -> str | None:
    """Repo-relative POSIX path; empty/None → None. Never host-absolute."""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    rel = repo_relative(raw)
    if rel is None:
        return None
    if Path(rel).is_absolute() or rel.startswith("/"):
        return Path(rel).as_posix().lstrip("/")
    return rel.replace("\\", "/")


def _require_rel(path: Path | str | None, *, field: str) -> str | None:
    rel = rel_path(path)
    if path is None or str(path).strip() == "":
        return None
    if rel is None or Path(rel).is_absolute() or rel.startswith("/"):
        raise SchemaError(
            f"{field} must be repo-relative",
            code="schema.path",
            details={"field": field, "path": str(path)},
        )
    return rel


def _fail(message: str, code: str, **details: Any) -> NoReturn:
    raise SchemaError(message, code=code, details=details)


def _int(
    value: Any,
    *,
    field: str,
    required: bool = True,
    default: int | None = None,
) -> int | None:
    if value is None or value == "":
        if required and default is None:
            _fail(f"{field} required", "schema.missing", field=field)
        return default
    if isinstance(value, bool):
        _fail(f"{field} must not be bool", "schema.type", field=field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"{field} must be int",
            code="schema.type",
            details={"field": field, "value": value},
        ) from exc


def _room(value: Any, *, field: str, required: bool = True) -> int | None:
    if value is None or value == "":
        if required:
            _fail(f"{field} required", "schema.missing", field=field)
        return None
    if isinstance(value, bool):
        _fail(f"{field} must not be bool", "schema.type", field=field)
    rid = parse_room_id(value)
    if rid is None:
        _fail(f"{field} is not a room id", "schema.room", field=field, value=value)
    return int(rid)


def _text(value: Any, *, field: str, required: bool = True) -> str | None:
    if value is None:
        if required:
            _fail(f"{field} required", "schema.missing", field=field)
        return None
    text = str(value).strip()
    if not text:
        if required:
            _fail(f"{field} must not be empty", "schema.missing", field=field)
        return None
    return text


def candidate_kind(candidate_id: str) -> str:
    """Kind encoded as ``kind`` or ``kind:identity``."""
    raw = str(candidate_id).strip()
    if not raw:
        raise SchemaError("empty candidate id", code="schema.selected")
    if raw in CANDIDATE_KINDS:
        return raw
    prefix = raw.split(":", 1)[0]
    if prefix in CANDIDATE_KINDS:
        return prefix
    raise SchemaError(
        f"candidate id {raw!r} is not in allowed kinds",
        code="schema.selected",
        details={"candidate_id": raw, "allowed_kinds": list(CANDIDATE_KINDS)},
    )


def _allowed_kinds(value: Any) -> tuple[str, ...]:
    if value is None:
        raise SchemaError("allowed_kinds required", code="schema.missing")
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise SchemaError("allowed_kinds must be a sequence", code="schema.type")
    out: list[str] = []
    for item in value:
        kind = str(item).strip()
        if not kind:
            raise SchemaError("empty candidate kind", code="schema.kind")
        if kind not in CANDIDATE_KINDS:
            raise SchemaError(
                f"unknown candidate kind {kind!r}",
                code="schema.kind",
                details={"kind": kind},
            )
        if kind not in out:
            out.append(kind)
    if not out:
        raise SchemaError("allowed_kinds must not be empty", code="schema.kind")
    return tuple(out)


def _selected(
    value: Any,
    *,
    allowed: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    rows: list[tuple[Any, Any]]
    if isinstance(value, Mapping):
        rows = list(value.items())
    elif isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise SchemaError("selected rows must be [profile, candidate_id]", code="schema.selected")
            rows.append((item[0], item[1]))
    else:
        raise SchemaError("selected must be a mapping or pair list", code="schema.selected")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for profile, cand in rows:
        prof = str(profile).strip()
        cid = str(cand).strip()
        if not prof:
            raise SchemaError("empty intervention profile", code="schema.profile")
        if prof not in INTERVENTION_PROFILES:
            raise SchemaError(
                f"unknown intervention profile {prof!r}",
                code="schema.profile",
                details={"profile": prof},
            )
        if not cid:
            raise SchemaError("empty candidate id", code="schema.selected")
        if prof in seen:
            raise SchemaError(
                f"duplicate profile {prof!r}",
                code="schema.profile",
                details={"profile": prof},
            )
        kind = candidate_kind(cid)
        if kind not in allowed:
            raise SchemaError(
                f"selected candidate id {cid!r} is not in allowed kinds {list(allowed)}",
                code="schema.selected",
                details={"candidate_id": cid, "kind": kind, "allowed_kinds": list(allowed)},
            )
        seen.add(prof)
        out.append((prof, cid))
    return tuple(out)


def leave_spec_digest(spec: LeaveSpec) -> str:
    return _digest_payload(asdict(spec))


def _xy_pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SchemaError(f"{field} must be [lo, hi]", code="schema.leave")
    lo = _int(value[0], field=f"{field}[0]")
    hi = _int(value[1], field=f"{field}[1]")
    assert lo is not None and hi is not None
    return (int(lo), int(hi))


@dataclass(frozen=True)
class Capacities:
    energy: int | None = None
    max_energy: int | None = None
    missiles: int | None = None
    max_missiles: int | None = None
    supers: int | None = None
    max_supers: int | None = None
    power_bombs: int | None = None
    max_power_bombs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Capacities:
        if not data:
            return cls()
        kwargs = {k: _int(data.get(k), field=k, required=False) for k in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass(frozen=True)
class EntryFingerprint:
    room_id: int
    x: int | None = None
    y: int | None = None
    pose: int | None = None
    velocity_x: int | None = None
    velocity_y: int | None = None
    sub_x: int | None = None
    sub_y: int | None = None
    momentum_x: int | None = None
    momentum_x_sub: int | None = None
    door_transition: int | None = None
    transition_direction: int | None = None
    speed_counter: int | None = None
    speed_flag: int | None = None
    items: int | None = None
    beams: int | None = None
    capacities: Capacities = field(default_factory=Capacities)
    boss_bits: int | None = None
    event_bits: int | None = None
    enemy_phase: str | None = None
    prior_room_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room"] = f"0x{self.room_id:04X}"
        if self.prior_room_id is not None:
            d["prior_room"] = f"0x{self.prior_room_id:04X}"
        d["capacities"] = self.capacities.to_dict()
        return _jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EntryFingerprint:
        raw = dict(data or {})
        room = _room(raw.get("room_id", raw.get("room")), field="room_id")
        assert room is not None
        ints = {
            name: _int(raw.get(name), field=name, required=False)
            for name in (
                "x",
                "y",
                "pose",
                "velocity_x",
                "velocity_y",
                "sub_x",
                "sub_y",
                "momentum_x",
                "momentum_x_sub",
                "door_transition",
                "transition_direction",
                "speed_counter",
                "speed_flag",
                "items",
                "beams",
                "boss_bits",
                "event_bits",
            )
        }
        caps = raw.get("capacities")
        return cls(
            room_id=int(room),
            capacities=Capacities.from_dict(caps if isinstance(caps, Mapping) else None),
            enemy_phase=_text(raw.get("enemy_phase"), field="enemy_phase", required=False),
            prior_room_id=_room(
                raw.get("prior_room_id", raw.get("prior_room")),
                field="prior_room_id",
                required=False,
            ),
            **ints,
        )


@dataclass(frozen=True)
class LeaveSpecRef:
    hop: str
    room: int
    digest: str
    x: tuple[int, int]
    y: tuple[int, int]
    pose_class: str = "any"
    gs: int = 8
    dt: int = 0
    boss_bit: int | None = None
    min_health: int = 1

    def to_leave_spec(self) -> LeaveSpec:
        return LeaveSpec(
            hop=self.hop,
            room=self.room,
            x=self.x,
            y=self.y,
            pose_class=self.pose_class,
            gs=self.gs,
            dt=self.dt,
            boss_bit=self.boss_bit,
            min_health=self.min_health,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room_hex"] = f"0x{self.room:04X}"
        d["x"] = list(self.x)
        d["y"] = list(self.y)
        return d

    @classmethod
    def from_leave_spec(cls, spec: LeaveSpec) -> LeaveSpecRef:
        return cls(
            hop=str(spec.hop),
            room=int(spec.room),
            digest=leave_spec_digest(spec),
            x=(int(spec.x[0]), int(spec.x[1])),
            y=(int(spec.y[0]), int(spec.y[1])),
            pose_class=str(spec.pose_class),
            gs=int(spec.gs),
            dt=int(spec.dt),
            boss_bit=None if spec.boss_bit is None else int(spec.boss_bit),
            min_health=int(spec.min_health),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeaveSpecRef:
        raw = dict(data)
        hop = _text(raw.get("hop"), field="leave.hop")
        room = _room(raw.get("room"), field="leave.room")
        assert hop is not None and room is not None
        spec = LeaveSpec(
            hop=hop,
            room=int(room),
            x=_xy_pair(raw.get("x"), field="leave.x"),
            y=_xy_pair(raw.get("y"), field="leave.y"),
            pose_class=str(raw.get("pose_class") or "any"),
            gs=int(_int(raw.get("gs"), field="leave.gs", required=False, default=8) or 8),
            dt=int(_int(raw.get("dt"), field="leave.dt", required=False, default=0) or 0),
            boss_bit=_int(raw.get("boss_bit"), field="leave.boss_bit", required=False),
            min_health=int(
                _int(raw.get("min_health"), field="leave.min_health", required=False, default=1) or 1
            ),
        )
        ref = cls.from_leave_spec(spec)
        given = _text(raw.get("digest"), field="leave.digest", required=False)
        if given and given != ref.digest:
            raise SchemaError(
                "leave digest does not match LeaveSpec",
                code="schema.leave",
                details={"digest": given, "expected": ref.digest},
            )
        return ref


@dataclass(frozen=True)
class EntryContract:
    fingerprint: EntryFingerprint
    state_path: str | None = None
    state_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "state_path": self.state_path,
            "state_digest": self.state_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EntryContract:
        raw = dict(data or {})
        fp_raw = raw.get("fingerprint")
        if isinstance(fp_raw, Mapping):
            fp = EntryFingerprint.from_dict(fp_raw)
        else:
            fp = EntryFingerprint.from_dict(raw)
        return cls(
            fingerprint=fp,
            state_path=_require_rel(raw.get("state_path"), field="state_path"),
            state_digest=_text(raw.get("state_digest"), field="state_digest", required=False),
        )


@dataclass(frozen=True)
class JoinPredicate:
    leave: LeaveSpecRef
    next_entry: EntryContract | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "leave": self.leave.to_dict(),
            "next_entry": None if self.next_entry is None else self.next_entry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JoinPredicate:
        raw = dict(data)
        leave_raw = raw.get("leave")
        if not isinstance(leave_raw, Mapping):
            raise SchemaError("join.leave required", code="schema.leave")
        nxt = raw.get("next_entry")
        return cls(
            leave=LeaveSpecRef.from_dict(leave_raw),
            next_entry=EntryContract.from_dict(nxt) if isinstance(nxt, Mapping) else None,
        )


def expected_hop_key(
    room_id: int,
    *,
    predecessor_room_id: int | None,
    next_room_id: int | None,
    items: int | None,
    goal: str | None,
) -> str:
    return make_hop_key(
        room_id,
        from_room_id=predecessor_room_id,
        to_room_id=next_room_id,
        items=items,
        goal=goal,
    )


@dataclass(frozen=True)
class RouteEdge:
    task_id: str
    hop_key: str
    room_id: int
    predecessor_room_id: int | None
    next_room_id: int | None
    entry: EntryContract
    successor_leave: LeaveSpecRef
    allowed_kinds: tuple[str, ...]
    selected: tuple[tuple[str, str], ...]
    owner_package: str
    integration_order: int
    max_frames: int
    max_no_progress: int
    goal: str | None = None
    required_items: int | None = None
    required_beams: int | None = None
    capacities: Capacities = field(default_factory=Capacities)
    boss_bits: int | None = None
    event_bits: int | None = None
    route_variant: str = "kpdr"
    predecessor_task_id: str | None = None
    successor_task_id: str | None = None
    invalid_room: bool = False
    segment: str | None = None
    hop_index: int = 0
    frame_start: int | None = None
    frame_end: int | None = None
    tape_path: str | None = None
    tape_digest: str | None = None
    source_notes: tuple[str, ...] = ()

    def selected_map(self) -> dict[str, str]:
        return {profile: cid for profile, cid in self.selected}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room"] = f"0x{self.room_id:04X}"
        if self.predecessor_room_id is not None:
            d["predecessor_room"] = f"0x{self.predecessor_room_id:04X}"
        if self.next_room_id is not None:
            d["next_room"] = f"0x{self.next_room_id:04X}"
        d["entry"] = self.entry.to_dict()
        d["successor_leave"] = self.successor_leave.to_dict()
        d["selected"] = dict(self.selected)
        d["allowed_kinds"] = list(self.allowed_kinds)
        d["capacities"] = self.capacities.to_dict()
        d["source_notes"] = list(self.source_notes)
        return _jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouteEdge:
        raw = dict(data)
        task_id = _text(raw.get("task_id"), field="task_id")
        assert task_id is not None
        room_id = _room(raw.get("room_id", raw.get("room")), field="room_id")
        assert room_id is not None
        pred_room = _room(
            raw.get("predecessor_room_id", raw.get("predecessor_room")),
            field="predecessor_room_id",
            required=False,
        )
        next_room = _room(
            raw.get("next_room_id", raw.get("next_room")),
            field="next_room_id",
            required=False,
        )
        goal = _text(raw.get("goal"), field="goal", required=False)
        items = _int(raw.get("required_items", raw.get("items")), field="required_items", required=False)
        expected = expected_hop_key(
            int(room_id),
            predecessor_room_id=pred_room,
            next_room_id=next_room,
            items=items,
            goal=goal,
        )
        hop_key = _text(raw.get("hop_key"), field="hop_key", required=False) or expected
        if hop_key != expected:
            _fail(
                f"hop_key {hop_key!r} does not match {expected!r}",
                "schema.hop_key",
                hop_key=hop_key,
                expected=expected,
            )
        allowed = _allowed_kinds(raw.get("allowed_kinds"))
        selected = _selected(raw.get("selected"), allowed=allowed)
        leave_raw = raw.get("successor_leave") or raw.get("leave")
        entry_raw = raw.get("entry")
        if not isinstance(leave_raw, Mapping):
            _fail("successor_leave required", "schema.leave")
        if not isinstance(entry_raw, Mapping):
            _fail("entry required", "schema.entry")
        owner = _require_rel(raw.get("owner_package"), field="owner_package")
        if owner is None:
            _fail("owner_package required", "schema.missing")
        notes = raw.get("source_notes") or ()
        if isinstance(notes, str):
            notes = (notes,)
        if not isinstance(notes, (list, tuple)):
            _fail("source_notes must be a sequence", "schema.type")
        max_frames = _int(raw.get("max_frames"), field="max_frames")
        max_np = _int(raw.get("max_no_progress"), field="max_no_progress")
        assert max_frames is not None and max_np is not None
        if int(max_frames) < 1 or int(max_np) < 1:
            _fail("frame budgets must be >= 1", "schema.budget")
        order = _int(raw.get("integration_order"), field="integration_order", required=False, default=0)
        assert order is not None
        return cls(
            task_id=task_id,
            hop_key=hop_key,
            room_id=int(room_id),
            predecessor_room_id=pred_room,
            next_room_id=next_room,
            entry=EntryContract.from_dict(entry_raw),
            successor_leave=LeaveSpecRef.from_dict(leave_raw),
            allowed_kinds=allowed,
            selected=selected,
            owner_package=owner,
            integration_order=int(order),
            max_frames=int(max_frames),
            max_no_progress=int(max_np),
            goal=goal,
            required_items=items,
            required_beams=_int(raw.get("required_beams"), field="required_beams", required=False),
            capacities=Capacities.from_dict(
                raw.get("capacities") if isinstance(raw.get("capacities"), Mapping) else None
            ),
            boss_bits=_int(raw.get("boss_bits"), field="boss_bits", required=False),
            event_bits=_int(raw.get("event_bits"), field="event_bits", required=False),
            route_variant=str(raw.get("route_variant") or "kpdr"),
            predecessor_task_id=_text(
                raw.get("predecessor_task_id"), field="predecessor_task_id", required=False
            ),
            successor_task_id=_text(
                raw.get("successor_task_id"), field="successor_task_id", required=False
            ),
            invalid_room=int(room_id) in INVALID_ROOMS,
            segment=_text(raw.get("segment"), field="segment", required=False),
            hop_index=int(_int(raw.get("hop_index"), field="hop_index", required=False, default=0) or 0),
            frame_start=_int(raw.get("frame_start"), field="frame_start", required=False),
            frame_end=_int(raw.get("frame_end"), field="frame_end", required=False),
            tape_path=_require_rel(raw.get("tape_path") or raw.get("tape"), field="tape_path"),
            tape_digest=_text(raw.get("tape_digest"), field="tape_digest", required=False),
            source_notes=tuple(str(n) for n in notes),
        )


def _link_edges(edges: Sequence[RouteEdge]) -> tuple[RouteEdge, ...]:
    linked: list[RouteEdge] = []
    for i, edge in enumerate(edges):
        pred = edges[i - 1].task_id if i else None
        succ = edges[i + 1].task_id if i + 1 < len(edges) else None
        given_pred = edge.predecessor_task_id
        given_succ = edge.successor_task_id
        if given_pred is not None and given_pred != pred:
            _fail(
                f"{edge.task_id} predecessor_task_id {given_pred!r} != manifest order {pred!r}",
                "schema.link",
                task_id=edge.task_id,
                expected=pred,
            )
        if given_succ is not None and given_succ != succ:
            _fail(
                f"{edge.task_id} successor_task_id {given_succ!r} != manifest order {succ!r}",
                "schema.link",
                task_id=edge.task_id,
                expected=succ,
            )
        if given_pred is None or given_succ is None:
            edge = replace(edge, predecessor_task_id=pred, successor_task_id=succ)
        linked.append(edge)
    return tuple(linked)


@dataclass(frozen=True)
class RouteManifest:
    route_id: str
    edges: tuple[RouteEdge, ...]
    variant: str = "kpdr"
    schema_version: int = SCHEMA_VERSION
    kind: str = MANIFEST_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "variant": self.variant,
            "edges": [e.to_dict() for e in self.edges],
        }

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version}",
                code="schema.version",
            )
        bad = [e.task_id for e in self.edges if e.invalid_room or e.room_id in INVALID_ROOMS]
        if bad:
            raise SchemaError(
                "invalid room 0x0000/0x5555",
                code="schema.room",
                details={"task_ids": bad},
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouteManifest:
        raw = dict(data)
        version = _int(raw.get("schema_version"), field="schema_version", required=False, default=SCHEMA_VERSION)
        assert version is not None
        if int(version) != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {version}",
                code="schema.version",
                details={"schema_version": version},
            )
        kind = str(raw.get("kind") or MANIFEST_KIND)
        if kind != MANIFEST_KIND:
            raise SchemaError(f"unknown manifest kind {kind!r}", code="schema.kind")
        route_id = _text(raw.get("route_id"), field="route_id")
        assert route_id is not None
        rows = raw.get("edges")
        if rows is None:
            raise SchemaError("edges required", code="schema.missing")
        if not isinstance(rows, (list, tuple)):
            raise SchemaError("edges must be a sequence", code="schema.type")
        edges = tuple(RouteEdge.from_dict(row) for row in rows if isinstance(row, Mapping))
        if len(edges) != len(rows):
            raise SchemaError("each edge must be an object", code="schema.type")
        ids = [e.task_id for e in edges]
        if len(ids) != len(set(ids)):
            raise SchemaError("duplicate task_id", code="schema.task_id", details={"task_ids": ids})
        return cls(
            route_id=route_id,
            edges=_link_edges(edges),
            variant=str(raw.get("variant") or "kpdr"),
            schema_version=int(version),
            kind=kind,
        )


@dataclass(frozen=True)
class CompletionReport:
    leftover_state_path: str
    screenshot_path: str
    trace_path: str
    replay_green: str = REPLAY_GREEN
    sync_green: str = SYNC_GREEN
    next_boot_on_red: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompletionReport:
        raw = dict(data)
        leftover = _require_rel(raw.get("leftover_state_path"), field="leftover_state_path")
        shot = _require_rel(raw.get("screenshot_path"), field="screenshot_path")
        trace = _require_rel(raw.get("trace_path"), field="trace_path")
        boot = _require_rel(raw.get("next_boot_on_red"), field="next_boot_on_red")
        if leftover is None or shot is None or trace is None:
            raise SchemaError("completion paths required", code="schema.missing")
        return cls(
            leftover_state_path=leftover,
            screenshot_path=shot,
            trace_path=trace,
            replay_green=str(raw.get("replay_green") or REPLAY_GREEN),
            sync_green=str(raw.get("sync_green") or SYNC_GREEN),
            next_boot_on_red=boot or leftover,
        )


@dataclass(frozen=True)
class TaskCard:
    task_id: str
    hop_key: str
    revision: int
    checkbox: str
    exact_residual: str
    entry_state_path: str | None
    entry_state_digest: str | None
    tape_digest: str | None
    segment: str | None
    hop_index: int
    frame_start: int | None
    frame_end: int | None
    source_notes: tuple[str, ...]
    entry_fingerprint: EntryFingerprint
    join: JoinPredicate
    adapter_kind: str
    intervention_profile: str
    timeout_frames: int
    commands: tuple[str, ...]
    owned_paths: tuple[str, ...]
    candidate_artifact_dir: str
    forbidden_files: tuple[str, ...]
    non_claims: tuple[str, ...]
    completion: CompletionReport
    next_task_id: str | None = None
    invalid_room: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_fingerprint"] = self.entry_fingerprint.to_dict()
        d["join"] = self.join.to_dict()
        d["completion"] = self.completion.to_dict()
        d["commands"] = list(self.commands)
        d["owned_paths"] = list(self.owned_paths)
        d["forbidden_files"] = list(self.forbidden_files)
        d["non_claims"] = list(self.non_claims)
        d["source_notes"] = list(self.source_notes)
        return _jsonable(d)


@dataclass(frozen=True)
class ReplayRow:
    trial: int
    passed: bool
    frames: int | None = None
    miss: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayRow:
        raw = dict(data)
        trial = _int(raw.get("trial"), field="replay.trial")
        assert trial is not None
        return cls(
            trial=int(trial),
            passed=bool(raw.get("passed")),
            frames=_int(raw.get("frames"), field="replay.frames", required=False),
            miss=_text(raw.get("miss"), field="replay.miss", required=False),
        )


@dataclass(frozen=True)
class JoinRow:
    trial: int
    predecessor_task_id: str
    candidate_id: str
    successor_task_id: str | None
    passed: bool
    miss: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JoinRow:
        raw = dict(data)
        trial = _int(raw.get("trial"), field="join.trial")
        pred = _text(raw.get("predecessor_task_id"), field="join.predecessor_task_id")
        cand = _text(raw.get("candidate_id"), field="join.candidate_id")
        assert trial is not None and pred is not None and cand is not None
        return cls(
            trial=int(trial),
            predecessor_task_id=pred,
            candidate_id=cand,
            successor_task_id=_text(
                raw.get("successor_task_id"), field="join.successor_task_id", required=False
            ),
            passed=bool(raw.get("passed")),
            miss=_text(raw.get("miss"), field="join.miss", required=False),
        )


@dataclass(frozen=True)
class MemoryWrite:
    frame: int
    address: int
    entity: str
    old: int
    new: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["address_hex"] = f"0x{self.address:04X}"
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryWrite:
        raw = dict(data)
        frame = _int(raw.get("frame"), field="write.frame")
        address = _int(raw.get("address"), field="write.address")
        old = _int(raw.get("old"), field="write.old")
        new = _int(raw.get("new"), field="write.new")
        entity = _text(raw.get("entity"), field="write.entity")
        reason = _text(raw.get("reason"), field="write.reason")
        assert frame is not None and address is not None and old is not None
        assert new is not None and entity is not None and reason is not None
        return cls(
            frame=int(frame),
            address=int(address),
            entity=entity,
            old=int(old),
            new=int(new),
            reason=reason,
        )


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    kind: str
    implementation_id: str
    task_id: str
    entry_fingerprint: EntryFingerprint
    source_digest: str | None = None
    card_digest: str | None = None
    rom_digest: str | None = None
    core_digest: str | None = None
    start_state_digest: str | None = None
    controller_digest: str | None = None
    tape_digest: str | None = None
    final_fingerprint: EntryFingerprint | None = None
    replay_rows: tuple[ReplayRow, ...] = ()
    join_rows: tuple[JoinRow, ...] = ()
    frame_count: int | None = None
    max_no_progress: int | None = None
    action_reasons: tuple[str, ...] = ()
    failure_class: str | None = None
    memory_writes: tuple[MemoryWrite, ...] = ()
    leftover_state_path: str | None = None
    screenshot_path: str | None = None
    trace_path: str | None = None
    parent_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_fingerprint"] = self.entry_fingerprint.to_dict()
        d["final_fingerprint"] = (
            None if self.final_fingerprint is None else self.final_fingerprint.to_dict()
        )
        d["memory_writes"] = [w.to_dict() for w in self.memory_writes]
        return _jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateArtifact:
        raw = dict(data)
        cid = _text(raw.get("candidate_id"), field="candidate_id")
        kind = candidate_kind(cid or str(raw.get("kind") or ""))
        given_kind = _text(raw.get("kind"), field="kind", required=False)
        if given_kind and given_kind != kind:
            raise SchemaError(
                f"candidate kind {given_kind!r} does not match id {cid!r}",
                code="schema.kind",
            )
        if kind not in CANDIDATE_KINDS:
            raise SchemaError(f"unknown candidate kind {kind!r}", code="schema.kind")
        impl = _text(raw.get("implementation_id"), field="implementation_id")
        task_id = _text(raw.get("task_id"), field="task_id")
        fp_raw = raw.get("entry_fingerprint")
        if not isinstance(fp_raw, Mapping):
            _fail("entry_fingerprint required", "schema.entry")
        assert cid is not None and impl is not None and task_id is not None
        final_raw = raw.get("final_fingerprint")
        digests = {
            name: _text(raw.get(name), field=name, required=False)
            for name in (
                "source_digest",
                "card_digest",
                "rom_digest",
                "core_digest",
                "start_state_digest",
                "controller_digest",
                "tape_digest",
            )
        }
        return cls(
            candidate_id=cid,
            kind=kind,
            implementation_id=impl,
            task_id=task_id,
            entry_fingerprint=EntryFingerprint.from_dict(fp_raw),
            **digests,
            final_fingerprint=EntryFingerprint.from_dict(final_raw)
            if isinstance(final_raw, Mapping)
            else None,
            replay_rows=tuple(
                ReplayRow.from_dict(r) for r in (raw.get("replay_rows") or ()) if isinstance(r, Mapping)
            ),
            join_rows=tuple(
                JoinRow.from_dict(r) for r in (raw.get("join_rows") or ()) if isinstance(r, Mapping)
            ),
            frame_count=_int(raw.get("frame_count"), field="frame_count", required=False),
            max_no_progress=_int(raw.get("max_no_progress"), field="max_no_progress", required=False),
            action_reasons=tuple(str(x) for x in (raw.get("action_reasons") or ())),
            failure_class=_text(raw.get("failure_class"), field="failure_class", required=False),
            memory_writes=tuple(
                MemoryWrite.from_dict(w)
                for w in (raw.get("memory_writes") or ())
                if isinstance(w, Mapping)
            ),
            leftover_state_path=_require_rel(raw.get("leftover_state_path"), field="leftover_state_path"),
            screenshot_path=_require_rel(raw.get("screenshot_path"), field="screenshot_path"),
            trace_path=_require_rel(raw.get("trace_path"), field="trace_path"),
            parent_candidate_id=_text(
                raw.get("parent_candidate_id"), field="parent_candidate_id", required=False
            ),
        )
