"""Artifact digest preflight (no emulator).

Snapshot hashes and availability for product-chain tapes, pins, joins, bodies,
late G4/Tourian tapes, ROM, and core. Rewrite host-absolute paths to
repo-relative at load. Report missing/corrupt artifacts, duplicate hop keys,
impossible inventory transitions, stale Gravity-gap docs, and the first
uncovered route edge.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import (
    load_anchors_index,
    match_anchor,
    parse_items_value,
    parse_room_id,
    resolve_anchor_path,
)
from super_metroid.human_tape.hops import hop_items_int
from super_metroid.human_tape.product_chain import (
    DEFAULT_BOARD,
    DEFAULT_TASK,
    _rel,
    build_product_chain_board,
    format_board_summary,
)
from super_metroid.human_tape.rta_clock import product_chain_segments
from super_metroid.human_tape.segment_archive import list_segment_ids, segments_dir_for
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR, REPO_DIR, SHARED_ROM, VANILLA_ROM_SHA1
from super_metroid.splice.errors import PreflightError

_CHUNK = 1 << 20
INVALID_ROOMS = frozenset({0x0000, 0x5555})
# Ceres / boot rooms may legally show items 0 even after a prior hop's mask.
KNOWN_DUMP_ROOMS = frozenset(
    {
        0x0000,
        0xDF45,
        0xDF8D,
        0xDFA4,
        0xDFD7,
        0xE0B5,
    }
)
POWER_ON_ALIASES = frozenset(
    {"power_on", "start", "power-on", "beginning", "full", "poweron"}
)
GRAVITY_ROOM = 0xCE40
GRAVITY_PATH_HUMAN = "gravity_path_human"
LATE_TAPES = (
    "g4_tourian_human",
    "g4_tourian_human_bb",
    "g4_tourian_human_mb",
)
STALE_DOC_PATHS = (
    GAME_DIR / "docs" / "tasks" / "FULL_STITCH_GAPS.md",
    GAME_DIR / "docs" / "tasks" / "HUMAN_TAPE_PIPELINE.md",
)
_PATH_KEYS = frozenset(
    {"path", "tape", "anchor_path", "task", "resolved_path", "written", "anchors_dir"}
)
_JSON_KINDS = frozenset({"tape", "anchors", "join", "extract", "body"})


def file_digest(path: Path | str | None) -> str | None:
    """SHA-256 of file bytes. Missing or empty → None; never raises."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size == 0:
            return None
    except OSError:
        return None
    try:
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def file_sha1(path: Path | str | None) -> str | None:
    """SHA-1 of file bytes (ROM pin vs ``VANILLA_ROM_SHA1``). Missing/empty → None."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size == 0:
            return None
    except OSError:
        return None
    try:
        digest = hashlib.sha1()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def repo_relative(path: Path | str | None, *, root: Path | str | None = None) -> str | None:
    """Repo-relative POSIX path. Never returns a host-absolute path."""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = Path(raw)
    base = Path(root).resolve() if root is not None else REPO_DIR.resolve()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p

    for candidate_root in (base, REPO_DIR.resolve(), GAME_DIR.resolve()):
        try:
            return resolved.relative_to(candidate_root).as_posix()
        except ValueError:
            continue

    if not p.is_absolute() and not raw.startswith("/"):
        game_rel = _rel(raw)
        if game_rel and not Path(game_rel).is_absolute() and not str(game_rel).startswith("/"):
            return str(game_rel).replace("\\", "/")
        return raw.replace("\\", "/")

    parts = resolved.parts
    if resolved.is_absolute() and len(parts) > 1:
        return Path(*parts[1:]).as_posix()
    return resolved.name


def _rewrite_paths(value: Any, *, root: Path | str | None = None) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _PATH_KEYS and isinstance(item, str):
                rel = repo_relative(item, root=root)
                out[str(key)] = rel if rel is not None else item
            else:
                out[str(key)] = _rewrite_paths(item, root=root)
        return out
    if isinstance(value, list):
        return [_rewrite_paths(item, root=root) for item in value]
    return value


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _json_corrupt(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return _safe_json(path) is None


def _resolve_on_disk(
    path: Path | str | None,
    *,
    extra: Sequence[Path] = (),
) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    p = Path(str(path))
    try:
        if p.is_file():
            return p
    except OSError:
        pass
    for base in (*extra, GAME_DIR, REPO_DIR, INTEGRATION_DIR):
        cand = base / p if not p.is_absolute() else p
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _artifact(
    kind: str,
    path: Path | str | None,
    *,
    root: Path | str | None = None,
    required: bool = False,
    allow_missing: bool = False,
    extra_missing: Sequence[str] = (),
) -> ArtifactRef:
    resolved = _resolve_on_disk(path)
    rel = repo_relative(resolved or path, root=root) if (resolved or path) else None
    exists = resolved is not None and resolved.is_file()
    digest = file_digest(resolved) if exists else None
    missing: list[str] = []
    if not exists:
        if required and not allow_missing:
            missing.append("file")
    elif digest is None:
        missing.append("empty")
    if exists and kind in _JSON_KINDS and _json_corrupt(resolved):  # type: ignore[arg-type]
        missing.append("corrupt")
    missing.extend(extra_missing)
    return ArtifactRef(
        kind=kind,
        path=rel,
        exists=exists,
        digest=digest,
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    path: str | None
    exists: bool
    digest: str | None
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentArtifacts:
    segment: str
    selected: bool
    tape: ArtifactRef
    anchors: ArtifactRef
    join: ArtifactRef
    extract: ArtifactRef
    body: ArtifactRef
    state: ArtifactRef

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HopPreflight:
    segment: str
    hop_index: int
    hop_key: str
    room: str
    room_id: int
    items: int | None
    enter_pin: str | None
    enter_pin_digest: str | None
    invalid_room: bool
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InventoryRegression:
    segment: str
    hop_index: int
    hop_key: str
    from_items: str
    to_items: str
    lost_bits: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RomPreflight:
    path: str | None
    exists: bool
    digest: str | None
    sha1: str | None
    expected_sha1: str
    matches_vanilla: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorePreflight:
    name: str
    version: str | None
    path: str | None
    digest: str | None
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    generated_at: str
    task: str | None
    rom: RomPreflight
    core: CorePreflight
    segments: tuple[SegmentArtifacts, ...]
    hops: tuple[HopPreflight, ...]
    late_tapes: tuple[ArtifactRef, ...]
    missing: tuple[str, ...]
    selected_missing: tuple[str, ...]
    duplicate_hop_keys: tuple[str, ...]
    impossible_inventory: tuple[InventoryRegression, ...]
    stale_docs: tuple[str, ...]
    first_uncovered_edge: dict[str, Any] | None
    gravity_path_human: dict[str, Any]
    board: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return json.loads(json.dumps(payload))


def discover_core_identity(*, root: Path | str | None = None) -> CorePreflight:
    """Best-effort core identity. Does not boot an emulator."""
    version: str | None = None
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        try:
            version = pkg_version("stable-retro")
        except PackageNotFoundError:
            version = None
    except Exception:
        version = None

    core_path: Path | None = None
    try:
        from importlib.metadata import files as pkg_files

        dist = pkg_files("stable-retro")
        if dist is not None:
            for entry in dist:
                name = str(getattr(entry, "name", "") or "").lower()
                if "snes9x" in name and Path(name).suffix in {".so", ".dylib", ".dll"}:
                    located = entry.locate()
                    if located is not None:
                        cand = Path(located)
                        if cand.is_file():
                            core_path = cand
                            break
    except Exception:
        core_path = None

    return CorePreflight(
        name="snes9x-stable-retro",
        version=version,
        path=repo_relative(core_path, root=root) if core_path else None,
        digest=file_digest(core_path),
        exists=core_path is not None and core_path.is_file(),
    )


def _rom_preflight(rom_path: Path | str | None, *, root: Path | str | None = None) -> RomPreflight:
    path = Path(rom_path) if rom_path is not None else SHARED_ROM
    sha1 = file_sha1(path)
    return RomPreflight(
        path=repo_relative(path, root=root),
        exists=path.is_file(),
        digest=file_digest(path),
        sha1=sha1,
        expected_sha1=VANILLA_ROM_SHA1,
        matches_vanilla=bool(sha1) and sha1 == VANILLA_ROM_SHA1,
    )


def _body_artifact(
    hops_dir: Path,
    *,
    root: Path | str | None = None,
    required: bool,
) -> ArtifactRef:
    files = sorted(p for p in hops_dir.glob("*.json") if p.is_file()) if hops_dir.is_dir() else []
    if not files:
        return ArtifactRef(
            kind="body",
            path=repo_relative(hops_dir, root=root),
            exists=False,
            digest=None,
            missing=("file",) if required else (),
        )
    digest = hashlib.sha256()
    for body in files:
        row = file_digest(body)
        digest.update(body.name.encode("utf-8"))
        digest.update(b":")
        digest.update((row or "").encode("ascii"))
        digest.update(b"\n")
    return ArtifactRef(
        kind="body",
        path=repo_relative(hops_dir, root=root),
        exists=True,
        digest=digest.hexdigest(),
        missing=(),
    )


def _start_state_path(join: Mapping[str, Any], sdir: Path) -> Path | None:
    raw = join.get("start_state")
    if raw is None:
        return None
    token = str(raw).strip()
    if not token or token.lower() in POWER_ON_ALIASES:
        return None
    return _resolve_on_disk(token, extra=(sdir, sdir.parent, INTEGRATION_DIR))


def _is_power_on_join(join: Mapping[str, Any]) -> bool:
    start = str(join.get("start_state") or "").strip()
    if start.lower() in POWER_ON_ALIASES:
        return True
    return bool(join.get("power_on")) and not start


def _segment_state(
    sdir: Path,
    join: Mapping[str, Any],
    anchors_idx: Mapping[str, Any] | None,
    tape_path: Path,
    *,
    root: Path | str | None = None,
    required: bool,
) -> ArtifactRef:
    start = str(join.get("start_state") or "").strip()
    if _is_power_on_join(join):
        return ArtifactRef(
            kind="state",
            path="power_on",
            exists=True,
            digest=None,
            missing=(),
        )
    pin = _start_state_path(join, sdir)
    if pin is None and anchors_idx:
        rows = anchors_idx.get("anchors") if isinstance(anchors_idx.get("anchors"), list) else []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            kind = str(row.get("kind") or "")
            if kind not in {"boot", "room_enter", "enter"}:
                continue
            resolved = resolve_anchor_path(row, anchors_index=anchors_idx, task_path=tape_path)
            if resolved is not None and resolved.is_file():
                pin = resolved
                break
    if pin is None:
        return ArtifactRef(
            kind="state",
            path=repo_relative(start, root=root) if start else None,
            exists=False,
            digest=None,
            missing=("file",) if required else (),
        )
    return _artifact("state", pin, root=root, required=required)


def _snapshot_segment(
    sdir: Path,
    label: str,
    *,
    selected: bool,
    root: Path | str | None = None,
) -> SegmentArtifacts:
    tape = sdir / "tape.json"
    anchors = sdir / "anchors.json"
    join_path = sdir / "join.json"
    extract = sdir / "extract.json"
    join = _safe_json(join_path) or {}
    anchors_idx = load_anchors_index(tape) if tape.is_file() or anchors.is_file() else None
    req = selected
    return SegmentArtifacts(
        segment=label,
        selected=selected,
        tape=_artifact("tape", tape, root=root, required=req),
        anchors=_artifact("anchors", anchors, root=root, required=req),
        join=_artifact("join", join_path, root=root, required=req),
        extract=_artifact("extract", extract, root=root, required=req),
        body=_body_artifact(sdir / "hops", root=root, required=False),
        state=_segment_state(
            sdir, join, anchors_idx, tape, root=root, required=False
        ),
    )


def _same_room(pin_room: int | None, room: int | None) -> bool:
    if pin_room is None or room is None:
        return False
    return int(pin_room) == int(room)


def _hop_pin(
    hop: Mapping[str, Any],
    tape: Path,
    anchors_idx: Mapping[str, Any] | None,
    *,
    root: Path | str | None = None,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not anchors_idx:
        return None, None, ("enter_pin",)
    room = parse_room_id(hop.get("room_id") if hop.get("room_id") is not None else hop.get("room"))
    start_i = int(hop.get("start_index") or hop.get("frame") or 0)
    hit = match_anchor(anchors_idx, start_i, room, task_path=tape) if room is not None else None
    pin_room = None
    if hit is not None:
        pin_room = parse_room_id(
            hit.get("room_id") if hit.get("room_id") is not None else hit.get("room")
        )
    if hit is not None and _same_room(pin_room, room):
        raw = hit.get("resolved_path") or hit.get("path")
        resolved = _resolve_on_disk(raw, extra=(tape.parent,))
        if resolved is None:
            return repo_relative(raw, root=root), None, ("enter_pin",)
        digest = file_digest(resolved)
        return repo_relative(resolved, root=root), digest, () if digest else ("enter_pin",)
    rows = anchors_idx.get("anchors") if isinstance(anchors_idx.get("anchors"), list) else []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("path"):
            continue
        row_room = parse_room_id(
            row.get("room_id") if row.get("room_id") is not None else row.get("room")
        )
        if not _same_room(row_room, room):
            continue
        resolved = resolve_anchor_path(row, anchors_index=anchors_idx, task_path=tape)
        if resolved is not None and resolved.is_file() and file_digest(resolved):
            continue
        return repo_relative(resolved or row.get("path"), root=root), None, ("enter_pin",)
    return None, None, ("enter_pin",)


def _inventory_regressions(hops: Sequence[Mapping[str, Any]]) -> tuple[InventoryRegression, ...]:
    out: list[InventoryRegression] = []
    prev_items: int | None = None
    prev_power_on = False
    for hop in hops:
        items = hop.get("items")
        if not isinstance(items, int):
            items = parse_items_value(hop.get("items_hex") if hop.get("items") is None else hop.get("items"))
        room = int(hop.get("room_id") or 0)
        segment = str(hop.get("segment") or "")
        power_on = bool(hop.get("_power_on"))
        known_dump = (
            room in KNOWN_DUMP_ROOMS
            or prev_items is None
            or items is None
            or (power_on and not prev_power_on)
        )
        if (
            not known_dump
            and prev_items is not None
            and items is not None
            and (prev_items & ~int(items))
        ):
            lost = prev_items & ~int(items)
            out.append(
                InventoryRegression(
                    segment=segment,
                    hop_index=int(hop.get("hop_index") or hop.get("index") or 0),
                    hop_key=str(hop.get("hop_key") or ""),
                    from_items=f"0x{int(prev_items):04X}",
                    to_items=f"0x{int(items):04X}",
                    lost_bits=f"0x{int(lost):04X}",
                )
            )
        if items is not None:
            prev_items = int(items)
        prev_power_on = power_on
    return tuple(out)


def _doc_treats_gravity_tape_as_route(text: str) -> bool:
    if "gravity_path_human" not in text:
        return False
    lower = text.lower()
    if "legacy" in lower or "not a hop board" in lower or "oracle" in lower:
        return False
    return True


def _doc_claims_missing_gravity_anchors(text: str) -> bool:
    for line in text.splitlines():
        lower = line.lower()
        if "gravity" not in lower and "0xce40" not in lower:
            continue
        if (
            "missing anchor" in lower
            or "no live anchor" in lower
            or "no live enter" in lower
        ):
            return True
    return False


def _stale_docs(hops: Sequence[HopPreflight], board_hops: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    flags: list[str] = []
    has_gravity_anchor = any(
        int(h.get("room_id") or 0) == GRAVITY_ROOM and bool(h.get("has_anchor"))
        for h in board_hops
    ) or any(h.room_id == GRAVITY_ROOM and h.enter_pin_digest for h in hops)
    for path in STALE_DOC_PATHS:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = repo_relative(path) or path.name
        if _doc_treats_gravity_tape_as_route(text):
            flags.append(
                f"{rel}: gravity_path_human is treated as a route tape; it is oracle-only"
            )
        if has_gravity_anchor and _doc_claims_missing_gravity_anchors(text):
            flags.append(
                f"{rel}: Gravity-anchor gap is stale vs generated inventory"
            )
    return tuple(dict.fromkeys(flags))


def _gravity_oracle(*, root: Path | str | None = None) -> dict[str, Any]:
    task = GAME_DIR / "tasks" / f"{GRAVITY_PATH_HUMAN}.json"
    resolved = task if task.is_file() else None
    return {
        "name": GRAVITY_PATH_HUMAN,
        "role": "oracle_only",
        "prefer": "full_start_v1_segments/s23 and s24 anchored material",
        "path": repo_relative(task, root=root),
        "exists": resolved is not None,
        "digest": file_digest(resolved),
        "note": "Legacy extract / snapshots only; not a hop board.",
    }


def _late_tape_artifacts(*, root: Path | str | None = None) -> tuple[ArtifactRef, ...]:
    rows: list[ArtifactRef] = []
    for name in LATE_TAPES:
        path = GAME_DIR / "tasks" / f"{name}.json"
        rows.append(_artifact("late_tape", path, root=root, required=False))
    return tuple(rows)


def _missing_labels(segments: Sequence[SegmentArtifacts], hops: Sequence[HopPreflight]) -> tuple[str, ...]:
    labels: list[str] = []
    for seg in segments:
        for art in (seg.tape, seg.anchors, seg.join, seg.extract, seg.body, seg.state):
            if not art.missing:
                continue
            labels.append(f"{seg.segment}:{art.kind}:{','.join(art.missing)}")
    for hop in hops:
        if hop.missing:
            labels.append(
                f"{hop.segment}:hop{hop.hop_index}:{','.join(hop.missing)}"
            )
    return tuple(labels)


def _first_uncovered(hops: Sequence[HopPreflight]) -> dict[str, Any] | None:
    for hop in hops:
        unresolved = (not hop.enter_pin_digest) or hop.invalid_room
        if unresolved:
            reasons: list[str] = []
            if hop.invalid_room:
                reasons.append(f"invalid_room:{hop.room}")
            if not hop.enter_pin_digest:
                reasons.append("missing_enter_pin")
            return {
                "segment": hop.segment,
                "hop_index": hop.hop_index,
                "hop_key": hop.hop_key,
                "room": hop.room,
                "room_id": hop.room_id,
                "enter_pin": hop.enter_pin,
                "enter_pin_digest": hop.enter_pin_digest,
                "reasons": reasons,
            }
    return None


def _selected_power_on(chain: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in chain:
        if row.get("power_on"):
            out.add(f"s{int(row['sid'])}")
    return out


def run_preflight(
    task_path: Path | str | None = None,
    *,
    include_live: bool = True,
    policy_dir: Path | str | None = None,
    bank_path: Path | str | None = None,
    rom_path: Path | str | None = None,
    write: bool = False,
    out: Path | str | None = None,
    strict: bool = False,
    repo_root: Path | str | None = None,
) -> PreflightReport:
    """Build the product-chain board, snapshot digests, rewrite absolute paths.

    Does not boot an emulator. ``strict`` raises :class:`PreflightError` when a
    selected product-chain artifact is missing or unresolved.
    """
    path = Path(task_path) if task_path is not None else DEFAULT_TASK
    kwargs: dict[str, Any] = {"include_live": include_live}
    if policy_dir is not None:
        kwargs["policy_dir"] = policy_dir
    if bank_path is not None:
        kwargs["bank_path"] = bank_path
    board = build_product_chain_board(path, **kwargs)
    board = _rewrite_paths(board, root=repo_root)

    chain, _notes = product_chain_segments(path)
    selected_sids = {int(r["sid"]) for r in chain}
    power_on_labels = _selected_power_on(chain)
    seg_root = segments_dir_for(path)
    segments: list[SegmentArtifacts] = []
    for sid in list_segment_ids(seg_root):
        segments.append(
            _snapshot_segment(
                seg_root / f"s{sid}",
                f"s{sid}",
                selected=sid in selected_sids,
                root=repo_root,
            )
        )

    if include_live and path.is_file():
        live_anchors = path.with_name(path.stem + "_anchors.json")
        live_extract = path.with_name(path.stem + "_extract.json")
        live_join = path.with_name(path.stem + "_join.json")
        hops_dir = path.with_name(path.stem + "_hops")
        segments.append(
            SegmentArtifacts(
                segment="live",
                selected=True,
                tape=_artifact("tape", path, root=repo_root, required=True),
                anchors=_artifact("anchors", live_anchors, root=repo_root, required=False),
                join=_artifact("join", live_join, root=repo_root, required=False),
                extract=_artifact("extract", live_extract, root=repo_root, required=False),
                body=_body_artifact(hops_dir, root=repo_root, required=False),
                state=ArtifactRef(
                    kind="state", path=None, exists=False, digest=None, missing=()
                ),
            )
        )

    tape_by_segment: dict[str, Path] = {}
    for row in chain:
        sid = int(row["sid"])
        tape_by_segment[f"s{sid}"] = seg_root / f"s{sid}" / "tape.json"
    if include_live:
        tape_by_segment["live"] = path

    hop_rows: list[HopPreflight] = []
    board_hops = [h for h in (board.get("hops") or []) if isinstance(h, Mapping)]
    for raw in board_hops:
        hop = dict(raw)
        segment = str(hop.get("segment") or "")
        tape = tape_by_segment.get(segment)
        if tape is None and hop.get("tape"):
            tape = _resolve_on_disk(str(hop["tape"]), extra=(path.parent, GAME_DIR)) or Path(
                str(hop["tape"])
            )
        anchors_idx = load_anchors_index(tape) if tape is not None else None
        pin_path, pin_digest, pin_missing = _hop_pin(
            hop, tape or path, anchors_idx, root=repo_root
        )
        room_id = int(hop.get("room_id") or parse_room_id(hop.get("room")) or 0)
        invalid = room_id in INVALID_ROOMS
        missing = list(pin_missing)
        if invalid:
            missing.append("invalid_room")
        hop_rows.append(
            HopPreflight(
                segment=segment,
                hop_index=int(hop.get("hop_index") or hop.get("index") or 0),
                hop_key=str(hop.get("hop_key") or ""),
                room=str(hop.get("room") or f"0x{room_id:04X}"),
                room_id=room_id,
                items=hop_items_int(hop),
                enter_pin=pin_path,
                enter_pin_digest=pin_digest,
                invalid_room=invalid,
                missing=tuple(dict.fromkeys(missing)),
            )
        )

    inventory = _inventory_regressions(
        [
            {
                "segment": h.segment,
                "hop_index": h.hop_index,
                "hop_key": h.hop_key,
                "room_id": h.room_id,
                "items": h.items,
                "_power_on": h.segment in power_on_labels,
            }
            for h in hop_rows
        ]
    )
    counts = Counter(h.hop_key for h in hop_rows if h.hop_key)
    duplicates = tuple(sorted(key for key, n in counts.items() if n > 1))

    late = _late_tape_artifacts(root=repo_root)
    gravity = _gravity_oracle(root=repo_root)
    hops_t = tuple(hop_rows)
    segs_t = tuple(segments)
    missing = _missing_labels(segs_t, hops_t)
    selected_missing = tuple(
        label
        for seg in segs_t
        for art in (seg.tape, seg.anchors, seg.join, seg.extract)
        if seg.selected and art.missing
        for label in (f"{seg.segment}:{art.kind}:{','.join(art.missing)}",)
    )
    # Index pointed at a pin path that does not resolve to bytes.
    pin_missing = tuple(
        f"{h.segment}:hop{h.hop_index}:enter_pin"
        for h in hops_t
        if h.enter_pin and not h.enter_pin_digest
    )
    selected_missing = tuple(dict.fromkeys((*selected_missing, *pin_missing)))

    if write:
        dest = Path(out) if out is not None else DEFAULT_BOARD
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
        board["written"] = repo_relative(dest, root=repo_root)

    report = PreflightReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        task=repo_relative(path, root=repo_root),
        rom=_rom_preflight(rom_path, root=repo_root),
        core=discover_core_identity(root=repo_root),
        segments=segs_t,
        hops=hops_t,
        late_tapes=late,
        missing=missing,
        selected_missing=selected_missing,
        duplicate_hop_keys=duplicates,
        impossible_inventory=inventory,
        stale_docs=_stale_docs(hops_t, board_hops),
        first_uncovered_edge=_first_uncovered(hops_t),
        gravity_path_human=gravity,
        board=board,
    )
    if strict and report.selected_missing:
        raise PreflightError(
            "selected artifacts missing or unresolved by digest",
            details={"missing": list(report.selected_missing)},
        )
    return report


def format_preflight_summary(report: PreflightReport) -> str:
    """Human-readable preflight + product-board summary."""
    rom = report.rom
    core = report.core
    rom_line = (
        f"rom exists={rom.exists} sha1={rom.sha1 or '-'} "
        f"vanilla={'yes' if rom.matches_vanilla else 'no'} digest={rom.digest or '-'}"
    )
    core_line = (
        f"core {core.name} version={core.version or '-'} "
        f"exists={core.exists} digest={core.digest or '-'}"
    )
    lines = [
        "splice preflight (no emulator)",
        f"  task {report.task or '-'}",
        f"  {rom_line}",
        f"  {core_line}",
        f"  segments={len(report.segments)} hops={len(report.hops)}",
        f"  selected_missing={len(report.selected_missing)} "
        f"duplicates={len(report.duplicate_hop_keys)} "
        f"inventory_regressions={len(report.impossible_inventory)}",
    ]
    if report.selected_missing:
        lines.append("  missing selected:")
        for label in report.selected_missing[:12]:
            lines.append(f"    - {label}")
        extra = len(report.selected_missing) - 12
        if extra > 0:
            lines.append(f"    … {extra} more")
    if report.duplicate_hop_keys:
        lines.append("  duplicate hop keys: " + ", ".join(report.duplicate_hop_keys[:8]))
    if report.impossible_inventory:
        row = report.impossible_inventory[0]
        lines.append(
            f"  inventory: {row.hop_key} {row.from_items} → {row.to_items} "
            f"(lost {row.lost_bits})"
        )
    if report.stale_docs:
        lines.append("  stale docs:")
        for flag in report.stale_docs:
            lines.append(f"    - {flag}")
    edge = report.first_uncovered_edge
    if not report.hops:
        lines.append("  first uncovered: no hops inventoried")
    elif edge:
        lines.append(
            f"  first uncovered: {edge.get('segment')} hop {edge.get('hop_index')} "
            f"{edge.get('room')} {edge.get('hop_key')} "
            f"reasons={','.join(edge.get('reasons') or ())}"
        )
    else:
        lines.append("  first uncovered: none (all hops digest-resolvable, rooms valid)")
    g = report.gravity_path_human
    lines.append(
        f"  {g.get('name')}: {g.get('role')} — {g.get('note')} "
        f"(prefer {g.get('prefer')})"
    )
    if report.board:
        lines.append(format_board_summary(report.board))
    return "\n".join(lines)
