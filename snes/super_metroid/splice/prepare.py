"""Fail-closed task prepare: validate artifacts and fingerprints, never boot.

``prepare(task_id)`` binds a card to digest-resolved artifacts. Missing or
mismatched ROM/core/start-state, inventory, boss/event bits, room, pose /
position / velocity, predecessor, intended exit, or required tape raises
:class:`PrepareError` before any emulator session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import (
    load_anchors_index,
    parse_room_id,
    resolve_anchor_path,
)
from super_metroid.paths import SHARED_ROM
from super_metroid.source_states import match_source_by_path
from super_metroid.splice.cards import generate_cards
from super_metroid.splice.errors import PrepareError, SchemaError
from super_metroid.splice.manifest import load_manifest, manifest_from_board
from super_metroid.splice.preflight import (
    INVALID_ROOMS,
    ArtifactRef,
    _artifact,
    _hop_pin,
    _resolve_on_disk,
    _same_room,
    discover_core_identity,
    file_digest,
    run_preflight,
)
from super_metroid.splice.schema import (
    INTERVENTION_PROFILES,
    EntryFingerprint,
    RouteEdge,
    RouteManifest,
    TaskCard,
)

_FP_REQUIRED: tuple[tuple[str, str], ...] = (
    ("items", "inventory"),
    ("x", "position"),
    ("y", "position"),
    ("pose", "pose"),
    ("velocity_x", "velocity"),
    ("velocity_y", "velocity"),
    ("boss_bits", "boss"),
    ("event_bits", "event"),
)


@dataclass(frozen=True)
class PreparedTask:
    """Immutable bind of a task card to digest-resolved artifacts."""

    task_id: str
    card: TaskCard
    artifacts: tuple[ArtifactRef, ...]
    entry_fingerprint: EntryFingerprint
    intervention_profile: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "intervention_profile": self.intervention_profile,
            "entry_fingerprint": self.entry_fingerprint.to_dict(),
            "source_id": self.source_id,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "card": self.card.to_dict(),
        }


def _as_manifest(manifest: RouteManifest | Mapping[str, Any] | Path | str) -> RouteManifest:
    if isinstance(manifest, RouteManifest):
        return manifest
    if isinstance(manifest, Mapping):
        return RouteManifest.from_dict(manifest)
    return load_manifest(manifest)


def _extra_roots(repo_root: Path | str | None) -> tuple[Path, ...]:
    if repo_root is None:
        return ()
    return (Path(repo_root),)


def _ref(
    kind: str,
    path: Path | str | None,
    *,
    repo_root: Path | str | None,
    extra: Sequence[Path],
    required: bool,
) -> ArtifactRef:
    found = _resolve_on_disk(path, extra=extra)
    return _artifact(kind, found or path, root=repo_root, required=required)


def _core_ref(
    core_path: Path | str | None,
    *,
    repo_root: Path | str | None,
    extra: Sequence[Path],
) -> ArtifactRef:
    if core_path is not None:
        return _ref("core", core_path, repo_root=repo_root, extra=extra, required=True)
    discovered = discover_core_identity(root=repo_root)
    missing: tuple[str, ...] = () if discovered.exists and discovered.digest else ("file",)
    return ArtifactRef(
        kind="core",
        path=discovered.path,
        exists=discovered.exists,
        digest=discovered.digest,
        missing=missing,
    )


def _anchor_room(row: Mapping[str, Any]) -> int | None:
    return parse_room_id(row.get("room_id") if row.get("room_id") is not None else row.get("room"))


def _pin_from_anchors(
    edge: RouteEdge,
    *,
    extra: Sequence[Path],
    repo_root: Path | str | None,
) -> Path | None:
    """Same-room enter pin only. Other-room / unlabelled rows do not cover."""
    tape = _resolve_on_disk(edge.tape_path, extra=extra)
    if tape is None:
        return None
    idx = load_anchors_index(tape)
    hop = {
        "room_id": edge.room_id,
        "room": edge.room_id,
        "start_index": edge.frame_start or 0,
        "frame": edge.frame_start or 0,
    }
    pin_path, pin_digest, missing = _hop_pin(hop, tape, idx, root=repo_root)
    if missing or not pin_digest:
        return None
    resolved = _resolve_on_disk(pin_path, extra=(tape.parent, *tuple(extra)))
    if resolved is None:
        return None
    rows = idx.get("anchors") if isinstance(idx.get("anchors"), list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_path = resolve_anchor_path(row, anchors_index=idx, task_path=tape)
        if row_path is None:
            continue
        try:
            same_file = row_path.resolve() == resolved.resolve()
        except OSError:
            same_file = False
        if same_file and _same_room(_anchor_room(row), edge.room_id):
            return resolved
    return None


def _mismatch(expected: str | None, actual: str | None, *, kind: str) -> str | None:
    if expected is None:
        return f"{kind}:digest"
    if actual is None:
        return f"{kind}:file"
    if actual != expected:
        return f"{kind}:digest"
    return None


def _room_issue(room_id: int) -> str | None:
    if int(room_id) in INVALID_ROOMS:
        return f"invalid_room:0x{int(room_id):04X}"
    return None


def _fingerprint_issues(fp: EntryFingerprint) -> list[str]:
    issues: list[str] = []
    bad = _room_issue(fp.room_id)
    if bad:
        issues.append(bad)
    seen: set[str] = set()
    for field, label in _FP_REQUIRED:
        if getattr(fp, field) is None and label not in seen:
            issues.append(f"{label}:missing")
            seen.add(label)
    return issues


def _catalog_issues(fp: EntryFingerprint, pin: Path | None) -> tuple[list[str], str | None]:
    if pin is None:
        return [], None
    source = match_source_by_path(pin)
    if source is None:
        return [], None
    issues: list[str] = []
    if int(source.room_id) != int(fp.room_id):
        issues.append(
            f"room:catalog:expected=0x{source.room_id:04X}:got=0x{fp.room_id:04X}"
        )
    if fp.x is None or fp.y is None:
        issues.append("position:missing")
    else:
        if source.x_min is not None and fp.x < source.x_min:
            issues.append(f"position:x<{source.x_min}")
        if source.x_max is not None and fp.x > source.x_max:
            issues.append(f"position:x>{source.x_max}")
        if source.y_min is not None and fp.y < source.y_min:
            issues.append(f"position:y<{source.y_min}")
        if source.y_max is not None and fp.y > source.y_max:
            issues.append(f"position:y>{source.y_max}")
    if source.poses is not None:
        if fp.pose is None:
            issues.append("pose:missing")
        elif fp.pose not in source.poses:
            issues.append(f"pose:{fp.pose}")
    return issues, source.source_id


def _require_equal(edge_val: int | None, fp_val: int | None, *, label: str) -> list[str]:
    if edge_val is None or fp_val is None:
        return [f"{label}:missing"]
    if int(edge_val) != int(fp_val):
        return [f"{label}:mismatch"]
    return []


def _contract_issues(edge: RouteEdge, fp: EntryFingerprint, card: TaskCard) -> list[str]:
    issues: list[str] = []
    issues.extend(_require_equal(edge.required_items, fp.items, label="inventory"))
    issues.extend(_require_equal(edge.boss_bits, fp.boss_bits, label="boss"))
    issues.extend(_require_equal(edge.event_bits, fp.event_bits, label="event"))
    pred_room = edge.predecessor_room_id
    if pred_room is not None:
        if fp.prior_room_id is None:
            issues.append("predecessor:missing")
        elif int(fp.prior_room_id) != int(pred_room):
            issues.append("predecessor:mismatch")
    leave = card.join.leave
    bad_leave = _room_issue(leave.room)
    if bad_leave:
        issues.append(f"exit:{bad_leave}")
    if not leave.digest or leave.x is None or leave.y is None:
        issues.append("exit:missing")
    if edge.next_room_id is not None and int(leave.room) != int(edge.next_room_id):
        issues.append("exit:mismatch")
    nxt = card.join.next_entry
    if edge.successor_task_id and nxt is None:
        issues.append("exit:next_entry")
    return issues


def _tape_required(edge: RouteEdge, card: TaskCard) -> bool:
    if card.adapter_kind == "tape":
        return True
    if edge.tape_path or edge.tape_digest or card.tape_digest:
        return True
    return False


def _load_route(
    manifest: RouteManifest | Mapping[str, Any] | Path | str | None,
    *,
    chain: Path | str | None,
    include_live: bool,
    rom_path: Path | str | None,
    repo_root: Path | str | None,
) -> RouteManifest:
    if manifest is not None:
        return _as_manifest(manifest)
    report = run_preflight(
        chain,
        include_live=include_live,
        rom_path=rom_path,
        repo_root=repo_root,
        strict=False,
    )
    return manifest_from_board(report.board)


def prepare(
    task_id: str,
    *,
    manifest: RouteManifest | Mapping[str, Any] | Path | str | None = None,
    profile: str = "scaffold",
    revision: int = 1,
    rom_path: Path | str | None = None,
    core_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    chain: Path | str | None = None,
    include_live: bool = True,
) -> PreparedTask:
    """Validate ``task_id`` and bind artifacts. Never starts an emulator."""
    if not str(task_id).strip():
        raise PrepareError("task_id required", code="prepare.task")
    if profile not in INTERVENTION_PROFILES:
        raise PrepareError(
            f"unknown intervention profile {profile!r}",
            code="prepare.profile",
            details={"profile": profile},
        )
    try:
        route = _load_route(
            manifest,
            chain=chain,
            include_live=include_live,
            rom_path=rom_path,
            repo_root=repo_root,
        )
        cards = generate_cards(route, profile=profile, revision=revision)
    except SchemaError as exc:
        raise PrepareError(
            str(exc),
            code=exc.code or "prepare.schema",
            details=exc.details,
        ) from exc

    card = next((c for c in cards if c.task_id == task_id), None)
    edge = next((e for e in route.edges if e.task_id == task_id), None)
    if card is None or edge is None:
        raise PrepareError(
            f"task {task_id!r} not in manifest",
            code="prepare.task",
            details={"task_id": task_id},
        )

    extra = _extra_roots(repo_root)
    issues: list[str] = []
    if not str(edge.selected_map().get(profile) or "").strip():
        issues.append(f"profile:unselected:{profile}")
    if card.invalid_room or edge.invalid_room or edge.room_id in INVALID_ROOMS:
        issues.append(f"invalid_room:0x{int(edge.room_id):04X}")

    rom = _ref(
        "rom",
        rom_path if rom_path is not None else SHARED_ROM,
        repo_root=repo_root,
        extra=extra,
        required=True,
    )
    if rom.missing or not rom.digest:
        issues.append(f"rom:{','.join(rom.missing) or 'file'}")

    core = _core_ref(core_path, repo_root=repo_root, extra=extra)
    if core.missing or not core.digest:
        issues.append(f"core:{','.join(core.missing) or 'file'}")

    pin_path = card.entry_state_path or edge.entry.state_path
    # Named path is authoritative: do not fall back to another-room tape pin.
    if pin_path:
        pin_file = _resolve_on_disk(pin_path, extra=extra)
    else:
        pin_file = _pin_from_anchors(edge, extra=extra, repo_root=repo_root)
    pin = _ref("state", pin_file or pin_path, repo_root=repo_root, extra=extra, required=True)
    expected_pin = card.entry_state_digest or edge.entry.state_digest
    actual_pin = file_digest(pin_file) if pin_file is not None else pin.digest
    pin_issue = _mismatch(expected_pin, actual_pin, kind="entry")
    if pin_file is None and not pin.exists:
        issues.append("entry_pin:file")
    elif pin_issue == "entry:digest" and expected_pin is None:
        issues.append("entry_digest:missing")
    elif pin_issue == "entry:digest":
        issues.append("entry_digest:mismatch")
    elif pin_issue == "entry:file":
        issues.append("entry_pin:file")

    tape_path = edge.tape_path
    need_tape = _tape_required(edge, card)
    tape = _ref("tape", tape_path, repo_root=repo_root, extra=extra, required=need_tape)
    expected_tape = card.tape_digest or edge.tape_digest
    if need_tape:
        if tape.missing:
            issues.append(f"tape:{','.join(tape.missing)}")
        tape_issue = _mismatch(expected_tape, tape.digest, kind="tape")
        if tape_issue is not None:
            issues.append(tape_issue)

    fp = card.entry_fingerprint
    issues.extend(_fingerprint_issues(fp))
    catalog_issues, source_id = _catalog_issues(fp, pin_file)
    issues.extend(catalog_issues)
    issues.extend(_contract_issues(edge, fp, card))

    issues = list(dict.fromkeys(issues))
    if issues:
        raise PrepareError(
            f"task {task_id!r} cannot be prepared before boot",
            details={"task_id": task_id, "missing": issues},
        )

    artifacts = tuple(a for a in (rom, core, pin, tape) if a.kind != "tape" or need_tape)
    return PreparedTask(
        task_id=task_id,
        card=card,
        artifacts=artifacts,
        entry_fingerprint=fp,
        intervention_profile=profile,
        source_id=source_id,
    )
