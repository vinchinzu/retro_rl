"""Ownership leases and planner-only candidate rollup.

A lease records task id, card revision, branch, owner paths, and expiry.
Two active leases may not overlap source paths. Artifact directories never
overlap among active leases. Rollup selects candidate ids and never writes
bank.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from super_metroid.splice.cards import artifact_dir as task_artifact_dir
from super_metroid.splice.errors import SpliceError
from super_metroid.splice.lanes import Lane, lane_artifact_dir, lane_owner_package
from super_metroid.splice.schema import (
    INTERVENTION_PROFILES,
    CandidateArtifact,
    RouteManifest,
    TaskCard,
    candidate_kind,
    rel_path,
)


class LeaseError(SpliceError):
    """Lease request rejected (overlap, missing fields, or bank write)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "lease.overlap",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


def _norm_path(path: str | None) -> str:
    rel = rel_path(path) if path else None
    text = (rel or str(path or "")).replace("\\", "/").strip()
    return text.rstrip("/")


def paths_overlap(left: str, right: str) -> bool:
    """True when paths are equal or one is a directory prefix of the other."""
    a, b = _norm_path(left), _norm_path(right)
    if not a or not b:
        return False
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _parse_expiry(value: str | None) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LeaseError(
            f"invalid expiry {value!r}",
            code="lease.expiry",
            details={"expiry": value},
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class Lease:
    """Coordinator record for one worktree/branch card."""

    task_id: str
    card_revision: int
    branch: str
    owner_paths: tuple[str, ...]
    expiry: str | None
    artifact_dir: str
    lane_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "card_revision": self.card_revision,
            "branch": self.branch,
            "owner_paths": list(self.owner_paths),
            "expiry": self.expiry,
            "artifact_dir": self.artifact_dir,
            "lane_id": self.lane_id,
        }


def _require_lease_fields(lease: Lease) -> None:
    if not str(lease.task_id).strip():
        raise LeaseError("task_id required", code="lease.task_id")
    if int(lease.card_revision) < 1:
        raise LeaseError("card_revision must be >= 1", code="lease.revision")
    if not str(lease.branch).strip():
        raise LeaseError("branch required", code="lease.branch")
    if not _norm_path(lease.artifact_dir):
        raise LeaseError("artifact_dir required", code="lease.artifact")
    if not lease.owner_paths:
        raise LeaseError("owner_paths required", code="lease.owner")


def is_active(lease: Lease, *, now: datetime | None = None) -> bool:
    expiry = _parse_expiry(lease.expiry)
    if expiry is None:
        return True
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return expiry > clock


def _owner_collision(request: Lease, other: Lease) -> str | None:
    for left in request.owner_paths:
        for right in other.owner_paths:
            if paths_overlap(left, right):
                return right
    return None


def _artifact_collision(request: Lease, other: Lease) -> bool:
    return paths_overlap(request.artifact_dir, other.artifact_dir)


@dataclass(frozen=True)
class GrantResult:
    granted: bool
    lease: Lease | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "granted": self.granted,
            "lease": None if self.lease is None else self.lease.to_dict(),
            "reason": self.reason,
        }


def grant_lease(
    request: Lease,
    existing: Sequence[Lease] = (),
    *,
    now: datetime | None = None,
) -> GrantResult:
    """Grant when owner paths and artifact dirs are disjoint from active leases."""
    _require_lease_fields(request)
    clock = now or datetime.now(timezone.utc)
    for other in existing:
        if not is_active(other, now=clock):
            continue
        hit = _owner_collision(request, other)
        if hit is not None:
            return GrantResult(
                False,
                None,
                f"owner_paths overlap with {other.task_id} ({hit})",
            )
        if _artifact_collision(request, other):
            return GrantResult(
                False,
                None,
                f"artifact_dir overlap with {other.task_id} ({other.artifact_dir})",
            )
    return GrantResult(True, request, "")


def lease_for_lane(
    lane: Lane,
    *,
    branch: str,
    card_revision: int = 1,
    expiry: str | None = None,
    task_id: str | None = None,
) -> Lease:
    """One coordinator lease covering a lane's unique owner and artifact dir."""
    owner = rel_path(lane.owner_package) or lane.owner_package or lane_owner_package(
        lane.lane_id
    )
    art = rel_path(lane.artifact_dir) or lane.artifact_dir or lane_artifact_dir(
        lane.lane_id
    )
    return Lease(
        task_id=task_id or (lane.task_ids[0] if lane.task_ids else lane.lane_id),
        card_revision=int(card_revision),
        branch=str(branch),
        owner_paths=(owner,),
        expiry=expiry,
        artifact_dir=art,
        lane_id=lane.lane_id,
    )


def lease_from_card(
    card: TaskCard,
    *,
    branch: str,
    expiry: str | None = None,
    lane_id: str | None = None,
    owner_paths: Sequence[str] | None = None,
) -> Lease:
    """Lease from an immutable card. Default owned_paths may share a package."""
    owned = tuple(owner_paths) if owner_paths is not None else tuple(card.owned_paths)
    art = card.candidate_artifact_dir or task_artifact_dir(card.task_id)
    return Lease(
        task_id=card.task_id,
        card_revision=int(card.revision),
        branch=str(branch),
        owner_paths=tuple(_norm_path(p) or p for p in owned if p),
        expiry=expiry,
        artifact_dir=rel_path(art) or art,
        lane_id=lane_id,
    )


@dataclass(frozen=True)
class SelectionRollup:
    """Planner-owned candidate ids. Never a bank.json write."""

    profile: str
    selected: tuple[tuple[str, str], ...]
    skipped: tuple[str, ...] = ()

    def as_map(self) -> dict[str, str]:
        return {task_id: cid for task_id, cid in self.selected}


def _as_candidate(row: CandidateArtifact | Mapping[str, Any]) -> CandidateArtifact:
    if isinstance(row, CandidateArtifact):
        return row
    return CandidateArtifact.from_dict(row)


def _candidate_score(cand: CandidateArtifact) -> tuple[int, int, int, str]:
    replay_ok = sum(1 for row in cand.replay_rows if row.passed)
    join_ok = sum(1 for row in cand.join_rows if row.passed)
    frames = cand.frame_count if cand.frame_count is not None else 10**9
    return (-replay_ok, -join_ok, int(frames), cand.candidate_id)


def rollup_candidates(
    manifest: RouteManifest,
    candidates: Sequence[CandidateArtifact | Mapping[str, Any]] = (),
    *,
    profile: str = "scaffold",
) -> SelectionRollup:
    """Select candidate ids per task. Planner-only; does not write bank.json."""
    if not str(profile).strip() or profile not in INTERVENTION_PROFILES:
        raise LeaseError(
            f"unknown intervention profile {profile!r}",
            code="lease.profile",
            details={"profile": profile},
        )
    grouped: dict[str, list[CandidateArtifact]] = {}
    for row in candidates:
        cand = _as_candidate(row)
        grouped.setdefault(cand.task_id, []).append(cand)
    selected: list[tuple[str, str]] = []
    skipped: list[str] = []
    for edge in manifest.edges:
        allowed = set(edge.allowed_kinds)
        pool = [
            c
            for c in grouped.get(edge.task_id, ())
            if c.kind in allowed or candidate_kind(c.candidate_id) in allowed
        ]
        if pool:
            best = min(pool, key=_candidate_score)
            selected.append((edge.task_id, best.candidate_id))
            continue
        fallback = edge.selected_map().get(profile, "")
        if fallback:
            selected.append((edge.task_id, fallback))
        else:
            skipped.append(edge.task_id)
    return SelectionRollup(profile=profile, selected=tuple(selected), skipped=tuple(skipped))
