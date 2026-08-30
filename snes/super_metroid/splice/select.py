"""Planner-only candidate selection. Never writes bank.json.

Workers emit candidate artifacts. ``select`` chooses ids per intervention
profile. Rollback is selecting the previous candidate id, not rewriting code.
If A's new leave cannot start B, A+B become one change and old A stays
selected until the pair is green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, Sequence

from super_metroid.splice.errors import SchemaError, SelectError
from super_metroid.splice.schema import (
    INTERVENTION_PROFILES,
    CandidateArtifact,
    RouteEdge,
    RouteManifest,
    candidate_kind,
)

# Prefer in-route Skills over tape when both are sync_green for the profile.
_KIND_RANK = {"controller": 0, "reactive_policy": 1, "boss": 2, "tape": 3}


@dataclass(frozen=True)
class CandidateOffer:
    """Worker-emitted candidate bound to the profile it was graded under."""

    artifact: CandidateArtifact
    profile: str

    @property
    def candidate_id(self) -> str:
        return self.artifact.candidate_id

    @property
    def task_id(self) -> str:
        return self.artifact.task_id

    @property
    def kind(self) -> str:
        return self.artifact.kind


@dataclass(frozen=True)
class Selection:
    """Planner rollup of selected candidate ids. Not a bank.json write."""

    route_id: str
    profile: str
    selected: tuple[tuple[str, str], ...]
    previous: tuple[tuple[str, str], ...] = ()
    coupled: tuple[tuple[str, str], ...] = ()
    offers: tuple[CandidateOffer, ...] = ()

    def selected_map(self) -> dict[str, str]:
        return {task: cid for task, cid in self.selected}

    def previous_map(self) -> dict[str, str]:
        return {task: cid for task, cid in self.previous}

    def offer_for(
        self,
        task_id: str,
        candidate_id: str,
        profile: str | None = None,
    ) -> CandidateOffer | None:
        for offer in self.offers:
            if offer.task_id != task_id or offer.candidate_id != candidate_id:
                continue
            if profile is not None and offer.profile != profile:
                continue
            return offer
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "profile": self.profile,
            "selected": dict(self.selected),
            "previous": dict(self.previous),
            "coupled": [list(pair) for pair in self.coupled],
        }


def _fail(message: str, code: str, **details: Any) -> NoReturn:
    raise SelectError(message, code=code, details=details)


def _as_manifest(manifest: RouteManifest | Mapping[str, Any]) -> RouteManifest:
    if isinstance(manifest, RouteManifest):
        return manifest
    try:
        return RouteManifest.from_dict(manifest)
    except SchemaError as exc:
        raise SelectError(str(exc), code=exc.code or "select.manifest", details=exc.details) from exc


def _require_profile(profile: str, *, code: str = "select.profile") -> str:
    prof = str(profile).strip()
    if not prof or prof not in INTERVENTION_PROFILES:
        _fail(
            f"unknown intervention profile {profile!r}",
            code,
            profile=profile,
        )
    return prof


def as_offer(
    raw: CandidateOffer | CandidateArtifact | Mapping[str, Any],
    *,
    default_profile: str,
) -> CandidateOffer:
    """Bind a worker artifact to its intervention profile. Does not write."""
    if isinstance(raw, CandidateOffer):
        return raw
    if isinstance(raw, CandidateArtifact):
        return CandidateOffer(artifact=raw, profile=default_profile)
    if not isinstance(raw, Mapping):
        _fail("candidate offer must be an object", "select.candidate")
    payload = dict(raw)
    tagged = payload.get("intervention_profile", payload.get("profile"))
    profile = _require_profile(str(tagged).strip() if tagged else default_profile)
    try:
        artifact = CandidateArtifact.from_dict(payload)
    except SchemaError as exc:
        raise SelectError(
            str(exc),
            code=exc.code or "select.candidate",
            details=exc.details,
        ) from exc
    return CandidateOffer(artifact=artifact, profile=profile)


def as_selection(
    value: Selection | Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    route_id: str,
    profile: str,
    offers: Sequence[CandidateOffer] = (),
    previous: Mapping[str, str] | None = None,
) -> Selection:
    """Coerce a mapping or pair list into a planner Selection."""
    if isinstance(value, Selection):
        if value.route_id != route_id:
            _fail(
                f"selection route {value.route_id!r} does not match {route_id!r}",
                "select.route",
                expected=route_id,
                actual=value.route_id,
            )
        if value.profile != profile:
            _fail(
                f"selection profile {value.profile!r} does not match {profile!r}",
                "select.profile",
                expected=profile,
                actual=value.profile,
            )
        return value
    rows: list[tuple[str, str]]
    if isinstance(value, Mapping):
        rows = [(str(k), str(v)) for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                _fail("selection rows must be [task_id, candidate_id]", "select.selected")
            rows.append((str(item[0]), str(item[1])))
    else:
        _fail("selection must be a mapping or pair list", "select.selected")
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for task_id, cid in rows:
        task = str(task_id).strip()
        cand = str(cid).strip()
        if not task or not cand:
            _fail("empty selected task or candidate id", "select.selected")
        try:
            candidate_kind(cand)
        except SchemaError as exc:
            raise SelectError(str(exc), code="select.selected", details=exc.details) from exc
        if task in seen:
            _fail(f"duplicate task {task!r}", "select.selected", task_id=task)
        seen.add(task)
        selected.append((task, cand))
    prev = tuple((str(k), str(v)) for k, v in dict(previous or {}).items() if str(k) and str(v))
    return Selection(
        route_id=route_id,
        profile=profile,
        selected=tuple(selected),
        previous=prev,
        offers=tuple(offers),
    )


def _replay_green(offer: CandidateOffer) -> bool:
    return sum(1 for row in offer.artifact.replay_rows if row.passed) >= 2


def _sync_rows(offer: CandidateOffer, successor_task_id: str | None) -> tuple[Any, ...]:
    cid = offer.candidate_id
    rows = []
    for row in offer.artifact.join_rows:
        if not row.passed or row.candidate_id != cid:
            continue
        if successor_task_id is None or row.successor_task_id == successor_task_id:
            rows.append(row)
    return tuple(rows)


def _sync_green(offer: CandidateOffer, successor_task_id: str | None) -> bool:
    if len(_sync_rows(offer, successor_task_id)) >= 2:
        return True
    if successor_task_id is None:
        return _replay_green(offer)
    return False


def _leave_starts_successor(offer: CandidateOffer, successor_task_id: str | None) -> bool:
    """False unless Join evidence shows this leave can start B."""
    if successor_task_id is None:
        return True
    return len(_sync_rows(offer, successor_task_id)) >= 2


def _rank(offer: CandidateOffer) -> tuple[int, int, str]:
    frames = offer.artifact.frame_count
    frame_key = int(frames) if frames is not None else 10**9
    kind_key = _KIND_RANK.get(offer.kind, 9)
    return (kind_key, frame_key, offer.candidate_id)


def _pick(
    edge: RouteEdge,
    *,
    profile: str,
    offers: Sequence[CandidateOffer],
    incumbent: str | None,
) -> tuple[str, bool]:
    """Return (candidate_id, coupled_to_successor).

    A replacement is selected only when it is sync_green with the successor.
    Otherwise the incumbent stays and A+B become one change.
    """
    matching = [
        o
        for o in offers
        if o.task_id == edge.task_id
        and o.profile == profile
        and o.kind in edge.allowed_kinds
    ]
    successor = edge.successor_task_id
    challengers = [o for o in matching if o.candidate_id != incumbent]
    green = [o for o in challengers if _sync_green(o, successor)]
    if green:
        return min(green, key=_rank).candidate_id, False
    blocked = bool(successor and challengers)
    if incumbent:
        # Old A stays until the pair is green. A+B become one change.
        return incumbent, blocked
    if matching:
        # Hole with no incumbent: pick best replay, still coupled if a successor exists.
        ready = [o for o in matching if _replay_green(o)] or list(matching)
        return min(ready, key=_rank).candidate_id, bool(successor)
    _fail(
        f"no selected candidate for {edge.task_id} profile {profile}",
        "select.selected",
        task_id=edge.task_id,
        profile=profile,
    )


def select(
    manifest: RouteManifest | Mapping[str, Any],
    candidates: Sequence[CandidateOffer | CandidateArtifact | Mapping[str, Any]] = (),
    *,
    profile: str = "scaffold",
    previous: Mapping[str, str] | None = None,
) -> Selection:
    """Choose selected ids per profile. Never mutates the manifest or bank.json."""
    prof = _require_profile(profile)
    route = _as_manifest(manifest)
    offers = tuple(as_offer(item, default_profile=prof) for item in candidates)
    selected: list[tuple[str, str]] = []
    prev_out: list[tuple[str, str]] = []
    coupled: list[tuple[str, str]] = []
    passed_prev = {str(k): str(v) for k, v in dict(previous or {}).items()}
    for edge in route.edges:
        incumbent = passed_prev.get(edge.task_id) or edge.selected_map().get(prof)
        chosen, couple = _pick(edge, profile=prof, offers=offers, incumbent=incumbent)
        selected.append((edge.task_id, chosen))
        if incumbent:
            prev_out.append((edge.task_id, incumbent))
        if couple and edge.successor_task_id:
            coupled.append((edge.task_id, edge.successor_task_id))
    return Selection(
        route_id=route.route_id,
        profile=prof,
        selected=tuple(selected),
        previous=tuple(prev_out),
        coupled=tuple(coupled),
        offers=offers,
    )


def rollback(selection: Selection, task_id: str | None = None) -> Selection:
    """Restore previous candidate id(s). Data-only; never rewrites code or bank.json."""
    if not isinstance(selection, Selection):
        _fail("selection required", "select.selection")
    current = selection.selected_map()
    previous = selection.previous_map()
    tasks = (task_id,) if task_id is not None else tuple(current)
    selected: list[tuple[str, str]] = []
    prev_out: list[tuple[str, str]] = []
    for task, cid in selection.selected:
        if task in tasks and task in previous:
            restored = previous[task]
            selected.append((task, restored))
            prev_out.append((task, cid))
        else:
            selected.append((task, cid))
            if task in previous:
                prev_out.append((task, previous[task]))
    return Selection(
        route_id=selection.route_id,
        profile=selection.profile,
        selected=tuple(selected),
        previous=tuple(prev_out),
        coupled=selection.coupled,
        offers=selection.offers,
    )
