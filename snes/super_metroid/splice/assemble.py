"""Assemble selected candidates through ``tips.play_hops`` in one session.

``assemble(route_id, selection)`` projects planner-selected candidates into
SpineHop play callables and drives the existing Composer. It never loads a
room state during the active assembly. Candidates whose intervention profile
does not match the assembly profile fail closed. Injectable session /
play_hops keep tests ROM-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, Sequence

from super_metroid.splice.errors import AssembleError, SchemaError, SelectError
from super_metroid.splice.schema import (
    INTERVENTION_PROFILES,
    CandidateArtifact,
    RouteEdge,
    RouteManifest,
    candidate_kind,
)
from super_metroid.splice.select import (
    CandidateOffer,
    Selection,
    as_offer,
    as_selection,
)

_LOAD_NAMES = ("load", "load_state", "load_state_data", "restore_state", "set_state")
_SURVIVAL_TOKS = ("energy", "ammo", "missile", "super", "power_bomb", "resource")
PlayHops = Callable[..., Any]


@dataclass(frozen=True)
class AssemblyHop:
    """SpineHop-shaped play row. play_hops only needs these fields."""

    hop_id: str
    play: Callable[[Any], Any]
    from_room: int
    to_room: int
    room_label: str
    tip_id: str
    use_transition_split: bool = False
    after: Callable[..., None] | None = None
    leave: Any = None


HopFactory = Callable[[RouteEdge, CandidateOffer], Any]


@dataclass(frozen=True)
class Assembly:
    """One continuous play_hops result. Not a promotion and not a bank write."""

    route_id: str
    profile: str
    selected: tuple[tuple[str, str], ...]
    hop_ids: tuple[str, ...]
    result: Any = None
    session: Any = None

    def selected_map(self) -> dict[str, str]:
        return {task: cid for task, cid in self.selected}

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "profile": self.profile,
            "selected": dict(self.selected),
            "hop_ids": list(self.hop_ids),
        }


def _fail(message: str, code: str, **details: Any) -> NoReturn:
    raise AssembleError(message, code=code, details=details)


def _as_manifest(manifest: RouteManifest | Mapping[str, Any]) -> RouteManifest:
    if isinstance(manifest, RouteManifest):
        return manifest
    try:
        return RouteManifest.from_dict(manifest)
    except SchemaError as exc:
        raise AssembleError(
            str(exc),
            code=exc.code or "assemble.manifest",
            details=exc.details,
        ) from exc


def _require_profile(profile: str) -> str:
    prof = str(profile).strip()
    if not prof or prof not in INTERVENTION_PROFILES:
        _fail(
            f"unknown intervention profile {profile!r}",
            "assemble.profile",
            profile=profile,
        )
    return prof


def _is_scaffold_write(reason: str) -> bool:
    text = str(reason).lower()
    return "scaffold" in text or "hp_clamp" in text or ("enemy" in text and "hp" in text)


def _is_survival_write(reason: str) -> bool:
    text = str(reason).lower()
    return any(tok in text for tok in _SURVIVAL_TOKS) and not _is_scaffold_write(text)


def write_allowed(reason: str, profile: str) -> bool:
    """Fail closed for unknown write reasons under the assembly profile."""
    if profile == "clean":
        return False
    if profile == "survival":
        return _is_survival_write(reason)
    if profile == "scaffold":
        return _is_scaffold_write(reason) or _is_survival_write(reason)
    return False


def _check_offer(offer: CandidateOffer, *, profile: str, edge: RouteEdge) -> None:
    if offer.profile != profile:
        _fail(
            f"candidate {offer.candidate_id!r} profile {offer.profile!r} "
            f"does not match assembly profile {profile!r}",
            "assemble.profile",
            candidate_id=offer.candidate_id,
            candidate_profile=offer.profile,
            profile=profile,
            task_id=edge.task_id,
        )
    if offer.kind not in edge.allowed_kinds:
        _fail(
            f"candidate kind {offer.kind!r} is not allowed for {edge.task_id}",
            "assemble.kind",
            candidate_id=offer.candidate_id,
            kind=offer.kind,
            allowed_kinds=list(edge.allowed_kinds),
        )
    for write in offer.artifact.memory_writes:
        if not write_allowed(write.reason, profile):
            _fail(
                f"candidate {offer.candidate_id!r} write {write.reason!r} "
                f"is not allowed under {profile}",
                "assemble.profile",
                candidate_id=offer.candidate_id,
                reason=write.reason,
                profile=profile,
            )


def _synthetic_offer(edge: RouteEdge, candidate_id: str, profile: str) -> CandidateOffer:
    try:
        kind = candidate_kind(candidate_id)
    except SchemaError as exc:
        raise AssembleError(str(exc), code="assemble.selected", details=exc.details) from exc
    impl = candidate_id.split(":", 1)[1] if ":" in candidate_id else candidate_id
    artifact = CandidateArtifact(
        candidate_id=candidate_id,
        kind=kind,
        implementation_id=impl,
        task_id=edge.task_id,
        entry_fingerprint=edge.entry.fingerprint,
    )
    return CandidateOffer(artifact=artifact, profile=profile)


def _offer_for(
    edge: RouteEdge,
    candidate_id: str,
    *,
    profile: str,
    selection: Selection,
    extras: Sequence[CandidateOffer],
) -> CandidateOffer:
    # Same kind:id is reused across profiles. Bind only a matching-profile
    # artifact; a clean-graded tape:a0 must not block scaffold tape:a0.
    found = selection.offer_for(edge.task_id, candidate_id, profile=profile)
    if found is not None:
        return found
    for offer in extras:
        if (
            offer.task_id == edge.task_id
            and offer.candidate_id == candidate_id
            and offer.profile == profile
        ):
            return offer
    other = [
        offer
        for offer in (*selection.offers, *extras)
        if offer.task_id == edge.task_id and offer.candidate_id == candidate_id
    ]
    incumbent = edge.selected_map().get(profile)
    if other and candidate_id != incumbent:
        _fail(
            f"candidate {candidate_id!r} profile {other[0].profile!r} "
            f"does not match assembly profile {profile!r}",
            "assemble.profile",
            candidate_id=candidate_id,
            candidate_profile=other[0].profile,
            profile=profile,
            task_id=edge.task_id,
        )
    return _synthetic_offer(edge, candidate_id, profile)


def project_hop(edge: RouteEdge, offer: CandidateOffer) -> AssemblyHop:
    """SpineHop-like row whose play never loads a room state."""

    def play(session: Any) -> None:
        # Adapters (tape / Skill / policy) land in later PRs. Projection only.
        _ = session
        _ = offer
        return None

    dest = edge.next_room_id if edge.next_room_id is not None else edge.room_id
    return AssemblyHop(
        hop_id=edge.task_id,
        play=play,
        from_room=int(edge.room_id),
        to_room=int(dest),
        room_label=edge.task_id,
        tip_id=edge.task_id,
        use_transition_split=False,
        leave=edge.successor_leave.to_leave_spec(),
    )


def _guard_loads(session: Any) -> Callable[[], None]:
    """Block env/session save-state loads for the duration of play_hops."""

    patched: list[tuple[Any, str, Any]] = []

    def blocked(method: str) -> Callable[..., Any]:
        def _blocked(*args: Any, **kwargs: Any) -> Any:
            _fail(
                "assemble never loads a room state during the active assembly",
                "assemble.load",
                method=method,
            )

        return _blocked

    env = getattr(session, "env", None)
    targets = [session, env]
    if env is not None:
        targets.append(getattr(env, "em", None))
        targets.append(getattr(env, "unwrapped", None))
        targets.append(getattr(env, "data", None))
    seen: set[int] = set()
    for obj in targets:
        if obj is None:
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        for name in _LOAD_NAMES:
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            try:
                setattr(obj, name, blocked(name))
            except (AttributeError, TypeError):
                for pobj, pname, orig in patched:
                    try:
                        setattr(pobj, pname, orig)
                    except (AttributeError, TypeError):
                        continue
                _fail(
                    "assemble cannot shadow mid-run state load",
                    "assemble.load",
                    method=name,
                )
            patched.append((obj, name, fn))

    def restore() -> None:
        for obj, name, orig in patched:
            try:
                setattr(obj, name, orig)
            except (AttributeError, TypeError):
                continue

    return restore


def _default_play_hops() -> PlayHops:
    from super_metroid.routes.tips import play_hops as _play_hops

    return _play_hops


def assemble(
    route_id: str,
    selection: Selection | Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    manifest: RouteManifest | Mapping[str, Any] | None = None,
    profile: str | None = None,
    candidates: Sequence[CandidateOffer | CandidateArtifact | Mapping[str, Any]] = (),
    play_hops: PlayHops | None = None,
    session: Any | None = None,
    session_factory: Callable[[], Any] | None = None,
    hop_factory: HopFactory | None = None,
    splits: list[Any] | None = None,
) -> Assembly:
    """Run selected candidates in one emulator session via play_hops."""
    if manifest is None:
        _fail("assemble requires a route manifest", "assemble.manifest", route_id=route_id)
    route = _as_manifest(manifest)
    if route.route_id != route_id:
        _fail(
            f"manifest route {route.route_id!r} does not match {route_id!r}",
            "assemble.route",
            expected=route_id,
            actual=route.route_id,
        )
    if isinstance(selection, Selection):
        prof = _require_profile(profile or selection.profile)
        if selection.profile != prof:
            _fail(
                f"selection profile {selection.profile!r} does not match {prof!r}",
                "assemble.profile",
                expected=prof,
                actual=selection.profile,
            )
    else:
        prof = _require_profile(profile or "scaffold")
    try:
        extras = tuple(as_offer(item, default_profile=prof) for item in candidates)
        chosen = as_selection(selection, route_id=route_id, profile=prof, offers=extras)
    except SelectError as exc:
        code = exc.code.replace("select.", "assemble.") if exc.code.startswith("select.") else exc.code
        raise AssembleError(str(exc), code=code, details=exc.details) from exc

    selected_map = chosen.selected_map()
    hops: list[Any] = []
    hop_ids: list[str] = []
    factory = hop_factory or project_hop
    for edge in route.edges:
        cid = selected_map.get(edge.task_id)
        if not cid:
            _fail(
                f"no selected candidate for {edge.task_id}",
                "assemble.selected",
                task_id=edge.task_id,
                profile=prof,
            )
        offer = _offer_for(edge, cid, profile=prof, selection=chosen, extras=extras)
        _check_offer(offer, profile=prof, edge=edge)
        hops.append(factory(edge, offer))
        hop_ids.append(edge.task_id)

    if session is None and session_factory is None:
        _fail(
            "assemble refuses to boot without a session factory",
            "assemble.session",
            route_id=route_id,
            profile=prof,
        )
    live = session if session is not None else session_factory()
    runner = play_hops if play_hops is not None else _default_play_hops()
    split_buf: list[Any] = [] if splits is None else splits
    restore = _guard_loads(live)
    try:
        result = runner(live, split_buf, hops, segments=None)
    except AssembleError:
        raise
    except Exception as exc:
        raise AssembleError(
            f"play_hops failed: {exc}",
            code="assemble.play",
            details={"error": type(exc).__name__},
        ) from exc
    finally:
        restore()
    return Assembly(
        route_id=route_id,
        profile=prof,
        selected=chosen.selected,
        hop_ids=tuple(hop_ids),
        result=result,
        session=live,
    )
