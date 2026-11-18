"""Assemble selected candidates through ``tips.play_hops`` in one session.

``assemble(route_id, selection)`` projects planner-selected candidates into
SpineHop play callables and drives the existing Composer. It never loads a
room state during the active assembly. Candidates whose intervention profile
does not match the assembly profile fail closed. Injectable session /
play_hops keep tests ROM-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        # Remaining tape adapters stay stubs. Gravity collect binds in
        # ``ranges.range_hop_factory``.
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


def _guard_loads(session: Any, counter: list[int] | None = None) -> Callable[[], None]:
    """Block env/session save-state loads for the duration of play_hops."""

    patched: list[tuple[Any, str, Any]] = []

    def blocked(method: str) -> Callable[..., Any]:
        def _blocked(*args: Any, **kwargs: Any) -> Any:
            if counter is not None:
                counter[0] += 1
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


def _guard_writes(env: Any, counter: list[int] | None = None) -> Callable[[], None]:
    """Reject external RAM writes; the core can still advance RAM normally."""
    patched, data = [], env.data

    def blocked(*args: Any, **kwargs: Any) -> NoReturn:
        if counter is not None:
            counter[0] += 1
        raise RuntimeError("Clean tape playback forbids RAM writes")

    class ReadOnlyMemory:
        assign = staticmethod(blocked)

        @property
        def blocks(self) -> Any:
            return {key: memoryview(block).toreadonly() for key, block in data.memory.blocks.items()}

        def __getattr__(self, name: str) -> Any:
            return getattr(data.memory, name)

    class ReadOnlyData:
        memory = ReadOnlyMemory()
        set_value = staticmethod(blocked)

        def __getattr__(self, name: str) -> Any:
            return getattr(data, name)

    # GameData.memory and Memory.assign are read-only pybind attributes.
    env.data = ReadOnlyData()
    for obj, name in ((env, "set_value"), (env, "set_ram"), (env.em, "set_ram")):
        original = getattr(obj, name, None)
        if callable(original):
            setattr(obj, name, blocked)
            patched.append((obj, name, original))

    def restore() -> None:
        for obj, name, original in patched:
            setattr(obj, name, original)
        env.data = data

    return restore


def _chain_session(env: Any) -> Any:
    from super_metroid.assist import UnlimitedAmmoAssist
    from super_metroid.progression import EARLY_GAME_GRAPH
    from super_metroid.routes.runtime import RouteSession

    return RouteSession(
        env, writer=None, assist=UnlimitedAmmoAssist(enabled=False), graph=EARLY_GAME_GRAPH
    )


def _same_chain_state(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    import json

    # Dataclass tuples become lists in JSON artifacts.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _chain_rows(board: Path) -> list[dict[str, Any]]:
    """Keep board row numbers stable, excluding boot garbage and stale live rows."""
    import json

    return [
        dict(hop, row=index)
        for index, hop in enumerate(json.loads(board.read_text())["hops"])
        if (hop["segment"] != "live" or "source_start" in hop) and int(hop["room"], 16) != 0x5555
    ]


def _chain_body(row: Mapping[str, Any], cache: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Trace F describes the state *after* input F: resume with F+1.

    Include the original door inputs through the successor's settled frame.
    Idling through that interval changes door momentum and breaks chaining.
    Segment-final rows retain their item-collection/tail inputs.
    """
    import hashlib
    import numpy as np
    from super_metroid.human_tape.hops import load_room_hops, load_task_json
    from super_metroid.paths import GAME_DIR

    path = Path(row["tape"])
    if not path.is_absolute():
        path = GAME_DIR / path
    key = str(path)
    if key not in cache:
        data = load_task_json(path)
        cache[key] = (data, load_room_hops(task_data=data), np.asarray(data["frames"], dtype=np.int8), hashlib.sha256(path.read_bytes()).hexdigest())
    data, hops, frames, digest = cache[key]
    index = int(row["hop_index"])
    hop = hops[index]
    if int(hop["room_id"]) != int(row["room"], 16):
        raise ValueError(f"board row {row['row']} does not match its tape slice")
    start = int(hop["start_index"]) + 1
    if int(row["room"], 16) == 0:
        start = 0  # Power-on includes the two boot-garbage samples.
    end = int(hops[index + 1]["start_index"]) if index + 1 < len(hops) else len(frames) - 1
    final = data["trace"][end]
    start = int(row.get("source_start", start))
    end = int(row.get("source_end", end))
    if not 0 <= start <= end < len(frames):
        raise ValueError(f"invalid source bounds for board row {row['row']}: {start}..{end}")
    final = data["trace"][end]
    reference = [dict(sample, action=frames[i + 1]) for i, sample in enumerate(data["trace"][start - 1 : min(end, start + 95)], start=start - 1)] if start else []
    return frames[start : end + 1], dict(final, source_start=start, source_end=end, source_sha256=digest, reference=reference)


def _chain_goal(state: Any, row: Mapping[str, Any], final: Mapping[str, Any]) -> bool:
    target = int(row.get("to_room") or row["room"], 16)
    items = int(str(final.get("items", 0)), 0)
    beams = int(str(final.get("beams", 0)), 0)
    ordinary = state.game_state == 8 and not state.door_transition
    if row.get("goal_phase") == "ending_or_credits":
        # Reaching the ending is a prefix, not evidence of end-of-credits.
        ordinary = state.phase.value == "ending_or_credits"
    if not (state.room_id == target and ordinary and state.health > 0):
        return False
    if state.collected_items & items != items or state.collected_beams & beams != beams:
        return False
    if row.get("goal_phase") == "ending_or_credits":
        return True
    if not row.get("to_room") or row.get("continuation"):
        # A segment tail may collect an item or position a same-room handoff.
        return abs(state.samus_x - int(final["x"])) <= 32 and abs(state.samus_y - int(final["y"])) <= 32
    return True


def _try_chain_body(env: Any, body: Any, row: Mapping[str, Any], final: Mapping[str, Any]) -> Any | None:
    """Pure input candidate; return its full buttons, including settle inputs."""
    import numpy as np
    from super_metroid.ram import GameplayPhase, parse_env_state

    actions = []
    if body.ndim != 2 or body.shape[1] != 12 or len(body) == 0 or not np.isin(body, (0, 1)).all():
        raise ValueError(f"board row {row['row']} requires nonempty binary SNES-12 actions")
    target = int(row.get("to_room") or row["room"], 16)
    for action in body:
        env.step(action)
        actions.append(action)
        state = parse_env_state(env, mode="nav")
        if state.phase is GameplayPhase.DEATH_OR_GAME_OVER or (state.game_state == 8 and state.health <= 0):
            return None
    if _chain_goal(state, row, final):
        return np.asarray(actions, dtype=np.int8)
    if not row.get("to_room") or row.get("continuation"):
        return None
    idle = np.zeros(12, dtype=np.int8)
    for _ in range(400):
        # Keep the final press until the door starts; then settle with no input.
        action = idle if state.room_id == target else body[-1]
        env.step(action)
        actions.append(action)
        state = parse_env_state(env, mode="nav")
        if state.phase is GameplayPhase.DEATH_OR_GAME_OVER or state.health <= 0:
            return None
        if _chain_goal(state, row, final):
            return np.asarray(actions, dtype=np.int8)
    return None


def _chain_controller(env: Any, row: Mapping[str, Any]) -> tuple[Any, int] | None:
    """Existing room owners supply searched or reactive button candidates."""
    import importlib
    import numpy as np
    from super_metroid.ram import parse_env_state

    room, to_room, controller = int(row["room"], 16), row.get("to_room"), row.get("controller_id")
    owners = {
        (0x9804, "0x9879", False): ("super_metroid.combat.bomb_torizo", "play_clean_bomb_torizo_leave"),
        (0x9D19, "0x9D9C", False): ("super_metroid.routes.kpdr.brinstar.big_pink", "play_clean_big_pink_leave"),
        (0x9DC7, "0x9B5B", False): ("super_metroid.routes.kpdr.brinstar.spore_spawn", "play_clean_spore_spawn_leave"),
        (0xA447, "0xA408", False): ("super_metroid.routes.kpdr.spazer.collect", "play_spazer_return_to_below"),
        (0xA59F, "0xA6E2", False): ("super_metroid.combat.kraid", "play_clean_kraid_leave"),
        (0xA6E2, None, True): ("super_metroid.combat.kraid", "play_varia_collect"),
        (0xA6E2, "0xA59F", False): ("super_metroid.routes.kpdr.kraid.varia_return", "play_varia_to_kraid"),
        (0xA59F, "0xA56B", False): ("super_metroid.routes.kpdr.kraid.varia_return", "play_kraid_to_eye_return"),
        (0xDA60, "0xD9AA", False): ("super_metroid.combat.draygon", "play_clean_draygon_leave"),
    }
    by_id = {
        "moat_cross": ("super_metroid.routes.kpdr.wrecked_ship.moat", "play_moat_cross"),
        "west_ocean_to_ws": ("super_metroid.routes.kpdr.wrecked_ship.west_ocean", "play_west_ocean_over_ocean_spark"),
    }
    spec = owners.get((room, to_room, bool(row.get("continuation")))) or by_id.get(controller or "")
    if spec is None:
        return None
    play = getattr(importlib.import_module(spec[0]), spec[1])
    actions = []

    class CandidateSession:
        def __init__(self) -> None:
            self.env, self.frame, self.state = env, 0, parse_env_state(env, mode="nav")

        def step(self, action: Any, reason: str = "") -> Any:
            env.step(action)
            self.frame += 1
            self.state = parse_env_state(env, mode="nav")
            actions.append(np.asarray(action, dtype=np.int8).copy())
            if self.state.health <= 0:
                raise RuntimeError("controller candidate died")
            return self.state

    result = play(CandidateSession())
    search = getattr(result, "fight", None)
    restores = int(getattr(search, "rollout_restores", 0))
    return np.asarray(actions, dtype=np.int8), restores


def _search_chain_body(env: Any, body: Any, row: Mapping[str, Any], final: Mapping[str, Any], *, max_delay: int) -> tuple[Any, str, int]:
    """Tape, bounded retimes, then reference-aware live entry search."""
    import numpy as np
    from super_metroid.ram import parse_env_state
    from super_metroid.reactive_policy import KinematicWeights, PolicyVariant, ReactivePolicyRunner, ReferenceSample, ReferenceTrajectory
    from super_metroid.room_adapter import AdapterSearchConfig, search_live_adapter

    original = env.em.get_state()
    restores = 0
    idle = np.zeros((1, 12), dtype=np.int8)

    def attempt(actions: Any) -> Any:
        nonlocal restores
        env.em.set_state(original)
        restores += 1
        return _try_chain_body(env, actions, row, final)

    try:
        for delay in range(max_delay + 1):
            actions = np.concatenate((np.repeat(idle, delay, axis=0), body))
            found = attempt(actions)
            if found is not None:
                return found, f"tape+idle:{delay}", restores
        env.em.set_state(original)
        restores += 1
        try:
            controlled = _chain_controller(env, row)
        except (RuntimeError, TimeoutError):
            controlled = None
        if controlled is not None:
            actions, controller_restores = controlled
            restores += controller_restores
            found = attempt(actions)
            if found is not None:
                return found, "room-controller", restores
        for trim in range(1, min(max_delay + 1, len(body))):
            found = attempt(body[trim:])
            if found is not None:
                return found, f"tape-trim:{trim}", restores

        # Search onto the human entry trajectory. Missing trace features carry
        # zero weight; every proposed rejoin must clear the full room in ROM.
        env.em.set_state(original)
        restores += 1
        samples = tuple(ReferenceSample.from_dict(sample) for sample in final.get("reference", []))
        if samples:
            runner = ReactivePolicyRunner(PolicyVariant(
                variant_id="chain_entry", trajectories=(ReferenceTrajectory("entry", samples),),
                weights=KinematicWeights(velocity=1, momentum=0, facing=0, movement_type=0, vertical_direction=0), rejoin_threshold=1,
            ))
            plan = search_live_adapter(env, runner, config=AdapterSearchConfig(beam_width=8, max_depth=4))
            restores += plan.expanded + 1
            if plan.frames:
                env.em.set_state(original)
                restores += 1
                for action in plan.frames:
                    env.step(np.asarray(action, dtype=np.int8))
                offset = runner.project_global(parse_env_state(env, mode="nav")).sample_index
                found = attempt(np.concatenate((np.asarray(plan.frames, dtype=np.int8), body[offset:])))
                if found is not None:
                    return found, f"live-adapter:{plan.expanded}", restores
        unique = {tuple(int(bit) for bit in action) for action in body[:96]}
        unique.add(tuple(int(bit) for bit in idle[0]))
        for index in (6, 7):
            action = [0] * 12
            action[index] = 1
            unique.add(tuple(action))
        for length in (2, 4, 8, 16, 32):
            for action in sorted(unique):
                prefix = np.repeat(np.asarray([action], dtype=np.int8), length, axis=0)
                found = attempt(np.concatenate((prefix, body)))
                if found is not None:
                    return found, f"entry-pulse:{length}:{action}", restores
        error = RuntimeError(f"no candidate clears board row {row['row']} ({row['name']})")
        error.rollout_restores = restores + 1
        raise error
    finally:
        env.em.set_state(original)


def solve_tape_chain(*, board: Path, stem: Path, through_row: int = 9999, max_delay: int = 12, resume: bool = True, overrides: Mapping[int, Path] | None = None) -> dict[str, Any]:
    """Develop one master tape through the Composer, then prove it from power-on.

    Development restores never enter RouteSession's frame/split ledger. Only
    accepted actions advance the Composer. Resume replays the accepted prefix
    from power-on; it never substitutes a human room state. The post-launch
    input tail follows the ending-entry row in the same tape.
    """
    import hashlib
    import json
    import time
    from datetime import datetime, timezone
    import numpy as np
    from super_metroid.combat.escape import play_credits_tail
    from super_metroid.dev.common import make_dev_env
    from super_metroid.paths import SCRATCH_STATE_DIR, SHARED_ROM, VANILLA_ROM_SHA1
    from super_metroid.ram import credits_complete, parse_env_state
    from super_metroid.routes.tips import play_hops
    from retro_harness.env import write_state_bytes

    if not stem.name.endswith("_clean"):
        raise ValueError("Clean chain artifacts require a _clean stem")
    if max_delay < 0:
        raise ValueError("max_delay must be nonnegative")
    rom_bytes = SHARED_ROM.read_bytes()
    rom_sha1 = hashlib.sha1(rom_bytes).hexdigest()
    if rom_sha1 != VANILLA_ROM_SHA1:
        raise ValueError(f"Clean chain requires vanilla ROM ({VANILLA_ROM_SHA1}), got {rom_sha1}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    report_path, tape_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if resume and report_path.is_file() != tape_path.is_file():
        raise ValueError("incomplete master tape artifacts; restore the pair or use --fresh")
    rows, cache, ledger, master = _chain_rows(board), {}, [], []
    report = {
        "kind": "clean_tape_chain", "schema_version": 1, "board": str(board),
        "rows": ledger, "rollout_restores": 0, "stopped": None,
        "credits_complete": False, "credits_tail_frames": 0,
        "rom_sha1": rom_sha1, "rom_sha256": hashlib.sha256(rom_bytes).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    env = make_dev_env()
    env.reset()
    restore_writes = _guard_writes(env)
    session = _chain_session(env)
    splits = []
    started = time.monotonic()

    def persist() -> None:
        actions = np.asarray(master, dtype=np.int8).reshape((-1, 12))
        np.savez_compressed(tape_path, actions=actions)
        report.update(
            total_frames=session.frame,
            rooms_cleared=sum(bool(r.get("to_room")) and r["room"] != "0x0000" for r in ledger),
            final_state=parse_env_state(env, frame=session.frame).to_dict(),
            state_parse="full_bank7e", seconds=round(time.monotonic() - started, 2),
            tape_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
            endpoint_sha256=hashlib.sha256(env.em.get_state()).hexdigest(),
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        write_state_bytes(SCRATCH_STATE_DIR / f"{stem.name}.state", env.em.get_state())

    def commit(actions: Any, label: str, target: int) -> None:
        import copy

        def play(live: Any) -> None:
            for action in actions:
                live.step(action, label)
        saved = env.em.get_state()
        metadata = {key: copy.copy(value) if isinstance(value, (dict, list)) else value for key, value in session.__dict__.items()}
        split_count = len(splits)
        try:
            play_hops(session, splits, [AssemblyHop(label, play, session.state.room_id, target, label, label)])
        except Exception:
            env.em.set_state(saved)
            report["rollout_restores"] += 1
            session.__dict__.clear()
            session.__dict__.update(metadata)
            del splits[split_count:]
            raise
        master.extend(actions)

    try:
        if resume and report_path.is_file() and tape_path.is_file():
            old = json.loads(report_path.read_text())
            report["rollout_restores"] = old.get("rollout_restores", 0)
            report["stopped"] = old.get("stopped")
            if old.get("rom_sha256", report["rom_sha256"]) != report["rom_sha256"]:
                raise ValueError("master tape belongs to a different ROM")
            actions = np.load(tape_path)["actions"]
            if hashlib.sha256(actions.tobytes()).hexdigest() != old["tape_sha256"]:
                raise ValueError("master tape digest mismatch")
            for previous in old["rows"]:
                current = next((row for row in rows if row["row"] == previous["row"]), None)
                identity = ("segment", "hop_index", "room", "to_room", "tape", "source_start", "source_end", "controller_id", "continuation", "goal_phase")
                if current is None or any(current.get(key) != previous.get(key) for key in identity if key in previous):
                    raise ValueError(f"board changed before accepted row {previous['row']}; use a new artifact stem")
                if current.get("tape"):
                    _, source = _chain_body(current, cache)
                    if previous.get("source_sha256", source["source_sha256"]) != source["source_sha256"]:
                        raise ValueError(f"source tape changed at accepted row {previous['row']}")
                    previous["source_sha256"] = source["source_sha256"]
                previous.update({key: current.get(key) for key in identity if key not in previous})
                count = int(previous["frames"])
                offset = len(master)
                commit(actions[offset : offset + count], f"resume:{previous['row']}", int(previous.get("to_room") or previous["room"], 16))
                ledger.append(previous)
            tail_count = int(old.get("credits_tail_frames", 0))
            if tail_count < 0 or len(master) + tail_count != len(actions):
                raise ValueError("master tape frame counts do not match its ledger")
            if tail_count:
                if not ledger or ledger[-1].get("goal_phase") != "ending_or_credits":
                    raise ValueError("credits tail requires an ending leave")
                commit(actions[len(master):], "resume:credits", session.state.room_id)
                if not credits_complete(env):
                    raise RuntimeError("saved credits tail no longer reaches the final screen")
                report["credits_tail_frames"] = tail_count
            actual = parse_env_state(env, frame=session.frame) if old.get("state_parse") == "full_bank7e" else session.state
            if not _same_chain_state(actual.to_dict(), old["final_state"]):
                raise RuntimeError("power-on prefix no longer reproduces its saved game state")
        done = {r["row"] for r in ledger}
        for row in rows:
            if row["row"] in done or row["row"] > through_row:
                continue
            try:
                if row["room"] != "0x0000" and session.state.room_id != int(row["room"], 16):
                    raise RuntimeError(f"coverage gap: live room 0x{session.state.room_id:04X}, next row starts {row['room']}")
                if row.get("controller_id"):
                    body = np.zeros((1, 12), dtype=np.int8)
                    final = {"x": 0, "y": 0, "items": session.state.collected_items}
                else:
                    body, final = _chain_body(row, cache)
                override = (overrides or {}).get(row["row"])
                if override:
                    body = np.load(override)["actions"]
                chosen, method, restores = _search_chain_body(env, body, row, final, max_delay=max_delay)
                report["rollout_restores"] += restores + 1
                commit(chosen, f"chain:{row['row']}", int(row.get("to_room") or row["room"], 16))
                ledger.append({k: row.get(k) for k in (
                    "row", "segment", "hop_index", "room", "to_room", "name", "tape",
                    "source_start", "source_end", "controller_id", "continuation", "goal_phase",
                )} | {
                    "source_sha256": final.get("source_sha256"), "frames": len(chosen),
                    "method": method, "energy": session.state.health,
                    "max_energy": session.state.max_health,
                    "leave_x": session.state.samus_x, "leave_y": session.state.samus_y,
                })
                report["stopped"] = None
                persist()
                print(f"row {row['row']:3} {row['name']}: {len(chosen)}f {method}, E={session.state.health}/{session.state.max_health}", flush=True)
            except (RuntimeError, ValueError, KeyError, IndexError, OSError) as exc:
                report["rollout_restores"] += getattr(exc, "rollout_restores", 0)
                report["stopped"] = {"row": row["row"], "room": row["room"], "name": row["name"], "error": str(exc)}
                print(str(exc), flush=True)
                break
        if (
            ledger
            and ledger[-1]["row"] == rows[-1]["row"]
            and ledger[-1].get("goal_phase") == "ending_or_credits"
        ):
            if not credits_complete(env):
                start = env.em.get_state()
                try:
                    tail = play_credits_tail(env)
                except RuntimeError as exc:
                    ending = ledger[-1]
                    report["stopped"] = {**{k: ending[k] for k in ("row", "room", "name")}, "error": str(exc)}
                else:
                    report["stopped"] = None
                finally:
                    env.em.set_state(start)
                    report["rollout_restores"] += 1
                if report["stopped"] is None:
                    commit(tail, "chain:credits", session.state.room_id)
                    report["credits_tail_frames"] = len(tail)
            report["credits_complete"] = credits_complete(env)
        persist()
    finally:
        restore_writes()
        env.close()
    proof = replay_tape_chain(stem=stem)
    report["proof"] = proof
    report["credits_complete"] = proof["success"] and proof["credits_complete"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def replay_tape_chain(*, stem: Path) -> dict[str, Any]:
    """Independent buttons-only power-on replay with load/write APIs blocked."""
    import hashlib
    import json
    import numpy as np
    from super_metroid.dev.common import make_dev_env
    from super_metroid.ram import GameplayPhase, credits_complete, parse_env_state
    from super_metroid.routes.tips import play_hops

    report = json.loads(stem.with_suffix(".json").read_text())
    from super_metroid.paths import SHARED_ROM, VANILLA_ROM_SHA1
    rom_bytes = SHARED_ROM.read_bytes()
    rom_sha1, rom_digest = hashlib.sha1(rom_bytes).hexdigest(), hashlib.sha256(rom_bytes).hexdigest()
    rom_vanilla = rom_sha1 == VANILLA_ROM_SHA1
    if not rom_vanilla:
        raise ValueError(f"Clean replay requires vanilla ROM ({VANILLA_ROM_SHA1}), got {rom_sha1}")
    if report.get("rom_sha256", rom_digest) != rom_digest or report.get("rom_sha1", rom_sha1) != rom_sha1:
        raise ValueError("master tape belongs to a different ROM")
    actions = np.load(stem.with_suffix(".npz"))["actions"]
    if actions.ndim != 2 or actions.shape[1] != 12 or not np.isin(actions, [0, 1]).all():
        raise ValueError("master tape must contain binary SNES-12 actions")
    if hashlib.sha256(actions.tobytes()).hexdigest() != report["tape_sha256"]:
        raise ValueError("master tape digest mismatch")
    env = make_dev_env()
    env.reset()
    session = _chain_session(env)
    core, load_attempts, write_attempts = env.em, [0], [0]

    class ReplayCore:
        def __getattr__(self, name: str) -> Any:
            if name in _LOAD_NAMES:
                def blocked(*args: Any, **kwargs: Any) -> NoReturn:
                    load_attempts[0] += 1
                    raise RuntimeError("Clean replay forbids state loads")
                return blocked
            return getattr(core, name)

    env.em = ReplayCore()
    restore_writes = _guard_writes(env, write_attempts)
    restore_loads = _guard_loads(session, load_attempts)
    deaths = offset = 0
    splits, detour_rows = [], []
    leaves_match = crossings_match = True
    transitions: list[dict[str, int | None]] = []

    def step(action: Any, label: str, row_number: int | None) -> None:
        nonlocal deaths
        previous_room = session.state.room_id
        state = session.step(action, label)
        deaths += int(state.phase is GameplayPhase.DEATH_OR_GAME_OVER)
        if previous_room != state.room_id and previous_room not in {0, 0x5555}:
            transitions.append({
                "row": row_number, "frame": session.frame,
                "source_room_id": previous_room, "target_room_id": state.room_id,
            })

    try:
        for row in report["rows"]:
            count, source = int(row["frames"]), int(row["room"], 16)
            target, row_number = int(row.get("to_room") or row["room"], 16), row["row"]
            leaves_match &= source == 0 or session.state.room_id == source
            transition_start, label = len(transitions), f"proof:{row_number}"

            def play(_live: Any) -> None:
                for action in actions[offset : offset + count]:
                    step(action, label, row_number)

            play_hops(session, splits, [AssemblyHop(str(row["row"]), play, session.state.room_id, target, row["name"], str(row["row"]))])
            observed = transitions[transition_start:]
            expected = int(source != 0 and source != target)
            if expected:
                crossings_match &= any(t["source_room_id"] == source and t["target_room_id"] == target for t in observed)
            if len(observed) > expected:
                detour_rows.append(row_number)
            state = session.state
            settled = state.game_state == 8 and not state.door_transition
            if row.get("goal_phase") == "ending_or_credits":
                settled = state.phase.value == "ending_or_credits"
            leaves_match &= settled and state.health > 0
            leaves_match &= (state.health, state.samus_x, state.samus_y) == (row["energy"], row["leave_x"], row["leave_y"])
            offset += count
        tail_count = int(report.get("credits_tail_frames", 0))
        rows = report["rows"]
        entered_ending = bool(rows and rows[-1].get("goal_phase") == "ending_or_credits")
        if tail_count < 0 or offset + tail_count != len(actions) or (tail_count and not entered_ending):
            raise ValueError("master tape credits tail does not match its ledger")
        for action in actions[offset:]:
            step(action, "proof:credits", None)
        complete = entered_ending and credits_complete(env)
        credits_match = complete == bool(report.get("credits_complete", False))
        final = parse_env_state(env, frame=session.frame) if report.get("state_parse") == "full_bank7e" else session.state
        matches = _same_chain_state(final.to_dict(), report["final_state"])
        core_matches = hashlib.sha256(env.em.get_state()).hexdigest() == report["endpoint_sha256"]
        prior_transitions = (report.get("proof") or {}).get("transitions")
        timeline_matches = prior_transitions is None or prior_transitions == transitions
        return {
            "success": (
                matches and leaves_match and crossings_match and timeline_matches
                and not deaths and credits_match and (not tail_count or complete)
                and rom_vanilla and load_attempts[0] == 0 and write_attempts[0] == 0
            ),
            "start_state": "power_on/retro.State.NONE",
            "rom_sha1": rom_sha1, "rom_vanilla": rom_vanilla,
            "state_loads": load_attempts[0], "ram_writes": write_attempts[0],
            "deaths": deaths, "total_frames": session.frame,
            "last_row": report["rows"][-1]["row"] if report["rows"] else None,
            "rooms_cleared": report["rooms_cleared"], "endpoint_matches": matches,
            "all_leaves_match": bool(leaves_match), "credits_complete": complete,
            "all_planned_crossings_match": crossings_match,
            "transition_timeline_matches": timeline_matches,
            "observed_transition_count": len(transitions), "detour_rows": detour_rows,
            "transitions": transitions,
            "credits_tail_frames": tail_count,
            "comparison": "vanilla ROM SHA-1, planned crossings, settled leaves, final game-state fields, and terminal credits when present",
            "core_snapshot_matches": core_matches, "final_state": final.to_dict(),
        }
    finally:
        restore_loads()
        restore_writes()
        env.em = core
        env.close()
