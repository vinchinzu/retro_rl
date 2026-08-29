"""Production adapter from ALTTP opening skills to SolverSession."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from alttp.ram import LINKS_HOUSE_ROOM, AlttpSnapshot
from alttp.startup import snapshot_env
from alttp_rando.house_to_uncle import play_house_to_uncle
from alttp_rando.logic_graph import N_LINKS_HOUSE, N_UNCLE
from alttp_rando.solver_bindings import (
    ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST,
    HOUSE_TO_UNCLE_SPEC,
    house_to_uncle_binding,
    load_house_to_uncle_evidence,
)
from retro_harness.adventure import (
    BindingCatalog,
    ExecutionReadiness,
    GraphEdge,
    GraphNode,
    PlanRequest,
    RouteGraph,
    plan,
)
from retro_harness.benchmark_claims import PolicyIdentity
from retro_harness.identity import sha256_bytes
from retro_harness.solver import (
    OneShotSkillPolicy,
    SkillInstance,
    SkillSignal,
    SkillStep,
    SolverObservation,
    SolverSession,
)

EDGE_HOUSE_TO_UNCLE = "house_to_uncle"
REAL_EDGE_IDS = (EDGE_HOUSE_TO_UNCLE,)
OBSERVATION_SCHEMA_DIGEST = ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST

HouseRunner = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class RouteSkillCommand:
    """A synchronous real-game controller invocation understood by the adapter."""

    edge_id: str
    target_node: str
    runner: HouseRunner

    def to_action_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "target_node": self.target_node,
            "runner": f"{self.runner.__module__}.{self.runner.__qualname__}",
        }


class RouteCommandPolicy(OneShotSkillPolicy):
    """Honest one-shot adapter for the existing house→uncle opening skill.

    Macro runners execute the full edge inside ``apply_route_command``;
    multi-frame :class:`~retro_harness.skill_policies.ScriptedSkillPolicy`
    consumers are the shared shape for tick-level RUNNING skills.
    """

    def __init__(self, command: RouteSkillCommand) -> None:
        self.command = command
        super().__init__(SkillStep(SkillSignal.SUCCESS, action=command))


@dataclass(frozen=True, slots=True)
class SolverAdapterBundle:
    graph: RouteGraph
    bindings: BindingCatalog
    skills: tuple[SkillInstance, ...]


class HouseSolverSession:
    """Frame-counting play env for SolverSession observe/apply hooks.

    ``em`` is the raw emulator (not AuditedEnv's load counter). Hole-drop
    candidate restores in ``castle_to_sword`` are search, not a published
    start skip; RAM writes still pass through the audited env.
    """

    def __init__(self, env: Any) -> None:
        self.audited = env
        self.frames = 0
        self.env = _PlayEnv(env, self)


class _PlayEnv:
    def __init__(self, inner: Any, session: HouseSolverSession) -> None:
        self._inner = inner
        self._session = session

    def step(self, action: Any) -> Any:
        result = self._inner.step(action)
        self._session.frames += 1
        return result

    @property
    def em(self) -> Any:
        inner = self._inner
        raw = getattr(inner, "env", inner)
        return raw.em

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def node_id_from_snapshot(snap: AlttpSnapshot) -> str:
    """Map an ALTTP RAM snapshot onto the bound house→uncle nodes."""
    if snap.has_fighter_sword and snap.in_secret_passage:
        return N_UNCLE
    if snap.indoors and snap.room_base_id == (LINKS_HOUSE_ROOM & 0xFF):
        return N_LINKS_HOUSE
    if snap.has_fighter_sword:
        return N_UNCLE
    return N_LINKS_HOUSE


def capabilities_from_snapshot(snap: AlttpSnapshot) -> frozenset[str]:
    caps: set[str] = set()
    if snap.has_fighter_sword:
        caps.add("sword")
    if snap.has_lamp:
        caps.add("lamp")
    return frozenset(caps)


def observation_from_session(session: HouseSolverSession) -> SolverObservation:
    snap = snapshot_env(session.env)
    return SolverObservation(
        frame=session.frames,
        node_id=node_id_from_snapshot(snap),
        schema_digest=OBSERVATION_SCHEMA_DIGEST,
        capabilities=capabilities_from_snapshot(snap),
        values={
            "game_mode": int(snap.game_mode),
            "room_base_id": int(snap.room_base_id),
            "indoors": bool(snap.indoors),
            "has_control": bool(snap.has_control),
            "has_fighter_sword": bool(snap.has_fighter_sword),
            "has_lamp": bool(snap.has_lamp),
        },
    )


def apply_route_command(session: HouseSolverSession, command: Any) -> int:
    if not isinstance(command, RouteSkillCommand):
        raise TypeError("solver adapter accepts RouteSkillCommand actions only")
    start_frame = session.frames
    result = command.runner(session.env)
    elapsed = session.frames - start_frame
    if elapsed < 1:
        raise RuntimeError(f"{command.edge_id} executed no emulator frames")
    snap = snapshot_env(session.env)
    if node_id_from_snapshot(snap) != command.target_node:
        blocker = getattr(result, "blocker", "")
        raise RuntimeError(
            f"{command.edge_id} ended at {node_id_from_snapshot(snap)!r}, "
            f"expected {command.target_node!r}"
            + (f" blocker={blocker}" if blocker else "")
        )
    if command.target_node == N_UNCLE and not snap.has_fighter_sword:
        raise RuntimeError(f"{command.edge_id} ended without fighter sword")
    return elapsed


def build_solver_adapter_bundle() -> SolverAdapterBundle:
    """Build the production one-edge graph with no experiment fault edge."""
    binding = house_to_uncle_binding()
    evidence = load_house_to_uncle_evidence()
    instance = SkillInstance(
        spec=HOUSE_TO_UNCLE_SPEC,
        binding=binding,
        policy=RouteCommandPolicy(
            RouteSkillCommand(
                EDGE_HOUSE_TO_UNCLE,
                N_UNCLE,
                play_house_to_uncle,
            )
        ),
        policy_identity=PolicyIdentity.from_policy(play_house_to_uncle),
        config={"source_report_sha256": evidence.artifact_digest or ""},
    )
    graph = RouteGraph(
        (
            GraphNode(N_LINKS_HOUSE, "Link's House", "light_world"),
            GraphNode(
                N_UNCLE,
                "Uncle / Fighter Sword",
                "hyrule_castle",
                tags=frozenset({"item"}),
            ),
        ),
        (
            GraphEdge(
                N_LINKS_HOUSE,
                N_UNCLE,
                edge_id=EDGE_HOUSE_TO_UNCLE,
                acquires=frozenset({"sword"}),
                verification="natural_entry",
                provenance=(
                    "alttp opening skills from FirstPlay; binding in "
                    "solver_bindings"
                ),
            ),
        ),
    )
    return SolverAdapterBundle(
        graph=graph,
        bindings=BindingCatalog((binding,)),
        skills=(instance,),
    )


def build_solver_session(
    session: HouseSolverSession,
    bundle: SolverAdapterBundle | None = None,
    *,
    minimum_readiness: ExecutionReadiness = ExecutionReadiness.NATURAL_ENTRY,
    max_replans: int = 0,
) -> SolverSession:
    active = bundle or build_solver_adapter_bundle()

    def plan_fn(observation: SolverObservation, excluded: frozenset[str]):
        edges = tuple(
            edge for edge in active.graph.edges if edge.edge_id not in excluded
        )
        return plan(PlanRequest(edges, observation.node_id, N_UNCLE))

    return SolverSession(
        observe=lambda: observation_from_session(session),
        apply_action=lambda command: apply_route_command(session, command),
        plan_fn=plan_fn,
        bindings=active.bindings,
        skills=active.skills,
        minimum_readiness=minimum_readiness,
        max_replans=max_replans,
    )


def composite_policy_identity(bundle: SolverAdapterBundle) -> PolicyIdentity:
    payload = "\n".join(
        binding.identity_digest for binding in bundle.bindings.bindings
    )
    return PolicyIdentity(
        "alttp_rando.solver_adapter",
        digest=sha256_bytes(payload.encode("utf-8")),
        version="1",
        source="alttp_rando.solver_adapter:build_solver_session",
        metadata={"binding_count": len(bundle.bindings.bindings)},
    )


__all__ = [
    "EDGE_HOUSE_TO_UNCLE",
    "OBSERVATION_SCHEMA_DIGEST",
    "REAL_EDGE_IDS",
    "HouseSolverSession",
    "RouteCommandPolicy",
    "RouteSkillCommand",
    "SolverAdapterBundle",
    "apply_route_command",
    "build_solver_adapter_bundle",
    "build_solver_session",
    "capabilities_from_snapshot",
    "composite_policy_identity",
    "node_id_from_snapshot",
    "observation_from_session",
]
