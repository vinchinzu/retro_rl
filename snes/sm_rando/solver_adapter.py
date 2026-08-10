"""Production adapter from Super Metroid route controllers to SolverSession."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from retro_harness.adventure import (
    BindingCatalog,
    EdgeEvidence,
    ExecutionReadiness,
    GraphEdge,
    GraphNode,
    PlanRequest,
    PromotionPolicy,
    RouteGraph,
    SkillBinding,
    plan,
)
from retro_harness.benchmark_claims import PolicyIdentity
from retro_harness.contracts import ObservationContract, ObservationField
from retro_harness.identity import canonical_json, sha256_bytes, sha256_file
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillInstance,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverObservation,
    SolverSession,
)
from sm_rando.paths import RECORDINGS_DIR, REPO_ROOT
from super_metroid.routes.kpdr.early_spine import (
    play_climb_to_pit,
    play_landing_to_parlor,
    play_parlor_to_climb,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CLIMB,
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.routes.runtime import RouteSession

VERTICAL_SLICE_EVIDENCE = RECORDINGS_DIR / "vertical_slice.evidence.json"

SOLVER_OBSERVATION_CONTRACT = ObservationContract(
    fields=(
        ObservationField("room_id", "uint16", semantic="current room node"),
        ObservationField("game_state", "uint16", semantic="engine game state"),
        ObservationField(
            "door_transition", "uint16", semantic="door transition phase"
        ),
        ObservationField("morph_ball", "bool", semantic="progression capability"),
        ObservationField("energy", "uint16", semantic="current energy resource"),
        ObservationField("missiles", "uint16", semantic="current missile resource"),
    ),
    preprocessing={"adapter": "sm_rando.solver_adapter/v1"},
)
OBSERVATION_SCHEMA_DIGEST = SOLVER_OBSERVATION_CONTRACT.identity_digest

EDGE_RECOVERY = "landing_to_parlor_recovery"
EDGE_PARLOR = "parlor_to_climb"
EDGE_CLIMB = "climb_to_pit"
REAL_EDGE_IDS = (EDGE_RECOVERY, EDGE_PARLOR, EDGE_CLIMB)

RouteRunner = Callable[[RouteSession], None]


@dataclass(frozen=True, slots=True)
class RouteSkillCommand:
    """A synchronous real-game controller invocation understood by the adapter."""

    edge_id: str
    target_room: int
    runner: RouteRunner

    def to_action_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "target_room": self.target_room,
            "runner": f"{self.runner.__module__}.{self.runner.__qualname__}",
        }


class RouteCommandPolicy:
    """Honest one-shot adapter for an existing synchronous route controller."""

    def __init__(self, command: RouteSkillCommand) -> None:
        self.command = command

    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None:
        del observation, config

    def step(self, observation: SolverObservation) -> SkillStep:
        del observation
        return SkillStep(SkillSignal.SUCCESS, action=self.command)


@dataclass(frozen=True, slots=True)
class SolverAdapterBundle:
    graph: RouteGraph
    bindings: BindingCatalog
    skills: tuple[SkillInstance, ...]


def _boundary_digest(report_sha256: str, frame: int, room_id: int) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "frame": frame,
                "report_sha256": report_sha256,
                "room_id": room_id,
                "schema_digest": OBSERVATION_SCHEMA_DIGEST,
            }
        ).encode("utf-8")
    )


def _validated_boundaries(
    evidence_path: Path = VERTICAL_SLICE_EVIDENCE,
) -> tuple[str, dict[int, tuple[int, str]]]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported vertical-slice evidence schema")
    source_value = evidence.get("source_report")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("vertical-slice evidence lacks source_report")
    source = (REPO_ROOT / source_value).resolve()
    try:
        source.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("source_report escapes repository root") from exc
    report_sha256 = sha256_file(source)
    if evidence.get("source_report_sha256") != report_sha256:
        raise ValueError("vertical-slice source report digest mismatch")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not report.get("success") or report.get("outcome") != "morph_ball_acquired":
        raise ValueError("vertical-slice source report did not clear Morph Ball")
    if report.get("state_loads") != 0 or report.get("progression_writes") != 0:
        raise ValueError("vertical-slice source report is not clean continuous play")

    transitions = {
        transition.get("edge_id"): transition
        for transition in report.get("transitions", ())
    }
    expected = (
        ("ceres_to_landing", ROOM_LANDING_SITE),
        ("landing_to_parlor", ROOM_PARLOR),
        ("parlor_to_climb", ROOM_CLIMB),
        ("climb_to_pit", ROOM_PIT),
    )
    boundaries: dict[int, tuple[int, str]] = {}
    last_frame = -1
    for edge_id, room_id in expected:
        transition = transitions.get(edge_id)
        if transition is None or int(transition.get("target_room_id", -1)) != room_id:
            raise ValueError(f"source report lacks transition {edge_id}")
        frame = int(transition["frame"])
        if frame <= last_frame:
            raise ValueError("source report transition order is invalid")
        boundaries[room_id] = (
            frame,
            _boundary_digest(report_sha256, frame, room_id),
        )
        last_frame = frame
    return report_sha256, boundaries


def route_skill_spec(edge_id: str, source_room: int, target_room: int) -> SkillSpec:
    requirement = ObservationRequirement(
        schema_digest=OBSERVATION_SCHEMA_DIGEST,
        allowed_nodes=(source_room,),
        # Successor controllers own settle frames after the first target-room
        # transition frame, so room identity is the executable handoff contract.
        required_values={"room_id": source_room},
    )
    return SkillSpec(
        skill_id=f"sm.vertical.{edge_id}",
        dispatch_key=f"sm_rando.solver_adapter:{edge_id}",
        observation_requirement=requirement,
        expected_delta=ProgressionDelta(target_node=target_room),
        timeout_frames=4_000,
    )


def scaffold_binding(spec: SkillSpec, edge_id: str) -> SkillBinding:
    return SkillBinding(
        edge_id=edge_id,
        skill_id=spec.skill_id,
        dispatch_key=spec.dispatch_key,
        entry_requirement_digest=spec.observation_requirement.identity_digest,
        progression_delta_digest=spec.expected_delta.identity_digest,
    )


def build_solver_adapter_bundle(
    evidence_path: Path = VERTICAL_SLICE_EVIDENCE,
) -> SolverAdapterBundle:
    """Build the production three-edge graph with no experiment fault edge."""
    source_sha256, boundaries = _validated_boundaries(evidence_path)
    rows: tuple[tuple[str, int, int, RouteRunner, str], ...] = (
        (
            EDGE_RECOVERY,
            ROOM_LANDING_SITE,
            ROOM_PARLOR,
            play_landing_to_parlor,
            "ceres_to_landing",
        ),
        (
            EDGE_PARLOR,
            ROOM_PARLOR,
            ROOM_CLIMB,
            play_parlor_to_climb,
            EDGE_RECOVERY,
        ),
        (
            EDGE_CLIMB,
            ROOM_CLIMB,
            ROOM_PIT,
            play_climb_to_pit,
            EDGE_PARLOR,
        ),
    )
    instances: list[SkillInstance] = []
    promoted: list[SkillBinding] = []
    for edge_id, source_room, target_room, runner, predecessor in rows:
        spec = route_skill_spec(edge_id, source_room, target_room)
        base = scaffold_binding(spec, edge_id)
        binding = PromotionPolicy().promote(
            base,
            EdgeEvidence(
                edge_id=edge_id,
                binding_digest=base.identity_digest,
                readiness=ExecutionReadiness.NATURAL_ENTRY,
                predecessor_edge_id=predecessor,
                predecessor_exit_observation_digest=boundaries[source_room][1],
                target_entry_observation_digest=boundaries[source_room][1],
                target_exit_observation_digest=boundaries[target_room][1],
                attempts=1,
                successes=1,
                artifact_digest=source_sha256,
            ),
        )
        promoted.append(binding)
        instances.append(
            SkillInstance(
                spec=spec,
                binding=binding,
                policy=RouteCommandPolicy(
                    RouteSkillCommand(edge_id, target_room, runner)
                ),
                policy_identity=PolicyIdentity.from_policy(runner),
                config={"source_report_sha256": source_sha256},
            )
        )

    graph = RouteGraph(
        (
            GraphNode(ROOM_LANDING_SITE, "Landing Site", "Crateria"),
            GraphNode(ROOM_PARLOR, "Parlor", "Crateria"),
            GraphNode(ROOM_CLIMB, "Climb", "Crateria"),
            GraphNode(ROOM_PIT, "Pit Room", "Crateria"),
        ),
        (
            GraphEdge(
                ROOM_LANDING_SITE,
                ROOM_PARLOR,
                edge_id=EDGE_RECOVERY,
            ),
            GraphEdge(ROOM_PARLOR, ROOM_CLIMB, edge_id=EDGE_PARLOR),
            GraphEdge(ROOM_CLIMB, ROOM_PIT, edge_id=EDGE_CLIMB),
        ),
    )
    return SolverAdapterBundle(
        graph=graph,
        bindings=BindingCatalog(promoted),
        skills=tuple(instances),
    )


def observation_from_route_session(session: RouteSession) -> SolverObservation:
    state = session.state
    capabilities = frozenset({"morph_ball"}) if state.morph_ball else frozenset()
    return SolverObservation(
        frame=session.frame,
        node_id=state.room_id,
        schema_digest=OBSERVATION_SCHEMA_DIGEST,
        capabilities=capabilities,
        resources={"energy": state.health, "missiles": state.missiles},
        values={
            "game_state": state.game_state,
            "room_id": state.room_id,
            "door_transition": state.door_transition,
        },
    )


def apply_route_command(session: RouteSession, command: Any) -> int:
    if not isinstance(command, RouteSkillCommand):
        raise TypeError("solver adapter accepts RouteSkillCommand actions only")
    start_frame = session.frame
    command.runner(session)
    if session.state.room_id != command.target_room:
        raise RuntimeError(
            f"{command.edge_id} ended in 0x{session.state.room_id:04X}, "
            f"expected 0x{command.target_room:04X}"
        )
    elapsed = session.frame - start_frame
    if elapsed < 1:
        raise RuntimeError(f"{command.edge_id} executed no emulator frames")
    return elapsed


def build_solver_session(
    route_session: RouteSession,
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
        return plan(PlanRequest(edges, observation.node_id, ROOM_PIT))

    return SolverSession(
        observe=lambda: observation_from_route_session(route_session),
        apply_action=lambda command: apply_route_command(route_session, command),
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
        "sm_rando.solver_adapter",
        digest=sha256_bytes(payload.encode("utf-8")),
        version="1",
        source="sm_rando.solver_adapter:build_solver_session",
        metadata={"binding_count": len(bundle.bindings.bindings)},
    )


__all__ = [
    "EDGE_CLIMB",
    "EDGE_PARLOR",
    "EDGE_RECOVERY",
    "OBSERVATION_SCHEMA_DIGEST",
    "REAL_EDGE_IDS",
    "RouteCommandPolicy",
    "RouteSkillCommand",
    "SOLVER_OBSERVATION_CONTRACT",
    "SolverAdapterBundle",
    "VERTICAL_SLICE_EVIDENCE",
    "apply_route_command",
    "build_solver_adapter_bundle",
    "build_solver_session",
    "composite_policy_identity",
    "observation_from_route_session",
    "route_skill_spec",
    "scaffold_binding",
]
