"""Audited three-edge SM solver slice with an explicit recovery replan."""

from __future__ import annotations

import hashlib
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
from retro_harness.benchmark import (
    AuditCapabilities,
    AttemptAudit,
    EvaluationContract,
    InterventionClass,
    PolicyIdentity,
    RuntimeObservationClass,
    StartIdentity,
    validate_claim,
)
from retro_harness.contracts import contract_digest
from retro_harness.play_spine import RunManifest, utc_now_iso
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillInstance,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverObservation,
    SolverSession,
    SolverSessionResult,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    counterexamples_from_solver_result,
    trajectory_from_solver_result,
)
from sm_rando.paths import GAME, GAME_DIR, RECORDINGS_DIR, REPO_ROOT, SHARED_SM_ROM
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.progression import MORPH_GRAPH
from super_metroid.routes.kpdr.early_spine import (
    MORPH_SPINE,
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
from super_metroid.routes.runtime import RouteSession, Split
from super_metroid.routes.tips import play_hops

VERTICAL_SLICE_EVIDENCE = RECORDINGS_DIR / "vertical_slice.evidence.json"
VERTICAL_SLICE_MANIFEST = RECORDINGS_DIR / "vertical_slice.run.json"
VERTICAL_SLICE_TRAJECTORY = RECORDINGS_DIR / "vertical_slice.trajectory.json"
VERTICAL_SLICE_COUNTEREXAMPLES = RECORDINGS_DIR / "vertical_slice_counterexamples"
OBSERVATION_SCHEMA_DIGEST = hashlib.sha256(
    b"sm-rando-vertical-slice-observation-v1"
).hexdigest()

EDGE_RECOVERY = "landing_to_parlor_recovery"
EDGE_PARLOR = "parlor_to_climb"
EDGE_CLIMB = "climb_to_pit"
EDGE_INJECTED = "landing_to_parlor_injected_failure"
REAL_EDGE_IDS = (EDGE_RECOVERY, EDGE_PARLOR, EDGE_CLIMB)
ACTION_SCHEMA_DIGEST = contract_digest(
    "sm-rando-vertical-action-v1",
    {"kind": "RouteSkillCommand", "edge_ids": [EDGE_INJECTED, *REAL_EDGE_IDS]},
)
REWARD_SCHEMA_DIGEST = contract_digest(
    "sm-rando-vertical-reward-v1",
    {"components": [{"name": "room_progress", "weight": 1.0}]},
)
CONTRACT_BUNDLE_DIGEST = contract_digest(
    "sm-rando-vertical-bundle-v1",
    {
        "observation": OBSERVATION_SCHEMA_DIGEST,
        "action": ACTION_SCHEMA_DIGEST,
        "reward": REWARD_SCHEMA_DIGEST,
        "environment": GAME,
        "start": "NONE",
    },
)

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
    """Expose an existing RouteSession controller as a SolverSession policy."""

    def __init__(self, command: RouteSkillCommand) -> None:
        self.command = command
        self._pending = True

    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None:
        del observation, config
        self._pending = True

    def step(self, observation: SolverObservation) -> SkillStep:
        del observation
        if self._pending:
            self._pending = False
            return SkillStep(SkillSignal.SUCCESS, action=self.command)
        return SkillStep(SkillSignal.TERMINAL_FAILURE, reason="command already ran")


class InjectedFailurePolicy:
    """Deterministic fault used to prove the runtime replans before real play."""

    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None:
        del observation, config

    def step(self, observation: SolverObservation) -> SkillStep:
        del observation
        return SkillStep(
            SkillSignal.RETRYABLE_FAILURE,
            reason="injected landing-door desync",
            recovery_hint="exclude_primary_and_replan",
        )


@dataclass(frozen=True, slots=True)
class VerticalSliceBundle:
    graph: RouteGraph
    bindings: BindingCatalog
    skills: tuple[SkillInstance, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary_digest(report_sha256: str, frame: int, room_id: int) -> str:
    payload = json.dumps(
        {
            "frame": frame,
            "report_sha256": report_sha256,
            "room_id": room_id,
            "schema_digest": OBSERVATION_SCHEMA_DIGEST,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    report_sha256 = _sha256(source)
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


def _spec(edge_id: str, source_room: int, target_room: int) -> SkillSpec:
    requirement = ObservationRequirement(
        schema_digest=OBSERVATION_SCHEMA_DIGEST,
        allowed_nodes=(source_room,),
        # The vanilla room tapes intentionally hand off on the first target-
        # room frame (game_state 11 / door_transition 1); the successor owns
        # the settle frames. Requiring ordinary game_state 8 here would reject
        # the real continuous boundary that produced the retained evidence.
        required_values={"room_id": source_room},
    )
    delta = ProgressionDelta(target_node=target_room)
    return SkillSpec(
        skill_id=f"sm.vertical.{edge_id}",
        dispatch_key=f"sm_rando.vertical_slice:{edge_id}",
        observation_requirement=requirement,
        expected_delta=delta,
        timeout_frames=4_000,
    )


def _base_binding(spec: SkillSpec, edge_id: str) -> SkillBinding:
    return SkillBinding(
        edge_id=edge_id,
        skill_id=spec.skill_id,
        dispatch_key=spec.dispatch_key,
        entry_contract_digest=spec.observation_requirement.identity_digest,
        exit_contract_digest=spec.expected_delta.identity_digest,
    )


def build_vertical_slice_bundle(
    evidence_path: Path = VERTICAL_SLICE_EVIDENCE,
) -> VerticalSliceBundle:
    """Build the graph and three evidence-promoted real controller bindings."""
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
    specs: dict[str, SkillSpec] = {}
    for edge_id, source_room, target_room, runner, predecessor in rows:
        spec = _spec(edge_id, source_room, target_room)
        specs[edge_id] = spec
        base = _base_binding(spec, edge_id)
        evidence = EdgeEvidence(
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
        )
        binding = PromotionPolicy().promote(base, evidence)
        promoted.append(binding)
        command = RouteSkillCommand(edge_id, target_room, runner)
        instances.append(
            SkillInstance(
                spec=spec,
                binding=binding,
                policy=RouteCommandPolicy(command),
                policy_identity=PolicyIdentity.from_policy(runner),
                config={"source_report_sha256": source_sha256},
            )
        )

    injected_spec = _spec(EDGE_INJECTED, ROOM_LANDING_SITE, ROOM_PARLOR)
    injected_binding = SkillBinding(
        edge_id=EDGE_INJECTED,
        skill_id=injected_spec.skill_id,
        dispatch_key=injected_spec.dispatch_key,
        entry_contract_digest=(
            injected_spec.observation_requirement.identity_digest
        ),
        exit_contract_digest=injected_spec.expected_delta.identity_digest,
        readiness=ExecutionReadiness.ISOLATED,
        evidence_digest=hashlib.sha256(b"fault-injection-v1").hexdigest(),
    )
    instances.append(
        SkillInstance(
            spec=injected_spec,
            binding=injected_binding,
            policy=InjectedFailurePolicy(),
            policy_identity=PolicyIdentity(
                "sm.vertical.injected_failure",
                version="1",
                source="sm_rando.vertical_slice:InjectedFailurePolicy",
            ),
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
                edge_id=EDGE_INJECTED,
                cost=0.5,
            ),
            GraphEdge(
                ROOM_LANDING_SITE,
                ROOM_PARLOR,
                edge_id=EDGE_RECOVERY,
                cost=1.0,
            ),
            GraphEdge(ROOM_PARLOR, ROOM_CLIMB, edge_id=EDGE_PARLOR),
            GraphEdge(ROOM_CLIMB, ROOM_PIT, edge_id=EDGE_CLIMB),
        ),
    )
    return VerticalSliceBundle(
        graph=graph,
        bindings=BindingCatalog((*promoted, injected_binding)),
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
        resources={
            "energy": state.health,
            "missiles": state.missiles,
        },
        values={
            "game_state": state.game_state,
            "room_id": state.room_id,
            "door_transition": state.door_transition,
        },
    )


def apply_route_command(session: RouteSession, command: Any) -> int:
    if not isinstance(command, RouteSkillCommand):
        raise TypeError("vertical slice accepts RouteSkillCommand actions only")
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
    bundle: VerticalSliceBundle | None = None,
) -> SolverSession:
    active = bundle or build_vertical_slice_bundle()

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
        minimum_readiness=ExecutionReadiness.ISOLATED,
        max_replans=1,
    )


def _composite_policy_identity(bundle: VerticalSliceBundle) -> PolicyIdentity:
    payload = "\n".join(
        binding.identity_digest for binding in bundle.bindings.bindings
    )
    return PolicyIdentity(
        "sm_rando.vertical_slice",
        digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        version="1",
        source="sm_rando.vertical_slice:build_solver_session",
        metadata={"binding_count": len(bundle.bindings.bindings)},
    )


def _route_attempt_audit(
    route_session: RouteSession,
    contract: EvaluationContract,
) -> AttemptAudit:
    telemetry = route_session.assist.telemetry
    ram_writes = telemetry.energy.writes + sum(
        counter.writes for counter in telemetry.ammo.values()
    )
    return AttemptAudit(
        ram_writes=ram_writes,
        mid_run_loads=0,
        assists={"resource_refill": ram_writes} if ram_writes else {},
        start_identity_digest=contract.start_identity.identity_digest,
        policy_identity_digest=contract.policy_identity.identity_digest,
        runtime_observation_class=contract.runtime_observation_class,
        intervention_class=contract.intervention_class,
        capabilities=AuditCapabilities.all(
            "super_metroid.RouteSession.vertical-slice-v1"
        ),
    )


def run_real_vertical_slice(
    output_path: Path = VERTICAL_SLICE_MANIFEST,
) -> tuple[SolverSessionResult, RunManifest]:
    """Power on vanilla SM, enter Landing naturally, and run the solver slice."""
    from retro_harness.env import make_env

    bundle = build_vertical_slice_bundle()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    manifest = RunManifest(
        game=GAME,
        package="sm_rando",
        started_at=utc_now_iso(),
        seed="vanilla-substrate",
        start_state="power_on/retro.State.NONE",
        mode="bot",
    )
    try:
        env.reset()
        assist = UnlimitedAmmoAssist(enabled=False)
        route_session = RouteSession(
            env,
            writer=None,
            assist=assist,
            graph=MORPH_GRAPH,
        )
        prefix_splits: list[Split] = []
        play_hops(route_session, prefix_splits, MORPH_SPINE[:3])
        if route_session.state.room_id != ROOM_LANDING_SITE:
            raise RuntimeError("natural predecessor did not reach Landing Site")

        solver_result = build_solver_session(route_session, bundle).run()
        policy_identity = _composite_policy_identity(bundle)
        trajectory = trajectory_from_solver_result(
            solver_result,
            action_schema_digest=ACTION_SCHEMA_DIGEST,
            reward_schema_digest=REWARD_SCHEMA_DIGEST,
            contract_bundle_digest=CONTRACT_BUNDLE_DIGEST,
            policy_identity_digest=policy_identity.identity_digest,
            provenance={
                "game": GAME,
                "start": "power_on_to_natural_landing",
                "evidence_path": str(VERTICAL_SLICE_EVIDENCE.relative_to(REPO_ROOT)),
            },
            reward_fn=lambda event: {
                "room_progress": float(
                    event.observation_before.node_id
                    != event.observation_after.node_id
                )
            },
        )
        trajectory.write(VERTICAL_SLICE_TRAJECTORY)
        counterexample_library = CounterexampleLibrary(
            VERTICAL_SLICE_COUNTEREXAMPLES
        )
        counterexamples = counterexamples_from_solver_result(
            solver_result,
            action_schema_digest=ACTION_SCHEMA_DIGEST,
            reward_schema_digest=REWARD_SCHEMA_DIGEST,
            contract_bundle_digest=CONTRACT_BUNDLE_DIGEST,
            policy_identity_digest=policy_identity.identity_digest,
            provenance={
                "game": GAME,
                "parent_trajectory_digest": trajectory.identity_digest,
            },
        )
        for counterexample in counterexamples:
            counterexample_library.add(counterexample)
        contract = EvaluationContract(
            runtime_observation_class=RuntimeObservationClass.BRONZE,
            intervention_class=InterventionClass.CLEAN,
            start_identity=StartIdentity(
                "power_on_to_natural_landing",
                rom_sha256=_sha256(SHARED_SM_ROM),
            ),
            policy_identity=policy_identity,
            benchmark_id="sm_rando_vertical_slice_v1",
            objective="Natural Landing Site to Pit Room with recovery replan",
        )
        audit = _route_attempt_audit(route_session, contract)
        validate_claim(contract, audit)

        manifest.frames = route_session.frame
        manifest.outcome = solver_result.status.value
        for outcome in solver_result.outcomes:
            manifest.add_milestone(
                outcome.edge_id,
                status=outcome.status.value,
                frames=outcome.frames,
            )
        manifest.meta.update(
            {
                "schema_version": 1,
                "solver_result": solver_result.to_record(),
                "benchmark_contract": contract.to_record(),
                "attempt_audit": audit.to_record(),
                "claim_valid": True,
                "evidence_path": str(VERTICAL_SLICE_EVIDENCE.relative_to(REPO_ROOT)),
                "real_edges": list(REAL_EDGE_IDS),
                "fault_edge": EDGE_INJECTED,
                "trajectory": {
                    "path": str(VERTICAL_SLICE_TRAJECTORY.relative_to(REPO_ROOT)),
                    "identity_digest": trajectory.identity_digest,
                    "schema_digest": trajectory.to_record()["schema_digest"],
                    "steps": len(trajectory.steps),
                },
                "counterexamples": [
                    {
                        "identity_digest": item.identity_digest,
                        "terminal_reason": item.terminal_reason,
                        "steps": len(item.steps),
                    }
                    for item in counterexamples
                ],
                "prefix_splits": [
                    {
                        "split_id": split.split_id,
                        "frame": split.frame,
                        "room_id": split.room_id,
                    }
                    for split in prefix_splits
                ],
            }
        )
        manifest.write(output_path)
        return solver_result, manifest
    except Exception as exc:
        manifest.outcome = "error"
        manifest.notes.append(f"{type(exc).__name__}: {exc}")
        manifest.write(output_path)
        raise
    finally:
        env.close()


__all__ = [
    "EDGE_CLIMB",
    "EDGE_INJECTED",
    "EDGE_PARLOR",
    "EDGE_RECOVERY",
    "ACTION_SCHEMA_DIGEST",
    "CONTRACT_BUNDLE_DIGEST",
    "OBSERVATION_SCHEMA_DIGEST",
    "REAL_EDGE_IDS",
    "REWARD_SCHEMA_DIGEST",
    "RouteCommandPolicy",
    "RouteSkillCommand",
    "VERTICAL_SLICE_EVIDENCE",
    "VERTICAL_SLICE_MANIFEST",
    "VERTICAL_SLICE_COUNTEREXAMPLES",
    "VERTICAL_SLICE_TRAJECTORY",
    "VerticalSliceBundle",
    "apply_route_command",
    "build_solver_session",
    "build_vertical_slice_bundle",
    "observation_from_route_session",
    "run_real_vertical_slice",
]
