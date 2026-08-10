"""Fault-injection experiment and real-ROM evidence for the SM solver adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from retro_harness.adventure import (
    BindingCatalog,
    ExecutionReadiness,
    GraphEdge,
    RouteGraph,
    SkillBinding,
)
from retro_harness.audit import (
    AuditCapabilities,
    AuditedEnv,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.benchmark_claims import (
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    validate_claim,
)
from retro_harness.contracts import contract_digest
from retro_harness.identity import sha256_bytes, sha256_file
from retro_harness.play_spine import RunManifest, utc_now_iso
from retro_harness.solver import (
    SkillInstance,
    SkillSignal,
    SkillStep,
    SolverObservation,
    SolverSessionResult,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    counterexamples_from_solver_result,
    trajectory_from_solver_result,
)
from sm_rando.paths import GAME, GAME_DIR, RECORDINGS_DIR, REPO_ROOT, SHARED_SM_ROM
from sm_rando.solver_adapter import (
    EDGE_CLIMB,
    EDGE_PARLOR,
    EDGE_RECOVERY,
    OBSERVATION_SCHEMA_DIGEST,
    REAL_EDGE_IDS,
    RouteCommandPolicy,
    RouteSkillCommand,
    SolverAdapterBundle,
    VERTICAL_SLICE_EVIDENCE,
    apply_route_command,
    build_solver_adapter_bundle,
    build_solver_session,
    composite_policy_identity,
    observation_from_route_session,
    route_skill_spec,
)
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.progression import MORPH_GRAPH
from super_metroid.routes.kpdr.early_spine import MORPH_SPINE
from super_metroid.routes.kpdr.room_ids import (
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
)
from super_metroid.routes.runtime import RouteSession, Split
from super_metroid.routes.tips import play_hops

VERTICAL_SLICE_MANIFEST = RECORDINGS_DIR / "vertical_slice.run.json"
VERTICAL_SLICE_TRAJECTORY = RECORDINGS_DIR / "vertical_slice.trajectory.json"
VERTICAL_SLICE_COUNTEREXAMPLES = RECORDINGS_DIR / "vertical_slice_counterexamples"

EDGE_INJECTED = "landing_to_parlor_injected_failure"
ACTION_SCHEMA_DIGEST = contract_digest(
    "sm-rando-vertical-action-v1",
    {"kind": "RouteSkillCommand", "edge_ids": [EDGE_INJECTED, *REAL_EDGE_IDS]},
)
REWARD_SCHEMA_DIGEST = contract_digest(
    "sm-rando-vertical-reward-v1",
    {"components": [{"name": "room_progress", "weight": 1.0}]},
)
CONTRACT_BUNDLE_DIGEST = contract_digest(
    "sm-rando-vertical-contract-set-v1",
    {
        "observation_contract": OBSERVATION_SCHEMA_DIGEST,
        "action_contract": ACTION_SCHEMA_DIGEST,
        "reward_contract": REWARD_SCHEMA_DIGEST,
        "environment": GAME,
        "start": "NONE",
    },
)

VerticalSliceBundle = SolverAdapterBundle


class InjectedFailurePolicy:
    """Deterministic experiment fault used to prove recovery replanning."""

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


def build_vertical_slice_bundle(
    evidence_path: Path = VERTICAL_SLICE_EVIDENCE,
) -> SolverAdapterBundle:
    """Add the isolated fault edge to a production-only adapter bundle."""
    production = build_solver_adapter_bundle(evidence_path)
    spec = route_skill_spec(EDGE_INJECTED, ROOM_LANDING_SITE, ROOM_PARLOR)
    binding = SkillBinding(
        edge_id=EDGE_INJECTED,
        skill_id=spec.skill_id,
        dispatch_key=spec.dispatch_key,
        entry_requirement_digest=spec.observation_requirement.identity_digest,
        progression_delta_digest=spec.expected_delta.identity_digest,
        readiness=ExecutionReadiness.ISOLATED,
        evidence_digest=sha256_bytes(b"fault-injection-v1"),
    )
    fault = SkillInstance(
        spec=spec,
        binding=binding,
        policy=InjectedFailurePolicy(),
        policy_identity=PolicyIdentity(
            "sm.vertical.injected_failure",
            version="1",
            source="sm_rando.vertical_slice:InjectedFailurePolicy",
        ),
    )
    graph = RouteGraph(
        production.graph.nodes.values(),
        (
            GraphEdge(
                ROOM_LANDING_SITE,
                ROOM_PARLOR,
                edge_id=EDGE_INJECTED,
                cost=0.5,
            ),
            *production.graph.edges,
        ),
    )
    return SolverAdapterBundle(
        graph=graph,
        bindings=BindingCatalog((*production.bindings.bindings, binding)),
        skills=(*production.skills, fault),
    )


def _attempt_identity(contract: EvaluationContract) -> dict[str, Any]:
    return {
        "start_identity_digest": contract.start_identity.identity_digest,
        "policy_identity_digest": contract.policy_identity.identity_digest,
        "runtime_observation_class": contract.runtime_observation_class,
        "intervention_class": contract.intervention_class,
    }


def run_real_vertical_slice(
    output_path: Path = VERTICAL_SLICE_MANIFEST,
) -> tuple[SolverSessionResult, RunManifest]:
    """Power on vanilla SM and run the fault-injection experiment."""
    from retro_harness.env import make_env

    bundle = build_vertical_slice_bundle()
    policy_identity = composite_policy_identity(bundle)
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(
            "power_on_to_natural_landing",
            rom_sha256=sha256_file(SHARED_SM_ROM),
        ),
        policy_identity=policy_identity,
        benchmark_id="sm_rando_vertical_slice_v1",
        objective="Natural Landing Site to Pit Room with recovery replan",
    )
    env = AuditedEnv(
        make_env(GAME, "NONE", GAME_DIR, render_mode=None),
        capabilities=AuditCapabilities.all("sm-rando.RouteSession.audit-v2"),
    )
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
        env.begin_attempt(**_attempt_identity(contract))
        route_session = RouteSession(
            env,
            writer=None,
            assist=UnlimitedAmmoAssist(enabled=False),
            graph=MORPH_GRAPH,
        )
        prefix_splits: list[Split] = []
        play_hops(route_session, prefix_splits, MORPH_SPINE[:3])
        if route_session.state.room_id != ROOM_LANDING_SITE:
            raise RuntimeError("natural predecessor did not reach Landing Site")

        solver_result = build_solver_session(
            route_session,
            bundle,
            minimum_readiness=ExecutionReadiness.ISOLATED,
            max_replans=1,
        ).run()
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
        counterexample_library = CounterexampleLibrary(
            VERTICAL_SLICE_COUNTEREXAMPLES
        )
        for counterexample in counterexamples:
            counterexample_library.add(counterexample)

        audit = env.audit()
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
                "schema_version": 2,
                "solver_result": solver_result.to_record(),
                "benchmark_contract": contract.to_record(),
                "attempt_audit": audit.to_record(),
                "claim_valid": True,
                "evidence_path": str(VERTICAL_SLICE_EVIDENCE.relative_to(REPO_ROOT)),
                "real_edges": list(REAL_EDGE_IDS),
                "fault_edge": EDGE_INJECTED,
                "production_graph_edge_count": len(REAL_EDGE_IDS),
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
    "ACTION_SCHEMA_DIGEST",
    "CONTRACT_BUNDLE_DIGEST",
    "EDGE_CLIMB",
    "EDGE_INJECTED",
    "EDGE_PARLOR",
    "EDGE_RECOVERY",
    "OBSERVATION_SCHEMA_DIGEST",
    "REAL_EDGE_IDS",
    "REWARD_SCHEMA_DIGEST",
    "RouteCommandPolicy",
    "RouteSkillCommand",
    "VERTICAL_SLICE_COUNTEREXAMPLES",
    "VERTICAL_SLICE_EVIDENCE",
    "VERTICAL_SLICE_MANIFEST",
    "VERTICAL_SLICE_TRAJECTORY",
    "VerticalSliceBundle",
    "apply_route_command",
    "build_solver_session",
    "build_vertical_slice_bundle",
    "observation_from_route_session",
    "run_real_vertical_slice",
]
