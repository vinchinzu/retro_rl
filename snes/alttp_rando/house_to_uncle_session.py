"""Fault-injection experiment and real-ROM evidence for the ALTTP solver adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alttp_rando.house_to_uncle import HOUSE_TO_UNCLE_EVIDENCE
from alttp_rando.logic_graph import N_LINKS_HOUSE, N_UNCLE
from alttp_rando.paths import (
    FIRST_PLAY_STATE,
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    REPO_ROOT,
    SHARED_Z3_JP_ROM,
)
from alttp_rando.solver_adapter import (
    EDGE_HOUSE_TO_UNCLE,
    OBSERVATION_SCHEMA_DIGEST,
    REAL_EDGE_IDS,
    HouseSolverSession,
    RouteCommandPolicy,
    RouteSkillCommand,
    SolverAdapterBundle,
    apply_route_command,
    build_solver_adapter_bundle,
    build_solver_session,
    composite_policy_identity,
    observation_from_session,
)
from alttp_rando.solver_bindings import HOUSE_TO_UNCLE_SPEC
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
    OneShotSkillPolicy,
    SkillInstance,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverSessionResult,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    counterexamples_from_solver_result,
    trajectory_from_solver_result,
)

HOUSE_TO_UNCLE_SESSION_MANIFEST = RECORDINGS_DIR / "house_to_uncle_session.run.json"
HOUSE_TO_UNCLE_SESSION_TRAJECTORY = (
    RECORDINGS_DIR / "house_to_uncle_session.trajectory.json"
)
HOUSE_TO_UNCLE_SESSION_COUNTEREXAMPLES = (
    RECORDINGS_DIR / "house_to_uncle_session_counterexamples"
)

EDGE_INJECTED = "house_to_uncle_injected_failure"
ACTION_SCHEMA_DIGEST = contract_digest(
    "alttp-rando-house-to-uncle-action-v1",
    {"kind": "RouteSkillCommand", "edge_ids": [EDGE_INJECTED, *REAL_EDGE_IDS]},
)
REWARD_SCHEMA_DIGEST = contract_digest(
    "alttp-rando-house-to-uncle-reward-v1",
    {"components": [{"name": "node_progress", "weight": 1.0}]},
)
CONTRACT_BUNDLE_DIGEST = contract_digest(
    "alttp-rando-house-to-uncle-contract-set-v1",
    {
        "observation_contract": OBSERVATION_SCHEMA_DIGEST,
        "action_contract": ACTION_SCHEMA_DIGEST,
        "reward_contract": REWARD_SCHEMA_DIGEST,
        "environment": GAME,
        "start": FIRST_PLAY_STATE,
    },
)


class InjectedFailurePolicy(OneShotSkillPolicy):
    """Deterministic experiment fault used to prove recovery replanning."""

    def __init__(self) -> None:
        super().__init__(
            SkillStep(
                SkillSignal.RETRYABLE_FAILURE,
                reason="injected house-exit desync",
                recovery_hint="exclude_primary_and_replan",
            )
        )


def _injected_spec() -> SkillSpec:
    return SkillSpec(
        skill_id="z3.house_to_uncle.injected_failure",
        dispatch_key="alttp_rando.house_to_uncle_session:injected_failure",
        observation_requirement=HOUSE_TO_UNCLE_SPEC.observation_requirement,
        expected_delta=HOUSE_TO_UNCLE_SPEC.expected_delta,
        timeout_frames=1,
        max_retries=0,
    )


def build_house_to_uncle_session_bundle() -> SolverAdapterBundle:
    """Add the isolated fault edge to a production-only adapter bundle."""
    production = build_solver_adapter_bundle()
    spec = _injected_spec()
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
            "z3.house_to_uncle.injected_failure",
            version="1",
            source="alttp_rando.house_to_uncle_session:InjectedFailurePolicy",
        ),
    )
    graph = RouteGraph(
        production.graph.nodes.values(),
        (
            GraphEdge(
                N_LINKS_HOUSE,
                N_UNCLE,
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


def run_real_house_to_uncle_session(
    output_path: Path = HOUSE_TO_UNCLE_SESSION_MANIFEST,
) -> tuple[SolverSessionResult, RunManifest]:
    """Load FirstPlay and run the house→uncle fault-injection experiment."""
    from retro_harness.env import make_env
    from retro_harness.play_spine import configure_display

    configure_display(headless=True)
    from alttp_rando.boot import ensure_first_play_state

    ensure_first_play_state()
    bundle = build_house_to_uncle_session_bundle()
    policy_identity = composite_policy_identity(bundle)
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(
            FIRST_PLAY_STATE,
            rom_sha256=sha256_file(SHARED_Z3_JP_ROM),
        ),
        policy_identity=policy_identity,
        benchmark_id="alttp_rando_house_to_uncle_session_v1",
        objective="Natural FirstPlay house→uncle with recovery replan",
    )
    env = AuditedEnv(
        make_env(GAME, FIRST_PLAY_STATE, GAME_DIR, render_mode=None),
        capabilities=AuditCapabilities.all(
            "alttp-rando.HouseSolverSession.audit-v1"
        ),
    )
    manifest = RunManifest(
        game=GAME,
        package="alttp_rando",
        started_at=utc_now_iso(),
        seed="vanilla-substrate",
        start_state=FIRST_PLAY_STATE,
        mode="bot",
    )
    try:
        env.reset()
        env.begin_attempt(**_attempt_identity(contract))
        session = HouseSolverSession(env)
        observation = observation_from_session(session)
        if observation.node_id != N_LINKS_HOUSE:
            raise RuntimeError("FirstPlay predecessor did not reach Link's House")
        mismatches = HOUSE_TO_UNCLE_SPEC.observation_requirement.mismatches(
            observation
        )
        if mismatches:
            raise RuntimeError(
                "FirstPlay observation failed house→uncle preconditions: "
                + ", ".join(mismatches)
            )

        solver_result = build_solver_session(
            session,
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
                "start": FIRST_PLAY_STATE,
                "evidence_path": str(HOUSE_TO_UNCLE_EVIDENCE.relative_to(REPO_ROOT)),
                "substrate": "vanilla",
            },
            reward_fn=lambda event: {
                "node_progress": float(
                    event.observation_before.node_id
                    != event.observation_after.node_id
                )
            },
        )
        trajectory.write(HOUSE_TO_UNCLE_SESSION_TRAJECTORY)
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
            HOUSE_TO_UNCLE_SESSION_COUNTEREXAMPLES
        )
        for counterexample in counterexamples:
            counterexample_library.add(counterexample)

        audit = env.audit()
        validate_claim(contract, audit)
        manifest.frames = session.frames
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
                "evidence_path": str(
                    HOUSE_TO_UNCLE_EVIDENCE.relative_to(REPO_ROOT)
                ),
                "real_edges": list(REAL_EDGE_IDS),
                "fault_edge": EDGE_INJECTED,
                "production_graph_edge_count": len(REAL_EDGE_IDS),
                "substrate": "vanilla",
                "seed_source": "fixture",
                "trajectory": {
                    "path": str(
                        HOUSE_TO_UNCLE_SESSION_TRAJECTORY.relative_to(REPO_ROOT)
                    ),
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
    "EDGE_HOUSE_TO_UNCLE",
    "EDGE_INJECTED",
    "HOUSE_TO_UNCLE_SESSION_COUNTEREXAMPLES",
    "HOUSE_TO_UNCLE_SESSION_MANIFEST",
    "HOUSE_TO_UNCLE_SESSION_TRAJECTORY",
    "OBSERVATION_SCHEMA_DIGEST",
    "REAL_EDGE_IDS",
    "REWARD_SCHEMA_DIGEST",
    "RouteCommandPolicy",
    "RouteSkillCommand",
    "apply_route_command",
    "build_house_to_uncle_session_bundle",
    "build_solver_session",
    "observation_from_session",
    "run_real_house_to_uncle_session",
]
