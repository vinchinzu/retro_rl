"""Offline and opt-in ROM checks for the ALTTP SolverSession second consumer."""

from __future__ import annotations

import pytest

from alttp.ram import AlttpSnapshot
from alttp_rando.house_to_uncle import play_house_to_uncle
from alttp_rando.house_to_uncle_session import (
    EDGE_INJECTED,
    OBSERVATION_SCHEMA_DIGEST,
    REAL_EDGE_IDS,
    RouteSkillCommand,
    build_house_to_uncle_session_bundle,
    run_real_house_to_uncle_session,
)
from alttp_rando.logic_graph import N_LINKS_HOUSE, N_UNCLE
from alttp_rando.solver_adapter import (
    EDGE_HOUSE_TO_UNCLE,
    build_solver_adapter_bundle,
    node_id_from_snapshot,
)
from alttp_rando.solver_bindings import HOUSE_TO_UNCLE_SPEC
from retro_harness.adventure import ExecutionReadiness, PlanRequest, plan
from retro_harness.solver import (
    SolverObservation,
    SolverResultStatus,
    SolverSession,
)


def _snap(**overrides: object) -> AlttpSnapshot:
    base = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=0x0104,
        indoors=True,
        screen_id=0,
        link_x=2368,
        link_y=8538,
        link_direction=2,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
    )
    base.update(overrides)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_house_to_uncle_is_natural_entry_and_uses_opening_skill() -> None:
    bundle = build_house_to_uncle_session_bundle()
    for edge_id in REAL_EDGE_IDS:
        binding = bundle.bindings.binding_for(edge_id)
        assert binding is not None
        assert binding.readiness is ExecutionReadiness.NATURAL_ENTRY
        instance = next(
            skill for skill in bundle.skills if skill.binding.edge_id == edge_id
        )
        assert instance.policy.command.runner is play_house_to_uncle
        assert instance.spec.skill_id == HOUSE_TO_UNCLE_SPEC.skill_id


def test_production_adapter_has_only_real_edges() -> None:
    bundle = build_solver_adapter_bundle()

    assert tuple(edge.edge_id for edge in bundle.graph.edges) == REAL_EDGE_IDS
    assert bundle.bindings.binding_for(EDGE_INJECTED) is None
    binding = bundle.bindings.binding_for(EDGE_HOUSE_TO_UNCLE)
    assert binding is not None
    assert binding.readiness is ExecutionReadiness.NATURAL_ENTRY


def test_node_id_from_snapshot_maps_house_and_uncle() -> None:
    house = _snap()
    uncle = _snap(room_id=0x55, sword_level=1)
    assert node_id_from_snapshot(house) == N_LINKS_HOUSE
    assert node_id_from_snapshot(uncle) == N_UNCLE


def test_recorded_boundary_replay_exercises_failure_replan_and_real_command() -> None:
    bundle = build_house_to_uncle_session_bundle()
    world = {"frame": 0, "node": N_LINKS_HOUSE, "sword": False}

    def observe() -> SolverObservation:
        at_house = world["node"] == N_LINKS_HOUSE
        return SolverObservation(
            frame=world["frame"],
            node_id=world["node"],
            schema_digest=OBSERVATION_SCHEMA_DIGEST,
            capabilities=frozenset({"sword"}) if world["sword"] else frozenset(),
            values={
                "game_mode": 0x07,
                "room_base_id": 0x04 if at_house else 0x55,
                "indoors": True,
                "has_control": True,
                "has_fighter_sword": world["sword"],
            },
        )

    def apply_action(action: RouteSkillCommand) -> int:
        assert isinstance(action, RouteSkillCommand)
        assert action.runner is play_house_to_uncle
        world["frame"] += 10
        world["node"] = action.target_node
        world["sword"] = True
        return 10

    def plan_fn(observation, excluded):
        return plan(
            PlanRequest(
                tuple(
                    edge
                    for edge in bundle.graph.edges
                    if edge.edge_id not in excluded
                ),
                observation.node_id,
                N_UNCLE,
            )
        )

    result = SolverSession(
        observe=observe,
        apply_action=apply_action,
        plan_fn=plan_fn,
        bindings=bundle.bindings,
        skills=bundle.skills,
        minimum_readiness=ExecutionReadiness.ISOLATED,
        max_replans=1,
    ).run()

    assert result.status is SolverResultStatus.COMPLETED
    assert result.replans == 1
    assert result.outcomes[0].edge_id == EDGE_INJECTED
    assert result.outcomes[0].replan
    assert result.completed_edges == REAL_EDGE_IDS


@pytest.mark.rom
@pytest.mark.rom_smoke
def test_real_house_to_uncle_session_rom(tmp_path) -> None:
    import stable_retro as retro

    if not hasattr(retro.data.Integrations, "CUSTOM"):
        pytest.skip("stable_retro test stub cannot execute ROM smoke")
    from alttp_rando.paths import FIRST_PLAY_STATE, INTEGRATION_DIR, SHARED_Z3_JP_ROM

    if not SHARED_Z3_JP_ROM.is_file():
        pytest.skip("JP 1.0 ROM not present")
    if not (INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state").is_file():
        pytest.skip("FirstPlay.state not present")
    result, manifest = run_real_house_to_uncle_session(tmp_path / "session.json")
    assert result.status is SolverResultStatus.COMPLETED
    assert result.replans == 1
    assert result.completed_edges == REAL_EDGE_IDS
    assert manifest.meta["claim_valid"] is True
    assert manifest.meta["substrate"] == "vanilla"
