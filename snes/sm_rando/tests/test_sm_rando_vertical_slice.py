"""Offline and opt-in ROM checks for the first real SolverSession consumer."""

from __future__ import annotations

import pytest

from retro_harness.adventure import ExecutionReadiness, PlanRequest, plan
from retro_harness.solver import (
    SolverObservation,
    SolverResultStatus,
    SolverSession,
)
from sm_rando.vertical_slice import (
    EDGE_INJECTED,
    OBSERVATION_SCHEMA_DIGEST,
    REAL_EDGE_IDS,
    RouteSkillCommand,
    build_vertical_slice_bundle,
    run_real_vertical_slice,
)
from super_metroid.routes.kpdr.room_ids import ROOM_LANDING_SITE, ROOM_PIT


def test_three_real_edges_are_natural_entry_and_use_vanilla_controllers() -> None:
    bundle = build_vertical_slice_bundle()
    for edge_id in REAL_EDGE_IDS:
        binding = bundle.bindings.binding_for(edge_id)
        assert binding is not None
        assert binding.readiness is ExecutionReadiness.NATURAL_ENTRY
        instance = next(
            skill for skill in bundle.skills if skill.binding.edge_id == edge_id
        )
        assert instance.policy.command.runner.__module__.startswith("super_metroid.")


def test_recorded_boundary_replay_exercises_failure_replan_and_real_commands() -> None:
    bundle = build_vertical_slice_bundle()
    world = {"frame": 0, "room": ROOM_LANDING_SITE}

    def observe() -> SolverObservation:
        return SolverObservation(
            frame=world["frame"],
            node_id=world["room"],
            schema_digest=OBSERVATION_SCHEMA_DIGEST,
            values={"game_state": 8, "room_id": world["room"]},
        )

    def apply_action(action: RouteSkillCommand) -> int:
        assert isinstance(action, RouteSkillCommand)
        assert action.runner.__module__.startswith("super_metroid.")
        world["frame"] += 10
        world["room"] = action.target_room
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
                ROOM_PIT,
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
def test_real_vertical_slice_rom(tmp_path) -> None:
    import stable_retro as retro

    if not hasattr(retro.data.Integrations, "CUSTOM"):
        pytest.skip("stable_retro test stub cannot execute ROM smoke")
    result, manifest = run_real_vertical_slice(tmp_path / "vertical.json")
    assert result.status is SolverResultStatus.COMPLETED
    assert result.replans == 1
    assert result.completed_edges == REAL_EDGE_IDS
    assert manifest.meta["claim_valid"] is True
