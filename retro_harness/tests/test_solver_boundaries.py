"""Compatibility checks for the decomposed solver public API."""

import retro_harness
from retro_harness import skill_policies, solver, solver_domain, solver_session


DOMAIN_SYMBOLS = (
    "ObservationRequirement",
    "ProgressionDelta",
    "SkillInstance",
    "SkillOutcome",
    "SkillOutcomeStatus",
    "SkillPolicy",
    "SkillSignal",
    "SkillSpec",
    "SkillStep",
    "SolverActionEvent",
    "SolverLifecycle",
    "SolverObservation",
    "SolverResultStatus",
    "SolverSessionResult",
    "SolverTraceEvent",
    "canonical_action_record",
)

SESSION_SYMBOLS = (
    "PlanFunction",
    "SolverSession",
)

POLICY_HELPER_SYMBOLS = (
    "OneShotSkillPolicy",
    "ScriptedSkillPolicy",
)

ROOT_FACADE_SYMBOLS = (
    "ObservationRequirement",
    "OneShotSkillPolicy",
    "ProgressionDelta",
    "ScriptedSkillPolicy",
    "SkillInstance",
    "SkillOutcome",
    "SkillOutcomeStatus",
    "SkillPolicy",
    "SkillSignal",
    "SkillSpec",
    "SkillStep",
    "SolverLifecycle",
    "SolverObservation",
    "SolverResultStatus",
    "SolverSession",
    "SolverSessionResult",
    "SolverActionEvent",
    "SolverTraceEvent",
    "canonical_action_record",
)


def test_solver_domain_facade_preserves_canonical_objects() -> None:
    for name in DOMAIN_SYMBOLS:
        assert getattr(solver, name) is getattr(solver_domain, name)


def test_solver_session_facade_preserves_canonical_objects() -> None:
    for name in SESSION_SYMBOLS:
        assert getattr(solver, name) is getattr(solver_session, name)


def test_solver_policy_helper_facade_preserves_canonical_objects() -> None:
    for name in POLICY_HELPER_SYMBOLS:
        assert getattr(solver, name) is getattr(skill_policies, name)


def test_root_facade_preserves_solver_objects() -> None:
    for name in ROOT_FACADE_SYMBOLS:
        assert getattr(retro_harness, name) is getattr(solver, name)
