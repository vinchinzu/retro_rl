"""Compatibility facade for solver domain types and SolverSession runtime.

Canonical immutable types live in :mod:`retro_harness.solver_domain`;
execution lifecycle lives in :mod:`retro_harness.solver_session`.
Concrete one-shot / multi-frame policy helpers live in
:mod:`retro_harness.skill_policies`.
"""

from retro_harness.skill_policies import OneShotSkillPolicy, ScriptedSkillPolicy
from retro_harness.solver_domain import (
    ObservationRequirement,
    ProgressionDelta,
    SkillInstance,
    SkillOutcome,
    SkillOutcomeStatus,
    SkillPolicy,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverActionEvent,
    SolverLifecycle,
    SolverObservation,
    SolverResultStatus,
    SolverSessionResult,
    SolverTraceEvent,
    canonical_action_record,
)
from retro_harness.solver_session import PlanFunction, SolverSession

__all__ = [
    "ObservationRequirement",
    "OneShotSkillPolicy",
    "PlanFunction",
    "ProgressionDelta",
    "ScriptedSkillPolicy",
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
    "SolverSession",
    "SolverSessionResult",
    "SolverTraceEvent",
    "canonical_action_record",
]
