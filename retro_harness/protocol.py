"""
Core harness interfaces for composable task-based game automation.

These legacy task interfaces remain intentionally minimal. New graph-solver
skills use :mod:`retro_harness.solver`; the aliases at the end of this module
make this file a thin compatibility facade while task consumers migrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol

import numpy as np


class TaskStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class WorldState:
    """Snapshot of the current game state used by tasks."""

    frame: int
    ram: np.ndarray
    info: Dict[str, Any]
    obs: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    """Action produced by a task step plus optional diagnostics."""

    action: np.ndarray
    reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a task step."""

    status: TaskStatus
    action: Optional[ActionResult] = None
    reason: Optional[str] = None
    checkpoint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    """Atomic behavior that emits actions until it completes."""

    name: str

    def reset(self, world: WorldState) -> None:
        ...

    def can_start(self, world: WorldState) -> bool:
        ...

    def step(self, world: WorldState) -> TaskResult:
        ...


# Thin facade for the typed solver lifecycle. Legacy Task/TaskResult remains
# supported for Harvest and bot_runner consumers.
from retro_harness.solver import (  # noqa: E402
    SkillInstance,
    SkillOutcome,
    SkillOutcomeStatus,
    SkillPolicy,
    SkillSpec,
)

Skill = SkillPolicy

__all__ = [
    "ActionResult",
    "Skill",
    "SkillInstance",
    "SkillOutcome",
    "SkillOutcomeStatus",
    "SkillPolicy",
    "SkillSpec",
    "Task",
    "TaskResult",
    "TaskStatus",
    "WorldState",
]
