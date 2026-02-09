"""
Core harness interfaces for composable task-based game automation.

These are intentionally minimal to allow multiple games and runners to share
common abstractions without coupling to a specific emulator or RAM layout.
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
