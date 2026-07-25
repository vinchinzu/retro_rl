"""Minimal behavior-tree nodes for scripted oneshot controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from snes_oneshot.actions import idle_action
from snes_oneshot.game_state import GameState
from snes_oneshot.primitives import FrameAction


class NodeStatus(Enum):
    """Behavior-tree tick result."""

    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class TickResult:
    """Outcome of one behavior-tree tick."""

    status: NodeStatus
    action: FrameAction | None = None
    reason: str = ""


class BehaviorNode(ABC):
    """Base behavior-tree node."""

    name: str = "BehaviorNode"

    @abstractmethod
    def tick(self, state: GameState) -> TickResult:
        """Evaluate this node against the current game state."""


class Condition(BehaviorNode):
    """Succeeds when predicate(state) is true; otherwise fails."""

    def __init__(
        self,
        predicate: Callable[[GameState], bool],
        *,
        name: str = "Condition",
    ) -> None:
        self._predicate = predicate
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        ok = self._predicate(state)
        return TickResult(
            status=NodeStatus.SUCCESS if ok else NodeStatus.FAILURE,
            reason=self.name,
        )


class ActionNode(BehaviorNode):
    """Emit a single FrameAction producer each tick while RUNNING."""

    def __init__(
        self,
        producer: Callable[[GameState], FrameAction],
        *,
        done: Callable[[GameState], bool] | None = None,
        name: str = "Action",
    ) -> None:
        self._producer = producer
        self._done = done
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        if self._done is not None and self._done(state):
            return TickResult(status=NodeStatus.SUCCESS, reason=self.name)
        return TickResult(
            status=NodeStatus.RUNNING,
            action=self._producer(state),
            reason=self.name,
        )


class Sequence(BehaviorNode):
    """Run children in order; fail on first failure (reactive each tick)."""

    def __init__(self, children: list[BehaviorNode], *, name: str = "Sequence") -> None:
        self.children = list(children)
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        for child in self.children:
            result = child.tick(state)
            if result.status is NodeStatus.RUNNING:
                return TickResult(
                    status=NodeStatus.RUNNING,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
            if result.status is NodeStatus.FAILURE:
                return TickResult(
                    status=NodeStatus.FAILURE,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
        return TickResult(status=NodeStatus.SUCCESS, reason=self.name)


class Selector(BehaviorNode):
    """Try children until one succeeds or runs; fail if all fail."""

    def __init__(self, children: list[BehaviorNode], *, name: str = "Selector") -> None:
        self.children = list(children)
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        for child in self.children:
            result = child.tick(state)
            if result.status is not NodeStatus.FAILURE:
                return TickResult(
                    status=result.status,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
        return TickResult(
            status=NodeStatus.FAILURE,
            action=FrameAction(action=idle_action(), reason="selector_idle"),
            reason=self.name,
        )
