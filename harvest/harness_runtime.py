"""Runtime harness utilities: step loop, observation cache, determinism hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional
from collections import deque

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState


@dataclass
class DeterminismHooks:
    """Hooks to keep executions deterministic and debuggable."""

    seed: Optional[int] = None
    sanitize_action: Optional[Callable[[np.ndarray], np.ndarray]] = None
    before_step: Optional[Callable[[WorldState, np.ndarray], None]] = None
    after_step: Optional[Callable[[WorldState], None]] = None


@dataclass
class ObservationCache:
    """Ring buffer of recent world states for debugging/replay."""

    maxlen: int = 1200
    _buffer: Deque[WorldState] = field(default_factory=deque, init=False)

    def push(self, state: WorldState) -> None:
        if len(self._buffer) >= self.maxlen:
            self._buffer.popleft()
        self._buffer.append(state)

    def recent(self) -> Deque[WorldState]:
        return self._buffer


def _zero_action(env) -> np.ndarray:
    if hasattr(env, "action_space"):
        space = env.action_space
        shape = getattr(space, "shape", None)
        if shape:
            dtype = getattr(space, "dtype", np.int32)
            return np.zeros(shape, dtype=dtype)
        if hasattr(space, "n"):
            return 0  # Discrete action
    return np.zeros(12, dtype=np.int32)


class HarnessRunner:
    """Generic runner to execute Task implementations against an env."""

    def __init__(self, env, hooks: Optional[DeterminismHooks] = None, cache: Optional[ObservationCache] = None):
        self.env = env
        self.hooks = hooks or DeterminismHooks()
        self.cache = cache or ObservationCache()
        self.frame = 0

    def reset(self, state: Optional[str] = None) -> WorldState:
        if self.hooks.seed is not None and hasattr(self.env, "seed"):
            self.env.seed(self.hooks.seed)

        if state is not None:
            obs, info = self.env.reset(state=state)
        else:
            obs, info = self.env.reset()

        ram = self._get_ram()
        world = WorldState(frame=self.frame, obs=obs, info=info, ram=ram)
        self.cache.push(world)
        return world

    def _get_ram(self) -> np.ndarray:
        if hasattr(self.env, "get_ram"):
            return self.env.get_ram()
        return np.zeros(0, dtype=np.uint8)

    def step_env(self, action: np.ndarray) -> WorldState:
        if self.hooks.sanitize_action is not None:
            action = self.hooks.sanitize_action(action)

        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, _, _, _, info = step_out
        else:
            obs, _, _, info = step_out
        self.frame += 1
        ram = self._get_ram()
        world = WorldState(frame=self.frame, obs=obs, info=info, ram=ram)
        self.cache.push(world)
        return world

    def run_task(self, task: Task, world: WorldState, max_steps: int = 5000) -> TaskResult:
        if not task.can_start(world):
            return TaskResult(status=TaskStatus.BLOCKED, reason="preconditions failed")

        task.reset(world)
        current = world

        for _ in range(max_steps):
            result = task.step(current)

            if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
                return result

            action = result.action.action if result.action else _zero_action(self.env)

            if self.hooks.before_step is not None:
                self.hooks.before_step(current, action)

            current = self.step_env(action)

            if self.hooks.after_step is not None:
                self.hooks.after_step(current)

        return TaskResult(status=TaskStatus.FAILURE, reason="max steps exceeded")

    def run_action_sequence(self, actions, world: WorldState, max_steps: Optional[int] = None) -> WorldState:
        current = world
        steps = 0
        for action in actions:
            if max_steps is not None and steps >= max_steps:
                break
            current = self.step_env(action)
            steps += 1
        return current
