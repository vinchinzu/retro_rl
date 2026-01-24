"""Recorded task adapter for harness Task protocol."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState


@dataclass
class RecordedTask(Task):
    name: str
    frames: List[List[int]]
    start_state: Optional[str] = None
    _idx: int = 0

    def reset(self, world: WorldState) -> None:
        self._idx = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        if self._idx >= len(self.frames):
            return TaskResult(status=TaskStatus.SUCCESS)
        action = np.array(self.frames[self._idx], dtype=np.int32)
        self._idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    @classmethod
    def load(cls, name: str, tasks_dir: str = "tasks") -> "RecordedTask":
        path = os.path.join(tasks_dir, f"{name}.json")
        with open(path, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not frames:
            print(f"[RECORDRD_TASK] Warning: {name} has no frames")
        else:
            # Check for "dead" recordings (mostly zeros)
            non_zero = sum(1 for f in frames if any(v != 0 for v in f))
            if non_zero < len(frames) * 0.1:
                print(f"[RECORDED_TASK] Warning: {name} is >90% empty ({non_zero}/{len(frames)} non-zero)")
        return cls(name=data.get("name", name), frames=frames, start_state=data.get("start_state"))
