"""Task graph with checkpoints and simple fallback chaining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from harness import Task, TaskResult, TaskStatus, WorldState


@dataclass
class TaskNode:
    name: str
    task: Task
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    checkpoint: Optional[str] = None


class TaskGraph:
    """BT-lite graph: nodes are tasks with explicit transitions."""

    def __init__(self, nodes: Iterable[TaskNode], start: str):
        self.nodes: Dict[str, TaskNode] = {node.name: node for node in nodes}
        if start not in self.nodes:
            raise ValueError(f"Unknown start node: {start}")
        self.start = start
        self.current: Optional[TaskNode] = None
        self.last_checkpoint: Optional[str] = None

    def reset(self, world: WorldState) -> None:
        self.current = self.nodes[self.start]
        self.last_checkpoint = self.current.checkpoint
        self.current.task.reset(world)

    def can_start(self, world: WorldState) -> bool:
        node = self.nodes[self.start]
        return node.task.can_start(world)

    def step(self, world: WorldState) -> TaskResult:
        if self.current is None:
            self.reset(world)

        node = self.current
        result = node.task.step(world)

        if result.checkpoint:
            self.last_checkpoint = result.checkpoint

        if result.status == TaskStatus.RUNNING:
            return result

        if result.status == TaskStatus.SUCCESS:
            if node.on_success is None:
                return result
            return self._transition(node.on_success, world, status_hint=TaskStatus.RUNNING)

        if result.status == TaskStatus.FAILURE:
            if node.on_failure is None:
                return result
            return self._transition(node.on_failure, world, status_hint=TaskStatus.RUNNING)

        return result

    def _transition(self, next_name: str, world: WorldState, status_hint: TaskStatus) -> TaskResult:
        if next_name not in self.nodes:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown node {next_name}")
        self.current = self.nodes[next_name]
        self.current.task.reset(world)
        if self.current.checkpoint:
            self.last_checkpoint = self.current.checkpoint
        return TaskResult(status=status_hint, checkpoint=self.last_checkpoint)

    def current_name(self) -> Optional[str]:
        return self.current.name if self.current else None


class LinearChain(TaskGraph):
    """Convenience for linear sequences of tasks."""

    @classmethod
    def from_tasks(cls, tasks: Iterable[Task]) -> "LinearChain":
        nodes = []
        prev_name = None
        for idx, task in enumerate(tasks):
            name = getattr(task, "name", f"task_{idx}")
            node = TaskNode(name=name, task=task)
            if prev_name is not None:
                nodes[-1].on_success = name
            nodes.append(node)
            prev_name = name
        if not nodes:
            raise ValueError("LinearChain requires at least one task")
        return cls(nodes=nodes, start=nodes[0].name)
