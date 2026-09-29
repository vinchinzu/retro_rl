"""Leave a festival map the bot did not choose to enter.

Star Night (Winter D24) and New Year (Y2+ Spring D1) pull the farmer into tilemap 0x0A the moment he wakes
or opens the house door; the day only ends when he talks to the right villager
after 18:00 (measured: a seeded random walk with A pulses reached it in ~575
legs, ~30k frames, then the ROM warped him home at 18:00). NPCs wander, so no
fixed route exists; a deterministic roam-and-talk is the robust answer.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.nav import make_action

# Festival tilemaps (Maps_Graphics.asm): 08 flower, 09 harvest, 0A star night
# square, 0B egg, 3A New Year mountain top. The New Year exit is walled by an
# NPC until the evening; the same roam-and-talk got home at 18:02 (seed 3,
# 37k frames, Y2 Spring D1 pin).
FESTIVAL_MAPS = frozenset({0x08, 0x09, 0x0A, 0x0B, 0x3A})
_DIRS = ("up", "down", "left", "right")


@dataclass
class FestivalEscapeTask(Task):
    name: str = "festival_escape"
    seed: int = 0
    timeout: int = 90_000
    talk_chance: float = 0.35
    maps: frozenset = FESTIVAL_MAPS

    _rng: random.Random = field(default_factory=random.Random, init=False)
    _queue: deque = field(default_factory=deque, init=False)
    _steps: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._rng = random.Random(self.seed)
        self._queue.clear()
        self._steps = 0

    def can_start(self, world: WorldState) -> bool:
        return int(world.ram[ADDR_TILEMAP]) in self.maps

    def _refill(self) -> None:
        d = self._rng.choice(_DIRS)
        for _ in range(self._rng.choice((10, 30, 60))):
            self._queue.append(make_action(**{d: True}))
        if self._rng.random() < self.talk_chance:
            for j in range(40):
                self._queue.append(make_action(a=True) if j < 2 else make_action())

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        if int(world.ram[ADDR_TILEMAP]) not in self.maps:
            return TaskResult(status=TaskStatus.SUCCESS, reason="left festival map")
        if self._steps > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="festival escape timeout")
        if not self._queue:
            self._refill()
        return TaskResult(
            status=TaskStatus.RUNNING, action=ActionResult(self._queue.popleft())
        )
