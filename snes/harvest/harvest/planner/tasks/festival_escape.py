"""Leave a festival map the bot did not choose to enter.

Star Night (Winter D24) and New Year (Y2+ Spring D1) pull the farmer onto a
festival map at 6:03 (open the door, press the bed, or wake). The ROM only sends
him home once he has talked to the villagers there after 18:00; measured with
NPC-tagged logs of a random roam, the exit fired on the LAST villager of the
set (Star Night: the boy, slot 4 of 0-4; New Year: the woman blocking the gap,
slot 1, after every other slot). NPCs wander, so this task tracks them live
from the game-object table and walks into each one in turn (nearest first),
looping the whole set, then falls back to a seeded random walk with a logged
bound. The clock idles at 18:xx forever, so there is no hour to wait for.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.nav import get_pos_from_ram, make_action

# Festival tilemaps (Maps_Graphics.asm): 08 flower, 09 harvest, 0A star night
# square, 0B egg, 3A New Year mountain top.
FESTIVAL_MAPS = frozenset({0x08, 0x09, 0x0A, 0x0B, 0x3A})


def day_plan_owns_festival(task: object) -> bool:
    """The live day plan is already walking the Flower Festival on map 0x08.

    That script talks and dances, then the ROM puts the farmer to bed.
    Replacing it with this roam is what left Spring D23 stuck in festival
    text. Star Night and New Year still use the roam; nobody scripts those.
    """
    child = getattr(task, "_current_task", None)
    if type(child).__name__ == "FlowerFestivalTask":
        return True
    phase = getattr(task, "current_phase", None)
    return getattr(phase, "kind", "") == "flower_festival"
_DIRS = ("up", "down", "left", "right")
_TALK_DX, _TALK_DY = 26, 32
_STUCK_FRAMES = 12
_ALIGN = 3
_SIDE_FRAMES = 450
_FROZEN_FRAMES = 300
# Approach spots relative to a villager's sprite pixel, with the facing to use.
# Measured: the Star Night boy only answers from below (27px, facing up); the
# New Year woman answered from her left (+23 -> face left) and right (-13).
_SIDES = (
    (0, 27, "up"),
    (23, 0, "left"),
    (-13, 0, "right"),
    (0, -27, "down"),
)


@dataclass
class FestivalEscapeTask(Task):
    name: str = "festival_escape"
    seed: int = 0
    timeout: int = 110_000
    # Frames spent on the scripted visit loop before the random fallback.
    scripted_budget: int = 60_000
    talk_chance: float = 0.35
    maps: frozenset = FESTIVAL_MAPS

    _rng: random.Random = field(default_factory=random.Random, init=False)
    _queue: deque = field(default_factory=deque, init=False)
    _steps: int = field(default=0, init=False)
    _visited: set = field(default_factory=set, init=False)
    _target: int | None = field(default=None, init=False)
    _last_pos: tuple = field(default=(0, 0), init=False)
    _still: int = field(default=0, init=False)
    _lock: int = field(default=1, init=False)
    _cooldown: int = field(default=0, init=False)
    _clock: tuple = field(default=(-1, -1), init=False)
    _clock_frame: int = field(default=0, init=False)
    _evening_seen: bool = field(default=False, init=False)
    _order: list = field(default_factory=lambda: [0, 1, 2, 3], init=False)
    _side: int = field(default=0, init=False)
    _side_start: int = field(default=0, init=False)
    _rounds: int = field(default=0, init=False)
    _fallback_logged: bool = field(default=False, init=False)
    talks: list = field(default_factory=list, init=False)

    def reset(self, world: WorldState) -> None:
        self._rng = random.Random(self.seed)
        self._queue.clear()
        self._steps = 0
        self._visited = set()
        self._target = None
        self._still = 0
        self._lock = 1
        self._cooldown = 0
        self._rounds = 0
        self._side = 0
        self._side_start = 0
        self._clock, self._clock_frame, self._evening_seen = (-1, -1), 0, False
        self._fallback_logged = False
        self.talks = []

    def can_start(self, world: WorldState) -> bool:
        return int(world.ram[ADDR_TILEMAP]) in self.maps

    # -- random fallback -------------------------------------------------
    def _refill(self) -> None:
        d = self._rng.choice(_DIRS)
        for _ in range(self._rng.choice((10, 30, 60))):
            self._queue.append(make_action(**{d: True}))
        if self._rng.random() < self.talk_chance:
            for j in range(40):
                self._queue.append(make_action(a=True) if j < 2 else make_action())

    # -- scripted visit --------------------------------------------------
    def _npcs(self, ram):
        return {o.slot: o.pixel for o in game_objects(ram) if not o.is_player}

    def _pick_target(self, npcs, pos):
        left = [s for s in npcs if s not in self._visited]
        if not left:
            self._visited.clear()
            self._rounds += 1
            left = list(npcs)
        if not self._evening_seen:
            # Clock frozen on arrival: the story beat (New Year prayer) is the
            # northernmost villager (the priest at the summit); go there first.
            return min(left, key=lambda s: npcs[s][1])
        return min(
            left,
            key=lambda s: (npcs[s][0] - pos[0]) ** 2 + (npcs[s][1] - pos[1]) ** 2,
        )

    def _scripted_step(self, ram) -> ActionResult:
        lock = int(read_ram_value(ram, "input_lock"))
        p = get_pos_from_ram(ram)
        pos = (int(p.x), int(p.y))
        npcs = self._npcs(ram)
        rising = lock == 2 and self._lock == 1
        self._lock = lock
        if rising and npcs:
            near = min(
                npcs,
                key=lambda s: (npcs[s][0] - pos[0]) ** 2 + (npcs[s][1] - pos[1]) ** 2,
            )
            self._visited.add(near)
            self.talks.append((self._steps, near))
            if self._target == near:
                self._target = None
        clock = (int(read_ram_value(ram, "hour")), int(read_ram_value(ram, "minute")))
        if clock != self._clock:
            self._clock, self._clock_frame = clock, self._steps
        evening = clock[0] >= 18
        if evening and not self._evening_seen:
            # Only villagers talked to after 18:00 count: start the set afresh.
            self._evening_seen = True
            self._visited.clear()
            self._target = None
        # Before evening the clock only runs once the story beat (New Year: the
        # priest) is done; while it is frozen keep talking, once it runs just wait.
        waiting = not evening and self._steps - self._clock_frame < _FROZEN_FRAMES
        if lock != 1:
            self._cooldown = 30
            phase = self._steps % 2
            return ActionResult(make_action(a=True) if phase == 0 else make_action())
        if self._cooldown > 0:
            self._cooldown -= 1
            return ActionResult(make_action())
        if waiting and self._steps > _FROZEN_FRAMES:
            return ActionResult(make_action())
        if not npcs:
            return ActionResult(make_action())
        if self._queue:
            self._last_pos = pos
            return ActionResult(self._queue.popleft())
        if self._target is None or self._target not in npcs:
            self._target = self._pick_target(npcs, pos)
            tx0, ty0 = npcs[self._target]
            # Nearest approach spot first.
            self._order = sorted(
                range(len(_SIDES)),
                key=lambda i: (tx0 + _SIDES[i][0] - pos[0]) ** 2
                + (ty0 + _SIDES[i][1] - pos[1]) ** 2,
            )
            self._side, self._side_start = 0, self._steps
        elif self._steps - self._side_start > _SIDE_FRAMES:
            # This approach side never produced a dialogue: try the next.
            self._side += 1
            self._side_start = self._steps
            if self._side >= len(self._order):
                self._visited.add(self._target)
                self._target = None
                return ActionResult(make_action())
        tx, ty = npcs[self._target]
        ox, oy, face = _SIDES[self._order[self._side]]
        gx, gy = tx + ox, ty + oy
        dx, dy = gx - pos[0], gy - pos[1]
        self._still = self._still + 1 if pos == self._last_pos else 0
        self._last_pos = pos
        if abs(dx) <= _ALIGN and abs(dy) <= _ALIGN:
            # In place: turn (hold toward the villager), release, lone A pulse.
            phase = self._steps % 24
            if phase < 8:
                return ActionResult(make_action(**{face: True}))
            if 10 <= phase < 12:
                return ActionResult(make_action(a=True))
            return ActionResult(make_action())
        horiz = abs(dx) > abs(dy)
        if self._still >= _STUCK_FRAMES:
            # Boxed in by a villager or scenery: a seeded detour, then resume.
            d = self._rng.choice(_DIRS)
            for _ in range(self._rng.choice((20, 40, 70))):
                self._queue.append(make_action(**{d: True}))
            self._still = 0
            return ActionResult(make_action())
        d = ("right" if dx > 0 else "left") if horiz else ("down" if dy > 0 else "up")
        return ActionResult(make_action(**{d: True}))

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        if int(world.ram[ADDR_TILEMAP]) not in self.maps:
            return TaskResult(status=TaskStatus.SUCCESS, reason="left festival map")
        if self._steps > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="festival escape timeout")
        if self._steps <= self.scripted_budget:
            return TaskResult(
                status=TaskStatus.RUNNING, action=self._scripted_step(world.ram)
            )
        if not self._fallback_logged:
            self._fallback_logged = True
            print(
                f"[FESTIVAL_ESCAPE] scripted visit failed after {self.scripted_budget}f "
                f"({len(self.talks)} talks, {self._rounds} rounds); random roam fallback"
            )
        if not self._queue:
            self._refill()
        return TaskResult(
            status=TaskStatus.RUNNING, action=ActionResult(self._queue.popleft())
        )
