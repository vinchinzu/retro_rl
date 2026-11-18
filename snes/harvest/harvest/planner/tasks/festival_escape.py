"""Leave a festival map the bot did not choose to enter.

Star Night (Winter D24) and New Year (Y2+ Spring D1) pull the farmer onto a
festival map at 6:03. Star Night talks after 18:00 and retains its bounded
roam fallback. New Year follows group 0x28's native $7F1F7A conversation
flags over a controller-probed walking grid, then talks to the exit woman.
New Year has no evening gate: waiting until 18:00 wastes the workday.
An unplanned Flower Festival uses its existing talk-and-dance task: its
clock stays at 07:00 and talking alone cannot end it.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value, read_ram_u16
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.nav import get_pos_from_ram, make_action

# Festival tilemaps (Maps_Graphics.asm): 08 flower, 09 harvest, 0A star night
# square, 0B egg, 3A New Year mountain top.
FESTIVAL_MAPS = frozenset({0x08, 0x09, 0x0A, 0x0B, 0x3A})


def day_plan_owns_festival(task: object) -> bool:
    """Keep scripted festival trips in their day plan; roam after they finish."""
    from harvest.tasks.power_berry import PowerBerryTask

    child = getattr(task, "_current_task", None)
    if isinstance(child, PowerBerryTask) and child.source in {"egg_festival", "harvest_festival"}:
        return True
    if type(child).__name__ in {"FlowerFestivalTask", "HarvestHappinessTask"}:
        return True
    phase = getattr(task, "current_phase", None)
    return getattr(phase, "kind", "") in {"flower_festival", "harvest_happiness"}
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
# Group 0x28 event 1: $7F1F7A bits acknowledge conversations; slot 1
# unlocks after the other eight. New Year has no evening-hour gate.
_NEW_YEAR_TALK_BITS = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2, 5: 5, 6: 6, 7: 7, 8: 8}
# Verified dialogue stands relative to the live actor anchors. Vertical
# approaches collide with several of these sprites before A can reach them.
_NEW_YEAR_STANDS = {
    0: (-13, 3, "right"), 1: (-14, -3, "right"),
    2: (-13, -3, "right"), 3: (-13, -3, "right"),
    4: (-13, 3, "right"), 5: (18, -6, "left"),
    6: (-14, -3, "right"), 7: (-13, 3, "right"), 8: (-13, -3, "right"),
}
# Harvest Festival G26.E0: each girl's first conversation clears her bit
# in $7F1F74. The woman at (88,232) then offers the dance (bit 5),
# and a second conversation with the chosen girl starts it.
_HARVEST_GIRLS = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
_HARVEST_DANCE_HOST = 4


def happiness_cash_floor(ram) -> int:
    from harvest.planner.field_cover import cash_floor_g
    from harvest.planner.social_calendar import date_index

    index = date_index(*(int(read_ram_value(ram, k)) for k in ("year", "season", "day")))
    return max(60_000, cash_floor_g(index))


def harvest_happiness_due(ram) -> bool:
    """The last Harvest Festival before the ending, paid from dated surplus."""
    from harvest.planner import run_strategy

    return (run_strategy.active().court_girls
            and tuple(int(read_ram_value(ram, k)) for k in ("year", "season", "day")) == (1, 2, 12)
            and int(read_ram_value(ram, "hour")) < 15
            and int(read_ram_value(ram, "happiness")) < 999
            and int(read_ram_value(ram, "money")) >= happiness_cash_floor(ram) + 300)


@dataclass
class FestivalEscapeTask(Task):
    name: str = "festival_escape"
    seed: int = 0
    timeout: int = 110_000
    # Frames spent on the scripted visit loop before the random fallback.
    scripted_budget: int = 60_000
    happiness_target: int = 0
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
    _flower: Task | None = field(default=None, init=False)
    _new_year: bool = field(default=False, init=False)
    _grid: object = field(default=None, init=False)
    _walk: deque = field(default_factory=deque, init=False)
    _walk_goal: tuple | None = field(default=None, init=False)
    _closed: set = field(default_factory=set, init=False)
    _walk_point: tuple | None = field(default=None, init=False)
    _walk_best: int = field(default=99999, init=False)
    _walk_stall: int = field(default=0, init=False)

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
        self.drinks = []
        self.happiness_failure = None
        self._drink_push = self._drink_frames = 0
        self._drink_before = None
        self._flower = None
        self._new_year = int(world.ram[ADDR_TILEMAP]) == 0x3A
        self._walk.clear()
        self._walk_goal = None
        self._closed.clear()
        self._walk_point, self._walk_best, self._walk_stall = None, 99999, 0
        if self._new_year:
            from harvest.tasks.flower_festival import WalkGrid
            self._grid = WalkGrid.load(Path(__file__).resolve().parents[2] / "maps" / "new_year_walk.json")
        if int(world.ram[ADDR_TILEMAP]) == 0x08:
            from harvest.tasks.flower_festival import FlowerFestivalTask
            self._flower = FlowerFestivalTask(timeout=self.timeout)
            self._flower.reset(world)

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
        if self._new_year:
            if 0 not in self._visited:
                return 0 if 0 in npcs else None
            remaining = [s for s in left if s != 1]
            if not remaining:
                return 1 if 1 in npcs else None
            return min(remaining, key=lambda s: (npcs[s][0] - pos[0]) ** 2 + (npcs[s][1] - pos[1]) ** 2)
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

    def _new_year_walk(self, pos, goal):
        """A* over controller-probed hops; close a live collision and replan."""
        from harvest.tasks.flower_festival import HOPS

        if max(abs(pos[i] - goal[i]) for i in (0, 1)) <= 6:
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            direction = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
            return make_action(**{direction: True})
        if goal != self._walk_goal:
            self._walk_goal = goal
            self._walk.clear()
            self._closed.clear()
        self._still = self._still + 1 if pos == self._last_pos else 0
        self._last_pos = pos
        if self._walk:
            point = self._walk[0][0]
            distance = abs(pos[0] - point[0]) + abs(pos[1] - point[1])
            if point != self._walk_point or distance < self._walk_best:
                self._walk_point, self._walk_best, self._walk_stall = point, distance, 0
            else:
                self._walk_stall += 1
        if (self._still >= _STUCK_FRAMES or self._walk_stall >= _STUCK_FRAMES) and self._walk:
            _point, edge = self._walk[0]
            self._closed.add(edge)
            self._walk.clear()
            self._still = 0
            self._walk_point, self._walk_best, self._walk_stall = None, 99999, 0
        while self._walk and pos == self._walk[0][0]:
            self._walk.popleft()
        if not self._walk:
            start, finish = self._grid.nearest(pos), self._grid.nearest(goal)
            if start is None or finish is None:
                return None
            route = self._grid.path(start, finish, frozenset(self._closed))
            if not route:
                return None
            if pos != self._grid.center(start):
                self._walk.append((self._grid.center(start), None))
            cell = start
            for direction in route:
                dx, dy, _bit = HOPS[direction]
                nxt = (cell[0] + dx, cell[1] + dy)
                self._walk.append((self._grid.center(nxt), (cell, direction)))
                cell = nxt
        point, _edge = self._walk[0]
        dx, dy = point[0] - pos[0], point[1] - pos[1]
        if _edge is None and dy:
            return make_action(**{"down" if dy > 0 else "up": True})
        direction = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
        return make_action(**{direction: True})

    def _scripted_step(self, ram) -> ActionResult:
        lock = int(read_ram_value(ram, "input_lock"))
        p = get_pos_from_ram(ram)
        pos = (int(p.x), int(p.y))
        npcs = self._npcs(ram)
        if self._new_year:
            flags = read_ram_u16(ram, 0x11F7A)
            self._visited = {slot for slot, bit in _NEW_YEAR_TALK_BITS.items() if flags & (1 << bit)}
            if self._target in self._visited:
                self._target = None
                self._queue.clear()
        rising = lock == 2 and self._lock == 1
        self._lock = lock
        if rising and npcs:
            near = min(
                npcs,
                key=lambda s: (npcs[s][0] - pos[0]) ** 2 + (npcs[s][1] - pos[1]) ** 2,
            )
            if not self._new_year:
                self._visited.add(near)
            self.talks.append((self._steps, near))
            if self._target == near:
                self._target = None
        clock = (int(read_ram_value(ram, "hour")), int(read_ram_value(ram, "minute")))
        if clock != self._clock:
            self._clock, self._clock_frame = clock, self._steps
        evening = clock[0] >= 18
        if evening and not self._evening_seen and not self._new_year:
            # Only villagers talked to after 18:00 count: start the set afresh.
            self._evening_seen = True
            self._visited.clear()
            self._target = None
        # Star Night keeps its evening wait. New Year uses conversation flags.
        waiting = not self._new_year and not evening and self._steps - self._clock_frame < _FROZEN_FRAMES
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
            if self._target is None:
                return ActionResult(make_action())
            tx0, ty0 = npcs[self._target]
            # Nearest approach spot first.
            self._order = sorted(
                range(len(_SIDES)),
                key=lambda i: (tx0 + _SIDES[i][0] - pos[0]) ** 2
                + (ty0 + _SIDES[i][1] - pos[1]) ** 2,
            )
            if self._new_year and self._target == 0:
                self._order = [2, 1, 0, 3]  # priest's west side is a verified talk stand
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
        ox, oy, face = _NEW_YEAR_STANDS[self._target] if self._new_year else _SIDES[self._order[self._side]]
        gx, gy = tx + ox, ty + oy
        dx, dy = gx - pos[0], gy - pos[1]
        self._still = self._still + 1 if pos == self._last_pos else 0
        self._last_pos = pos
        align = 0 if self._new_year else _ALIGN
        if abs(dx) <= align and abs(dy) <= align:
            if self._new_year:
                self._queue.extend([make_action(), make_action(), make_action(a=True), make_action(a=True)])
                self._queue.extend(make_action() for _ in range(8))
                return ActionResult(make_action(**{face: True}))
            # In place: turn (hold toward the villager), release, lone A pulse.
            phase = self._steps % 24
            if phase < (1 if self._new_year else 8):
                return ActionResult(make_action(**{face: True}))
            if 10 <= phase < 12:
                return ActionResult(make_action(a=True))
            return ActionResult(make_action())
        if self._new_year:
            action = self._new_year_walk(pos, (gx, gy))
            if action is None:
                self._side_start = self._steps - _SIDE_FRAMES - 1
                return ActionResult(make_action())
            return ActionResult(action)
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

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="new_year_talk" if self._new_year else "festival_talk",
            step_count=self._steps,
            details=(("talked", tuple(sorted(self._visited))), ("target", self._target), ("side", self._side)),
        )

    def _happiness_step(self, ram) -> ActionResult | None:
        """Buy native +3 dumplings after the girls, before the dance starts.

        The counter needs a blocked UP step to refresh contact, like a crop.
        Each purchase must cost 300G and add three (clamped at 999).
        """
        from harvest.planner.tasks.inventory_exit import room_walk_action

        if not self.happiness_target or self.happiness_failure:
            return None
        hearts, money = (int(read_ram_value(ram, k)) for k in ("happiness", "money"))
        ready = (int(read_ram_value(ram, "input_lock")) == 1
                 and not int(read_ram_value(ram, "game_state")) & 0x40
                 and int(read_ram_value(ram, "player_action")) == 0)
        if self._drink_before is None:
            self._drink_frames += 1
        if self._drink_before is not None:
            happy0, money0 = self._drink_before
            if hearts != happy0 and ready:
                if hearts != min(999, happy0 + 3) or money0 - money != 300:
                    self.happiness_failure = f"unverified dumpling {happy0}->{hearts}, debit {money0-money}"
                    return None
                self.drinks.append({"happiness_before": happy0, "happiness_after": hearts, "spent": 300})
                self._drink_before = None
                self._drink_frames = 0
            elif self._drink_frames > 1800:
                self.happiness_failure = "dumpling purchase timed out"
                return None
        if self._drink_before is None and ready:
            if hearts >= min(999, self.happiness_target) or money < happiness_cash_floor(ram) + 300:
                return None
            pos = get_pos_from_ram(ram)
            if not self._drink_push and max(abs(pos.x - 344), abs(pos.y - 120)) > 1:
                if self._drink_frames > 8_000:
                    self.happiness_failure = "dumpling counter approach timed out"
                    return None
                return ActionResult(room_walk_action(ram, (344, 120)))
            if self._drink_push < 24:
                self._drink_push += 1
                return ActionResult(make_action(up=True))
            self._drink_before = (hearts, money)
            self._drink_frames = 0
        return ActionResult(make_action(a=self._drink_before is not None and self._steps % 24 < 2))

    def _harvest_step(self, ram) -> ActionResult:
        from harvest.tasks.relationships import stand_walk_action
        from harvest.planner.tasks.inventory_exit import room_walk_action

        lock = int(read_ram_value(ram, "input_lock"))
        if lock != 1:
            self._cooldown = 24
            return ActionResult(make_action(a=self._steps % 24 < 2))
        if self._cooldown:
            self._cooldown -= 1
            return ActionResult(make_action())
        npcs = self._npcs(ram)
        flags = read_ram_u16(ram, 0x11F74)
        remaining = [slot for slot, bit in _HARVEST_GIRLS.items() if flags & (1 << bit) and slot in npcs]
        pos = get_pos_from_ram(ram)
        if remaining:
            target = self._target if self._target in remaining else min(
                remaining, key=lambda s: abs(npcs[s][0] - pos.x) + abs(npcs[s][1] - pos.y))
        else:
            drink = self._happiness_step(ram)
            if drink is not None:
                return drink
            # The mayor first walks north out of the host's narrow entrance.
            target = 9 if flags & 0x20 else (1 if npcs.get(1, (0, 0))[1] > 220 else _HARVEST_DANCE_HOST)
        if target not in npcs:
            return ActionResult(make_action())
        if self._target != target:
            self._target = target
            self._queue.clear()
        if self._queue:
            return ActionResult(self._queue.popleft())
        x, y = npcs[target]
        # Ann stands in the north fence opening; its west neighbour is solid.
        if target == 6:
            stand, face = (x, y + 27), "up"
        elif target == 1:
            stand, face = (x + 23, y + 3), "left"
        elif target == _HARVEST_DANCE_HOST:
            # The host sits in a one-tile fence notch. Talk diagonally from
            # the open crossing below, rather than BFS into the fence.
            stand, face = (x + 11, y + 16), "up"
        else:
            stand, face = (x - 13, y + 3), "right"
        align = 0 if target == _HARVEST_DANCE_HOST else 2
        if abs(pos.x - stand[0]) <= align and abs(pos.y - stand[1]) <= align:
            self._queue.extend([make_action(), make_action(a=True), make_action(a=True)])
            self._queue.extend(make_action() for _ in range(12))
            return ActionResult(make_action(**{face: True}))
        if target == _HARVEST_DANCE_HOST:
            return ActionResult(room_walk_action(ram, stand))
        return ActionResult(stand_walk_action(ram, stand, face))

    def step(self, world: WorldState) -> TaskResult:
        if self._flower is not None:
            return self._flower.step(world)
        self._steps += 1
        if self._drink_before is not None:
            self._drink_frames += 1
            if self._drink_frames > 1800:
                self.happiness_failure = f"dumpling order timed out after {self._drink_frames} frames at {get_pos_from_ram(world.ram)}"
                self._drink_before = None
        if int(world.ram[ADDR_TILEMAP]) not in self.maps:
            return TaskResult(status=TaskStatus.SUCCESS, reason="left festival map")
        if self._steps > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="festival escape timeout")
        if int(world.ram[ADDR_TILEMAP]) == 0x09:
            return TaskResult(TaskStatus.RUNNING, action=self._harvest_step(world.ram))
        if self._steps <= self.scripted_budget:
            return TaskResult(
                status=TaskStatus.RUNNING, action=self._scripted_step(world.ram)
            )
        if self._new_year:
            return TaskResult(status=TaskStatus.BLOCKED, reason="New Year conversation route exhausted")
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


@dataclass
class HarvestHappinessTask(Task):
    """Existing festival skills composed into the final Fall happiness trip."""

    name: str = "harvest_happiness"
    timeout: int = 660_000  # at most 333 orders, each bounded at 1,800 frames

    def reset(self, world):
        from harvest.planner.tasks.inventory import ExitToFarmTask

        self._steps = self._stage = 0
        self._before = int(read_ram_value(world.ram, "happiness"))
        self._day = int(read_ram_value(world.ram, "day"))
        self._child = ExitToFarmTask()
        self._child.reset(world)
        self.escape = None
        if int(read_ram_value(world.ram, "tilemap")) == 0x09:
            self.escape = FestivalEscapeTask(happiness_target=999, timeout=self.timeout)
            self._child, self._stage = self.escape, 2
            self._child.reset(world)

    def can_start(self, world):
        return harvest_happiness_due(world.ram)

    def step(self, world):
        self._steps += 1
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason="Harvest happiness trip timed out")
        result = self._child.step(world)
        if result.status != TaskStatus.SUCCESS:
            return result
        from harvest.tasks.power_berry import FestivalEntryTask

        if self._stage == 0:
            self._child = FestivalEntryTask()
        elif self._stage == 1:
            self.escape = FestivalEscapeTask(happiness_target=999, timeout=self.timeout)
            self._child = self.escape
        else:
            if int(read_ram_value(world.ram, "day")) == self._day:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(
                    a=int(read_ram_value(world.ram, "input_lock")) != 1 and self._steps % 24 < 2)),
                    reason="wait for native festival overnight")
            happiness = int(read_ram_value(world.ram, "happiness"))
            remaining = max(0, 999 - happiness)
            meta = {"drinks": self.escape.drinks, "happiness_failure": self.escape.happiness_failure,
                    "target_reached": remaining == 0, "happiness_remaining": remaining}
            return TaskResult(TaskStatus.SUCCESS, reason=f"happiness {self._before}->{happiness}; dumplings={len(self.escape.drinks)} remaining={remaining}", meta=meta)
        self._stage += 1
        self._child.reset(world)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))


# ── Star Night: pick the wife's scene ───────────────────────────────────────
# Winter D24 06:00, in the house: text 0x02B6 "Where should I go?" (group 0x0F event 0). One A turns the page, then
# the menu is live ~200 frames later and Down/Up step $018F (Left/Right do nothing); choice n plays scene n+1, and
# each scene takes 30 from the wife unless she is its girl. Measured on PR_e2ec1_Y1_Winter_D24 (Eve the wife, 430):
# the default index 0 is the square, Ellen +30 and Eve -30; index 2 transits to the bar (0x1E) at 18:03, Eve kept.
STAR_NIGHT_TEXT = 0x02B6
STAR_NIGHT_INDEX = {"ellen": 0, "maria": 1, "eve": 2, "nina": 3, "ann": 4}
STAR_NIGHT_CURSOR = 0x018F
# Only Eve's scene has measured stands; another wife keeps the old default.
STAR_NIGHT_SCENE_MAPS = {"eve": 0x1E}
# Eve's scene (group 0x0F event 3, $B48402): six villagers each signal once when talked to, then Eve's line (after
# all six) adds +30 and ends the scene. The farmer lands at (144,440). Seated actors seal the room BFS (+-12 px
# boxes close column x=9, the only way north), so the walk is a fixed path through the open corridors of the bar
# grid (column 9, row 21, column 4); a point with a facing is a talk stop, the press reaching 16 px ahead +-12.
STAR_NIGHT_EVE_PATH = (
    ((144, 436), "down"),                       # door villager (144,456); text 0x03D9
    ((152, 436), None), ((152, 417), "right"),  # (180,416) on the south stool
    ((152, 344), None), ((180, 344), "down"),   # (180,368)
    ((136, 344), "up"),                         # counter (136,328)
    ((72, 344), None), ((72, 368), None), ((80, 368), "right"),  # (100,368): not from above (no box at (100,344))
    ((72, 344), None), ((40, 344), None), ((40, 384), "down"),   # (40,408)
    ((40, 344), None), ((72, 344), None), ((72, 446), None), ((78, 446), "left"),  # Eve (56,448): the bar stand
)
# Eve's scene actor wanders (op22), so her stop follows her live sprite: her right side, facing left.
STAR_NIGHT_EVE_ANCHOR = (56, 448)
STAR_NIGHT_EVE_OFFSET = (22, -2)
# From Eve's stand back to the door without crossing the door villager's row (a straight walk sticks at (131,452)).
STAR_NIGHT_EVE_RETURN = (((72, 446), None), ((72, 344), None), ((152, 344), None), ((152, 436), None))


def _star_night_wife(ram):
    from harvest.planner.social_calendar import enabled, wife_of

    if not enabled():
        return None
    wife = wife_of(ram)
    return wife if wife in STAR_NIGHT_SCENE_MAPS else None


def star_night_menu_press(ram, state: dict, step: int) -> TaskResult | None:
    """Drive the Star Night menu to the wife's scene, or ``None`` to leave the default handling."""
    from harvest.core.ram_catalog import read_ram_u8

    if (int(read_ram_value(ram, "dialog_text_id")) != STAR_NIGHT_TEXT or int(read_ram_value(ram, "input_lock")) != 2
            or (int(read_ram_value(ram, "season")), int(read_ram_value(ram, "day"))) != (3, 24)):
        return None
    wife = _star_night_wife(ram)
    year = int(read_ram_value(ram, "year"))
    if state.get("year") != year:
        state.clear()
        state["year"] = year
    if wife is None or state.get("gave_up"):
        return None
    target = STAR_NIGHT_INDEX[wife]
    idle = TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="star night menu")
    seen = state.setdefault("seen", step)
    if "advanced" not in state:
        if step - seen < 300:
            return idle
        state["advanced"] = step
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=True)), reason="star night page")
    if step - state["advanced"] < 200:
        return idle
    cursor = int(read_ram_u8(ram, STAR_NIGHT_CURSOR, live_offset=False))
    if cursor != target:
        if state.get("moves", 0) > 12:
            state["gave_up"] = True
            print(f"[STAR_NIGHT] cursor stuck at {cursor}, leaving the default choice", flush=True)
            return None
        if step % 16:
            return idle
        state["moves"] = state.get("moves", 0) + 1
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(down=cursor < target, up=cursor > target)),
                          reason=f"star night cursor {cursor}->{target}")
    if step % 12:
        return idle
    print(f"[STAR_NIGHT] choosing index {target} ({wife})", flush=True)
    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=True)), reason=f"star night {wife}")


def star_night_scene_map(ram) -> bool:
    """The farmer is in the wife's Star Night scene (not a festival map, so the planner would not hand it over)."""
    wife = _star_night_wife(ram)
    return bool(wife and int(ram[ADDR_TILEMAP]) == STAR_NIGHT_SCENE_MAPS[wife]
                and (int(read_ram_value(ram, "season")), int(read_ram_value(ram, "day"))) == (3, 24)
                and int(read_ram_value(ram, "hour")) >= 18)


@dataclass
class StarNightSceneTask(Task):
    """Talk to every actor of the wife's Star Night scene from its stand, Eve last; the ROM then ends the scene."""
    name: str = "star_night_scene"
    timeout: int = 40_000
    talks: list = field(default_factory=list, init=False)
    misses: list = field(default_factory=list, init=False)

    def reset(self, world: WorldState) -> None:
        self._map = int(world.ram[ADDR_TILEMAP])
        self.misses = []
        self._queue = list(STAR_NIGHT_EVE_PATH)
        self._steps = self._rounds = self._ready = 0
        self._stand = None
        self.talks = []

    def can_start(self, world: WorldState) -> bool:
        return star_night_scene_map(world.ram)

    def _talk(self, ram) -> TaskResult | None:
        """One path point: walk there; at a talk stop face, A until a box opens, wait it closed. ``None`` when done."""
        (x, y), face = self._stand["at"]
        if (x, y) == STAR_NIGHT_EVE_PATH[-1][0]:
            ax, ay = STAR_NIGHT_EVE_ANCHOR
            eve = min((o.pixel for o in game_objects(ram) if o.is_npc_candidate),
                      key=lambda p: abs(p[0] - ax) + abs(p[1] - ay), default=None)
            if eve is not None and abs(eve[0] - ax) + abs(eve[1] - ay) <= 48:
                x, y = eve[0] + STAR_NIGHT_EVE_OFFSET[0], eve[1] + STAR_NIGHT_EVE_OFFSET[1]
        lock = int(read_ram_value(ram, "input_lock"))
        stage = self._stand["stage"]
        self._stand["frames"] += 1
        if self._stand["frames"] > 2_400:
            pos = get_pos_from_ram(ram)
            self.misses.append(((x, y), (pos.x, pos.y), stage))
            return None
        if stage == "walk":
            pos = get_pos_from_ram(ram)
            # A seated villager's body stops the farmer short of the anchor offset (153 for a stand at 167); the
            # press reaches 16 px ahead +-12, so a blocked step within 20 px of the stand is arrival.
            still = self._stand.get("last") == (pos.x, pos.y)
            self._stand["still"] = self._stand.get("still", 0) + 1 if still else 0
            self._stand["last"] = (pos.x, pos.y)
            dx, dy = x - pos.x, y - pos.y
            near = max(abs(dx), abs(dy))
            if near <= 2 or (face and near <= 20 and self._stand["still"] >= 20):
                if face is None:
                    return None
                self._stand.update(stage="face", n=0)
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            move = (("right" if dx > 0 else "left") if abs(dx) > 2 else ("down" if dy > 0 else "up"))
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{move: True})))
        if stage == "face":
            self._stand["n"] += 1
            if self._stand["n"] <= 6:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{face: True})))
            self._stand.update(stage="press", n=0)
        if stage in ("face", "press"):
            if lock != 1:
                self._stand.update(stage="read", n=0)
            else:
                self._stand["n"] += 1
                if self._stand["n"] % 60 == 59:
                    self._stand.update(stage="walk")   # she may have moved: re-approach and face again
                if self._stand["n"] > 600:
                    self.misses.append(((x, y), "no box", stage))
                    return None
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=self._stand["n"] % 12 < 2)))
        self._stand["n"] = 0 if lock != 1 else self._stand["n"] + 1
        if self._stand["n"] >= 8:
            self.talks.append(((x, y), self._steps))
            return None
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=lock != 1 and self._steps % 12 < 2)))

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        ram = world.ram
        if int(ram[ADDR_TILEMAP]) != self._map:
            return TaskResult(TaskStatus.SUCCESS, reason=f"star night scene ended talks={len(self.talks)}")
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"star night scene timeout talks={len(self.talks)}")
        if self._stand is not None:
            result = self._talk(ram)
            if result is not None:
                return result
            self._stand = None
        if int(read_ram_value(ram, "input_lock")) != 1:
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=self._steps % 12 < 2)))
        from harvest.planner.tasks.transitions import screen_faded_in

        # The pad is dead until the arrival fade-in ends; a talk pressed then opens no box.
        self._ready = self._ready + 1 if screen_faded_in(ram) else 0
        if self._ready < 30 and not self.talks:
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
        if not self._queue:
            # A stand that missed its actor leaves Eve on "not yet": go round again, twice at most.
            self._rounds += 1
            if self._rounds > 2:
                return TaskResult(TaskStatus.FAILURE, reason=f"star night scene unfinished talks={len(self.talks)}")
            self._queue = list(STAR_NIGHT_EVE_RETURN) + list(STAR_NIGHT_EVE_PATH)
        self._stand = {"at": self._queue.pop(0), "stage": "walk", "frames": 0, "n": 0}
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
