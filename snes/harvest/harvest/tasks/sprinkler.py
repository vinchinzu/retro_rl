"""Sprinkler watering, verified against every covered crop, down to the last dry tile.

One use costs eight stamina and no can water. The sprinkler *replaces* the can: both are one shelf slot
(shed row 3, bit $10 over the can's), so an owner has no can left for sparse leftovers. Asking for it failed
"tool 0x10 missing after shelf" and ended 41 days of the 531 run at 13:00, before the evening field cover.
Stands take the most crops first and then any stand that reaches one.
Works on old blocks as well as plots with a bare centre and side entrance.
"""
from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.carry import selected_tool
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import Tool
from harvest.tasks.crop_geometry import tile_needs_watering
from harvest.tasks.nav import Pathfinder, get_pos_from_ram, get_tile_at, make_action


def dry_tiles(ram, bounds):
    x0, y0, x1, y1 = bounds
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
            if tile_needs_watering(get_tile_at(ram, x, y))}


def sprinkler_stand(ram, dry, start, excluded=(), *, min_crops=1):
    """Most crops per use, then shortest actual path. No walking through crops."""
    candidates = {}
    for x, y in dry:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                stand = x + dx, y + dy
                if stand not in excluded:
                    candidates.setdefault(stand, set()).add((x, y))
    finder = Pathfinder()
    # Prefer clusters before paying for BFS; keep distance deterministic.
    ranked = sorted(candidates, key=lambda s: (-len(candidates[s]), abs(s[0]-start[0])+abs(s[1]-start[1]), s))
    for stand in ranked:
        if len(candidates[stand]) < min_crops:
            break
        path = finder.find_path(ram, start, stand)
        if path is not None:
            return stand, candidates[stand]
    return None


@dataclass
class SprinklerWaterTask(Task):
    name: str = "sprinkler_water"
    bounds: tuple = (2, 3, 62, 60)
    skip_water_tiles: set = field(default_factory=set)
    stop_hour: int | None = None
    timeout: int = 24000
    _child: Task | None = field(default=None, init=False)
    _phase: str = field(default="fetch", init=False)
    _steps: int = field(default=0, init=False)
    _wait: int = field(default=0, init=False)
    _covered: set = field(default_factory=set, init=False)
    _excluded: set = field(default_factory=set, init=False)
    _stand: tuple | None = field(default=None, init=False)
    _watered: int = field(default=0, init=False)
    _initial: set = field(default_factory=set, init=False)

    def reset(self, world):
        from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
        self._steps = self._wait = self._watered = 0
        self._phase = "fetch"
        self._covered, self._excluded = set(), set()
        self._initial = dry_tiles(world.ram, self.bounds) - self.skip_water_tiles
        self._child = EnsureCarryToolTask(tool_id=int(Tool.SPRINKLER), exit_when_done=True)
        self._child.reset(world)

    def can_start(self, world):
        return True

    def progress_snapshot(self):
        return ProgressSnapshot(self.name, self._phase, step_count=self._steps,
                                details=(("watered", self._watered), ("stand", self._stand)),
                                child=task_progress_snapshot(self._child) if self._child else None)

    def _idle(self, reason, **buttons):
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)), reason=reason)

    def step(self, world):
        from harvest.tasks.skills import NavSkill
        self._steps += 1
        ram = world.ram
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"sprinkler timeout watered={self._watered}")
        if self._child:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                if self._phase == "fetch":
                    return result
                self._excluded.add(self._stand)
                self._phase = "select"
            else:
                self._phase = "use" if self._phase == "nav" else "select"
                self._wait = 0
        if self._phase == "use":
            # Wait for the animation to finish, then compare every crop. Never
            # credit the Y press or stamina change as successful watering.
            self._wait += 1
            remaining = {t for t in self._covered if tile_needs_watering(get_tile_at(ram, *t))}
            if not remaining or self._wait > 240:
                self._watered += len(self._covered) - len(remaining)
                if remaining:
                    self._excluded.add(self._stand)
                self._phase = "select"
            elif int(selected_tool(ram)) != int(Tool.SPRINKLER):
                return self._idle("select sprinkler", x=self._wait % 6 == 1)
            else:
                return self._idle("sprinkler verify", y=6 <= self._wait < 26)
        dry = dry_tiles(ram, self.bounds) - self.skip_water_tiles
        if not dry:
            return TaskResult(TaskStatus.SUCCESS, reason=f"sprinkler watered={self._watered} dry_crops=0")
        if self.stop_hour is not None and int(read_ram_value(ram, "hour")) >= self.stop_hour:
            return TaskResult(TaskStatus.SUCCESS, reason=f"sprinkler stop_hour watered={self._watered}")
        pos = get_pos_from_ram(ram)
        choice = sprinkler_stand(ram, dry, (pos.x // 16, pos.y // 16), self._excluded)
        if choice:
            if int(read_ram_value(ram, "stamina")) < 8:
                return TaskResult(TaskStatus.SUCCESS, reason=f"stamina_low watered={self._watered} unfunded={len(dry)}")
            self._stand, self._covered = choice
            self._child = NavSkill(name="nav_sprinkler", target_px=(self._stand[0]*16+8, self._stand[1]*16+8),
                                   radius=5, soft_radius=5, timeout=1200, require_tilemap=0)
            self._phase = "nav"
        else:
            # Every stand left is walled off by crops or already failed. Dry leftovers are one night's growth;
            # a required phase that fails here throws away the rest of the day.
            return TaskResult(TaskStatus.SUCCESS, reason=f"sprinkler watered={self._watered} unreachable={len(dry)}")
        self._child.reset(world)
        return self._idle(self._phase)


def sprinkler_purchase_due(ram):
    """Buy once unlocked; leave the crop reserve and all missing cow tools funded."""
    from harvest.tasks.buy_seeds import tool_owned
    from harvest.planner.livestock_econ import SHOP_PRICE_G, TOOL_PRICE, missing_cow_tools, WALLET_RESERVE_G
    year, season, day = (int(read_ram_value(ram, k, raw=True)) for k in ("year", "season", "day"))
    unlocked = year >= 1 or season >= 2 or (season == 1 and day >= 20)
    reserve = WALLET_RESERVE_G + sum(TOOL_PRICE[t] for t in missing_cow_tools(ram))
    return unlocked and not tool_owned(ram, int(Tool.SPRINKLER)) and int(read_ram_value(ram, "money")) >= SHOP_PRICE_G["sprinkler"] + reserve
