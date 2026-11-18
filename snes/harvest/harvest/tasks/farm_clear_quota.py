"""D2 leftover clear quotas — RAM-count stop, not whole-farm wipe.

``FarmClearTask(handoff="quota", quota=...)`` succeeds when the clearer has
removed at least the requested counts. Small rocks are tile ``0x06``;
large boulders are one 2×2 (TL ``0x0D``/damage). Do not count four cells
of one boulder as four rocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set

from harvest.core.tile_catalog import (
    FENCE,
    is_path_tilemap,
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TILES,
    ROCK as SMALL_ROCK_TILE,
    STALE_TILE_IDS,
    STONE,
    STUMP_TILES,
    WEED,
    DebrisType,
)
from harvest.tasks.farm_ops import TileScanner


@dataclass(frozen=True)
class ClearQuota:
    weeds: int = 0
    stones: int = 0
    small_rocks: int = 0
    large_rocks: int = 0
    stumps: int = 0
    fences: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ClearQuota":
        if not data:
            return cls()
        return cls(
            weeds=int(data.get("weeds", 0) or 0),
            stones=int(data.get("stones", 0) or 0),
            small_rocks=int(data.get("small_rocks", 0) or 0),
            large_rocks=int(data.get("large_rocks", 0) or 0),
            stumps=int(data.get("stumps", 0) or 0),
            fences=int(data.get("fences", 0) or 0),
        )

    def is_empty(self) -> bool:
        return not any(
            (
                self.weeds,
                self.stones,
                self.small_rocks,
                self.large_rocks,
                self.stumps,
                self.fences,
            )
        )


@dataclass(frozen=True)
class DebrisCounts:
    weeds: int = 0
    stones: int = 0
    small_rocks: int = 0
    large_rocks: int = 0
    stumps: int = 0
    fences: int = 0

    def as_dict(self) -> dict:
        return {
            "weeds": self.weeds,
            "stones": self.stones,
            "small_rocks": self.small_rocks,
            "large_rocks": self.large_rocks,
            "stumps": self.stumps,
            "fences": self.fences,
        }

    def cleared_since(self, now: "DebrisCounts") -> "DebrisCounts":
        return DebrisCounts(
            weeds=self.weeds - now.weeds,
            stones=self.stones - now.stones,
            small_rocks=self.small_rocks - now.small_rocks,
            large_rocks=self.large_rocks - now.large_rocks,
            stumps=self.stumps - now.stumps,
            fences=self.fences - now.fences,
        )

    def meets(self, quota: ClearQuota) -> bool:
        return (
            self.weeds >= quota.weeds
            and self.stones >= quota.stones
            and self.small_rocks >= quota.small_rocks
            and self.large_rocks >= quota.large_rocks
            and self.stumps >= quota.stumps
            and self.fences >= quota.fences
        )


def classify_target(tile_id: int, debris_type: DebrisType) -> str:
    if debris_type == DebrisType.WEED or tile_id == WEED:
        return "weeds"
    if debris_type == DebrisType.STONE or tile_id == STONE:
        return "stones"
    if debris_type == DebrisType.FENCE or tile_id == FENCE:
        return "fences"
    if debris_type == DebrisType.STUMP or tile_id in STUMP_TILES:
        return "stumps"
    if tile_id == SMALL_ROCK_TILE:
        return "small_rocks"
    if tile_id in LARGE_ROCK_TILES or tile_id in LARGE_ROCK_DAMAGE_TILES:
        return "large_rocks"
    if debris_type == DebrisType.ROCK:
        return "large_rocks"
    return "other"


def count_debris(ram, bounds=None, *, types=None) -> DebrisCounts:
    targets = TileScanner().scan(ram, bounds, types=types)
    tallies = {
        "weeds": 0,
        "stones": 0,
        "small_rocks": 0,
        "large_rocks": 0,
        "stumps": 0,
        "fences": 0,
    }
    for target in targets:
        key = classify_target(int(target.tile_id), target.debris_type)
        if key in tallies:
            tallies[key] += 1
    return DebrisCounts(**tallies)


def capped_quota(want: ClearQuota, start: DebrisCounts) -> ClearQuota:
    """Do not demand more than the pin actually spawned.

    D2 ``Y1_After_Buy_Potato`` has 0× ``0x06`` small boulders. A leftover
    quota of 10 small rocks (pond-tossed ``0x04``) must not fail the
    4-boulder hammer pass.
    """
    return ClearQuota(
        weeds=min(want.weeds, start.weeds),
        stones=min(want.stones, start.stones),
        small_rocks=min(want.small_rocks, start.small_rocks),
        large_rocks=min(want.large_rocks, start.large_rocks),
        stumps=min(want.stumps, start.stumps),
        fences=min(want.fences, start.fences),
    )


def quota_counts_met(
    start: DebrisCounts, now: DebrisCounts, want: ClearQuota
) -> bool:
    if want.is_empty():
        return False
    effective = capped_quota(want, start)
    if effective.is_empty():
        return True
    return start.cleared_since(now).meets(effective)


def unmet_debris_types(
    start: DebrisCounts | None,
    now: DebrisCounts,
    quota: Mapping[str, Any] | ClearQuota | None,
) -> Optional[Set[DebrisType]]:
    """Debris kinds still below the (capped) quota, or None if no quota."""
    want = quota if isinstance(quota, ClearQuota) else ClearQuota.from_mapping(quota)
    if want.is_empty() or not isinstance(start, DebrisCounts):
        return None
    effective = capped_quota(want, start)
    if effective.is_empty():
        return set()
    cleared = start.cleared_since(now)
    unmet: Set[DebrisType] = set()
    if cleared.weeds < effective.weeds:
        unmet.add(DebrisType.WEED)
    if cleared.stones < effective.stones:
        unmet.add(DebrisType.STONE)
    if (
        cleared.small_rocks < effective.small_rocks
        or cleared.large_rocks < effective.large_rocks
    ):
        unmet.add(DebrisType.ROCK)
    if cleared.stumps < effective.stumps:
        unmet.add(DebrisType.STUMP)
    if cleared.fences < effective.fences:
        unmet.add(DebrisType.FENCE)
    return unmet


# The whole-field cover (score family from Year 2) sows grass across the field, not only the pasture.
COVER_FIELD_BOUNDS = (2, 3, 62, 60)


def _grass_bounds() -> tuple:
    """Where grass stages (0x72/0x75/0x76) are legitimate tiles: the pasture, or the whole field under the cover."""
    from harvest.planner import run_strategy
    from harvest.tasks.grass_planter import PASTURE_BOUNDS

    return COVER_FIELD_BOUNDS if run_strategy.field_cover() else PASTURE_BOUNDS


def _in_pasture(tile) -> bool:
    x0, y0, x1, y1 = _grass_bounds()
    return x0 <= tile[0] <= x1 and y0 <= tile[1] <= y1


def stale_under_player(ram) -> bool:
    """The farmer's tile reads a stale viewport id; pasture grass shares those ids."""
    from harvest.tasks.nav import get_pos_from_ram, get_tile_at, TILE_SIZE

    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    return not _in_pasture(tile) and int(get_tile_at(ram, *tile)) in STALE_TILE_IDS


def farm_map_loaded(ram) -> bool:
    """False off the farm, on shed-door 0xFF, or viewport unload.

    Shed ``0x26`` has a real metatile map, so an FF-stale check alone treats
    it as a wiped farm and CLEAR_ROCKS no-ops. Standing on a8 next to the
    door still unloads distant farm metatiles to 0xFF.

    0x72/0x75/0x76 are also pasture grass stages, so tiles inside
    ``PASTURE_BOUNDS`` are not counted: a Year 2 Summer pasture held 57 tiles of
    0x76 and the stump clearer walked north-west into the barn for good. Under
    the whole-field cover the grass bounds are the whole field (``COVER_FIELD_BOUNDS``):
    389 young-grass tiles read as an unloaded map all of Y2 Spring D8 and wedged
    ``ENSURE_ANIMAL_TOOLS`` for 2M frames. ``0xFF`` still counts everywhere.
    """
    from harvest.core.tile_catalog import ADDR_MAP, ADDR_TILEMAP, MAP_WIDTH

    if ADDR_TILEMAP < len(ram) and int(ram[ADDR_TILEMAP]) != 0x00:
        return False
    if stale_under_player(ram):
        return False
    x0, y0, x1, y1 = _grass_bounds()
    end = min(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, len(ram))
    if end <= ADDR_MAP:
        return False
    import numpy as np

    chunk = np.zeros(MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
    chunk[: end - ADDR_MAP] = np.asarray(ram[ADDR_MAP:end], dtype=np.uint8)
    grid = chunk.reshape(MAP_WIDTH, MAP_WIDTH).copy()
    inside_unloaded = int((grid[y0 : y1 + 1, x0 : x1 + 1] == 0xFF).sum()) if (x0, y0, x1, y1) == COVER_FIELD_BOUNDS else 0
    grid[y0 : y1 + 1, x0 : x1 + 1] = 0
    stale = int(np.isin(grid, list(STALE_TILE_IDS)).sum()) + inside_unloaded   # 0xFF is never grass
    return stale < 64


def needs_shed_door_step_off(ram) -> bool:
    """True on the shed warp or while the farm map is still unloaded.

    Adjacent a8 is walkable but still unloads distant metatiles — keep
    walking toward the open yard until :func:`farm_map_loaded`.
    """
    from harvest.tasks.farm_ops import SHED_DOOR_TILE
    from harvest.tasks.nav import get_pos_from_ram, TILE_SIZE

    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    if tile == SHED_DOOR_TILE:
        return True
    if stale_under_player(ram):
        return True
    return not farm_map_loaded(ram)


YARD_LOAD_TILE = (25, 32)


def yard_load_action(ram):
    """Run toward open ground south of the shed until the farm map loads.

    Shed-door stale is west of that tile; west-gate FF (spa return) is east.
    Choose the direction from the live position at either entrance.
    """
    from harvest.tasks.nav import TILE_SIZE, get_pos_from_ram, make_action

    pos = get_pos_from_ram(ram)
    # The shed has a solid facade beside its door. Leave south before aiming
    # west; an upward correction at the threshold enters the shed again.
    if pos.x // TILE_SIZE == 26 and pos.y // TILE_SIZE == 30:
        return make_action(down=True, b=True)
    tx, ty = YARD_LOAD_TILE
    dx = tx * TILE_SIZE + 8 - pos.x
    dy = ty * TILE_SIZE + 8 - pos.y
    if abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"
    return make_action(**{direction: True, "b": True})


def quota_satisfied(
    ram,
    quota: Mapping[str, Any] | ClearQuota | None,
    *,
    clearer: Optional[Any] = None,
    bounds=None,
) -> bool:
    """True when this pass has cleared at least the requested counts.

    Honest path: ``FarmClearTask.reset`` snapshots ``quota_start_counts``,
    then start-minus-now via ``count_debris`` (one target per 2×2 TL).
    Requested counts cap at what the pin spawned. A capped-empty quota
    (pin spawned none of the requested debris) is a no-op when the farm
    map is loaded, even if ``cleared_count`` is 0. Viewport-unload zeros
    stay False. ``cleared_by_kind`` is only a fallback if no snapshot
    exists.
    """
    want = (
        quota
        if isinstance(quota, ClearQuota)
        else ClearQuota.from_mapping(quota)
    )
    if want.is_empty() or clearer is None:
        return False
    if not farm_map_loaded(ram):
        return False
    start = getattr(clearer, "quota_start_counts", None)
    if isinstance(start, DebrisCounts):
        scan_bounds = bounds
        if scan_bounds is None:
            scan_bounds = getattr(clearer, "farm_bounds", None)
        if not quota_counts_met(start, count_debris(ram, scan_bounds), want):
            return False
        if int(getattr(clearer, "cleared_count", 0) or 0) > 0:
            return True
        # Non-empty effective quota still needs a swing so shed-door unload
        # cannot fake a wipe. Capped-empty is an honest no-op.
        return capped_quota(want, start).is_empty()
    got = getattr(clearer, "cleared_by_kind", None)
    if isinstance(got, Mapping):
        return DebrisCounts(
            weeds=int(got.get("weeds", 0) or 0),
            stones=int(got.get("stones", 0) or 0),
            small_rocks=int(got.get("small_rocks", 0) or 0),
            large_rocks=int(got.get("large_rocks", 0) or 0),
            stumps=int(got.get("stumps", 0) or 0),
            fences=int(got.get("fences", 0) or 0),
        ).meets(want)
    return False


__all__ = [
    "ClearQuota",
    "DebrisCounts",
    "capped_quota",
    "classify_target",
    "count_debris",
    "farm_map_loaded",
    "needs_shed_door_step_off",
    "yard_load_action",
    "quota_counts_met",
    "quota_satisfied",
    "unmet_debris_types",
]

# FarmClearer handoff (pocket, drop, quota/type_clear stop). Installed as
# methods on FarmClearer so that module stays the one clear Task.

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    CLEARABLE_DEBRIS_TYPES,
    STALE_TILE_IDS,
    TILE_SIZE,
    TILE_TO_DEBRIS,
)
from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
from harvest.maps.map_config import FARM_POND_ACCESS_FENCE_ROW
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.farm_toss import (
    FENCE_WALL_Y,
    FenceJumpTossSkill,
    needs_south_fence_drop,
    open_toss_face,
)
from harvest.tasks.nav import Point, get_pos_from_ram, get_tile_at, make_action

_FENCE_WALL_PX_Y = FARM_POND_ACCESS_FENCE_ROW * 16
_EAST_STAGING_X = 480


def progress_text(self) -> str:
    phase = self.current_phase
    phase_name = phase.name if phase else self.state
    return f"{phase_name} cleared={self.cleared_count} failed={len(self.failed_tiles)}"

def progress_snapshot(self) -> ProgressSnapshot:
    target = self.current_target
    approach = self.approach_tile
    approach_position = (
        (approach[0] * TILE_SIZE + TILE_SIZE // 2, approach[1] * TILE_SIZE + TILE_SIZE // 2)
        if approach is not None else None
    )
    phase = self.current_phase
    details = (
        ("cleared", self.cleared_count),
        ("failed", len(self.failed_tiles)),
        ("state", self.state),
        ("clearing_phase", phase.name if phase else None),
        ("target", target.tile if target is not None else None),
        ("hits", int(self.target_hits)),
        ("approach", approach),
        ("approach_position", approach_position),
        ("stamina_exhausted", self.stamina_exhausted),
    )
    return ProgressSnapshot(
        task_name=self.name, phase_text=self.state, step_count=self._step_count, details=details
    )

def _scan_bounds(self):
    if self.handoff != "quota" and self.farm_bounds is not None and self._pocket_arrived:
        return self._plot_scan_bounds()
    return self.farm_bounds or self._locked_bounds or self._work_bounds

def _remaining_debris(self, ram) -> list:
    scan_types = set(CLEARABLE_DEBRIS_TYPES)
    if self.handoff == "type_clear" and self._priority_spec:
        scan_types = set(self._priority_spec)
    elif self.tools_missing and self.priority:
        # Rock/stump without hammer/axe is not work this clear can finish;
        # counting it made every Fall CLEAR_FIELD "partial_clear remaining=1".
        scan_types &= set(self.priority)
    return TileScanner().scan(ram, self._scan_bounds(), types=scan_types)

def _player_tile(self, ram) -> Tile:
    pos = get_pos_from_ram(ram)
    return (pos.x // TILE_SIZE, pos.y // TILE_SIZE)

def _in_pocket(self, ram) -> bool:
    bounds = self.farm_bounds
    if bounds is None:
        return True
    tx, ty = self._player_tile(ram)
    if not (bounds[0] <= tx <= bounds[2] and bounds[1] <= ty <= bounds[3]):
        return False
    # West-fence (3,28) is inside the box but is not the plant stand.
    cx, cy = WEST_POCKET_PLANT_CENTER
    return abs(tx - cx) <= 3 and abs(ty - cy) <= 2

def _pocket_tiles_ready(self, ram) -> bool:
    """False while the plant-notch 5x5 is still stale 0x72 (gate viewport)."""
    cx, cy = WEST_POCKET_PLANT_CENTER
    stale = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if int(get_tile_at(ram, cx + dx, cy + dy)) in STALE_TILE_IDS:
                stale += 1
    return stale < 8

def _pocket_is_ready(self, world: WorldState) -> bool:
    from harvest.planner.day_plan_status import is_farm_tilemap
    ram = world.ram
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    return is_farm_tilemap(tilemap) and self._in_pocket(ram) and self._pocket_tiles_ready(ram)

def _plot_cells_to_clear(self) -> set:
    """3x3 ring + notch + HOE_PLAN stands (2 tiles out)."""
    from harvest.tasks.crop_geometry import hoe_plan, plot_tiles
    cx, cy = WEST_POCKET_PLANT_CENTER
    cells = set(plot_tiles((cx, cy), include_center=True))
    cells.add((cx, cy))
    for target, stand, _face in hoe_plan((cx, cy)):
        cells.add(target)
        cells.add(stand)
    return cells

def _plot_scan_bounds(self) -> Tuple[int, int, int, int]:
    cells = self._plot_cells_to_clear()
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (min(xs), min(ys), max(xs), max(ys))

def _lock_clearer_to_plot(self) -> None:
    """Stop roaming the pocket. Quota/type_clear keep the requested box."""
    if self.handoff in ("quota", "type_clear"):
        return
    bounds = self._plot_scan_bounds()
    self._locked_bounds = bounds
    self._work_bounds = None
    print(f"[CLEAR] Plot scan bounds {bounds}")

def _plant_notch_is_clear(self, ram) -> bool:
    from harvest.planner.tasks.transitions import hands_are_clear
    if self.farm_bounds is None or not self._pocket_arrived or not hands_are_clear(ram):
        return False
    for tx, ty in self._plot_cells_to_clear():
        tile_id = int(get_tile_at(ram, tx, ty))
        if tile_id in STALE_TILE_IDS or tile_id in TILE_TO_DEBRIS:
            return False
    return True

def _pocket_stand_px(self) -> Point:
    cx, cy = WEST_POCKET_PLANT_CENTER
    return Point((cx - 1) * TILE_SIZE + 8, cy * TILE_SIZE + 8)

def _make_pocket_approach(self, world: WorldState) -> Task:
    from harvest.maps.map_config import (
        ROUTES,
        SEGMENTS,
        mountain_downhill_escape,
        mountain_exit_then_farm,
        path_return_to_farm,
        slice_route_from_position,
    )
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask
    from harvest.planner.tasks.multi_nav import MultiMapNavTask
    from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
    from harvest.planner.tasks.navigation import NavTask
    ram = world.ram
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap == 0x26 or is_house_tilemap(tilemap):
        return ExitToFarmTask(tasks_dir=self.tasks_dir)
    if is_farm_tilemap(tilemap):
        return NavTask(
            name="nav_clear_plot_pocket",
            target_px=self._pocket_stand_px(),
            radius=14,
            timeout=3500,
        )
    pos = get_pos_from_ram(ram)
    if tilemap == 0x10:
        if int(pos.y) >= 380:
            hops = mountain_exit_then_farm(
                mountain_downhill_escape(int(pos.x), int(pos.y), tilemap=tilemap)
            )
        else:
            hops = list(ROUTES.get("mountain_to_farm", []))
    elif is_path_tilemap(tilemap):
        hops = path_return_to_farm(int(pos.x), int(pos.y), tilemap=tilemap)
    else:
        hops = list(SEGMENTS.get("path_to_farm", []))
        if tilemap == 0x04:
            hops = list(SEGMENTS.get("town_shop_to_path", [])) + hops
        elif tilemap == 0x1C:
            hops = (
                list(SEGMENTS.get("shop_to_town", []))
                + list(SEGMENTS.get("town_shop_to_path", []))
                + hops
            )
    sliced = slice_route_from_position(hops, pos.x, pos.y, tilemap=tilemap)
    return MultiMapNavTask(
        name="nav_clear_plot_farm",
        waypoints=sliced or hops,
        timeout=4000,
        initial_settle_frames=8,
    )

def _uses_pocket_approach(self) -> bool:
    """West-pocket CLEAR_PLOT. Quota/type_clear bounds are a leftover chunk."""
    return self.farm_bounds is not None and self.handoff not in ("quota", "type_clear")

def _step_pocket_approach(self, world: WorldState) -> Optional[TaskResult]:
    if not self._uses_pocket_approach() or self._pocket_arrived:
        return None
    if self._pocket_is_ready(world):
        if not self._pocket_arrived:
            pos = get_pos_from_ram(world.ram)
            print(f"[CLEAR] Pocket ready pos=({pos.x},{pos.y}) tile={self._player_tile(world.ram)}")
        self._pocket_arrived = True
        self._lock_clearer_to_plot()
        self._approach = None
        return None
    if self._approach is None:
        self._approach = self._make_pocket_approach(world)
        self._approach.reset(world)
        pos = get_pos_from_ram(world.ram)
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        print(
            f"[CLEAR] Approach plant pocket via {self._approach.name} "
            f"tm=0x{tilemap:02X} pos=({pos.x},{pos.y})"
        )
    result = self._approach.step(world)
    if result.status == TaskStatus.RUNNING:
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=result.action,
            reason=result.reason or "approach plant pocket",
        )
    self._approach = None
    if self._pocket_is_ready(world):
        self._pocket_arrived = True
        self._lock_clearer_to_plot()
        return None
    return TaskResult(
        status=TaskStatus.RUNNING,
        action=ActionResult(make_action()),
        reason="approach plant pocket",
    )

def can_start(self, world: WorldState) -> bool:
    from harvest.planner.day_plan_status import FARM_TILEMAP, is_farm_tilemap
    ram = world.ram
    if ram is None or ADDR_TILEMAP >= len(ram):
        return False
    # Pocket clear after shop: the west-gate scan is empty because (13,28)
    # is still stale. Always start and walk in before scanning.
    if self.farm_bounds is not None:
        return True
    tilemap = int(ram[ADDR_TILEMAP])
    if not is_farm_tilemap(tilemap) and tilemap != FARM_TILEMAP:
        return True
    return TileScanner().has_clearable_debris(ram, self._scan_bounds())

def _on_farm(self, world: WorldState) -> bool:
    from harvest.planner.day_plan_status import FARM_TILEMAP, is_farm_tilemap
    ram = world.ram
    if ram is None or ADDR_TILEMAP >= len(ram):
        return False
    tilemap = int(ram[ADDR_TILEMAP])
    return is_farm_tilemap(tilemap) or tilemap == FARM_TILEMAP

def _exit_stand_px(self, pos: Point) -> Tuple[Point, str]:
    if pos.x < 176:
        return Point(4 * TILE_SIZE + 8, 27 * TILE_SIZE + 8), "north (west of fence)"
    if pos.x >= _EAST_STAGING_X:
        label = "north (east of fence)"
    else:
        label = "east past fence then north"
    return Point(30 * TILE_SIZE + 8, 27 * TILE_SIZE + 8), label

def _queue_south_exit_staging(self, world: WorldState) -> None:
    from harvest.planner.tasks.navigation import NavTask
    pos = get_pos_from_ram(world.ram)
    stand, route = self._exit_stand_px(pos)
    print(f"[CLEAR] Exit-staging from south pocket pos=({pos.x},{pos.y}) → {route}")
    self._exit_nav = NavTask(
        name="nav_clear_exit_north", target_px=stand, radius=16, timeout=2500
    )
    self._exit_nav.reset(world)

def _quota_met(self, ram) -> bool:
    if self.handoff != "quota" or not self.quota:
        return False
    return quota_satisfied(ram, self.quota, clearer=self, bounds=self.farm_bounds)

def _complete_status(self, world: WorldState, remaining) -> TaskStatus:
    """Unbounded whole-farm SUCCESS only with empty debris on the farm."""
    if self.handoff == "quota":
        return TaskStatus.SUCCESS if self._quota_met(world.ram) else TaskStatus.FAILURE
    if self.farm_bounds is not None:
        if self._plant_notch_is_clear(world.ram):
            return TaskStatus.SUCCESS
        return TaskStatus.FAILURE
    if remaining or not self._on_farm(world):
        return TaskStatus.FAILURE
    return TaskStatus.SUCCESS

def _step_exit_nav(self, world: WorldState) -> Optional[TaskResult]:
    if self._exit_nav is None:
        return None
    result = self._exit_nav.step(world)
    if result.status == TaskStatus.RUNNING:
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=result.action,
            reason=result.reason or "clear exit-staging south pocket",
        )
    self._exit_nav = None
    reason = self._pending_finish_reason or "clear exit-staging south pocket"
    status = self._pending_finish_status
    self._pending_finish_reason = ""
    self._pending_finish_status = TaskStatus.SUCCESS
    return TaskResult(status=status, reason=reason)

def _unsinkable_lift(self, ram, player, target) -> bool:
    """True when a hand-lift cannot be finished.

    A large rock needs a hammer. Tile 0x04 becomes held 0x0D and tile
    0x05 becomes held 0x0F; with no open landing the clear walks away
    carrying it. A stone north of the fence can still use the south jump.
    """
    if target is None or target.debris_type == DebrisType.ROCK:
        return True
    if target.debris_type == DebrisType.WEED:
        return False
    if open_toss_face(ram, player, blocked=(target.tile,)) is not None:
        return False
    return not (
        target.debris_type == DebrisType.STONE and int(player[1]) <= FENCE_WALL_Y
    )


def _lift_current(self, ram, player, target):
    held = read_held_item(ram)
    if held:
        self._toss_before_lift += 1
        if self._toss_before_lift > 3:
            print(f"[CLEARER] Still held=0x{held:02X}; stop clear at {target}")
            self.failed_tiles.add(target)
            self.current_target = None
            self.clearing_start_frame = 0
            self._toss_before_lift = 0
            return "complete"
        print(f"[CLEARER] Toss held=0x{held:02X} before next lift")
        self._queue_held_toss(ram, player, held, face="down")
        return None
    self._toss_before_lift = 0
    if self._unsinkable_lift(ram, player, self.current_target):
        kind = self.current_target.debris_type.name if self.current_target else "debris"
        print(f"[CLEARER] Skip unsinkable {kind} at {target}")
        self.failed_tiles.add(target)
        self.current_target = None
        self.clearing_start_frame = 0
        return "scanning"
    lift_key = (target[0], target[1], int(self.current_target.tile_id))
    attempts = self.tile_attempts.get(lift_key, 0)
    if attempts >= 2 or target in self.failed_tiles:
        print(
            f"[CLEARER] Skipping lift thrash at {target} "
            f"({self.current_target.debris_type.name})"
        )
        self.failed_tiles.add(target)
        self.current_target = None
        self.clearing_start_frame = 0
        return "scanning"
    self.tile_attempts[lift_key] = attempts + 1
    print(
        f"[CLEARER] Lifting {self.current_target.debris_type.name} "
        f"at {target} (attempt {attempts + 1}/2)"
    )
    direction = self._face_dir(player, target)
    self.action_queue.extend(make_action(**{direction: True}) for _ in range(3))
    self.action_queue.extend(make_action() for _ in range(4))
    self.action_queue.extend(make_action(a=True) for _ in range(18))
    self.action_queue.extend(make_action() for _ in range(20))
    self._pending_lift_verify = target
    return None


def _maybe_stage_then_success(self, world: WorldState, reason: str) -> TaskResult:
    from harvest.planner.day_plan_status import is_farm_tilemap
    status = self._pending_finish_status
    tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
    if not self._did_south_staging and is_farm_tilemap(tilemap):
        pos = get_pos_from_ram(world.ram)
        if pos.y >= _FENCE_WALL_PX_Y + 8 and pos.x < _EAST_STAGING_X:
            self._did_south_staging = True
            self._pending_finish_reason = reason
            self._queue_south_exit_staging(world)
            stepped = self._step_exit_nav(world)
            if stepped is not None:
                return stepped
    return TaskResult(status=status, reason=reason)

def _finish_or_drop(
    self, world: WorldState, reason: str, *, status: Optional[TaskStatus] = None
) -> TaskResult:
    """Drop a carried weed/rock before the next phase. South of y=31, stage north."""
    from harvest.planner.tasks.transitions import (
        hands_are_clear,
        multi_face_toss_actions,
        toss_held_actions,
    )
    if status is not None:
        self._pending_finish_status = status
    if self._finish_toss is not None:
        result = self._finish_toss.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        self._finish_toss = None
        if hands_are_clear(world.ram):
            return self._maybe_stage_then_success(world, reason)
    if self._drop_queue:
        return self._running(self._drop_queue.popleft(), "drop carried before clear done")
    if self._staging_queue:
        return self._running(self._staging_queue.popleft(), "clear exit-staging south pocket")
    if hands_are_clear(world.ram):
        return self._maybe_stage_then_success(world, reason)
    if self._drop_attempts >= 6:
        held = read_held_item(world.ram)
        print(f"[CLEAR] Drop failed; not leaving clear held=0x{held:02X}")
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"{reason}; held=0x{held:02X}",
        )
    self._drop_attempts += 1
    self._pending_finish_reason = reason
    held = read_held_item(world.ram)
    pos = get_pos_from_ram(world.ram)
    tile = (pos.x // 16, pos.y // 16)
    if self._drop_attempts == 1 and needs_south_fence_drop(tile, held):
        self._finish_toss = FenceJumpTossSkill()
        self._finish_toss.reset(world)
        return self._finish_toss.step(world)
    if self._drop_attempts == 1:
        self._drop_queue.extend(toss_held_actions(face="down", step_away=True))
        self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
    else:
        self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
    print(
        f"[CLEAR] Dropping held item before done ({self._drop_attempts}/6 "
        f"held=0x{read_held_item(world.ram):02X})"
    )
    return self._running(self._drop_queue.popleft(), "drop carried before clear done")

def _running(self, action, reason: str) -> TaskResult:
    wrapped = action if isinstance(action, ActionResult) else ActionResult(action)
    return TaskResult(status=TaskStatus.RUNNING, action=wrapped, reason=reason)

def _finish_idle(self, world: WorldState, remaining) -> TaskResult:
    if (
        self.handoff == "type_clear"
        and remaining
        and self._type_clear_retries < 3
        and (self.timeout <= 0 or self._step_count <= self.timeout)
    ):
        self._type_clear_retries += 1
        self.failed_tiles.clear()
        self.failed_approaches.clear()
        self.state = "scanning"
        self.current_target = None
        return self._running(
            make_action(),
            f"retry type clear pass {self._type_clear_retries} remaining={len(remaining)}",
        )
    lift_note = " lift_only" if self.tools_missing else ""
    finish_status = self._complete_status(world, remaining)
    if self.stamina_exhausted:
        return self._finish_or_drop(
            world, f"stamina_low cleared={self.cleared_count}", status=finish_status
        )
    if remaining:
        return self._finish_or_drop(
            world,
            f"partial_clear cleared={self.cleared_count} remaining={len(remaining)}{lift_note}",
            status=finish_status,
        )
    if self._uses_pocket_approach() and not self._pocket_arrived:
        retry = self._step_pocket_approach(world)
        if retry is not None:
            return retry
    if self.farm_bounds is None and not self._on_farm(world):
        return self._finish_or_drop(
            world,
            f"partial_clear cleared={self.cleared_count} remaining={len(remaining)}{lift_note}",
            status=TaskStatus.FAILURE,
        )
    return self._finish_or_drop(
        world, f"field_clear cleared={self.cleared_count}{lift_note}", status=finish_status
    )

def step(self, world: WorldState) -> TaskResult:
    self._step_count += 1
    # Field clear has no off-farm recovery. run12 walked into the house,
    # the clock froze, and every later farm phase map-locked. Bail on frame 1.
    from harvest.planner.tasks.transitions import hands_are_clear
    if self.startup_done and not self._uses_pocket_approach() and not self._on_farm(world):
        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"off_farm tilemap=0x{tilemap:02X} cleared={self.cleared_count}",
        )
    if self._finish_toss is not None:
        result = self._finish_toss.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        self._finish_toss = None
    if self._drop_queue:
        return self._running(self._drop_queue.popleft(), "drop carried before clear done")
    if self._staging_queue:
        return self._running(self._staging_queue.popleft(), "clear exit-staging south pocket")
    exited = self._step_exit_nav(world)
    if exited is not None:
        return exited
    if self._pending_finish_reason and not hands_are_clear(world.ram):
        return self._finish_or_drop(world, self._pending_finish_reason)
    if self._pending_finish_reason and hands_are_clear(world.ram):
        reason = self._pending_finish_reason
        self._pending_finish_reason = ""
        return self._maybe_stage_then_success(world, reason)
    approached = self._step_pocket_approach(world)
    if approached is not None:
        return approached
    if self.handoff not in ("quota", "type_clear") and self._plant_notch_is_clear(world.ram):
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=f"field_clear plot_ring_clear cleared={self.cleared_count}",
        )
    if self._quota_met(world.ram):
        noop = " no-op" if int(self.cleared_count or 0) <= 0 else ""
        return self._finish_or_drop(
            world, f"field_clear quota_met{noop} cleared={self.cleared_count} quota={self.quota}",
            status=TaskStatus.SUCCESS,
        )
    if self.timeout > 0 and self._step_count > self.timeout:
        remaining = self._remaining_debris(world.ram)
        lift_note = " lift_only" if self.tools_missing else ""
        return self._finish_or_drop(
            world,
            f"clear_budget cleared={self.cleared_count} remaining={len(remaining)}{lift_note}",
            status=self._complete_status(world, remaining),
        )
    action = self.tick(world.ram)
    if action is None:
        return self._finish_idle(world, self._remaining_debris(world.ram))
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action=action))

