"""Day-plan / autoplay Task wrapper around :class:`FarmClearer`."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    CLEARABLE_DEBRIS_TYPES,
    STALE_TILE_IDS,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
)
from harvest.maps.map_config import FARM_POND_ACCESS_FENCE_ROW
from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
from harvest.paths import TASKS_DIR as PROJECT_TASKS_DIR
from harvest.planner.day_plan_status import (
    FARM_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
)
from harvest.planner.tasks.transitions import (
    hands_are_clear,
    multi_face_toss_actions,
    toss_held_actions,
)
from harvest.tasks.farm_toss import FenceJumpTossSkill, needs_south_fence_drop
from harvest.tasks.nav import (
    Point,
    TILE_SIZE,
    get_pos_from_ram,
    get_tile_at,
    make_action,
)
from harvest.tasks.farm_clearer import (
    FarmClearer,
    TileScanner,
)


# y=31 fence wall px — clear finishing south of this leaves return_home stuck
# in the SW rock pocket (rr-5in D8). Walk east past the wall end first.
_FENCE_WALL_PX_Y = FARM_POND_ACCESS_FENCE_ROW * 16  # 496
_EAST_STAGING_X = 480

DEFAULT_TASKS_DIR = os.fspath(PROJECT_TASKS_DIR)


@dataclass
class FarmClearTask(Task):
    """Clear weeds, stones, rocks, and stumps on the farm map.

    Unbounded (no farm_bounds) SUCCESS only when remaining clearable debris
    is empty on the farm map. Incomplete stamina/budget/lift-only leftover
    is FAILURE so the optional day-plan policy can skip/defer. Pocket
    ``CLEAR_PLOT`` (farm_bounds set) still SUCCESS on the 3x3 plot + hoe
    stands, not a single notch cell.
    Leftover smash (``handoff="quota"``) SUCCESS when RAM counts drop by
    the requested quota, even if other debris remains. Unmet quota
    (including ``stamina_exhausted``) is FAILURE — not a plot_ring or
    empty-farm lie.
    Always tries to drop a held weed/rock before finishing so return-home
    is not blocked.
    """

    name: str = "farm_clear"
    priority: Optional[List[DebrisType]] = None
    tasks_dir: str = DEFAULT_TASKS_DIR
    timeout: int = 120000
    fetch_tools: bool = True
    prefer_lift_for_weeds: bool = True
    prefer_lift_for_stones: bool = False
    farm_bounds: Optional[Tuple[int, int, int, int]] = None
    # "plot_ring" (pocket plant) | "quota" (leftover smash) | "" (empty farm).
    handoff: str = ""
    quota: Optional[dict] = None

    _clearer: FarmClearer = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _drop_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _pending_finish_reason: str = field(default="", init=False)
    _staging_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _did_south_staging: bool = field(default=False, init=False)
    _toss_skill: Optional[FenceJumpTossSkill] = field(default=None, init=False)
    _exit_nav: Optional[Task] = field(default=None, init=False)
    # Pocket clear (farm_bounds set): do not trust an empty scan from the
    # shop-return / west-gate pin. Walk into the yard first so (13,28) loads.
    _approach: Optional[Task] = field(default=None, init=False)
    _pocket_arrived: bool = field(default=False, init=False)
    _pending_finish_status: TaskStatus = field(
        default=TaskStatus.SUCCESS, init=False
    )

    def __post_init__(self) -> None:
        self._clearer = FarmClearer(priority=self.priority)
        self._configure_clearer()

    def _configure_clearer(self) -> None:
        self._clearer.tasks_dir = self.tasks_dir
        self._clearer.configure(
            prefer_lift_for_weeds=self.prefer_lift_for_weeds,
            prefer_lift_for_stones=self.prefer_lift_for_stones,
            farm_bounds=self.farm_bounds,
        )
        if self.fetch_tools:
            self._register_default_tool_startup()
        else:
            # Skip shed recordings. Carry scan in reset() drops only
            # types whose tool is actually missing (rr-20w.2.12).
            self._clearer.startup_done = True
            self._clearer._tool_scan_done = True

    def _apply_carry_tools(self, ram) -> None:
        """Keep ROCK/STUMP when hammer/axe is already in the pair."""
        if self.fetch_tools:
            return
        self._clearer.tool_manager.update(ram)
        missing = []
        if not self._clearer.tool_manager.has(int(Tool.HAMMER)):
            missing.append(int(Tool.HAMMER))
        if not self._clearer.tool_manager.has(int(Tool.AXE)):
            missing.append(int(Tool.AXE))
        if missing:
            self._clearer.tools_missing = True
            self._clearer._enable_lift_only_mode(missing)
        else:
            self._clearer.tools_missing = False

    def _register_default_tool_startup(self) -> None:
        # Position-locked get_hammer/get_axe recordings only work from their
        # original start states. Prefer inventory scan + lift-only weeds/stones
        # unless explicitly re-enabled for tool-fetch experiments.
        use_recordings = os.getenv(
            "FETCH_CLEAR_TOOL_RECORDINGS", ""
        ).lower() in ("1", "true", "yes")
        if not use_recordings:
            return

        if not os.getenv("SKIP_HAMMER", "").lower() in ("1", "true", "yes"):
            get_hammer = os.path.join(self.tasks_dir, "get_hammer.json")
            shed_grab = os.path.join(
                self.tasks_dir, "shed_grab_hammer_smash_rock.json"
            )
            if os.path.exists(get_hammer):
                self._clearer.add_startup_task("task", name="get_hammer")
            elif os.path.exists(shed_grab):
                self._clearer.add_startup_task(
                    "nav",
                    name="go_shed",
                    target=Point(342, 489),
                    radius=12,
                    timeout=1800,
                )
                self._clearer.add_startup_task(
                    "task", name="shed_grab_hammer_smash_rock"
                )

        if not os.getenv("SKIP_AXE", "").lower() in ("1", "true", "yes"):
            get_axe = os.path.join(self.tasks_dir, "get_axe.json")
            if os.path.exists(get_axe):
                self._clearer.add_startup_task("task", name="get_axe")

    @property
    def clearer(self) -> FarmClearer:
        return self._clearer

    @property
    def progress_text(self) -> str:
        phase = self._clearer.current_phase
        phase_name = phase.name if phase else self._clearer.state
        return (
            f"{phase_name} cleared={self._clearer.cleared_count} "
            f"failed={len(self._clearer.failed_tiles)}"
        )

    def progress_snapshot(self) -> ProgressSnapshot:
        target = self._clearer.current_target
        details = (
            ("cleared", self._clearer.cleared_count),
            ("failed", len(self._clearer.failed_tiles)),
            ("state", self._clearer.state),
            (
                "target",
                target.tile if target is not None else None,
            ),
            ("stamina_exhausted", self._clearer.stamina_exhausted),
        )
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self._clearer.state,
            step_count=self._step_count,
            details=details,
        )

    def reset(self, world: WorldState) -> None:
        priority = list(self.priority) if self.priority else None
        self._clearer = FarmClearer(priority=priority)
        self._configure_clearer()
        self._apply_carry_tools(world.ram)
        self._step_count = 0
        self._started = True
        self._drop_queue.clear()
        self._drop_attempts = 0
        self._pending_finish_reason = ""
        self._pending_finish_status = TaskStatus.SUCCESS
        self._staging_queue.clear()
        self._did_south_staging = False
        self._toss_skill = None
        self._approach = None
        self._exit_nav = None
        if self.handoff == "quota":
            from harvest.tasks.farm_clear_quota import count_debris

            self._clearer.quota_start_counts = count_debris(
                world.ram, self.farm_bounds
            )
        self._pocket_arrived = False

    def _scan_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        if self.farm_bounds is not None and self._pocket_arrived:
            return self._plot_scan_bounds()
        return self.farm_bounds or self._clearer._locked_bounds or self._clearer.farm_bounds

    def _remaining_debris(self, ram) -> list:
        return TileScanner().scan(
            ram, self._scan_bounds(), types=set(CLEARABLE_DEBRIS_TYPES)
        )

    def _player_tile(self, ram) -> Tuple[int, int]:
        pos = get_pos_from_ram(ram)
        return (pos.x // TILE_SIZE, pos.y // TILE_SIZE)

    def _in_pocket(self, ram) -> bool:
        bounds = self.farm_bounds
        if bounds is None:
            return True
        tx, ty = self._player_tile(ram)
        if not (bounds[0] <= tx <= bounds[2] and bounds[1] <= ty <= bounds[3]):
            return False
        # West-fence (3,28) is in the box but not the plant stand. Ready only
        # near (13,28) so we do not scan-and-wander from the west wall.
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
        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        return (
            is_farm_tilemap(tilemap)
            and self._in_pocket(ram)
            and self._pocket_tiles_ready(ram)
        )

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
        """Stop roaming the full pocket after arrival — only the 3x3 + stands."""
        bounds = self._plot_scan_bounds()
        self._clearer.farm_bounds = bounds
        self._clearer._locked_bounds = bounds
        print(f"[CLEAR] Plot scan bounds {bounds}")

    def _plant_notch_is_clear(self, ram) -> bool:
        """True when the 8-tile plot and hoe stands are free of debris."""
        if self.farm_bounds is None or not self._pocket_arrived:
            return False
        if not hands_are_clear(ram):
            return False
        for tx, ty in self._plot_cells_to_clear():
            tile_id = int(get_tile_at(ram, tx, ty))
            if tile_id in STALE_TILE_IDS or tile_id in TILE_TO_DEBRIS:
                return False
        return True

    def _pocket_stand_px(self) -> Point:
        # (13,28) is the weed notch after shop. Stand on the untilled west
        # neighbor so BFS does not have to target the bush tile itself.
        cx, cy = WEST_POCKET_PLANT_CENTER
        return Point((cx - 1) * TILE_SIZE + 8, cy * TILE_SIZE + 8)

    def _make_pocket_approach(self, world: WorldState) -> Task:
        from harvest.maps.map_config import SEGMENTS, slice_route_from_position
        from harvest.planner.tasks.inventory_exit import ExitToFarmTask
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        from harvest.planner.tasks.navigation import NavTask

        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap == 0x26:
            # Shed door walk-in during pocket approach — leave, then retry farm nav.
            return ExitToFarmTask(tasks_dir=self.tasks_dir)
        if is_farm_tilemap(tilemap):
            stand = self._pocket_stand_px()
            return NavTask(
                name="nav_clear_plot_pocket",
                target_px=stand,
                radius=14,
                timeout=3500,
            )
        if is_house_tilemap(tilemap):
            return ExitToFarmTask(tasks_dir=self.tasks_dir)
        hops = list(SEGMENTS.get("path_to_farm", []))
        if tilemap == 0x04:
            hops = list(SEGMENTS.get("town_shop_to_path", [])) + hops
        elif tilemap == 0x1C:
            hops = (
                list(SEGMENTS.get("shop_to_town", []))
                + list(SEGMENTS.get("town_shop_to_path", []))
                + hops
            )
        pos = get_pos_from_ram(ram)
        sliced = slice_route_from_position(hops, pos.x, pos.y, tilemap=tilemap)
        return MultiMapNavTask(
            name="nav_clear_plot_farm",
            waypoints=sliced or hops,
            timeout=4000,
            initial_settle_frames=8,
        )

    def _step_pocket_approach(self, world: WorldState) -> Optional[TaskResult]:
        """Walk into the west plant pocket before trusting an empty scan."""
        if self.farm_bounds is None or self._pocket_arrived:
            return None
        if self._pocket_is_ready(world):
            if not self._pocket_arrived:
                pos = get_pos_from_ram(world.ram)
                print(
                    f"[CLEAR] Pocket ready pos=({pos.x},{pos.y}) "
                    f"tile={self._player_tile(world.ram)}"
                )
            self._pocket_arrived = True
            self._lock_clearer_to_plot()
            self._approach = None
            return None
        if self._approach is None:
            self._approach = self._make_pocket_approach(world)
            self._approach.reset(world)
            pos = get_pos_from_ram(world.ram)
            tilemap = (
                int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            )
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
        # Hop finished (farm gate) but the notch is still off-screen — next
        # frame starts the in-pocket NavTask.
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason="approach plant pocket",
        )

    def can_start(self, world: WorldState) -> bool:
        ram = world.ram
        if ram is None or ADDR_TILEMAP >= len(ram):
            return False
        # Pocket clear after shop: the west-gate / path scan is empty because
        # (13,28) is still stale. Always start and walk in before scanning.
        if self.farm_bounds is not None:
            return True
        tilemap = int(ram[ADDR_TILEMAP])
        if not is_farm_tilemap(tilemap) and tilemap != FARM_TILEMAP:
            # Allow house/shed startup tool fetches; clearer handles map travel
            # via recordings. Debris presence is checked from farm RAM when on
            # farm; off-farm we still allow start so tools can be fetched.
            return True
        return TileScanner().has_clearable_debris(ram, self._scan_bounds())

    def _exit_stand_px(self, pos: Point) -> Tuple[Point, str]:
        """North-of-fence stand. Never cycle left/right in place."""
        if pos.x < 176:
            return Point(4 * TILE_SIZE + 8, 27 * TILE_SIZE + 8), "north (west of fence)"
        return (
            Point(30 * TILE_SIZE + 8, 27 * TILE_SIZE + 8),
            "north (east of fence)" if pos.x >= _EAST_STAGING_X else "east past fence then north",
        )

    def _queue_south_exit_staging(self, world: WorldState) -> None:
        """Leave south-of-fence pocket via BFS, not a left/right thrash cycle."""
        from harvest.planner.tasks.navigation import NavTask

        pos = get_pos_from_ram(world.ram)
        stand, route = self._exit_stand_px(pos)
        print(
            f"[CLEAR] Exit-staging from south pocket "
            f"pos=({pos.x},{pos.y}) → {route}"
        )
        self._exit_nav = NavTask(
            name="nav_clear_exit_north",
            target_px=stand,
            radius=16,
            timeout=2500,
        )
        self._exit_nav.reset(world)

    def _on_farm(self, world: WorldState) -> bool:
        ram = world.ram
        if ram is None or ADDR_TILEMAP >= len(ram):
            return False
        tilemap = int(ram[ADDR_TILEMAP])
        return is_farm_tilemap(tilemap) or tilemap == FARM_TILEMAP

    def _quota_met(self, ram) -> bool:
        if self.handoff != "quota" or not self.quota:
            return False
        from harvest.tasks.farm_clear_quota import quota_satisfied

        return quota_satisfied(
            ram, self.quota, clearer=self._clearer, bounds=self.farm_bounds
        )

    def _complete_status(self, world: WorldState, remaining) -> TaskStatus:
        """Unbounded whole-farm SUCCESS only with empty debris on farm."""
        if self.handoff == "quota":
            return (
                TaskStatus.SUCCESS
                if self._quota_met(world.ram)
                else TaskStatus.FAILURE
            )
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

    def _maybe_stage_then_success(
        self, world: WorldState, reason: str
    ) -> TaskResult:
        """After drop, leave the south-of-fence pocket before finishing."""
        status = self._pending_finish_status
        if not self._did_south_staging and is_farm_tilemap(
            int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        ):
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
        self,
        world: WorldState,
        reason: str,
        *,
        status: Optional[TaskStatus] = None,
    ) -> TaskResult:
        """Do not hand a carried weed/rock to the next day-plan phase.

        Prefer multi-face stationary A-drop (fence_flow proven). Still SUCCESS
        with held after budget so the day can sleep — ReturnHomeTask then
        relocates to open ground and retries (rr-6g7g).
        When finishing south of the y=31 fence wall, B-run east first so
        return_home is not born in the SW rock pocket (rr-5in D8).
        """
        if status is not None:
            self._pending_finish_status = status
        if self._toss_skill is not None:
            result = self._toss_skill.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._toss_skill = None
            if hands_are_clear(world.ram):
                return self._maybe_stage_then_success(world, reason)
        if self._drop_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason="drop carried before clear done",
            )
        if self._staging_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._staging_queue.popleft()),
                reason="clear exit-staging south pocket",
            )
        if hands_are_clear(world.ram):
            return self._maybe_stage_then_success(world, reason)
        drop_limit = 6
        if self._drop_attempts >= drop_limit:
            held = read_held_item(world.ram)
            print(
                f"[CLEAR] Leaving clear with held=0x{held:02X} after drop attempts"
            )
            # Still try exit-staging even with held — return_home clears hands.
            return self._maybe_stage_then_success(
                world, f"{reason}; held=0x{held:02X}"
            )
        self._drop_attempts += 1
        self._pending_finish_reason = reason
        held = read_held_item(world.ram)
        pos = get_pos_from_ram(world.ram)
        tile = (pos.x // 16, pos.y // 16)
        if self._drop_attempts == 1 and needs_south_fence_drop(tile, held):
            self._toss_skill = FenceJumpTossSkill()
            self._toss_skill.reset(world)
            return self._toss_skill.step(world)
        if self._drop_attempts == 1:
            self._drop_queue.extend(toss_held_actions(face="down", step_away=True))
            self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
        else:
            self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
        print(
            f"[CLEAR] Dropping held item before done "
            f"({self._drop_attempts}/{drop_limit} "
            f"held=0x{read_held_item(world.ram):02X})"
        )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._drop_queue.popleft()),
            reason="drop carried before clear done",
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._toss_skill is not None:
            result = self._toss_skill.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._toss_skill = None
        if self._drop_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason="drop carried before clear done",
            )
        if self._staging_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._staging_queue.popleft()),
                reason="clear exit-staging south pocket",
            )
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

        # CLEAR_PLOT feeds the 3x3 hoe/plant skill at (13,28). Hand off when
        # the ring + hoe stands are clear instead of roaming the full pocket.
        # Quota leftover (CLEAR_BUSHES / ROCKS / STUMPS) must not use this.
        if self.handoff != "quota" and self._plant_notch_is_clear(world.ram):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    f"field_clear plot_ring_clear "
                    f"cleared={self._clearer.cleared_count}"
                ),
            )
        if self._quota_met(world.ram):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    f"field_clear quota_met "
                    f"cleared={self._clearer.cleared_count} quota={self.quota}"
                ),
            )

        if self._step_count > self.timeout:
            remaining = self._remaining_debris(world.ram)
            lift_note = " lift_only" if self._clearer.tools_missing else ""
            return self._finish_or_drop(
                world,
                (
                    f"clear_budget cleared={self._clearer.cleared_count} "
                    f"remaining={len(remaining)}{lift_note}"
                ),
                status=self._complete_status(world, remaining),
            )

        action = self._clearer.tick(world.ram)
        if action is None:
            remaining = self._remaining_debris(world.ram)
            lift_note = (
                " lift_only" if self._clearer.tools_missing else ""
            )
            finish_status = self._complete_status(world, remaining)
            if self._clearer.stamina_exhausted:
                return self._finish_or_drop(
                    world,
                    f"stamina_low cleared={self._clearer.cleared_count}",
                    status=finish_status,
                )
            if remaining:
                return self._finish_or_drop(
                    world,
                    (
                        f"partial_clear cleared={self._clearer.cleared_count} "
                        f"remaining={len(remaining)}{lift_note}"
                    ),
                    status=finish_status,
                )
            if self.farm_bounds is not None and not self._pocket_arrived:
                retry = self._step_pocket_approach(world)
                if retry is not None:
                    return retry
            if self.farm_bounds is None and not self._on_farm(world):
                return self._finish_or_drop(
                    world,
                    (
                        f"partial_clear cleared={self._clearer.cleared_count} "
                        f"remaining={len(remaining)}{lift_note}"
                    ),
                    status=TaskStatus.FAILURE,
                )
            return self._finish_or_drop(
                world,
                f"field_clear cleared={self._clearer.cleared_count}{lift_note}",
                status=finish_status,
            )

        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action=action),
        )


__all__ = ["FarmClearTask"]
