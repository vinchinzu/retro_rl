"""Farm clear Task. Nav and tile-scan helpers are re-exported from here."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from retro_harness import Task, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.stamina import SWING_STAMINA_COST, Stamina
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_STAMINA,
    CLEARABLE_DEBRIS_TYPES,
    MAP_WIDTH,
    STALE_TILE_IDS,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
)
from harvest.paths import TASKS_DIR as PROJECT_TASKS_DIR
from harvest.tasks.farm_clear_quota import (
    count_debris,
    farm_map_loaded,
    needs_shed_door_step_off,
    unmet_debris_types,
)
from harvest.tasks.farm_ops import (  # noqa: F401
    DEFAULT_PRIORITY,
    Target,
    TileScanner,
    ToolManager,
    action_to_names,
    choose_clear_target,
    cycle_tool,
    drop_unarmed_debris,
    find_unfailed_approach,
    parse_priority_list,
    shed_door_step_off_actions,
    snap_debris_anchor,
    sort_targets_cluster,
    start_progress_watch,
    use_tool,
    use_tool_facing,
)
from harvest.tasks.farm_toss import (
    evaluate_lift_verify,
    in_place_toss_actions,
    needs_south_fence_drop,
    start_fence_jump_skill,
    step_fence_jump_skill,
)
from harvest.tasks.nav import (  # noqa: F401
    VIEWPORT_HOP_TILES,
    WALKABLE_TILES,
    Navigator,
    Pathfinder,
    Point,
    get_pos_from_ram,
    get_tile_at,
    make_action,
    manhattan,
    tile_dist,
)

# Hammer/axe hits cost 2 stamina; do not start a multi-hit below this.
# Lifts may continue at 1. The 8-swing start budget is Stamina.can_finish_multi_hit.
MIN_CLEAR_STAMINA = 4
FACE_SETTLE_FRAMES = 8
Y_HOLD_FRAMES = 20
SWING_COOLDOWN_FRAMES = 20
POST_SWING_OBSERVE_FRAMES = 2
MAX_OBSERVE_EXTRA = 4
MAX_STAND_MISSES = 3
_STARTUP_TOOL_NAMES = {
    "get_hammer": Tool.HAMMER,
    "get_axe": Tool.AXE,
    "get_sickle": Tool.SICKLE,
    "get_hoe": Tool.HOE,
}
Tile = Tuple[int, int]


class FarmClearer(Task):
    """Clear weeds, stones, rocks, and stumps. One ``step`` drives the farm."""

    def __init__(
        self,
        priority: Optional[List[DebrisType]] = None,
        *,
        name: str = "farm_clear",
        tasks_dir: Optional[str] = None,
        timeout: int = 120000,
        fetch_tools: bool = True,
        prefer_lift_for_weeds: bool = True,
        prefer_lift_for_stones: bool = False,
        farm_bounds: Optional[Tuple[int, int, int, int]] = None,
        handoff: str = "",
        quota: Optional[dict] = None,
    ):
        self.name = name
        self.tasks_dir = os.fspath(
            PROJECT_TASKS_DIR if tasks_dir is None else tasks_dir
        )
        self.timeout = int(timeout)
        self.fetch_tools = bool(fetch_tools)
        self._lift_weeds = bool(prefer_lift_for_weeds)
        self._lift_stones = bool(prefer_lift_for_stones)
        self._requested_bounds = (
            tuple(int(v) for v in farm_bounds) if farm_bounds is not None else None
        )
        self.handoff = str(handoff or "")
        self.quota = quota
        self._priority_spec = list(priority) if priority else None
        self._boot()

    # Requested bounds stay distinct from the scan box. Assigning farm_bounds
    # after reset must not shrink a clearer that was built unbounded.
    @property
    def farm_bounds(self):
        return self._requested_bounds

    @farm_bounds.setter
    def farm_bounds(self, value) -> None:
        self._requested_bounds = (
            tuple(int(v) for v in value) if value is not None else None
        )

    @property
    def clearer(self) -> "FarmClearer":
        return self

    def configure(self, **kwargs) -> None:
        if "prefer_lift_for_weeds" in kwargs:
            self._lift_weeds = bool(kwargs["prefer_lift_for_weeds"])
        if "prefer_lift_for_stones" in kwargs:
            self._lift_stones = bool(kwargs["prefer_lift_for_stones"])
        if "priority" in kwargs and kwargs["priority"] is not None:
            self._priority_spec = list(kwargs["priority"])
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.farm_bounds is not None:
            self._locked_bounds = tuple(self.farm_bounds)
            self._work_bounds = None

    def _boot(self) -> None:
        spec = self._priority_spec
        self.priority = list(spec) if spec else DEFAULT_PRIORITY.copy()
        self.scanner = TileScanner()
        self.pathfinder = Pathfinder(self.scanner)
        self.navigator = Navigator(self.pathfinder)
        self.tool_manager = ToolManager()
        self.action_queue = deque()
        self.task_queue = deque()
        self._drop_queue = deque()
        self._staging_queue = deque()
        self.state = "scanning"
        self.failed_tiles, self.tiles_cleared = set(), set()
        self.tile_attempts, self.failed_approaches = {}, set()
        self.startup_tasks = []
        self.prefer_lift_for_weeds = self._lift_weeds
        self.prefer_lift_for_stones = self._lift_stones
        self.max_stasis, self.max_nav_no_progress = 120, 360
        self.debug_interval, self.min_stamina = 300, MIN_CLEAR_STAMINA
        self.max_scan_misses = 90
        self._pending_finish_reason = ""
        self._pending_finish_status = TaskStatus.SUCCESS
        for name in (
            "current_phase", "current_target", "approach_tile", "quota_start_counts",
            "_locked_bounds", "_work_bounds", "_nav_best_distance", "searching_tool",
            "_pending_lift_verify", "_pending_toss_origin", "_toss_skill", "_finish_toss",
            "_approach", "_exit_nav",
        ):
            setattr(self, name, None)
        for name, value in (
            ("cleared_count", 0), ("frame_count", 0), ("tool_search_frames", 0),
            ("startup_index", 0), ("target_hits", 0), ("clearing_start_frame", 0),
            ("suppress_move_frames", 0), ("_toss_before_lift", 0), ("_tool_scan_frames", 0),
            ("_step_count", 0), ("_drop_attempts", 0), ("_type_clear_retries", 0),
            ("scan_miss_streak", 0), ("_nav_last_progress_frame", 0),
        ):
            setattr(self, name, value)
        self.stamina_exhausted = self.tools_missing = False
        self.startup_done = self._tool_scan_done = False
        self._did_south_staging = self._pocket_arrived = False
        self._reset_tool_seq()
        self._init_no_go()
        if self.farm_bounds is not None:
            self._locked_bounds = tuple(self.farm_bounds)
        if self.fetch_tools:
            self._register_default_tool_startup()
        else:
            self.startup_done = True
            self._tool_scan_done = True

    def _init_no_go(self) -> None:
        default = "9,26;9,27;9,28;11,26;11,27;11,28;8,12;9,12;10,12"
        raw = os.getenv("NO_GO_TILES", default).replace("|", ";")
        for entry in raw.split(";"):
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            try:
                self.pathfinder.no_go_tiles.add((int(parts[0]), int(parts[1])))
            except ValueError:
                pass

    def add_startup_task(self, task_type: str, **kwargs) -> None:
        self.startup_tasks.append({"type": task_type, **kwargs})

    def _register_default_tool_startup(self) -> None:
        # Position-locked shed tapes only match their record start. Default is
        # an inventory scan plus lift-only for whatever tool is actually missing.
        flag = os.getenv("FETCH_CLEAR_TOOL_RECORDINGS", "").lower()
        if flag not in ("1", "true", "yes"):
            return
        if os.getenv("SKIP_HAMMER", "").lower() not in ("1", "true", "yes"):
            hammer = os.path.join(self.tasks_dir, "get_hammer.json")
            shed = os.path.join(self.tasks_dir, "shed_grab_hammer_smash_rock.json")
            if os.path.exists(hammer):
                self.add_startup_task("task", name="get_hammer")
            elif os.path.exists(shed):
                self.add_startup_task(
                    "nav", name="go_shed", target=Point(342, 489), radius=12, timeout=1800
                )
                self.add_startup_task("task", name="shed_grab_hammer_smash_rock")
        if os.getenv("SKIP_AXE", "").lower() not in ("1", "true", "yes"):
            if os.path.exists(os.path.join(self.tasks_dir, "get_axe.json")):
                self.add_startup_task("task", name="get_axe")

    def _apply_carry_tools(self, ram) -> None:
        """Keep ROCK/STUMP when hammer/axe is already in the pair."""
        if self.fetch_tools:
            return
        self.tool_manager.update(ram)
        missing = []
        if not self.tool_manager.has(int(Tool.HAMMER)):
            missing.append(int(Tool.HAMMER))
        if not self.tool_manager.has(int(Tool.AXE)):
            missing.append(int(Tool.AXE))
        if missing:
            self.tools_missing = True
            self._enable_lift_only_mode(missing)
        else:
            self.tools_missing = False

    def reset(self, world: WorldState) -> None:
        self._boot()
        self._apply_carry_tools(world.ram)
        if self.handoff == "quota":
            self.quota_start_counts = count_debris(world.ram, self.farm_bounds)

    def _active_bounds(self):
        if self._locked_bounds is not None:
            return self._locked_bounds
        if self._work_bounds is not None:
            return self._work_bounds
        return None

    def _reset_tool_seq(self) -> None:
        self._tool_swing_pending = False
        self._tool_last_hits = 0
        self._tool_last_stam = None
        self._tool_misses = 0
        self._tool_observe_extra = 0
        self._tool_faced = False
        self._tool_seq_key = None

    def _reset_target(self) -> None:
        self.current_target = None
        self.approach_tile = None
        self.clearing_start_frame = 0
        self.target_hits = 0
        self._reset_tool_seq()

    def _hit_edge(self, ram: np.ndarray) -> bool:
        """True when RAM shows a registered swing after the observe wait."""
        stam = Stamina.from_ram(ram)
        last_hits = int(self._tool_last_hits or 0)
        if stam.tool_hits > last_hits:
            self.target_hits = stam.tool_hits
            return True
        last_stam = self._tool_last_stam
        if last_stam is not None and stam.current <= int(last_stam) - SWING_STAMINA_COST:
            # $096D can lag the stamina debit by one frame.
            self.target_hits = max(int(self.target_hits), last_hits + 1)
            return True
        return False

    def _retry_other_stand(self, ram: np.ndarray, target: Tile) -> str:
        current = self.current_target
        player = self.navigator.current_tile
        self.failed_approaches.add((target, player))
        nxt = find_unfailed_approach(self, ram, current) if current is not None else None
        print(f"[CLEARER] Three planted misses at {player} vs {target}; next stand={nxt}")
        if nxt is None:
            self.failed_tiles.add(target)
            self._reset_target()
            return "scanning"
        self.approach_tile = nxt
        self.target_hits = 0
        self.clearing_start_frame = 0
        self._reset_tool_seq()
        return "navigating"

    def _observe_pending_swing(self, ram: np.ndarray, target: Tile) -> Optional[str]:
        if not self._tool_swing_pending:
            return None
        if self._hit_edge(ram):
            self._tool_swing_pending = False
            self._tool_observe_extra = 0
            self._tool_misses = 0
            self.clearing_start_frame = self.frame_count or 1
            print(f"[CLEARER] Hit registered {self.target_hits} stam={Stamina.from_ram(ram)} (planted)")
            return None
        if self._tool_observe_extra < MAX_OBSERVE_EXTRA:
            self._tool_observe_extra += 1
            self.action_queue.append(make_action())
            return None
        self._tool_swing_pending = False
        self._tool_observe_extra = 0
        self._tool_misses += 1
        print(
            f"[CLEARER] Swing miss {self._tool_misses}/{MAX_STAND_MISSES} "
            f"at {target} from {self.navigator.current_tile}; stay planted"
        )
        if self._tool_misses >= MAX_STAND_MISSES:
            return self._retry_other_stand(ram, target)
        return None

    def _queue_planted_swing(self, ram: np.ndarray, player: Tile, target: Tile) -> None:
        current = self.current_target
        stam = Stamina.from_ram(ram)
        if not self._tool_faced:
            cells = tuple(current.footprint) if current is not None else (target,)
            face_tile = min(cells or (target,), key=lambda c: abs(c[0] - player[0]) + abs(c[1] - player[1]))
            direction = self._face_dir(player, face_tile)
            self.action_queue.append(make_action(**{direction: True}))
            self.action_queue.extend(make_action() for _ in range(FACE_SETTLE_FRAMES))
            self._tool_faced = True
            print(f"[CLEARER] Face {direction} at {target} from {player}, then Y-only until it breaks")
        self._tool_last_hits = stam.tool_hits
        self._tool_last_stam = stam.current
        self.action_queue.extend(use_tool(frames=Y_HOLD_FRAMES, cooldown=SWING_COOLDOWN_FRAMES))
        self.action_queue.extend(make_action() for _ in range(POST_SWING_OBSERVE_FRAMES))
        self._tool_swing_pending = True

    def handle_tool_clear(self, ram: np.ndarray, *, player: Tile, target: Tile) -> Optional[str]:
        """Queue one planted tool attempt. Credit hits only from a RAM edge."""
        current = self.current_target
        if current is None:
            return "scanning"
        if current.required_tool is None:
            self.failed_tiles.add(target)
            self._reset_target()
            return "scanning"
        key = (target[0], target[1], player[0], player[1])
        if self._tool_seq_key != key:
            self._reset_tool_seq()
            self._tool_seq_key = key
            self.target_hits = 0
        tool = current.required_tool
        if self.tool_manager.current != tool:
            print(f"[CLEARER] Need {tool.name}, have 0x{self.tool_manager.current:02X}")
            self.searching_tool = tool
            self.tool_manager.start_search()
            self.tool_search_frames = 0
            self._reset_tool_seq()
            return "tool_switch"
        observe = self._observe_pending_swing(ram, target)
        if observe is not None:
            return observe
        stamina = Stamina.from_ram(ram)
        if stamina.tool_hits > int(self.target_hits):
            self.target_hits = stamina.tool_hits
        if self.target_hits == 0 and not self._can_afford_target(ram, current):
            print(
                f"[CLEARER] Skip {current.debris_type.name} at {target}: need "
                f"{stamina.cost_to_clear(current.required_hits)} stam for "
                f"{current.required_hits}+miss budget, have {stamina}"
            )
            self.stamina_exhausted = True
            self._reset_target()
            return "scanning"
        if self.target_hits > 0 and stamina < 2:
            self.stamina_exhausted = True
            self._reset_target()
            return "complete"
        if self.target_hits == 0 and not self._tool_faced:
            tile_key = (target[0], target[1], current.tile_id)
            attempts = self.tile_attempts.get(tile_key, 0)
            if attempts >= 3:
                print(
                    f"[CLEARER] Giving up on {current.debris_type.name} at {target} "
                    f"tile=0x{current.tile_id:02X} (3 failed attempts)"
                )
                self.failed_tiles.add(target)
                self._reset_target()
                return "scanning"
            self.tile_attempts[tile_key] = attempts + 1
            verb = "Clearing" if attempts == 0 else "Re-targeting"
            print(
                f"[CLEARER] {verb} {current.debris_type.name} at {target} "
                f"tile=0x{current.tile_id:02X} from {player} "
                f"({current.required_hits} hits, attempt {attempts + 1}/3)"
            )
        self._queue_planted_swing(ram, player, target)
        return None

    def tool_clear_is_planted(self) -> bool:
        """True once the first face is committed; d-pad must stay off."""
        current = self.current_target
        if current is None or self._should_lift(current):
            return False
        return bool(self._tool_faced or self._tool_swing_pending or int(self.target_hits) > 0)

    def _load_task(self, name: str) -> Optional[List[np.ndarray]]:
        if not self.tasks_dir:
            return None
        path = os.path.join(self.tasks_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            data = json.load(handle)
        return [np.array(frame, dtype=np.int32) for frame in data.get("frames", [])]

    def _requested_startup_tools(self) -> Set[int]:
        wanted: Set[int] = set()
        for step in self.startup_tasks:
            if step.get("type") != "task":
                continue
            tool_id = _STARTUP_TOOL_NAMES.get(str(step.get("name", "")))
            if tool_id is not None:
                wanted.add(int(tool_id))
        return wanted

    def _enable_lift_only_mode(self, missing: List[int]) -> None:
        """Drop only debris whose required tool is actually missing."""
        self.prefer_lift_for_weeds = True
        if int(Tool.HAMMER) in missing:
            self.prefer_lift_for_stones = True
        self.priority = drop_unarmed_debris(self.priority, missing)
        names = ", ".join(f"0x{tool:02X}" for tool in missing) or "lift-only"
        kept = ", ".join(dt.name for dt in self.priority)
        print(f"[CLEARER] Startup missing tools: {names}; priority={kept}")

    def _finalize_startup_tools(self) -> None:
        """Re-scan carry (selected + backpack) and drop unarmed debris types."""
        have = set(self.tool_manager.seen)
        have.add(self.tool_manager.current)
        if self.tool_manager.has(int(Tool.HAMMER)):
            have.add(int(Tool.HAMMER))
        if self.tool_manager.has(int(Tool.AXE)):
            have.add(int(Tool.AXE))
        missing = set(self._requested_startup_tools() - have)
        if DebrisType.ROCK in self.priority and not self.tool_manager.has(int(Tool.HAMMER)):
            missing.add(int(Tool.HAMMER))
        if DebrisType.STUMP in self.priority and not self.tool_manager.has(int(Tool.AXE)):
            missing.add(int(Tool.AXE))
        if missing:
            self.tools_missing = True
            self._enable_lift_only_mode(sorted(missing))
        else:
            self.tools_missing = False

    def _run_startup(self, ram: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if self.startup_done:
            return False, None
        if not self._tool_scan_done:
            self._tool_scan_frames += 1
            self.tool_manager.record()
            if self.tool_manager.cycle_complete() or self._tool_scan_frames > 60:
                self._tool_scan_done = True
                found = [f"0x{t:02X}" for t in sorted(self.tool_manager.seen)]
                print(f"[CLEARER] Tool inventory: {', '.join(found)}")
            else:
                if self._tool_scan_frames % 6 == 0:
                    self.action_queue.extend(cycle_tool())
                queued = self.action_queue.popleft() if self.action_queue else make_action()
                return True, queued
        if self.task_queue:
            return True, self.task_queue.popleft()
        if self.startup_index >= len(self.startup_tasks):
            self._finalize_startup_tools()
            self.startup_done = True
            print("[CLEARER] Startup complete")
            return False, None
        step = self.startup_tasks[self.startup_index]
        step_type = step.get("type", "")
        if step_type == "task":
            task_name = step.get("name", "")
            tool_id = _STARTUP_TOOL_NAMES.get(task_name)
            if tool_id is not None and self.tool_manager.has(int(tool_id)):
                print(f"[CLEARER] Skipping {task_name} (already have {tool_id.name})")
                self.startup_index += 1
                return True, make_action()
            frames = self._load_task(task_name)
            if frames:
                print(f"[CLEARER] Task: {task_name} ({len(frames)} frames)")
                self.task_queue.extend(frames)
            else:
                print(f"[CLEARER] Task not found: {task_name}")
            self.startup_index += 1
            queued = self.task_queue.popleft() if self.task_queue else make_action()
            return True, queued
        if step_type == "nav":
            return self._startup_nav(ram, step)
        self.startup_index += 1
        return True, make_action()

    def _startup_nav(self, ram: np.ndarray, step: dict) -> Tuple[bool, Optional[np.ndarray]]:
        target = step.get("target")
        radius = step.get("radius", 12)
        timeout = step.get("timeout", 0)
        if "start_frame" not in step:
            step["start_frame"] = self.frame_count
        if timeout and self.frame_count - step["start_frame"] >= timeout:
            print(f"[CLEARER] Nav timeout: {step.get('name')}")
            self.startup_index += 1
            self.navigator.path = []
            return True, make_action()
        pos = self.navigator.current_pos
        if target and abs(target.x - pos.x) <= radius and abs(target.y - pos.y) <= radius:
            print(f"[CLEARER] Nav done: {step.get('name')}")
            self.startup_index += 1
            self.navigator.path = []
            return True, make_action()
        if self.navigator.stasis > self.max_stasis:
            if self.navigator.path:
                self.pathfinder.temp_blocked.add(self.navigator.path[0])
            self.navigator.path = []
            self.navigator.stasis = 0
        if target and not self.navigator.path:
            target_tile = (target.x // TILE_SIZE, target.y // TILE_SIZE)
            approach = self.pathfinder.find_approach(ram, target_tile, pos)
            if not approach:
                approach = self.pathfinder.find_nearest_walkable(ram, target_tile, max_radius=4)
            if approach:
                path = self.pathfinder.find_path(ram, self.navigator.current_tile, approach)
                if path:
                    self.navigator.path = path
        action = self.navigator.follow_path(ram)
        return True, action if action is not None else make_action()

    def _emit_action(self, action: np.ndarray, src: str) -> np.ndarray:
        if self.suppress_move_frames > 0:
            self.suppress_move_frames -= 1
            # Strip d-pad on the Y swing so the farmer stays planted. A direction-only
            # face tap still passes so the sprite turns before the tool comes out.
            if action[1] == 1:
                action = action.copy()
                action[4:8] = 0
                src = f"{src}+suppress"
        if os.getenv("ACTION_DEBUG") == "1":
            buttons = action_to_names(action)
            noisy = buttons != "none" or (
                os.getenv("ACTION_DEBUG_VERBOSE") == "1" and self.frame_count % 30 == 0
            )
            if noisy:
                print(f"[ACTION] frame={self.frame_count} state={self.state} src={src} buttons={buttons}")
        return action

    def _should_lift(self, target: Target) -> bool:
        if not target.is_liftable:
            return False
        if target.debris_type == DebrisType.WEED:
            return self.prefer_lift_for_weeds
        if target.debris_type == DebrisType.STONE:
            return self.prefer_lift_for_stones
        return target.debris_type == DebrisType.FENCE

    def _face_dir(self, player: Tile, target: Tile) -> str:
        dx, dy = target[0] - player[0], target[1] - player[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _stamina(self, ram: np.ndarray) -> Stamina:
        return Stamina.from_ram(ram)

    def _can_afford_target(self, ram: np.ndarray, target: Target) -> bool:
        return self._stamina(ram).can_afford_clear(
            target.required_hits, lifting=self._should_lift(target)
        )

    def _try_adjacent_opportunity(self, ram: np.ndarray, player_tile: Tile) -> Optional[str]:
        """Clear priority debris that is already cardinally adjacent."""
        best: Optional[Target] = None
        best_rank: Optional[int] = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = player_tile[0] + dx, player_tile[1] + dy
            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH):
                continue
            snapped = snap_debris_anchor(ram, nx, ny, get_tile_at(ram, nx, ny))
            if snapped is None:
                continue
            nx, ny, tile_id, debris = snapped
            if ((nx, ny), player_tile) in self.failed_approaches:
                continue
            if debris not in CLEARABLE_DEBRIS_TYPES or (nx, ny) in self.failed_tiles:
                continue
            try:
                rank = self.priority.index(debris)
            except ValueError:
                continue
            candidate = Target(
                tile=(nx, ny),
                pos=Point(nx * TILE_SIZE + 8, ny * TILE_SIZE + 8),
                debris_type=debris,
                tile_id=tile_id,
            )
            if not self._can_afford_target(ram, candidate):
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = candidate
        if best is None:
            return None
        self.current_target = best
        self.approach_tile = player_tile
        self.navigator.path = []
        self.navigator.stasis = 0
        self.target_hits = 0
        self.clearing_start_frame = 0
        print(f"[CLEARER] Adjacent {best.debris_type.name} at {best.tile} -> clear now")
        return "clearing"

    def _step_off_stale(self, ram: np.ndarray) -> bool:
        """Hold west/NW toward (25,28) until shed-door 0xFF loads a8/a1."""
        if not needs_shed_door_step_off(ram):
            return False
        if self.action_queue:
            return True
        self.action_queue.extend(shed_door_step_off_actions())
        return True

    def _handle_scanning(self, ram: np.ndarray) -> Optional[str]:
        stam = self._stamina(ram)
        if stam < 1:
            self.stamina_exhausted = True
            print("[CLEARER] Stamina empty; stopping clear")
            return "complete"
        if self._step_off_stale(ram):
            return None
        scan_bounds = self._active_bounds()
        scan_types = set(self.priority) if self.priority else set(CLEARABLE_DEBRIS_TYPES)
        if self.quota:
            remaining = unmet_debris_types(
                self.quota_start_counts, count_debris(ram, scan_bounds), self.quota
            )
            if remaining is not None:
                if not farm_map_loaded(ram):
                    remaining = scan_types
                scan_types &= remaining
                if not scan_types:
                    return "complete"
        scanned = self.scanner.scan(ram, scan_bounds, types=scan_types)
        targets = [t for t in scanned if self._can_afford_target(ram, t)]
        if not targets:
            if scanned:
                self.stamina_exhausted = True
                print(
                    f"[CLEARER] Stamina low ({stam}); skip multi-hit "
                    f"(need {stam.cost_to_clear(6)} for 8-swing rock)"
                )
            return "complete"
        player_tile = self.navigator.current_tile
        opportunity = self._try_adjacent_opportunity(ram, player_tile)
        if opportunity:
            return opportunity
        if self._locked_bounds is None:
            xs = [t.tile[0] for t in targets]
            ys = [t.tile[1] for t in targets]
            self._work_bounds = (
                max(2, min(xs)), max(2, min(ys)), min(61, max(xs)), min(61, max(ys))
            )
        counts: Dict[DebrisType, int] = {}
        for target in targets:
            counts[target.debris_type] = counts.get(target.debris_type, 0) + 1
        new_phase = next((dt for dt in self.priority if counts.get(dt, 0) > 0), None)
        if new_phase != self.current_phase:
            if new_phase:
                print(f"[CLEARER] Phase: {new_phase.name}")
            self.current_phase = new_phase
        if not self.current_phase:
            return "complete"
        phase_targets = [
            t for t in targets
            if t.debris_type == self.current_phase and t.tile not in self.failed_tiles
        ]
        phase_targets = sort_targets_cluster(phase_targets, self.navigator.current_pos)
        chosen = choose_clear_target(self, ram, phase_targets)
        if chosen is not None:
            target, approach, path = chosen
            self.scan_miss_streak = 0
            self.current_target = target
            self.approach_tile = approach
            self.navigator.path = path
            self.navigator.stasis = 0
            start_progress_watch(self, approach)
            self.target_hits = 0
            self.clearing_start_frame = 0
            tool = target.required_tool.name if target.required_tool else "HANDS"
            print(f"[CLEARER] Target: {target.debris_type.name} at {target.tile} ({tool})")
            return "navigating"
        self.scan_miss_streak += 1
        if self.scan_miss_streak >= self.max_scan_misses:
            phase = self.current_phase.name if self.current_phase else "debris"
            print(
                f"[CLEARER] No reachable {phase} after {self.scan_miss_streak} scans; "
                f"stopping with cleared={self.cleared_count}"
            )
            return "complete"
        return None

    def _queue_held_toss(self, ram, player, held: int, *, face: str = "down", origin=None) -> None:
        blocked = {tuple(origin)} if origin is not None else set()
        if origin is not None or needs_south_fence_drop(player, held):
            self.action_queue.clear()
            self._toss_skill = start_fence_jump_skill(
                frame=self.frame_count, ram=ram, blocked=blocked
            )
            return
        self.action_queue.extend(in_place_toss_actions(face=face))

    def _replan_nav_hop(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target or not self.approach_tile:
            return "scanning"
        path = self.pathfinder.find_path(
            ram, self.navigator.current_tile, self.approach_tile, max_steps=VIEWPORT_HOP_TILES
        )
        if path is None:
            self.failed_approaches.add((self.current_target.tile, self.approach_tile))
            self.current_target = None
            self.approach_tile = None
            return "scanning"
        self.navigator.path = path
        self.navigator.stasis = 0
        return None

    def _handle_navigating(self, ram: np.ndarray) -> Optional[str]:
        from harvest.tasks.farm_ops import handle_navigating
        return handle_navigating(self, ram)

    def _clear_gone(self, ram: np.ndarray, origin: Tile) -> str:
        self._pending_lift_verify = None
        held = int(read_held_item(ram))
        if held:
            self._pending_toss_origin = origin
            self._queue_held_toss(ram, self.navigator.current_tile, held, origin=origin)
        elif origin not in self.tiles_cleared:
            self.tiles_cleared.add(origin)
            self.cleared_count += 1
            self.pathfinder.no_go_tiles.discard(origin)
        self.current_target = None
        self.clearing_start_frame = 0
        return "scanning"

    def _handle_clearing(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target:
            return "scanning"
        if self.clearing_start_frame == 0:
            self.clearing_start_frame = self.frame_count
            self.action_queue.clear()
            self.task_queue.clear()
            self.navigator.path = []
        # Timeout is a stuck approach, not a planted multi-hit. Walking mid-hammer
        # STZs $096D, so stay until the tile breaks or the stand misses out.
        if (
            self.frame_count - self.clearing_start_frame > 600
            and not self.tool_clear_is_planted()
        ):
            print(f"[CLEARER] Clearing timeout at {self.current_target.tile}, moving on")
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"
        current_tile_id = get_tile_at(ram, *self.current_target.tile)
        if current_tile_id != self.current_target.tile_id:
            new_debris = TILE_TO_DEBRIS.get(current_tile_id)
            if new_debris is None:
                return self._clear_gone(ram, self.current_target.tile)
            if new_debris != self.current_target.debris_type:
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=new_debris,
                tile_id=current_tile_id,
            )
        player = self.navigator.current_tile
        target = self.current_target.tile
        # A 2x2 anchor can be two or three steps from a valid lower/right stand.
        footprint = set(self.current_target.footprint)
        adjacent = player not in footprint and any(tile_dist(player, cell) == 1 for cell in footprint)
        if not adjacent:
            self.clearing_start_frame = 0
            return "navigating"
        if self.action_queue:
            return None
        if self._pending_lift_verify is not None:
            verify_tile = self._pending_lift_verify
            self._pending_lift_verify = None
            lift_key = (verify_tile[0], verify_tile[1], int(get_tile_at(ram, *verify_tile)))
            if evaluate_lift_verify(ram, verify_tile) == "blocked":
                attempts = self.tile_attempts.get(lift_key, 0) + 1
                self.tile_attempts[lift_key] = attempts
                print(f"[CLEARER] Lift did not clear {verify_tile} (attempt {attempts}/2)")
                if attempts >= 2:
                    self.failed_tiles.add(verify_tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1 or self.navigator.stasis < 6:
            return None
        # Re-center only before the first face. A walk mid-hammer STZs $096D.
        if self.approach_tile and not self.tool_clear_is_planted():
            center_action = self.navigator.center_on_tile(self.approach_tile, tolerance=2)
            if center_action is not None:
                self.action_queue.append(center_action)
                return None
        if self._should_lift(self.current_target):
            return self._lift_current(ram, player, target)
        return self.handle_tool_clear(ram, player=player, target=target)

    def _lift_current(self, ram: np.ndarray, player: Tile, target: Tile) -> Optional[str]:
        held = read_held_item(ram)
        if held:
            self._toss_before_lift += 1
            if self._toss_before_lift > 3:
                print(f"[CLEARER] Still held=0x{held:02X}; skip lift at {target}")
                self.failed_tiles.add(target)
                self.current_target = None
                self.clearing_start_frame = 0
                self._toss_before_lift = 0
                return "scanning"
            print(f"[CLEARER] Toss held=0x{held:02X} before next lift")
            self._queue_held_toss(ram, player, held, face="down")
            return None
        self._toss_before_lift = 0
        lift_key = (target[0], target[1], int(self.current_target.tile_id))
        attempts = self.tile_attempts.get(lift_key, 0)
        if attempts >= 2 or target in self.failed_tiles:
            print(f"[CLEARER] Skipping lift thrash at {target} ({self.current_target.debris_type.name})")
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

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[str]:
        del ram
        if not self.searching_tool:
            return "clearing"
        self.tool_search_frames += 1
        if self.tool_manager.current == self.searching_tool:
            print(f"[CLEARER] Found {self.searching_tool.name}")
            self.searching_tool = None
            return "clearing"
        self.tool_manager.record()
        if self.tool_manager.cycle_complete() or self.tool_search_frames > 300:
            print(f"[CLEARER] Can't find {self.searching_tool.name}")
            frames = None
            if not self.tools_missing:
                frames = self._load_task(f"get_{self.searching_tool.name.lower()}")
            if frames:
                print(f"[CLEARER] Running get_{self.searching_tool.name.lower()}")
                self.task_queue.extend(frames)
                self.searching_tool = None
                self.tool_manager.start_search()
                return None
            if self.current_target:
                self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.searching_tool = None
            self.clearing_start_frame = 0
            return "scanning"
        self.action_queue.extend(cycle_tool())
        return None

    def tick(self, ram: np.ndarray) -> Optional[np.ndarray]:
        """One clearer frame. ``step`` is the Task; this is the field advance."""
        self.frame_count += 1
        self.navigator.update(ram)
        self.tool_manager.update(ram)
        if self.frame_count % self.debug_interval == 0:
            stamina = ram[ADDR_STAMINA] if ADDR_STAMINA < len(ram) else 0
            targets = self.scanner.scan(ram, self._active_bounds() or self.farm_bounds)
            print(
                f"[CLEARER] Debug @ {self.frame_count}f pos={self.navigator.current_pos} "
                f"tool=0x{self.tool_manager.current:02X} stamina={stamina} state={self.state} "
                f"targets={len(targets)} cleared={self.cleared_count} failed={len(self.failed_tiles)}"
            )
        running, action = self._run_startup(ram)
        if running:
            return action if action is not None else make_action()
        if self.task_queue:
            return self._emit_action(self.task_queue.popleft(), "task")
        prev_toss = self._toss_skill
        self._toss_skill, toss_action = step_fence_jump_skill(
            self._toss_skill, ram, frame=self.frame_count
        )
        if toss_action is not None:
            return self._emit_action(toss_action, "fence_jump")
        if prev_toss is not None and self._pending_toss_origin is not None:
            self._credit_toss(ram)
        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        on_stale = (
            self.navigator.current_tile is not None
            and int(get_tile_at(ram, *self.navigator.current_tile)) in STALE_TILE_IDS
        )
        if input_lock != 1 and not on_stale:
            unlock = make_action(a=True) if self.frame_count % 2 == 0 else make_action(b=True)
            return self._emit_action(unlock, "unlock")
        if self.state == "complete":
            return None
        handlers = {
            "scanning": self._handle_scanning,
            "navigating": self._handle_navigating,
            "clearing": self._handle_clearing,
            "tool_switch": self._handle_tool_switch,
        }
        if self.state in handlers:
            nxt = handlers[self.state](ram)
            if nxt == "complete":
                self.state = "complete"
                return None
            if nxt:
                self.state = nxt
        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")
        return self._emit_action(make_action(), "idle")

    def _credit_toss(self, ram: np.ndarray) -> None:
        origin = self._pending_toss_origin
        self._pending_toss_origin = None
        verdict = evaluate_lift_verify(ram, origin)
        if verdict == "cleared":
            if origin not in self.tiles_cleared:
                self.tiles_cleared.add(origin)
                self.cleared_count += 1
            self.pathfinder.no_go_tiles.discard(origin)
            return
        key = (origin[0], origin[1], int(get_tile_at(ram, *origin)))
        attempts = self.tile_attempts.get(key, 0) + 1
        self.tile_attempts[key] = attempts
        print(f"[CLEARER] Toss did not free {origin} ({verdict}) (attempt {attempts}/2)")
        if attempts >= 2:
            self.failed_tiles.add(origin)


def _install_clear_handoff() -> None:
    from harvest.tasks import farm_clear_quota as policy
    names = (
        "progress_snapshot", "_scan_bounds", "_remaining_debris", "_player_tile",
        "_in_pocket", "_pocket_tiles_ready", "_pocket_is_ready", "_plot_cells_to_clear",
        "_plot_scan_bounds", "_lock_clearer_to_plot", "_plant_notch_is_clear",
        "_pocket_stand_px", "_make_pocket_approach", "_uses_pocket_approach",
        "_step_pocket_approach", "can_start", "_on_farm", "_exit_stand_px",
        "_queue_south_exit_staging", "_quota_met", "_complete_status", "_step_exit_nav",
        "_maybe_stage_then_success", "_finish_or_drop", "_running", "_finish_idle", "step",
    )
    for name in names:
        setattr(FarmClearer, name, getattr(policy, name))
    FarmClearer.progress_text = property(policy.progress_text)


_install_clear_handoff()

__all__ = [
    "DEFAULT_PRIORITY", "FACE_SETTLE_FRAMES", "FarmClearer", "MAX_OBSERVE_EXTRA",
    "MAX_STAND_MISSES", "POST_SWING_OBSERVE_FRAMES", "Target", "TileScanner",
    "ToolManager", "choose_clear_target", "cycle_tool", "parse_priority_list", "use_tool",
    "use_tool_facing",
]
