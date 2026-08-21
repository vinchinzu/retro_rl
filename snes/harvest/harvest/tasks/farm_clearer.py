"""
Farm clearing module - Phase-based debris clearing with tool management.

Navigation primitives live in ``harvest.tasks.nav``; tile-scan / tool helpers
live in ``harvest.tasks.farm_ops``. Both are re-exported here for backward
compatibility.
"""

from typing import Optional, List, Dict, Tuple, Set
from collections import deque
import os
import json

import numpy as np

from harvest.core.animal_status import read_held_item
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_STAMINA,
    CLEARABLE_DEBRIS_TYPES,
    MAP_WIDTH,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
)

# Nav primitives — re-exported so existing importers keep working.
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

# Tile-scan / tool helpers — re-exported for backward compatibility.
from harvest.tasks.farm_ops import (  # noqa: F401
    Target,
    TileScanner,
    ToolManager,
    action_to_names,
    cycle_tool,
    drop_unarmed_debris,
    snap_debris_anchor,
    use_tool,
    use_tool_facing,
)
from harvest.tasks.farm_toss import (
    FenceJumpTossSkill,
    in_place_toss_actions,
    needs_south_fence_drop,
    start_fence_jump_skill,
    step_fence_jump_skill,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Hard obstacles first so pathing opens up, then cheap lifts.
DEFAULT_PRIORITY: List[DebrisType] = [
    DebrisType.ROCK,
    DebrisType.STUMP,
    DebrisType.STONE,
    DebrisType.WEED,
]

# Hammer/axe hits cost 2 stamina; stop before a multi-hit cannot finish.
MIN_CLEAR_STAMINA = 4


# =============================================================================
# FARM CLEARER
# =============================================================================

class FarmClearer:
    """Phase-based farm clearing: rock → stump → stone → weed."""

    def __init__(self, priority: Optional[List[DebrisType]] = None):
        self.priority = priority or DEFAULT_PRIORITY.copy()

        self.scanner = TileScanner()
        self.pathfinder = Pathfinder(self.scanner)
        self.navigator = Navigator(self.pathfinder)
        self.tool_manager = ToolManager()

        self.current_phase: Optional[DebrisType] = None
        self.current_target: Optional[Target] = None
        self.approach_tile: Optional[Tuple[int, int]] = None
        self.action_queue: deque = deque()
        self.state = "scanning"

        self.failed_tiles: Set[Tuple[int, int]] = set()
        self.cleared_count = 0
        self.tiles_cleared: Set[Tuple[int, int]] = set()
        self.tile_attempts: Dict[Tuple[int, int, int], int] = {}
        self.frame_count = 0
        self.farm_bounds: Optional[Tuple[int, int, int, int]] = None
        self._locked_bounds: Optional[Tuple[int, int, int, int]] = None

        self.prefer_lift_for_weeds = True
        self.prefer_lift_for_stones = False
        self.max_stasis = 120
        self.debug_interval = 300
        self.min_stamina = MIN_CLEAR_STAMINA
        self.stamina_exhausted = False
        self.tools_missing = False
        self.scan_miss_streak = 0
        self.max_scan_misses = 90

        self.searching_tool: Optional[Tool] = None
        self.tool_search_frames = 0

        self.startup_tasks: List[Dict] = []
        self.startup_index = 0
        self.startup_done = False
        self.task_queue: deque = deque()
        self.tasks_dir: Optional[str] = None

        self.target_hits = 0
        self.clearing_start_frame = 0
        self.suppress_move_frames = 0
        self._pending_lift_verify: Optional[Tuple[int, int]] = None
        self._toss_before_lift = 0
        self._toss_skill: Optional[FenceJumpTossSkill] = None
        self._init_no_go()

    def _init_no_go(self):
        default = "9,26;9,27;9,28;11,26;11,27;11,28;8,12;9,12;10,12"
        for entry in os.getenv("NO_GO_TILES", default).replace("|", ";").split(";"):
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    self.pathfinder.no_go_tiles.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.farm_bounds is not None:
            self._locked_bounds = tuple(self.farm_bounds)

    def add_startup_task(self, task_type: str, **kwargs):
        self.startup_tasks.append({"type": task_type, **kwargs})

    def _load_task(self, name: str) -> Optional[List[np.ndarray]]:
        if not self.tasks_dir:
            return None
        path = os.path.join(self.tasks_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return [np.array(frame, dtype=np.int32) for frame in data.get("frames", [])]

    def _emit_action(self, action: np.ndarray, src: str) -> np.ndarray:
        if self.suppress_move_frames > 0:
            self.suppress_move_frames -= 1
            # Strip directional inputs on tool-swing frames to prevent drift.
            # Direction-only frames (the initial face tap) pass through so the
            # character actually turns toward the target before swinging.
            if action[1] == 1:  # Y button pressed (tool use)
                action = action.copy()
                action[4:8] = 0
                src = f"{src}+suppress"
        if os.getenv("ACTION_DEBUG") == "1":
            buttons = action_to_names(action)
            if buttons != "none" or os.getenv("ACTION_DEBUG_VERBOSE") == "1" and self.frame_count % 30 == 0:
                print(f"[ACTION] frame={self.frame_count} state={self.state} src={src} buttons={buttons}")
        return action

    def _requested_startup_tools(self) -> Set[int]:
        wanted: Set[int] = set()
        for step in self.startup_tasks:
            if step.get("type") != "task":
                continue
            name = str(step.get("name", ""))
            mapping = {
                "get_hammer": int(Tool.HAMMER),
                "get_axe": int(Tool.AXE),
                "get_sickle": int(Tool.SICKLE),
                "get_hoe": int(Tool.HOE),
            }
            tool_id = mapping.get(name)
            if tool_id is not None:
                wanted.add(tool_id)
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
        missing = sorted(self._requested_startup_tools() - have)

        if DebrisType.ROCK in self.priority and not self.tool_manager.has(
            int(Tool.HAMMER)
        ):
            missing = sorted(set(missing) | {int(Tool.HAMMER)})
        if DebrisType.STUMP in self.priority and not self.tool_manager.has(
            int(Tool.AXE)
        ):
            missing = sorted(set(missing) | {int(Tool.AXE)})

        if missing:
            self.tools_missing = True
            self._enable_lift_only_mode(missing)
        else:
            self.tools_missing = False

    def _run_startup(self, ram: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if self.startup_done:
            return False, None

        # One-time tool inventory scan at the very beginning
        if not hasattr(self, '_tool_scan_done'):
            self._tool_scan_done = False
            self._tool_scan_frames = 0
            self.tool_manager.start_search()

        if not self._tool_scan_done:
            self._tool_scan_frames += 1
            self.tool_manager.record()

            # Scan complete after one full cycle or timeout
            if self.tool_manager.cycle_complete() or self._tool_scan_frames > 60:
                self._tool_scan_done = True
                tools_found = [f"0x{t:02X}" for t in sorted(self.tool_manager.seen)]
                print(f"[CLEARER] Tool inventory: {', '.join(tools_found)}")
            else:
                # Continue cycling
                if self._tool_scan_frames % 6 == 0:  # Cycle every 6 frames
                    self.action_queue.extend(cycle_tool())
                return True, self.action_queue.popleft() if self.action_queue else make_action()

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

            # Check if we should skip tool acquisition tasks using pre-scanned inventory
            if task_name in ("get_hammer", "get_axe", "get_sickle", "get_hoe"):
                tool_map = {
                    "get_hammer": Tool.HAMMER,
                    "get_axe": Tool.AXE,
                    "get_sickle": Tool.SICKLE,
                    "get_hoe": Tool.HOE,
                }
                required_tool = tool_map.get(task_name)

                if required_tool and self.tool_manager.has(int(required_tool)):
                    print(
                        f"[CLEARER] Skipping {task_name} "
                        f"(already have {required_tool.name})"
                    )
                    self.startup_index += 1
                    return True, make_action()

            # Execute the task
            frames = self._load_task(task_name)
            if frames:
                print(f"[CLEARER] Task: {task_name} ({len(frames)} frames)")
                self.task_queue.extend(frames)
            else:
                print(f"[CLEARER] Task not found: {task_name}")
            self.startup_index += 1
            return True, self.task_queue.popleft() if self.task_queue else make_action()

        elif step_type == "nav":
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

            if target and abs(target.x - self.navigator.current_pos.x) <= radius and abs(target.y - self.navigator.current_pos.y) <= radius:
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
                approach = self.pathfinder.find_approach(ram, target_tile, self.navigator.current_pos)
                if not approach:
                    approach = self.pathfinder.find_nearest_walkable(ram, target_tile, max_radius=4)
                if approach:
                    path = self.pathfinder.find_path(ram, self.navigator.current_tile, approach)
                    if path:
                        self.navigator.path = path

            action = self.navigator.follow_path(ram)
            return True, action if action is not None else make_action()

        self.startup_index += 1
        return True, make_action()

    def _should_lift(self, target: Target) -> bool:
        if not target.is_liftable:
            return False
        if target.debris_type == DebrisType.WEED:
            return self.prefer_lift_for_weeds
        if target.debris_type == DebrisType.STONE:
            return self.prefer_lift_for_stones
        if target.debris_type == DebrisType.FENCE:
            return True
        return False

    def _face_dir(self, player: Tuple[int, int], target: Tuple[int, int]) -> str:
        dx, dy = target[0] - player[0], target[1] - player[1]
        return 'right' if abs(dx) >= abs(dy) and dx > 0 else 'left' if abs(dx) >= abs(dy) else 'down' if dy > 0 else 'up'

    def _stamina(self, ram: np.ndarray) -> int:
        if ADDR_STAMINA >= len(ram):
            return 0
        return int(ram[ADDR_STAMINA])

    def _can_afford_target(self, ram: np.ndarray, target: Target) -> bool:
        return self._stamina(ram) >= target.stamina_to_clear(
            lifting=self._should_lift(target)
        )

    def _sort_targets_cluster(
        self, targets: List[Target], player_pos: Point
    ) -> List[Target]:
        """Nearest-neighbor with north bias so day-plan clear stays returnable.

        Prefer targets north of / near the y=31 fence; deep-south debris
        (y>38) is a softlock trap for return_home after water days (rr-5in).
        """
        remaining = list(targets)
        ordered: List[Target] = []
        cur = player_pos
        row_dir = 1
        while remaining:
            remaining.sort(
                key=lambda t: (
                    2 if t.tile[1] > 40 else (1 if t.tile[1] > 32 else 0),
                    manhattan(t.pos, cur),
                    t.tile[1],
                    t.tile[0] * row_dir,
                )
            )
            nxt = remaining.pop(0)
            ordered.append(nxt)
            if ordered and len(ordered) >= 2:
                prev_y = ordered[-2].tile[1]
                if nxt.tile[1] != prev_y:
                    row_dir *= -1
            cur = nxt.pos
        return ordered

    def _try_adjacent_opportunity(
        self, ram: np.ndarray, player_tile: Tuple[int, int]
    ) -> Optional[str]:
        """Clear any priority debris already adjacent to the player."""
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
        print(
            f"[CLEARER] Adjacent {best.debris_type.name} at {best.tile} "
            "-> clear now"
        )
        return "clearing"

    def _handle_scanning(self, ram: np.ndarray) -> Optional[str]:
        stam = self._stamina(ram)
        if stam < 1:
            self.stamina_exhausted = True
            print("[CLEARER] Stamina empty; stopping clear")
            return "complete"

        scan_bounds = self._locked_bounds or self.farm_bounds
        scan_types = set(self.priority) if self.priority else set(CLEARABLE_DEBRIS_TYPES)
        scanned = self.scanner.scan(ram, scan_bounds, types=scan_types)
        targets = [t for t in scanned if self._can_afford_target(ram, t)]
        if not targets:
            if scanned:
                self.stamina_exhausted = True
                print(f"[CLEARER] Stamina low ({stam}); stopping clear")
            return "complete"

        player_tile = self.navigator.current_tile
        opportunity = self._try_adjacent_opportunity(ram, player_tile)
        if opportunity:
            return opportunity

        if self._locked_bounds is None:
            xs = [t.tile[0] for t in targets]
            ys = [t.tile[1] for t in targets]
            self.farm_bounds = (
                max(2, min(xs)),
                max(2, min(ys)),
                min(61, max(xs)),
                min(61, max(ys)),
            )

        counts: Dict[DebrisType, int] = {}
        for t in targets:
            counts[t.debris_type] = counts.get(t.debris_type, 0) + 1

        new_phase = None
        for dt in self.priority:
            if counts.get(dt, 0) > 0:
                new_phase = dt
                break

        if new_phase != self.current_phase:
            if new_phase:
                print(f"[CLEARER] Phase: {new_phase.name}")
            self.current_phase = new_phase

        if not self.current_phase:
            return "complete"

        phase_targets = [
            t
            for t in targets
            if t.debris_type == self.current_phase
            and t.tile not in self.failed_tiles
        ]
        phase_targets = self._sort_targets_cluster(
            phase_targets, self.navigator.current_pos
        )

        for target in phase_targets:
            approach = self.pathfinder.find_approach(
                ram,
                target.tile,
                self.navigator.current_pos,
                footprint=target.footprint,
            )
            if approach:
                path = self.pathfinder.find_path(
                    ram,
                    self.navigator.current_tile,
                    approach,
                    max_steps=VIEWPORT_HOP_TILES,
                )
                if path is not None:
                    self.scan_miss_streak = 0
                    self.current_target = target
                    self.approach_tile = approach
                    self.navigator.path = path
                    self.navigator.stasis = 0
                    self.target_hits = 0
                    self.clearing_start_frame = 0
                    tool = (
                        target.required_tool.name
                        if target.required_tool
                        else "HANDS"
                    )
                    print(
                        f"[CLEARER] Target: {target.debris_type.name} "
                        f"at {target.tile} ({tool})"
                    )
                    return "navigating"

        self.scan_miss_streak += 1
        if self.scan_miss_streak >= self.max_scan_misses:
            print(
                f"[CLEARER] No reachable {self.current_phase.name if self.current_phase else 'debris'} "
                f"after {self.scan_miss_streak} scans; stopping with "
                f"cleared={self.cleared_count}"
            )
            return "complete"
        return None

    def _queue_held_toss(self, ram, player, held: int, *, face: str = "down") -> None:
        if needs_south_fence_drop(player, held):
            self.action_queue.clear()
            self._toss_skill = start_fence_jump_skill(frame=self.frame_count, ram=ram)
            return
        self.action_queue.extend(in_place_toss_actions(face=face))

    def _replan_nav_hop(self, ram: np.ndarray) -> Optional[str]:
        """Plan a viewport-limited hop toward the current approach tile."""
        if not self.current_target or not self.approach_tile:
            return "scanning"
        path = self.pathfinder.find_path(
            ram,
            self.navigator.current_tile,
            self.approach_tile,
            max_steps=VIEWPORT_HOP_TILES,
        )
        if path is None:
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            return "scanning"
        self.navigator.path = path
        self.navigator.stasis = 0
        return None

    def _handle_navigating(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target or not self.approach_tile:
            return "scanning"

        live_id = get_tile_at(ram, *self.current_target.tile)
        live_debris = TILE_TO_DEBRIS.get(live_id)
        if live_debris is None:
            self.current_target = None
            return "scanning"
        if live_debris != self.current_target.debris_type:
            self.current_target = None
            return "scanning"
        if live_id != self.current_target.tile_id:
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=live_debris,
                tile_id=live_id,
            )

        if self.navigator.current_tile == self.approach_tile:
            return "clearing"

        if self.navigator.stasis > self.max_stasis:
            print(
                f"[NAV] Stuck at {self.navigator.current_tile}, "
                "trying alternate path"
            )
            if self.navigator.path:
                self.pathfinder.temp_blocked.add(self.navigator.path[0])
            self.navigator.path = []
            self.navigator.stasis = 0
            return self._replan_nav_hop(ram)

        action = self.navigator.follow_path(ram)
        if action is not None:
            self.action_queue.append(action)
            return None

        # Hop segment finished short of the approach — replan next hop.
        if self.navigator.current_tile != self.approach_tile:
            return self._replan_nav_hop(ram)
        return "clearing"

    def _handle_clearing(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target:
            return "scanning"

        # Track when we entered clearing state for timeout
        if self.clearing_start_frame == 0:
            self.clearing_start_frame = self.frame_count
            self.action_queue.clear()
            self.task_queue.clear()
            self.navigator.path = []

        # Timeout (600 frames: per-hit clearing with centering between each)
        if self.frame_count - self.clearing_start_frame > 600:
            print(f"[CLEARER] Clearing timeout at {self.current_target.tile}, moving on")
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        # Re-validate target tile.  Rocks change tile ID as they take damage;
        # keep hitting as long as the debris *type* is unchanged.
        current_tile_id = get_tile_at(ram, *self.current_target.tile)
        if current_tile_id != self.current_target.tile_id:
            new_debris = TILE_TO_DEBRIS.get(current_tile_id)
            if new_debris is None:
                # Tile fully cleared
                pos_key = self.current_target.tile
                if pos_key not in self.tiles_cleared:
                    self.tiles_cleared.add(pos_key)
                    self.cleared_count += 1
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            if new_debris != self.current_target.debris_type:
                # Changed to a different debris type, rescan
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            # Same debris type, different visual (rock taking damage) — continue
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=new_debris,
                tile_id=current_tile_id,
            )

        player = self.navigator.current_tile
        target = self.current_target.tile

        if tile_dist(player, target) > 1:
            return "navigating"

        # Wait for any queued actions (current hit animation) to finish
        if self.action_queue:
            return None

        # Finish verifying a lift after the queued A presses drain.
        if self._pending_lift_verify is not None:
            verify_tile = self._pending_lift_verify
            self._pending_lift_verify = None
            lift_key = (
                verify_tile[0],
                verify_tile[1],
                int(get_tile_at(ram, *verify_tile)),
            )
            if TILE_TO_DEBRIS.get(get_tile_at(ram, *verify_tile)) is None:
                if verify_tile not in self.tiles_cleared:
                    self.tiles_cleared.add(verify_tile)
                    self.cleared_count += 1
                held = read_held_item(ram)
                face = self._face_dir(verify_tile, player)
                if face not in {"up", "down", "left", "right"}:
                    face = "down"
                self._queue_held_toss(ram, player, held, face=face)
                # Do not re-target this cell this clear pass — toss often
                # re-deposits the same rock one tile over / back.
                self.failed_tiles.add(verify_tile)
            else:
                attempts = self.tile_attempts.get(lift_key, 0) + 1
                self.tile_attempts[lift_key] = attempts
                print(
                    f"[CLEARER] Lift did not clear {verify_tile} "
                    f"(attempt {attempts}/2)"
                )
                if attempts >= 2:
                    self.failed_tiles.add(verify_tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        # Wait until inputs are accepted and player is stationary
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1 or self.navigator.stasis < 6:
            return None

        # Re-center on approach tile before every hit to correct animation drift
        if self.approach_tile:
            center_action = self.navigator.center_on_tile(
                self.approach_tile, tolerance=2
            )
            if center_action is not None:
                self.action_queue.append(center_action)
                return None

        # Lift check — cap attempts so unliftable / re-deposited stones stop thrashing.
        if self._should_lift(self.current_target):
            held = read_held_item(ram)
            if held:
                self._toss_before_lift += 1
                if self._toss_before_lift > 3:
                    print(
                        f"[CLEARER] Still held=0x{held:02X}; "
                        f"skip lift at {target}"
                    )
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
            self.action_queue.extend(
                [make_action(**{direction: True}) for _ in range(3)]
            )
            self.action_queue.extend([make_action() for _ in range(4)])
            self.action_queue.extend([make_action(a=True) for _ in range(18)])
            self.action_queue.extend([make_action() for _ in range(20)])
            self._pending_lift_verify = target
            return None

        # Tool check
        tool = self.current_target.required_tool
        if tool is None:
            self.failed_tiles.add(target)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        if self.tool_manager.current != tool:
            print(
                f"[CLEARER] Need {tool.name}, "
                f"have 0x{self.tool_manager.current:02X}"
            )
            self.searching_tool = tool
            self.tool_manager.start_search()
            self.tool_search_frames = 0
            return "tool_switch"

        # Do not start a 6-hit (or any tool swing we cannot finish). Lifts
        # already returned above and may continue at stamina 1–3.
        if self.target_hits == 0 and not self._can_afford_target(
            ram, self.current_target
        ):
            self.stamina_exhausted = True
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"
        if self.target_hits > 0 and self._stamina(ram) < 2:
            self.stamina_exhausted = True
            self.current_target = None
            self.clearing_start_frame = 0
            return "complete"

        # First hit: attempt tracking and logging
        if self.target_hits == 0:
            tile_key = (target[0], target[1], self.current_target.tile_id)
            attempts = self.tile_attempts.get(tile_key, 0)
            if attempts >= 3:
                print(
                    f"[CLEARER] Giving up on "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} "
                    "(3 failed attempts)"
                )
                self.failed_tiles.add(target)
                self.current_target = None
                return "scanning"
            self.tile_attempts[tile_key] = attempts + 1
            direction = self._face_dir(player, target)
            if attempts == 0:
                print(
                    f"[CLEARER] Clearing "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} from {player} "
                    f"facing {direction} "
                    f"({self.current_target.required_hits} hits)"
                )
            else:
                print(
                    f"[CLEARER] Re-targeting "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} "
                    f"(attempt {attempts + 1}/3)"
                )

        # Hits delivered — only count after the tile is actually gone.
        if self.target_hits >= self.current_target.required_hits:
            if TILE_TO_DEBRIS.get(current_tile_id) is None:
                pos_key = self.current_target.tile
                if pos_key not in self.tiles_cleared:
                    self.tiles_cleared.add(pos_key)
                    self.cleared_count += 1
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            # Still present after claimed hits — keep swinging a bit more,
            # then fail the tile.
            if self.target_hits >= self.current_target.required_hits + 3:
                print(
                    f"[CLEARER] Hits exhausted but tile remains at {target}"
                )
                self.failed_tiles.add(target)
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"

        # Queue a SINGLE hit: face → settle → swing → cooldown
        direction = self._face_dir(player, target)
        self.action_queue.append(make_action(**{direction: True}))
        self.action_queue.extend([make_action() for _ in range(8)])
        self.action_queue.extend(use_tool(frames=20, cooldown=20))
        self.target_hits += 1

        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[str]:
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
        self.frame_count += 1
        self.navigator.update(ram)
        self.tool_manager.update(ram)

        if self.frame_count % self.debug_interval == 0:
            stamina = ram[ADDR_STAMINA] if ADDR_STAMINA < len(ram) else 0
            targets = self.scanner.scan(ram, self.farm_bounds)
            print(
                f"[CLEARER] Debug @ {self.frame_count}f "
                f"pos={self.navigator.current_pos} "
                f"tool=0x{self.tool_manager.current:02X} "
                f"stamina={stamina} state={self.state} "
                f"targets={len(targets)} cleared={self.cleared_count} "
                f"failed={len(self.failed_tiles)}"
            )

        running, action = self._run_startup(ram)
        if running:
            return action if action is not None else make_action()

        if self.task_queue:
            return self._emit_action(self.task_queue.popleft(), "task")

        self._toss_skill, toss_action = step_fence_jump_skill(
            self._toss_skill, ram, frame=self.frame_count
        )
        if toss_action is not None:
            return self._emit_action(toss_action, "fence_jump")

        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")

        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1:
            action = (
                make_action(a=True)
                if self.frame_count % 2 == 0
                else make_action(b=True)
            )
            return self._emit_action(action, "unlock")

        if self.state == "complete":
            return None

        handlers = {
            "scanning": self._handle_scanning,
            "navigating": self._handle_navigating,
            "clearing": self._handle_clearing,
            "tool_switch": self._handle_tool_switch,
        }

        if self.state in handlers:
            next_state = handlers[self.state](ram)
            if next_state == "complete":
                self.state = "complete"
                return None
            if next_state:
                self.state = next_state

        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")

        return self._emit_action(make_action(), "idle")


# =============================================================================
# PRIORITY PARSING
# =============================================================================

DEBRIS_NAMES = {
    "weed": DebrisType.WEED, "weeds": DebrisType.WEED, "bush": DebrisType.WEED,
    "stone": DebrisType.STONE, "stones": DebrisType.STONE,
    "rock": DebrisType.ROCK, "rocks": DebrisType.ROCK,
    "stump": DebrisType.STUMP, "stumps": DebrisType.STUMP,
}


def parse_priority_list(raw: Optional[str], priority_only: bool = False) -> List[DebrisType]:
    if not raw:
        return list(DEFAULT_PRIORITY)

    parsed = []
    for name in raw.split(","):
        debris = DEBRIS_NAMES.get(name.strip().lower())
        if debris and debris not in parsed:
            parsed.append(debris)

    if not parsed:
        return list(DEFAULT_PRIORITY)

    if not priority_only:
        for dt in DEFAULT_PRIORITY:
            if dt not in parsed:
                parsed.append(dt)

    return parsed
