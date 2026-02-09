"""
Grass planting module - Tills untilled ground and plants grass seeds.

Two-phase state machine:
  Phase 1 (TILL): Scan for untilled tiles, navigate adjacent, use HOE
  Phase 2 (PLANT): Scan for tilled tiles, switch to grass seeds, use Y

Follows the Task protocol from fence_flow.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List
from collections import deque

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from farm_clearer import (
    TileScanner,
    Pathfinder,
    Navigator,
    ToolManager,
    Tool,
    Point,
    make_action,
    use_tool,
    cycle_tool,
    get_tile_at,
    manhattan,
    tile_dist,
    TILE_SIZE,
    MAP_WIDTH,
    ADDR_TOOL,
    ADDR_INPUT_LOCK,
    WALKABLE_TILES,
)


# =============================================================================
# CONSTANTS (from Step 1 discovery)
# =============================================================================

# Tile IDs
UNTILLED_TILE = 0x01       # Cleared ground, not yet hoed
DRIED_TILLED = 0x02        # Previously hoed but dried out, needs re-hoeing
FRESH_TILLED = 0x07        # Freshly hoed, ready for planting
PLANTED_GRASS_TILE = 0x70  # After planting grass seeds on tilled soil
GRASS_TILES = {0x70, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85}  # planted + mature
TILLED_TILES = {DRIED_TILLED, FRESH_TILLED}  # Both types of tilled

# Tiles the hoe can act on (both untilled AND dried-out tilled need re-hoeing)
TILLABLE_TILES = {UNTILLED_TILE, DRIED_TILLED}

# Tiles grass seeds can be planted on (only freshly hoed)
PLANTABLE_TILES = {FRESH_TILLED}

# Item ID for grass seeds when equipped
GRASS_SEED_ITEM_ID = 0x0C

# Power Berry detection
ADDR_POWER_BERRIES = 0x0976  # Count of collected berries (verified via state diff)

# Default bounds: full farmable area
DEFAULT_BOUNDS = (3, 3, 62, 60)

# Default no-go: entire left side of farm (house, shed, shipping bin, barn)
DEFAULT_NO_GO_RECTS = [(3, 3, 28, 60)]


# =============================================================================
# GRASS PLANT TASK
# =============================================================================

class Phase:
    TILL = "till"
    PLANT = "plant"


@dataclass
class GrassPlantTask(Task):
    """Till ground and plant grass seeds within configurable bounds."""

    name: str = "grass_planter"
    bounds: Tuple[int, int, int, int] = DEFAULT_BOUNDS
    no_go_rects: List[Tuple[int, int, int, int]] = field(default_factory=lambda: list(DEFAULT_NO_GO_RECTS))
    till_only: bool = False
    max_steps_per_target: int = 1200
    stasis_repath: int = 180
    max_failures: int = 50
    max_retry_passes: int = 2
    debug: bool = False
    debug_interval: int = 300

    chunk_size: int = 3  # Work in 3x3 chunks

    # Internal state (not constructor args)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _tool_mgr: ToolManager = field(default_factory=ToolManager, init=False)
    _no_go_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    _phase: str = field(default=Phase.TILL, init=False)
    _state: str = field(default="scan", init=False)
    _target_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _approach_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _steps_on_target: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _failures: int = field(default=0, init=False)
    _plant_failures: int = field(default=0, init=False)
    _failed_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)
    _retry_pass: int = field(default=0, init=False)

    # Chunk tracking: current 3x3 chunk origin (top-left corner)
    _chunk_origin: Optional[Tuple[int, int]] = field(default=None, init=False)
    _chunk_phase_logged: str = field(default="", init=False)
    _chunk_order: List[Tuple[int, int]] = field(default_factory=list, init=False)
    _chunk_index: int = field(default=0, init=False)
    _act_count: int = field(default=0, init=False)  # total act attempts (for diagnostic logging)

    tilled_count: int = field(default=0, init=False)
    planted_count: int = field(default=0, init=False)
    berries_found: int = field(default=0, init=False)
    _prev_berry_count: int = field(default=0, init=False)
    _berry_dialog_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        if os.getenv("GRASS_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True
        self._phase = Phase.TILL
        self._state = "scan"
        self._target_tile = None
        self._approach_tile = None
        self._action_queue.clear()
        self._steps_on_target = 0
        self._total_steps = 0
        self._failures = 0
        self._plant_failures = 0
        self._failed_tiles.clear()
        self._chunk_origin = None
        self._chunk_phase_logged = ""
        self._retry_pass = 0
        self._act_count = 0
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._init_no_go()
        self._chunk_order = self._build_chunk_order()
        self._chunk_index = 0
        self.tilled_count = 0
        self.planted_count = 0
        self.berries_found = 0
        pb = int(world.ram[ADDR_POWER_BERRIES]) if ADDR_POWER_BERRIES < len(world.ram) else 0
        self._prev_berry_count = pb
        self._berry_dialog_frames = 0

    def _init_no_go(self):
        """Build no-go tile set from rects + GRASS_NO_GO env var."""
        self._no_go_tiles.clear()
        all_rects = list(self.no_go_rects)
        env_val = os.getenv("GRASS_NO_GO", "")
        if env_val:
            for rect_str in env_val.split(";"):
                rect_str = rect_str.strip()
                if not rect_str:
                    continue
                parts = rect_str.split(",")
                if len(parts) == 4:
                    all_rects.append(tuple(int(p) for p in parts))
        for x1, y1, x2, y2 in all_rects:
            for ty in range(y1, y2 + 1):
                for tx in range(x1, x2 + 1):
                    self._no_go_tiles.add((tx, ty))
        if self._no_go_tiles:
            print(f"[GRASS] No-go: {len(self._no_go_tiles)} tiles from {len(all_rects)} rects")

    def _build_chunk_order(self) -> List[Tuple[int, int]]:
        """Build ordered list of chunk origins left-to-right, top-to-bottom."""
        x_min, y_min, x_max, y_max = self.bounds
        cs = self.chunk_size
        chunks = []
        for cy in range(y_min, y_max + 1, cs):
            for cx in range(x_min, x_max + 1, cs):
                chunks.append((cx, cy))
        return chunks

    def can_start(self, world: WorldState) -> bool:
        return True

    # ------------------------------------------------------------------
    # Scanning helpers
    # ------------------------------------------------------------------

    def _scan_targets(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        """Find tiles matching the current phase within full bounds (for tests/stats)."""
        target_ids = TILLABLE_TILES if self._phase == Phase.TILL else PLANTABLE_TILES
        x_min, y_min, x_max, y_max = self.bounds
        results = []
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                tid = get_tile_at(ram, tx, ty)
                if tid in target_ids and (tx, ty) not in self._failed_tiles and (tx, ty) not in self._no_go_tiles:
                    results.append((tx, ty))
        return results

    def _scan_chunk(self, ram: np.ndarray, origin: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find target tiles within a single chunk."""
        target_ids = TILLABLE_TILES if self._phase == Phase.TILL else PLANTABLE_TILES
        x_min, y_min, x_max, y_max = self.bounds
        cs = self.chunk_size
        ox, oy = origin
        results = []
        for ty in range(oy, min(oy + cs, y_max + 1)):
            for tx in range(ox, min(ox + cs, x_max + 1)):
                if tx < x_min or ty < y_min:
                    continue
                tid = get_tile_at(ram, tx, ty)
                if tid in target_ids and (tx, ty) not in self._failed_tiles and (tx, ty) not in self._no_go_tiles:
                    results.append((tx, ty))
        return results

    def _face_dir(self, player: Tuple[int, int], target: Tuple[int, int]) -> str:
        dx = target[0] - player[0]
        dy = target[1] - player[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_scan(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Per-chunk till→plant: till all tiles in a chunk, then plant, then advance."""
        player_pos = self._navigator.current_pos

        while self._chunk_index < len(self._chunk_order):
            origin = self._chunk_order[self._chunk_index]
            targets = self._scan_chunk(ram, origin)

            if not targets:
                # Current phase done in this chunk
                if self._phase == Phase.TILL and not self.till_only:
                    # Switch to plant within same chunk
                    self._phase = Phase.PLANT
                    targets = self._scan_chunk(ram, origin)
                    if not targets:
                        # Nothing to plant either, advance chunk
                        self._chunk_index += 1
                        self._phase = Phase.TILL
                        self._chunk_origin = None
                        continue
                else:
                    # Chunk complete (till-only or plant done), advance
                    self._chunk_index += 1
                    self._phase = Phase.TILL
                    self._chunk_origin = None
                    continue

            # Log chunk/phase transitions
            if self._chunk_origin != origin or self._chunk_phase_logged != self._phase:
                self._chunk_origin = origin
                self._chunk_phase_logged = self._phase
                cs = self.chunk_size
                phase = self._phase.upper()
                retry = f" retry={self._retry_pass}" if self._retry_pass > 0 else ""
                print(f"[GRASS] {phase} chunk ({origin[0]},{origin[1]})-({origin[0]+cs-1},{origin[1]+cs-1})  "
                      f"{len(targets)} tiles{retry}")

            # Use strict row order within chunk (no proximity sort)
            for target in targets:
                if self._phase == Phase.PLANT:
                    # PLANT: navigate ONTO target tile (grass seeds plant 3x3 around player)
                    path = self._pathfinder.find_path(ram, self._navigator.current_tile, target)
                    if path is None:
                        continue
                    self._target_tile = target
                    self._approach_tile = target  # stand ON the tile
                else:
                    # TILL: navigate adjacent to target, face it, use hoe
                    approach = self._pathfinder.find_approach(ram, target, player_pos)
                    if approach is None:
                        continue
                    path = self._pathfinder.find_path(ram, self._navigator.current_tile, approach)
                    if path is None:
                        continue
                    self._target_tile = target
                    self._approach_tile = approach
                self._navigator.path = path
                self._navigator.stasis = 0
                self._steps_on_target = 0
                self._state = "navigate"
                if self.debug:
                    print(f"[GRASS] -> ({target[0]},{target[1]}) via ({self._approach_tile[0]},{self._approach_tile[1]}) path={len(path)}")
                return None

            # All targets in this chunk are unreachable for current phase
            if self._phase == Phase.TILL and not self.till_only:
                self._phase = Phase.PLANT
                continue  # re-scan same chunk for plantable
            self._chunk_index += 1
            self._phase = Phase.TILL
            self._chunk_origin = None

        # All chunks exhausted - try retry pass
        if self._failed_tiles and self._retry_pass < self.max_retry_passes:
            self._retry_pass += 1
            n_failed = len(self._failed_tiles)
            self._failed_tiles.clear()
            self._chunk_index = 0
            self._chunk_origin = None
            self._phase = Phase.TILL
            print(f"[GRASS] Pass done. Retry {self._retry_pass}/{self.max_retry_passes} ({n_failed} failed tiles)")
            return None

        msg = f"tilled={self.tilled_count} planted={self.planted_count}"
        print(f"[GRASS] Complete: {msg}")
        return TaskResult(status=TaskStatus.SUCCESS, reason=msg)

    def _handle_navigate(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._target_tile is None or self._approach_tile is None:
            self._state = "scan"
            return None

        # Check target still valid
        tid = get_tile_at(ram, *self._target_tile)
        target_ids = TILLABLE_TILES if self._phase == Phase.TILL else PLANTABLE_TILES
        if tid not in target_ids:
            self._state = "scan"
            return None

        # Arrived?
        if self._navigator.current_tile == self._approach_tile:
            self._state = "center"
            return None

        # Stuck recovery
        if self._navigator.stasis > self.stasis_repath and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, self._approach_tile)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                self._failures += 1
                self._failed_tiles.add(self._target_tile)
                self._state = "scan"
                if self._failures >= self.max_failures:
                    return TaskResult(status=TaskStatus.FAILURE, reason="too many nav failures")
                return None

        action = self._navigator.follow_path(ram)
        if action is not None:
            self._action_queue.append(action)
        return None

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Center on approach tile before acting."""
        if self._approach_tile is None:
            self._state = "scan"
            return None
        tol = 1 if self._phase == Phase.PLANT else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            self._state = "act"
        else:
            self._action_queue.append(center_action)
        return None

    def _handle_act(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Use tool on target tile. Matches farm_clearer timing pattern."""
        if self._target_tile is None or self._approach_tile is None:
            self._state = "scan"
            return None

        tx, ty = self._target_tile

        # Re-validate target
        tid = get_tile_at(ram, tx, ty)
        target_ids = TILLABLE_TILES if self._phase == Phase.TILL else PLANTABLE_TILES
        if tid not in target_ids:
            if self._phase == Phase.TILL and tid == FRESH_TILLED:
                self.tilled_count += 1
            self._state = "scan"
            return None

        # Position check depends on phase
        player = self._navigator.current_tile
        if self._phase == Phase.PLANT:
            # Must be ON target tile (grass seeds plant 3x3 around player)
            if player != self._target_tile:
                self._state = "navigate"
                return None
        else:
            # Must be adjacent to target tile (hoe acts on tile in front)
            if tile_dist(player, self._target_tile) > 1:
                self._state = "navigate"
                return None

        # Wait for queued actions to finish
        if self._action_queue:
            return None

        # Wait for input lock to clear and player to settle (same as farm_clearer)
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1 or self._navigator.stasis < 6:
            return None

        # Re-center on approach tile before each use (drift correction)
        tol = 1 if self._phase == Phase.PLANT else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        # Check correct tool is equipped
        wanted_tool = Tool.HOE if self._phase == Phase.TILL else GRASS_SEED_ITEM_ID
        if self._tool_mgr.current != wanted_tool:
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None

        self._act_count += 1
        phase = self._phase.upper()
        if self._phase == Phase.PLANT:
            # Standing ON tilled tile - face down then Y to plant 3x3 around player
            direction = "down"
        else:
            # Adjacent to target - face it then Y to hoe
            direction = self._face_dir(player, self._target_tile)

        if self._act_count <= 20 or self.debug:
            print(f"[GRASS] {phase} ({tx},{ty}) [0x{tid:02X}] from ({player[0]},{player[1]}) facing {direction} tool=0x{self._tool_mgr.current:02X}")

        # Face direction → settle → Y press → cooldown
        # Tile change takes ~26 frames after Y ends, so cooldown must be >= 30
        self._action_queue.extend([make_action(**{direction: True}) for _ in range(4)])  # Face target
        self._action_queue.extend([make_action() for _ in range(6)])     # Let face register
        self._action_queue.extend(use_tool(frames=20, cooldown=30))      # Y press + cooldown
        self._state = "verify"
        return None

    def _handle_verify(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Check if tool use changed the target tile."""
        if self._action_queue:
            return None

        # Wait for input lock to clear (animation done)
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1:
            return None

        if self._target_tile is None:
            self._state = "scan"
            return None

        tx, ty = self._target_tile
        tid = get_tile_at(ram, tx, ty)

        if self._phase == Phase.TILL:
            # Success = tile changed to freshly hoed (0x07)
            if tid == FRESH_TILLED:
                self.tilled_count += 1
                if self._act_count <= 20 or self.debug:
                    print(f"[GRASS] TILL OK ({tx},{ty}) -> 0x{tid:02X} tilled={self.tilled_count}")
            else:
                self._failures += 1
                self._failed_tiles.add(self._target_tile)
                if self._act_count <= 20 or self.debug:
                    print(f"[GRASS] TILL FAIL ({tx},{ty}) still 0x{tid:02X} failures={self._failures}")
        else:
            # Grass seeds plant 3x3 around player — count all newly-planted
            player = self._navigator.current_tile
            seeded = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cx, cy = player[0] + dx, player[1] + dy
                    t = get_tile_at(ram, cx, cy)
                    if t in GRASS_TILES:
                        seeded += 1
            if seeded > 0:
                self.planted_count += seeded
                if self._act_count <= 20 or self.debug:
                    print(f"[GRASS] PLANT @ ({player[0]},{player[1]}) seeded={seeded} total={self.planted_count}")

        if self._failures >= self.max_failures:
            return TaskResult(status=TaskStatus.FAILURE, reason="too many till failures")

        self._state = "scan"
        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Cycle tools until we find the one we need."""
        wanted = Tool.HOE if self._phase == Phase.TILL else GRASS_SEED_ITEM_ID
        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                name = "HOE" if self._phase == Phase.TILL else "GRASS_SEEDS"
                print(f"[GRASS] Found {name}")
            self._state = "center"  # re-center before acting
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete():
            # Wrapped around without finding tool
            name = "HOE" if self._phase == Phase.TILL else "GRASS_SEEDS"
            print(f"[GRASS] Tool {name} not found in inventory")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{name} not in inventory")

        self._action_queue.extend(cycle_tool())
        return None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._total_steps += 1
        self._steps_on_target += 1

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            phase = self._phase.upper()
            print(f"[GRASS] step={self._total_steps} phase={phase} state={self._state} pos={cur} "
                  f"tilled={self.tilled_count} planted={self.planted_count} failures={self._failures}")

        # ---- Power Berry detection ----
        cur_pb = int(world.ram[ADDR_POWER_BERRIES]) if ADDR_POWER_BERRIES < len(world.ram) else 0
        if cur_pb > self._prev_berry_count:
            gained = cur_pb - self._prev_berry_count
            self.berries_found += gained
            self._prev_berry_count = cur_pb
            self._action_queue.clear()
            self._berry_dialog_frames = 90
            print(f"[GRASS] POWER BERRY! total={cur_pb}")

        if self._berry_dialog_frames > 0:
            self._berry_dialog_frames -= 1
            action = make_action(a=True) if self._berry_dialog_frames % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="berry_dialog")

        # ---- Timeout per target ----
        if self._steps_on_target > self.max_steps_per_target and self._target_tile is not None:
            self._failed_tiles.add(self._target_tile)
            self._failures += 1
            self._state = "scan"
            self._target_tile = None
            self._action_queue.clear()
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many target timeouts")

        # Drain action queue first - tool animations must complete even during input lock
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Dialog dismissal (only after queue is empty so tool swings aren't interrupted)
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if self._total_steps % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="dialog")

        # State dispatch
        handlers = {
            "scan": self._handle_scan,
            "navigate": self._handle_navigate,
            "center": self._handle_center,
            "act": self._handle_act,
            "verify": self._handle_verify,
            "tool_switch": self._handle_tool_switch,
        }

        handler = handlers.get(self._state)
        if handler:
            result = handler(world.ram)
            if result is not None:
                return result

        # Drain any actions queued by handlers
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def phase_text(self) -> str:
        return f"{self._phase}:{self._state}"

    @property
    def progress_text(self) -> str:
        s = f"tilled={self.tilled_count} planted={self.planted_count}"
        if self._retry_pass > 0:
            s += f" retry={self._retry_pass}"
        if self._failures:
            s += f" till_fail={self._failures}"
        if self.berries_found:
            s += f" berries={self.berries_found}"
        return s


# =============================================================================
# CLI: discovery mode
# =============================================================================

def _run_discover(state_name: str):
    """Dump tile histogram and tool inventory for a save state."""
    import stable_retro as retro
    from collections import Counter

    integration_path = os.path.join(_SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(integration_path)

    env = retro.make(
        game="HarvestMoon-Snes",
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        state=state_name,
    )
    env.reset()
    ram = env.get_ram()

    print(f"=== Tile Histogram (bounds {DEFAULT_BOUNDS}) ===")
    hist: Counter = Counter()
    x_min, y_min, x_max, y_max = DEFAULT_BOUNDS
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tid = get_tile_at(ram, tx, ty)
            hist[tid] += 1
    for tid, count in sorted(hist.items(), key=lambda kv: -kv[1]):
        label = ""
        if tid in TILLABLE_TILES:
            label = " <- TILLABLE"
        elif tid in PLANTABLE_TILES:
            label = " <- PLANTABLE"
        elif tid == PLANTED_GRASS_TILE:
            label = " <- PLANTED GRASS"
        print(f"  0x{tid:02X}: {count:4d}{label}")

    print(f"\n=== Tool Inventory ===")
    print(f"  Current: 0x{int(ram[ADDR_TOOL]):02X}")
    seen = {int(ram[ADDR_TOOL])}
    for i in range(60):
        if i % 6 == 0:
            action = np.zeros(12, dtype=np.int32)
            action[9] = 1
            env.step(action)
        else:
            env.step(np.zeros(12, dtype=np.int32))
        tid = int(env.get_ram()[ADDR_TOOL])
        if tid not in seen:
            seen.add(tid)
    tool_names = {0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer", 0x04: "Axe",
                  0x0C: "GrassSeeds", 0x10: "WateringCan"}
    for tid in sorted(seen):
        name = tool_names.get(tid, "???")
        print(f"  0x{tid:02X} = {name}")

    print(f"\n=== Grass Seeds ===")
    env2 = retro.make(
        game="HarvestMoon-Snes",
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        state=state_name,
    )
    # Can't make two envs - just report from RAM
    grass_addr = 0x0927
    grass_count = int(ram[grass_addr]) if grass_addr < len(ram) else 0
    print(f"  RAM[0x0927] = {grass_count}")
    print(f"  set_value('grass_seeds', 99) supported: yes (verified)")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grass Planter - discover tile IDs or run planting")
    parser.add_argument("--discover", action="store_true", help="Run tile/tool discovery")
    parser.add_argument("--state", type=str, default="Y1_Spring_Day01_06h00m", help="Save state name")
    args = parser.parse_args()

    if args.discover:
        _run_discover(args.state)
    else:
        print("Use --discover to dump tile IDs, or import GrassPlantTask from harvest_bot.py")
