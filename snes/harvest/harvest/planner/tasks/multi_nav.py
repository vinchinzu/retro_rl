"""Multi-map waypoint navigation used by day-plan phases.

Corridor policy (soft solids, entities, lift_throw, fail-closed seal) lives in
:mod:`nav_corridor`. This module owns the waypoint FSM and A/B-loop loader.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Pathfinder,
    Navigator,
    make_action,
    get_tile_at,
    TILE_SIZE,
)
from harvest.core.animal_status import read_held_item
from harvest.core.tile_catalog import ADDR_TILEMAP, LIFTABLE_TILES
from harvest.tasks.farm_ops import TileScanner

from harvest.maps.map_config import Waypoint, get_walkable_tiles
from harvest.tasks.primitives import (
    drain_action_queue,
    press_button_sequence,
)
from harvest.planner.day_plan_status import tilemaps_match
from harvest.planner.tasks.nav_corridor import (
    dirs_toward,
    entity_blocks,
    farm_soft_blocks,
    hop_target,
    liftable_gate_toward,
    micro_center_action,
    opportunistic_clear_waypoint,
    queue_lift_throw,
    replace_no_go,
    safe_walk_action,
    tile_blocks_charge,
)
from harvest.planner.tasks.navigation import (
    STALE_TILE_IDS,
    find_frontier_path,
    find_loaded_direction,
    _neighbor_tile,
    _nav_needs_menu_dismiss,
)

# ── MultiMapNavTask ───────────────────────────────────────────────

@dataclass
class MultiMapNavTask(Task):
    """Navigate a sequence of waypoints across multiple maps using BFS.

    State machine per waypoint:
      nav        → BFS navigate to target_px using per-map walkable tiles
      action     → Face direction, press button, cooldown
      exit_walk  → Walk exit_direction + B until tilemap changes
      exit_settle → Idle frames for tile data to load, rebuild pathfinder
    """

    name: str = "multi_nav"
    waypoints: List[Waypoint] = field(default_factory=list)
    timeout: int = 8000
    initial_settle_frames: int = 60
    # Cargo routes (egg/crop/forage already in hand) must fail closed at a
    # blocked corridor.  Opportunistic lift/throw would throw the cargo away.
    allow_opportunistic_clear: bool = True

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _step_count: int = field(default=0, init=False)
    _wp_index: int = field(default=0, init=False)
    _phase: str = field(default="nav", init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _settle_frames: int = field(default=0, init=False)
    _exit_walk_frames: int = field(default=0, init=False)
    _no_path_frames: int = field(default=0, init=False)
    _initial_settle: int = field(default=0, init=False)
    # Stagnant no-path recovery (SW farm pocket after CLEAR → house).
    _stuck_anchor: Optional[Tuple[int, int]] = field(default=None, init=False)
    _stuck_frames: int = field(default=0, init=False)
    # Tile-stasis misses L/R wiggle across a tile boundary (stasis resets).
    _pixel_anchor: Optional[Tuple[int, int]] = field(default=None, init=False)
    _pixel_stuck: int = field(default=0, init=False)
    _pixel_replans: int = field(default=0, init=False)
    _farm_soft_blocks: Set[Tuple[int, int]] = field(default_factory=set, init=False)
    _entity_blocks: Set[Tuple[int, int]] = field(default_factory=set, init=False)
    _lift_throw_attempts: int = field(default=0, init=False)
    _soft_solid_pin_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self._scanner = TileScanner()
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._wp_index = 0
        self._phase = "nav"
        self._action_queue.clear()
        self._settle_frames = 0
        self._exit_walk_frames = 0
        self._no_path_frames = 0
        self._initial_settle = 0
        self._stuck_anchor = None
        self._stuck_frames = 0
        self._pixel_anchor = None
        self._pixel_stuck = 0
        self._pixel_replans = 0
        self._farm_soft_blocks.clear()
        self._entity_blocks.clear()
        self._lift_throw_attempts = 0
        self._soft_solid_pin_frames = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        # Set initial walkable tiles based on current tilemap
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        self._rebuild_pathfinder(tilemap)
        if self.waypoints:
            print(f"[MULTI_NAV] Start: {len(self.waypoints)} waypoints, tilemap=0x{tilemap:02X}")

    def can_start(self, world: WorldState) -> bool:
        return len(self.waypoints) > 0

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        if self._phase != "exit_settle":
            self._phase = "nav"
        self._settle_frames = 0
        self._exit_walk_frames = 0
        self._no_path_frames = 0
        self._stuck_anchor = None
        self._stuck_frames = 0
        self._pixel_anchor = None
        self._pixel_stuck = 0
        self._pixel_replans = 0
        self._farm_soft_blocks.clear()
        self._entity_blocks.clear()
        self._lift_throw_attempts = 0
        self._soft_solid_pin_frames = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def _rebuild_pathfinder(self, tilemap_id: int) -> None:
        """Rebuild pathfinder with walkable tiles for the given map."""
        walkable = get_walkable_tiles(tilemap_id)
        self._pathfinder = Pathfinder(self._scanner, walkable_tiles=set(walkable))
        self._navigator = Navigator(self._pathfinder)
        self._navigator.stasis = 0
        self._farm_soft_blocks.clear()
        self._entity_blocks.clear()

    def _clear_dynamic_blocks(self) -> None:
        self._pathfinder.no_go_tiles.difference_update(self._farm_soft_blocks)
        self._pathfinder.no_go_tiles.difference_update(self._entity_blocks)
        self._farm_soft_blocks.clear()
        self._entity_blocks.clear()

    def _sync_farm_soft_blocks(self, ram: np.ndarray, tilemap: int) -> None:
        nxt = farm_soft_blocks(self._scanner, ram, tilemap)
        replace_no_go(self._pathfinder, self._farm_soft_blocks, nxt)
        self._farm_soft_blocks = nxt

    def _sync_entity_blocks(self, ram: np.ndarray) -> None:
        nxt = entity_blocks(ram, self._navigator.current_tile)
        replace_no_go(self._pathfinder, self._entity_blocks, nxt)
        self._entity_blocks = nxt

    def _sync_travel_blocks(self, ram: np.ndarray, tilemap: int) -> None:
        self._sync_farm_soft_blocks(ram, tilemap)
        self._sync_entity_blocks(ram)

    # Back-compat for unit tests that assert weed no-go membership.
    @property
    def _farm_weed_blocks(self) -> Set[Tuple[int, int]]:
        return set(self._farm_soft_blocks)

    def _queue_lift_throw(self, ram: np.ndarray, wp: Waypoint) -> Optional[str]:
        return queue_lift_throw(
            self._action_queue, self._navigator.current_tile, ram, wp
        )

    def _start_waypoint_action(self, world: WorldState, wp: Waypoint) -> TaskResult:
        """Begin the action for the current waypoint (same-frame on arrival)."""
        if wp.action_on_arrive == "lift_throw":
            reason = self._queue_lift_throw(world.ram, wp)
            if reason is None:
                print(
                    f"[MULTI_NAV] lift_throw skip (gate clear) "
                    f"face={wp.action_face} at {self._navigator.current_tile}"
                )
                self._lift_throw_attempts = 0
                self._advance_waypoint()
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="lift_throw already clear",
                )
            self._lift_throw_attempts += 1
            print(
                f"[MULTI_NAV] Action: lift_throw {reason} "
                f"attempt={self._lift_throw_attempts}"
            )
            self._phase = "lift_throw_drain"
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="lift_throw empty queue",
            )

        button = {"press_a": "a", "press_b": "b", "press_y": "y"}.get(
            wp.action_on_arrive or ""
        )
        if button:
            self._action_queue.extend(
                press_button_sequence(
                    button,
                    face=wp.action_face,
                    face_frames=1 if wp.action_face else 0,
                    pre_press_settle_frames=5 if wp.action_face else 0,
                    hold_frames=wp.action_frames,
                    settle_frames=wp.action_cooldown,
                )
            )

        print(f"[MULTI_NAV] Action: {wp.action_on_arrive} face={wp.action_face}")
        self._phase = "action_drain"
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued
        self._advance_waypoint()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _current_wp(self) -> Optional[Waypoint]:
        if self._wp_index < len(self.waypoints):
            return self.waypoints[self._wp_index]
        return None

    def _waypoint_tilemap_matches(self, tilemap: int, wp: Waypoint) -> bool:
        return tilemaps_match(tilemap, wp.tilemap)

    def _at_wp_target(self, wp: Waypoint) -> bool:
        pos = self._navigator.current_pos
        return (abs(pos.x - wp.target_px[0]) <= wp.radius and
                abs(pos.y - wp.target_px[1]) <= wp.radius)

    def _tile_blocks_charge(self, ram: np.ndarray, tx: int, ty: int) -> bool:
        return tile_blocks_charge(self._pathfinder, ram, tx, ty)

    def _safe_walk_action(
        self,
        ram: np.ndarray,
        preferred: str,
        *,
        secondary: Optional[str] = None,
        allow_detour: bool = False,
    ) -> Optional[np.ndarray]:
        return safe_walk_action(
            self._pathfinder,
            self._navigator,
            ram,
            preferred,
            secondary=secondary,
            allow_detour=allow_detour,
        )

    def _update_pixel_stuck(self) -> None:
        """Count frames with no real movement. Tile-stasis misses L/R wiggle."""
        if self._phase != "nav":
            return
        cur = self._navigator.current_pos
        anchor = self._pixel_anchor
        if (
            anchor is not None
            and max(abs(cur.x - anchor[0]), abs(cur.y - anchor[1])) < 4
        ):
            self._pixel_stuck += 1
        else:
            self._pixel_anchor = (cur.x, cur.y)
            self._pixel_stuck = 0
            self._pixel_replans = 0

    def _pixel_stuck_replan(self) -> Optional[TaskResult]:
        """Break an in-place left/right pin instead of burning the timeout."""
        if self._pixel_stuck < 48:
            return None
        cur = self._navigator.current_pos
        pin = self._navigator.path[0] if self._navigator.path else None
        self._navigator.path = []
        # Drop push-facing neighbors from a short charge; keep only the L/R pin.
        self._pathfinder.temp_blocked.clear()
        if pin is not None:
            self._pathfinder.temp_blocked.add(pin)
        self._navigator.stasis = 0
        self._navigator._push_tile = None
        self._navigator._push_px = None
        self._navigator._push_hold = 0
        self._pixel_stuck = 0
        self._pixel_replans += 1
        print(
            f"[MULTI_NAV] Pixel-stuck pos=({cur.x},{cur.y}) "
            f"replans={self._pixel_replans} — skip L/R center"
        )
        if self._pixel_replans >= 4:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"pixel_stuck pos=({cur.x},{cur.y}) "
                    f"replans={self._pixel_replans}"
                ),
            )
        return None

    def _advance_waypoint(self) -> None:
        """Move to next waypoint."""
        self._wp_index += 1
        self._phase = "nav"
        self._action_queue.clear()
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._lift_throw_attempts = 0
        self._soft_solid_pin_frames = 0
        self._pixel_anchor = None
        self._pixel_stuck = 0
        self._pixel_replans = 0
        wp = self._current_wp()
        if wp:
            print(f"[MULTI_NAV] Waypoint {self._wp_index + 1}/{len(self.waypoints)}"
                  f" tilemap=0x{wp.tilemap:02X} target={wp.target_px}")

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._step_count += 1
        self._update_pixel_stuck()

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="multi_nav timeout")

        # Initial settle: walk toward first waypoint to trigger tile loading.
        # After map transition, SNES tile RAM reads stale 0x72 until the
        # player moves and the viewport scrolls to load new tile data.
        SETTLE_FRAMES = self.initial_settle_frames
        if self._initial_settle < SETTLE_FRAMES:
            self._initial_settle += 1
            dismissed = _nav_needs_menu_dismiss(world.ram, self._step_count)
            if dismissed is not None:
                return dismissed
            # Walk toward first waypoint during settle to trigger tile loading
            wp = self._current_wp()
            if wp:
                cur = self._navigator.current_pos
                primary, secondary = dirs_toward(
                    wp.target_px[0] - cur.x, wp.target_px[1] - cur.y
                )
                action = self._safe_walk_action(
                    world.ram, primary, secondary=secondary
                )
                if action is None:
                    action = make_action()
            else:
                action = make_action()
            if self._initial_settle == SETTLE_FRAMES:
                tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
                self._rebuild_pathfinder(tilemap)
                self._navigator.update(world.ram)
                print(f"[MULTI_NAV] Settle done, pos=({self._navigator.current_pos.x},"
                      f"{self._navigator.current_pos.y}) tilemap=0x{tilemap:02X}")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # All waypoints done
        wp = self._current_wp()
        if wp is None:
            return TaskResult(status=TaskStatus.SUCCESS, reason="all waypoints reached")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if (
            not self._waypoint_tilemap_matches(tilemap, wp)
            and self._phase not in {"exit_walk", "exit_settle"}
        ):
            for idx in range(self._wp_index + 1, len(self.waypoints)):
                if self._waypoint_tilemap_matches(tilemap, self.waypoints[idx]):
                    print(f"[MULTI_NAV] Relocalized from waypoint {self._wp_index + 1}"
                          f" to {idx + 1} on tilemap=0x{tilemap:02X}")
                    self._wp_index = idx
                    self._phase = "nav"
                    self._navigator.path = []
                    self._navigator.stasis = 0
                    self._pathfinder.temp_blocked.clear()
                    # Some exits flip tilemap just outside their waypoint
                    # radius.  Rebuild for the new map and re-run the short
                    # coordinate/tile settle before moving toward its first
                    # waypoint; otherwise stale origin coordinates can walk
                    # straight back through the transition.
                    self._rebuild_pathfinder(tilemap)
                    self._initial_settle = 0
                    wp = self._current_wp()
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                        reason="relocalized after map transition",
                    )
            if wp is not None and not self._waypoint_tilemap_matches(tilemap, wp):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"expected tilemap 0x{wp.tilemap:02X}, got 0x{tilemap:02X}",
                )

        # Dialog / menu dismissal
        dismissed = _nav_needs_menu_dismiss(world.ram, self._step_count)
        if dismissed is not None:
            return dismissed

        # Drain queued actions
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        # ── Phase: exit_settle ──
        if self._phase == "exit_settle":
            self._settle_frames += 1
            if self._settle_frames <= int(wp.exit_push_frames):
                direction = wp.exit_direction or "left"
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(**{direction: True})),
                    reason=f"push into destination {direction}",
                )
            if self._settle_frames >= 30:
                # Rebuild pathfinder for new map
                self._rebuild_pathfinder(tilemap)
                self._navigator.update(world.ram)
                print(f"[MULTI_NAV] Settled on tilemap 0x{tilemap:02X}"
                      f" pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y})")
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # ── Phase: exit_walk ──
        if self._phase == "exit_walk":
            self._exit_walk_frames += 1
            # Check if tilemap changed
            if not self._waypoint_tilemap_matches(tilemap, wp):
                print(f"[MULTI_NAV] Exited map 0x{wp.tilemap:02X} → 0x{tilemap:02X}"
                      f" after {self._exit_walk_frames} frames")
                self._phase = "exit_settle"
                self._settle_frames = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            # Timeout: give up after 500 frames of walking toward exit
            if self._exit_walk_frames > 500:
                print("[MULTI_NAV] Exit walk timeout (500 frames)")
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            direction = wp.exit_direction or "left"
            action = make_action(**{direction: True, "b": True})
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # ── Phase: action ──
        if self._phase == "action":
            return self._start_waypoint_action(world, wp)

        if self._phase == "lift_throw_drain":
            if self._action_queue:
                queued = drain_action_queue(self._action_queue)
                if queued is not None:
                    return queued
            held = int(read_held_item(world.ram))
            face = wp.action_face or "up"
            target = _neighbor_tile(
                self._navigator.current_tile[0],
                self._navigator.current_tile[1],
                face,
            )
            tid = int(get_tile_at(world.ram, *target))
            # Also treat opportunistic mid-nav clears (no lift_throw action on wp).
            waypoint_owned = wp.action_on_arrive == "lift_throw"
            if held == 0:
                # Re-scan: facing may not be the cleared cell after throw.
                still_blocked = tid in LIFTABLE_TILES
                if not still_blocked or not waypoint_owned:
                    self._lift_throw_attempts = 0
                    self._stuck_frames = 0
                    self._no_path_frames = 0
                    self._navigator.path = []
                    # Refresh soft-solid no-go so BFS can use the opened cell.
                    self._sync_travel_blocks(world.ram, tilemap)
                    if waypoint_owned:
                        self._advance_waypoint()
                    else:
                        self._phase = "nav"
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                        reason="lift_throw cleared",
                    )
            if self._lift_throw_attempts >= 4:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"lift_throw failed held=0x{held:02X} "
                        f"target={target} tid=0x{tid:02X}"
                    ),
                )
            # Retry: waypoint-owned goes back to action; opportunistic re-queues.
            if waypoint_owned:
                self._phase = "action"
            else:
                self._phase = "nav"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="lift_throw retry",
            )

        if self._phase == "action_drain":
            if not self._action_queue:
                # Action sequence done, advance
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued

        # ── Phase: nav ──
        # Check arrival. Process action waypoints in the same frame (no idle
        # thrash frame) so berry lift_throw / bin drop stay frame-tight.
        if self._at_wp_target(wp) and self._phase == "nav":
            if wp.is_exit:
                print(f"[MULTI_NAV] Reached exit waypoint, walking {wp.exit_direction}")
                self._phase = "exit_walk"
                self._exit_walk_frames = 0
                direction = wp.exit_direction or "left"
                action = make_action(**{direction: True, "b": True})
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if wp.action_on_arrive:
                return self._start_waypoint_action(world, wp)
            # Just a nav waypoint, advance
            self._advance_waypoint()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # Direct run: if waypoint specifies run_direction, just hold that
        # direction + B. Much faster than BFS for known clear paths.
        # Check the axis of travel to detect overshoot.
        if wp.run_direction:
            cur = self._navigator.current_pos
            d = wp.run_direction
            if d in {"left", "right"} and abs(cur.y - wp.target_px[1]) > wp.radius:
                align = "down" if wp.target_px[1] > cur.y else "up"
                safe = self._safe_walk_action(world.ram, align)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(safe if safe is not None else make_action()),
                )
            if d in {"up", "down"} and abs(cur.x - wp.target_px[0]) > wp.radius:
                align = "right" if wp.target_px[0] > cur.x else "left"
                safe = self._safe_walk_action(world.ram, align)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(safe if safe is not None else make_action()),
                )
            overshot = False
            if d == "down" and cur.y > wp.target_px[1] + wp.radius:
                overshot = True
            elif d == "up" and cur.y < wp.target_px[1] - wp.radius:
                overshot = True
            elif d == "right" and cur.x > wp.target_px[0] + wp.radius:
                overshot = True
            elif d == "left" and cur.x < wp.target_px[0] - wp.radius:
                overshot = True
            if overshot:
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if wp.force_run:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(**{d: True, "b": True})),
                )
            safe = self._safe_walk_action(world.ram, d)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(safe if safe is not None else make_action()),
            )

        # Close-range direct walk: when within ~5 tiles, walk directly toward
        # the target without BFS — but NEVER into fence/weed/solid tiles.
        if not wp.is_exit:
            cur = self._navigator.current_pos
            dx_close = abs(wp.target_px[0] - cur.x)
            dy_close = abs(wp.target_px[1] - cur.y)
            stasis = self._navigator.stasis
            if (
                dx_close <= 80
                and dy_close <= 80
                and stasis < 40
                and self._pixel_stuck < 20
            ):  # ~5 tiles; bail if L/R pin
                primary, secondary = dirs_toward(
                    wp.target_px[0] - cur.x, wp.target_px[1] - cur.y
                )
                preferred = primary if stasis < 20 else secondary
                safe = self._safe_walk_action(
                    world.ram, preferred, secondary=secondary
                )
                if safe is not None:
                    return TaskResult(
                        status=TaskStatus.RUNNING, action=ActionResult(safe)
                    )
                # All neighbors blocked at close range → BFS / fail, no thrash.

        stuck = self._pixel_stuck_replan()
        if stuck is not None:
            return stuck

        # Stuck recovery — also replan around entities that walked onto the path.
        if self._navigator.stasis > 90 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []
            self._navigator.stasis = 0
            self._sync_travel_blocks(world.ram, tilemap)

        # Soft-solid / thrash pin during nav only (not mid lift/throw A-hold).
        held_now = int(read_held_item(world.ram))
        if self._phase == "nav":
            if self._navigator.stasis > 0:
                self._soft_solid_pin_frames += 1
            else:
                self._soft_solid_pin_frames = 0
                # Tile progress: allow more opportunistic clears later on route.
                if self._lift_throw_attempts > 0:
                    self._lift_throw_attempts = max(0, self._lift_throw_attempts - 1)
            pin_limit = (
                300
                if held_now != 0 and not self.allow_opportunistic_clear
                else 120
                if held_now != 0
                else 240
            )
            if self._soft_solid_pin_frames >= pin_limit:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"soft_solid pin held=0x{held_now:02X} "
                        f"pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y}) "
                        f"stasis={self._navigator.stasis}"
                    ),
                )
        else:
            self._soft_solid_pin_frames = 0

        # BFS path (viewport-aware hopping)
        if not self._navigator.path:
            self._sync_travel_blocks(world.ram, tilemap)
            hop = hop_target(self._navigator.current_tile, wp.target_px)
            goal = self._pathfinder.find_nearest_walkable(world.ram, hop, max_radius=4)
            if goal is None:
                goal = hop
            path = self._pathfinder.find_path(world.ram, self._navigator.current_tile, goal)
            # find_path returns [] when already on the goal tile (not None).
            # That is success: micro-center with close-range walk, do not seal.
            if path is None:
                final = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
                path = find_frontier_path(
                    self._pathfinder, world.ram, self._navigator.current_tile, final
                )
            if path is not None:
                self._navigator.path = path
                self._navigator.stasis = 0
                self._no_path_frames = 0
                self._stuck_anchor = None
                self._stuck_frames = 0
                # Same-tile goal: empty path means center with a pixel nudge.
                # Do NOT use _safe_walk_action — that refuses a step whose
                # *neighbor tile* is a weed even when we only need a few px
                # inside the current tile (berry lift_throw radius=4).
                if not path:
                    cur = self._navigator.current_pos
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(
                            micro_center_action(cur.x, cur.y, wp.target_px)
                        ),
                        reason="micro_center same tile",
                    )
            else:
                # BFS failed. Safe walk only — never B-run into fence/bushes.
                self._no_path_frames += 1
                cur = self._navigator.current_pos
                anchor = (cur.x, cur.y)
                if self._stuck_anchor is not None and max(
                    abs(cur.x - self._stuck_anchor[0]),
                    abs(cur.y - self._stuck_anchor[1]),
                ) < 8:
                    self._stuck_frames += 1
                else:
                    self._stuck_anchor = anchor
                    self._stuck_frames = 0
                if self._no_path_frames == 1 or self._no_path_frames % 300 == 0:
                    tx, ty = self._navigator.current_tile
                    neighbor_ids = {
                        direction: int(get_tile_at(world.ram, *_neighbor_tile(tx, ty, direction)))
                        for direction in ("up", "down", "left", "right")
                    }
                    print(
                        f"[MULTI_NAV] No BFS path from ({cur.x},{cur.y}) "
                        f"toward {wp.target_px}; safe walk only "
                        f"(frame {self._no_path_frames} stuck={self._stuck_frames} "
                        f"neighbors={neighbor_ids} entities={len(self._entity_blocks)})"
                    )
                if self._stuck_frames > 0 and self._stuck_frames % 120 == 0:
                    self._pathfinder.temp_blocked.clear()
                    self._navigator.stasis = 0
                    self._sync_travel_blocks(world.ram, tilemap)
                # Soft-solid gate: if a liftable weed/stone blocks progress toward
                # the waypoint, lift+throw it instead of sealing (live weed layout
                # differs from static route dumps every morning).
                if (
                    self.allow_opportunistic_clear
                    and self._stuck_frames >= 30
                    and self._stuck_frames < 90
                ):
                    gate = liftable_gate_toward(
                        self._navigator.current_tile, world.ram, wp
                    )
                    if gate is not None and self._lift_throw_attempts < 4:
                        face, _target, _tid = gate
                        reason = self._queue_lift_throw(
                            world.ram, opportunistic_clear_waypoint(wp, face)
                        )
                        if reason is not None:
                            self._lift_throw_attempts += 1
                            print(
                                f"[MULTI_NAV] Opportunistic {reason} "
                                f"(stuck={self._stuck_frames})"
                            )
                            self._phase = "lift_throw_drain"
                            queued = drain_action_queue(self._action_queue)
                            if queued is not None:
                                return queued
                # Fail fast when sealed (e.g. y=31 fence) — do not thrash.
                if self._stuck_frames >= 90:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            f"no_path sealed pos=({cur.x},{cur.y}) "
                            f"target={wp.target_px} stuck={self._stuck_frames}"
                        ),
                    )
                final = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
                loaded_direction = find_loaded_direction(
                    world.ram, self._navigator.current_tile, final
                )
                if (
                    get_tile_at(world.ram, *self._navigator.current_tile) in STALE_TILE_IDS
                    and loaded_direction is not None
                ):
                    safe = self._safe_walk_action(world.ram, loaded_direction)
                    if safe is not None:
                        return TaskResult(
                            status=TaskStatus.RUNNING, action=ActionResult(safe)
                        )
                primary, secondary = dirs_toward(
                    wp.target_px[0] - cur.x, wp.target_px[1] - cur.y
                )
                safe = self._safe_walk_action(
                    world.ram, primary, secondary=secondary
                )
                if safe is not None:
                    return TaskResult(
                        status=TaskStatus.RUNNING, action=ActionResult(safe)
                    )
                # Completely boxed in by solids — idle (0 thrash frames).
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="no_safe_step",
                )

        action = self._navigator.follow_path(world.ram)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


__all__ = ["MultiMapNavTask"]
