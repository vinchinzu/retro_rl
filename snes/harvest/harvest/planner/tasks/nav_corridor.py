"""Travel corridor policy for MultiMapNavTask.

Soft solids, live entities, hop clamp, safe B-charge, lift-throw, and the
stall / yield / run-direction / pin-recovery drive. The waypoint FSM stays
on MultiMapNavTask; :class:`NavCorridor` is mixed into that task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Set, Tuple

import numpy as np

from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState
from harvest.core.animal_status import read_held_item
from harvest.core.npc_catalog import game_objects
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    DebrisType,
    FENCE,
    LIFTABLE_TILES,
    WEED,
)
from harvest.maps.map_config import Waypoint
from harvest.planner.day_plan_status import tilemaps_match
from harvest.planner.tasks.navigation import (
    MAX_HOP,
    _DIR_DELTA,
    _OPPOSITE_FACE,
    neighbor_tile,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import (
    Navigator,
    Pathfinder,
    get_tile_at,
    make_action,
    TILE_SIZE,
)
from harvest.tasks.primitives import drain_action_queue, press_a_sequence


Tile = Tuple[int, int]

# A wandering NPC/animal clears one tile in well under a second. Waiting is
# both faster and safer than re-routing a proven corridor around it.
ENTITY_YIELD_FRAMES = 90
# Frames the close-range walk may run without closing on the waypoint before
# it hands the waypoint to BFS. A left/right bounce across a tile boundary
# resets ``Navigator.stasis`` every crossing, so no existing guard sees it.
CLOSE_RANGE_STALL_FRAMES = 30
# Frames a run_direction/force_run hop may make no progress before dropping
# to BFS. At run speed anything over ~0.5 s of this is a wall, not traffic.
RUN_DIR_STALL_FRAMES = 45
# Soft-solid pin recoveries per nav task before failing the leg.
PIN_RECOVERY_LIMIT = 4
# Live entity blocks are re-read on this cadence during nav (not only on a
# BFS replan, which never happens while the close-range walk is driving).
ENTITY_SYNC_PERIOD = 4


def dirs_toward(dx: int, dy: int) -> Tuple[str, str]:
    """Primary/secondary cardinals toward a pixel or tile delta."""
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left", "down" if dy > 0 else "up")
    return ("down" if dy > 0 else "up", "right" if dx > 0 else "left")


def hop_target(cur: Tile, target_px: Tuple[int, int]) -> Tile:
    """BFS target clamped so each axis stays inside the loaded viewport."""
    final = (target_px[0] // TILE_SIZE, target_px[1] // TILE_SIZE)
    dx = final[0] - cur[0]
    dy = final[1] - cur[1]
    if abs(dx) <= MAX_HOP and abs(dy) <= MAX_HOP:
        return final
    cx = max(-MAX_HOP, min(MAX_HOP, dx))
    cy = max(-MAX_HOP, min(MAX_HOP, dy))
    limit = 7
    if abs(cx) > limit or abs(cy) > limit:
        scale = limit / max(abs(cx), abs(cy))
        cx = int(cx * scale)
        cy = int(cy * scale)
    return (cur[0] + cx, cur[1] + cy)


def tile_blocks_charge(pathfinder: Pathfinder, ram: np.ndarray, tx: int, ty: int) -> bool:
    """True when charging into this tile wastes frames (fence/solid/bush)."""
    if not (0 <= tx < 64 and 0 <= ty < 64):
        return True
    if not pathfinder.is_walkable(ram, tx, ty):
        return True
    tid = int(get_tile_at(ram, tx, ty))
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    return bool(tilemaps_match(tilemap, 0x00) and tid in {FENCE, WEED})


def safe_walk_action(
    pathfinder: Pathfinder,
    navigator: Navigator,
    ram: np.ndarray,
    preferred: str,
    *,
    secondary: Optional[str] = None,
    allow_detour: bool = False,
) -> Optional[np.ndarray]:
    """Hold B+dir only if the next tile is not a solid/bush thrash cell."""
    cur = navigator.current_tile
    order: List[str] = []
    candidates = (
        (preferred, secondary, "down", "right", "left", "up")
        if allow_detour
        else (preferred, secondary)
    )
    for direction in candidates:
        if direction and direction not in order:
            order.append(direction)
    for direction in order:
        nx, ny = neighbor_tile(cur[0], cur[1], direction)
        if tile_blocks_charge(pathfinder, ram, nx, ny):
            continue
        if navigator.note_push_facing(ram, (nx, ny)):
            continue
        return make_action(**{direction: True, "b": True})
    return None


@dataclass
class CloseRangeLatch:
    """Distance watchdog for the close-range walk.

    A sideways bounce crosses a tile boundary every few frames, which resets
    ``Navigator.stasis`` — so the pin is invisible to every other guard and
    only distance-to-target tells the truth (run13 path 0x0C (8,6), 6x
    byte-identical). Frozen-in-place is *not* a bounce: that is the
    post-transition tile-load wait, owned by the pixel-stuck guard.
    """

    stall_frames: int = CLOSE_RANGE_STALL_FRAMES
    best: Optional[int] = None
    stall: int = 0

    def reset(self) -> None:
        self.best = None
        self.stall = 0

    def stalled(self, dist: int, *, moving: bool) -> bool:
        """True once the walk has stopped closing and BFS should take over."""
        if self.best is None or dist < self.best:
            self.best = dist
            self.stall = 0
        elif moving:
            self.stall += 1
        return self.stall >= self.stall_frames


def close_range_action(
    pathfinder: Pathfinder,
    navigator: Navigator,
    ram: np.ndarray,
    wp: Waypoint,
    *,
    stasis: int,
) -> Optional[np.ndarray]:
    """Walk straight at a waypoint from within ~5 tiles, or None for BFS.

    The secondary cardinal is offered only while its axis still has real
    error. A secondary that is already inside the arrival radius buys
    nothing and costs a step back, which is what turns a blocked primary
    into a left/right bounce against a concave cell.
    """
    cur = navigator.current_pos
    dx = wp.target_px[0] - cur.x
    dy = wp.target_px[1] - cur.y
    primary, secondary = dirs_toward(dx, dy)
    secondary_error = abs(dy) if secondary in ("up", "down") else abs(dx)
    if secondary_error <= max(wp.radius, 8):
        secondary = None
    preferred = primary if stasis < 20 else (secondary or primary)
    return safe_walk_action(
        pathfinder, navigator, ram, preferred, secondary=secondary
    )


def sprite_ahead(
    navigator: Navigator,
    wp: Waypoint,
    sprites: Set[Tile],
    *,
    run_direction: Optional[str] = None,
) -> Optional[Tile]:
    """The live sprite tile the farmer is about to walk into, if any."""
    if not sprites:
        return None
    if navigator.path:
        nxt = navigator.path[0]
    else:
        cur = navigator.current_pos
        tile = navigator.current_tile
        direction = run_direction or dirs_toward(
            wp.target_px[0] - cur.x, wp.target_px[1] - cur.y
        )[0]
        nxt = neighbor_tile(tile[0], tile[1], direction)
        if nxt == tile:
            return None
    return nxt if nxt in sprites else None


def farm_soft_blocks(
    scanner: TileScanner, ram: np.ndarray, tilemap: int
) -> Set[Tile]:
    """Weed/stone/fence cells that travel BFS must treat as no-go."""
    if not tilemaps_match(tilemap, 0x00):
        return set()
    return {
        target.tile
        for target in scanner.scan(
            ram,
            types={DebrisType.WEED, DebrisType.STONE, DebrisType.FENCE},
        )
    }


def _blocking_entity(obj) -> bool:
    """Live sprites the farmer physically collides with."""
    if getattr(obj, "is_player", False):
        return False
    kind = str(getattr(obj, "kind", "") or "")
    label = str(getattr(obj, "label", "") or "")
    if kind in {"animal", "npc_candidate"} or label in {"dog", "chicken", "cow"}:
        return True
    return bool(getattr(obj, "is_npc_candidate", False))


def entity_tiles(ram: np.ndarray, player_tile: Tile, *, radius: int = 10) -> Set[Tile]:
    """Tiles a live non-player sprite currently stands on (no padding)."""
    tiles: Set[Tile] = set()
    try:
        objects = game_objects(ram)
    except Exception:
        return tiles
    for obj in objects:
        tile = getattr(obj, "tile", None)
        if not tile or not _blocking_entity(obj):
            continue
        tx, ty = int(tile[0]), int(tile[1])
        if (tx, ty) == player_tile:
            continue
        if abs(tx - player_tile[0]) > radius or abs(ty - player_tile[1]) > radius:
            continue
        tiles.add((tx, ty))
    return tiles


def pad_entity_blocks(
    ram: np.ndarray, tiles: Set[Tile], player_tile: Tile
) -> Set[Tile]:
    """Ring-pad mountain sprites so BFS does not thread a wandering NPC.

    Mountain 0x10 only: an NPC BFS'd tight against is re-collided with on
    its next step. Padding that would seal every exit from the farmer's own
    tile is dropped — a boxed-in corridor stalls worse than one bumped NPC.
    """
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != 0x10 or not tiles:
        return set(tiles)
    padded = set(tiles)
    for bx, by in tiles:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                padded.add((bx + dx, by + dy))
    padded.discard(player_tile)
    neighbors = {
        neighbor_tile(player_tile[0], player_tile[1], direction)
        for direction in ("up", "down", "left", "right")
    }
    return set(tiles) if neighbors.issubset(padded) else padded


def entity_blocks(ram: np.ndarray, player_tile: Tile) -> Set[Tile]:
    """Reroute around live dog / NPC / animal sprites (not the player)."""
    return pad_entity_blocks(ram, entity_tiles(ram, player_tile), player_tile)


def replace_no_go(
    pathfinder: Pathfinder, previous: Set[Tile], nxt: Set[Tile]
) -> None:
    pathfinder.no_go_tiles.difference_update(previous)
    pathfinder.no_go_tiles.update(nxt)


def liftable_gate_toward(
    current_tile: Tile, ram: np.ndarray, wp: Waypoint
) -> Optional[Tuple[str, Tile, int]]:
    """If a liftable soft solid blocks progress toward wp, return face/tile/id."""
    goal = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
    dx = goal[0] - current_tile[0]
    dy = goal[1] - current_tile[1]
    faces: List[str] = []
    if abs(dx) >= abs(dy):
        faces.append("right" if dx > 0 else "left")
        if dy != 0:
            faces.append("down" if dy > 0 else "up")
    else:
        faces.append("down" if dy > 0 else "up")
        if dx != 0:
            faces.append("right" if dx > 0 else "left")
    for face in faces:
        nx, ny = neighbor_tile(current_tile[0], current_tile[1], face)
        if not (0 <= nx < 64 and 0 <= ny < 64):
            continue
        tid = int(get_tile_at(ram, nx, ny))
        if tid in LIFTABLE_TILES:
            return face, (nx, ny), tid
    return None


def _lift_face_toward(current_tile: Tile, cand: Tile) -> str:
    if cand[1] < current_tile[1]:
        return "up"
    if cand[1] > current_tile[1]:
        return "down"
    if cand[0] < current_tile[0]:
        return "left"
    if cand[0] > current_tile[0]:
        return "right"
    return "up"


def queue_lift_throw(
    action_queue: Deque[np.ndarray],
    current_tile: Tile,
    ram: np.ndarray,
    wp: Waypoint,
) -> Optional[str]:
    """Queue lift then throw, or skip when the gate is already open."""
    face = wp.action_face or "up"
    throw_face = _OPPOSITE_FACE.get(face, "down")
    if tilemaps_match(
        int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0, 0x00
    ):
        throw_face = "down"
    held = int(read_held_item(ram))
    face_tile = neighbor_tile(current_tile[0], current_tile[1], face)
    dx, dy = _DIR_DELTA.get(face, (0, 0))
    candidates = [face_tile, (face_tile[0] + dx, face_tile[1] + dy)]
    target: Optional[Tile] = None
    tid = 0
    lift_face = face
    for cand in candidates:
        if not (0 <= cand[0] < 64 and 0 <= cand[1] < 64):
            continue
        cand_tid = int(get_tile_at(ram, *cand))
        if cand_tid in LIFTABLE_TILES:
            target = cand
            tid = cand_tid
            lift_face = _lift_face_toward(current_tile, cand)
            break
    hold = max(12, int(wp.action_frames))
    settle = max(12, int(wp.action_cooldown))

    if held != 0:
        action_queue.extend(
            press_a_sequence(
                throw_face,
                face_frames=6,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        return f"throw held=0x{held:02X} face={throw_face}"

    if target is not None:
        action_queue.extend(
            press_a_sequence(
                lift_face,
                face_frames=8,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        action_queue.extend(
            press_a_sequence(
                throw_face,
                face_frames=6,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        return (
            f"lift_throw stand={current_tile} "
            f"target={target} tid=0x{tid:02X} face={lift_face}"
        )
    return None


def opportunistic_clear_waypoint(wp: Waypoint, face: str) -> Waypoint:
    """Waypoint-shaped lift_throw used when a weed seals a travel corridor."""
    return Waypoint(
        tilemap=wp.tilemap,
        target_px=wp.target_px,
        radius=wp.radius,
        action_on_arrive="lift_throw",
        action_face=face,
        action_frames=22,
        action_cooldown=24,
    )


def micro_center_action(cur_x: int, cur_y: int, target_px: Tuple[int, int]) -> np.ndarray:
    """Walk without B to center inside the current tile (no neighbor step)."""
    dx = target_px[0] - cur_x
    dy = target_px[1] - cur_y
    if abs(dx) >= abs(dy) and abs(dx) > 0:
        return make_action(right=dx > 0, left=dx < 0)
    if abs(dy) > 0:
        return make_action(down=dy > 0, up=dy < 0)
    return make_action()


class NavCorridor:
    """Corridor drive mixed into MultiMapNavTask.

    Stall, yield, close-range, run-direction, pin recovery, lift-throw, and
    entity-block sync. Waypoint order and phase dispatch stay on the task.
    """

    def _sync_farm_soft_blocks(self, ram: np.ndarray, tilemap: int) -> None:
        nxt = farm_soft_blocks(self._scanner, ram, tilemap)
        replace_no_go(self._pathfinder, self._farm_soft_blocks, nxt)
        self._farm_soft_blocks = nxt

    def _sync_entity_blocks(self, ram: np.ndarray) -> None:
        tile = self._navigator.current_tile
        self._entity_tiles = entity_tiles(ram, tile)
        nxt = pad_entity_blocks(ram, self._entity_tiles, tile)
        replace_no_go(self._pathfinder, self._entity_blocks, nxt)
        self._entity_blocks = nxt

    def _sync_travel_blocks(self, ram: np.ndarray, tilemap: int) -> None:
        self._sync_farm_soft_blocks(ram, tilemap)
        self._sync_entity_blocks(ram)

    def _entity_yield_result(self, wp: Waypoint) -> Optional[TaskResult]:
        """Hold still while a live sprite stands in the next tile.

        The mountain/path NPCs and the farm dog walk across proven corridors.
        Charging one burns the stasis budget and fails the leg; rerouting
        around it leaves the corridor. Both are worse than waiting a beat.
        """
        run_dir = (
            wp.run_direction if self._wp_index != self._run_dir_bail_wp else None
        )
        nxt = sprite_ahead(
            self._navigator, wp, self._entity_tiles, run_direction=run_dir
        )
        if nxt is None:
            self._yield_frames = 0
            return None
        self._yield_frames += 1
        if self._yield_frames > ENTITY_YIELD_FRAMES:
            # Parked, not passing. Let BFS route around the no-go it sits on.
            return None
        # A yield is not a pin: keep the stall guards off the wait.
        self._soft_solid_pin_frames = 0
        self._navigator.stasis = 0
        self._pixel_stuck = 0
        if self._yield_frames == 1:
            print(f"[MULTI_NAV] Yield to sprite at {nxt} (wp {self._wp_index + 1})")
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"yield to sprite at {nxt}",
        )

    def _recover_from_pin(self, world: WorldState, tilemap: int) -> TaskResult:
        """Break a soft-solid pin instead of failing the whole leg.

        Every pin seen in run13 was a concave cell the walk kept re-entering,
        not an impassable route: blocking the cell and replanning clears it.
        Waypoints are guides, so a second pin on the same one skips it —
        except on mountain 0x10, where skipping a corridor hop is how the
        farmer ends up in Gotz's dialogue.
        """
        self._pin_recoveries += 1
        cur = self._navigator.current_tile
        head = self._navigator.path[0] if self._navigator.path else None
        if head is not None and head != cur:
            self._pathfinder.temp_blocked.add(head)
        self._close_bail_wp = self._wp_index
        self._navigator.path = []
        self._navigator.stasis = 0
        self._soft_solid_pin_frames = 0
        self._close_latch.reset()
        self._sync_travel_blocks(world.ram, tilemap)
        skipped = False
        after = self._wp_index + 1
        nxt = self.waypoints[after] if after < len(self.waypoints) else None
        if (
            self._pin_recoveries >= 2
            and tilemap != 0x10
            and nxt is not None
            and self._waypoint_tilemap_matches(tilemap, nxt)
        ):
            self._advance_waypoint()
            skipped = True
        print(
            f"[MULTI_NAV] Pin recovery {self._pin_recoveries}/{PIN_RECOVERY_LIMIT} "
            f"at {cur} block={head} "
            f"{'skip to wp ' + str(self._wp_index + 1) if skipped else 'replan'}"
        )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"pin recovery {self._pin_recoveries} at {cur}",
        )

    # Back-compat for unit tests that assert weed no-go membership.
    @property
    def _farm_weed_blocks(self) -> Set[Tuple[int, int]]:
        return set(self._farm_soft_blocks)

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

    def _begin_lift_throw(self, world: WorldState, wp: Waypoint) -> TaskResult:
        reason = queue_lift_throw(
            self._action_queue, self._navigator.current_tile, world.ram, wp
        )
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

    def _drain_lift_throw(
        self, world: WorldState, wp: Waypoint, tilemap: int
    ) -> TaskResult:
        if self._action_queue:
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
        held = int(read_held_item(world.ram))
        face = wp.action_face or "up"
        target = neighbor_tile(
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

    def _run_direction_result(
        self, world: WorldState, wp: Waypoint
    ) -> Optional[TaskResult]:
        """Hold run_direction, or None when this waypoint is on BFS."""
        if not wp.run_direction or self._wp_index == self._run_dir_bail_wp:
            return None
        cur = self._navigator.current_pos
        d = wp.run_direction
        # Progress guard: a force-run / run_direction hop that stops
        # moving (grape return pins ~(505,633) on the mountain-exit
        # force-run) drops to BFS for this waypoint rather than holding
        # the direction into a wall forever.
        anchor = self._run_dir_anchor
        if anchor is None or abs(cur.x - anchor[0]) + abs(cur.y - anchor[1]) > 6:
            self._run_dir_anchor = (cur.x, cur.y)
            self._run_dir_stall = 0
        else:
            self._run_dir_stall += 1
        if self._run_dir_stall >= RUN_DIR_STALL_FRAMES:
            self._run_dir_bail_wp = self._wp_index
            self._run_dir_anchor = None
            self._run_dir_stall = 0
            self._navigator.path = []
            self._navigator.stasis = 0
            safe = self._safe_walk_action(world.ram, d)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(safe if safe is not None else make_action()),
                reason=f"run_direction {d} pinned; BFS for wp {self._wp_index}",
            )
        if d in {"left", "right"} and abs(cur.y - wp.target_px[1]) >= wp.radius:
            align = "down" if wp.target_px[1] > cur.y else "up"
            safe = self._safe_walk_action(world.ram, align)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(safe if safe is not None else make_action()),
            )
        if d in {"up", "down"} and abs(cur.x - wp.target_px[0]) >= wp.radius:
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

    def _close_range_result(
        self, world: WorldState, wp: Waypoint
    ) -> Optional[TaskResult]:
        """Walk straight from within ~5 tiles, or None so BFS can run."""
        if wp.is_exit or self._wp_index == self._close_bail_wp:
            return None
        cur = self._navigator.current_pos
        dx_close = abs(wp.target_px[0] - cur.x)
        dy_close = abs(wp.target_px[1] - cur.y)
        stasis = self._navigator.stasis
        if not (
            dx_close <= 80
            and dy_close <= 80
            and stasis < 40
            and self._pixel_stuck < 20
        ):  # ~5 tiles; bail if L/R pin
            return None
        dist = max(dx_close, dy_close)
        if self._close_latch.stalled(dist, moving=self._pixel_stuck == 0):
            self._close_bail_wp = self._wp_index
            self._navigator.path = []
            print(
                f"[MULTI_NAV] Close-range stalled at ({cur.x},{cur.y}) "
                f"dist={dist} — BFS for wp {self._wp_index + 1}"
            )
            self._close_latch.reset()
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"close-range stalled at wp {self._wp_index + 1}",
            )
        safe = close_range_action(
            self._pathfinder,
            self._navigator,
            world.ram,
            wp,
            stasis=stasis,
        )
        if safe is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(safe))
        # All neighbors blocked at close range → BFS / fail, no thrash.
        return None

    def _soft_solid_pin_result(
        self, world: WorldState, tilemap: int
    ) -> Optional[TaskResult]:
        """Count a nav pin and recover or fail. Always updates the counter."""
        held_now = int(read_held_item(world.ram))
        if self._phase != "nav":
            self._soft_solid_pin_frames = 0
            return None
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
        if self._soft_solid_pin_frames < pin_limit:
            return None
        if self._pin_recoveries < PIN_RECOVERY_LIMIT:
            return self._recover_from_pin(world, tilemap)
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=(
                f"soft_solid pin held=0x{held_now:02X} "
                f"pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y}) "
                f"stasis={self._navigator.stasis} "
                f"recoveries={self._pin_recoveries}"
            ),
        )

    def _opportunistic_lift_result(
        self, world: WorldState, wp: Waypoint
    ) -> Optional[TaskResult]:
        """Lift a soft solid that seals the hop, or None to keep failing closed."""
        if not (
            self.allow_opportunistic_clear
            and self._stuck_frames >= 30
            and self._stuck_frames < 90
        ):
            return None
        gate = liftable_gate_toward(self._navigator.current_tile, world.ram, wp)
        if gate is None or self._lift_throw_attempts >= 4:
            return None
        face, _target, _tid = gate
        reason = queue_lift_throw(
            self._action_queue,
            self._navigator.current_tile,
            world.ram,
            opportunistic_clear_waypoint(wp, face),
        )
        if reason is None:
            return None
        self._lift_throw_attempts += 1
        print(
            f"[MULTI_NAV] Opportunistic {reason} "
            f"(stuck={self._stuck_frames})"
        )
        self._phase = "lift_throw_drain"
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued
        return None


__all__ = [
    "CLOSE_RANGE_STALL_FRAMES",
    "ENTITY_SYNC_PERIOD",
    "ENTITY_YIELD_FRAMES",
    "NavCorridor",
    "PIN_RECOVERY_LIMIT",
    "RUN_DIR_STALL_FRAMES",
    "CloseRangeLatch",
    "close_range_action",
    "dirs_toward",
    "entity_blocks",
    "entity_tiles",
    "farm_soft_blocks",
    "hop_target",
    "liftable_gate_toward",
    "micro_center_action",
    "opportunistic_clear_waypoint",
    "pad_entity_blocks",
    "queue_lift_throw",
    "replace_no_go",
    "safe_walk_action",
    "sprite_ahead",
    "tile_blocks_charge",
]
