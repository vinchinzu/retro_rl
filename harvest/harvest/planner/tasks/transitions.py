"""Map transition and building-exit tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.farm_clearer import (
    TileScanner,
    Pathfinder,
    Navigator,
    make_action,
    get_pos_from_ram,
    tile_dist,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
)
from harvest.tasks.primitives import dismiss_dialogue_result
from harvest.core.animal_status import read_held_item
from harvest.core.ram_catalog import field_spec, live_wram_base, read_ram_u8
from harvest.planner.day_plan_status import (
    BARN_TILEMAP,
    COOP_TILEMAP,
    FARM_TILEMAP,
    HOUSE_TILEMAPS,
    SHED_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
    tilemaps_match,
)

# ── ExitBuildingTask ──────────────────────────────────────────────

# House exit path: (direction, frames) pairs.
# Player starts near px(136,120) after the morning wake-up in farmhouse
# interiors. The same relocalized path works for the base house (0x15) and
# first remodel (0x16); 0x17 is registered for the second remodel but should be
# verified once that save exists.
HOUSE_EXIT_PATH: List[Tuple[str, int]] = [
    ("right", 80),
    ("down", 120),
    ("left", 50),
    ("down", 240),
]
HOUSE_EXIT_TILE: Tuple[int, int] = (8, 12)
HOUSE_EXIT_RIGHT_X = 216
HOUSE_EXIT_LOWER_Y = 201
HOUSE_EXIT_DOOR_X = 136
# Outdoor house threshold — stand here, then walk up into the doorway.
HOUSE_ENTER_STAND_TILE: Tuple[int, int] = (8, 26)
HOUSE_ENTER_DOOR_X = 136
# North of the door tile (~y=344) without a map change means we walked into
# the wall past the threshold; re-stand instead of blind-pushing forever.
HOUSE_ENTER_OVERSHOOT_Y = 328
# Shed outdoor threshold (route ends at px 424,489).
SHED_ENTER_STAND_TILE: Tuple[int, int] = (26, 30)
SHED_ENTER_DOOR_X = 424
# Door is a few tiles north of the shed stand (~y=489).
SHED_ENTER_OVERSHOOT_Y = 440
ADDR_PLAYER_STATE = field_spec("player_state").address
PLAYER_STATE_TRANSITION_BIT = 0x80
PLAYER_STATE_CARRYING_BIT = 0x02
BARN_EXIT_TROUGH_X = 113
BARN_EXIT_TROUGH_MAX_X = 130
BARN_EXIT_RIGHT_AISLE_X = 204
BARN_EXIT_BYPASS_X = 216
BARN_EXIT_LOWER_Y = 20 * 16 + 8
BARN_EXIT_DOOR_X = 8 * 16 + 8
# Coop door stand is (8,12). The tilemap's false-open vertical strip around
# x=5 traps generic BFS between the left service lane and the door.
COOP_EXIT_DOOR_X = 8 * 16 + 8
COOP_EXIT_CORRIDOR_Y = 12 * 16 + 8
COOP_EXIT_LEFT_LANE_X = 2 * 16 + 8
COOP_EXIT_FALSE_OPEN_X = 5 * 16 + 8
# Tiles with y >= 8 around x=5 are false-open; stay above y=7 while crossing.
COOP_EXIT_SAFE_CROSS_MAX_Y = 7 * 16 + 8


def hands_are_clear(ram: np.ndarray) -> bool:
    """True when no debris/item is held (doors reject carries)."""
    if read_held_item(ram) != 0:
        return False
    idx = ADDR_PLAYER_STATE + live_wram_base(ram)
    if idx < len(ram) and (int(ram[idx]) & PLAYER_STATE_CARRYING_BIT):
        return False
    return True


def toss_held_actions(*, face: str = "down") -> List[np.ndarray]:
    """Face away from a doorway and throw whatever is in hands.

    Matches farm-clearer and toss_* recordings: face, pause, face+A hold,
    settle. A short walk in the throw direction helps when standing against
    a wall (house door / fence edge).
    """
    actions: List[np.ndarray] = []
    actions.extend(make_action(**{face: True}) for _ in range(6))
    actions.extend(make_action() for _ in range(4))
    # Step into open ground so the throw is not blocked by the wall.
    actions.extend(make_action(**{face: True, "b": True}) for _ in range(10))
    actions.extend(make_action() for _ in range(4))
    # Hold face+A like toss_bush_pond; plain A alone often fails mid-carry.
    actions.extend(make_action(**{face: True, "a": True}) for _ in range(18))
    actions.extend(make_action(a=True) for _ in range(10))
    actions.extend(make_action() for _ in range(24))
    actions.extend(make_action(**{face: True, "b": True}) for _ in range(8))
    actions.extend(make_action() for _ in range(10))
    return actions


def multi_face_toss_actions() -> List[np.ndarray]:
    """Try each cardinal throw direction once — used before house entry."""
    actions: List[np.ndarray] = []
    for face in ("down", "left", "right", "up"):
        actions.extend(toss_held_actions(face=face))
    return actions


@dataclass
class ExitBuildingTask(Task):
    """Dismiss morning dialog, then relocalize and walk out of the house."""

    name: str = "exit_building"
    target_tilemap: int = 0x00
    dialog_frames: int = 120
    timeout: int = 600
    settle_frames: int = 30

    _step_count: int = field(default=0, init=False)
    _path_index: int = field(default=0, init=False)
    _path_frame: int = field(default=0, init=False)
    _target_seen_frames: int = field(default=0, init=False)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._path_index = 0
        self._path_frame = 0
        self._target_seen_frames = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._pathfinder.extra_walkable.clear()

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._target_seen_frames = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._pathfinder.extra_walkable.clear()

    def _fallback_action(self, goal: Tuple[int, int]) -> np.ndarray:
        current = self._navigator.current_tile
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        if abs(dx) >= abs(dy):
            primary = "right" if dx > 0 else "left"
            secondary = "down" if dy > 0 else "up"
        else:
            primary = "down" if dy > 0 else "up"
            secondary = "right" if dx > 0 else "left"
        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        stasis = self._navigator.stasis
        if stasis < 30:
            direction = primary
        elif stasis < 60:
            direction = secondary
        elif stasis < 90:
            direction = opposites[primary]
        else:
            direction = opposites[secondary]
        return make_action(**{direction: True, "b": True})

    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        self._pathfinder.extra_walkable.add(goal)
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal, tolerance=2):
            return self._navigator.center_on_tile(goal, tolerance=2)

        if self._navigator.stasis > 120 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, goal)
            if path is None:
                return self._fallback_action(goal)
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            return self._fallback_action(goal)
        return action

    def _house_exit_action(self) -> np.ndarray:
        pos = self._navigator.current_pos
        if pos.y < HOUSE_EXIT_LOWER_Y - 10:
            if pos.x < HOUSE_EXIT_RIGHT_X - 8:
                return make_action(right=True, b=True)
            return make_action(down=True, b=True)

        if pos.x > HOUSE_EXIT_DOOR_X + 4:
            return make_action(left=True, b=True)
        if pos.x < HOUSE_EXIT_DOOR_X - 4:
            return make_action(right=True, b=True)
        return make_action(down=True, b=True)

    def _target_ready(self, tilemap: int, ram: Optional[np.ndarray] = None) -> bool:
        if not tilemaps_match(tilemap, self.target_tilemap):
            return False
        if is_farm_tilemap(self.target_tilemap):
            pos = self._navigator.current_pos
            if pos.x == 0 and pos.y == 0:
                return False
            # The farmhouse exit flips tilemap to farm before the transition
            # finishes and before outdoor coordinates settle near the house
            # frontage. Handing off early lets nav target stale door tiles and
            # can immediately re-enter the house.
            # Settled outdoor y after a house exit is ~344 (door) for base and
            # remodel fronts; mid-warp values sit well below that.
            if pos.y < 330:
                return False
            if ram is not None:
                idx = ADDR_PLAYER_STATE + live_wram_base(ram)
                if idx < len(ram) and (int(ram[idx]) & PLAYER_STATE_TRANSITION_BIT):
                    return False
        return True

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._navigator.update(world.ram)

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="exit timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemaps_match(tilemap, self.target_tilemap) and self._step_count > 30:
            if self._target_ready(tilemap, world.ram):
                self._target_seen_frames += 1
                if self._target_seen_frames >= self.settle_frames:
                    return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
            else:
                self._target_seen_frames = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="exit settle")
        if not tilemaps_match(tilemap, self.target_tilemap):
            self._target_seen_frames = 0

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1 and self._step_count <= self.dialog_frames:
            return dismiss_dialogue_result(self._step_count, buttons=("b", "a"), pulse_every=1)

        if is_house_tilemap(tilemap):
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._house_exit_action()))

        # At the exit tile, or in another farm building fallback, walk through
        # the south-facing doorway.
        action = make_action(down=True, b=True)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


@dataclass
class DirectionalTransitionTask(Task):
    """Walk to a door stand, clear hands if needed, then push through a doorway.

    Building doors reject carried debris. Blind direction holds also overshoot
    past the threshold without entering — recover by re-standing on the door
    tile and retrying with column alignment.
    """

    name: str = "directional_transition"
    direction: str = "down"
    target_tilemap: int = FARM_TILEMAP
    target_tilemaps: Optional[Tuple[int, ...]] = None
    origin_tilemap: Optional[int] = None
    timeout: int = 600
    min_frames_before_success: int = 15
    settle_frames: int = 0
    stand_tile: Optional[Tuple[int, int]] = None
    stand_tolerance: int = 0
    target_stand_tile: Optional[Tuple[int, int]] = None
    target_stand_tolerance: int = 0
    # Keep the player on this pixel axis while pushing into the door.
    door_align_px: Optional[int] = None
    door_align_tolerance: int = 4
    # If travel goes past this limit without a map change, re-stand.
    overshoot_limit_px: Optional[int] = None
    require_empty_hands: bool = False
    clear_hands_limit: int = 4

    _step_count: int = field(default=0, init=False)
    _target_seen_frames: int = field(default=0, init=False)
    _stand_reached: bool = field(default=False, init=False)
    _barn_exit_bypass: bool = field(default=False, init=False)
    _hands_attempts: int = field(default=0, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._target_seen_frames = 0
        self._stand_reached = False
        self._barn_exit_bypass = False
        self._hands_attempts = 0
        self._action_queue.clear()
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._pathfinder.extra_walkable.clear()

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._target_seen_frames = 0
        self._stand_reached = False
        self._barn_exit_bypass = False
        self._action_queue.clear()
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._pathfinder.extra_walkable.clear()

    def can_start(self, world: WorldState) -> bool:
        return True

    def _fallback_action(self, goal: Tuple[int, int]) -> np.ndarray:
        current = self._navigator.current_tile
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return make_action(**{direction: True, "b": True})

    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        self._pathfinder.extra_walkable.add(goal)
        if tile_dist(self._navigator.current_tile, goal) <= self.stand_tolerance:
            return self._navigator.center_on_tile(goal, tolerance=max(1, self.stand_tolerance))

        if self._navigator.stasis > 120 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, goal)
            if path is None:
                return self._fallback_action(goal)
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            return self._fallback_action(goal)
        return action

    def _queue_clear_hands(self) -> None:
        face = "down" if self.direction == "up" else "up"
        if self.direction in {"left", "right"}:
            face = "down"
        self._action_queue.extend(toss_held_actions(face=face))

    def _overshot_door(self) -> bool:
        if self.overshoot_limit_px is None:
            return False
        pos = self._navigator.current_pos
        if self.direction == "up":
            return pos.y < self.overshoot_limit_px
        if self.direction == "down":
            return pos.y > self.overshoot_limit_px
        if self.direction == "left":
            return pos.x < self.overshoot_limit_px
        if self.direction == "right":
            return pos.x > self.overshoot_limit_px
        return False

    def _reset_to_stand(self) -> None:
        self._stand_reached = False
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def _door_push_action(self) -> np.ndarray:
        """Align to the door column/row, then press into the doorway."""
        pos = self._navigator.current_pos
        if self.door_align_px is not None:
            if self.direction in {"up", "down"}:
                if abs(pos.x - self.door_align_px) > self.door_align_tolerance:
                    return make_action(
                        right=pos.x < self.door_align_px,
                        left=pos.x > self.door_align_px,
                        b=True,
                    )
            else:
                if abs(pos.y - self.door_align_px) > self.door_align_tolerance:
                    return make_action(
                        down=pos.y < self.door_align_px,
                        up=pos.y > self.door_align_px,
                        b=True,
                    )
        return make_action(**{self.direction: True, "b": True})

    def _barn_exit_action(self) -> np.ndarray:
        pos = self._navigator.current_pos
        if self._barn_exit_bypass:
            if pos.y >= BARN_EXIT_LOWER_Y - 2:
                self._barn_exit_bypass = False
            elif pos.x < BARN_EXIT_BYPASS_X - 3:
                return make_action(right=True, b=True)
            elif pos.x > BARN_EXIT_BYPASS_X + 3:
                return make_action(left=True, b=True)
            else:
                return make_action(down=True, b=True)

        if pos.y < BARN_EXIT_LOWER_Y - 2:
            if pos.x <= BARN_EXIT_TROUGH_MAX_X:
                if abs(pos.x - BARN_EXIT_TROUGH_X) > 2:
                    return make_action(right=pos.x < BARN_EXIT_TROUGH_X, left=pos.x > BARN_EXIT_TROUGH_X, b=True)
                return make_action(down=True, b=True)
            # Cow stalls between the right aisle and the door frequently pin
            # exit nav with little tile-stasis growth. Bypass sooner and with
            # a wider aisle band so EXIT_BARN does not burn the day timeout.
            near_right_aisle = abs(pos.x - BARN_EXIT_RIGHT_AISLE_X) <= 16
            if self._navigator.stasis > 45 and near_right_aisle:
                self._barn_exit_bypass = True
                return make_action(right=True, b=True)
            if abs(pos.x - BARN_EXIT_RIGHT_AISLE_X) > 3:
                return make_action(
                    right=pos.x < BARN_EXIT_RIGHT_AISLE_X,
                    left=pos.x > BARN_EXIT_RIGHT_AISLE_X,
                    b=True,
                )
            return make_action(down=True, b=True)
        if pos.x > BARN_EXIT_DOOR_X + 4:
            return make_action(left=True, b=True)
        if pos.x < BARN_EXIT_DOOR_X - 4:
            return make_action(right=True, b=True)
        return make_action(down=True, b=True)

    def _coop_exit_action(self) -> np.ndarray:
        """Exit via top-of-false-open bypass, then the door column.

        Bottom-left shipping bin blocks a direct south path from the left
        service lane, and x=5 tiles at y>=8 are false-open. Recorded-safe
        route: climb above y=128, run east to the door x, then press down.
        """
        pos = self._navigator.current_pos
        # Shipping-bin interact pocket.
        if pos.x <= 50 and pos.y >= 165:
            if abs(pos.x - 38) > 2:
                return make_action(right=pos.x < 38, left=pos.x > 38, b=True)
            return make_action(up=True, b=True)
        # Climb above the false-open band before crossing east.
        if pos.x < COOP_EXIT_DOOR_X - 4 and pos.y > COOP_EXIT_SAFE_CROSS_MAX_Y:
            return make_action(up=True, b=True)
        if abs(pos.x - COOP_EXIT_DOOR_X) > 4:
            return make_action(
                right=pos.x < COOP_EXIT_DOOR_X,
                left=pos.x > COOP_EXIT_DOOR_X,
                b=True,
            )
        return make_action(down=True, b=True)

    def _target_ready(self, tilemap: int) -> bool:
        if not self._is_target_tilemap(tilemap):
            return False
        if self.target_stand_tile is not None:
            pos = self._navigator.current_pos
            if pos.x == 0 and pos.y == 0:
                return False
            if tile_dist(self._navigator.current_tile, self.target_stand_tile) > self.target_stand_tolerance:
                return False
        if (
            is_farm_tilemap(self.target_tilemap)
            and self.origin_tilemap in {SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP}
        ):
            pos = self._navigator.current_pos
            # The tilemap byte flips before the game copies outdoor farm
            # coordinates in from some building exits. Do not hand off to the
            # next nav phase while the player still has indoor-local coords.
            if pos.x < 250 or pos.y < 300:
                return False
        return True

    def _is_target_tilemap(self, tilemap: int) -> bool:
        if tilemaps_match(tilemap, self.target_tilemap):
            return True
        return self.target_tilemaps is not None and any(
            tilemaps_match(tilemap, target) for target in self.target_tilemaps
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        self._navigator.update(world.ram)

        if self._action_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
                reason="clear hands for door",
            )

        if self._step_count > self.timeout:
            pos = self._navigator.current_pos
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"directional transition timeout pos=({pos.x},{pos.y}) "
                    f"tile={self._navigator.current_tile} tilemap=0x{tilemap:02X}"
                ),
            )
        if (
            self.origin_tilemap is not None
            and self._step_count == 1
            and not tilemaps_match(tilemap, self.origin_tilemap)
        ):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected origin 0x{self.origin_tilemap:02X}, got 0x{tilemap:02X}",
            )
        if self._is_target_tilemap(tilemap) and self._step_count >= self.min_frames_before_success:
            if self._target_ready(tilemap):
                self._target_seen_frames += 1
                if self._target_seen_frames >= self.settle_frames:
                    return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
            else:
                self._target_seen_frames = 0
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._door_push_action()),
                    reason="transition target settle",
                )
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="transition settle")
        if not self._is_target_tilemap(tilemap):
            self._target_seen_frames = 0

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count)

        if (
            self.require_empty_hands
            and not self._is_target_tilemap(tilemap)
            and not hands_are_clear(world.ram)
        ):
            if self._hands_attempts >= self.clear_hands_limit:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"hands not clear for door (held=0x{read_held_item(world.ram):02X})",
                )
            self._hands_attempts += 1
            self._queue_clear_hands()
            print(
                f"[DOOR] Clearing hands before {self.name} "
                f"(attempt {self._hands_attempts}/{self.clear_hands_limit} "
                f"held=0x{read_held_item(world.ram):02X})"
            )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
                reason="clear hands for door",
            )

        if (
            self.origin_tilemap == BARN_TILEMAP
            and is_farm_tilemap(self.target_tilemap)
            and tilemap == BARN_TILEMAP
        ):
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._barn_exit_action()))
        if (
            self.origin_tilemap == COOP_TILEMAP
            and is_farm_tilemap(self.target_tilemap)
            and tilemap == COOP_TILEMAP
        ):
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._coop_exit_action()))

        if self.stand_tile is not None and not self._is_target_tilemap(tilemap):
            if self._overshot_door():
                print(
                    f"[DOOR] Overshoot on {self.name} "
                    f"pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y}); "
                    "re-standing"
                )
                self._reset_to_stand()
            if not self._stand_reached:
                action = self._navigate_to_tile(world.ram, self.stand_tile)
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
                self._stand_reached = True

        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._door_push_action()),
        )


__all__ = [
    "ExitBuildingTask",
    "DirectionalTransitionTask",
    "hands_are_clear",
    "toss_held_actions",
    "multi_face_toss_actions",
    "HOUSE_ENTER_STAND_TILE",
    "HOUSE_ENTER_DOOR_X",
    "HOUSE_ENTER_OVERSHOOT_Y",
    "SHED_ENTER_STAND_TILE",
    "SHED_ENTER_DOOR_X",
    "SHED_ENTER_OVERSHOOT_Y",
]
