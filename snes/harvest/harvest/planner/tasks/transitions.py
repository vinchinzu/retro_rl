"""Map transition and building-exit tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Pathfinder,
    Navigator,
    make_action,
    get_pos_from_ram,
    tile_dist,
)
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.cow_geometry import BARN_DOOR_PX, BarnWalker, barn_cow_pixels
from harvest.tasks.primitives import dismiss_dialogue_result
from harvest.core.animal_status import read_held_item
from harvest.core.ram_catalog import field_spec, live_wram_base, read_ram_u8, read_ram_value
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

# $0025 fade_current_brightness. ScreenFadein ($8087CE) steps it to 15 after a
# door warp, and the pad ignores input until then: about 70 frames after a coop
# exit lands at (456, 360), and the tilemap already reads the farm well before.
ADDR_FADE_BRIGHTNESS = 0x0025
FULL_BRIGHTNESS = 0x0F


def screen_faded_in(ram: np.ndarray) -> bool:
    idx = ADDR_FADE_BRIGHTNESS + live_wram_base(ram)
    return idx >= len(ram) or int(ram[idx]) >= FULL_BRIGHTNESS


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
HOUSE_EXIT_UNSTICK_FROM_YEAR = 2     # 0-based: Year 3 only, so Years 1-2 keep their frames
HOUSE_EXIT_STILL_FRAMES = 30
HOUSE_EXIT_SIDESTEP_FRAMES = 24
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
# Coop door stand is (8,12). The tilemap's false-open vertical strip around
# x=5 traps generic BFS between the left service lane and the door.
COOP_EXIT_DOOR_X = 8 * 16 + 8
COOP_EXIT_CORRIDOR_Y = 12 * 16 + 8
COOP_EXIT_LEFT_LANE_X = 2 * 16 + 8
COOP_EXIT_FALSE_OPEN_X = 5 * 16 + 8
# Tiles with y >= 8 around x=5 are false-open; stay above y=7 while crossing.
COOP_EXIT_SAFE_CROSS_MAX_Y = 7 * 16 + 8
# A hen parked on the crossing row blocks the run east for good (pin
# LN_y23_coop_hen_block: one at (8,7) held the farmer at (7,7) through the
# whole timeout). Past the false-open strip the door takes the farmer from
# x=126, so a stalled crossing goes down instead.
COOP_EXIT_DESCEND_MIN_X = COOP_EXIT_FALSE_OPEN_X + 16
# Frames without a pixel of movement before a barn or coop exit changes lane.
EXIT_STALL_FRAMES = 45
_BARN_DOOR_NUDGE_XS = (120, 136, 128)  # the doorway spans x=113..143


def hands_are_clear(ram: np.ndarray) -> bool:
    """True when no debris/item is held (doors reject carries)."""
    if read_held_item(ram) != 0:
        return False
    idx = ADDR_PLAYER_STATE + live_wram_base(ram)
    if idx < len(ram) and (int(ram[idx]) & PLAYER_STATE_CARRYING_BIT):
        return False
    return True


def toss_held_actions(*, face: str = "down", step_away: bool = False) -> List[np.ndarray]:
    """Face a direction and throw whatever is in hands.

    Stationary face+A is the ROM-proven local drop (fence_flow corridor, crop
    planter). Long B-walks before A often pin the player against walls/debris
    or re-pickup the just-dropped stone (held 0x0D / weed 0x09) — that is what
    left power-on soaks stuck at return_home with hands full after CLEAR_FIELD.
    Optional short step-away is only for wall-hug door stands.
    """
    actions: List[np.ndarray] = []
    # Face and settle before A so the throw direction is latched.
    actions.extend(make_action(**{face: True}) for _ in range(10))
    actions.extend(make_action() for _ in range(4))
    if step_away:
        # One short nudge into open ground (door/fence edge only).
        actions.extend(make_action(**{face: True, "b": True}) for _ in range(6))
        actions.extend(make_action() for _ in range(4))
    # Hold face+A; plain A alone often fails mid-carry.
    actions.extend(make_action(**{face: True, "a": True}) for _ in range(20))
    actions.extend(make_action(a=True) for _ in range(8))
    actions.extend(make_action() for _ in range(16))
    return actions


def multi_face_toss_actions(*, prefer_south: bool = True) -> List[np.ndarray]:
    """Try each cardinal throw direction once — used before house entry.

    Prefer south/sides first so we do not seal a northern approach (fence y=31
    residual). ``up`` last as a last-resort face.
    """
    actions: List[np.ndarray] = []
    faces = ("down", "left", "right", "up") if prefer_south else ("down", "up", "left", "right")
    for face in faces:
        actions.extend(toss_held_actions(face=face, step_away=False))
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
    _exit_last_x: int = field(default=-1, init=False)
    _exit_stall: int = field(default=0, init=False)
    _exit_down_left: int = field(default=0, init=False)
    _path_frame: int = field(default=0, init=False)
    _target_seen_frames: int = field(default=0, init=False)
    _still_at: Tuple[int, int] = field(default=(-1, -1), init=False)
    _still: int = field(default=0, init=False)
    _side_dir: int = field(default=1, init=False)
    _side_left: int = field(default=0, init=False)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._still_at, self._still, self._side_dir, self._side_left = (-1, -1), 0, 1, 0
        self._exit_last_x = -1
        self._exit_stall = 0
        self._exit_down_left = 0
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

    def _sidestep_if_blocked(self, ram: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Year 3: step round whatever stands in the way of the blind exit walk.

        The walk only presses down and left/right, so a child or the dog on the farmer's column holds it for good (D r45 Summer D18:
        farmer (327,156), Eve (328,120), an NPC at (328,169); the exit timed out, the herd stayed unfed, cow happiness -25).
        After ``HOUSE_EXIT_STILL_FRAMES`` without moving it sidesteps for ``HOUSE_EXIT_SIDESTEP_FRAMES``, left first, then right.
        """

        pos = self._navigator.current_pos
        self._still = self._still + 1 if (pos.x, pos.y) == self._still_at else 0
        self._still_at = (pos.x, pos.y)
        if int(read_ram_value(ram, "year", raw=True) or 0) < HOUSE_EXIT_UNSTICK_FROM_YEAR:
            return action
        if self._side_left > 0:
            self._side_left -= 1
            return make_action(left=self._side_dir < 0, right=self._side_dir > 0)
        if self._still >= HOUSE_EXIT_STILL_FRAMES:
            self._still, self._side_dir, self._side_left = 0, -self._side_dir, HOUSE_EXIT_SIDESTEP_FRAMES
            return make_action(left=self._side_dir < 0, right=self._side_dir > 0)
        return action

    def _house_exit_action(self) -> np.ndarray:
        pos = self._navigator.current_pos
        if pos.y < HOUSE_EXIT_LOWER_Y - 10:
            # From the bed stand (70,86) the blind rightward walk runs into the
            # furniture at x~89 and stalls forever (rr-bu62.20). Step down off
            # that row for a while whenever the x walk stops making progress.
            if self._exit_down_left > 0:
                self._exit_down_left -= 1
                return make_action(down=True)
            if pos.x == self._exit_last_x:
                self._exit_stall += 1
            else:
                self._exit_stall = 0
                self._exit_last_x = pos.x
            if self._exit_stall >= 20:
                self._exit_stall = 0
                self._exit_down_left = 30
                return make_action(down=True)
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
            from harvest.core.npc_catalog import game_objects
            from harvest.tasks.nav import get_pos_from_ram

            pos = get_pos_from_ram(world.ram)
            others = [(o.kind, tuple(o.pixel)) for o in game_objects(world.ram) if not o.is_player][:6]
            return TaskResult(status=TaskStatus.FAILURE, reason=f"exit timeout pos=({pos.x},{pos.y}) tilemap=0x{int(world.ram[ADDR_TILEMAP]):02X} others={others}")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemaps_match(tilemap, self.target_tilemap) and self._step_count > 30:
            if self._target_ready(tilemap, world.ram):
                self._target_seen_frames += 1
                if self._target_seen_frames >= self.settle_frames:
                    return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
            else:
                self._target_seen_frames = 0
            # Mid-warp outdoor (y < 330): keep holding down. Neutral settle while
            # player_state is still transitioning freezes control (rr-bhr).
            if is_farm_tilemap(self.target_tilemap):
                pos = self._navigator.current_pos
                if pos.y < 330:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(down=True, b=True)),
                        reason="exit mid-warp push",
                    )
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="exit settle")
        if not tilemaps_match(tilemap, self.target_tilemap):
            self._target_seen_frames = 0

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1 and self._step_count <= self.dialog_frames:
            return dismiss_dialogue_result(self._step_count, buttons=("b", "a"), pulse_every=1)

        if is_house_tilemap(tilemap):
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._sidestep_if_blocked(world.ram, self._house_exit_action())))

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
    # Shed (and similar tight frames): walk the column, never B-run. Holding
    # B every frame at 2x clips into the doorjamb and then left/right-thrashes.
    walk_into_door: bool = False

    _step_count: int = field(default=0, init=False)
    _target_seen_frames: int = field(default=0, init=False)
    _stand_reached: bool = field(default=False, init=False)
    _door_nudge: int = field(default=0, init=False)
    _barn_walker: BarnWalker = field(default_factory=BarnWalker, init=False)
    _exit_last: Tuple[int, int] = field(default=(-1, -1), init=False)
    _exit_stall: int = field(default=0, init=False)
    _coop_descend: bool = field(default=False, init=False)
    _coop_detour_clear_y: int = field(default=0, init=False)
    _coop_detour_idx: int = field(default=-1, init=False)
    _coop_cross_y: int = field(default=0, init=False)
    _coop_cross_idx: int = field(default=0, init=False)
    _coop_climb_x: int = field(default=0, init=False)
    _hands_attempts: int = field(default=0, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    @staticmethod
    def _objects_near(ram: np.ndarray, radius: int = 48) -> list:
        """Sprites within ``radius`` px of the farmer: what a door that will not open has in front of it."""
        from harvest.core.npc_catalog import game_objects
        from harvest.tasks.nav import get_pos_from_ram

        pos = get_pos_from_ram(ram)
        return [(o.kind, tuple(o.pixel)) for o in game_objects(ram)
                if not o.is_player and abs(o.pixel[0] - pos.x) <= radius and abs(o.pixel[1] - pos.y) <= radius][:5]

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._target_seen_frames = 0
        self._stand_reached = False
        self._barn_walker.clear()
        self._exit_last, self._exit_stall, self._coop_descend = (-1, -1), 0, False
        self._coop_detour_clear_y, self._coop_detour_idx = 0, -1
        self._coop_cross_y, self._coop_cross_idx, self._coop_climb_x = 0, 0, 0
        self._hands_attempts = 0
        self._action_queue.clear()
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._pathfinder.extra_walkable.clear()

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._target_seen_frames = 0
        self._stand_reached = False
        self._barn_walker.clear()
        self._coop_detour_clear_y, self._coop_detour_idx = 0, -1
        self._coop_cross_y, self._coop_cross_idx, self._coop_climb_x = 0, 0, 0
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
        # First attempt: step away from the door then toss. Later: multi-face
        # stationary (stones/weeds often refuse a single wall-facing throw).
        face = "down" if self.direction == "up" else "up"
        if self.direction in {"left", "right"}:
            face = "down"
        if self._hands_attempts <= 1:
            self._action_queue.extend(toss_held_actions(face=face, step_away=True))
        else:
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))

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
        # Escape mid-wall clip on up-doors: strafe off the column then walk
        # south. Pure down while embedded in the house sprite often no-ops
        # (rr-6g7g residual y≈372), leaving soft-restand thrashing in place.
        if self.direction == "up" and self.stand_tile is not None:
            self._action_queue.extend(make_action(left=True) for _ in range(8))
            self._action_queue.extend(make_action(down=True, b=True) for _ in range(28))
            self._action_queue.extend(make_action(right=True) for _ in range(10))
            self._action_queue.extend(make_action(down=True, b=True) for _ in range(20))
            self._action_queue.extend(make_action() for _ in range(6))

    def _door_push_action(self) -> np.ndarray:
        """Align to the door column/row, then press into the doorway.

        Near the threshold, alternate walk (no B) with run. Blind B-hold can
        overshoot the farmhouse door into a mid-wall clip at y≈372 without a
        map change (rr-6g7g residual after hands-clear). Shed doors also clip
        when B is held every frame at 2x — ``walk_into_door`` is walk+idle only.
        """
        pos = self._navigator.current_pos
        run = not self.walk_into_door
        if self.door_align_px is not None:
            if self.direction in {"up", "down"}:
                if abs(pos.x - self.door_align_px) > self.door_align_tolerance:
                    return make_action(
                        right=pos.x < self.door_align_px,
                        left=pos.x > self.door_align_px,
                        b=run,
                    )
            else:
                if abs(pos.y - self.door_align_px) > self.door_align_tolerance:
                    return make_action(
                        down=pos.y < self.door_align_px,
                        up=pos.y > self.door_align_px,
                        b=run,
                    )
        if self.walk_into_door:
            # 8 walk / 4 idle so the ROM can accept the threshold. Continuous
            # Up (or B+dir) at 2x embeds in the shed jamb.
            if (self._step_count % 12) >= 8:
                return make_action()
            return make_action(**{self.direction: True})
        # Walk into the door most frames; brief run every 4th frame for speed.
        use_run = (self._step_count % 4) == 0
        return make_action(**{self.direction: True, "b": use_run})

    def _note_exit_stall(self) -> None:
        pos = self._navigator.current_pos
        here = (pos.x, pos.y)
        # A push into a body jitters one pixel (y 347 <-> 348), so that counts as still.
        still = abs(here[0] - self._exit_last[0]) + abs(here[1] - self._exit_last[1]) <= 1
        self._exit_stall = self._exit_stall + 1 if still else 0
        self._exit_last = here

    def _barn_exit_action(self, ram: np.ndarray) -> np.ndarray:
        """Walk the barn map to the door, then push down through it.

        A calf parked in the doorway overlaps the door column (Y2 Spring D30, a rescue): a push that stands
        still sidesteps along the doorway, which spans x=113..143.
        """
        pos = self._navigator.current_pos
        if abs(pos.x - BARN_DOOR_PX[0]) > 15 or pos.y < BARN_DOOR_PX[1] - 4:
            self._exit_stall, self._door_nudge = 0, 0
            action = self._barn_walker.step((pos.x, pos.y), BARN_DOOR_PX, tolerance=4, cows=barn_cow_pixels(ram))
            if action is not None:
                return action
        self._note_exit_stall()
        if self._exit_stall >= EXIT_STALL_FRAMES:
            self._door_nudge, self._exit_stall = self._door_nudge + 1, 0
        if self._door_nudge:
            tx = _BARN_DOOR_NUDGE_XS[(self._door_nudge - 1) % len(_BARN_DOOR_NUDGE_XS)]
            if abs(pos.x - tx) > 2:
                return make_action(left=pos.x > tx, right=pos.x < tx)
            return make_action(down=True)
        if self.door_align_px is not None and abs(pos.x - self.door_align_px) > self.door_align_tolerance:
            return make_action(left=pos.x > self.door_align_px, right=pos.x < self.door_align_px)
        return make_action(down=True)  # a run pulse here can shove a cow through the door

    def _coop_exit_action(self) -> np.ndarray:
        """Exit via top-of-false-open bypass, then the door column.

        Bottom-left shipping bin blocks a direct south path from the left
        service lane, and x=5 tiles at y>=8 are false-open. Recorded-safe
        route: climb above y=128, run east to the door x, then press down.
        """
        pos = self._navigator.current_pos
        self._note_exit_stall()
        # Shipping-bin interact pocket.
        if pos.x <= 50 and pos.y >= 165:
            if abs(pos.x - 38) > 2:
                return make_action(right=pos.x < 38, left=pos.x > 38, b=True)
            return make_action(up=True, b=True)

        detour_xs = (164, 108, 188, 92)
        cross_ys = (62, 98, 76, 114)

        if self._coop_detour_clear_y > 0:
            if pos.y >= self._coop_detour_clear_y:
                self._coop_detour_clear_y = 0
            else:
                target_x = detour_xs[self._coop_detour_idx % len(detour_xs)]
                if self._exit_stall >= EXIT_STALL_FRAMES:
                    self._coop_detour_idx += 1
                    self._exit_stall = 0
                    target_x = detour_xs[self._coop_detour_idx % len(detour_xs)]
                if abs(pos.x - target_x) > 3:
                    return make_action(right=pos.x < target_x, left=pos.x > target_x, b=True)
                return make_action(down=True, b=True)

        if (
            self._exit_stall >= EXIT_STALL_FRAMES
            and COOP_EXIT_DESCEND_MIN_X <= pos.x < COOP_EXIT_DOOR_X - 4
        ):
            self._coop_descend, self._exit_stall = not self._coop_descend, 0
        if self._coop_descend and pos.x >= COOP_EXIT_DESCEND_MIN_X:
            return make_action(down=True, b=True)

        # Climb above the false-open band before crossing east.
        if pos.x < COOP_EXIT_DOOR_X - 4 and pos.y > COOP_EXIT_SAFE_CROSS_MAX_Y:
            if self._coop_climb_x > 0:
                if abs(pos.x - self._coop_climb_x) <= 3 or self._exit_stall >= EXIT_STALL_FRAMES:
                    self._coop_climb_x = 0
                    self._exit_stall = 0
                else:
                    return make_action(
                        right=pos.x < self._coop_climb_x,
                        left=pos.x > self._coop_climb_x,
                        b=True,
                    )
            if self._exit_stall >= EXIT_STALL_FRAMES:
                self._coop_climb_x = 92 if pos.x < 70 else 42
                self._exit_stall = 0
                return make_action(
                    right=pos.x < self._coop_climb_x,
                    left=pos.x > self._coop_climb_x,
                    b=True,
                )
            return make_action(up=True, b=True)

        if abs(pos.x - COOP_EXIT_DOOR_X) > 4:
            if self._coop_cross_y > 0:
                if abs(pos.y - self._coop_cross_y) <= 3:
                    self._coop_cross_y = 0
                elif self._exit_stall >= EXIT_STALL_FRAMES:
                    self._coop_cross_idx += 1
                    self._coop_cross_y = cross_ys[self._coop_cross_idx % len(cross_ys)]
                    self._exit_stall = 0
                    return make_action(
                        down=pos.y < self._coop_cross_y,
                        up=pos.y > self._coop_cross_y,
                        b=True,
                    )
                else:
                    return make_action(
                        down=pos.y < self._coop_cross_y,
                        up=pos.y > self._coop_cross_y,
                        b=True,
                    )
            if self._exit_stall >= EXIT_STALL_FRAMES:
                self._coop_cross_y = cross_ys[self._coop_cross_idx % len(cross_ys)]
                if abs(pos.y - self._coop_cross_y) <= 4:
                    self._coop_cross_idx += 1
                    self._coop_cross_y = cross_ys[self._coop_cross_idx % len(cross_ys)]
                self._exit_stall = 0
                return make_action(
                    down=pos.y < self._coop_cross_y,
                    up=pos.y > self._coop_cross_y,
                    b=True,
                )
            return make_action(
                right=pos.x < COOP_EXIT_DOOR_X,
                left=pos.x > COOP_EXIT_DOOR_X,
                b=True,
            )

        if self._exit_stall >= EXIT_STALL_FRAMES:
            self._coop_detour_clear_y = min(196, pos.y + 28)
            self._coop_detour_idx += 1
            self._exit_stall = 0
            target_x = detour_xs[self._coop_detour_idx % len(detour_xs)]
            return make_action(right=pos.x < target_x, left=pos.x > target_x, b=True)
        return make_action(down=True, b=True)

    def _target_ready(self, tilemap: int, ram: np.ndarray) -> bool:
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
            if not screen_faded_in(ram):
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
                    f"tile={self._navigator.current_tile} tilemap=0x{tilemap:02X} "
                    f"held=0x{int(read_ram_value(world.ram, 'held_item')):02X} lock={int(world.ram[ADDR_INPUT_LOCK])} "
                    f"near={self._objects_near(world.ram)}"
                ),
            )
        if (
            self.origin_tilemap is not None
            and self._step_count == 1
            and not tilemaps_match(tilemap, self.origin_tilemap)
            and not self._is_target_tilemap(tilemap)
        ):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected origin 0x{self.origin_tilemap:02X}, got 0x{tilemap:02X}",
            )
        if self._is_target_tilemap(tilemap) and self._step_count >= self.min_frames_before_success:
            if self._target_ready(tilemap, world.ram):
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
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._barn_exit_action(world.ram)))
        if (
            self.origin_tilemap == COOP_TILEMAP
            and is_farm_tilemap(self.target_tilemap)
            and tilemap == COOP_TILEMAP
        ):
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._coop_exit_action()))

        if self.stand_tile is not None and not self._is_target_tilemap(tilemap):
            # Only arm re-stand while in door-push mode. While correcting, keep
            # walking (scripted south nudge and/or BFS) — do not re-clear path.
            if self._stand_reached and self._overshot_door():
                print(
                    f"[DOOR] Overshoot on {self.name} "
                    f"pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y}); "
                    "re-standing"
                )
                self._reset_to_stand()
            # Mid-wall pin above the outdoor stand without crossing the deep
            # overshoot line (base house y≈360–380). Cooldown via stasis reset
            # in _reset_to_stand so we do not thrash every frame at stasis=91.
            elif (
                self.direction == "up"
                and self._stand_reached
                and self._navigator.stasis > 120
                and self.stand_tile is not None
            ):
                stand_y = self.stand_tile[1] * 16 + 8
                if self._navigator.current_pos.y < stand_y - 48:
                    print(
                        f"[DOOR] Soft mid-wall restand on {self.name} "
                        f"pos=({self._navigator.current_pos.x},"
                        f"{self._navigator.current_pos.y}) stasis="
                        f"{self._navigator.stasis}"
                    )
                    self._reset_to_stand()
            # Drain scripted re-stand walk before BFS (queued by _reset_to_stand).
            if self._action_queue:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._action_queue.popleft()),
                    reason="re-stand after overshoot",
                )
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
