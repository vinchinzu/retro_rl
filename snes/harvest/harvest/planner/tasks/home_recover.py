"""Failure / thrash recovery policy for ReturnHomeTask.

``home_approach`` owns approach geometry; this module owns what to do when
child phases fail (exit-to-farm dialogue mash, enter-house south recover,
nav near-door force enter, mid-yard re-nav, south softlock escape).

Pure helpers only — ReturnHomeTask applies decisions (counters, queues, prints).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.planner.tasks.home_approach import (
    EAST_AROUND_FENCE_X,
    deep_south_of_house,
    south_of_fence_wall,
)
from harvest.planner.tasks.transitions import (
    multi_face_toss_actions,
    toss_held_actions,
)
from harvest.tasks.nav import Point, make_action


class RecoverKind(str, Enum):
    """What ReturnHomeTask should do after a child FAILURE/BLOCKED."""

    QUEUE_EXIT_MASH = "queue_exit_mash"
    FAIL_EXIT = "fail_exit"
    RETRY_ENTER_SOUTH = "retry_enter_south"
    RETRY_ENTER_RESTART = "retry_enter_restart"
    FORCE_ENTER = "force_enter"
    MID_YARD_RENAV = "mid_yard_renav"
    SOUTH_ESCAPE = "south_escape"
    RETRY_DROP_THEN_NAV = "retry_drop_then_nav"
    HARD_FAIL = "hard_fail"


@dataclass(frozen=True)
class RecoverDecision:
    kind: RecoverKind
    reason: str = ""
    # SOUTH_ESCAPE: prefer far-east thrash (west then north).
    far_east: bool = False
    # Enter-fail: reset drop budget when reason mentions hands.
    hands_not_clear: bool = False
    # HARD_FAIL after hands-full path mutates phase before reporting.
    clear_task: bool = False
    set_phase: Optional[str] = None
    # SOUTH_ESCAPE log / result.reason variant (drop vs house nav).
    escape_from_drop: bool = False


def exit_to_farm_recover_actions() -> List[np.ndarray]:
    """Multi-face A/B mash used after ExitToFarm dialogue/unknown thrash."""
    actions: List[np.ndarray] = []
    for face in ("down", "left", "right", "up"):
        actions.extend(make_action(**{face: True}) for _ in range(4))
        actions.extend(make_action(a=True, b=True) for _ in range(8))
        actions.extend(make_action() for _ in range(4))
    return actions


def enter_fail_south_recovery_actions(
    pos: Point, front: Point
) -> Optional[List[np.ndarray]]:
    """South walk-out when enter failed north of the outdoor stand.

    Returns None when geometry does not apply (``pos.y >= front.y - 16``).
    """
    if pos.y >= front.y - 16:
        return None
    actions: List[np.ndarray] = []
    actions.extend(make_action(left=True) for _ in range(10))
    actions.extend(make_action(down=True, b=True) for _ in range(40))
    actions.extend(make_action(right=True) for _ in range(12))
    actions.extend(make_action(down=True, b=True) for _ in range(24))
    actions.extend(make_action() for _ in range(8))
    return actions


def decide_exit_to_farm_failure(
    *,
    retries: int,
    retry_limit: int,
    reason: str,
) -> RecoverDecision:
    if retries < retry_limit:
        return RecoverDecision(kind=RecoverKind.QUEUE_EXIT_MASH, reason=reason)
    return RecoverDecision(kind=RecoverKind.FAIL_EXIT, reason=reason)


def decide_enter_house_failure(
    *,
    pos: Point,
    front: Point,
    enter_retries: int,
    enter_retry_limit: int = 4,
    reason: str,
) -> Optional[RecoverDecision]:
    """Return a retry decision, or None if enter retries are exhausted."""
    if enter_retries >= enter_retry_limit:
        return None
    hands_not_clear = "hands not clear" in reason
    if enter_fail_south_recovery_actions(pos, front) is not None:
        return RecoverDecision(
            kind=RecoverKind.RETRY_ENTER_SOUTH,
            reason=reason,
            hands_not_clear=hands_not_clear,
        )
    return RecoverDecision(
        kind=RecoverKind.RETRY_ENTER_RESTART,
        reason=reason,
        hands_not_clear=hands_not_clear,
    )


def decide_nav_failure(
    *,
    phase: str,
    pos: Point,
    front: Point,
    hands_clear: bool,
    drop_attempts: int,
    drop_attempt_limit: int,
    offstand_corrections: int,
    south_escape_attempts: int,
    south_escape_limit: int,
    reason: str,
) -> RecoverDecision:
    """Policy for nav_house_front / nav_drop_spot FAILURE or BLOCKED."""
    # Hands still full: drop then re-nav. Exhausted budget mutates phase to
    # start then hard-fails (same as prior inline path).
    if phase in {"nav_house_front", "nav_drop_spot"} and not hands_clear:
        if drop_attempts < drop_attempt_limit:
            return RecoverDecision(
                kind=RecoverKind.RETRY_DROP_THEN_NAV,
                reason=reason,
            )
        return RecoverDecision(
            kind=RecoverKind.HARD_FAIL,
            reason=reason,
            clear_task=True,
            set_phase="start",
        )

    if phase == "nav_house_front":
        dx = abs(pos.x - front.x)
        dy = abs(pos.y - front.y)
        # Generous near-door force-enter after multi_nav timeout:
        # - D12 residual (118,486) ~62px south of stand
        # - Gate B D5 (190,423) vs front (136,424): dx=54 lateral, dy=1
        #   (old dx<=48 left that as HARD_FAIL → terminal return_home)
        near_door = (dx <= 48 and dy <= 80) or (dx <= 72 and dy <= 32)
        if near_door:
            return RecoverDecision(kind=RecoverKind.FORCE_ENTER, reason=reason)
        # Mid-yard south of door (north of fence): simple re-nav north.
        if (
            not south_of_fence_wall(pos)
            and pos.y > front.y + 24
            and dx <= 80
            and offstand_corrections < 6
        ):
            return RecoverDecision(kind=RecoverKind.MID_YARD_RENAV, reason=reason)
        # South-of-fence / far-from-door multi_nav timeout: B-run escape.
        # Lateral miss past force-enter but not deep south: still try renav once
        # via escape only when truly far.
        door_far = dx + dy > 80
        if (
            (
                deep_south_of_house(pos, front)
                or south_of_fence_wall(pos)
                or door_far
            )
            and south_escape_attempts < south_escape_limit
        ):
            return RecoverDecision(
                kind=RecoverKind.SOUTH_ESCAPE,
                reason=reason,
                far_east=pos.x > EAST_AROUND_FENCE_X,
                escape_from_drop=False,
            )

    if phase == "nav_drop_spot":
        if (
            (deep_south_of_house(pos, front) or south_of_fence_wall(pos))
            and south_escape_attempts < south_escape_limit
        ):
            return RecoverDecision(
                kind=RecoverKind.SOUTH_ESCAPE,
                reason=reason,
                far_east=pos.x > EAST_AROUND_FENCE_X,
                escape_from_drop=True,
            )

    return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)


def decide_child_failure(
    *,
    phase: str,
    pos: Point,
    front: Point,
    reason: str,
    hands_clear: bool,
    exit_to_farm_retries: int,
    exit_to_farm_retry_limit: int,
    enter_retries: int,
    enter_retry_limit: int = 4,
    drop_attempts: int,
    drop_attempt_limit: int,
    offstand_corrections: int,
    south_escape_attempts: int,
    south_escape_limit: int,
) -> RecoverDecision:
    """Top-level policy for ReturnHomeTask child FAILURE/BLOCKED."""
    if phase == "exit_to_farm":
        return decide_exit_to_farm_failure(
            retries=exit_to_farm_retries,
            retry_limit=exit_to_farm_retry_limit,
            reason=reason,
        )
    if phase == "enter_house":
        enter = decide_enter_house_failure(
            pos=pos,
            front=front,
            enter_retries=enter_retries,
            enter_retry_limit=enter_retry_limit,
            reason=reason,
        )
        if enter is not None:
            return enter
        return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)
    if phase in {"nav_house_front", "nav_drop_spot"}:
        return decide_nav_failure(
            phase=phase,
            pos=pos,
            front=front,
            hands_clear=hands_clear,
            drop_attempts=drop_attempts,
            drop_attempt_limit=drop_attempt_limit,
            offstand_corrections=offstand_corrections,
            south_escape_attempts=south_escape_attempts,
            south_escape_limit=south_escape_limit,
            reason=reason,
        )
    return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)


def _charge_mix(direction: str, frames: int, *, a_every: int) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    for i in range(frames):
        if i % a_every == 0:
            actions.append(make_action(**{direction: True, "a": True}))
        else:
            actions.append(make_action(**{direction: True, "b": True}))
    return actions


def drop_carried_actions(attempt: int) -> List[np.ndarray]:
    """Toss held debris so building doors accept entry."""
    if attempt <= 1:
        return list(toss_held_actions(face="down", step_away=True)) + list(
            multi_face_toss_actions(prefer_south=True)
        )
    if attempt <= 3:
        return list(multi_face_toss_actions(prefer_south=True))
    actions: List[np.ndarray] = []
    for face in ("down", "left", "right"):
        actions.extend(toss_held_actions(face=face, step_away=True))
    return actions


def south_escape_actions(
    *, long_east: bool = False, far_east: bool = False
) -> List[np.ndarray]:
    """Leave south-of-fence softlock for re-nav.

    Far-east pond latitude: west toward free lane then north. SW pocket:
    north-first then east. Mid-wall: east first. Mix A so a blocking weed
    can be lifted.
    """
    if far_east:
        actions = (
            _charge_mix("left", 90, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 70, a_every=20)
            + _charge_mix("up", 90, a_every=20)
            + _charge_mix("left", 40, a_every=20)
            + _charge_mix("up", 60, a_every=20)
        )
    elif long_east:
        actions = (
            _charge_mix("up", 80, a_every=20)
            + _charge_mix("right", 70, a_every=20)
            + _charge_mix("up", 90, a_every=20)
            + _charge_mix("right", 60, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 20, a_every=20)
        )
    else:
        actions = (
            _charge_mix("right", 70, a_every=20)
            + _charge_mix("up", 70, a_every=20)
            + _charge_mix("right", 50, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 16, a_every=20)
        )
    actions.extend(make_action() for _ in range(8))
    return actions


def short_east_north_actions() -> List[np.ndarray]:
    """Compact east→north charge when outer timeout is almost gone."""
    actions = _charge_mix("right", 50, a_every=16) + _charge_mix(
        "up", 70, a_every=16
    )
    actions.extend(make_action() for _ in range(6))
    return actions


_RESCUE_DIRECTIONS = ("up", "right", "down", "left")


class RescueWalkTask(Task):
    """Shake a wedged farmer loose before ``return_home`` is rebuilt.

    Clears any open dialogue/menu, then walks a short square whose starting
    side rotates with ``attempt`` so successive rescues push different ways
    (a fence corner, a debris tile, a door lip). Always ends SUCCESS: the
    caller judges progress by whether the next ``return_home`` gets home.
    """

    name = "rescue_walk"

    def __init__(self, attempt: int = 0, leg_frames: int = 36, legs: int = 8) -> None:
        self.attempt = int(attempt)
        self.leg_frames = leg_frames
        self.legs = legs
        self._step = 0

    def reset(self, world: WorldState) -> None:
        self._step = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.scene import classify_scene_from_ram
        from harvest.planner.day_plan_status import is_house_tilemap
        from harvest.tasks.primitives import dismiss_dialogue_result

        self._step += 1
        scene = classify_scene_from_ram(world.ram)
        if is_house_tilemap(scene.tilemap):
            return TaskResult(status=TaskStatus.SUCCESS, reason="rescue: already in house")
        if scene.needs_input_dismiss and self._step < 600:
            return dismiss_dialogue_result(
                self._step, buttons=("b", "a"), pulse_every=1, reason="rescue: clear scene"
            )
        leg = (self._step - 1) // self.leg_frames
        if leg >= self.legs:
            return TaskResult(status=TaskStatus.SUCCESS, reason="rescue walk done")
        direction = _RESCUE_DIRECTIONS[(self.attempt + leg) % 4]
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**{direction: True})),
            reason=f"rescue walk {direction}",
        )


# ── Dog pickup flag (rr-bu62.19) ──────────────────────────────────────────
# bank_84 Event_UNKDogRelated: a fresh A with empty hands beside the dog starts
# the pickup/drop animation and sets game_state 0x0800.  The dog script only
# finishes it outdoors, so walking into the house mid-animation leaves 0x0800
# stuck: bed A, lunch and every other interaction are dead for the rest of the
# day (Fall soak D27, power-on D16).  A second A beside the dog clears it.
DOG_PICKUP_ANIM_FLAG = 0x0800
DOG_POS_ADDR = 0x11F2C  # WRAM $7F1F2C dog_pos_X, then dog_pos_Y
DOG_A_RADIUS_PX = 24
_A_INDEX = int(np.argmax(make_action(a=True)))
_B_INDEX = int(np.argmax(make_action(b=True)))


def _game_state(ram: np.ndarray) -> int:
    return int(ram[0xD2]) | (int(ram[0xD3]) << 8)


def dog_pickup_stuck(ram: np.ndarray) -> bool:
    """game_state bit 0x0800 is set, whether or not 0x0080 is set.

    A door warp sets 0x0080 for a few frames (0x4C81). That is still the
    pickup animation, not a finished release.
    """
    return bool(_game_state(ram) & DOG_PICKUP_ANIM_FLAG)


def dog_position(ram: np.ndarray) -> Optional[Point]:
    from harvest.core.ram_catalog import live_wram_base

    idx = DOG_POS_ADDR + live_wram_base(ram)
    if idx + 3 >= len(ram):
        return None
    x = int(ram[idx]) | (int(ram[idx + 1]) << 8)
    y = int(ram[idx + 2]) | (int(ram[idx + 3]) << 8)
    if x == 0 and y == 0:
        return None
    return Point(x, y)


# The door dog on power-on D5 sits at (139, 422). A stand on that row warps
# into the house, and the flag never clears.
_HOUSE_DOOR = (136, 424)


def dog_release_stand(dog: Point) -> tuple[tuple[int, int], str]:
    """Where to face the dog from. The house-door dog is met from the south."""
    if abs(dog.x - _HOUSE_DOOR[0]) <= 32 and abs(dog.y - _HOUSE_DOOR[1]) <= 32:
        return (_HOUSE_DOOR[0], _HOUSE_DOOR[1] + 32), "up"
    return (int(dog.x) - 16, int(dog.y)), "right"


def dog_adjacent(ram: np.ndarray, radius: int = DOG_A_RADIUS_PX) -> bool:
    from harvest.core.animal_status import read_held_item
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan_status import is_farm_tilemap

    if not is_farm_tilemap(int(read_ram_value(ram, "tilemap"))):
        return False
    dog = dog_position(ram)
    if dog is None or read_held_item(ram) != 0:
        return False
    px = int(read_ram_value(ram, "player_x"))
    py = int(read_ram_value(ram, "player_y"))
    return abs(px - dog.x) <= radius and abs(py - dog.y) <= radius


# The farm's two holes, (50,55) tile 0xC7 and (37,48) tile 0xC8, lead to the Harvest
# Sprites' cave, map 0x2A. They are not BFS-walkable, but blind charges end under
# them and walk in (the far-east escape "left 90, up 80"; a walk down the east side
# and west along y=857); the cave has no exit the bot can find, and the scene reader
# calls it a dialogue (stale text 0x031A/0x032A), so return_home mashed A/B for 8
# rescues twice (Y2 Summer D4, D16). A step that would enter one is held.
# (0xC0-0xC6 are the doors of the house, barn, coop, shed and west gate.)
FARM_PIT_TILES = frozenset({0xC7, 0xC8})
FARM_PIT_TILE = 0xC7
_PIT_LOOKAHEAD_PX = 12
_UP, _DOWN, _LEFT, _RIGHT = (int(np.argmax(make_action(**{k: True}))) for k in ("up", "down", "left", "right"))


def hold_back_from_pit(action, ram: np.ndarray):
    """Neutral pad instead of a farm step whose next ~12px lands on the pit tile."""
    from harvest.core.tile_catalog import ADDR_TILEMAP, get_tile_at
    from harvest.tasks.nav import get_pos_from_ram

    if action is None or len(action) <= _RIGHT or int(ram[ADDR_TILEMAP]) not in (0, 1, 2, 3):
        return action
    dx, dy = int(bool(action[_RIGHT])) - int(bool(action[_LEFT])), int(bool(action[_DOWN])) - int(bool(action[_UP]))
    if not (dx or dy):
        return action
    pos = get_pos_from_ram(ram)
    ahead = get_tile_at(ram, (pos.x + dx * _PIT_LOOKAHEAD_PX) // 16, (pos.y + dy * _PIT_LOOKAHEAD_PX) // 16)
    return make_action() if ahead in FARM_PIT_TILES else action


PLAYER_STATE_RIDING = 0x11  # player_state 1 is walking; probe: one A press dismounts
# Riding is harmless anywhere but the house door, which refuses a rider; walking mounted
# works. The press is limited to the door stand's neighbourhood (stand (136,424), walkway
# above it) because a held A anywhere else froze the walk (Y2 Fall D23, pixel_stuck at
# (259,376)) and starved the 17:00 shipping scene of its A edge (Y3 Spring D20, 1.6M frames).
_DOOR_ZONE_X = (104, 168)
_DOOR_ZONE_Y = (340, 470)


def dismount_horse(action, ram: np.ndarray):
    """A press while the farmer rides the horse at the house door stand.

    An A pulse beside the parked horse mounts it, and the house door refuses a rider:
    the enter push timed out 3 x 11k frames and cost eight rescues (Y2 Spring D19,
    jitter run p150b, farmer on sprite 0x0097 at the stand (8,26)). The bot never
    rides on purpose. Only when input is free and the farmer is idle, so the press
    is an edge (the dismount animation then runs on neutral frames).
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.tasks.nav import get_pos_from_ram

    if int(ram[ADDR_TILEMAP]) not in (0, 1, 2, 3) or int(read_ram_value(ram, "player_state")) != PLAYER_STATE_RIDING:
        return action
    pos = get_pos_from_ram(ram)
    if not (_DOOR_ZONE_X[0] <= pos.x <= _DOOR_ZONE_X[1] and _DOOR_ZONE_Y[0] <= pos.y <= _DOOR_ZONE_Y[1]):
        return action
    if int(read_ram_value(ram, "input_lock")) != 1:
        return action
    return make_action() if int(read_ram_value(ram, "player_action")) != 0 else make_action(a=True)


def sanitize_action(action, ram: np.ndarray):
    """Pad filters applied to every planner action: the pit, the dog hug, then a dismount."""
    if action is None:
        return action
    return dismount_horse(strip_dog_hug_press(hold_back_from_pit(action, ram), ram), ram)


def strip_dog_hug_press(action, ram: np.ndarray):
    """Swap A for B while the dog is beside the farmer with empty hands.

    A dismiss/interact pulse there would start the pickup animation.  B still
    closes text boxes; a task that truly needs A simply retries once the dog
    wanders off.
    """
    if action is None or len(action) <= max(_A_INDEX, _B_INDEX):
        return action
    if not action[_A_INDEX] or not dog_adjacent(ram):
        return action
    # An open text box (the 17:00 shipping scene) needs its A to close; only
    # the follow-up A after it closes can hug the dog.  Stripping the closing
    # press deadlocked the scene for 150k frames (power-on D4).
    from harvest.core.ram_catalog import read_ram_value

    if int(read_ram_value(ram, "input_lock")) != 1:
        return action
    from harvest.core.scene import classify_scene_from_ram

    if classify_scene_from_ram(ram).needs_input_dismiss:
        return action
    out = np.array(action, copy=True)
    out[_A_INDEX] = 0
    out[_B_INDEX] = 1
    return out


class DogReleaseTask(Task):
    """Outdoors, walk to the dog and press A until the stuck 0x0800 clears."""

    name = "dog_release"
    # Planner reads this; it must not special-case the class.
    allows_dog_a = True

    def __init__(self, timeout: int = 4000, max_tries: int = 8) -> None:
        self.timeout = timeout
        self.max_tries = max_tries
        self._step = 0
        self._tries = 0
        self._child: Optional[Task] = None
        self._queue: List[np.ndarray] = []
        self._off_farm = 0

    def reset(self, world: WorldState) -> None:
        self._step = 0
        self._tries = 0
        self._child = None
        self._queue = []
        self._off_farm = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.ram_catalog import read_ram_value
        from harvest.maps.map_types import Waypoint
        from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
        from harvest.planner.tasks.inventory_exit import ExitToFarmTask
        from harvest.planner.tasks.navigation import MultiMapNavTask

        self._step += 1
        # Clear only when 0x0800 itself drops. 0x0080 during a warp is not that.
        if not dog_pickup_stuck(world.ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason="dog flag clear")
        if self._step > self.timeout or self._tries > self.max_tries:
            return TaskResult(status=TaskStatus.FAILURE, reason="dog flag stuck")
        if self._queue:
            return TaskResult(
                status=TaskStatus.RUNNING, action=ActionResult(self._queue.pop(0))
            )
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        # Farm id arrives before the doorway warp finishes. Dropping the exit
        # child here stranded the remodeled-house farmer at (136,344), where
        # the stale door tiles seal every path toward the dog.
        if isinstance(self._child, ExitToFarmTask) or is_house_tilemap(tilemap):
            if not isinstance(self._child, ExitToFarmTask):
                self._child = ExitToFarmTask()
                self._child.reset(world)
            res = self._child.step(world)
            if res.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                return res
            if res.status == TaskStatus.SUCCESS:
                self._child = None
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=res.action or ActionResult(make_action()),
                reason="dog release: leaving building",
            )
        if not is_farm_tilemap(tilemap):
            self._off_farm += 1
            if self._off_farm > 600:
                return TaskResult(status=TaskStatus.FAILURE, reason="dog release: off farm")
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="dog release: waiting out a warp",
            )
        dog = dog_position(world.ram)
        if dog is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="dog release: no dog")
        stand, face = dog_release_stand(dog)
        if self._child is None or isinstance(self._child, ExitToFarmTask):
            self._child = MultiMapNavTask(
                name="nav_dog",
                waypoints=[Waypoint(tilemap=0x00, target_px=stand, radius=8)],
                timeout=1200,
                initial_settle_frames=0,
            )
            self._child.reset(world)
        res = self._child.step(world)
        if res.status == TaskStatus.RUNNING:
            return res
        # Arrived (or gave up): face the dog and press A once, then re-check.
        self._child = None
        self._tries += 1
        self._queue = (
            [make_action(**{face: True})] * 6
            + [make_action()] * 4
            + [make_action(a=True)]
            + [make_action()] * 20
        )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


# One bounded barn retry before bed when a barn adult is still unfed: the
# night pass then takes happiness -8 and a sick roll, and a sick cow dies seven
# nights later. Every cow death in the y23_final run followed such a day (a
# deferred ENTER_BARN, a chores stall, hay at 0). Hours past 18:00 are free, so
# the retry costs no field work. Feed, talk and brush; milking is skipped, it is
# the step the Y2 Summer D23 chores stalled after.
BARN_FEED_CHECK_FRAMES = 9000
_FEED_CHECK_MAPS = frozenset({0x27, 0x28})


def barn_feed_check_needed(ram: np.ndarray) -> bool:
    from harvest.core.animal_status import read_hay_count, unfed_adult_cow_slots
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap

    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    nearby = is_farm_tilemap(tilemap) or is_house_tilemap(tilemap) or tilemap in _FEED_CHECK_MAPS
    return nearby and read_hay_count(ram) > 0 and bool(unfed_adult_cow_slots(ram))


def barn_feed_check_phases() -> list:
    from dataclasses import replace

    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_phase_cow import (
        COW_CHORES_PHASE,
        ENTER_BARN_PHASE,
        EXIT_BARN_PHASE,
        NAV_TO_BARN_PHASE,
    )

    chores = replace(COW_CHORES_PHASE, params={
        **COW_CHORES_PHASE.params, "milk": False, "timeout": BARN_FEED_CHECK_FRAMES})
    return [EXIT_TO_FARM_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE, chores, EXIT_BARN_PHASE]


class BarnFeedCheckTask(Task):
    """The barn phases once more, inside ``BARN_FEED_CHECK_FRAMES``."""

    name = "barn_feed_check"

    def __init__(self, tasks_dir: str, budget: int = BARN_FEED_CHECK_FRAMES) -> None:
        self.tasks_dir = tasks_dir
        self.budget = budget
        self._step = 0
        self._plan: Optional[Task] = None

    def reset(self, world: WorldState) -> None:
        from dataclasses import replace

        from harvest.planner.day_phase_types import DayPlannerPolicy
        from harvest.planner.day_plan_orchestrator import DayPlanTask

        self._step = 0
        self._plan = DayPlanTask(
            tasks_dir=self.tasks_dir,
            phase_sequence=barn_feed_check_phases(),
            policy=replace(DayPlannerPolicy(), include_end_day=False),
        )
        self._plan.reset(world)

    def step(self, world: WorldState) -> TaskResult:
        self._step += 1
        if self._step > self.budget:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"feed check over {self.budget}f")
        return self._plan.step(world)


def pond_toss_px() -> Point:
    """Pixel of the main pond's south-lip toss stand."""
    from harvest.tasks.nav import TILE_SIZE
    from harvest.tasks.pond_policy import PRIMARY_POND_STAND

    return Point(PRIMARY_POND_STAND[0] * TILE_SIZE + 8, PRIMARY_POND_STAND[1] * TILE_SIZE + 8)


def pond_is_nearer(pos: Point, drop: Point) -> bool:
    """The pond lip is a shorter walk than the house drop spot."""
    pond = pond_toss_px()
    return abs(pos.x - pond.x) + abs(pos.y - pond.y) < abs(pos.x - drop.x) + abs(pos.y - drop.y)


class PondTossTask(Task):
    """Carry held debris to the main pond's south lip and toss it into the water.

    The ground drops near the house can all refuse a stone (Summer D27: one
    lifted by the far-east escape, every face and the wander refused it, 8
    rescues). The pond always takes debris: the Day 2 clear dumps every stone
    there. Debris and unrecognised items (``pond_takes``); a hen, a crop, an egg or milk is never thrown in.
    """

    name = "pond_toss"

    def __init__(self, timeout: int = 4000) -> None:
        self.timeout = timeout
        self._step = 0
        self._nav: Optional[Task] = None
        self._carry = None

    def reset(self, world: WorldState) -> None:
        from harvest.maps.map_types import Waypoint
        from harvest.planner.tasks.navigation import MultiMapNavTask
        from harvest.tasks.carry_toss import CarryToPondStand
        from harvest.tasks.nav import get_pos_from_ram

        self._step = 0
        self._carry = CarryToPondStand()
        # The carry skill owns the north-yard highway and east spur. A single
        # far-away F0 waypoint drives into the barns before that skill can run.
        if get_pos_from_ram(world.ram).y < 26 * 16:
            self._nav = None
            self._carry.reset(world)
            return
        stand = (pond_toss_px().x, pond_toss_px().y)
        self._nav = MultiMapNavTask(
            name="nav_pond_lip",
            waypoints=[Waypoint(tilemap=0x00, target_px=stand, radius=40)],
            timeout=max(600, self.timeout - 900),
            initial_settle_frames=0,
            allow_opportunistic_clear=False,
        )
        self._nav.reset(world)

    def can_start(self, world: WorldState) -> bool:
        from harvest.core.animal_status import read_held_item
        from harvest.tasks.farm_toss import pond_takes

        return pond_takes(read_held_item(world.ram))

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.animal_status import read_held_item
        from harvest.tasks.farm_toss import pond_takes

        self._step += 1
        held = int(read_held_item(world.ram))
        if held == 0 and self._nav is None:
            return self._carry.step(world)
        if held == 0:
            return TaskResult(status=TaskStatus.SUCCESS, reason="pond toss: hands clear")
        if not pond_takes(held):
            return TaskResult(status=TaskStatus.FAILURE, reason=f"pond toss: held 0x{held:02X} is not for the pond")
        if self._step > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="pond toss: timeout")
        if self._nav is not None:
            result = self._nav.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            # Arrived, or the walk gave up close by: the stand path is BFS from here.
            self._nav = None
            self._carry.reset(world)
        return self._carry.step(world)


def _tile_ring(ram: np.ndarray, pos: Point) -> str:
    """Tile ids of the 3x3 around the farmer, rows north to south, for failure logs."""
    from harvest.tasks.nav import TILE_SIZE, get_tile_at

    tx, ty = int(pos.x) // TILE_SIZE, int(pos.y) // TILE_SIZE
    rows = []
    for dy in (-1, 0, 1):
        try:
            rows.append(" ".join(f"{int(get_tile_at(ram, tx + dx, ty + dy)):02X}" for dx in (-1, 0, 1)))
        except Exception:
            rows.append("?? ?? ??")
    return f"at ({tx},{ty}) [{' / '.join(rows)}]"


class WanderDropTask(Task):
    """Find open ground by walking, then drop the held item in front.

    A carried rock/weed cannot be put down facing a blocked tile, and the
    fixed drop spots south of the house are ringed by south-field debris
    (power-on D25: held 0x0F, 8 rescues).  Walk each direction in turn until
    the farmer really moves, then pulse A facing that free tile.
    """

    name = "wander_drop"
    _DIRS = ("right", "left", "up", "down")

    def __init__(self, start_dir: int = 0, timeout: int = 900) -> None:
        self.start_dir = start_dir
        self.timeout = timeout
        self._step = 0
        self._leg = 0
        self._leg_frames = 0
        self._origin: Optional[Point] = None
        self._phase = "walk"
        self._drop_frames = 0

    def reset(self, world: WorldState) -> None:
        self._step = 0
        self._leg = 0
        self._leg_frames = 0
        self._origin = None
        self._phase = "walk"
        self._drop_frames = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.animal_status import read_held_item
        from harvest.tasks.nav import get_pos_from_ram

        self._step += 1
        if read_held_item(world.ram) == 0:
            return TaskResult(status=TaskStatus.SUCCESS, reason="wander drop: hands clear")
        pos = get_pos_from_ram(world.ram)
        if self._step > self.timeout or self._leg >= 8:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"wander drop: no open ground {_tile_ring(world.ram, pos)}",
            )
        if self._origin is None:
            self._origin = pos
        direction = self._DIRS[(self.start_dir + self._leg) % 4]
        if self._phase == "walk":
            self._leg_frames += 1
            if abs(pos.x - self._origin.x) + abs(pos.y - self._origin.y) >= 14:
                self._phase = "drop"
                self._drop_frames = 0
            elif self._leg_frames > 30:
                self._leg += 1
                self._leg_frames = 0
                self._origin = pos
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(**{direction: True})),
                reason=f"wander drop: walk {direction}",
            )
        self._drop_frames += 1
        if self._drop_frames > 30:
            # Front tile refused the item; try the next direction from here.
            self._leg += 1
            self._phase = "walk"
            self._leg_frames = 0
            self._origin = pos
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if self._drop_frames % 8 in (1, 2):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(a=True, **{direction: True})),
                reason="wander drop: A",
            )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


def toss_held_first(water, world):
    """Dump debris still in hand into the pond before a tool phase (hands full means no tool).

    gdet_20261115 Y1 Summer D11: a stone lifted by a skipped `clear_plant_stands` stayed in hand, every can use idled on
    "drop held" (its pad was empty) and CROP_WATER hit its 24,000 f watchdog 14 times.
    """
    from harvest.core.animal_status import read_held_item
    from harvest.tasks.farm_toss import pond_takes
    from harvest.tasks.primitives import TaskSequence

    held = int(read_held_item(world.ram))
    if not held or not pond_takes(held):
        return water
    print(f"[DAY_PLAN] {getattr(water, 'name', 'water')} starts with held=0x{held:02X}: pond toss first")
    return TaskSequence(name="pond_toss_then_tool", tasks=(PondTossTask(), water), keep_last_reason=True)
