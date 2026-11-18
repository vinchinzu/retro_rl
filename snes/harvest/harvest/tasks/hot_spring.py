"""Mountain outdoor hot-spring stamina refill task.

ROM facts from human recording ``tasks/hot_spring_bath.json`` (2026-07-31):

  **True spa = upper outdoor pond on mountain 0x10**, not camp tent pond,
  not MapMountainCave 0x29.

  Farm→spa path (grape dirt corridor, not the east fish pond):
    south land → carpenter gap → west climb → east mid y~360
    → north/east to upper lip y=201. Fish/camp starts still use
    fish_spot_to_outdoor_spa.

  Soak (A button — not B-alone at camp F0):
    Stand A0 at tile(38,12) / px~(619,201), hold **Right+A** into water tile
    **0xF7** at (39,12), walk across to (40,12), then **Left+A** back.
    ``player_action=3`` for ~30–35f while on/through F7.
    Stamina: start 100/130 → end **130/130** (full restore verified).

  Mid-right camp/tent pond (F0 ~tx43,ty25) is the wrong water — no restore.
  West cave 0x29 is MapMountainCave (blue feather), not spa.

  Noon lunch (HaveLunch @ 12:00) is a separate +20; not spa.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec, read_ram_value
from harvest.core.stamina import Stamina
from harvest.core.scene import SPA_POOL_TILE_BOX, classify_scene_from_ram
from harvest.maps.map_config import (
    ROUTES,
    Waypoint,
    farm_to_spa_waypoints,
    slice_route_from_position,
)
from harvest.planner.day_plan_status import TASKS_DIR, is_farm_tilemap
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import dismiss_dialogue_result, drain_action_queue

# Outdoor mountain map — spa stays on this tilemap (no interior transition).
MOUNTAIN_TILEMAP = 0x10
PATH_TILEMAP = 0x0C
# Historical cave interior (MapMountainCave). Not the hot spring.
CAVE_TILEMAP = 0x29
# Back-compat alias used by older probes/tests.
SPA_TILEMAP = MOUNTAIN_TILEMAP

# Match farm_clearer / crop tasks: direct WRAM offsets on the task RAM view.
ADDR_STAMINA = field_spec("stamina").address
ADDR_MAX_STAMINA = field_spec("max_stamina").address
ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_GAME_STATE = field_spec("game_state").address

# Upper-pond bath lip from hot_spring_bath recording (A into 0xF7).
# West stand on A0 facing right into water tile (39,12)=0xF7.
SPA_OUTDOOR_STAND_TILE = (38, 12)
SPA_OUTDOOR_STAND_PX = (619, 201)
SPA_WATER_TILE = (39, 12)
SPA_WATER_TILE_ID = 0xF7
SPA_EAST_STAND_PX = (644, 201)
# Begin soak when within this Manhattan px of the lip. 48px let a west
# stand (x=582) start bathing before it reached 0xF7 at ~(624,201).
SPA_ARRIVAL_RADIUS_PX = 24

# player_action == 3 is jump / in-water anim (decomp + recording).
PLAYER_ACTION_JUMP = 3
# Recording: Right+A into F7 ~13f, coast, Left+A back ~14f.
ENTER_HOLD_FRAMES = 14
COAST_FRAMES = 20


def _read_u16(ram: np.ndarray, addr: int) -> int:
    if addr + 1 >= len(ram):
        return 0
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def read_player_xy(ram: np.ndarray) -> tuple[int, int]:
    return _read_u16(ram, ADDR_X), _read_u16(ram, ADDR_Y)


def near_outdoor_spa(ram: np.ndarray, radius: int = SPA_ARRIVAL_RADIUS_PX) -> bool:
    """True when on mountain near the upper-pond lip (not camp pond)."""
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != MOUNTAIN_TILEMAP:
        return False
    px, py = read_player_xy(ram)
    sx, sy = SPA_OUTDOOR_STAND_PX
    return abs(px - sx) + abs(py - sy) <= radius


# The pool box starts at tile x=38, which is the dry lip (SPA_OUTDOOR_STAND_PX);
# the water begins at x=39, so a farmer east of x=624 is in the pond.
SPA_WATER_MIN_X_PX = 624
SPA_WATER_MAX_X_PX = 640   # the 0xF7 strip is one tile wide; the farmer lands on the dry shelf east of it

# Closed-loop bath (ROM probe, 2026-11-08): the old fixed-length button script slipped when the clock phase moved. A left
# jump overshot the west lip, the walk back pressed A before the farmer stood flush against the water, the press became a tool
# swing, and every later cycle repeated it: stamina stuck at 132/160 for 3,600 frames in 4 of ~170 sweep soaks, farmer in the pond.
# Now the farmer walks flush to the edge, holds A only there until ``player_action`` reads 3, rides the jump, and picks the next
# direction from where it landed.
BATH_STILL_FRAMES = 3      # an unchanged x this long means the water blocks the walk: flush against the edge
BATH_APPROACH_MAX = 90     # frames of walking toward the edge before the press goes ahead anyway
BATH_PRESS_MAX = 40        # frames of A at the edge without a jump before it counts as a miss
BATH_COAST_MAX = 90        # a jump lasts ~40 frames; this only guards a stuck animation
BATH_BACKOFF_FRAMES = 14   # walk away from the edge after a miss so the next press starts from a fresh approach
BATH_SETTLE_FRAMES = 4
BATH_MAX_MISSES = 4        # misses in a row before the soak gives up and walks home


def in_spa_pond(ram: np.ndarray) -> bool:
    """True when the farmer stands in the upper-pond water, not on the lip."""
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != MOUNTAIN_TILEMAP:
        return False
    px, py = read_player_xy(ram)
    x0, y0, x1, y1 = SPA_POOL_TILE_BOX
    return px >= SPA_WATER_MIN_X_PX and x0 <= px // 16 <= x1 and y0 <= py // 16 <= y1


def spa_pond_exit_actions() -> list:
    """Pad frames that leave the pond westward and stop on solid ground.

    Plain left walking collides with water ``0xF7``; re-cross it twice with a
    Left+A jump (the same input that entered), then push west so the return
    route starts on the path. Shared by the post-soak settle and ``return_home``
    (a day floor that fires mid-soak strands the farmer in the water, where the
    mountain route's first waypoint is unreachable: Y2 Spring D19).
    """
    actions = [make_action() for _ in range(6)]
    for _attempt in range(2):
        actions += [make_action(left=True, b=True) for _ in range(4)]
        actions += [make_action(left=True, b=True, a=True) for _ in range(ENTER_HOLD_FRAMES + 8)]
        actions += [make_action(left=True, b=True) for _ in range(COAST_FRAMES)]
        actions += [make_action() for _ in range(6)]
    actions += [make_action(left=True, b=True) for _ in range(28)]
    actions += [make_action() for _ in range(12)]
    return actions


def read_stamina(ram: np.ndarray) -> int:
    return Stamina.from_ram(ram).current


def read_max_stamina(ram: np.ndarray) -> int:
    return Stamina.from_ram(ram).maximum


def read_stamina_state(ram: np.ndarray) -> Stamina:
    return Stamina.from_ram(ram)


def read_player_action(ram: np.ndarray) -> int:
    if ADDR_PLAYER_ACTION >= len(ram):
        return 0
    return int(ram[ADDR_PLAYER_ACTION])


_OPPOSITE = {"left": "right", "right": "left"}


def _bath_miss_state(ram: np.ndarray) -> str:
    """What the farmer and its neighbours were doing when a flush A press never jumped (Y3 Summer D14: 4 misses, no cause in the log)."""
    from harvest.core.npc_catalog import game_objects
    from harvest.core.ram_catalog import read_ram_value

    px, py = read_player_xy(ram)
    near = sorted(
        (abs(o.pixel[0] - px) + abs(o.pixel[1] - py), o.sprite_table_idx, o.pixel)
        for o in game_objects(ram) if not o.is_player
    )[:4]
    fields = ("player_state", "held_item", "input_lock", "player_direction", "stamina", "hour")
    state = " ".join(f"{k}={read_ram_value(ram, k)}" for k in fields)
    return f"pos={(px, py)} act={read_player_action(ram)} {state} near={[(d, hex(i), p) for d, i, p in near]}"



class BathCrossing:
    """Closed-loop crossing of the one-tile 0xF7 strip: walk flush to the edge, hold A there, ride the jump, settle.

    ``tasks/hot_spring_bath.json`` holds **B+direction+A** while crossing 0xF7 (tool was brush 0x0F). A+dir alone with the
    watering can selected does not enter water (edge collision, no ``player_action=3``). One jump pays +12 stamina and carries
    the farmer to the far shelf, so every crossing flips the side. The direction comes from the farmer's x on each start, never
    from a counter: a jump that overshoots a lip, or a soak resumed anywhere, still walks back to the edge it needs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stage = ""            # "" idle, then approach / press / coast / backoff / settle
        self.n = 0
        self.dir = "right"
        self.x: Optional[int] = None
        self.still = 0
        self.misses = 0            # presses in a row that never produced a jump
        self.crossings = 0         # crossings started

    def start(self, px: int, *, to_west: bool = False) -> None:
        if to_west or px >= SPA_WATER_MAX_X_PX:
            self.dir = "left"
        elif px < SPA_WATER_MIN_X_PX:
            self.dir = "right"
        self.stage, self.n, self.x, self.still = "approach", 0, None, 0
        self.crossings += 1

    def action(self, ram: np.ndarray, *, to_west: bool = False):
        """The pad for this frame."""
        px, _py = read_player_xy(ram)
        jumping = read_player_action(ram) == PLAYER_ACTION_JUMP
        if not self.stage:
            self.start(px, to_west=to_west)
        self.n += 1
        d = self.dir
        if self.stage == "approach":
            self.still = self.still + 1 if px == self.x else 0
            self.x = px
            if self.still < BATH_STILL_FRAMES and self.n <= BATH_APPROACH_MAX:
                return make_action(**{d: True, "b": True})
            self.stage, self.n = "press", 0     # A before the farmer is flush became a tool swing, not a jump
        if self.stage == "press":
            if jumping:
                self.stage, self.n, self.misses = "coast", 0, 0
            elif self.n <= BATH_PRESS_MAX:
                return make_action(**{d: True, "b": True, "a": True})
            else:
                self.misses += 1
                print(f"[SPA] press {self.misses} never jumped: {_bath_miss_state(ram)}")
                self.stage, self.n = "backoff", 0
        if self.stage == "coast":
            if jumping and self.n <= BATH_COAST_MAX:
                return make_action(**{d: True, "b": True})
            self.stage, self.n = "settle", 0
        if self.stage == "backoff":
            if self.n <= BATH_BACKOFF_FRAMES:
                return make_action(**{_OPPOSITE[d]: True, "b": True})
            self.start(px, to_west=to_west)
            return make_action()
        if self.n <= BATH_SETTLE_FRAMES:
            return make_action()
        self.start(px, to_west=to_west)
        return make_action()


# A sprite (the farm dog) can park in the narrow west corridor and seal the walk to the spa. A cover evening soaks sixty times,
# so one blocked walk must not end it: leave, wait for the dog to move, then walk again (ROM probes: 2 of 9 evening soaks blocked; one 12-start
# sample needed ~15,000f of waiting).
GATE_RETRIES = 8
GATE_WAIT_FRAMES = 600              # by day the clock runs through a wait
GATE_WAIT_NIGHT_FRAMES = 3000       # after 18:00 it is frozen, and the dog moves a tile every few thousand frames (ROM: freed at ~15,000f)
GATE_RETREAT_PX = (232, 392)        # the open yard east of the corridor: the farmer leaves so the dog can wander off
GATE_BLOCKS = ("pixel_stuck", "no_path", "exit_walk timeout")


@dataclass
class HotSpringStaminaTask(Task):
    """Travel to the outdoor mountain hot spring, A-bathe until stamina recovers.

    Ladder:
      already_full → SUCCESS
      not on farm → ExitToFarm
      farm/path → MultiMapNav farm_to_spa (upper pond)
      mountain at pond → A+dir bath cycles until target / plateau / max
      optional MultiMapNav mountain_to_farm
    """

    name: str = "hot_spring_stamina"
    # None = soak until current == max (full restore).
    min_stamina: int | None = None
    return_to_farm: bool = True
    tasks_dir: str = TASKS_DIR
    timeout: int = 24000
    soak_timeout: int = 3600
    soak_plateau_frames: int = 240
    # Farm → path → full mountain corridor needs ~2–4k frames when clear;
    # return is similar length. Budget headroom for stasis recovery.
    nav_timeout: int = 20000
    # A bath cycle is two crossings (one jump, +12 stamina, each). A numeric ``min_stamina`` gives up after this many cycles with no
    # gain, or ends at the plateau; a full restore keeps bathing until current == max or ``soak_timeout``.
    max_jump_cycles: int = 10

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _steps: int = field(default=0, init=False)
    _soak_steps: int = field(default=0, init=False)
    _soak_start: int = field(default=0, init=False)
    _last_stamina: int = field(default=-1, init=False)
    _plateau: int = field(default=0, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _stam_before_trip: int = field(default=0, init=False)
    _bath: BathCrossing = field(default_factory=BathCrossing, init=False)
    _soak_failure: str = field(default="", init=False)
    _exit_tail: bool = field(default=False, init=False)
    _jumps_seen: int = field(default=0, init=False)
    _was_jumping: bool = field(default=False, init=False)
    _soak_done_reason: str = field(default="", init=False)
    _route_retries: int = field(default=0, init=False)
    _gate_wait: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._steps = 0
        self._soak_steps = 0
        self._soak_start = 0
        self._last_stamina = -1
        self._plateau = 0
        self._action_queue.clear()
        self._stam_before_trip = int(read_stamina(world.ram))
        self._bath.reset()
        self._soak_failure = ""
        self._exit_tail = False
        self._jumps_seen = 0
        self._was_jumping = False
        self._soak_done_reason = ""
        self._route_retries = 0
        self._gate_wait = 0

    @property
    def _jump_cycles(self) -> int:
        """Crossings started this soak (one jump each, +12 stamina)."""
        return self._bath.crossings

    def can_start(self, world: WorldState) -> bool:
        return True

    def _stamina_target(self, ram: np.ndarray) -> int:
        stam = Stamina.from_ram(ram)
        if self.min_stamina is None:
            return stam.maximum
        return min(int(self.min_stamina), stam.maximum)

    def _stamina_ok(self, ram: np.ndarray) -> bool:
        stam = Stamina.from_ram(ram)
        return stam.current >= self._stamina_target(ram)

    def _activate(self, phase: str, task: Task, world: WorldState) -> TaskResult:
        self._phase = phase
        self._task = task
        task.reset(world)
        return task.step(world)

    def _queue_lip_approach(self, ram: np.ndarray) -> None:
        """Run to the recorded lip before the first B+A water pass."""
        px, py = read_player_xy(ram)
        sx, sy = SPA_OUTDOOR_STAND_PX
        dx, dy = sx - px, sy - py
        # Outdoor run is ~1.5px/frame; pad so we do not start short of 0xF7.
        if abs(dx) > 8:
            axis = "right" if dx > 0 else "left"
            frames = min(90, abs(dx) * 2 // 3 + 12)
            for _ in range(frames):
                self._action_queue.append(make_action(**{axis: True, "b": True}))
        if abs(dy) > 8:
            axis = "down" if dy > 0 else "up"
            frames = min(40, abs(dy) * 2 // 3 + 8)
            for _ in range(frames):
                self._action_queue.append(make_action(**{axis: True, "b": True}))
        for _ in range(8):
            self._action_queue.append(make_action())

    def _begin_soak(self, world: WorldState) -> TaskResult:
        self._phase = "soak"
        self._task = None
        self._soak_steps = 0
        self._soak_start = read_stamina(world.ram)
        self._last_stamina = self._soak_start
        self._plateau = 0
        self._bath.reset()
        self._soak_failure = ""
        self._exit_tail = False
        self._jumps_seen = 0
        self._was_jumping = False
        self._action_queue.clear()
        for _ in range(8):
            self._action_queue.append(make_action())
        self._queue_lip_approach(world.ram)
        stam = Stamina.from_ram(world.ram)
        target = self._stamina_target(world.ram)
        print(
            f"[SPA] Upper-pond A-bath start {stam} target>={target} "
            f"max_cycles={self.max_jump_cycles} stand~{SPA_OUTDOOR_STAND_PX} "
            f"pos={read_player_xy(world.ram)} "
            f"water_tile={SPA_WATER_TILE} id=0x{SPA_WATER_TILE_ID:02X}"
        )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _finish_success(self, world: WorldState, reason: str) -> TaskResult:
        stam = read_stamina(world.ram)
        msg = f"{reason}; stamina={stam} (was {self._stam_before_trip})"
        print(f"[SPA] SUCCESS {msg}")
        return TaskResult(status=TaskStatus.SUCCESS, reason=msg)

    def _sliced_route(
        self,
        route_name: str,
        world: WorldState,
        *,
        fallback: Optional[str] = None,
    ) -> list:
        """Named route, sliced from nearest hop to current player position."""
        waypoints = list(ROUTES.get(route_name, []) or [])
        if not waypoints and fallback:
            waypoints = list(ROUTES.get(fallback, []) or [])
        if not waypoints:
            return []
        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        px, py = read_player_xy(ram)
        sliced = slice_route_from_position(waypoints, px, py, tilemap=tilemap)
        if len(sliced) < len(waypoints):
            print(
                f"[SPA] Route {route_name}: sliced {len(waypoints)} → {len(sliced)} "
                f"hops from pos=({px},{py}) map=0x{tilemap:02X}"
            )
        return sliced

    def _start_return_or_done(self, world: WorldState, reason: str) -> TaskResult:
        if not self._stamina_ok(world.ram) and not self._soak_failure:
            stam = Stamina.from_ram(world.ram)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"refusing return unrestored ({reason}); {stam}",
            )
        if self.return_to_farm:
            waypoints = self._sliced_route(
                "outdoor_spa_to_farm", world, fallback="mountain_to_farm"
            )
            if waypoints:
                return self._activate(
                    "return_farm",
                    MultiMapNavTask(
                        name="spa_return_farm",
                        waypoints=waypoints,
                        timeout=self.nav_timeout,
                        initial_settle_frames=15,
                    ),
                    world,
                )
        return self._finish_success(world, reason)

    def _start_next(self, world: WorldState) -> TaskResult:
        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

        if self._stamina_ok(ram) and (
            is_farm_tilemap(tilemap) or not self.return_to_farm
        ):
            return self._finish_success(world, "stamina already sufficient")

        # Accidentally inside west cave — walk out, then re-route to outdoor pond.
        if tilemap == CAVE_TILEMAP:
            print("[SPA] On cave 0x29 (not the outdoor spa); exiting to mountain")
            from harvest.planner.tasks.transitions import DirectionalTransitionTask

            return self._activate(
                "exit_cave",
                DirectionalTransitionTask(
                    name="exit_mountain_cave",
                    direction="down",
                    origin_tilemap=CAVE_TILEMAP,
                    target_tilemap=MOUNTAIN_TILEMAP,
                    timeout=1800,
                    min_frames_before_success=10,
                    settle_frames=20,
                ),
                world,
            )

        if tilemap == MOUNTAIN_TILEMAP:
            if self._stamina_ok(ram):
                return self._start_return_or_done(world, "soaked; on mountain")
            # Already at upper pond lip — soak without re-walking the corridor.
            if near_outdoor_spa(ram):
                print(
                    f"[SPA] Already at outdoor spa lip "
                    f"pos={read_player_xy(ram)}; begin A-bath"
                )
                return self._begin_soak(world)
            waypoints = self._sliced_route(
                "mountain_entry_to_outdoor_spa",
                world,
                fallback="mountain_entry_to_spa",
            )
            # East camp/fish/Gotz pocket. Manhattan-slice onto the grape
            # corridor would pick the spa ridge, which is cliff-blocked from
            # here — walk the fish→spa dirt instead.
            px, py = read_player_xy(ram)
            if px >= 600 and py >= 350:
                fish = self._sliced_route("fish_spot_to_outdoor_spa", world)
                if fish:
                    waypoints = fish
            if waypoints:
                return self._activate(
                    "route_spa",
                    MultiMapNavTask(
                        name="mountain_to_outdoor_spa",
                        waypoints=waypoints,
                        timeout=self.nav_timeout,
                        initial_settle_frames=10,
                    ),
                    world,
                )
            return self._begin_soak(world)

        if is_farm_tilemap(tilemap) or tilemap == PATH_TILEMAP:
            if self._stamina_ok(ram) and is_farm_tilemap(tilemap):
                return self._finish_success(world, "stamina already sufficient")
            px, py = read_player_xy(ram)
            route = list(farm_to_spa_waypoints(px, py, tilemap))
            if route:
                sliced = slice_route_from_position(route, px, py, tilemap=tilemap)
                if len(sliced) < len(route):
                    print(
                        f"[SPA] Route farm_to_spa: sliced {len(route)} → {len(sliced)} "
                        f"hops from pos=({px},{py}) map=0x{tilemap:02X}"
                    )
                route = sliced
            if not route:
                route = self._sliced_route("farm_to_spa", world)
            if not route:
                route = (
                    list(ROUTES.get("farm_to_mountain", []))
                    + list(
                        ROUTES.get("mountain_entry_to_outdoor_spa", [])
                        or ROUTES.get("mountain_entry_to_spa", [])
                    )
                )
            if not route:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="missing farm_to_spa / farm_to_mountain route",
                )
            return self._activate(
                "route_mountain",
                MultiMapNavTask(
                    name="farm_to_spa",
                    waypoints=list(route),
                    timeout=self.nav_timeout,
                    initial_settle_frames=20,
                ),
                world,
            )

        return self._activate(
            "exit_to_farm",
            ExitToFarmTask(tasks_dir=self.tasks_dir),
            world,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        if self._steps > self.timeout:
            stam = read_stamina(world.ram)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"hot spring timeout stamina={stam} phase={self._phase}",
            )

        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss:
            return dismiss_dialogue_result(
                self._steps,
                reason=f"spa {scene.mode.value}",
            )

        if self._phase == "soak":
            return self._step_soak(world)

        if self._phase == "gate_wait":
            # The farm dog parks in the west corridor (3,26) for a while; idle frames move it. Then walk the route again.
            self._gate_wait -= 1
            if self._gate_wait % 200 == 0:
                from harvest.planner.tasks.home_recover import dog_position

                print(f"[SPA] gate wait {self._gate_wait}f left: farmer={read_player_xy(world.ram)} dog={dog_position(world.ram)}")
            if self._gate_wait > 0:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._phase = "start"
            return self._start_next(world)

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        if self._phase == "post_soak_settle":
            if not self._exit_tail:
                exiting = self._exit_pond_step(world)
                if exiting is not None:
                    return exiting
            px, _py = read_player_xy(world.ram)
            # Still east of pond — keep pushing west before multi-nav.
            if px > 600:
                print(f"[SPA] post-soak still east lip x={px}; extra west push")
                for _ in range(10):
                    self._action_queue.append(
                        make_action(left=True, b=True, a=True)
                    )
                for _ in range(20):
                    self._action_queue.append(make_action(left=True, b=True))
                queued = drain_action_queue(self._action_queue)
                if queued is not None:
                    return queued
            reason = self._soak_done_reason or "soaked"
            self._phase = "start"
            print(
                f"[SPA] post-soak settle done pos={read_player_xy(world.ram)}; "
                f"starting return"
            )
            return self._start_return_or_done(world, f"soaked ({reason})")

        if self._task is None:
            return self._start_next(world)

        # Mid-route restore (e.g. noon HaveLunch +20) — skip remaining spa walk.
        if self._phase in {"route_mountain", "route_spa"} and self._stamina_ok(
            world.ram
        ):
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            print(
                f"[SPA] Stamina ok mid-{self._phase} "
                f"({read_stamina(world.ram)}); abort route"
            )
            self._task = None
            if tilemap == MOUNTAIN_TILEMAP:
                return self._start_return_or_done(world, "stamina ok mid-route")
            if is_farm_tilemap(tilemap) or not self.return_to_farm:
                return self._finish_success(world, "stamina ok mid-route")
            return self._start_return_or_done(world, "stamina ok mid-route")

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result

        if result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            if self._phase == "gate_retreat":
                self._task = None                      # no matter how far it got: wait where it stands
                self._phase = "gate_wait"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if (
                self._phase == "route_mountain"
                and is_farm_tilemap(int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0)
                and any(block in reason for block in GATE_BLOCKS)
                and self._route_retries < GATE_RETRIES
            ):
                from harvest.planner.tasks.home_recover import dog_position

                from harvest.core.ram_catalog import read_ram_value

                self._route_retries += 1
                night = int(read_ram_value(world.ram, "hour")) >= 18
                self._gate_wait = GATE_WAIT_NIGHT_FRAMES if night else GATE_WAIT_FRAMES
                print(f"[SPA] Walk to the spa blocked ({reason}) dog={dog_position(world.ram)}; "
                      f"retreat and wait {self._gate_wait}f, try {self._route_retries}/{GATE_RETRIES}")
                retreat = MultiMapNavTask(
                    name="spa_gate_retreat", waypoints=[Waypoint(0x00, GATE_RETREAT_PX, radius=16)],
                    timeout=1500, initial_settle_frames=0,
                )
                return self._activate("gate_retreat", retreat, world)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self._phase} failed: {reason}",
            )

        if self._phase == "gate_retreat":
            self._task = None
            self._phase = "gate_wait"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "exit_to_farm":
            self._task = None
            return self._start_next(world)

        if self._phase in {"route_mountain", "route_spa", "exit_cave"}:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == MOUNTAIN_TILEMAP:
                if self._stamina_ok(world.ram):
                    return self._start_return_or_done(world, "stamina ok after route")
                # Prefer soak only when near lip; otherwise re-route (nav may
                # have "succeeded" on a wrong intermediate or partial path).
                if near_outdoor_spa(world.ram) or self._phase == "route_spa":
                    return self._begin_soak(world)
                self._task = None
                return self._start_next(world)
            if tilemap == CAVE_TILEMAP:
                self._task = None
                return self._start_next(world)
            self._task = None
            return self._start_next(world)

        if self._phase == "return_farm":
            if not self._stamina_ok(world.ram):
                stam = Stamina.from_ram(world.ram)
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"{self._soak_failure}; walked home" if self._soak_failure else f"returned unrestored; {stam}",
                )
            return self._finish_success(world, "soaked; returned to farm")

        self._task = None
        return self._start_next(world)

    def _finish_soak(self, world: WorldState, reason: str) -> TaskResult:
        stam = Stamina.from_ram(world.ram)
        print(
            f"[SPA] Soak done ({reason}) {stam} "
            f"(start={self._soak_start}) frames={self._soak_steps} "
            f"jumps_seen={self._jumps_seen} crossings={self._jump_cycles}"
        )
        self._task = None
        if not self._stamina_ok(world.ram):
            failure = f"soak ended unrestored ({reason}); {stam}"
            if not self.return_to_farm:
                return TaskResult(status=TaskStatus.FAILURE, reason=failure)
            # An optional phase that fails in the water leaves the farmer on the mountain, where every later phase fails its
            # map lock (have=0x10 need=0x00, Y3 Spring D12): leave the pond and walk home first, then report the failure.
            self._soak_failure = failure
        # A caller that does not need to return has reached its contract as
        # soon as stamina is restored.  The east-to-west pond crossing below
        # only prepares a safe starting point for the return navigation.
        elif not self.return_to_farm:
            return self._finish_success(world, f"soaked ({reason})")

        self._phase = "post_soak_settle"
        self._action_queue.clear()
        self._bath.reset()
        self._exit_tail = False
        self._soak_done_reason = reason
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _exit_pond_step(self, world: WorldState) -> Optional[TaskResult]:
        """Cross back to the west lip with the closed-loop bath, then queue the walk clear of the water.

        The bath often ends on the east shelf (x~646), where a plain walk west collides with 0xF7. None once the farmer is
        west of the water and the clear-the-lip walk has been queued and drained.
        """
        ram = world.ram
        px, _py = read_player_xy(ram)
        in_water = px >= SPA_WATER_MIN_X_PX or read_player_action(ram) == PLAYER_ACTION_JUMP
        if in_water and not self._exit_tail:
            if self._bath.misses < BATH_MAX_MISSES:
                action = self._bath.action(ram, to_west=True)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            print(f"[SPA] exit stalled at {read_player_xy(ram)}; pad fallback")
            self._action_queue.extend(spa_pond_exit_actions())
            self._exit_tail = True
        elif not self._exit_tail:
            self._action_queue.extend(make_action(left=True, b=True) for _ in range(28))
            self._action_queue.extend(make_action() for _ in range(12))
            self._exit_tail = True
        return drain_action_queue(self._action_queue)

    def _step_soak(self, world: WorldState) -> TaskResult:
        ram = world.ram
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        stam = read_stamina(ram)
        action = read_player_action(ram)
        self._soak_steps += 1

        if tilemap != MOUNTAIN_TILEMAP:
            print(f"[SPA] Left mountain mid-soak (map=0x{tilemap:02X}); re-route")
            self._task = None
            self._phase = "start"
            self._action_queue.clear()
            return self._start_next(world)

        if action == PLAYER_ACTION_JUMP:
            self._was_jumping = True
        elif self._was_jumping:
            self._was_jumping = False
            self._jumps_seen += 1
            print(
                f"[SPA] Jump/water anim #{self._jumps_seen} "
                f"stamina={stam} (start={self._soak_start})"
            )

        if stam > self._last_stamina:
            print(f"[SPA] Stamina {self._last_stamina} -> {stam}")
            self._last_stamina = stam
            self._plateau = 0
        else:
            self._plateau += 1

        target_hit = self._stamina_ok(ram)
        gained = stam > self._soak_start
        fill_to_max = self.min_stamina is None
        # Partial plateau is not a full restore. Only accept it when the
        # caller asked for a numeric threshold (or we already hit target).
        plateau_done = (
            gained
            and self._plateau >= self.soak_plateau_frames
            and (target_hit or not fill_to_max)
        )
        timed_out = self._soak_steps >= self.soak_timeout
        cycles_done = self._jump_cycles >= 2 * self.max_jump_cycles and not self._action_queue
        stalled = self._bath.misses >= BATH_MAX_MISSES
        # Full restore needs ~14 jumps at 160 max stamina. Do not treat the cycle budget
        # as "done enough" while current < max.
        cycles_complete_partial = cycles_done and gained and not fill_to_max

        if target_hit or plateau_done or timed_out or cycles_complete_partial or stalled:
            reason = (
                "target reached"
                if target_hit
                else "soak plateau"
                if plateau_done
                else "cycles done"
                if cycles_complete_partial
                else "bath stalled"
                if stalled
                else "soak timeout"
            )
            self._action_queue.clear()
            return self._finish_soak(world, reason)

        if cycles_done and not gained:
            self._action_queue.clear()
            reason = (
                f"no restore after {self._jump_cycles} crossings "
                f"(jumps_seen={self._jumps_seen})"
            )
            print(f"[SPA] FAIL {reason}; {Stamina.from_ram(ram)}")
            return self._finish_soak(world, reason)

        queued = drain_action_queue(self._action_queue)    # the walk to the lip that _begin_soak queued
        if queued is not None:
            return queued
        before = self._bath.crossings
        pad = self._bath.action(ram)
        if self._bath.crossings != before and (self._bath.crossings == 1 or self._bath.crossings % 4 == 0):
            print(
                f"[SPA] crossing {self._bath.crossings} dir={self._bath.dir} stam={stam} "
                f"pos={read_player_xy(ram)} act={action} hour={read_ram_value(ram, 'hour')}"
            )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(pad))

    @property
    def phase_text(self) -> str:
        return self._phase

    @property
    def progress_text(self) -> str:
        return (
            f"phase={self._phase} steps={self._steps} "
            f"jumps={self._jumps_seen}/{self._jump_cycles}"
        )


__all__ = [
    "SPA_TILEMAP",
    "MOUNTAIN_TILEMAP",
    "CAVE_TILEMAP",
    "SPA_OUTDOOR_STAND_TILE",
    "SPA_OUTDOOR_STAND_PX",
    "SPA_WATER_TILE",
    "SPA_WATER_TILE_ID",
    "SPA_ARRIVAL_RADIUS_PX",
    "PLAYER_ACTION_JUMP",
    "read_stamina",
    "read_max_stamina",
    "read_stamina_state",
    "read_player_action",
    "read_player_xy",
    "near_outdoor_spa",
    "HotSpringStaminaTask",
]
