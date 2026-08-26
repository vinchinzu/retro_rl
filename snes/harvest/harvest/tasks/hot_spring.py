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

from harvest.core.ram_catalog import field_spec
from harvest.core.stamina import Stamina
from harvest.core.scene import classify_scene_from_ram
from harvest.maps.map_config import (
    ROUTES,
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
SETTLE_FRAMES = 12


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


def _queue_a_bath_cycle(queue: deque, *, enter_right: bool = True) -> None:
    """One human-style bath pass into 0xF7 water, coast, return.

    ``tasks/hot_spring_bath.json`` holds **B+direction+A** while crossing
    0xF7 (tool was brush 0x0F). A+dir alone with watering can selected does
    not enter water (edge collision, no ``player_action=3``). Alternating
    right/left matches the five bath runs in the capture.
    """
    into = "right" if enter_right else "left"
    back = "left" if enter_right else "right"
    # Brief face + run without A so direction sticks before interact.
    for _ in range(4):
        queue.append(make_action(**{into: True, "b": True}))
    for _ in range(ENTER_HOLD_FRAMES):
        queue.append(make_action(**{into: True, "b": True, "a": True}))
    for _ in range(COAST_FRAMES):
        queue.append(make_action(**{into: True, "b": True}))
    for _ in range(4):
        queue.append(make_action())
    for _ in range(ENTER_HOLD_FRAMES):
        queue.append(make_action(**{back: True, "b": True, "a": True}))
    for _ in range(COAST_FRAMES):
        queue.append(make_action(**{back: True, "b": True}))
    for _ in range(SETTLE_FRAMES):
        queue.append(make_action())


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
    # None = soak until current >= max (full restore).
    min_stamina: int | None = None
    return_to_farm: bool = True
    tasks_dir: str = TASKS_DIR
    timeout: int = 24000
    soak_timeout: int = 3600
    soak_plateau_frames: int = 240
    # Farm → path → full mountain corridor needs ~2–4k frames when clear;
    # return is similar length. Budget headroom for stasis recovery.
    nav_timeout: int = 20000
    # Live fill is ~5–6 jump-exits of 0xF7. This is a queue budget, not a
    # "leave now" cap — full restore keeps bathing until current == max.
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
    _jump_cycles: int = field(default=0, init=False)
    _jumps_seen: int = field(default=0, init=False)
    _was_jumping: bool = field(default=False, init=False)
    _soak_done_reason: str = field(default="", init=False)

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
        self._jump_cycles = 0
        self._jumps_seen = 0
        self._was_jumping = False
        self._soak_done_reason = ""

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
        self._jump_cycles = 0
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
        if not self._stamina_ok(world.ram):
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

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        if self._phase == "post_soak_settle":
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
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self._phase} failed: {reason}",
            )

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
                    reason=f"returned unrestored; {stam}",
                )
            return self._finish_success(world, "soaked; returned to farm")

        self._task = None
        return self._start_next(world)

    def _finish_soak(self, world: WorldState, reason: str) -> TaskResult:
        stam = Stamina.from_ram(world.ram)
        print(
            f"[SPA] Soak done ({reason}) {stam} "
            f"(start={self._soak_start}) frames={self._soak_steps} "
            f"jumps_seen={self._jumps_seen} jump_cycles={self._jump_cycles}"
        )
        self._task = None
        if not self._stamina_ok(world.ram):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"soak ended unrestored ({reason}); {stam}",
            )
        # A caller that does not need to return has reached its contract as
        # soon as stamina is restored.  The east-to-west pond crossing below
        # only prepares a safe starting point for the return navigation.
        if not self.return_to_farm:
            return self._finish_success(world, f"soaked ({reason})")

        self._phase = "post_soak_settle"
        self._action_queue.clear()
        # Bath often ends on the *east* lip (x~640+). Plain left walk collides
        # with 0xF7. Re-cross water (B+A+left) up to twice, then push west to
        # ~x560 so return multi-nav starts on solid A0 path.
        for _ in range(6):
            self._action_queue.append(make_action())
        for _attempt in range(2):
            for _ in range(4):
                self._action_queue.append(make_action(left=True, b=True))
            for _ in range(ENTER_HOLD_FRAMES + 8):
                self._action_queue.append(make_action(left=True, b=True, a=True))
            for _ in range(COAST_FRAMES):
                self._action_queue.append(make_action(left=True, b=True))
            for _ in range(6):
                self._action_queue.append(make_action())
        for _ in range(28):
            self._action_queue.append(make_action(left=True, b=True))
        for _ in range(12):
            self._action_queue.append(make_action())
        self._soak_done_reason = reason
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

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
        cycles_done = self._jump_cycles >= self.max_jump_cycles and not self._action_queue
        # Full restore needs ~5–6 jump-exits. Do not treat the cycle budget
        # as "done enough" while current < max.
        cycles_complete_partial = cycles_done and gained and not fill_to_max

        if target_hit or plateau_done or timed_out or cycles_complete_partial:
            reason = (
                "target reached"
                if target_hit
                else "soak plateau"
                if plateau_done
                else "cycles done"
                if cycles_complete_partial
                else "soak timeout"
            )
            self._action_queue.clear()
            return self._finish_soak(world, reason)

        if cycles_done and not gained:
            self._action_queue.clear()
            reason = (
                f"no restore after {self._jump_cycles} A-bath cycles "
                f"(jumps_seen={self._jumps_seen})"
            )
            if fill_to_max or not target_hit:
                stam_now = Stamina.from_ram(ram)
                print(f"[SPA] FAIL {reason}; {stam_now}")
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"{reason}; stamina={stam_now}",
                )
            return self._finish_soak(world, reason)

        keep_bathing = self._jump_cycles < self.max_jump_cycles or (
            fill_to_max and not target_hit and not timed_out
        )
        if not self._action_queue and keep_bathing:
            enter_right = self._jump_cycles % 2 == 0
            _queue_a_bath_cycle(self._action_queue, enter_right=enter_right)
            self._jump_cycles += 1
            if self._jump_cycles == 1 or self._jump_cycles % 2 == 0:
                cap = self.max_jump_cycles
                print(
                    f"[SPA] A-bath cycle {self._jump_cycles}/{cap} "
                    f"enter={'right' if enter_right else 'left'} stam={stam}"
                )

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

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
