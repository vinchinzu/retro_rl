"""Exit-to-farm, farm-exit, and outdoor morning intro tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Point,
    make_action,
    get_pos_from_ram,
)
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
)
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.tasks.primitives import dismiss_dialogue_result
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    FARM_TILEMAP,
    SHED_TILEMAP,
    BARN_TILEMAP,
    COOP_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
)
from harvest.core.scene import (
    SceneLocation,
    SceneMode,
    classify_scene_from_ram,
    scene_indicates_ending,
)
from harvest.core.ram_catalog import read_ram_value
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.planner.tasks.transitions import (
    DirectionalTransitionTask,
    ExitBuildingTask,
)
from harvest.planner.tasks.inventory_common import (
    farm_free_move_ready,
    outdoor_intro_flags_ready,
)

FARM_BUILDING_EXIT_STAND_TILES: Dict[int, Tuple[int, int]] = {
    SHED_TILEMAP: (8, 12),
    BARN_TILEMAP: (8, 22),
    COOP_TILEMAP: (8, 12),
}
# The shed tilemap flips to farm while coordinates are still settling through
# y~392.  Starting another shelf trip there selects the upper-farm route, then
# snaps to the real door at (26,30) behind collision.  Gate shed success on the
# settled outdoor stand; other buildings retain their existing looser gate.
FARM_BUILDING_OUTDOOR_STAND_TILES: Dict[int, Tuple[int, int]] = {
    SHED_TILEMAP: (26, 30),
}
FARM_BUILDING_EXIT_DOOR_X = 8 * 16 + 8
BARN_EXIT_TROUGH_X = 113
BARN_EXIT_TROUGH_MAX_X = 130
BARN_EXIT_RIGHT_AISLE_X = 204
BARN_EXIT_BYPASS_X = 216
BARN_EXIT_LOWER_Y = 20 * 16 + 8
BARN_EXIT_DOOR_X = 8 * 16 + 8

# Dog-name entry (tilemap 0x5F) is the intentional end of CODE_83CEAE outdoor
# morning intro. $099F=3 marks dog naming (HM-Decomp TODO: "name being asked").
_NAME_TILEMAP = 0x5F
_NAME_KIND_ADDR = 0x099F
_NAME_CURSOR_ADDR = 0x0991
_NAME_LENGTH_ADDR = 0x0994
_NAME_KIND_DOG = 3
_NAME_INPUT_COOLDOWN = 40
_NAME_READY_SETTLE = 20


def _raw_u8(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) if address < len(ram) else 0


@dataclass
class CompleteOutdoorMorningIntroTask(Task):
    """Pure-complete D2 outdoor morning dog intro so free-move is restored.

    After pure truck→sleep, first house→farm with ``event_flags_1f68=0x0011``
    fires ``CODE_83CEAE``: ORA morning bit ``0x0020``, clear free-move, scripted
    walk to house-front, dialogue, then dog **name entry** (tilemap ``0x5F``,
    ``$099F=3``). Completing the name (deterministic ``AAAA``) sets dog-owned
    ``0x0080`` (flags → ``0x00B1``) and restores free-move ``0x4000``.

    Verified Clean 2026-08-09 from ``town_day1_rest_end`` (~3.5k frames after
    ExitToFarm). No RAM writes. Idempotent when flags already ready + free.
    """

    name: str = "complete_outdoor_morning_intro"
    timeout: int = 12_000
    exit_timeout: int = 4000
    tasks_dir: str = TASKS_DIR
    dog_name_chars: int = 4  # AAAA — same deterministic short name as power-on

    _step_count: int = field(default=0, init=False)
    _last_input_step: int = field(default=-_NAME_INPUT_COOLDOWN, init=False)
    _exit_task: Optional[Task] = field(default=None, init=False)
    _name_submitted: bool = field(default=False, init=False)
    _ready_frames: int = field(default=0, init=False)
    _last_reason: str = field(default="start", init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._last_input_step = -_NAME_INPUT_COOLDOWN
        self._exit_task = None
        self._name_submitted = False
        self._ready_frames = 0
        self._last_reason = "start"

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        if self._name_submitted:
            return "INTRO_SETTLE"
        if self._exit_task is not None:
            return "INTRO_EXIT"
        return "INTRO_OUTDOOR"

    def _can_press(self) -> bool:
        return self._step_count - self._last_input_step >= _NAME_INPUT_COOLDOWN

    def _press(self, *, reason: str, **buttons: bool) -> TaskResult:
        self._last_input_step = self._step_count
        self._last_reason = reason
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**buttons)),
            reason=reason,
        )

    def _idle(self, reason: str) -> TaskResult:
        self._last_reason = reason
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=reason,
        )

    def _is_complete(self, ram: np.ndarray) -> bool:
        return outdoor_intro_flags_ready(ram) and farm_free_move_ready(ram)

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        ram = world.ram
        if self._is_complete(ram):
            # Already ready (Y1 / pre-set): succeed immediately.
            # After dog-name submit: brief settle so free-move sticks.
            if not self._name_submitted:
                f68 = int(read_ram_value(ram, "event_flags_1f68", raw=True))
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=(
                        f"outdoor intro already ready f1f68=0x{f68:04X} "
                        f"frames={self._step_count}"
                    ),
                )
            self._ready_frames += 1
            if self._ready_frames >= _NAME_READY_SETTLE:
                f68 = int(read_ram_value(ram, "event_flags_1f68", raw=True))
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=(
                        f"outdoor intro complete f1f68=0x{f68:04X} "
                        f"free-move frames={self._step_count}"
                    ),
                )
            return self._idle("intro complete; free-move settle")
        self._ready_frames = 0

        if self._step_count > self.timeout:
            f68 = -1
            gs = -1
            try:
                f68 = int(read_ram_value(ram, "event_flags_1f68", raw=True))
                gs = int(read_ram_value(ram, "game_state", raw=True))
            except Exception:
                pass
            pos = get_pos_from_ram(ram)
            tm = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"outdoor morning intro timeout f1f68=0x{f68:04X} gs=0x{gs:04X} "
                    f"tm=0x{tm:02X} pos=({pos.x},{pos.y}) phase={self.phase_text} "
                    f"last={self._last_reason}"
                ),
            )

        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1

        # Still indoors: exit first (intro fires on house→farm).
        if is_house_tilemap(tilemap) or (
            not is_farm_tilemap(tilemap) and tilemap != _NAME_TILEMAP
        ):
            if self._exit_task is None:
                self._exit_task = ExitToFarmTask(
                    tasks_dir=self.tasks_dir,
                    cutscene_mash_limit=300,
                )
                self._exit_task.reset(world)
            result = self._exit_task.step(world)
            if result.status == TaskStatus.FAILURE:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"intro exit failed: {result.reason}",
                )
            if result.status == TaskStatus.SUCCESS:
                self._exit_task = None
                return self._idle("exit done; await outdoor intro")
            self._last_reason = result.reason or "exiting to farm"
            return result

        # Dog name entry screen (intentional CODE_83CEAE outcome).
        if tilemap == _NAME_TILEMAP and input_lock == 5:
            name_len = _raw_u8(ram, _NAME_LENGTH_ADDR)
            name_cursor = _raw_u8(ram, _NAME_CURSOR_ADDR)
            if self._can_press():
                if name_len < self.dog_name_chars:
                    return self._press(
                        reason=f"dog name char {name_len + 1}/{self.dog_name_chars}",
                        a=True,
                    )
                if name_cursor == 0:
                    # Same reversed-grid quirk as PowerOnStartTask.
                    return self._press(reason="dog name move to OK", left=True)
                if name_cursor == 40:
                    return self._press(reason="dog name move to OK", up=True)
                if name_cursor == 70:
                    self._name_submitted = True
                    return self._press(reason="confirm dog name AAAA", a=True)
                return self._press(reason=f"dog name confirm cursor={name_cursor}", a=True)
            return self._idle("dog name input cooldown")

        # Dialogue (lock 2) or menus — mash A/B.
        if input_lock in (0, 2, 4):
            if self._can_press():
                return self._press(reason=f"intro dialogue lock={input_lock}", a=True)
            return self._idle(f"dialogue cooldown lock={input_lock}")

        # Scripted walk / cutscene on farm without free-move: stay neutral so
        # we do not fight auto-walk to house-front, occasional A for stalls.
        if is_farm_tilemap(tilemap) and not farm_free_move_ready(ram):
            if self._can_press() and (self._step_count % 90) < 3:
                return self._press(reason="cutscene pulse A", a=True)
            return self._idle("await scripted outdoor intro walk")

        # Free-move but flags incomplete: rare; pulse A in case dog NPC talk.
        if farm_free_move_ready(ram) and not outdoor_intro_flags_ready(ram):
            if self._can_press() and (self._step_count % 120) < 3:
                return self._press(reason="free-move wait dog bit", a=True)
            return self._idle("free-move; waiting dog-owned bit")

        return self._idle(self._last_reason or "outdoor intro settle")


# Maps with a known pure exit / return route to farm.
_EXIT_ROUTE_TILEMAPS = frozenset({0x0C, 0x04, 0x05, 0x10})
_BUILDING_EXIT_TILEMAPS = frozenset({SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP})


@dataclass
class ExitToFarmTask(Task):
    """Exit the current farm building without assuming a house-only opener."""

    name: str = "exit_to_farm"
    tasks_dir: str = TASKS_DIR
    house_timeout: int = 2200

    cutscene_mash_limit: int = 240
    # Sticky budget for dialogue / unknown-map thrash (rr-uru1).
    # Power-on D23 hung on dialogue@unknown tilemap=0x08 because the old
    # dismiss counter reset whenever a single free frame cleared needs_input
    # dismiss — mashing forever until the outer return_home / planner budget.
    # This limit only resets on reset() or farm SUCCESS, never mid-thrash.
    dismiss_mash_limit: int = 360

    _task: Optional[Task] = field(default=None, init=False)
    _blocked_reason: str = field(default="", init=False)
    _step_count: int = field(default=0, init=False)
    _cutscene_mash_count: int = field(default=0, init=False)
    _dismiss_mash_count: int = field(default=0, init=False)
    # Cumulative frames spent off known exit routes (sticky; rr-uru1).
    _offroute_stuck_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._blocked_reason = ""
        self._step_count = 0
        self._cutscene_mash_count = 0
        self._dismiss_mash_count = 0
        self._offroute_stuck_frames = 0
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_farm_tilemap(tilemap):
            self._task = None
            return
        if is_house_tilemap(tilemap):
            self._task = ExitBuildingTask(
                target_tilemap=FARM_TILEMAP,
                dialog_frames=120,
                timeout=self.house_timeout,
            )
        elif tilemap in _BUILDING_EXIT_TILEMAPS:
            self._task = DirectionalTransitionTask(
                name="directional_transition_exit_to_farm",
                direction="down",
                origin_tilemap=tilemap,
                target_tilemap=FARM_TILEMAP,
                timeout=1800 if tilemap == BARN_TILEMAP else 1200,
                min_frames_before_success=15,
                stand_tile=FARM_BUILDING_EXIT_STAND_TILES.get(tilemap),
                stand_tolerance=1,
                target_stand_tile=FARM_BUILDING_OUTDOOR_STAND_TILES.get(tilemap),
                target_stand_tolerance=1,
                door_align_px=FARM_BUILDING_EXIT_DOOR_X,
                settle_frames=5,
            )
        elif tilemap in _EXIT_ROUTE_TILEMAPS:
            route_name = {
                0x0C: "path_to_farm",
                0x04: "town_to_farm",
                0x05: "event_town_to_farm",
                0x10: "mountain_to_farm",
            }[tilemap]
            waypoints = ROUTES.get(route_name, [])
            self._task = MultiMapNavTask(
                name=f"return_{route_name}",
                waypoints=list(waypoints),
                timeout=6000,
                initial_settle_frames=30,
            )
        else:
            scene = classify_scene_from_ram(world.ram)
            if scene_indicates_ending(scene) or scene.mode == SceneMode.INVALID_COORDINATES:
                self._task = None
                self._blocked_reason = f"cannot exit to farm from {scene.summary()}"
                return
            if (
                scene.mode == SceneMode.CUTSCENE_EVENT
                or scene.location == SceneLocation.UNKNOWN
                or scene.mode == SceneMode.UNKNOWN_TILEMAP
            ):
                # Mash through event cutscenes in step(); do not invent a house exit.
                self._task = None
                return
            self._task = ExitBuildingTask(
                target_tilemap=FARM_TILEMAP,
                dialog_frames=120,
                timeout=self.house_timeout,
            )
        self._task.reset(world)

    @staticmethod
    def _has_known_exit_route(tilemap: int) -> bool:
        return (
            is_farm_tilemap(tilemap)
            or is_house_tilemap(tilemap)
            or tilemap in _BUILDING_EXIT_TILEMAPS
            or tilemap in _EXIT_ROUTE_TILEMAPS
        )

    def _sticky_offroute_timeout(
        self, scene, *, kind: str
    ) -> Optional[TaskResult]:
        """Fail after sticky off-route budget (does not reset mid-thrash)."""
        self._offroute_stuck_frames += 1
        limit = max(self.dismiss_mash_limit, self.cutscene_mash_limit)
        if self._offroute_stuck_frames > limit:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"{kind} timeout after {self._offroute_stuck_frames}f "
                    f"from {scene.summary()}"
                ),
            )
        return None

    def _unknown_map_escape_result(self, scene) -> TaskResult:
        """Multi-face walk + A/B while stuck on unregistered tilemaps.

        Power-on residual tilemap 0x08 at farm-like coords often needs a
        direction change, not pure A-mash, after dialogue text idles.
        """
        faces = ("down", "left", "right", "up")
        face = faces[(self._offroute_stuck_frames // 20) % len(faces)]
        pulse = self._offroute_stuck_frames % 4
        if pulse == 0:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(**{face: True}, a=True, b=True)),
                reason=f"unknown-map escape face={face} {scene.summary()}",
            )
        if pulse == 1:
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason=f"unknown-map dismiss {scene.summary()}",
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**{face: True}, b=True)),
            reason=f"unknown-map walk {face} {scene.summary()}",
        )

    def resume_after_hotswap(self, world: WorldState) -> None:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_farm_tilemap(tilemap):
            self._task = None
            return
        if self._task is None:
            self.reset(world)
            return
        resume = getattr(self._task, "resume_after_hotswap", None)
        if callable(resume):
            resume(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene):
            return TaskResult(
                status=TaskStatus.BLOCKED,
                reason=f"cannot exit to farm from {scene.summary()}",
            )

        # Settled on farm with no active child → done (clears sticky thrash).
        # When a DirectionalTransition child is still settling on farm, defer
        # to the child so stand/settle frames stay behavior-identical.
        if (
            is_farm_tilemap(tilemap)
            and not scene.is_transient
            and not scene.needs_input_dismiss
            and self._task is None
        ):
            self._offroute_stuck_frames = 0
            self._dismiss_mash_count = 0
            self._cutscene_mash_count = 0
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")

        # Drop stale child routes when the live tilemap left their origin
        # (house exit warping onto unregistered 0x08 must not keep house logic).
        if self._task is not None and not self._has_known_exit_route(tilemap):
            self._task = None

        if scene.is_transient:
            if scene.mode == SceneMode.CUTSCENE_EVENT:
                self._cutscene_mash_count += 1
                timed = self._sticky_offroute_timeout(scene, kind="cutscene/unknown")
                if timed is not None:
                    # Prefer FAILURE so ReturnHome can recover-mash then retry
                    # (BLOCKED was treated as hard stop in older paths).
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=timed.reason or f"cannot exit to farm from {scene.summary()}",
                    )
                if self._cutscene_mash_count > self.cutscene_mash_limit:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            f"cutscene mash timeout after {self._cutscene_mash_count}f "
                            f"from {scene.summary()}"
                        ),
                    )
                # Unregistered tilemaps: walk+mash, not pure A idle.
                if not self._has_known_exit_route(tilemap):
                    return self._unknown_map_escape_result(scene)
                return dismiss_dialogue_result(
                    self._step_count,
                    buttons=("a", "b"),
                    pulse_every=1,
                    reason=f"waiting through {scene.mode.value}",
                )
            # House→farm mid-warp: tilemap flips to farm with y≈212 while
            # player_state still has the transition bit. Neutral wait freezes
            # outdoor control (Gate B / power-on D2). Keep pushing south until
            # coordinates settle near the door front (~344).
            if is_farm_tilemap(tilemap):
                # Building exits own their destination stand.  In particular,
                # the shed passes through a plausible farm y before snapping
                # to its real outside door; do not let the generic house gate
                # finish the transition early.
                if isinstance(self._task, DirectionalTransitionTask) and (
                    self._task.origin_tilemap in _BUILDING_EXIT_TILEMAPS
                ):
                    return self._task.step(world)
                pos_y = get_pos_from_ram(world.ram).y
                if pos_y < 330:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(down=True, b=True)),
                        reason=f"finish farm mid-warp y={pos_y}",
                    )
                # Mid-warp finished enough to count as on-farm.
                self._offroute_stuck_frames = 0
                self._task = None
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"tilemap=0x{tilemap:02X} mid-warp settle y={pos_y}",
                )
            # Non-farm transition (sleep/map): sticky budget so we never idle forever.
            timed = self._sticky_offroute_timeout(scene, kind="transition")
            if timed is not None:
                return timed
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"waiting through {scene.summary()}",
            )

        if scene.needs_input_dismiss:
            self._dismiss_mash_count += 1
            timed = self._sticky_offroute_timeout(scene, kind="dialogue dismiss")
            if timed is not None:
                return timed
            if self._dismiss_mash_count > self.dismiss_mash_limit:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"dialogue dismiss timeout after {self._dismiss_mash_count}f "
                        f"from {scene.summary()}"
                    ),
                )
            if not self._has_known_exit_route(tilemap):
                return self._unknown_map_escape_result(scene)
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason=f"dismiss {scene.mode.value} before exit",
            )

        # Free input on unregistered / no-route map: sticky escape, do not
        # reset thrash counters (intermittent dialogue was the D23 hang).
        if not self._has_known_exit_route(tilemap):
            timed = self._sticky_offroute_timeout(scene, kind="unknown-map")
            if timed is not None:
                return timed
            return self._unknown_map_escape_result(scene)

        # Known building / town / path route — progress may clear sticky budget
        # only after a short free streak is unnecessary; keep sticky until farm.
        if self._blocked_reason:
            return TaskResult(status=TaskStatus.BLOCKED, reason=self._blocked_reason)
        if is_farm_tilemap(tilemap) and self._task is None:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
        if self._task is None:
            # Rebuild route without wiping sticky counters (manual partial reset).
            blocked = self._blocked_reason
            stuck = self._offroute_stuck_frames
            dismiss = self._dismiss_mash_count
            cut = self._cutscene_mash_count
            steps = self._step_count
            self.reset(world)
            self._offroute_stuck_frames = stuck
            self._dismiss_mash_count = dismiss
            self._cutscene_mash_count = cut
            self._step_count = steps
            self._blocked_reason = blocked or self._blocked_reason
            if self._blocked_reason:
                return TaskResult(status=TaskStatus.BLOCKED, reason=self._blocked_reason)
            if is_farm_tilemap(
                int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else tilemap
            ):
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
            if self._task is None:
                timed = self._sticky_offroute_timeout(scene, kind="no-exit-route")
                if timed is not None:
                    return timed
                return self._unknown_map_escape_result(scene)
        return self._task.step(world)


@dataclass
class FarmExitTask(Task):
    """BFS walk to the west farm gate (shop/town approach).

    Pure B-run left/up thrashes from the shipping bin / south farm after a
    berry ship. Delegate to NavTask so debris and long hops path correctly.
    """

    name: str = "farm_exit"
    target_px: Point = field(default_factory=lambda: Point(40, 424))
    radius: int = 16
    timeout: int = 10000

    _nav: Optional[Task] = field(default=None, init=False, repr=False)

    def reset(self, world: WorldState) -> None:
        pos = get_pos_from_ram(world.ram)
        if pos.y >= 32 * 16:
            # Post-berry return: use the known clear south lane and cross the
            # long fence only after reaching its west end. Generic straight
            # NavTask can spend its full budget trying to walk north through
            # the fence from the shipping bin.
            waypoints = list(ROUTES["farm_south_to_west_gate"])
        else:
            waypoints = [
                Waypoint(
                    tilemap=FARM_TILEMAP,
                    target_px=(self.target_px.x, self.target_px.y),
                    radius=self.radius,
                )
            ]
        self._nav = MultiMapNavTask(
            name=f"{self.name}_multi_nav",
            waypoints=waypoints,
            timeout=self.timeout,
            initial_settle_frames=0,
        )
        self._nav.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(0)

        if self._nav is None:
            self.reset(world)
        assert self._nav is not None
        result = self._nav.step(world)
        if result.status == TaskStatus.SUCCESS:
            return TaskResult(status=TaskStatus.SUCCESS, reason="arrived west gate")
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=result.reason or "farm_exit timeout",
            )
        return result
