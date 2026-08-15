"""Reactive first-mountain-berry skill (not a recorded input tape).

Spring D2 house → farm west gate → path crossroads → mountain south
corridor north on x=20 → first grape stand. Town trips reuse
``farm_to_path`` + ``path_to_town``.

Default is approach-only: arrive at the stand and stop. Pickup is a
separate arm: stand on the ground grape at ``(326, 409)``, face down,
tap A, then choose **Don't eat**. Mash-A eats the grape. Mountain
dialogue with ``held=0`` is Gotz — fail closed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.game_clock import SEGMENT_FPS, format_segment_time
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.scene import SceneMode, classify_scene_from_ram
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, MOUNTAIN_WALKABLE, tile_label
from harvest.maps.map_config import (
    SEGMENTS,
    path_coords_leaked,
    segment_waypoints,
    slice_route_from_position,
)
from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
    repeat_action,
)

_LOCKED = frozenset({0, 2, 4})
_EAT_CURSOR_DONT = 1

# HM-Decomp held-item table (0x091D), not tool-slot ids.
MOUNTAIN_FORAGE_ITEMS = {
    0x01: "mushroom",
    0x02: "poison_mushroom",
    0x03: "grapes",
    0x04: "green_fruit",
    0x05: "flower",
    0x07: "fish",
    0x08: "power_berry",
}
MOUNTAIN_BERRY_ITEMS = frozenset({0x03, 0x08})
PATH_TILEMAP = 0x0C
MOUNTAIN_TILEMAP = 0x10
SHED_TILEMAP = 0x26
BARN_TILEMAP = 0x27
COOP_TILEMAP = 0x28

# Walk these named hops in order. Live tilemap picks the first remaining hop.
BERRY_NAV_SEGMENTS: Tuple[str, ...] = (
    "farm_to_path",
    "path_to_mountain",
    "mountain_entry_to_first_berry",
)
TOWN_NAV_SEGMENTS: Tuple[str, ...] = (
    "farm_to_path",
    "path_to_town",
)


def held_forage_name(held: int) -> Optional[str]:
    return MOUNTAIN_FORAGE_ITEMS.get(int(held))


def is_mountain_forage(held: int) -> bool:
    return int(held) in MOUNTAIN_FORAGE_ITEMS


def _tilemap(world: WorldState) -> int:
    ram = world.ram
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0


def _needs_building_exit(tilemap: int) -> bool:
    return is_house_tilemap(tilemap) or tilemap in (SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP)


def first_remaining_segment(
    tilemap: int,
    segments: Sequence[str] = BERRY_NAV_SEGMENTS,
) -> Optional[str]:
    """Skip hops the live map has already finished."""
    if _needs_building_exit(tilemap):
        return None
    if is_farm_tilemap(tilemap):
        return segments[0] if segments else None
    if tilemap == PATH_TILEMAP:
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and hops[-1].tilemap in (PATH_TILEMAP, MOUNTAIN_TILEMAP) and name != "farm_to_path":
                return name
        return None
    if tilemap == MOUNTAIN_TILEMAP:
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and hops[0].tilemap == MOUNTAIN_TILEMAP:
                return name
        return None
    return segments[0] if segments else None


# mountain_grape_stand tape: stand on this pixel, face down, A. Ground
# grape (not carpenter 2x2). Item box is Eat / Don't eat — keep it.
GRAPE_STAND_PX = (326, 409)
GRAPE_STAND_TILE = (20, 25)
GRAPE_STAND_RADIUS = 16
GRAPE_STAND_NUDGE = 3
def mountain_corridor_segments(samples: Sequence[dict]) -> dict:
    """Measure mountain land → grape and grape → mountain exit.

    ``samples`` are time-ordered snapshots with ``frame``, ``tilemap``,
    ``x``, ``y``, and optional ``held_item``. Pick/keep time between the
    two hops is reported separately and is not part of either corridor.
    """
    enter = None
    grape = None
    leave_stand = None
    mountain_exit = None
    for row in samples:
        frame = int(row.get("frame", 0))
        tilemap = int(row.get("tilemap", -1))
        x = int(row.get("x", 0))
        y = int(row.get("y", 0))
        held = int(row.get("held_item", 0))
        at_stand = (
            tilemap == MOUNTAIN_TILEMAP
            and abs(x - GRAPE_STAND_PX[0]) <= GRAPE_STAND_RADIUS
            and abs(y - GRAPE_STAND_PX[1]) <= GRAPE_STAND_RADIUS
        )
        if enter is None and tilemap == MOUNTAIN_TILEMAP:
            enter = frame
        if grape is None and at_stand:
            grape = frame
        if (
            leave_stand is None
            and grape is not None
            and tilemap == MOUNTAIN_TILEMAP
            and is_mountain_forage(held)
            and not at_stand
        ):
            leave_stand = frame
        if (
            mountain_exit is None
            and grape is not None
            and tilemap == PATH_TILEMAP
        ):
            mountain_exit = frame
    inbound = (grape - enter) if enter is not None and grape is not None else None
    outbound_start = leave_stand if leave_stand is not None else grape
    outbound = (
        (mountain_exit - outbound_start)
        if outbound_start is not None and mountain_exit is not None
        else None
    )
    pick = (
        (leave_stand - grape)
        if grape is not None and leave_stand is not None
        else None
    )
    return {
        "mountain_entry_to_grape": format_segment_time(inbound),
        "grape_to_mountain_exit": format_segment_time(outbound),
        "pick_keep": format_segment_time(pick),
        "marks": {
            "mountain_enter": enter,
            "grape_stand": grape,
            "leave_stand": leave_stand,
            "mountain_exit": mountain_exit,
        },
    }


def nearby_tile_scan(ram, *, radius: int = 3) -> list[dict]:
    """Live  (2r+1)^2 tile window for probes and bush facing."""
    pos = get_pos_from_ram(ram)
    tx, ty = pos.x // 16, pos.y // 16
    rows: list[dict] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = tx + dx, ty + dy
            tid = int(get_tile_at(ram, x, y))
            rows.append(
                {
                    "tile": [x, y],
                    "id": tid,
                    "hex": f"0x{tid:02X}",
                    "label": tile_label(tid),
                    "walkable": tid in MOUNTAIN_WALKABLE,
                }
            )
    return rows


def at_grape_stand(ram, *, radius: int = GRAPE_STAND_RADIUS) -> bool:
    pos = get_pos_from_ram(ram)
    return (
        abs(int(pos.x) - GRAPE_STAND_PX[0]) <= radius
        and abs(int(pos.y) - GRAPE_STAND_PX[1]) <= radius
    )


def on_grape_pixel(ram, *, slop: int = GRAPE_STAND_NUDGE) -> bool:
    pos = get_pos_from_ram(ram)
    return (
        abs(int(pos.x) - GRAPE_STAND_PX[0]) <= slop
        and abs(int(pos.y) - GRAPE_STAND_PX[1]) <= slop
    )


def face_toward_grape(ram) -> str:
    """Face the ground-grape cell. Already on-tile → down (tape last step)."""
    pos = get_pos_from_ram(ram)
    tx, ty = int(pos.x) // 16, int(pos.y) // 16
    dx = GRAPE_STAND_TILE[0] - tx
    dy = GRAPE_STAND_TILE[1] - ty
    if dx == 0 and dy == 0:
        return "down"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _menu_cursor(ram) -> int:
    return int(read_ram_value(ram, "dialog_menu_cursor", raw=True))


def _box_open(scene, lock: int) -> bool:
    return scene.mode in {SceneMode.DIALOGUE, SceneMode.MENU} or lock in _LOCKED


@dataclass
class MountainBerryTask(Task):
    """House/farm → shared path fork → first mountain berry, RAM-closed."""

    name: str = "mountain_berry"
    timeout: int = 16_000
    nav_timeout: int = 6_000
    pick_attempts: int = 0
    approach_only: bool = True
    segments: Tuple[str, ...] = BERRY_NAV_SEGMENTS

    _step_count: int = field(default=0, init=False)
    _held_before: int = field(default=0, init=False)
    _child: Optional[Task] = field(default=None, init=False)
    _child_name: str = field(default="", init=False)
    _done_segments: set[str] = field(default_factory=set, init=False, repr=False)
    _pick_queue: deque = field(default_factory=deque, init=False, repr=False)
    _picks: int = field(default=0, init=False)
    _last_reason: str = field(default="start", init=False)
    _eat_phase: str = field(default="idle", init=False)
    _eat_wait: int = field(default=0, init=False)
    _post_pick_wait: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._held_before = int(read_held_item(world.ram))
        self._child = None
        self._child_name = ""
        self._done_segments.clear()
        self._pick_queue.clear()
        self._picks = 0
        self._last_reason = "start"
        self._eat_phase = "idle"
        self._eat_wait = 0
        self._post_pick_wait = 0
        self._arm_next(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        return self._child_name or "idle"

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._child) if self._child is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self.phase_text,
            step_count=self._step_count,
            details=(
                ("held_before", self._held_before),
                ("picks", self._picks),
                ("approach_only", int(self.approach_only)),
                ("eat_phase", self._eat_phase),
            ),
            child=child,
        )

    def _success(self, held: int) -> TaskResult:
        label = held_forage_name(held) or f"0x{held:02X}"
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=f"mountain forage held={label} (0x{held:02X})",
        )

    def _nav_for(self, name: str, world: WorldState) -> MultiMapNavTask:
        hops = list(SEGMENTS.get(name, []))
        if not hops:
            hops = segment_waypoints(name)
        pos = get_pos_from_ram(world.ram)
        tilemap = _tilemap(world)
        # North-edge spawn (y≈10) is still the map transition. Manhattan
        # would skip the south-land hops and send us through Gotz.
        if name == "mountain_entry_to_first_berry" and int(pos.y) < 80:
            sliced = hops
        elif tilemap == PATH_TILEMAP and path_coords_leaked(pos.x, pos.y):
            # Farm/mountain pixels linger on 0x0C. Do not skip the plaza.
            sliced = hops
        else:
            sliced = slice_route_from_position(hops, pos.x, pos.y, tilemap=tilemap)
        task = MultiMapNavTask(
            name=f"seg_{name}",
            waypoints=sliced or hops,
            timeout=self.nav_timeout,
            initial_settle_frames=12,
        )
        task.reset(world)
        return task

    def _arm_next(self, world: WorldState) -> None:
        tilemap = _tilemap(world)
        held = int(read_held_item(world.ram))
        if is_mountain_forage(held):
            self._child = None
            self._child_name = "done"
            self._last_reason = held_forage_name(held) or "picked"
            return
        if _needs_building_exit(tilemap):
            self._child = ExitToFarmTask()
            self._child.reset(world)
            self._child_name = "exit_to_farm"
            self._last_reason = "leaving building"
            return
        if tilemap == MOUNTAIN_TILEMAP and at_grape_stand(world.ram):
            self._child = None
            if self.approach_only or self.pick_attempts <= 0:
                self._child_name = "at_stand"
                self._last_reason = "at first grape stand"
            else:
                self._child_name = "pick"
                self._last_reason = "ready to pick"
            return
        remaining = [name for name in self.segments if name not in self._done_segments]
        nxt = first_remaining_segment(tilemap, remaining)
        if nxt is None:
            self._child = None
            if self.approach_only or self.pick_attempts <= 0:
                self._child_name = "at_stand"
                self._last_reason = "at first grape stand"
            else:
                self._child_name = "pick"
                self._last_reason = "ready to pick"
            return
        self._child = self._nav_for(nxt, world)
        self._child_name = nxt
        self._last_reason = f"nav {nxt}"

    def _queue_stand_nudge(self, ram) -> None:
        pos = get_pos_from_ram(ram)
        dx = GRAPE_STAND_PX[0] - int(pos.x)
        dy = GRAPE_STAND_PX[1] - int(pos.y)
        if abs(dx) > GRAPE_STAND_NUDGE:
            face = "right" if dx > 0 else "left"
            self._pick_queue.extend(
                repeat_action(make_action(**{face: True}), min(20, abs(dx)))
            )
        if abs(dy) > GRAPE_STAND_NUDGE:
            face = "down" if dy > 0 else "up"
            self._pick_queue.extend(
                repeat_action(make_action(**{face: True}), min(20, abs(dy)))
            )

    def _queue_pick(self, world: WorldState) -> TaskResult:
        if self._picks >= self.pick_attempts:
            pos = get_pos_from_ram(world.ram)
            held = int(read_held_item(world.ram))
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"mountain berry unverified after {self._picks} picks "
                    f"held=0x{held:02X} pos=({pos.x},{pos.y})"
                ),
            )
        self._picks += 1
        self._post_pick_wait = 0
        self._eat_phase = "idle"
        self._eat_wait = 0
        if not on_grape_pixel(world.ram):
            self._queue_stand_nudge(world.ram)
        # Tape: last step is already Down onto (326,409), idle, then A 14f
        # in place. Extra face-walk steps off the grape. Do not cycle
        # faces — south/east carpenter NPCs talk.
        face = face_toward_grape(world.ram)
        self._pick_queue.extend(
            press_a_sequence(
                None,
                face_frames=0,
                pre_press_settle_frames=10,
                hold_frames=14,
                settle_frames=12,
            )
        )
        self._last_reason = f"pick attempt={self._picks} face={face}"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._pick_queue.popleft()),
            reason=self._last_reason,
        )

    def _handle_eat_box(self, world: WorldState, *, lock: int) -> TaskResult:
        """Keep the grape. Tape: wait for the text box, A, Down, A.

        ``It's the berry of wild grape!`` draws first; Eat / Don't eat is
        the next page. Early A is ignored, so keep cycling until lock
        clears. Cursor 1 is Don't eat when the menu is live.
        """
        cursor = _menu_cursor(world.ram)
        if self._eat_phase == "idle":
            self._eat_phase = "wait"
            self._eat_wait = 0
        self._eat_wait += 1
        if self._eat_wait > 600:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"grape eat-box timeout phase={self._eat_phase} lock={lock}",
            )
        if lock == 1:
            self._eat_phase = "done"
            self._last_reason = "kept grape"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )
        # Tape cadence from box-open: ~128 idle, A 7, ~52 idle, Down 5, ~6 idle, A 7.
        cycle = (self._eat_wait - 1) % 80
        if self._eat_wait < 80:
            self._last_reason = "wait grape item box"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )
        if cursor == _EAT_CURSOR_DONT:
            self._last_reason = "confirm don't-eat"
            hold_a = cycle < 8 or 48 <= cycle < 56
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(a=hold_a)),
                reason=self._last_reason,
            )
        if cycle < 8:
            self._last_reason = "advance grape item box"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(a=True)),
                reason=self._last_reason,
            )
        if 32 <= cycle < 40:
            self._last_reason = "move eat cursor down"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(down=True)),
                reason=self._last_reason,
            )
        if 48 <= cycle < 56:
            self._last_reason = "confirm don't-eat"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(a=True)),
                reason=self._last_reason,
            )
        self._last_reason = "grape item box settle"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=self._last_reason,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout after {self._step_count}f phase={self.phase_text}",
            )

        held = int(read_held_item(world.ram))
        scene = classify_scene_from_ram(world.ram)
        lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        box = _box_open(scene, lock)

        if is_mountain_forage(held) and box:
            # Item box after A-pick is Eat / Don't eat — keep the grape.
            return self._handle_eat_box(world, lock=lock)
        if is_mountain_forage(held) and lock == 1:
            if self._picks > 0 and self._eat_phase == "idle":
                self._post_pick_wait += 1
                if self._post_pick_wait < 40:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                        reason="wait grape item box",
                    )
            return self._success(held)
        if is_mountain_forage(held):
            return self._success(held)

        if box:
            # Ate the grape (held dropped) or Gotz talk. Both fail closed.
            if _tilemap(world) == MOUNTAIN_TILEMAP:
                why = "ate grape" if self._picks > 0 else "talked on mountain"
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"{why} scene={scene.mode.value} lock={lock}",
                )
            return dismiss_dialogue_result(
                self._step_count,
                reason="dismiss mountain dialogue",
            )

        queued = drain_action_queue(self._pick_queue, reason=self._last_reason)
        if queued is not None:
            return queued

        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status in {TaskStatus.FAILURE, TaskStatus.BLOCKED}:
                # Do not skip a blocked carpenter-corridor hop — that is how
                # we walk into Gotz. Approach-only fails closed.
                if (
                    not self.approach_only
                    and _tilemap(world) == MOUNTAIN_TILEMAP
                    and self._child_name in self.segments
                ):
                    self._done_segments.add(self._child_name)
                    self._child = None
                    self._arm_next(world)
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(np.zeros(12, dtype=np.int32)),
                        reason=f"{self._child_name} soft-fail → stand",
                    )
                return TaskResult(
                    status=result.status,
                    action=result.action,
                    reason=f"{self._child_name}: {result.reason or result.status.value}",
                )
            if self._child_name in self.segments:
                self._done_segments.add(self._child_name)
            self._child = None
            self._arm_next(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(np.zeros(12, dtype=np.int32)),
                reason=f"{self._child_name or 'pick'} next",
            )

        if self.approach_only or self.pick_attempts <= 0:
            pos = get_pos_from_ram(world.ram)
            if at_grape_stand(world.ram):
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"at first grape stand pos=({pos.x},{pos.y}) pick unverified",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"grape stand miss pos=({pos.x},{pos.y}) tilemap=0x{_tilemap(world):02X}",
            )
        return self._queue_pick(world)


def go_to_town_waypoints() -> List[Waypoint]:
    """Farm → path fork → town. Same ``farm_to_path`` hop as mountain berry."""
    return segment_waypoints("farm_to_path", "path_to_town")


__all__ = [
    "MOUNTAIN_FORAGE_ITEMS",
    "MOUNTAIN_BERRY_ITEMS",
    "BERRY_NAV_SEGMENTS",
    "TOWN_NAV_SEGMENTS",
    "GRAPE_STAND_PX",
    "GRAPE_STAND_TILE",
    "MountainBerryTask",
    "at_grape_stand",
    "face_toward_grape",
    "first_remaining_segment",
    "held_forage_name",
    "is_mountain_forage",
    "go_to_town_waypoints",
    "nearby_tile_scan",
    "on_grape_pixel",
    "format_segment_time",
    "mountain_corridor_segments",
    "SEGMENT_FPS",
]
