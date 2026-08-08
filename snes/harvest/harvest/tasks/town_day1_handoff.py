"""Spring D1 town handoff — six talks, truck leave, shed pickups, sleep → D2.

Precomputed from ``docs/town_day1_recon.md`` and verified stands in
``tasks/town_day1_rest.json`` (2026-08-01). Controller-only; no RAM writes.

Completion mask is ``d1_town_event_mask`` ``0x3F`` before the truck leave
response. Flower owner + livestock gift the watering can and grass seed bag
onto the shed shelf (already present from new-game init as row2 ``0x88``);
this handoff also *picks them up* into the 2-slot carry pair after return.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.carry import seed_item_id, tool_in_carry_pair
from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import SceneMode, classify_scene_from_ram
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import Tool
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.planner.tasks.home import GoToSleepTask
from harvest.planner.day_plan_status import TASKS_DIR
from harvest.planner.tasks.inventory import (
    EnsureCarryToolTask,
    RecordingSliceSpec,
    SHED_SEED_SPECS,
    ShedFetchItemTask,
    load_recording_slice,
)
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.farm_clearer import make_action
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
    press_button_sequence,
)

TARGET_MASK = 0x3F
TOWN_TILEMAP = 0x04
PATH_TILEMAP = 0x0C
FARM_TILEMAP = 0x00
FLOWER_SHOP_TILEMAP = 0x1C
FLOWER_BACK_TILEMAP = 0x1D
CHURCH_TILEMAP = 0x1B
ANIMAL_SHOP_TILEMAP = 0x24

BIT_ANN = 0x01
BIT_EVE = 0x02
BIT_NINA = 0x04
BIT_FLOWER_OWNER = 0x08
BIT_LIVESTOCK = 0x10
BIT_MARIA = 0x20
BIT_LEFT_TOWN = 0x40  # set by truck leave (decomp)


def read_mask(ram) -> int:
    return int(read_ram_value(ram, "d1_town_event_mask")) & 0xFF


def mask_has(ram, bit: int) -> bool:
    return bool(read_mask(ram) & bit)


def _clone_route(name: str) -> List[Waypoint]:
    route = ROUTES.get(name)
    if not route:
        raise KeyError(f"missing route {name!r}")
    return list(route)


@dataclass
class WaitForMaskBitTask(Task):
    """Mash A through dialogue until a D1 town bit is set (or already set)."""

    name: str = "wait_mask_bit"
    bit: int = 0
    timeout: int = 900
    stable_clear: int = 8

    _step_count: int = field(default=0, init=False)
    _clear_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._clear_frames = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if mask_has(world.ram, self.bit):
            scene = classify_scene_from_ram(world.ram)
            input_lock = int(read_ram_value(world.ram, "input_lock"))
            if input_lock == 1 and not scene.needs_input_dismiss:
                self._clear_frames += 1
                if self._clear_frames >= self.stable_clear:
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        reason=f"mask bit 0x{self.bit:02X} set (mask=0x{read_mask(world.ram):02X})",
                    )
            else:
                self._clear_frames = 0
                return dismiss_dialogue_result(self._step_count, reason="clear after bit set")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"mask bit 0x{self.bit:02X} not set after {self.timeout}f "
                f"(mask=0x{read_mask(world.ram):02X})",
            )
        return dismiss_dialogue_result(self._step_count, reason=f"talk bit 0x{self.bit:02X}")


@dataclass
class ScriptedWalkTask(Task):
    """Hold a direction (optional B) for a fixed frame budget, then succeed."""

    name: str = "scripted_walk"
    direction: str = "up"
    frames: int = 30
    run: bool = True

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.frames:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"walked {self.direction} {self.frames}f")
        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
            return dismiss_dialogue_result(self._step_count)
        kwargs = {self.direction: True}
        if self.run:
            kwargs["b"] = True
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**kwargs)),
            reason=f"scripted {self.direction}",
        )


@dataclass
class PressAUntilBitOrTimeout(Task):
    """Face, press A, mash dialogue; success if bit sets, soft-success on timeout optional."""

    name: str = "press_a_until_bit"
    bit: int = 0
    face: Optional[str] = None
    attempts: int = 4
    attempt_timeout: int = 220
    required: bool = True

    _step_count: int = field(default=0, init=False)
    _attempt: int = field(default=0, init=False)
    _queue: deque = field(default_factory=deque, init=False)
    _phase: str = field(default="press", init=False)
    _phase_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._attempt = 0
        self._queue = deque()
        self._phase = "press"
        self._phase_frames = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _queue_press(self) -> None:
        self._queue.extend(
            press_a_sequence(
                self.face,
                face_frames=3 if self.face else 0,
                pre_press_settle_frames=4,
                hold_frames=20,
                settle_frames=10,
            )
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if mask_has(world.ram, self.bit):
            scene = classify_scene_from_ram(world.ram)
            if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
                return dismiss_dialogue_result(self._step_count, reason="dismiss after success")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"bit 0x{self.bit:02X} set",
            )

        if self._step_count > self.attempts * self.attempt_timeout + 60:
            if self.required:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"failed bit 0x{self.bit:02X} mask=0x{read_mask(world.ram):02X}",
                )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"soft-skip bit 0x{self.bit:02X} mask=0x{read_mask(world.ram):02X}",
            )

        queued = drain_action_queue(self._queue, reason="press queue")
        if queued is not None:
            return queued

        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
            return dismiss_dialogue_result(self._step_count)

        if self._phase == "press":
            self._attempt += 1
            self._queue_press()
            self._phase = "wait"
            self._phase_frames = 0
            queued = drain_action_queue(self._queue, reason="press queue")
            if queued is not None:
                return queued

        self._phase_frames += 1
        if self._phase_frames > self.attempt_timeout:
            self._phase = "press"
            self._phase_frames = 0
        # Idle between attempts. Do not hold face directions — on this ROM
        # holding a D-pad walks and can overshoot the talk stand (Ann probe).
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class SkipIfBitSet(Task):
    """Decorator: if bit already set at start, succeed; else run child to completion.

    Once the child starts, we keep running it even after the bit flips (talk
    success) so exit routes still execute.
    """

    name: str = "skip_if_bit"
    bit: int = 0
    child: Optional[Task] = None

    _started: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._started = False

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="skip" if not self._started else "child",
            details=(("bit", self.bit),),
            child=task_progress_snapshot(self.child) if self._started else None,
        )

    def step(self, world: WorldState) -> TaskResult:
        if not self._started and mask_has(world.ram, self.bit):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"bit 0x{self.bit:02X} already set",
            )
        if self.child is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="skip_if_bit missing child")
        if not self._started:
            self.child.reset(world)
            self._started = True
        return self.child.step(world)


@dataclass
class SequenceTask(Task):
    """Ordered child tasks with shared name for progress."""

    name: str = "sequence"
    tasks: Sequence[Task] = field(default_factory=tuple)

    _index: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._index = 0
        self._started = False

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        child = None
        if self._started and self._index < len(self.tasks):
            child = task_progress_snapshot(self.tasks[self._index])
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=(
                getattr(self.tasks[self._index], "name", "?")
                if self._index < len(self.tasks)
                else "done"
            ),
            phase_index=self._index,
            child=child,
        )

    def step(self, world: WorldState) -> TaskResult:
        while self._index < len(self.tasks):
            if not self._started:
                self.tasks[self._index].reset(world)
                self._started = True
            result = self.tasks[self._index].step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                return TaskResult(
                    status=result.status,
                    action=result.action,
                    reason=f"{self.name}/{getattr(self.tasks[self._index], 'name', self._index)}: "
                    f"{result.reason or result.status.value}",
                )
            # success → next
            self._index += 1
            self._started = False
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"{self.name} advance {self._index}/{len(self.tasks)}",
            )
        return TaskResult(status=TaskStatus.SUCCESS, reason=f"{self.name} complete")


def _nav(name: str, waypoints: List[Waypoint], *, timeout: int = 6000, settle: int = 20) -> MultiMapNavTask:
    return MultiMapNavTask(
        name=name,
        waypoints=waypoints,
        timeout=timeout,
        initial_settle_frames=settle,
    )


def _talk_route(
    name: str,
    route_name: str,
    bit: int,
    *,
    face: str,
    timeout: int = 6000,
    exit_route: str | None = None,
    required: bool = True,
) -> Task:
    """Nav to stand, face+mash until bit, optional exit route back to town."""
    steps: list[Task] = [
        _nav(f"nav_{name}", _clone_route(route_name), timeout=timeout),
        PressAUntilBitOrTimeout(
            name=f"talk_{name}",
            bit=bit,
            face=face,
            attempts=6,
            attempt_timeout=200,
            required=required,
        ),
    ]
    if exit_route:
        steps.append(_nav(f"exit_{name}", _clone_route(exit_route), timeout=timeout, settle=15))
    return SkipIfBitSet(
        name=f"skip_{name}",
        bit=bit,
        child=SequenceTask(name=name, tasks=tuple(steps)),
    )


def _shed_starter_tools(*, exit_when_done: bool = True, required: bool = True) -> Task:
    """Pick free D1 grass seeds + watering can from the tool shed.

    New-game init puts both on shed_items_row_2 (``0x88`` = can 0x80 | grass
    0x08). Stock ``grass_seeds`` is already 1; equipping requires a shelf A.
    Carry only holds two slots — order is grass then can so both stay ready.

    Verified from ``house_size=0`` morning house. Some D1 fixtures incorrectly
    have ``house_size=2`` (AnnEve / rest_end), which breaks ExitToFarm — set
    ``required=False`` to soft-continue the handoff in that case.
    """
    grass_shelf = SHED_SEED_SPECS["grass"]
    seq = SequenceTask(
        name="shed_starter_tools",
        tasks=(
            ShedFetchItemTask(
                name="pick_grass_seeds",
                item_id=seed_item_id("grass"),
                shelf=grass_shelf,
                exit_when_done=False,
            ),
            EnsureCarryToolTask(
                name="pick_watering_can",
                tool_id=int(Tool.WATERING_CAN),
                exit_when_done=exit_when_done,
            ),
            _AssertCarryToolsTask(
                name="assert_starter_tools",
                required_ids=(seed_item_id("grass"), int(Tool.WATERING_CAN)),
            ),
        ),
    )
    if required:
        return seq
    return _SoftOptionalTask(name="shed_starter_tools_optional", child=seq)


def build_day1_handoff_tasks(
    *,
    include_sleep: bool = True,
    require_full_mask: bool = True,
    pick_starter_tools: bool = True,
    require_starter_tools: bool = False,
    use_rest_recording: bool = True,
) -> SequenceTask:
    """Build the full D1 town → truck → shed pickups → optional sleep sequence.

    Talk stands verified from ``tasks/town_day1_rest.json`` (Ann|Eve start,
    full mask 0x3F, truck cutscene → house, sleep → D2). Outdoor talks first.

    ``require_starter_tools`` forces shed grass+can into carry (Gate B /
    ``house_size=0``). Soft-optional when false so AnnEve ``house_size=2``
    fixtures can still finish the truck/sleep path.

    ``use_rest_recording``: when True and the rest capture exists, replay it
    for the remaining four talks + truck + sleep (AnnEve oracle path). When
    False (clean power-on / Town_Gate with Ann|Eve still open), use composed
    pure routes — the rest recording desyncs if Ann|Eve were just run pure
    (input_lock / path drift).
    """
    # Flower owner: enter shop, remap, push to counter stand ~(34,347), face down A.
    # town_day1_rest bit 0x08 at (34,347) Down+A.
    flower_owner = SkipIfBitSet(
        name="skip_flower_owner",
        bit=BIT_FLOWER_OWNER,
        child=SequenceTask(
            name="flower_owner",
            tasks=(
                _nav("to_flower_shop", _clone_route("d1_town_to_flower_shop"), timeout=5000),
                # Remap interior coords (probe: left then up settles ~144,456).
                ScriptedWalkTask(name="shop_remap_left", direction="left", frames=40, run=True),
                ScriptedWalkTask(name="shop_remap_up", direction="up", frames=20, run=False),
                # Push past counter lip toward owner object ~(40,360) / stand y~347.
                _HoldButtonsTask(name="counter_push", buttons=("up", "a"), frames=140),
                ScriptedWalkTask(name="to_owner_x", direction="left", frames=55, run=True),
                ScriptedWalkTask(name="to_owner_y", direction="up", frames=20, run=False),
                PressAUntilBitOrTimeout(
                    name="owner_a",
                    bit=BIT_FLOWER_OWNER,
                    face="down",
                    attempts=10,
                    attempt_timeout=180,
                    required=True,
                ),
                # Exit front room → town (same door as nina's shop exit).
                ScriptedWalkTask(name="owner_to_door_x", direction="right", frames=40, run=True),
                ScriptedWalkTask(name="owner_exit_down", direction="down", frames=100, run=True),
                _nav(
                    "owner_exit_town",
                    [
                        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=18),
                        Waypoint(
                            tilemap=0x1C,
                            target_px=(144, 480),
                            radius=12,
                            is_exit=True,
                            exit_direction="down",
                        ),
                        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
                    ],
                    timeout=4000,
                    settle=10,
                ),
            ),
        ),
    )

    # Nina: play town_day1_rest from flower-door spawn through talk.
    # Slice assumes shop entry coords ~(598,218) → remap → back room → (101,102) A.
    nina_rest = load_recording_slice(
        RecordingSliceSpec("town_day1_rest", start_frame=4564, end_frame=5300),
        TASKS_DIR,
    )
    nina_rest.name = "nina_rest_talk"
    nina = SkipIfBitSet(
        name="skip_nina",
        bit=BIT_NINA,
        child=SequenceTask(
            name="nina",
            tasks=(
                # Stop at door spawn so the recording slice lines up.
                _nav(
                    "to_flower_door",
                    [
                        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
                        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
                        Waypoint(
                            tilemap=0x04,
                            target_px=(600, 262),
                            radius=10,
                            is_exit=True,
                            exit_direction="up",
                        ),
                        Waypoint(tilemap=0x1C, target_px=(598, 218), radius=18),
                    ],
                    timeout=5000,
                    settle=25,
                ),
                nina_rest,
                # Slice can desync on talk; force stand + A if bit not yet set.
                _nav("nina_stand", _clone_route("d1_flower_back_to_nina"), timeout=4000, settle=15),
                PressAUntilBitOrTimeout(
                    name="talk_nina",
                    bit=BIT_NINA,
                    face="left",
                    attempts=10,
                    attempt_timeout=180,
                    required=True,
                ),
                _nav("nina_exit", _clone_route("d1_flower_back_exit_to_town"), timeout=5000, settle=15),
            ),
        ),
    )

    # Truck leave often cutscenes into the farmhouse (town_day1_rest). Soft-nav
    # to farm is optional; _TruckLeaveTask succeeds on non-town tilemap.
    truck = SequenceTask(
        name="truck_leave",
        tasks=(
            _nav("to_truck", _clone_route("d1_town_to_truck"), timeout=7000, settle=15),
            _TruckLeaveTask(),
        ),
    )

    # Outdoor first (Ann + Eve ROM-verified on clean D1 entry).
    parts: List[Task] = [
        _talk_route("ann", "d1_town_to_ann", BIT_ANN, face="left", timeout=5000),
        _talk_route("eve", "d1_town_to_eve", BIT_EVE, face="up", timeout=5000),
    ]
    if require_full_mask:
        # Prefer the verified human capture only when the run already matches
        # its AnnEve entry (mask 0x03). Clean power-on/Town_Gate must use
        # composed pure routes — rest desyncs after pure Ann|Eve (rr-bhr).
        rest_path = __import__("os").path.join(TASKS_DIR, "town_day1_rest.json")
        if use_rest_recording and __import__("os").path.isfile(rest_path):
            # Full human rest capture (Ann|Eve → mask 0x3F → truck → house sleep
            # → D2). Mask clears on day advance, so assert peak mask mid-run is
            # not possible after the fact; success is day≥2 + optional shed.
            rest = load_recording_slice(
                RecordingSliceSpec("town_day1_rest", start_frame=0, end_frame=None),
                TASKS_DIR,
            )
            rest.name = "town_day1_rest_recording"
            parts.append(rest)
            # Do not re-sleep: recording already ends D2 morning house.
            include_sleep = False
        else:
            parts.extend(
                [
                    SkipIfBitSet(
                        name="skip_livestock",
                        bit=BIT_LIVESTOCK,
                        child=SequenceTask(
                            name="livestock",
                            tasks=(
                                _nav(
                                    "nav_livestock",
                                    _clone_route("d1_town_to_livestock"),
                                    timeout=7000,
                                ),
                                ScriptedWalkTask(
                                    name="livestock_up", direction="up", frames=30, run=True
                                ),
                                ScriptedWalkTask(
                                    name="livestock_right",
                                    direction="right",
                                    frames=45,
                                    run=True,
                                ),
                                ScriptedWalkTask(
                                    name="livestock_down",
                                    direction="down",
                                    frames=22,
                                    run=False,
                                ),
                                PressAUntilBitOrTimeout(
                                    name="talk_livestock",
                                    bit=BIT_LIVESTOCK,
                                    face="down",
                                    attempts=12,
                                    attempt_timeout=200,
                                    required=True,
                                ),
                                _nav(
                                    "exit_livestock",
                                    _clone_route("d1_livestock_to_town"),
                                    timeout=5000,
                                    settle=15,
                                ),
                                ScriptedWalkTask(
                                    name="livestock_clear_door",
                                    direction="down",
                                    frames=40,
                                    run=True,
                                ),
                            ),
                        ),
                    ),
                    nina,
                    flower_owner,
                    SkipIfBitSet(
                        name="skip_maria",
                        bit=BIT_MARIA,
                        child=SequenceTask(
                            name="maria",
                            tasks=(
                                _nav(
                                    "nav_maria_door",
                                    _clone_route("d1_town_to_maria"),
                                    timeout=7000,
                                ),
                                ScriptedWalkTask(
                                    name="church_door_up",
                                    direction="up",
                                    frames=80,
                                    run=True,
                                ),
                                _nav(
                                    "nav_maria_stand",
                                    _clone_route("d1_church_to_maria"),
                                    timeout=4000,
                                    settle=20,
                                ),
                                PressAUntilBitOrTimeout(
                                    name="talk_maria",
                                    bit=BIT_MARIA,
                                    face="up",
                                    attempts=10,
                                    attempt_timeout=180,
                                    required=True,
                                ),
                                _nav(
                                    "exit_maria",
                                    _clone_route("d1_maria_to_town"),
                                    timeout=5000,
                                    settle=15,
                                ),
                            ),
                        ),
                    ),
                    _AssertMaskTask(name="assert_mask", expected=TARGET_MASK),
                    truck,
                ]
            )
        if pick_starter_tools:
            # Free D1 grass bag + watering can into carry.
            # Required on clean house_size=0 (power-on / Gate B). Soft-optional
            # when house_size!=0: AnnEve/rest fixtures are size2 and ExitToFarm
            # can fall into tilemap 0x5F.
            parts.append(
                _shed_starter_tools(
                    exit_when_done=True,
                    required=bool(require_starter_tools),
                )
            )
        if include_sleep:
            parts.append(GoToSleepTask(name="sleep_to_d2", timeout=12000))
    else:
        # Baseline progress run: only the proven outdoor pair.
        parts.append(_AssertMaskTask(name="assert_ann_eve", expected=BIT_ANN | BIT_EVE))
    return SequenceTask(name="town_day1_handoff", tasks=tuple(parts))


@dataclass
class _HoldButtonsTask(Task):
    """Hold a set of buttons for N frames (used for A+Up counter push)."""

    name: str = "hold_buttons"
    buttons: Tuple[str, ...] = ()
    frames: int = 30

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.frames:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"held {self.buttons} {self.frames}f")
        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
            return dismiss_dialogue_result(self._step_count)
        kwargs = {b: True for b in self.buttons}
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**kwargs)),
            reason=f"hold {self.buttons}",
        )


@dataclass
class _AssertMaskTask(Task):
    name: str = "assert_mask"
    expected: int = TARGET_MASK

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        mask = read_mask(world.ram)
        if mask == self.expected or (mask & self.expected) == self.expected:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"mask 0x{mask:02X} covers 0x{self.expected:02X}",
            )
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"mask 0x{mask:02X} missing bits 0x{(self.expected & ~mask):02X}",
        )


@dataclass
class _AssertCarryToolsTask(Task):
    """Require the given tool/item ids to be in the 2-slot carry pair."""

    name: str = "assert_carry_tools"
    required_ids: Tuple[int, ...] = ()

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        missing = [i for i in self.required_ids if not tool_in_carry_pair(world.ram, i)]
        if not missing:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"carry has {[f'0x{i:02X}' for i in self.required_ids]}",
            )
        sel = int(read_ram_value(world.ram, "tool_selected"))
        back = int(read_ram_value(world.ram, "tool_backpack"))
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=(
                f"carry missing {[f'0x{i:02X}' for i in missing]} "
                f"(have 0x{sel:02X}/0x{back:02X})"
            ),
        )


@dataclass
class _SoftOptionalTask(Task):
    """Run child; convert failure into soft success so the handoff can continue."""

    name: str = "soft_optional"
    child: Optional[Task] = None

    _started: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._started = False

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="child" if self._started else "start",
            child=task_progress_snapshot(self.child) if self._started else None,
        )

    def step(self, world: WorldState) -> TaskResult:
        if self.child is None:
            return TaskResult(status=TaskStatus.SUCCESS, reason="no child")
        if not self._started:
            self.child.reset(world)
            self._started = True
        result = self.child.step(world)
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                action=result.action,
                reason=f"soft-skip: {result.reason or result.status.value}",
            )
        return result


@dataclass
class _TruckLeaveTask(Task):
    """Talk to the truck/shipper and accept leave until we leave town or set 0x40."""

    name: str = "truck_leave_dialog"
    timeout: int = 3600

    _step_count: int = field(default=0, init=False)
    _queue: deque = field(default_factory=deque, init=False)
    _pressed: bool = field(default=False, init=False)
    _face_cycle: Tuple[str, ...] = ("left", "up", "left", "down")

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._queue = deque()
        self._pressed = False

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        mask = read_mask(world.ram)
        if tilemap != TOWN_TILEMAP or (mask & BIT_LEFT_TOWN):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"left town tm=0x{tilemap:02X} mask=0x{mask:02X}",
            )
        if self._step_count > self.timeout:
            # Still in town — fail so shed is not attempted from the road.
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"truck leave stuck in town mask=0x{mask:02X}",
            )

        queued = drain_action_queue(self._queue)
        if queued is not None:
            return queued

        scene = classify_scene_from_ram(world.ram)
        input_lock = int(read_ram_value(world.ram, "input_lock"))
        if input_lock != 1 or scene.needs_input_dismiss or scene.mode in {
            SceneMode.DIALOGUE,
            SceneMode.MENU,
        }:
            # Leave menu: cycle down a few times then A (rest recording uses A/down).
            phase = self._step_count % 24
            if phase in {0, 6, 12}:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(down=True)),
                    reason="truck menu cursor",
                )
            return dismiss_dialogue_result(self._step_count, reason="truck dialog")

        # Re-approach from a few facings — truck sprite stand is picky.
        if not self._pressed or self._step_count % 160 == 0:
            self._pressed = True
            face = self._face_cycle[(self._step_count // 160) % len(self._face_cycle)]
            self._queue.extend(
                press_a_sequence(
                    face,
                    face_frames=6,
                    pre_press_settle_frames=8,
                    hold_frames=24,
                    settle_frames=14,
                )
            )
            queued = drain_action_queue(self._queue)
            if queued is not None:
                return queued
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class TownDay1HandoffTask(Task):
    """Full precomputed D1 town handoff for natural-entry automation."""

    name: str = "town_day1_handoff"
    include_sleep: bool = True
    require_full_mask: bool = True
    pick_starter_tools: bool = True
    # None = auto: require shed when house_size==0 (clean power-on / Gate B).
    require_starter_tools: Optional[bool] = None
    timeout: int = 90_000

    _inner: Optional[SequenceTask] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _require_starter_tools_effective: bool = field(default=False, init=False)
    _house_size_at_start: int = field(default=-1, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        house_size = int(read_ram_value(world.ram, "house_size", raw=True))
        self._house_size_at_start = house_size
        if self.require_starter_tools is None:
            # Clean starter house is size 0; upgraded fixtures (AnnEve) are 2.
            require_shed = house_size == 0
        else:
            require_shed = bool(self.require_starter_tools)
        self._require_starter_tools_effective = require_shed
        # Rest recording only when Ann|Eve bits already set (AnnEve oracle).
        # Clean power-on/Town_Gate (mask 0) must compose pure routes (rr-bhr).
        start_mask = read_mask(world.ram)
        ann_eve_ready = (start_mask & (BIT_ANN | BIT_EVE)) == (BIT_ANN | BIT_EVE)
        use_rest = ann_eve_ready and not require_shed
        self._inner = build_day1_handoff_tasks(
            include_sleep=self.include_sleep,
            require_full_mask=self.require_full_mask,
            pick_starter_tools=self.pick_starter_tools,
            require_starter_tools=require_shed,
            use_rest_recording=use_rest,
        )
        self._inner.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._inner) if self._inner else None
        phase = ""
        if self._inner is not None:
            phase = self._inner.progress_snapshot().phase_text or ""
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=phase,
            step_count=self._step_count,
            child=child,
        )

    def summary(self, world: WorldState) -> dict:
        mask = read_mask(world.ram)
        sel = int(read_ram_value(world.ram, "tool_selected"))
        back = int(read_ram_value(world.ram, "tool_backpack"))
        return {
            "mask": mask,
            "mask_hex": f"0x{mask:02X}",
            "mask_complete": (mask & TARGET_MASK) == TARGET_MASK,
            "day": int(read_ram_value(world.ram, "day")),
            "season": int(read_ram_value(world.ram, "season")),
            "hour": int(read_ram_value(world.ram, "hour")),
            "tilemap": int(read_ram_value(world.ram, "tilemap")),
            "x": int(read_ram_value(world.ram, "player_x")),
            "y": int(read_ram_value(world.ram, "player_y")),
            "tool_selected": sel,
            "tool_backpack": back,
            "has_watering_can": tool_in_carry_pair(world.ram, int(Tool.WATERING_CAN)),
            "has_grass_seeds": tool_in_carry_pair(world.ram, seed_item_id("grass")),
            "grass_seeds_stock": int(read_ram_value(world.ram, "grass_seeds")),
            "house_size": int(read_ram_value(world.ram, "house_size", raw=True)),
            "house_size_at_start": self._house_size_at_start,
            "require_starter_tools": self._require_starter_tools_effective,
            "frames": self._step_count,
            "phase": (
                self._inner.progress_snapshot().phase_text if self._inner is not None else ""
            ),
        }

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._inner is None:
            self.reset(world)
        assert self._inner is not None
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"handoff timeout mask=0x{read_mask(world.ram):02X}",
            )
        return self._inner.step(world)


__all__ = [
    "TARGET_MASK",
    "TownDay1HandoffTask",
    "build_day1_handoff_tasks",
    "read_mask",
]
