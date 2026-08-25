"""Spring D1 town handoff — task helpers and mask constants.

Extracted from ``town_day1_handoff`` (LOC budget). Controllers only; no RAM writes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.carry import tool_in_carry_pair
from harvest.core.ram_catalog import read_ram_value
from harvest.core.npc_catalog import game_objects
from harvest.core.scene import SceneMode, classify_scene_from_ram
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
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
                return dismiss_dialogue_result(
                    self._step_count,
                    pulse_every=1,
                    reason="clear after bit set",
                )
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"mask bit 0x{self.bit:02X} not set after {self.timeout}f "
                f"(mask=0x{read_mask(world.ram):02X})",
            )
        return dismiss_dialogue_result(
            self._step_count,
            pulse_every=1,
            reason=f"talk bit 0x{self.bit:02X}",
        )


@dataclass
class WalkUntilCoordTask(Task):
    """Hold a direction until the camera remaps onto a tilemap + coord box.

    Building doors keep the previous map's pixels until the player walks off
    the trigger.  MultiMapNav ``run_direction`` would first align X and miss
    the remap.  Rest tape: hold Up into the animal shop until (128,200);
    hold Down out until town-space ~(600,888).
    """

    name: str = "walk_until_coord"
    direction: str = "up"
    tilemap: int = ANIMAL_SHOP_TILEMAP
    max_x: int | None = None
    min_x: int | None = None
    min_y: int | None = None
    max_y: int | None = None
    timeout: int = 240
    run: bool = True

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _in_box(self, px: int, py: int) -> bool:
        if self.max_x is not None and px >= self.max_x:
            return False
        if self.min_x is not None and px < self.min_x:
            return False
        if self.max_y is not None and py >= self.max_y:
            return False
        if self.min_y is not None and py < self.min_y:
            return False
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        px = int(read_ram_value(world.ram, "player_x"))
        py = int(read_ram_value(world.ram, "player_y"))
        remapped = tilemap == self.tilemap and self._in_box(px, py)
        if remapped:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"coord remap tm=0x{tilemap:02X} ({px},{py})",
            )
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"coord remap timeout tm=0x{tilemap:02X} ({px},{py}) "
                    f"want tm=0x{self.tilemap:02X}"
                ),
            )
        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
            return dismiss_dialogue_result(self._step_count)
        kwargs = {self.direction: True}
        if self.run:
            kwargs["b"] = True
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**kwargs)),
            reason=f"remap {self.direction} tm=0x{tilemap:02X} ({px},{py})",
        )


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
    attempt_timeout: int = 140  # was 220 — talks felt sluggish live
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
                face_frames=2 if self.face else 0,
                pre_press_settle_frames=2,
                hold_frames=10,
                settle_frames=4,
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
class TrackNpcUntilBitTask(Task):
    """Follow the nearest live NPC object and talk until an event bit sets."""

    name: str = "track_npc_until_bit"
    bit: int = 0
    timeout: int = 2400
    face_hint: Optional[str] = None

    _step_count: int = field(default=0, init=False)
    _queue: deque = field(default_factory=deque, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._queue = deque()

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if mask_has(world.ram, self.bit):
            scene = classify_scene_from_ram(world.ram)
            if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
                return dismiss_dialogue_result(self._step_count, reason="dismiss after tracked talk")
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"bit 0x{self.bit:02X} set")
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"moving NPC talk timeout bit=0x{self.bit:02X} mask=0x{read_mask(world.ram):02X}",
            )

        queued = drain_action_queue(self._queue, reason="tracked NPC press")
        if queued is not None:
            return queued
        scene = classify_scene_from_ram(world.ram)
        if scene.needs_input_dismiss or int(read_ram_value(world.ram, "input_lock")) != 1:
            return dismiss_dialogue_result(self._step_count)

        objects = game_objects(world.ram)
        player = next((obj for obj in objects if obj.is_player), None)
        candidates = [obj for obj in objects if obj.is_npc_candidate]
        if player is None or not candidates:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="waiting for live NPC object",
            )
        px, py = player.pixel
        npc = min(candidates, key=lambda obj: abs(obj.pixel[0] - px) + abs(obj.pixel[1] - py))
        dx, dy = npc.pixel[0] - px, npc.pixel[1] - py

        # NPC/object origins are not sprite centers; live livestock is
        # interactable at an object delta around (31, 14).
        if (abs(dx) <= 36 and abs(dy) <= 18) or (abs(dy) <= 36 and abs(dx) <= 18):
            preferred = "right" if abs(dx) >= abs(dy) and dx > 0 else "left"
            if abs(dy) > abs(dx):
                preferred = "down" if dy > 0 else "up"
            face = self.face_hint or preferred
            self._queue.extend(
                press_a_sequence(face, face_frames=1, pre_press_settle_frames=1, hold_frames=2, settle_frames=3)
            )
            return drain_action_queue(
                self._queue,
                reason=(
                    f"talk to moving NPC slot={npc.slot} "
                    f"sprite=0x{npc.sprite_table_idx:04X} at={npc.pixel}"
                ),
            )

        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**{direction: True})),
            reason=(
                f"track moving NPC slot={npc.slot} sprite=0x{npc.sprite_table_idx:04X} "
                f"at={npc.pixel} dx={dx} dy={dy}"
            ),
        )


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
    """Shipper leave dialog then east-gate walk (town_day1_rest 9270–9780).

    Rest capture: stand ~(715,421), Right until lock=2, mash A/B through leave
    menu, then walk Right out to path ``0x0C`` (cutscene continues into house).
    """

    name: str = "truck_leave_dialog"
    timeout: int = 4800

    _step_count: int = field(default=0, init=False)
    _queue: deque = field(default_factory=deque, init=False)
    _phase: str = field(default="engage", init=False)
    _dialog_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._queue = deque()
        self._phase = "engage"
        self._dialog_frames = 0

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
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"truck leave stuck in town mask=0x{mask:02X} phase={self._phase}",
            )

        queued = drain_action_queue(self._queue)
        if queued is not None:
            return queued

        scene = classify_scene_from_ram(world.ram)
        input_lock = int(read_ram_value(world.ram, "input_lock"))
        in_dialog = (
            input_lock != 1
            or scene.needs_input_dismiss
            or scene.mode in {SceneMode.DIALOGUE, SceneMode.MENU}
        )

        if self._phase == "engage":
            if in_dialog:
                self._phase = "dialog"
                self._dialog_frames = 0
            else:
                # Nudge right into the shipper trigger (rest used Right into lock=2).
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(right=True)),
                    reason="truck engage right",
                )

        if self._phase == "dialog":
            self._dialog_frames += 1
            if not in_dialog and self._dialog_frames > 40:
                self._phase = "exit_east"
            else:
                # Rest mixes A and B; occasional B cancels side-branches.
                if self._dialog_frames % 50 < 8:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(b=True)),
                        reason="truck dialog B",
                    )
                return dismiss_dialogue_result(self._step_count, reason="truck dialog")

        # After leave accepted, walk east out of town (path 0x0C).
        if self._phase == "exit_east":
            px = int(read_ram_value(world.ram, "player_x"))
            if px < 750:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(right=True, b=True)),
                    reason="truck exit east",
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(right=True)),
                reason="truck push gate",
            )

        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
