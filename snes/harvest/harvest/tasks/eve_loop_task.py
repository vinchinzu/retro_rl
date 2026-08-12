"""Repeatable Eve relationship dialogue loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
)
from harvest.tasks.nav import (
    get_pos_from_ram,
    make_action,
)
from harvest.core.npc_catalog import ROMANCE_HEART_THRESHOLDS, game_objects, romance_points_for_hearts
from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16
from harvest.tasks.recorded_task import RecordedTask
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState


ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_DIALOG_MENU_CURSOR = field_spec("dialog_menu_cursor").address


@dataclass
class EveTalkLoopTask(Task):
    """Repeat the Eve "No" dialogue attempt until the requested heart target."""

    name: str = "eve_talk_loop"
    task_name: str = "talk_eve_loop"
    tasks_dir: str = "tasks"
    target_hearts: int = 10
    max_loops: int = 300
    timeout: int = 360000
    origin_tilemap: int = 0x04
    bar_tilemap: int = 0x1E
    origin_px: Tuple[int, int] = (152, 872)
    origin_radius: int = 8
    eve_stand_px: Tuple[int, int] = (69, 450)
    bar_exit_px: Tuple[int, int] = (137, 456)
    align_timeout: int = 1200
    min_success_gain: int = 8
    max_talk_attempts_before_reset: int = 2
    daily_question_cap_points: int = 120

    _frames: List[List[int]] = field(default_factory=list, init=False)
    _talk_frames: List[List[int]] = field(default_factory=list, init=False)
    _frame_idx: int = field(default=0, init=False)
    _loop_count: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _phase: str = field(default="align_outside", init=False)
    _phase_count: int = field(default=0, init=False)
    _align_count: int = field(default=0, init=False)
    _talk_attempts: int = field(default=0, init=False)
    _attempt_start_points: int = field(default=0, init=False)
    _target_points: int = field(default=999, init=False)
    _start_points: int = field(default=0, init=False)
    _choice_frame: int = field(default=0, init=False)
    _small_gain_count: int = field(default=0, init=False)

    def _eve_points(self, ram: np.ndarray) -> int:
        return read_ram_u16(ram, field_spec("eve_hearts").address)

    def _player_action(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_PLAYER_ACTION)

    def _dialog_menu_cursor(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_DIALOG_MENU_CURSOR, live_offset=False)

    def _set_phase(self, phase: str) -> None:
        if self._phase != phase:
            self._phase = phase
            self._phase_count = 0
            self._frame_idx = 0
            self._choice_frame = 0

    def _at_recording_origin(self, ram: np.ndarray) -> bool:
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap != self.origin_tilemap:
            return False
        pos = get_pos_from_ram(ram)
        return abs(pos.x - self.origin_px[0]) <= self.origin_radius and abs(pos.y - self.origin_px[1]) <= self.origin_radius

    def _input_locked(self, ram: np.ndarray) -> bool:
        value = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        return value != 1

    def _choice_active(self, ram: np.ndarray) -> bool:
        return self._input_locked(ram) and self._player_action(ram) == 9

    def _choose_no_action(self, ram: np.ndarray) -> ActionResult:
        self._choice_frame += 1
        if self._dialog_menu_cursor(ram) == 1 and self._choice_frame > 4:
            return ActionResult(make_action(a=self._choice_frame % 16 < 10))
        phase_frame = (self._choice_frame - 1) % 48
        if phase_frame < 14:
            return ActionResult(make_action(down=True))
        if phase_frame < 22:
            return ActionResult(make_action())
        if phase_frame < 38:
            return ActionResult(make_action(a=True))
        return ActionResult(make_action())

    def _dynamic_eve_stand_px(self, ram: np.ndarray) -> Tuple[int, int]:
        candidates = []
        for obj in game_objects(ram):
            if obj.is_player:
                continue
            x, y = obj.pixel
            if obj.kind == "npc_candidate" or 0x0200 <= obj.sprite_table_idx <= 0x02FF:
                if 24 <= x <= 128 and 392 <= y <= 512:
                    candidates.append(obj)
        if not candidates:
            return self.eve_stand_px
        target = min(
            candidates,
            key=lambda obj: abs(obj.pixel[0] - self.eve_stand_px[0]) + abs(obj.pixel[1] - self.eve_stand_px[1]),
        )
        x, y = target.pixel
        return (max(48, min(128, x + 16)), max(424, min(472, y)))

    def reset(self, world: WorldState) -> None:
        recording = RecordedTask.load(self.task_name, self.tasks_dir)
        self._frames = recording.frames
        # Replay only the standing talk/prompt rhythm; live choice RAM overrides
        # the recorded menu timing so the answer stays on "No".
        self._talk_frames = self._frames[566:1039] or self._frames
        self._frame_idx = 0
        self._loop_count = 0
        self._step_count = 0
        self._phase = "align_outside"
        self._phase_count = 0
        self._align_count = 0
        self._talk_attempts = 0
        self._attempt_start_points = self._eve_points(world.ram)
        self._target_points = romance_points_for_hearts(self.target_hearts)
        self._start_points = self._attempt_start_points
        self._choice_frame = 0
        self._small_gain_count = 0

    def _align_to_origin_action(self, ram: np.ndarray) -> ActionResult | None:
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap != self.origin_tilemap:
            return None
        pos = get_pos_from_ram(ram)
        if pos.y < 700 and self._align_count < 120:
            return ActionResult(make_action())
        dx = self.origin_px[0] - pos.x
        dy = self.origin_px[1] - pos.y
        if abs(dx) > self.origin_radius:
            return ActionResult(make_action(right=dx > 0, left=dx < 0))
        if abs(dy) > self.origin_radius:
            return ActionResult(make_action(down=dy > 0, up=dy < 0))
        return ActionResult(make_action())

    def _move_to_bar_px(self, ram: np.ndarray, target: Tuple[int, int], *, radius_x: int = 8, radius_y: int = 6) -> ActionResult:
        if self._input_locked(ram):
            return ActionResult(make_action(a=self._phase_count % 16 < 8))
        pos = get_pos_from_ram(ram)
        if pos.x == 0 and pos.y == 0:
            return ActionResult(make_action())
        dx = target[0] - pos.x
        dy = target[1] - pos.y
        if abs(dy) > radius_y:
            return ActionResult(make_action(down=dy > 0, up=dy < 0, b=True))
        if abs(dx) > radius_x:
            return ActionResult(make_action(right=dx > 0, left=dx < 0, b=True))
        return ActionResult(make_action())

    def _exit_bar_action(self, ram: np.ndarray) -> ActionResult | None:
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap != self.bar_tilemap:
            return None
        if self._input_locked(ram):
            return ActionResult(make_action(a=self._phase_count % 16 < 8))
        pos = get_pos_from_ram(ram)
        dx = self.bar_exit_px[0] - pos.x
        dy = self.bar_exit_px[1] - pos.y
        if abs(dx) > 10:
            return ActionResult(make_action(right=dx > 0, left=dx < 0, b=True))
        if abs(dy) > 6:
            return ActionResult(make_action(down=dy > 0, up=dy < 0, b=True))
        return ActionResult(make_action(down=True, b=True))

    def can_start(self, world: WorldState) -> bool:
        return bool(self._frames) or os.path.exists(os.path.join(self.tasks_dir, f"{self.task_name}.json"))

    @property
    def progress_text(self) -> str:
        return f"{self._phase} loops={self._loop_count} frame={self._frame_idx}/{len(self._talk_frames)} target={self._target_points}"

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._phase_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="eve talk loop timeout")
        if not self._talk_frames:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"recording {self.task_name} has no frames")

        current_points = self._eve_points(world.ram)
        if current_points >= self._target_points:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"eve hearts {current_points}/{self._target_points} loops={self._loop_count}",
            )
        if self._target_points > self.daily_question_cap_points and current_points >= self.daily_question_cap_points:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"eve daily question cap {current_points}/{self._target_points} loops={self._loop_count}",
            )
        if self._loop_count >= self.max_loops:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"eve loop cap reached hearts={current_points}/{self._target_points}",
            )

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0

        if self._phase == "align_outside":
            if tilemap == self.bar_tilemap:
                self._set_phase("approach_eve")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="inside_bar")
            if self._at_recording_origin(world.ram):
                self._align_count = 0
                self._set_phase("enter_bar")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="origin_ready")
            self._align_count += 1
            if self._align_count <= self.align_timeout:
                action = self._align_to_origin_action(world.ram)
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=action, reason="align_outside")
            pos = get_pos_from_ram(world.ram)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"expected Eve loop origin tilemap=0x{self.origin_tilemap:02X} "
                    f"near={self.origin_px}, got tilemap=0x{tilemap:02X} pos=({pos.x},{pos.y})"
                ),
            )

        if self._phase == "enter_bar":
            if tilemap == self.origin_tilemap:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(up=True, b=True)), reason="enter_bar")
            if tilemap == self.bar_tilemap:
                if self._phase_count < 45:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="bar_settle")
                self._set_phase("approach_eve")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="bar_ready")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"expected town/bar tilemap, got 0x{tilemap:02X}")

        if self._phase == "approach_eve":
            if tilemap != self.bar_tilemap:
                self._set_phase("align_outside")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="outside_bar")
            pos = get_pos_from_ram(world.ram)
            target = self._dynamic_eve_stand_px(world.ram)
            if abs(pos.x - target[0]) <= 8 and abs(pos.y - target[1]) <= 6:
                self._attempt_start_points = current_points
                self._talk_attempts = 0
                self._set_phase("talk")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True)), reason="talk_ready")
            return TaskResult(status=TaskStatus.RUNNING, action=self._move_to_bar_px(world.ram, target), reason="approach_eve")

        if self._phase == "talk":
            gain = current_points - self._attempt_start_points
            if gain >= self.min_success_gain:
                self._loop_count += 1
                print(
                    f"[EVE] loop={self._loop_count} hearts={current_points}/{self._target_points} "
                    f"delta={current_points - self._start_points} gain={gain} attempts={self._talk_attempts + 1}"
                )
                self._set_phase("exit_bar")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="heart_gain")
            if gain > 0:
                self._small_gain_count += 1
                print(
                    f"[EVE] ignored small gain +{gain}; hearts={current_points}/{self._target_points} "
                    f"small_gains={self._small_gain_count}"
                )
                self._attempt_start_points = current_points
                self._talk_attempts = 0
                self._set_phase("exit_bar")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="small_heart_gain")
            if tilemap != self.bar_tilemap:
                self._set_phase("align_outside")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="outside_bar")
            if self._choice_active(world.ram):
                return TaskResult(status=TaskStatus.RUNNING, action=self._choose_no_action(world.ram), reason="choose_no")
            self._choice_frame = 0
            action = np.array(self._talk_frames[self._frame_idx], dtype=np.int32)
            self._frame_idx += 1
            if self._frame_idx >= len(self._talk_frames):
                self._talk_attempts += 1
                self._frame_idx = 0
                if self._talk_attempts >= self.max_talk_attempts_before_reset:
                    print(
                        f"[EVE] reset after missed talk attempts={self._talk_attempts} "
                        f"hearts={current_points}/{self._target_points}"
                    )
                    self._set_phase("exit_bar")
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                        reason="reset_after_missed_talk",
                    )
                print(f"[EVE] retry talk attempts={self._talk_attempts} hearts={current_points}/{self._target_points}")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="talk")

        if self._phase == "exit_bar":
            if tilemap == self.origin_tilemap:
                self._set_phase("align_outside")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="outside_bar")
            action = self._exit_bar_action(world.ram)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=action, reason="exit_bar")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"expected bar tilemap, got 0x{tilemap:02X}")

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown eve phase {self._phase}")


__all__ = [
    "EveTalkLoopTask",
    "ROMANCE_HEART_THRESHOLDS",
    "romance_points_for_hearts",
]
