"""Clean Harvest Moon power-on → new-game bootstrap.

The ROM starts in an attract sequence rather than a controllable farm scene.
``PowerOnStartTask`` uses ordinary controller input to skip that sequence,
select ``START``, create the first diary, enter the deterministic ``AAAA``
name, and dismiss the opening dialogue.  It intentionally never loads a
save state or writes RAM.

The task finishes only after name entry has completed and the game reports a
controllable Spring day-1 scene.  The opening automatically walks from the
farm to the town gate at 07:00, so the task waits through that sequence rather
than handing a planner a briefly unlocked but still scripted farm scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram
from harvest.tasks.farm_clearer import make_action


# These are title/name-screen scratch registers, not persistent game facts.
# Keeping them local prevents the gameplay RAM catalog from acquiring UI-only
# implementation details.
_TITLE_MENU_KIND_ADDR = 0x0095
_TITLE_CURSOR_ADDR = 0x098D
_NAME_CURSOR_ADDR = 0x0991
_NAME_LENGTH_ADDR = 0x0994
_TRANSITION_PARAM_ADDR = 0x0094

_TITLE_TILEMAP = 0x5C
_DIARY_TILEMAP = 0x5E
_NAME_TILEMAP = 0x5F

_STARTUP_SKIP_FRAME = 600
_INPUT_COOLDOWN = 100
_DIARY_READY_DELAY = 1200
_READY_SETTLE_FRAMES = 120
_OPENING_READY_HOUR = 7


def _raw_u8(ram, address: int) -> int:
    return int(ram[address]) if address < len(ram) else 0


@dataclass
class PowerOnStartTask(Task):
    """Drive a no-state-load boot to a usable Spring day-1 entry point.

    The four ``A`` presses at the name screen deliberately produce ``AAAA``.
    A fixed short name makes the flow deterministic and avoids depending on a
    mutable SRAM diary or a hand-authored fixture.  The title menu's horizontal
    name-grid labels are reversed in this ROM, so the physical LEFT input moves
    to the game's logical ``right`` target (the ``OK`` button path).
    """

    name: str = "power_on_start"
    timeout: int = 20_000

    _step_count: int = field(default=0, init=False)
    _last_input_step: int = field(default=-_INPUT_COOLDOWN, init=False)
    _boot_skip_sent: bool = field(default=False, init=False)
    _title_confirmed: bool = field(default=False, init=False)
    _title_confirmed_step: int = field(default=0, init=False)
    _diary_selected: bool = field(default=False, init=False)
    _name_submitted: bool = field(default=False, init=False)
    _ready_frames: int = field(default=0, init=False)
    _last_reason: str = field(default="booting attract sequence", init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._last_input_step = -_INPUT_COOLDOWN
        self._boot_skip_sent = False
        self._title_confirmed = False
        self._title_confirmed_step = 0
        self._diary_selected = False
        self._name_submitted = False
        self._ready_frames = 0
        self._last_reason = "booting attract sequence"

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        if self._name_submitted:
            return "POWER_ON_SETTLE"
        if self._diary_selected:
            return "POWER_ON_OPENING"
        if self._title_confirmed:
            return "POWER_ON_NEW_DIARY"
        if self._boot_skip_sent:
            return "POWER_ON_TITLE"
        return "POWER_ON_ATTRACT"

    @property
    def progress_text(self) -> str:
        return f"phase={self.phase_text} step={self._step_count}"

    def summary(self, world: WorldState) -> dict[str, object]:
        ram = world.ram
        scene = classify_scene_from_ram(ram)
        return {
            "completed": self._name_submitted and self._ready_frames >= _READY_SETTLE_FRAMES,
            "frames": self._step_count,
            "phase": self.phase_text,
            "reason": self._last_reason,
            "date": {
                key: int(read_ram_value(ram, key))
                for key in ("year", "season", "day", "weekday", "hour", "minute")
            },
            "scene": scene.to_dict(),
            "player_name": "AAAA",
            "initial_state_loads": 0,
            "ram_writes": 0,
        }

    def _can_press(self) -> bool:
        return self._step_count - self._last_input_step >= _INPUT_COOLDOWN

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

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        ram = world.ram
        input_lock = int(read_ram_value(ram, "input_lock"))
        tilemap = int(read_ram_value(ram, "tilemap"))
        day = int(read_ram_value(ram, "day"))
        menu_kind = _raw_u8(ram, _TITLE_MENU_KIND_ADDR)
        title_cursor = _raw_u8(ram, _TITLE_CURSOR_ADDR)
        transition_param = _raw_u8(ram, _TRANSITION_PARAM_ADDR)
        name_length = _raw_u8(ram, _NAME_LENGTH_ADDR)
        name_cursor = _raw_u8(ram, _NAME_CURSOR_ADDR)

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"power-on bootstrap timed out during {self.phase_text}",
            )

        if self._can_press():
            if not self._boot_skip_sent and self._step_count >= _STARTUP_SKIP_FRAME:
                self._boot_skip_sent = True
                return self._press(reason="skip attract sequence", start=True)

            if (
                not self._title_confirmed
                and input_lock == 4
                and menu_kind == 5
                and tilemap == _TITLE_TILEMAP
            ):
                if title_cursor != 1:
                    return self._press(reason="select START on title", up=True)
                self._title_confirmed = True
                self._title_confirmed_step = self._step_count
                return self._press(reason="confirm START on title", a=True)

            if (
                self._title_confirmed
                and not self._diary_selected
                and self._step_count - self._title_confirmed_step >= _DIARY_READY_DELAY
                and input_lock == 4
                and menu_kind == 7
                and tilemap == _DIARY_TILEMAP
                and transition_param == 0
            ):
                self._diary_selected = True
                return self._press(reason="create first diary", a=True)

            if input_lock == 2:
                return self._press(reason="advance opening dialogue", a=True)

            if input_lock == 5 and tilemap == _NAME_TILEMAP:
                if name_length < 4:
                    return self._press(reason="enter deterministic player name", a=True)
                if name_cursor == 0:
                    # The game's logical right action is wired to the physical
                    # left controller bit in this name-grid implementation.
                    return self._press(reason="move to name OK", left=True)
                if name_cursor == 40:
                    return self._press(reason="move to name OK", up=True)
                if name_cursor == 70:
                    self._name_submitted = True
                    return self._press(reason="confirm player name", a=True)

        scene = classify_scene_from_ram(ram)
        ready = (
            self._name_submitted
            and day == 1
            and int(read_ram_value(ram, "hour")) >= _OPENING_READY_HOUR
            and input_lock == 1
            and scene.is_normal_map
        )
        self._ready_frames = self._ready_frames + 1 if ready else 0
        if self._ready_frames >= _READY_SETTLE_FRAMES:
            self._last_reason = "Spring day 1 controllable after power-on"
            return TaskResult(status=TaskStatus.SUCCESS, reason=self._last_reason)

        return self._idle("waiting for power-on scene to settle")


__all__ = ["PowerOnStartTask"]
