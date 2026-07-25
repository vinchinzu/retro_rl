"""Shot preparation and swing timing for Hal's Hole in One Golf.

Shot pipeline (from the manual):
  Command → Shot → Aim (A) → Lie check (A) → Club (A) → Stance (A) →
  Swing: A (start) → A (power) → A (impact)

Putting uses two A presses (or one for a tap-in).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from retro_harness.protocol import (
    ActionResult,
    TaskResult,
    TaskStatus,
    WorldState,
)

from hals_golf.core.actions import (
    CONFIRM,
    DOWN,
    LEFT,
    RIGHT,
    idle,
    named_script,
    press,
)
from hals_golf.core.ram import (
    LIE_PUTTING_GREEN,
    WRAM_LIE_TYPE,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
    read_u16_le,
    read_u8,
)
from hals_golf.core.scene import is_command_screen


def _stroke_count(world: WorldState) -> int:
    if "stroke_count" in world.info:
        return int(world.info["stroke_count"])
    return read_u8(world.ram, WRAM_STROKE_COUNT)


def _rest_distance(world: WorldState) -> int:
    if "rest_distance" in world.info:
        return int(world.info["rest_distance"])
    return read_u16_le(world.ram, WRAM_REST_DISTANCE)


def _lie_type(world: WorldState) -> int:
    if "lie_type" in world.info:
        return int(world.info["lie_type"])
    return read_u8(world.ram, WRAM_LIE_TYPE)


def _flight_settled(
    *,
    require_rest_change: bool,
    start_rest: int,
    start_strokes: int,
    start_lie: int,
    world: WorldState,
) -> bool:
    """True when WAIT_FLIGHT may accept success.

    VS HAL can bump the stroke byte without moving the ball. Require a REST
    change for full swings, but allow a stroke bump on green/short putts where
    hole-outs often leave REST stale until the scorecard.
    """
    if not require_rest_change:
        return True
    rest_now = _rest_distance(world)
    if rest_now != start_rest or rest_now == 0:
        return True
    if _stroke_count(world) == start_strokes:
        return False
    return start_lie == LIE_PUTTING_GREEN


class ShotPhase(Enum):
    OPEN_SHOT = auto()
    CONFIRM_AIM = auto()
    CONFIRM_LIE = auto()
    CONFIRM_CLUB = auto()
    CONFIRM_STANCE = auto()
    SWING_START = auto()
    SWING_POWER = auto()
    SWING_IMPACT = auto()
    WAIT_FLIGHT = auto()
    DONE = auto()


@dataclass
class ShotTask:
    """Execute one full shot with tuned 3-click timing."""

    name: str = "shot"
    # Frames after backswing start before stopping the meter (power click).
    power_delay: int = 42
    # Frames after power click before impact click.
    impact_delay: int = 26
    # Wait after impact for ball flight / next command menu.
    flight_wait: int = 240
    # Optional down-presses to move command cursor onto Shot if needed.
    cursor_downs: int = 0
    # Signed aim taps before confirming the aiming screen. Negative is left.
    aim_steps: int = 0
    # Down taps on the club card (0=1W; the default bag reaches SW at 12).
    club_downs: int = 0
    # When True, WAIT_FLIGHT also requires REST to change before success so a
    # VS HAL opponent command panel cannot end the shot early.
    require_rest_change: bool = False
    # VS HAL switches straight to Hal after a hole-out. Stroke play should
    # keep its original command/timeout gate because REST can flash zero
    # transiently during ordinary ball flight.
    complete_on_rest_zero: bool = False
    _phase: ShotPhase = ShotPhase.OPEN_SHOT
    _wait: int = 0
    _queue: list[np.ndarray] = field(default_factory=list)
    _start_strokes: int = -1
    _start_rest: int = -1
    _start_lie: int = -1
    _flight_elapsed: int = 0

    def reset(self, world: WorldState) -> None:
        self._phase = ShotPhase.OPEN_SHOT
        self._wait = 0
        self._queue = []
        self._start_strokes = _stroke_count(world)
        self._start_rest = _rest_distance(world)
        self._start_lie = _lie_type(world)
        self._flight_elapsed = 0
        downs = [("DOWN", 2), ("IDLE", 8)] * max(0, self.cursor_downs)
        self._queue = named_script(
            [
                *downs,
                ("B", 3),  # Shot (B = ENTER)
                ("IDLE", 40),
            ]
        )

    def can_start(self, world: WorldState) -> bool:
        del world
        return True

    def resume_after_hotswap(self, world: WorldState) -> None:
        """Re-enter from command menu after a human intervention."""
        self.reset(world)

    def step(self, world: WorldState) -> TaskResult:
        if self._queue:
            action = self._queue.pop(0)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=action, reason=self._phase.name),
                meta={"phase": self._phase.name},
            )

        if self._wait > 0:
            self._wait -= 1
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason="wait"),
                meta={"phase": self._phase.name, "wait": self._wait},
            )

        if self._phase == ShotPhase.OPEN_SHOT:
            self._phase = ShotPhase.CONFIRM_AIM
            aim_button = LEFT if self.aim_steps < 0 else RIGHT
            self._queue = []
            for _ in range(abs(self.aim_steps)):
                self._queue.extend([press(aim_button), press(aim_button)])
                self._queue.extend(idle() for _ in range(3))
            self._queue.extend(named_script([("B", 3), ("IDLE", 35)]))
        elif self._phase == ShotPhase.CONFIRM_AIM:
            self._phase = ShotPhase.CONFIRM_LIE
            self._queue = named_script([("B", 3), ("IDLE", 35)])
        elif self._phase == ShotPhase.CONFIRM_LIE:
            self._phase = ShotPhase.CONFIRM_CLUB
            self._queue = []
            for _ in range(max(0, self.club_downs)):
                self._queue.extend([press(DOWN), press(DOWN)])
                self._queue.extend(idle() for _ in range(5))
            self._queue.extend(named_script([("B", 3), ("IDLE", 35)]))
        elif self._phase == ShotPhase.CONFIRM_CLUB:
            self._phase = ShotPhase.CONFIRM_STANCE
            self._queue = named_script([("B", 3), ("IDLE", 40)])
        elif self._phase == ShotPhase.CONFIRM_STANCE:
            self._phase = ShotPhase.SWING_START
            self._queue = [press(CONFIRM), press(CONFIRM), idle(), idle()]
            # Account for the four-frame click/release pulse so the public
            # delay is the actual first-click to next-click spacing.
            self._wait = max(0, self.power_delay - len(self._queue))
        elif self._phase == ShotPhase.SWING_START:
            self._phase = ShotPhase.SWING_POWER
            self._queue = [press(CONFIRM), press(CONFIRM), idle(), idle()]
            self._wait = max(0, self.impact_delay - len(self._queue))
        elif self._phase == ShotPhase.SWING_POWER:
            self._phase = ShotPhase.SWING_IMPACT
            self._queue = [press(CONFIRM), press(CONFIRM), idle(), idle()]
            self._wait = 10
        elif self._phase == ShotPhase.SWING_IMPACT:
            self._phase = ShotPhase.WAIT_FLIGHT
            self._flight_elapsed = 0
        elif self._phase == ShotPhase.WAIT_FLIGHT:
            # A hole-out has no following command panel: VS HAL immediately
            # starts Hal's turn instead.  Waiting for the normal command/
            # timeout gate lets Hal's REST changes masquerade as another one
            # of our shots, so finish as soon as the cup has held the ball.
            if (
                self.complete_on_rest_zero
                and self._flight_elapsed >= 120
                and _rest_distance(world) == 0
            ):
                self._phase = ShotPhase.DONE
                return TaskResult(status=TaskStatus.SUCCESS)
            rest_ok = _flight_settled(
                require_rest_change=self.require_rest_change,
                start_rest=self._start_rest,
                start_strokes=self._start_strokes,
                start_lie=self._start_lie,
                world=world,
            )
            cmd_ready = (
                self._flight_elapsed >= 120
                and is_command_screen(world.obs)
                and rest_ok
            )
            timed_out = self._flight_elapsed >= self.flight_wait and (
                not self.require_rest_change or rest_ok
            )
            if cmd_ready or timed_out:
                self._phase = ShotPhase.DONE
            else:
                self._flight_elapsed += 1
                # Soft escape: give up so the mission can nudge aim/power
                # instead of looping identical chips forever.
                if (
                    self.require_rest_change
                    and self._flight_elapsed >= self.flight_wait + 480
                ):
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        action=ActionResult(
                            action=idle(), reason="rest_unchanged"
                        ),
                        meta={"phase": self._phase.name},
                    )
        elif self._phase == ShotPhase.DONE:
            return TaskResult(status=TaskStatus.SUCCESS)

        if self._queue:
            action = self._queue.pop(0)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=action, reason=self._phase.name),
                meta={"phase": self._phase.name},
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action=idle(), reason=self._phase.name),
            meta={"phase": self._phase.name},
        )


@dataclass
class PuttTask:
    """Simplified two-click putt sequence."""

    name: str = "putt"
    power_delay: int = 60
    flight_wait: int = 900
    require_rest_change: bool = False
    complete_on_rest_zero: bool = False
    # Signed aim taps on the putting aim screen before confirming the line.
    aim_steps: int = 0
    _phase: str = "prepare"
    _wait: int = 0
    _queue: list[np.ndarray] = field(default_factory=list)
    _flight_elapsed: int = 0
    _start_rest: int = -1
    _start_strokes: int = -1
    _start_lie: int = -1

    def reset(self, world: WorldState) -> None:
        self._phase = "prepare"
        self._wait = 0
        self._flight_elapsed = 0
        self._start_rest = _rest_distance(world)
        self._start_strokes = _stroke_count(world)
        self._start_lie = _lie_type(world)
        # On the green: command -> aim -> ball/putter -> meter. The meter then
        # uses two clicks (start and power), unlike a full swing's three.
        self._queue = named_script([("B", 3), ("IDLE", 55)])
        aim_button = LEFT if self.aim_steps < 0 else RIGHT
        for _ in range(abs(self.aim_steps)):
            self._queue.extend([press(aim_button), press(aim_button)])
            self._queue.extend(idle() for _ in range(3))
        self._queue.extend(
            named_script(
                [
                    ("B", 3),
                    ("IDLE", 55),
                    ("B", 3),
                    ("IDLE", 55),
                ]
            )
        )

    def can_start(self, world: WorldState) -> bool:
        del world
        return True

    def step(self, world: WorldState) -> TaskResult:
        if self._queue:
            action = self._queue.pop(0)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=action, reason=self._phase),
            )
        if self._wait > 0:
            self._wait -= 1
            if self._wait == 0 and self._phase == "power_wait":
                self._queue = [press(CONFIRM), press(CONFIRM)]
                self._phase = "flight"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason=self._phase),
            )
        if self._phase == "flight":
            # Hole-outs transition directly to the opponent/result sequence
            # and never restore our command panel.  Report success before
            # Hal's subsequent ball movement can be attributed to this putt.
            if (
                self.complete_on_rest_zero
                and self._flight_elapsed >= 120
                and _rest_distance(world) == 0
            ):
                return TaskResult(status=TaskStatus.SUCCESS)
            rest_ok = _flight_settled(
                require_rest_change=self.require_rest_change,
                start_rest=self._start_rest,
                start_strokes=self._start_strokes,
                start_lie=self._start_lie,
                world=world,
            )
            cmd_ready = (
                self._flight_elapsed >= 120
                and is_command_screen(world.obs)
                and rest_ok
            )
            timed_out = self._flight_elapsed >= self.flight_wait and (
                not self.require_rest_change or rest_ok
            )
            if cmd_ready or timed_out:
                return TaskResult(status=TaskStatus.SUCCESS)
            self._flight_elapsed += 1
            if (
                self.require_rest_change
                and self._flight_elapsed >= self.flight_wait + 480
            ):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    action=ActionResult(
                        action=idle(), reason="rest_unchanged"
                    ),
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason=self._phase),
            )
        if self._phase == "prepare":
            self._phase = "power_wait"
            self._queue = [press(CONFIRM), press(CONFIRM), idle(), idle()]
            self._wait = max(0, self.power_delay - len(self._queue))
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=self._queue.pop(0), reason=self._phase),
            )
        return TaskResult(status=TaskStatus.SUCCESS)
