"""Generic leftover-safe rupee/occupancy farm (no ``$066D`` writes).

Extracted from ``OverworldToArrowShopController._farm_step`` (the 0x49<->0x4A
wooden-arrow farm in ``zelda_i.level1.arrow_shop``) so any shop-buy controller
can farm rupees to a target ``N`` without poking the rupee count. The farm:

- Patrols live prey objects on ``farm_screen`` (``type_id`` not in
  ``(0, 0xFF, 0x60)``, ``hp > 0``, in-bounds) plus rupee-drop sprites
  (``type_id == 0x60``), preferring drops when both are present.
- Leaves ``farm_screen`` for ``restock_neighbor_screen`` (and back) once
  nothing is left to farm — the same 0x49<->0x4A scroll-out/scroll-in
  restock, generalized to any adjacent screen pair and a configurable
  ``restock_direction``. The leave is an occupancy walk to the open lane
  (``heart_farm.LEAVE_GOALS``), not a bare directional hold: a bush between
  Link and the edge used to leave ``max_frames`` as the only exit.
- Gives up (``farm_screen_dead``) when a restock returns a screen with
  nothing on it. Overworld waves are one-shot and every restock pair here is
  a one-screen hop, so the restock is a give-up detector, not a rupee supply
  — see ``_screen_dead``.
- Stops once ``snap.rupees >= target_rupees``.
- Returns to ``leftover_screen`` (``farm_screen`` by default; may also be
  ``restock_neighbor_screen``) before reporting done, so the caller is never
  stranded on the farm screen when it differs from the leftover screen.

This module never writes RAM. A timeout (or an unreachable ``leftover_screen``
outside the ``farm_screen``/``restock_neighbor_screen`` pair) fails closed
with a RAM glance (``screen``/``x``/``y``/``rupees``/``mode``) recorded in
``report()["glance"]`` — never a poke.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.ids import (
    FIVE_RUPEE_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.overworld.common import walk_or_swing
from zelda_i.overworld.heart_farm import FarmOccupancy, leave_goal
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "DEATH_MODE",
    "DEFAULT_EMPTY_WAIT_FRAMES",
    "DEFAULT_FARM_MAX_FRAMES",
    "DEFAULT_FARM_X_HI",
    "DEFAULT_FARM_X_LO",
    "DEFAULT_FARM_Y_HI",
    "DEFAULT_FARM_Y_LO",
    "DEFAULT_SWING_HOLD",
    "DEFAULT_SWING_PERIOD",
    "MAX_LEAVE_STALLS",
    "RUPEE_DROP_TYPE_ID",
    "RupeeFarmController",
    "RupeeFarmPhase",
]

DEATH_MODE = 17
RUPEE_DROP_TYPE_ID = RUPEE_DROP_OBJECT_TYPE
_RUPEE_STATES = frozenset({RUPEE_DROP_STATE, FIVE_RUPEE_DROP_STATE})
DEFAULT_FARM_MAX_FRAMES = 36000
DEFAULT_EMPTY_WAIT_FRAMES = 90
DEFAULT_SWING_PERIOD = 8
DEFAULT_SWING_HOLD = 3
DEFAULT_FARM_Y_LO = 88
DEFAULT_FARM_Y_HI = 210
DEFAULT_FARM_X_LO = 16
DEFAULT_FARM_X_HI = 240
DEFAULT_LEAVE_MAX_FRAMES = 400
# Two blocked leaves is a walled exit, not bad luck: fail closed instead of
# cycling wait -> leave -> wait until ``max_frames``.
MAX_LEAVE_STALLS = 2

_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}


class RupeeFarmPhase(Enum):
    FARM = auto()
    RETURN = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class RupeeFarmController:
    """Patrol ``farm_screen`` until ``snap.rupees >= target_rupees``.

    ``leftover_screen`` must be either ``farm_screen`` (default) or
    ``restock_neighbor_screen`` — the two screens this farm ever occupies.
    Anything else fails closed once the target is met rather than stranding
    or poking.
    """

    target_rupees: int
    farm_screen: int
    restock_neighbor_screen: int
    restock_direction: str
    leftover_screen: int | None = None

    max_frames: int = DEFAULT_FARM_MAX_FRAMES
    empty_wait_frames: int = DEFAULT_EMPTY_WAIT_FRAMES
    swing_period: int = DEFAULT_SWING_PERIOD
    swing_hold: int = DEFAULT_SWING_HOLD
    farm_y_lo: int = DEFAULT_FARM_Y_LO
    farm_y_hi: int = DEFAULT_FARM_Y_HI
    farm_x_lo: int = DEFAULT_FARM_X_LO
    farm_x_hi: int = DEFAULT_FARM_X_HI
    leave_max_frames: int = DEFAULT_LEAVE_MAX_FRAMES

    phase: RupeeFarmPhase = RupeeFarmPhase.FARM
    frames: int = 0
    empty_frames: int = 0
    leaving: bool = False
    leave_frames: int = 0
    leave_pending: bool = False
    leave_stalls: int = 0
    restocks: int = 0
    saw_prey: bool = False
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    restock_inland: bool = True
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_rupees: int = -1
    _glance: dict[str, int] = field(default_factory=dict)
    _occ: FarmOccupancy = field(default_factory=FarmOccupancy, repr=False)

    def __post_init__(self) -> None:
        if self.leftover_screen is None:
            self.leftover_screen = self.farm_screen

    def reset(self) -> None:
        self.phase = RupeeFarmPhase.FARM
        self.frames = 0
        self.empty_frames = 0
        self.leaving = False
        self.leave_frames = 0
        self.leave_pending = False
        self.leave_stalls = 0
        self.restocks = 0
        self.saw_prey = False
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.restock_inland = True
        self.success = False
        self.notes.clear()
        self.start_rupees = -1
        self._glance = {}
        self._occ.reset()

    def already_satisfied(self, snap: ZeldaSnapshot) -> bool:
        return snap.rupees >= self.target_rupees and snap.screen == self.leftover_screen

    # ------------------------------------------------------------------ #
    # Terminal helpers — never write RAM, only read the glance fields.
    # ------------------------------------------------------------------ #

    def _glance_of(self, snap: ZeldaSnapshot) -> dict[str, int]:
        return {
            "screen": int(snap.screen),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "rupees": int(snap.rupees),
            "mode": int(snap.mode),
        }

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.phase = RupeeFarmPhase.FAILED
        self.success = False
        self._glance = self._glance_of(snap)
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def _finish(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.phase = RupeeFarmPhase.DONE
        self._glance = self._glance_of(snap)
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def _screen_dead(self, snap: ZeldaSnapshot) -> FrameAction:
        """Give up: a restock cycle returned an empty screen.

        Overworld waves are one-shot (AGENTS.md, measured 2026-09-12): once
        0x4A's tektites are dead the screen stays empty through a depth-1
        (0x49) *and* a depth-2 (0x49-0x59-0x49) round trip — and every
        ``locations._RESTOCK`` pair this farm is built on (0x4A<->0x49,
        0x68<->0x78) is a one-screen hop, so the depth-1 restock is a no-op on
        that corridor. The restock is therefore a give-up detector, not a rupee
        supply: one empty cycle after a restock is the answer, not a reason to
        spend the remaining ~35,000 frames on it.
        """
        if snap.rupees >= self.target_rupees:
            return self._return_step(snap)
        return self._fail(
            snap,
            f"farm_screen_dead_{snap.rupees}_of_{self.target_rupees}",
        )

    # ------------------------------------------------------------------ #
    # Screen topology (farm_screen <-> restock_neighbor_screen only)
    # ------------------------------------------------------------------ #

    def _transition_direction(self, snap: ZeldaSnapshot) -> str:
        """Direction to hold through a scroll, inferred from screen topology."""
        if snap.next_screen == self.farm_screen:
            return _OPPOSITE[self.restock_direction]
        if snap.next_screen == self.restock_neighbor_screen:
            return self.restock_direction
        if snap.screen == self.farm_screen:
            return self.restock_direction
        return _OPPOSITE[self.restock_direction]

    def _return_direction_from(self, screen: int) -> str | None:
        """Direction from ``screen`` toward ``leftover_screen``, or None."""
        if screen == self.leftover_screen:
            return None
        if screen == self.farm_screen and self.leftover_screen == self.restock_neighbor_screen:
            return self.restock_direction
        if screen == self.restock_neighbor_screen and self.leftover_screen == self.farm_screen:
            return _OPPOSITE[self.restock_direction]
        return None

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.start_rupees < 0:
            self.start_rupees = int(snap.rupees)

        if (
            abs(snap.link_x - self.last_x) <= 2
            and abs(snap.link_y - self.last_y) <= 2
            and snap.screen == self.last_screen
            and not snap.transitioning
        ):
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_x = snap.link_x
        self.last_y = snap.link_y
        self.last_screen = snap.screen

        if snap.mode == DEATH_MODE:
            return self._fail(snap, "link_death")
        if snap.level != 0:
            return self._fail(snap, f"farm_left_level_{snap.level}")
        if self.frames >= self.max_frames:
            return self._fail(snap, f"farm_timeout_rupees_{snap.rupees}")

        done = snap.rupees >= self.target_rupees

        if snap.transitioning:
            self.leaving = False
            self.leave_frames = 0
            self.empty_frames = 0
            self.stuck = 0
            self.restock_inland = False
            direction = self._transition_direction(snap)
            reason = "farm_return_scroll" if done else "farm_scroll"
            return FrameAction(nes_action(direction), reason)

        if snap.mode not in (PLAY_MODE, 8):
            self.empty_frames = 0
            self.stuck = 0
            return FrameAction(nes_idle_action(), f"farm_mode_{snap.mode}")

        if done:
            self.leaving = False
            self.leave_frames = 0
            if self.phase is not RupeeFarmPhase.RETURN:
                self.phase = RupeeFarmPhase.RETURN
            return self._return_step(snap)

        return self._farm_step(snap)

    def _return_step(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == self.leftover_screen:
            return self._finish(snap, f"farm_ok_{snap.rupees}")
        direction = self._return_direction_from(snap.screen)
        if direction is None:
            return self._fail(snap, f"farm_return_unreachable_{snap.screen:02x}")
        return self._edge_walk(snap, direction, "farm_return")

    def _leave_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return self._edge_walk(snap, self.restock_direction, "farm_leave")

    def _edge_walk(
        self, snap: ZeldaSnapshot, direction: str, reason: str
    ) -> FrameAction:
        """Walk the open lane to the screen edge, then push it.

        A bare directional hold has no answer to a bush or a rock between Link
        and the edge — the only exit left is the 36,000-frame ``max_frames``.
        ``heart_farm.LEAVE_GOALS`` aims at the y=141 corridor every overworld
        screen the spine crosses has (that is how the hop crossed it) and the
        occupancy walk learns what is in the way by bumping it, then routes
        around. At the edge cell, hold the direction: the scroll needs the
        button, not a path.
        """
        goal = leave_goal(direction)
        if goal is None:
            return FrameAction(nes_action(direction), reason)
        gx, gy = goal
        at_edge = (
            (direction == "LEFT" and snap.link_x <= gx)
            or (direction == "RIGHT" and snap.link_x >= gx)
            or (direction == "UP" and snap.link_y <= gy)
            or (direction == "DOWN" and snap.link_y >= gy)
        )
        if at_edge:
            return FrameAction(nes_action(direction), f"{reason}_push")
        return self._occ.walk(snap, self.frames, goal, reason, slash=False)

    def _farm_step(self, snap: ZeldaSnapshot) -> FrameAction:
        # Overworld enemies do not respawn while we stay put; toggle out to
        # restock_neighbor_screen and back to force fresh spawns.
        if snap.screen == self.restock_neighbor_screen:
            if self.leave_pending:
                self.leave_pending = False
                self.restocks += 1
                self.notes.append(f"farm_restock_{self.restocks}")
            self.leaving = False
            self.leave_frames = 0
            self.empty_frames = 0
            if not self.restock_inland:
                if self.restock_direction == "LEFT":
                    if snap.link_x > 200:
                        return walk_or_swing(
                            self.frames,
                            "LEFT",
                            "farm_inland",
                            snap,
                            period=self.swing_period,
                            hold=self.swing_hold,
                        )
                elif self.restock_direction == "RIGHT":
                    if snap.link_x < 56:
                        return walk_or_swing(
                            self.frames,
                            "RIGHT",
                            "farm_inland",
                            snap,
                            period=self.swing_period,
                            hold=self.swing_hold,
                        )
                elif self.restock_direction == "UP":
                    if snap.link_y > 180:
                        return walk_or_swing(
                            self.frames,
                            "UP",
                            "farm_inland",
                            snap,
                            period=self.swing_period,
                            hold=self.swing_hold,
                        )
                elif self.restock_direction == "DOWN":
                    if snap.link_y < 90:
                        return walk_or_swing(
                            self.frames,
                            "DOWN",
                            "farm_inland",
                            snap,
                            period=self.swing_period,
                            hold=self.swing_hold,
                        )
                self.restock_inland = True
            direction = _OPPOSITE[self.restock_direction]
            return walk_or_swing(
                self.frames,
                direction,
                "farm_respawn",
                snap,
                period=self.swing_period,
                hold=self.swing_hold,
            )
        if snap.screen != self.farm_screen:
            return self._fail(snap, f"farm_left_{snap.screen:02x}")

        if not self.restock_inland:
            respawn_dir = _OPPOSITE[self.restock_direction]
            if respawn_dir == "RIGHT" and snap.link_x < 48:
                return FrameAction(nes_action("RIGHT"), "farm_inland")
            elif respawn_dir == "LEFT" and snap.link_x > 208:
                return FrameAction(nes_action("LEFT"), "farm_inland")
            elif respawn_dir == "DOWN" and snap.link_y < 90:
                return FrameAction(nes_action("DOWN"), "farm_inland")
            elif respawn_dir == "UP" and snap.link_y > 180:
                return FrameAction(nes_action("UP"), "farm_inland")
            self.restock_inland = True

        drops = [
            obj
            for obj in snap.objects
            if obj.slot >= 1
            and obj.type_id == RUPEE_DROP_TYPE_ID
            and int(obj.state) in _RUPEE_STATES
        ]
        if drops and self.leaving:
            self.leaving = False
            self.leave_frames = 0
            self.empty_frames = 0

        if self.leaving:
            self.leave_frames += 1
            if self.leave_frames > self.leave_max_frames:
                self.leaving = False
                self.leave_frames = 0
                self.leave_pending = False
                self.empty_frames = 0
                self.leave_stalls += 1
                self.notes.append(
                    f"farm_leave_stalled_{snap.link_x}_{snap.link_y}"
                )
                if self.leave_stalls >= MAX_LEAVE_STALLS:
                    return self._fail(
                        snap, f"farm_leave_blocked_{snap.link_x}_{snap.link_y}"
                    )
            else:
                return self._leave_action(snap)

        if snap.link_y < self.farm_y_lo:
            return FrameAction(nes_action("DOWN"), "farm_south")

        prey = drops or [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 10
            and obj.type_id not in (0, 0xFF, 0x64, RUPEE_DROP_TYPE_ID)
            and 0 < obj.hp < 200
            and self.farm_y_lo < obj.y < self.farm_y_hi
            and self.farm_x_lo < obj.x < self.farm_x_hi
        ]
        if not prey:
            self.empty_frames += 1
            if self.empty_frames < self.empty_wait_frames:
                return FrameAction(nes_idle_action(), "farm_wait")
            self.empty_frames = 0
            # One restock already came back with nothing to farm: the wave is
            # one-shot and this corridor is dead. Bail with a reason instead of
            # cycling wait -> leave -> respawn for the rest of ``max_frames``.
            if self.restocks >= 1 and not self.saw_prey:
                return self._screen_dead(snap)
            self.saw_prey = False
            self.leaving = True
            self.leave_pending = True
            self.leave_frames = 1
            self.stuck = 0
            return self._leave_action(snap)

        self.empty_frames = 0
        self.saw_prey = True
        self.leaving = False
        self.leave_frames = 0
        nearest = min(
            prey,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y

        primary_x = abs(dx) >= abs(dy)
        if self.stuck > 15:
            primary_x = not primary_x

        if primary_x:
            if abs(dx) > 4:
                direction = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 4:
                direction = "DOWN" if dy > 0 else "UP"
            else:
                direction = "RIGHT" if dx >= 0 else "LEFT"
        else:
            if abs(dy) > 4:
                direction = "DOWN" if dy > 0 else "UP"
            elif abs(dx) > 4:
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy >= 0 else "UP"

        if self.stuck > 30:
            if snap.link_y < 141:
                direction = "DOWN"
            elif snap.link_y > 148:
                direction = "UP"
            else:
                direction = "RIGHT" if dx >= 0 else "LEFT"

        if direction == "UP" and snap.link_y <= self.farm_y_lo:
            direction = "DOWN"
        reason = "farm_rupee" if drops else "farm_chase"
        if drops:
            return FrameAction(nes_action(direction), "farm_rupee")
        return walk_or_swing(
            self.frames,
            direction,
            reason,
            snap,
            period=self.swing_period,
            hold=self.swing_hold,
        )

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "target_rupees": self.target_rupees,
            "farm_screen": self.farm_screen,
            "restock_neighbor_screen": self.restock_neighbor_screen,
            "restock_direction": self.restock_direction,
            "leftover_screen": self.leftover_screen,
            "start_rupees": self.start_rupees,
            "restocks": self.restocks,
            "saw_prey": self.saw_prey,
            "leave_stalls": self.leave_stalls,
            "occupancy_misses": self._occ.misses,
            "notes": list(self.notes),
            "glance": dict(self._glance),
        }
