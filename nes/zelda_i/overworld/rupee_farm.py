"""Generic leftover-safe rupee/occupancy farm (no ``$066D`` writes).

Extracted from ``OverworldToArrowShopController._farm_step`` (the 0x49<->0x4A
wooden-arrow farm in ``zelda_i.level1.arrow_shop``) so any shop-buy controller
can farm rupees to a target ``N`` without poking the rupee count. The farm:

- Patrols live prey objects on ``farm_screen`` (``type_id`` not in
  ``(0, 0xFF, 0x60)``, ``hp > 0``, in-bounds) plus rupee-drop sprites
  (``type_id == 0x60``), preferring drops when both are present.
- Leaves ``farm_screen`` for ``restock_neighbor_screen`` (and back) to force
  overworld enemy respawns once nothing is left to farm — the same
  0x49<->0x4A scroll-out/scroll-in restock, generalized to any adjacent
  screen pair and a configurable ``restock_direction``.
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
from zelda_i.ram import ZeldaSnapshot

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
DEFAULT_FARM_Y_LO = 120
DEFAULT_FARM_Y_HI = 210
DEFAULT_FARM_X_LO = 16
DEFAULT_FARM_X_HI = 240

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

    phase: RupeeFarmPhase = RupeeFarmPhase.FARM
    frames: int = 0
    empty_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_rupees: int = -1
    _glance: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.leftover_screen is None:
            self.leftover_screen = self.farm_screen

    def reset(self) -> None:
        self.phase = RupeeFarmPhase.FARM
        self.frames = 0
        self.empty_frames = 0
        self.success = False
        self.notes.clear()
        self.start_rupees = -1
        self._glance = {}

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

    # ------------------------------------------------------------------ #
    # Screen topology (farm_screen <-> restock_neighbor_screen only)
    # ------------------------------------------------------------------ #

    def _transition_direction(self, snap: ZeldaSnapshot) -> str:
        """Direction to hold through a scroll, inferred from screen topology."""
        returning_to_farm = (
            snap.screen == self.restock_neighbor_screen
            or snap.next_screen == self.farm_screen
        )
        if returning_to_farm:
            return _OPPOSITE[self.restock_direction]
        return self.restock_direction

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

        if snap.mode == DEATH_MODE:
            return self._fail(snap, "link_death")
        if snap.level != 0:
            return self._fail(snap, f"farm_left_level_{snap.level}")
        if self.frames >= self.max_frames:
            return self._fail(snap, f"farm_timeout_rupees_{snap.rupees}")

        done = snap.rupees >= self.target_rupees

        if snap.transitioning:
            direction = self._transition_direction(snap)
            reason = "farm_return_scroll" if done else "farm_scroll"
            return FrameAction(nes_action(direction), reason)

        if done:
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
        return FrameAction(nes_action(direction), "farm_return")

    def _farm_step(self, snap: ZeldaSnapshot) -> FrameAction:
        # Overworld enemies do not respawn while we stay put; toggle out to
        # restock_neighbor_screen and back to force fresh spawns.
        if snap.screen == self.restock_neighbor_screen:
            direction = _OPPOSITE[self.restock_direction]
            return FrameAction(nes_action(direction), "farm_respawn")
        if snap.screen != self.farm_screen:
            return self._fail(snap, f"farm_left_{snap.screen:02x}")

        if snap.link_y < self.farm_y_lo:
            return FrameAction(nes_action("DOWN"), "farm_south")

        drops = [
            obj
            for obj in snap.objects
            if obj.slot >= 1
            and obj.type_id == RUPEE_DROP_TYPE_ID
            and int(obj.state) in _RUPEE_STATES
        ]
        prey = drops or [
            obj
            for obj in snap.objects
            if obj.slot >= 1
            and obj.type_id not in (0, 0xFF, RUPEE_DROP_TYPE_ID)
            and obj.hp > 0
            and self.farm_y_lo < obj.y < self.farm_y_hi
            and self.farm_x_lo < obj.x < self.farm_x_hi
        ]
        if not prey:
            self.empty_frames += 1
            if self.empty_frames < self.empty_wait_frames:
                direction = "RIGHT" if snap.link_x < 160 else "LEFT"
                return FrameAction(nes_action(direction), "farm_wait")
            self.empty_frames = 0
            return FrameAction(nes_action(self.restock_direction), "farm_leave")
        self.empty_frames = 0
        nearest = min(
            prey,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) >= abs(dy) and abs(dx) > 4:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 4:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "RIGHT" if dx >= 0 else "LEFT"
        if direction == "UP" and snap.link_y < self.farm_y_lo + 16:
            direction = "DOWN"
        reason = "farm_rupee" if drops else "farm_chase"
        # Drops are not threats; walk_or_swing will not pulse A at empty air.
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
            "notes": list(self.notes),
            "glance": dict(self._glance),
        }
