"""Level 7 shard rooms: 0x2A east shutter -> 0x2B Triforce -> settled OW leave.

Live 2/2 on the walk-on lineage (``20260904_W3``/``W4``, rr-8t4.3): from the
Aquamentus room the east shutter opens on the kill, ``0x2B`` spawns Link on
the west mouth ``(16,141)``, and the shard sits behind a diamond floor -- DOWN
at ``x=16`` does not move, so the walk is south-around
``(32,141) -> (32,189) -> (120,189) -> (128,141)``.  The fanfare is then idled
out (never walked, same shape as ``Level6ExitController``) and the game returns
Link to the overworld by itself.

This factory's leftover is **not** the Survival packet: the lineage pin
starts at TF 0, so a fixture run reads TF ``0x40`` on OW ``0x42``.
``MEASURED_POST_L7_EXIT`` is filled from power-on ``--through level7``
2/2; ``report()["measured_post_l7_exit_verified"]`` stays False because
this controller does not claim that leftover.

Success is the **rising edge** of the L7 Triforce bit plus a settled overworld
frame -- a level check alone would pass on any pin that already has the bit.
No RAM writes: position, progression and capacity all stay 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import TF_BIT_L7
from zelda_i.level7.stairs import AQUAMENTUS_ROM, TRIFORCE_ROM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "FANFARE_MODE",
    "LEVEL7",
    "SHARD_LEAVE_MAX_FRAMES",
    "SHARD_WAYPOINTS",
    "SHUTTER_MAX_FRAMES",
    "Level7ShardLeaveController",
    "ShardLeavePhase",
    "make_level7_shard_leave_controller",
]

LEVEL7 = 7
DEATH_MODE = 17
FANFARE_MODE = 18
SHUTTER_MAX_FRAMES = 1200
SHARD_LEAVE_MAX_FRAMES = SHUTTER_MAX_FRAMES + 3600
# Diamond floor: the shard is reached from the south, never straight east.
SHARD_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (32, 141),
    (32, 189),
    (120, 189),
    (128, 141),
)
_ARRIVE_TOL = 3


class ShardLeavePhase(Enum):
    EAST_SHUTTER = auto()
    SHARD_WALK = auto()
    FANFARE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level7ShardLeaveController:
    """0x2A -> 0x2B shard -> idle the fanfare -> settled overworld frame."""

    phase: ShardLeavePhase = ShardLeavePhase.EAST_SHUTTER
    max_frames: int = SHARD_LEAVE_MAX_FRAMES
    frames: int = 0
    shutter_frames: int = 0
    waypoint_index: int = 0
    success: bool = False
    failed: bool = False
    initial_triforce: int | None = None
    shard_taken: bool = False
    leftover: dict[str, int] | None = None
    notes: list[str] = field(default_factory=list)

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _fail(self, note: str, reason: str) -> FrameAction:
        self.failed = True
        self.phase = ShardLeavePhase.FAILED
        self._note(note)
        return FrameAction(nes_idle_action(), reason)

    def _shard_walk(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.waypoint_index >= len(SHARD_WAYPOINTS):
            return FrameAction(nes_idle_action(), "wait_shard_pickup")
        tx, ty = SHARD_WAYPOINTS[self.waypoint_index]
        dx, dy = tx - int(snap.link_x), ty - int(snap.link_y)
        if abs(dx) <= _ARRIVE_TOL and abs(dy) <= _ARRIVE_TOL:
            self.waypoint_index += 1
            self._note(f"shard_wp_{self.waypoint_index}")
            return FrameAction(nes_idle_action(), "shard_wp")
        if abs(dy) > _ARRIVE_TOL:
            return FrameAction(
                nes_action("DOWN" if dy > 0 else "UP"), "shard_walk_y"
            )
        return FrameAction(
            nes_action("RIGHT" if dx > 0 else "LEFT"), "shard_walk_x"
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.initial_triforce is None:
            self.initial_triforce = int(snap.triforce)
            self._note(f"triforce_in_0x{self.initial_triforce:02x}")
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == DEATH_MODE:
            return self._fail("death", "death")
        if self.frames > self.max_frames:
            return self._fail("budget_exhausted", "budget_exhausted")

        took_shard = bool(int(snap.triforce) & TF_BIT_L7) and not (
            self.initial_triforce & TF_BIT_L7
        )
        if took_shard and not self.shard_taken:
            self.shard_taken = True
            self.phase = ShardLeavePhase.FANFARE
            self._note("triforce_bit_set")

        if snap.level == 0 and snap.mode == PLAY_MODE and not snap.transitioning:
            if not self.shard_taken:
                return self._fail("overworld_without_shard", "left_without_shard")
            self.leftover = {
                "screen": int(snap.screen),
                "link_x": int(snap.link_x),
                "link_y": int(snap.link_y),
                "mode": int(snap.mode),
                "triforce": int(snap.triforce),
            }
            self.success = True
            self.phase = ShardLeavePhase.DONE
            self._note(f"ow_leave_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), "ow_leave")

        if snap.mode == FANFARE_MODE or self.phase is ShardLeavePhase.FANFARE:
            return FrameAction(nes_idle_action(), "idle_fanfare")
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.level != LEVEL7:
            return self._fail(f"left_L7_to_L{snap.level}", "left_level7")

        if self.phase is ShardLeavePhase.EAST_SHUTTER:
            if int(snap.screen) == TRIFORCE_ROM:
                self.phase = ShardLeavePhase.SHARD_WALK
                self._note("arrived_0x2b")
                return FrameAction(nes_idle_action(), "arrived_0x2b")
            if int(snap.screen) != AQUAMENTUS_ROM:
                return self._fail(
                    f"unexpected_room_0x{snap.screen:02x}", "unexpected_room"
                )
            self.shutter_frames += 1
            if self.shutter_frames > SHUTTER_MAX_FRAMES:
                return self._fail("east_shutter_stalled", "east_shutter_stalled")
            return FrameAction(nes_action("RIGHT"), "push_east_shutter")

        if int(snap.screen) != TRIFORCE_ROM:
            return self._fail(
                f"left_0x2b_to_0x{snap.screen:02x}", "left_shard_room"
            )
        return self._shard_walk(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_shard_and_settled_leave",
            "phase": self.phase.name,
            "shard_taken": self.shard_taken,
            "initial_triforce": self.initial_triforce,
            "waypoint_index": self.waypoint_index,
            "leftover": dict(self.leftover) if self.leftover else None,
            "measured_post_l7_exit_verified": False,
            "route_eligible": False,
            "notes": list(self.notes),
        }


def make_level7_shard_leave_controller() -> Level7ShardLeaveController:
    """Fresh 0x2A -> 0x2B -> OW controller (never share instances)."""
    return Level7ShardLeaveController()
