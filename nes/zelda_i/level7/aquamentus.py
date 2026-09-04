"""Level 7 boss room 0x2A: Aquamentus kill plus the heart-container pickup.

The fight reuses ``level1.finish.Level1AquamentusController`` (one combat
engine, aliased onto the live ``$EB=0x2A``).  Only the pickup is L7 code: the
L1 engine walks a single fixed cell ``(192, 141)`` and then idles, which in
0x2A collected only because the C1 walk happened to cross the real item.  On
the walk-on lineage the boss dies at ``(190, 128)`` and Link idled at
``(190, 141)`` for 5.5k frames with ``heart_containers`` still 3 (W1/W2).

Probe ``scratch/probe_l7_2a_heart.py`` (``20260904_H1``) dumped all 13 object
slots after the kill: the container is **not** an object slot -- ``$00AB``
holds room item ``0x1A`` and the pickup happened at **``(136, 141)``**, the
room centre, not at ``(192, 141)``.  So this controller walks the centre row
first and then serpentines the floor band, skipping a waypoint it cannot
reach, until the container count rises.

Success is the **rising edge** of ``heart_containers`` measured from this
controller's own first frame -- never a level check, which a fixture pin that
already carries 4 containers would satisfy without picking anything up.
No RAM writes: position, progression and capacity all stay 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level1.finish import (
    AQUAMENTUS_MAX_FRAMES,
    AquamentusPhase,
    Level1AquamentusController,
    ROOM_AQUAMENTUS,
)
from zelda_i.level7.stairs import AQUAMENTUS_ROM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "HEART_CELL",
    "HEART_STALL_FRAMES",
    "HEART_SWEEP_MAX_FRAMES",
    "HEART_SWEEP_WAYPOINTS",
    "LEVEL7",
    "Level7AquamentusHeartController",
    "make_level7_aquamentus_heart_controller",
]

LEVEL7 = 7
DEATH_MODE = 17
_ARRIVE_TOL = 3
# Boss floor band of 0x2A (west mouth (32,141), east shutter (208,141)).
# The centre row comes first: 20260904_H1 collected at (136,141).  The rest
# is a serpentine fallback in case a later lineage drops it elsewhere.
HEART_CELL = (136, 141)
HEART_SWEEP_WAYPOINTS: tuple[tuple[int, int], ...] = (
    HEART_CELL,
    (64, 141),
    (200, 141),
    (200, 125),
    (64, 125),
    (64, 157),
    (200, 157),
    (200, 173),
    (64, 173),
    (64, 109),
    (200, 109),
    HEART_CELL,
)
HEART_SWEEP_MAX_FRAMES = 2400
# Frames of no movement before a waypoint is written off as unreachable.
HEART_STALL_FRAMES = 40


@dataclass
class Level7AquamentusHeartController:
    """Kill Aquamentus in live ``0x2A`` and collect one heart container."""

    live_room: int = AQUAMENTUS_ROM
    max_frames: int = AQUAMENTUS_MAX_FRAMES + HEART_SWEEP_MAX_FRAMES
    frames: int = 0
    sweep_frames: int = 0
    waypoint_index: int = 0
    success: bool = False
    failed: bool = False
    initial_containers: int | None = None
    boss_defeated: bool = False
    stalled: int = 0
    last_xy: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)
    fight: Level1AquamentusController = field(
        default_factory=lambda: Level1AquamentusController(
            phase=AquamentusPhase.ALIGN, tank_hits=True
        )
    )

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _alias(self, snap: ZeldaSnapshot) -> ZeldaSnapshot:
        if snap.screen == self.live_room:
            return replace(snap, screen=ROOM_AQUAMENTUS)
        return snap

    def _heart_taken(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.initial_containers is not None
            and snap.heart_containers >= self.initial_containers + 1
        )

    def _sweep(self, snap: ZeldaSnapshot) -> FrameAction:
        self.sweep_frames += 1
        if self.sweep_frames > HEART_SWEEP_MAX_FRAMES:
            self.failed = True
            self._note(f"heart_sweep_exhausted_{snap.link_x}_{snap.link_y}")
            return FrameAction(nes_idle_action(), "heart_sweep_exhausted")
        if self.waypoint_index >= len(HEART_SWEEP_WAYPOINTS):
            self.waypoint_index = 0
        xy = (int(snap.link_x), int(snap.link_y))
        self.stalled = self.stalled + 1 if xy == self.last_xy else 0
        self.last_xy = xy
        tx, ty = HEART_SWEEP_WAYPOINTS[self.waypoint_index]
        dx, dy = tx - xy[0], ty - xy[1]
        if self.stalled > HEART_STALL_FRAMES:
            self.waypoint_index += 1
            self.stalled = 0
            self._note(f"heart_wp_unreachable_{xy[0]}_{xy[1]}")
            return FrameAction(nes_idle_action(), "heart_wp_unreachable")
        if abs(dx) <= _ARRIVE_TOL and abs(dy) <= _ARRIVE_TOL:
            self.waypoint_index += 1
            self._note(f"heart_wp_{self.waypoint_index}")
            return FrameAction(nes_idle_action(), "heart_wp")
        if abs(dy) > _ARRIVE_TOL:
            return FrameAction(
                nes_action("DOWN" if dy > 0 else "UP"), "heart_sweep_y"
            )
        return FrameAction(
            nes_action("RIGHT" if dx > 0 else "LEFT"), "heart_sweep_x"
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.initial_containers is None:
            self.initial_containers = int(snap.heart_containers)
            self._note(f"containers_in_{self.initial_containers}")
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == DEATH_MODE:
            self.failed = True
            self._note("death")
            return FrameAction(nes_idle_action(), "death")
        if self.frames > self.max_frames:
            self.failed = True
            self._note("budget_exhausted")
            return FrameAction(nes_idle_action(), "budget_exhausted")
        if snap.level != LEVEL7 or int(snap.screen) != int(self.live_room):
            self.failed = True
            self._note(f"left_room_L{snap.level}_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), "left_boss_room")

        if self._heart_taken(snap):
            self.success = True
            self._note("heart_container_collected")
            return FrameAction(nes_idle_action(), "heart_container_collected")

        if not self.boss_defeated:
            action = self.fight.step(self._alias(snap))
            if self.fight.phase is AquamentusPhase.FAILED:
                # The engine only fails the fight; a timeout after the kill is
                # a pickup problem, so fall through to the sweep instead.
                if not self.fight.boss_seen:
                    self.failed = True
                    self._note("aquamentus_fight_red")
                    return FrameAction(nes_idle_action(), "aquamentus_red")
            if self.fight.phase in (
                AquamentusPhase.COLLECT_HEART,
                AquamentusPhase.DONE,
                AquamentusPhase.FAILED,
            ):
                self.boss_defeated = True
                self._note("aquamentus_defeated")
            elif snap.mode == PLAY_MODE:
                return action
            else:
                return action

        return self._sweep(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_aquamentus_heart",
            "live_room": f"0x{self.live_room:02X}",
            "aliased_l1_room": f"0x{ROOM_AQUAMENTUS:02X}",
            "boss_defeated": self.boss_defeated,
            "initial_containers": self.initial_containers,
            "sweep_frames": self.sweep_frames,
            "waypoint_index": self.waypoint_index,
            "fight": self.fight.report(),
            "route_eligible": False,
            "notes": list(self.notes),
        }


def make_level7_aquamentus_heart_controller(
    *, live_room: int = AQUAMENTUS_ROM
) -> Level7AquamentusHeartController:
    """Fresh 0x2A kill-and-collect controller (never share instances)."""
    return Level7AquamentusHeartController(live_room=live_room)
