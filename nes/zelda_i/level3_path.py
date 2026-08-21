"""Level 3 multi-room path controllers (door micros, west key, north chain).

Room specs and stop predicates remain in ``level3_dungeon``. Raft path lives in
``level3_raft_path`` (import there, or via ``level3_dungeon`` shim).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
)
from zelda_i.level3_geometry import (
    NORTH_DOOR_X,
    NORTH_DOOR_X_TOL,
    ROOM_6B_BAND_Y,
    ROOM_6B_DOOR_Y,
    WEST_DOOR_APPROACH_Y,
    WEST_DOOR_WALL_X,
)
from zelda_i.level3_occupancy import room_6b_grid
from zelda_i.level3_dungeon import (
    ROOM_6B_SPEC,
    ROOM_7B_SPEC,
    ROOM_L3_DARKNUTS,
    ROOM_L3_NORTH_ZOLS,
    ROOM_L3_WEST_KEY,
)
from zelda_i.level3_overworld import LEVEL3, SCREEN_LEVEL3_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

# Path timing knobs (not room-table data).
WEST_ENTER_MAX_FRAMES = 1200
NORTH_ENTER_MAX_FRAMES = 1500
NORTH_EXIT_6B_MAX_FRAMES = 6000

# Re-export geometry for callers that imported door bands from this module.
# Re-export for boss path / scripts that imported ROOM_L3_ENTRY from here.
ROOM_L3_ENTRY = SCREEN_LEVEL3_ENTRY_ROOM

def west_door_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of 0x7c → 0x7b west-door policy (diagonal residual)."""
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3")
    if snap.transitioning:
        return FrameAction(nes_action("LEFT", "UP"), "west_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_L3_WEST_KEY:
        return FrameAction(nes_idle_action(), "west_arrived")
    if snap.screen != ROOM_L3_ENTRY:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    # South mouth → room body
    if snap.link_y > 165:
        return FrameAction(nes_action("UP"), "west_leave_mouth")
    # Horizontal approach on y≈149 band (reaches x≈32 wall)
    if snap.link_x > WEST_DOOR_WALL_X:
        if abs(snap.link_y - WEST_DOOR_APPROACH_Y) > 3:
            direction = "UP" if snap.link_y > WEST_DOOR_APPROACH_Y else "DOWN"
            return FrameAction(nes_action(direction), "west_align_y")
        return FrameAction(nes_action("LEFT"), "west_approach")
    # Door plane: LEFT alone sticks; LEFT+UP corner-clips into 0x7b
    return FrameAction(nes_action("LEFT", "UP"), "west_diagonal_push")


def north_door_7b_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of 0x7b → 0x6b north-door policy (strict x≈120)."""
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_L3_NORTH_ZOLS:
        return FrameAction(nes_idle_action(), "north_arrived_6b")
    if snap.screen != ROOM_L3_WEST_KEY:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "north_align_x")
    return FrameAction(nes_action("UP"), "north_push")


@dataclass
class Level3WestDoorController:
    """Route 0x7c → 0x7b only (no combat). Success when room-ready on 0x7b."""

    max_frames: int = WEST_ENTER_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        action = west_door_step(snap)
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_WEST_KEY
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x7b")
            return FrameAction(nes_idle_action(), "west_arrived")
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "policy": "LEFT+UP diagonal at west wall; approach y≈149",
        }


@dataclass
class Level3NorthDoor7bController:
    """Route 0x7b → 0x6b only (no combat). Strict x≈120 UP residual."""

    max_frames: int = NORTH_ENTER_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        action = north_door_7b_step(snap)
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_NORTH_ZOLS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x6b")
            return FrameAction(nes_idle_action(), "north_arrived_6b")
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "policy": "UP @ x≈120 (|dx|≤4); wider align sticks at x≈112",
        }


_ROOM_6B_DOOR = (NORTH_DOOR_X, ROOM_6B_DOOR_Y)
_ROOM_6B_BAND = (NORTH_DOOR_X, ROOM_6B_BAND_Y)


def _room_6b_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=room_6b_grid())


@dataclass
class Level3NorthExit6bController:
    """Route 0x6b → 0x5b after Zols cleared (occupancy BFS + door push).

    Predicted 1px walks; a miss blocks the cell ahead and replans. No path
    stands. North band UP @ x≈120 is the verified door residual (not a 1px walk).
    """

    max_frames: int = NORTH_EXIT_6B_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    walker: OccupancyWalker = field(default_factory=_room_6b_walker)

    @property
    def grid(self) -> OccupancyGrid:
        return self.walker.grid

    @property
    def misses(self) -> int:
        return self.walker.misses

    def _goal(self) -> tuple[int, int]:
        if self.grid.passable(*_ROOM_6B_DOOR):
            return _ROOM_6B_DOOR
        return _ROOM_6B_BAND

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_DARKNUTS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x5b")
            return FrameAction(nes_idle_action(), "north_arrived_5b")

        if snap.transitioning:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north6b_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L3_NORTH_ZOLS:
            self.walker.last_dir = None
            return FrameAction(
                nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
            )

        xy = (snap.link_x, snap.link_y)
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}")

        # Traversable north band starts at y=109 (y<=100 stranded on the diamond).
        if snap.link_y <= ROOM_6B_BAND_Y:
            self.walker.last_dir = None
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north6b_align_door")
            return FrameAction(nes_action("UP"), "north6b_push")

        # South mouth / door plane. Live dest combat ends at (120,181):
        # UP does not move (v3); LEFT+UP slides west to (48,181) (v4). Walk
        # LEFT off the door column, then UP inland. Do not occupancy-grade.
        if snap.link_y >= 173:
            self.walker.last_dir = None
            if abs(snap.link_x - NORTH_DOOR_X) <= 16:
                return FrameAction(nes_action("LEFT"), "north6b_leave_mouth_x")
            return FrameAction(nes_action("UP"), "north6b_leave_mouth")

        direction = self.walker.next_dir(xy, self._goal())
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"wait_f{self.frames}_{xy}")
            return FrameAction(nes_idle_action(), "north6b_wait")
        return FrameAction(nes_action(direction), "north6b_path")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "misses": self.misses,
            "blocked": len(self.grid.blocked),
            "notes": list(self.notes),
            "policy": "occupancy BFS + UP @ x≈120 on north band",
        }


@dataclass
class Level3WestKeyController:
    """Full isolated pure: Level3Entrance 0x7c → 0x7b clear + key.

    Phase 1: ``Level3WestDoorController`` (diagonal residual).
    Phase 2: ``GenericDungeonRoomController(ROOM_7B_SPEC)`` combat/reward.
    """

    door: Level3WestDoorController = field(default_factory=Level3WestDoorController)
    combat: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_7B_SPEC)
    )
    frames: int = 0
    success: bool = False
    phase: str = "door"
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")

        if self.phase == "door":
            action = self.door.step(snap)
            if self.door.success:
                self.phase = "combat"
                self.notes.append("door_ok")
                # Hand off; combat controller sees room_id and enters FIGHT.
                return self.combat.step(snap)
            if self.door.failed:
                self.phase = "failed"
                self.notes.append("door_failed")
            return action

        if self.phase == "combat":
            action = self.combat.step(snap)
            if self.combat.success:
                self.success = True
                self.phase = "done"
                self.notes.append("key_ok")
            elif self.combat.phase is DungeonPhase.FAILED:
                self.phase = "failed"
                self.notes.append("combat_failed")
            return action

        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "door": self.door.report(),
            "combat": self.combat.report(),
            "spec_id": ROOM_7B_SPEC.spec_id,
            "intervention_class": "clean",
            "track": "clean",
        }


@dataclass
class Level3NorthChainController:
    """Isolated pure from ``Level3WestKey``: 0x7b → 0x6b clear → 0x5b.

    Phase 1: ``Level3NorthDoor7bController`` (UP @ x≈120).
    Phase 2: ``GenericDungeonRoomController(ROOM_6B_SPEC)`` Zol clear.
    Phase 3: ``Level3NorthExit6bController`` north to Darknut room.
    Stop: ``level3_reached_5b``.
    """

    door: Level3NorthDoor7bController = field(
        default_factory=Level3NorthDoor7bController
    )
    combat: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_6B_SPEC)
    )
    north_exit: Level3NorthExit6bController = field(
        default_factory=Level3NorthExit6bController
    )
    frames: int = 0
    success: bool = False
    phase: str = "door"
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")

        # Early success if already in 0x5b (reload / resume).
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_DARKNUTS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.phase = "done"
            self.notes.append("already_0x5b")
            return FrameAction(nes_idle_action(), "done")

        if self.phase == "door":
            action = self.door.step(snap)
            if self.door.success:
                self.phase = "combat"
                self.notes.append("door_6b_ok")
                return self.combat.step(snap)
            if self.door.failed:
                self.phase = "failed"
                self.notes.append("door_6b_failed")
            return action

        if self.phase == "combat":
            action = self.combat.step(snap)
            if self.combat.success:
                self.phase = "north_exit"
                self.notes.append("zols_cleared")
                return self.north_exit.step(snap)
            if self.combat.phase is DungeonPhase.FAILED:
                self.phase = "failed"
                self.notes.append("combat_failed")
            return action

        if self.phase == "north_exit":
            action = self.north_exit.step(snap)
            if self.north_exit.success:
                self.success = True
                self.phase = "done"
                self.notes.append("reached_0x5b")
            elif self.north_exit.failed:
                self.phase = "failed"
                self.notes.append("north_exit_failed")
            return action

        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "door": self.door.report(),
            "combat": self.combat.report(),
            "north_exit": self.north_exit.report(),
            "spec_id": ROOM_6B_SPEC.spec_id,
            "stop": "level3_reached_5b",
            "intervention_class": "clean",
            "track": "clean",
        }




# Raft path: import from ``level3_raft_path`` (canonical) or
# ``level3_dungeon`` (compatibility shim). No re-export here (rr-iji5).
