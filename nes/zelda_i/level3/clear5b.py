"""Level 3 dest_6b: 0x7b → 0x6b clear → 0x5b south-mouth Darknut clear.

``Level3NorthChainController`` lives here so ``level3_path`` does not grow.
Re-exported from ``level3_path`` for existing imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.engine import (
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
)
from zelda_i.level3.dungeon import (
    LEVEL3,
    ROOM_5B_SPEC,
    ROOM_6B_SPEC,
    ROOM_L3_DARKNUTS,
)
from zelda_i.level3.raft_path import SPAWN_SETTLE_FRAMES
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

if TYPE_CHECKING:
    from zelda_i.level3.path import (
        Level3NorthDoor7bController,
        Level3NorthExit6bController,
    )


def _north_door():
    from zelda_i.level3.path import Level3NorthDoor7bController

    return Level3NorthDoor7bController()


def _north_exit():
    from zelda_i.level3.path import Level3NorthExit6bController

    return Level3NorthExit6bController()


class Level3Clear5bController(GenericDungeonRoomController):
    """0x5b Darknut clear. Engine clips dest to occupancy_bounds (ymin=109)."""

    def __init__(self, spec: DungeonRoomSpec | None = None) -> None:
        super().__init__(spec if spec is not None else ROOM_5B_SPEC)


@dataclass
class Level3NorthChainController:
    """From ``Level3WestKey``: 0x7b → 0x6b clear → 0x5b Darknut clear.

    Phase 1: ``Level3NorthDoor7bController`` (UP @ x≈120).
    Phase 2: ``GenericDungeonRoomController(ROOM_6B_SPEC)`` Zol clear.
    Phase 3: ``Level3NorthExit6bController`` north to Darknut room.
    Phase 4: ``Level3Clear5bController`` occupancy Darknut clear.
    Stop: play 0x5b with no live Darknuts.
    """

    door: Level3NorthDoor7bController = field(default_factory=_north_door)
    combat: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_6B_SPEC)
    )
    north_exit: Level3NorthExit6bController = field(default_factory=_north_exit)
    clear_5b: Level3Clear5bController = field(
        default_factory=Level3Clear5bController
    )
    # Spine dest_6b clears 0x5b. Isolated Clean north_chain still stops on enter.
    clear_darknuts: bool = True
    frames: int = 0
    spawn_frames: int = 0
    success: bool = False
    phase: str = "door"
    notes: list[str] = field(default_factory=list)

    def _in_5b_playable(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_DARKNUTS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def _5b_cleared(self, snap: ZeldaSnapshot) -> bool:
        return self._in_5b_playable(snap) and not ROOM_5B_SPEC.live_enemies(snap)

    def _enter_clear_5b(self, snap: ZeldaSnapshot) -> FrameAction:
        self.phase = "clear_5b"
        self.notes.append("entered_0x5b_clear")
        return self.clear_5b.step(snap)

    def _enter_spawn_5b(self, snap: ZeldaSnapshot) -> FrameAction:
        self.phase = "spawn_5b"
        self.spawn_frames = 0
        self.notes.append("entered_0x5b_spawn")
        return FrameAction(nes_idle_action(), "spawn_5b_wait")

    def _advance_spawn_5b(self, snap: ZeldaSnapshot) -> FrameAction:
        self.spawn_frames += 1
        live = ROOM_5B_SPEC.live_enemies(snap)
        if live:
            return self._enter_clear_5b(snap)
        if self.spawn_frames >= SPAWN_SETTLE_FRAMES:
            self.success = True
            self.phase = "done"
            self.notes.append("spawn_5b_empty")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "spawn_5b_wait")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")

        # Already in 0x5b. Isolated enter-only succeeds. Spine waits for
        # Darknut HP bytes (same lag as raft spawn_59) before clear/skip.
        if self.phase == "spawn_5b":
            return self._advance_spawn_5b(snap)
        if self.phase != "clear_5b" and self._in_5b_playable(snap):
            if not self.clear_darknuts:
                self.success = True
                self.phase = "done"
                self.notes.append("already_0x5b")
                return FrameAction(nes_idle_action(), "done")
            if ROOM_5B_SPEC.live_enemies(snap):
                return self._enter_clear_5b(snap)
            return self._enter_spawn_5b(snap)

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
                if not self.clear_darknuts:
                    self.success = True
                    self.phase = "done"
                    self.notes.append("reached_0x5b")
                    return FrameAction(nes_idle_action(), "done")
                if ROOM_5B_SPEC.live_enemies(snap):
                    return self._enter_clear_5b(snap)
                return self._enter_spawn_5b(snap)
            if self.north_exit.failed:
                self.phase = "failed"
                self.notes.append("north_exit_failed")
            return action

        if self.phase == "clear_5b":
            action = self.clear_5b.step(snap)
            if self._5b_cleared(snap):
                self.success = True
                self.phase = "done"
                self.notes.append("darknuts_cleared")
                return FrameAction(nes_idle_action(), "done")
            if self.clear_5b.success:
                self.success = True
                self.phase = "done"
                self.notes.append("darknuts_cleared")
            elif self.clear_5b.phase is DungeonPhase.FAILED:
                self.phase = "failed"
                self.notes.append("clear_5b_failed")
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
            "clear_5b": self.clear_5b.report(),
            "spec_id": ROOM_6B_SPEC.spec_id,
            "clear_spec_id": ROOM_5B_SPEC.spec_id,
            "stop": (
                "level3_cleared_5b" if self.clear_darknuts else "level3_reached_5b"
            ),
            "clear_darknuts": self.clear_darknuts,
            "intervention_class": "survival" if self.clear_darknuts else "clean",
            "track": "assisted" if self.clear_darknuts else "clean",
        }
