"""Level 3 (Manji) dungeon room specs and west-key pure helpers.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Lives outside ``dungeon.py`` so L2 room tables stay untouched.

Live pure (2026-08-06, Clean isolated from ``Level3Entrance``)::

    0x7c entry --(LEFT+UP corner residual)--> 0x7b
    0x7b: 6× Zol type ``0x13`` (HP>0) + fixed key RoomItemId ``0x19``
    Clear + key pickup ~658 combat frames after room-ready (3/3 trials).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level3_overworld import LEVEL3, SCREEN_LEVEL3_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live L3 room / enemy anchors (isolated pure 2026-08-06) ---
ROOM_L3_ENTRY = SCREEN_LEVEL3_ENTRY_ROOM  # 0x7C
ROOM_L3_WEST_KEY = 0x7B
ZOL_OBJECT_TYPE = 0x13  # live type on 0x7b; wooden sword splits → Gel 0x15
ROOM_ITEM_SMALL_KEY = 0x19

# West door residual: pure LEFT sticks at x≈32 (mask==0). LEFT+UP at the west
# wall corner-clips into the scroll (mode 6/7 → room 0x7b). Approach band y≈149
# reaches the wall; y≈141 alone often blocks mid-room at x≈112.
WEST_DOOR_APPROACH_Y = 149
WEST_DOOR_WALL_X = 48
WEST_ENTER_MAX_FRAMES = 1200

_ROOM_7B_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 141),
    (160, 181),
    (112, 181),
    (64, 181),
    (64, 141),
    (120, 141),
)

ROOM_7B_SPEC = DungeonRoomSpec(
    spec_id="level3_room7b_west_key",
    source_room=ROOM_L3_ENTRY,
    room_id=ROOM_L3_WEST_KEY,
    entry=DoorRoute(
        "LEFT",
        ((120, WEST_DOOR_APPROACH_Y), (WEST_DOOR_WALL_X, WEST_DOOR_APPROACH_Y)),
    ),
    enemy_types=(ZOL_OBJECT_TYPE,),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_7B_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=10000,
    level=LEVEL3,
)

register_room_spec(ROOM_7B_SPEC)


def level3_room_7b_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x7b with keys≥1 and no live Zols.

    FIXED_INVENTORY stop: inventory + TYPE_AND_HP liveness only (RoomAllDead
    may lag after the last kill / key touch).
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_WEST_KEY
        and snap.mode == PLAY_MODE
        and snap.keys >= 1
        and not ROOM_7B_SPEC.live_enemies(snap)
    )


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
