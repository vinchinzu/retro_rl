"""Level 3 (Manji) dungeon room specs and pure helpers.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Lives outside ``dungeon.py`` so L2 room tables stay untouched.

Live pure (2026-08-06, Clean isolated from ``Level3Entrance``)::

    0x7c entry --(LEFT+UP corner residual)--> 0x7b
    0x7b: 6× Zol type ``0x13`` (HP>0) + fixed key RoomItemId ``0x19``
    Clear + key pickup ~658 combat frames after room-ready (3/3 trials).

Live pure chain from ``Level3WestKey`` (2026-08-06 recon + encode)::

    0x7b --(UP @ x≈120 strict)--> 0x6b
    0x6b: 5× Zol type ``0x13`` on diagonal-block floor; RoomItemId ``0x19``
          (key drop residual — type-0 HP leftovers stall RoomAllDead)
    0x6b --(UP @ x≈120 after type-0x13 clear)--> 0x5b
    0x5b: 3× Darknut type ``0x0b`` (HP 64); north open → 0x4b (3× Zol+key)
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
ROOM_L3_NORTH_ZOLS = 0x6B  # north of west-key; diagonal blocks
ROOM_L3_DARKNUTS = 0x5B  # north of 0x6b after zol clear
ROOM_L3_ZOL_KEY_4B = 0x4B  # north of darknuts (source next key)
ZOL_OBJECT_TYPE = 0x13  # live type on 0x7b/0x6b; wooden sword can leave type-0 HP residual
DARKNUT_OBJECT_TYPE = 0x0B  # live type on 0x5b (red Darknut, HP 64)
ROOM_ITEM_SMALL_KEY = 0x19

# West door residual: pure LEFT sticks at x≈32 (mask==0). LEFT+UP at the west
# wall corner-clips into the scroll (mode 6/7 → room 0x7b). Approach band y≈149
# reaches the wall; y≈141 alone often blocks mid-room at x≈112.
WEST_DOOR_APPROACH_Y = 149
WEST_DOOR_WALL_X = 48
WEST_ENTER_MAX_FRAMES = 1200

# North door residual from 0x7b: UP only works with |x-120|≤4. Threshold 8
# leaves Link at x≈112 and sticks on the north wall (live probe 2026-08-06).
NORTH_DOOR_X = 120
NORTH_DOOR_X_TOL = 4
NORTH_ENTER_MAX_FRAMES = 1500
NORTH_EXIT_6B_MAX_FRAMES = 6000

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

# 0x6b diagonal-block floor: prefer south/mid bands that stay walkable.
_ROOM_6B_PATROL: tuple[tuple[int, int], ...] = (
    (100, 181),
    (140, 181),
    (160, 173),
    (150, 157),
    (120, 157),
    (100, 157),
    (100, 173),
    (128, 141),
    (112, 173),
    (136, 165),
)

# After clear, snake toward north door plane (live free-explore path).
_ROOM_6B_NORTH_EXIT: tuple[tuple[int, int], ...] = (
    (120, 189),
    (144, 181),
    (152, 165),
    (144, 141),
    (136, 125),
    (128, 109),
    (120, 100),
    (120, 93),
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
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=10000,
    level=LEVEL3,
)

# Type-0x13 clear only: RoomAllDead often stays 0 (type-0 HP leftovers after
# wooden-sword hits). settle_all_dead=0 so CLEAR_ONLY trips when live Zols==0.
ROOM_6B_SPEC = DungeonRoomSpec(
    spec_id="level3_room6b_north_zols",
    source_room=ROOM_L3_WEST_KEY,
    room_id=ROOM_L3_NORTH_ZOLS,
    entry=DoorRoute(
        "UP",
        ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93)),
    ),
    enemy_types=(ZOL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_6B_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.CLEAR_ONLY,
        settle_all_dead=0,
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("DOWN", ((NORTH_DOOR_X, 205),)),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=12000,
    level=LEVEL3,
)

# Darknut room graph node (combat not pure-encoded yet — side/back hits only).
ROOM_5B_SPEC = DungeonRoomSpec(
    spec_id="level3_room5b_darknuts",
    source_room=ROOM_L3_NORTH_ZOLS,
    room_id=ROOM_L3_DARKNUTS,
    entry=DoorRoute(
        "UP",
        ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93)),
    ),
    enemy_types=(DARKNUT_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=(
            (80, 141),
            (120, 117),
            (160, 141),
            (160, 173),
            (120, 173),
            (80, 173),
            (120, 141),
        ),
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    exit_routes=(
        DoorRoute("DOWN", ((NORTH_DOOR_X, 205),)),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=15000,
    level=LEVEL3,
)

register_room_spec(ROOM_7B_SPEC)
register_room_spec(ROOM_6B_SPEC)
register_room_spec(ROOM_5B_SPEC)


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


def level3_room_6b_zols_cleared(ram: np.ndarray) -> bool:
    """0x6b with no live type-0x13 Zols (RoomAllDead not required)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_NORTH_ZOLS
        and snap.mode == PLAY_MODE
        and not ROOM_6B_SPEC.live_enemies(snap)
    )


def level3_reached_5b(ram: np.ndarray) -> bool:
    """Isolated pure stop: play mode inside 0x5b (Darknut room)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_DARKNUTS
        and snap.mode == PLAY_MODE
        and not snap.transitioning
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


_RIGHT_OF = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
_LEFT_OF = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}


def north_exit_6b_step(
    snap: ZeldaSnapshot,
    *,
    facing: str,
    prefer_goal: bool,
) -> tuple[FrameAction, str]:
    """One frame of 0x6b → 0x5b after Zol clear.

    Live residual: diagonal raised blocks partition the floor. Pure waypoint
    snakes stall after combat. Right-hand wall-follow reaches the north band
    (y≈93); once there, center x≈120 and hold UP into 0x5b.
    """
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3"), facing
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north6b_scroll"), facing
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}"), facing
    if snap.screen == ROOM_L3_DARKNUTS:
        return FrameAction(nes_idle_action(), "north_arrived_5b"), facing
    if snap.screen != ROOM_L3_NORTH_ZOLS:
        return (
            FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"),
            facing,
        )

    # Door plane / north band: center and hold UP.
    if snap.link_y <= 105:
        if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "north6b_align_door"), direction
        return FrameAction(nes_action("UP"), "north6b_push"), "UP"

    # Bias: if south of mid, prefer RIGHT then UP corridor (live free-explore).
    if prefer_goal and snap.link_y > 150:
        if snap.link_x < 140:
            return FrameAction(nes_action("RIGHT"), "north6b_bias_right"), "RIGHT"
        return FrameAction(nes_action("UP"), "north6b_bias_up"), "UP"

    # Right-hand wall follow (facing is last successful move).
    face = facing if facing in _RIGHT_OF else "UP"
    # Order: right of face, face, left, back — caller probes via stuck reset.
    order = (_RIGHT_OF[face], face, _LEFT_OF[face], _RIGHT_OF[_RIGHT_OF[face]])
    # Emit preferred first; Level3NorthExit6bController may override on stuck.
    direction = order[0]
    return FrameAction(nes_action(direction), f"north6b_follow_{direction}"), direction


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


# Free-explore target grid (live 2026-08-06: exits 0x6b→0x5b after combat).
_ROOM_6B_HUNT: tuple[tuple[int, int], ...] = tuple(
    (x, y) for y in range(90, 210, 8) for x in range(72, 200, 8)
) + tuple(
    (x, y) for y in range(90, 112, 4) for x in range(96, 152, 8)
) + (
    (120, 93),
    (120, 93),
    (120, 100),
    (120, 93),
)


@dataclass
class Level3NorthExit6bController:
    """Route 0x6b → 0x5b after Zols cleared (free-explore grid + door push).

    Live: after combat, walk a coarse grid; when blocked try alternate
    directions; on north band hold UP @ x≈120 into 0x5b.
    """

    max_frames: int = NORTH_EXIT_6B_MAX_FRAMES
    frames: int = 0
    hunt_index: int = 0
    target_steps: int = 0
    pending_alt: str | None = None
    success: bool = False
    failed: bool = False
    last_xy: tuple[int, int] | None = None
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
            return FrameAction(nes_action("UP"), "north6b_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L3_NORTH_ZOLS:
            return FrameAction(
                nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
            )

        # Consume one-shot alternate direction after a blocked step.
        if self.pending_alt is not None:
            direction = self.pending_alt
            self.pending_alt = None
            return FrameAction(nes_action(direction), "north6b_alt")

        xy = (snap.link_x, snap.link_y)
        blocked = self.last_xy == xy
        self.last_xy = xy

        # North band: center and push door.
        if snap.link_y <= 100 and abs(snap.link_x - NORTH_DOOR_X) <= 8:
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north6b_align_door")
            return FrameAction(nes_action("UP"), "north6b_push")
        if snap.link_y <= 100:
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north6b_align_door")
            return FrameAction(nes_action("UP"), "north6b_push")

        # Advance hunt target after a short attempt window or arrival.
        tx, ty = _ROOM_6B_HUNT[self.hunt_index % len(_ROOM_6B_HUNT)]
        self.target_steps += 1
        if (
            abs(snap.link_x - tx) <= 6 and abs(snap.link_y - ty) <= 6
        ) or self.target_steps >= 45:
            self.hunt_index = (self.hunt_index + 1) % len(_ROOM_6B_HUNT)
            self.target_steps = 0
            tx, ty = _ROOM_6B_HUNT[self.hunt_index % len(_ROOM_6B_HUNT)]

        dx, dy = tx - snap.link_x, ty - snap.link_y
        if abs(dx) > 3 and abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 3:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "UP"

        # If last frame did not move, queue an alternate direction.
        if blocked:
            alts = [d for d in ("UP", "RIGHT", "DOWN", "LEFT") if d != direction]
            self.pending_alt = alts[self.frames % len(alts)]
            if self.frames % 60 == 0:
                self.notes.append(f"block_f{self.frames}_hunt{self.hunt_index}")

        return FrameAction(nes_action(direction), f"north6b_hunt_{self.hunt_index}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "hunt_index": self.hunt_index,
            "notes": list(self.notes),
            "policy": "free-explore grid hunt + UP @ x≈120 on north band",
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
