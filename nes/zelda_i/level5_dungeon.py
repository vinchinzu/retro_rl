"""Level 5 (Lizard) dungeon room specs and stop predicates.

Isolated pure for early L5 rooms. Imports combat infrastructure from
``dungeon`` only — do not edit ``dungeon.py`` from L5 agents.

Live recon (updated 2026-08-14)::

    Entry **0x76** (south mouth). North open → **0x66**.
    Room **0x66** has 3× Gibdo and supplies the first key. Return south to
    **0x76**, then spend that key at the east door to **0x77** (5× Pols Voice
    type ``0x16`` + replacement key ``0x19``). Direct east from a zero-key
    entrance is correctly blocked. Combat pure from ``L5_Room_77`` is 2/2;
    composed east-door navigation remains the active route boundary.

    Cleared **0x66** east → **0x67**: 2× Bubble ``0x40`` (hp240, sword-immune)
    + 1× type ``0x4e`` (hp0). doors=0x02 (LEFT only) → dead-end residual.

    West of 0x66 (door poke): **0x65** 5× Gibdo ``0x30``; north **0x55**
    5× Zol ``0x13`` + item 0x19. Natural west/north from 0x66 still blocked
    with doors=0x08 (dark-room graph PARTIAL).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
    dungeon_room_cleared,
    inventory_reward_success,
    register_room_spec,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

LEVEL_5 = 5
ROOM_L5_ENTRY = 0x76
ROOM_L5_GIBDO_66 = 0x66
# East residual of cleared 0x66 (live probe 2026-08-06); Bubble dead-end.
ROOM_L5_EAST_67 = 0x67
# East of entry 0x76 — key door to Pols Voice + replacement small key.
ROOM_L5_POLS_77 = 0x77
# West of 0x66 when west door forced (PARTIAL natural).
ROOM_L5_WEST_65 = 0x65
# North of 0x65 when doors forced (PARTIAL / dark-room chain).
ROOM_L5_NORTH_55 = 0x55

# Type 0x30 — Gibdo-correlated (HP=112 at spawn; TYPE_AND_HP liveness).
GIBDO_OBJECT_TYPE = 0x30
# Type 0x16 — Pols Voice (HP=160; sword works with backstep; key 0x19).
POLS_VOICE_OBJECT_TYPE = 0x16
# Type 0x40 — Bubble (HP=240; sword does not reduce HP; invincible residual).
BUBBLE_OBJECT_TYPE = 0x40
# Type 0x4e — non-combat residual on 0x67 (hp0; trap/fire-correlated).
ROOM_67_TRAP_TYPE = 0x4E
# Type 0x13 — Zol (same id as L3 west-key room); seen on 0x55.
ZOL_OBJECT_TYPE = 0x13

ROOM_ITEM_SMALL_KEY = 0x19

# After clear of 0x66, ``cur_opened_doors`` becomes 0x08 and east opens → 0x67.
# North/west still blocked from this room without further items/geometry.
ROOM_66_EAST_DOOR_BIT = 0x08
# 0x67 settles with left doorway open back to 0x66.
ROOM_67_WEST_DOOR_BIT = 0x02
# 0x65 settles with east doorway open back to 0x66 (when entered west).
ROOM_65_EAST_DOOR_BIT = 0x01

# Stalfos-style room sweep; engage tighter for multi-hit Gibdos.
_ROOM_66_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 117),
    (192, 149),
    (160, 149),
    (112, 149),
    (64, 149),
    (64, 181),
    (112, 181),
    (160, 181),
    (192, 181),
)

_ROOM_77_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (160, 125),
    (96, 157),
)

_ROOM_67_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (120, 117),
    (176, 117),
    (176, 141),
    (120, 141),
    (64, 141),
)

# From entry 0x76 south mouth ~(120,205) walk north into 0x66.
# Also valid when already in 0x66 at south spawn (L5_Room_66).
ROOM_66_SPEC = DungeonRoomSpec(
    spec_id="level5_room66_gibdos",
    source_room=ROOM_L5_ENTRY,
    room_id=ROOM_L5_GIBDO_66,
    entry=DoorRoute(
        "UP",
        ((120, 205), (120, 93)),
    ),
    enemy_types=(GIBDO_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_66_PATROL,
        engage_distance=56,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    required_open_doors=ROOM_66_EAST_DOOR_BIT,
    exit_routes=(
        DoorRoute("DOWN", ((120, 205),)),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=12000,
    level=LEVEL_5,
)

# East residual of cleared 0x66 — Bubbles only; no clear pure (invincible).
# Graph node: enter RIGHT from 0x66, exit LEFT only.
ROOM_67_SPEC = DungeonRoomSpec(
    spec_id="level5_room67_bubbles",
    source_room=ROOM_L5_GIBDO_66,
    room_id=ROOM_L5_EAST_67,
    entry=DoorRoute(
        "RIGHT",
        ((120, 141), (208, 141)),
    ),
    enemy_types=(BUBBLE_OBJECT_TYPE,),
    expected_enemy_count=2,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_67_PATROL,
        engage_distance=32,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    # Left doorway open on settle (back to 0x66); R/U/D solid.
    required_open_doors=0,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (32, 141))),),
    max_frames=4000,
    level=LEVEL_5,
)

# East of entry: 5× Pols Voice 0x16 + fixed small key 0x19.
# A key from room 0x66 is required. Spec is pure once room-ready on 0x77.
ROOM_77_SPEC = DungeonRoomSpec(
    spec_id="level5_room77_pols_voice",
    source_room=ROOM_L5_ENTRY,
    room_id=ROOM_L5_POLS_77,
    entry=DoorRoute(
        "RIGHT",
        # Approach geometry lives in level5_path.EAST_DOOR_* .
        (
            (120, 157),
            (200, 157),
            (200, 141),
        ),
    ),
    enemy_types=(POLS_VOICE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_77_PATROL,
        engage_distance=72,
        engage_attack_period=5,
        engage_attack_hold=3,
        patrol_attack_period=6,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(120, 141),
        waypoints=(
            (120, 141),
            (96, 117),
            (144, 165),
            (80, 141),
            (160, 141),
            (120, 157),
            (120, 125),
        ),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (32, 141))),),
    max_frames=18000,
    level=LEVEL_5,
)

# West of 0x66 (PARTIAL: needs west door open). 5× Gibdo clear-only.
ROOM_65_SPEC = DungeonRoomSpec(
    spec_id="level5_room65_gibdos",
    source_room=ROOM_L5_GIBDO_66,
    room_id=ROOM_L5_WEST_65,
    entry=DoorRoute(
        "LEFT",
        ((120, 141), (32, 141)),
    ),
    enemy_types=(GIBDO_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_66_PATROL,
        engage_distance=56,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    exit_routes=(DoorRoute("RIGHT", ((120, 141), (208, 141))),),
    max_frames=12000,
    level=LEVEL_5,
)

register_room_spec(ROOM_66_SPEC)
register_room_spec(ROOM_67_SPEC)
register_room_spec(ROOM_77_SPEC)
register_room_spec(ROOM_65_SPEC)


def level5_room_66_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x66 3× Gibdo dead, RoomAllDead≥20, east door bit 0x08."""
    return dungeon_room_cleared(ram, ROOM_66_SPEC)


def level5_in_room_66(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x66 (pre- or post-clear)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_GIBDO_66
        and snap.mode == PLAY_MODE
    )


def level5_in_room_67(ram: np.ndarray) -> bool:
    """Play mode inside L5 residual room 0x67 (east of cleared 0x66)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_EAST_67
        and snap.mode == PLAY_MODE
    )


def level5_room_67_arrived(ram: np.ndarray) -> bool:
    """Graph stop: room-ready 0x67 with west door bit.

    Bubbles (type 0x40) spawn after a short settle; do not require them for the
    graph stop (sword-immune residual — arrival only, not clear).
    """
    snap = read_snapshot(ram)
    return (
        level5_in_room_67(ram)
        and (snap.cur_opened_doors & ROOM_67_WEST_DOOR_BIT) == ROOM_67_WEST_DOOR_BIT
    )


def level5_in_room_77(ram: np.ndarray) -> bool:
    """Play mode inside L5 Pols Voice room 0x77."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_POLS_77
        and snap.mode == PLAY_MODE
    )


def level5_room_77_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x77 keys≥1 and no live Pols Voice (type 0x16).

    FIXED_INVENTORY stop: inventory + liveness only (RoomAllDead may lag).
    """
    return inventory_reward_success(ram, ROOM_77_SPEC, min_value=1)


def level5_in_room_65(ram: np.ndarray) -> bool:
    """Play mode inside L5 west room 0x65 (PARTIAL natural entry)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_WEST_65
        and snap.mode == PLAY_MODE
    )


@dataclass
class Level5PolsVoiceController(GenericDungeonRoomController):
    """Pols Voice clear + backstep when stuck overlapping without kills.

    Live: HP=160 multi-hit; overlapping without damage stalls the generic
    engage loop. Retreat then re-engage (same pattern as L6 wizzrobes).
    """

    last_progress_frame: int = 0
    prev_live_count: int = -1
    backstep_frames: int = 0

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        self.combat_frames += 1
        n_live = len(live)
        if self.prev_live_count < 0:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
        elif n_live < self.prev_live_count:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
            self.backstep_frames = 0
            self.notes.append(f"kill_to_{n_live}_f{self.frames}")

        if not live:
            return self._patrol(snap)

        nearest = min(
            live,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        stuck_close = dist < 18 and (self.frames - self.last_progress_frame) > 80
        if stuck_close or self.backstep_frames > 0:
            if self.backstep_frames <= 0:
                self.backstep_frames = 28
                self.notes.append(f"backstep_f{self.frames}_d{dist}")
            self.backstep_frames -= 1
            if self.backstep_frames == 0:
                self.last_progress_frame = self.frames
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                direction = "LEFT" if dx >= 0 else "RIGHT"
            else:
                direction = "UP" if dy >= 0 else "DOWN"
            if snap.link_x < 48:
                direction = "RIGHT"
            elif snap.link_x > 200:
                direction = "LEFT"
            if snap.link_y < 100:
                direction = "DOWN"
            elif snap.link_y > 190:
                direction = "UP"
            return FrameAction(nes_action(direction), "pols_backstep")

        if dist < self.spec.combat.engage_distance:
            return self._engage(snap, nearest)
        return self._patrol(snap)


def make_pols_voice_controller() -> Level5PolsVoiceController:
    """Factory for room-77 Pols Voice + key controller."""
    return Level5PolsVoiceController(spec=ROOM_77_SPEC)


@dataclass
class Level5East67Controller:
    """Walk east from cleared 0x66 into residual 0x67; stop on arrival.

    No combat clear — Bubbles are sword-immune. Success = room-ready 0x67.
    Standalone (not GenericDungeonRoomController) so we skip clear logic.
    """

    max_frames: int = 4000
    settle_frames: int = 45
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": ROOM_67_SPEC.spec_id,
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        from retro_harness.nes import nes_idle_action

        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_EAST_67
            and snap.mode == PLAY_MODE
        ):
            if self.settle_left <= 0 and "settling_67" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_67")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_67")
            self.success = True
            self.notes.append("arrived_67")
            return FrameAction(nes_idle_action(), "arrived_67")

        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_action("RIGHT"), "scroll")

        # Leave south-ish positions then push RIGHT at y≈141.
        if snap.link_y > 170:
            return FrameAction(nes_action("UP"), "leave_south")
        if abs(snap.link_y - 141) > 4:
            btn = "UP" if snap.link_y > 141 else "DOWN"
            return FrameAction(nes_action(btn), "align_east_y")
        return FrameAction(nes_action("RIGHT"), "enter_67")


def make_east_67_controller() -> Level5East67Controller:
    return Level5East67Controller()


__all__ = [
    "LEVEL_5",
    "ROOM_L5_ENTRY",
    "ROOM_L5_GIBDO_66",
    "ROOM_L5_EAST_67",
    "ROOM_L5_POLS_77",
    "ROOM_L5_WEST_65",
    "ROOM_L5_NORTH_55",
    "GIBDO_OBJECT_TYPE",
    "POLS_VOICE_OBJECT_TYPE",
    "BUBBLE_OBJECT_TYPE",
    "ROOM_67_TRAP_TYPE",
    "ZOL_OBJECT_TYPE",
    "ROOM_ITEM_SMALL_KEY",
    "ROOM_66_EAST_DOOR_BIT",
    "ROOM_67_WEST_DOOR_BIT",
    "ROOM_65_EAST_DOOR_BIT",
    "ROOM_66_SPEC",
    "ROOM_67_SPEC",
    "ROOM_77_SPEC",
    "ROOM_65_SPEC",
    "level5_room_66_cleared",
    "level5_in_room_66",
    "level5_in_room_67",
    "level5_room_67_arrived",
    "level5_in_room_77",
    "level5_room_77_key_success",
    "level5_in_room_65",
    "Level5PolsVoiceController",
    "Level5East67Controller",
    "make_pols_voice_controller",
    "make_east_67_controller",
]
