"""Level 6 (Dragon) dungeon room specs and stop predicates.

Owned by L6 pure wave — do not put these specs into ``dungeon.py``.
Import ``GenericDungeonRoomController`` / dataclasses from ``dungeon`` only.

Live recon (2026-08-06 / 2026-08-07)::

    Entry **0x79** (empty combat, RoomItemId 0x03).
    East **0x7a**: 5× type 0x24 (orange wizzrobe-correlated) + key 0x19.
    RIGHT from entry: wall-first y≈157 → x≈208 → y≈144–149 (see
    ``level6_overworld.Level6EntryRightController``; no A while aligning).
    West **0x78**: key-LEFT from 0x79 (fire-bypass y≈157→141); 5× type 0x24.
    Trap: UP from 0x7a spends key on Old Man **0x6a** — do not.

Wizzrobe combat: sword misses when overlapping at the door; controller
backsteps when stuck too close without a kill, then re-engages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retro_harness.nes import nes_action
from retro_harness.input_script import FrameAction
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.dungeon_ids import (
    GEL_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

# Re-export for runners / docs.
ROOM_L6_ENTRY = LEVEL6_ENTRY_ROOM  # 0x79
ROOM_L6_EAST_KEY = LEVEL6_EAST_KEY_ROOM  # 0x7a
ROOM_L6_WEST_WIZZROBE = LEVEL6_WEST_WIZZROBE_ROOM  # 0x78
ROOM_L6_COMPASS = LEVEL6_COMPASS_ROOM  # 0x68
# After clear of 0x78, open_doorway_mask includes UP (0x08) → compass room 0x68.
ROOM_78_UP_DOOR_BIT = 0x08
# ADDR_COMPASS bitfield: one bit per dungeon (L6 → bit5 → 0x20).
LEVEL6_COMPASS_BIT = 1 << (LEVEL6 - 1)

# Open-floor patrol; wizzrobes teleport — cover mid lanes.
_ROOM_7A_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)

# Entry room — no live combat on room-ready; graph node + door routes.
ROOM_79_SPEC = DungeonRoomSpec(
    spec_id="level6_room79_entry",
    source_room=LEVEL6_ENTRY_ROOM,
    room_id=LEVEL6_ENTRY_ROOM,
    entry=DoorRoute("UP", ((120, 205),)),
    enemy_types=(),
    expected_enemy_count=0,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=((120, 141),),
        engage_distance=48,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        # Fire-block bypass: wall-first then door channel (controller owns timing).
        DoorRoute("RIGHT", ((120, 157), (208, 157), (208, 144))),
        DoorRoute("DOWN", ((120, 205),)),
    ),
    max_frames=2000,
    level=LEVEL6,
)

# East of entry: 5× type 0x24 + fixed RoomItemId small key (0x19).
# Key pickup observed near center after clear (keys 0→1); target (136,141).
ROOM_7A_SPEC = DungeonRoomSpec(
    spec_id="level6_room7a_east_key",
    source_room=LEVEL6_ENTRY_ROOM,
    room_id=LEVEL6_EAST_KEY_ROOM,
    entry=DoorRoute(
        "RIGHT",
        ((120, 157), (208, 157), (208, 144)),
    ),
    enemy_types=(WIZZROBE_ORANGE_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_7A_PATROL,
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=12000,
    level=LEVEL6,
)

# West of entry: key door from 0x79 (fire-bypass) then 5× type 0x24.
# Clear opens UP (mask bit 0x08) → 0x68 compass Zols. No room key drop.
ROOM_78_SPEC = DungeonRoomSpec(
    spec_id="level6_room78_west_wizzrobes",
    source_room=LEVEL6_ENTRY_ROOM,
    room_id=LEVEL6_WEST_WIZZROBE_ROOM,
    entry=DoorRoute(
        "LEFT",
        ((120, 157), (32, 157), (32, 141)),
    ),
    enemy_types=(WIZZROBE_ORANGE_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_7A_PATROL,
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    # Post-clear live: cur_opened_doors=0x01 (RIGHT), open_doorway_mask=0x09 (R+U).
    # UP kill-door is walkable via mask; do not gate CLEAR on door bit 0x08.
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=12000,
    level=LEVEL6,
)

register_room_spec(ROOM_79_SPEC)
register_room_spec(ROOM_7A_SPEC)
register_room_spec(ROOM_78_SPEC)

# North of cleared 0x78: 5× Zol 0x13 + RoomItemId 0x16 compass.
# Wooden sword splits Zols → gel 0x14/0x15. Ignore invuln 0x2b / block 0x68.
# Spine leftover is south mouth (120,205); occupancy miss-blocks the two
# statue clusters. Compass inventory is ADDR_COMPASS bit 0x20.
_ROOM_68_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (120, 141),
    (160, 173),
    (160, 189),
    (120, 109),
    (80, 109),
    (160, 109),
    (120, 93),
)

ROOM_68_SPEC = DungeonRoomSpec(
    spec_id="level6_room68_compass",
    source_room=LEVEL6_WEST_WIZZROBE_ROOM,
    room_id=LEVEL6_COMPASS_ROOM,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(GEL_SPLIT_OBJECT_TYPE,),
    combat=CombatTuning(
        patrol=_ROOM_68_PATROL,
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
        occupancy_patrol=True,
        occupancy_bounds=(16, 216, 77, 205),
        inland_dash=24,
        avoid_walls=True,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="compass",
        target=(120, 141),
        waypoints=(
            (120, 141),
            (120, 109),
            (80, 141),
            (160, 141),
            (120, 173),
            (120, 189),
            (80, 109),
            (160, 109),
            (64, 173),
            (176, 173),
        ),
    ),
    room_item_id=0x16,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=12000,
    level=LEVEL6,
)

register_room_spec(ROOM_68_SPEC)


def level6_room_7a_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x7a with keys≥1 and no live type-0x24 enemies.

    Same FIXED_INVENTORY stop as L2 west/east keys: inventory + liveness only.
    Do not require RoomAllDead lag after key pickup.
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.screen == LEVEL6_EAST_KEY_ROOM
        and snap.mode == PLAY_MODE
        and snap.keys >= 1
        and not ROOM_7A_SPEC.live_enemies(snap)
    )


@dataclass
class Level6EastKeyController(GenericDungeonRoomController):
    """Generic room clear + wizzrobe backstep when overlapping without kills.

    Live: sword swings at distance 0 on the west door miss forever; retreat
    when stuck too close, then re-engage from a short offset.
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
        stuck_close = (
            dist < 16 and (self.frames - self.last_progress_frame) > 100
        )
        if stuck_close or self.backstep_frames > 0:
            if self.backstep_frames <= 0:
                self.backstep_frames = 24
                self.notes.append(f"backstep_f{self.frames}_d{dist}")
            self.backstep_frames -= 1
            if self.backstep_frames == 0:
                # Allow a fresh engage window after retreat.
                self.last_progress_frame = self.frames
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                direction = "LEFT" if dx >= 0 else "RIGHT"
            else:
                direction = "UP" if dy >= 0 else "DOWN"
            # Prefer center when pinned on a door edge.
            if snap.link_x < 40:
                direction = "RIGHT"
            elif snap.link_x > 200:
                direction = "LEFT"
            return FrameAction(nes_action(direction), "wizzrobe_backstep")

        if dist < self.spec.combat.engage_distance:
            return self._engage(snap, nearest)
        return self._patrol(snap)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["last_progress_frame"] = self.last_progress_frame
        base["prev_live_count"] = self.prev_live_count
        return base


def make_east_key_controller() -> Level6EastKeyController:
    """Factory: GenericDungeonRoomController subclass bound to ROOM_7A_SPEC."""
    return Level6EastKeyController(spec=ROOM_7A_SPEC)


def level6_room_78_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x78 cleared — no live type-0x24, play mode.

    Does not require UP door bit (mask lag) or inventory change.
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.screen == LEVEL6_WEST_WIZZROBE_ROOM
        and snap.mode == PLAY_MODE
        and not ROOM_78_SPEC.live_enemies(snap)
    )


@dataclass
class Level6WestWizzrobeController(Level6EastKeyController):
    """Same wizzrobe backstep combat as east key, bound to ROOM_78_SPEC."""


def make_west_wizzrobe_controller() -> Level6WestWizzrobeController:
    """Factory: backstep combat controller for 0x78 west wizzrobes."""
    return Level6WestWizzrobeController(spec=ROOM_78_SPEC)


def level6_room_68_compass_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x68 Zols/gels dead and L6 compass bit set.

    Compass-style bitfield — do not use keys-style min_value.
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.screen == LEVEL6_COMPASS_ROOM
        and snap.mode == PLAY_MODE
        and (snap.compass & LEVEL6_COMPASS_BIT) != 0
        and not ROOM_68_SPEC.live_enemies(snap)
    )


def make_compass_68_controller() -> GenericDungeonRoomController:
    """Occupancy-patrol Zol clear + compass hunt on 0x68. Ignore 0x2b/0x68."""
    return GenericDungeonRoomController(spec=ROOM_68_SPEC)


__all__ = [
    "ROOM_L6_ENTRY",
    "ROOM_L6_EAST_KEY",
    "ROOM_L6_WEST_WIZZROBE",
    "ROOM_L6_COMPASS",
    "ROOM_79_SPEC",
    "ROOM_7A_SPEC",
    "ROOM_78_SPEC",
    "ROOM_68_SPEC",
    "ROOM_78_UP_DOOR_BIT",
    "LEVEL6_COMPASS_BIT",
    "Level6EastKeyController",
    "Level6WestWizzrobeController",
    "make_east_key_controller",
    "make_west_wizzrobe_controller",
    "make_compass_68_controller",
    "level6_room_7a_key_success",
    "level6_room_78_clear_success",
    "level6_room_68_compass_success",
]
