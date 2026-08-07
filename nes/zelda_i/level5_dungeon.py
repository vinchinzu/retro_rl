"""Level 5 (Lizard) dungeon room specs and stop predicates.

Isolated pure for early L5 rooms. Imports combat infrastructure from
``dungeon`` only — do not edit ``dungeon.py`` from L5 agents.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    RewardKind,
    RewardSpec,
    dungeon_room_cleared,
    register_room_spec,
)
from zelda_i.ram import PLAY_MODE, read_snapshot

LEVEL_5 = 5
ROOM_L5_ENTRY = 0x76
ROOM_L5_GIBDO_66 = 0x66
# East residual of cleared 0x66 (live probe 2026-08-06); not Pols Voice.
ROOM_L5_EAST_67 = 0x67

# Type 0x30 — Gibdo-correlated (HP=112 at spawn; TYPE_AND_HP liveness).
GIBDO_OBJECT_TYPE = 0x30

# After clear of 0x66, ``cur_opened_doors`` becomes 0x08 and east opens → 0x67.
# North/west still blocked from this room without further items/geometry.
ROOM_66_EAST_DOOR_BIT = 0x08

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

register_room_spec(ROOM_66_SPEC)


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


__all__ = [
    "LEVEL_5",
    "ROOM_L5_ENTRY",
    "ROOM_L5_GIBDO_66",
    "ROOM_L5_EAST_67",
    "GIBDO_OBJECT_TYPE",
    "ROOM_66_EAST_DOOR_BIT",
    "ROOM_66_SPEC",
    "level5_room_66_cleared",
    "level5_in_room_66",
]
