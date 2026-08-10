"""Level 4 (Snake) dungeon room specs and live anchors.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Interior recon (assisted, rr-5lu 2026-08-09) — **no walkthrough hardcodes**.

Live path from ``Level4Entrance`` (room **0x71**)::

    0x71 entry (empty combat) --UP@x≈120--> 0x61
    0x61: 3× Vire type ``0x12`` (HP 64) → wooden sword splits to type ``0x1c``
    0x61 --BOMB_UP stand≈(120,105) face UP--> 0x51
    0x51: 8× Keese type ``0x1b`` (TYPE-only) + RoomItemId ``0x19`` key
    0x51 --LEFT @ y≈141--> 0x50 (5× Vire ``0x12``)  **tip residual**

Not Clean STATUS. Stepladder / Gleeok / TF ``0x08`` still residual.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level4_overworld import LEVEL4, LEVEL4_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live L4 room anchors (assisted recon rr-5lu 2026-08-09) ---
ROOM_L4_ENTRY = LEVEL4_ENTRY_ROOM  # 0x71 — empty combat mouth
ROOM_L4_VIRES_61 = 0x61  # north of entry; 3× Vire 0x12
ROOM_L4_KEESE_KEY_51 = 0x51  # bomb-N of 0x61; 8× Keese + key 0x19
ROOM_L4_VIRES_50 = 0x50  # west of 0x51; 5× Vire 0x12 (LIVE exit only)

VIRE_OBJECT_TYPE = 0x12  # live on 0x61/0x50; HP 64; splits on sword hit
VIRE_SPLIT_KEESE_TYPE = 0x1C  # live split residual from Vire (not standard 0x1B)
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_NONE = 0x03

# Bomb-north wall 0x61 → 0x51 (live stand ≈ y105, face UP).
BOMB_61_NORTH_STAND = (120, 105)
BOMB_61_NORTH_FACE = "UP"
BOMB_61_OPENS_TO = ROOM_L4_KEESE_KEY_51

_PATROL_MID: tuple[tuple[int, int], ...] = (
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


def level4_room_ready(snap: ZeldaSnapshot, room: int) -> bool:
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == room
        and not snap.transitioning
    )


def level4_entry_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_ENTRY)


def level4_room_61_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_VIRES_61)


def level4_room_51_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_KEESE_KEY_51)


def level4_room_61_cleared(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_61):
        return False
    enemies = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in (VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE)
    ]
    live = [o for o in enemies if o.hp > 0] or [
        o for o in enemies if o.type_id == VIRE_SPLIT_KEESE_TYPE
    ]
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_room_51_key_success(ram: np.ndarray) -> bool:
    """Keese clear + at least one key collected in room 0x51."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_KEESE_KEY_51):
        return False
    keese = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == KEESE_OBJECT_TYPE
    ]
    return len(keese) == 0 and snap.room_all_dead >= 20 and snap.keys >= 1


# --- Specs (assisted geometry; not Clean promote) ---
ROOM_71_SPEC = DungeonRoomSpec(
    spec_id="level4_room71_entry",
    source_room=ROOM_L4_ENTRY,
    room_id=ROOM_L4_ENTRY,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(),
    expected_enemy_count=0,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(patrol=((120, 150),)),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(DoorRoute("UP", ((120, 150), (120, 93))),),
    max_frames=2000,
    level=LEVEL4,
)

ROOM_61_SPEC = DungeonRoomSpec(
    spec_id="level4_room61_vires",
    source_room=ROOM_L4_ENTRY,
    room_id=ROOM_L4_VIRES_61,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=3,  # Vires; split increases count mid-fight
    alive_rule=AliveRule.TYPE_AND_HP,  # Vire uses HP; split 0x1c may be type-only
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=56,
        attack_phase=0,
        engage_attack_period=8,
        engage_attack_hold=4,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(),  # bomb wall, not free door
    max_frames=12000,
    level=LEVEL4,
)

ROOM_51_SPEC = DungeonRoomSpec(
    spec_id="level4_room51_keese_key",
    source_room=ROOM_L4_VIRES_61,
    room_id=ROOM_L4_KEESE_KEY_51,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE,  # Keese HP stays 0 while alive
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=48,
        attack_phase=0,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(128, 141),
        waypoints=((100, 125), (150, 157), (128, 141), (80, 160), (160, 120)),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (40, 141))),),
    max_frames=12000,
    level=LEVEL4,
)

register_room_spec(ROOM_71_SPEC)
register_room_spec(ROOM_61_SPEC)
register_room_spec(ROOM_51_SPEC)


class BombWall61North:
    """Geometry stand for ``BombWallController``: 0x61 bomb-UP → 0x51."""

    room = ROOM_L4_VIRES_61
    stand = BOMB_61_NORTH_STAND
    face = BOMB_61_NORTH_FACE
    opens_to = BOMB_61_OPENS_TO


def planning_interior_report() -> dict:
    """Machine-readable live interior facts for probes / docs."""
    return {
        "level": LEVEL4,
        "bead": "rr-5lu",
        "track": "assisted",
        "status": "interior_partial",
        "date": "2026-08-09",
        "entry_room": hex(ROOM_L4_ENTRY),
        "live_graph": {
            hex(ROOM_L4_ENTRY): {"UP": hex(ROOM_L4_VIRES_61)},
            hex(ROOM_L4_VIRES_61): {
                "BOMB_UP": hex(ROOM_L4_KEESE_KEY_51),
                "enemies": {"0x12": 3, "split": "0x1c"},
            },
            hex(ROOM_L4_KEESE_KEY_51): {
                "LEFT": hex(ROOM_L4_VIRES_50),
                "enemies": {"0x1b": 8},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
            },
            hex(ROOM_L4_VIRES_50): {"enemies": {"0x12": 5}, "note": "exit_only_recon"},
        },
        "bomb_61_north": {
            "stand": list(BOMB_61_NORTH_STAND),
            "face": BOMB_61_NORTH_FACE,
            "opens_to": hex(BOMB_61_OPENS_TO),
        },
        "not_yet": [
            "stepladder room / ADDR_LADDER",
            "Gleeok boss type",
            "TF bit 0x08 natural",
            "Clean promote",
        ],
    }
