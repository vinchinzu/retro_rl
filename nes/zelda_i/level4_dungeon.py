"""Level 4 (Snake) dungeon room specs and live anchors.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Interior recon (assisted/pure, rr-5lu / rr-2ysf 2026-08-09/10) —
**no walkthrough hardcodes** beyond live IDs.

Live path from ``Level4Entrance`` (room **0x71**)::

    0x71 entry (empty combat) --UP@x≈120--> 0x61
    0x61: 3× Vire type ``0x12`` (HP 64) → wooden sword splits to type ``0x1c``
    0x61 --BOMB_UP stand≈(120,105) face UP--> 0x51
    0x51: 8× Keese type ``0x1b`` (TYPE-only) + RoomItemId ``0x19`` key
    0x51 --LEFT @ y≈141--> 0x50 (5× Vire ``0x12``)
    0x51 --DOWN @ x≈120--> 0x61
    0x61 --KEY-RIGHT @ y≈141 (keys 1→0)--> 0x62
    0x62: 5× Vire + RoomItemId ``0x16`` Compass (dark maze)
    0x62 --maze compass + return LEFT--> 0x61 (ADDR_COMPASS bit 0x08)
    **Post-compass expand (rr-xc3x / rr-q8eq / rr-n1wn live):** 0x50 is
    **not** a dead-end. Scripted north → **0x40** (5× Zol ``0x13`` → gel
    ``0x14`` + key ``0x19`` via east-corridor path). Free UP → **0x30**
    (3× Vire + 2× invuln ``0x2b``; ignore invuln for clear). First rooms
    outside early component {0x71, 0x61, 0x51, 0x50, 0x62}. 0x51 UP/RIGHT
    sealed; 0x40 L/R sealed. ADDR_LADDER residual (stepladder further
    north/east of 0x30 after clear+exit probe).

Not Clean STATUS. Stepladder / Gleeok / TF ``0x08`` still residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.bomb_wall_path import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level4_overworld import LEVEL4, LEVEL4_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live L4 room anchors (rr-5lu / rr-2ysf 2026-08-09/10) ---
ROOM_L4_ENTRY = LEVEL4_ENTRY_ROOM  # 0x71 — empty combat mouth
ROOM_L4_VIRES_61 = 0x61  # north of entry; 3× Vire 0x12
ROOM_L4_KEESE_KEY_51 = 0x51  # bomb-N of 0x61; 8× Keese + key 0x19
ROOM_L4_VIRES_50 = 0x50  # west of 0x51; 5× Vire 0x12 (north exit → 0x40)
ROOM_L4_COMPASS_62 = 0x62  # KEY-RIGHT of 0x61; 5× Vire + compass 0x16 dark maze
ROOM_L4_ZOLS_40 = 0x40  # north of 0x50; 5× Zol 0x13 + key 0x19 (rr-xc3x)

VIRE_OBJECT_TYPE = 0x12  # live on 0x61/0x50/0x62; HP 64; sword splits → 0x1c
VIRE_SPLIT_KEESE_TYPE = 0x1C  # live split residual from Vire (not standard 0x1B)
ZOL_OBJECT_TYPE = 0x13  # live on 0x40; HP 32; wooden sword splits → gel 0x14
GEL_SPLIT_OBJECT_TYPE = 0x14  # Zol split residual (HP stays 0 while alive)
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_COMPASS = 0x16  # live room item on 0x62
ROOM_ITEM_NONE = 0x03
LEVEL4_COMPASS_BIT = 0x08  # ADDR_COMPASS bit for dungeon level 4

# Bomb-north wall 0x61 → 0x51 (live stand ≈ y105, face UP).
BOMB_61_NORTH_STAND = (120, 105)
BOMB_61_NORTH_FACE = "UP"
BOMB_61_OPENS_TO = ROOM_L4_KEESE_KEY_51

# Key-east door 0x61 → 0x62 (live: y≈141 hold RIGHT; keys 1→0).
KEY_61_EAST_Y = 141
KEY_61_EAST_Y_TOL = 4
KEY_61_OPENS_TO = ROOM_L4_COMPASS_62

# Free LEFT 0x51 → 0x50.
LEFT_51_Y = 141

# Dark-maze 0x62 compass (rr-9so0 live BFS from Level4Room62Cleared).
# Hold each token for MAZE_IN_HOLD / MAZE_OUT_HOLD frames. Pickup ~ (136,132).
# After compass, corridor path back to west vestibule then LEFT → 0x61 play.
MAZE_IN_HOLD = 6
MAZE_OUT_HOLD = 4
MAZE_62_TO_COMPASS: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "UP",
)
MAZE_62_RETURN_WEST: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "LEFT",
)
COMPASS_PICKUP_XY = (136, 132)

# 0x50 → 0x40 north (rr-xc3x live). Interior blocks block center+UP.
# Prefer waypoint seek (robust to clear_50 end pose) then long UP.
# Token path kept as fallback / docs (hold MAZE_50_HOLD from ≈(160,149)).
MAZE_50_HOLD = 6
MAZE_50_LONG_UP = 280
MAZE_50_TO_NORTH: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
)
# Live intermediate cells on successful BFS (tol ±8).
MAZE_50_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (160, 181),
    (112, 181),
    (112, 120),
    (128, 100),
    (120, 72),
    (120, 56),
)

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
    """0x61 with no live Vire/split and RoomAllDead settle."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_61):
        return False
    live = ROOM_61_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_room_51_key_success(ram: np.ndarray) -> bool:
    """Keese clear + at least one key collected in room 0x51."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_KEESE_KEY_51):
        return False
    keese = ROOM_51_SPEC.live_enemies(snap)
    return len(keese) == 0 and snap.room_all_dead >= 20 and snap.keys >= 1


def level4_room_50_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_VIRES_50)


def level4_room_50_cleared(ram: np.ndarray) -> bool:
    """0x50 with no live Vire/split and RoomAllDead settle."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_50):
        return False
    live = ROOM_50_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_room_62_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_COMPASS_62)


def level4_room_62_cleared(ram: np.ndarray) -> bool:
    """0x62 Vires cleared (compass pickup residual)."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_COMPASS_62):
        return False
    live = ROOM_62_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_compass_collected(ram: np.ndarray) -> bool:
    """L4 compass inventory bit set (ADDR_COMPASS & 0x08)."""
    snap = read_snapshot(ram)
    return bool(snap.compass & LEVEL4_COMPASS_BIT)


def level4_compass_route_success(ram: np.ndarray) -> bool:
    """Compass bit set and back on 0x61 play-ready (maze return complete)."""
    snap = read_snapshot(ram)
    return (
        bool(snap.compass & LEVEL4_COMPASS_BIT)
        and level4_room_ready(snap, ROOM_L4_VIRES_61)
    )


def level4_room_40_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_ZOLS_40)


def level4_room_40_cleared(ram: np.ndarray) -> bool:
    """0x40 Zols+gels cleared (key pickup residual).

    RoomAllDead often stays 0 after wooden-sword Zol→Gel splits (type-only
    residuals), so clear is live-enemy emptiness only (settle_all_dead=0).
    """
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_ZOLS_40):
        return False
    live = ROOM_40_SPEC.live_enemies(snap)
    return len(live) == 0


def level4_room_40_key_success(ram: np.ndarray) -> bool:
    """Zols+gels clear + ≥1 key on 0x40 (RoomItemId 0x19)."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_ZOLS_40):
        return False
    live = ROOM_40_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.keys >= 1


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
    alive_rule=AliveRule.TYPE_AND_HP,  # Vire uses HP
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),  # split 0x1c HP stays 0
    object_slot_max=12,  # splits land in slots 10–11+
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(),  # bomb wall, not free door
    max_frames=16000,
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
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=56,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        # Live pickup ~ (136,149) after clear; dense mid-room hunt.
        target=(136, 149),
        waypoints=(
            (128, 141),
            (136, 149),
            (120, 149),
            (144, 141),
            (112, 141),
            (128, 157),
            (128, 125),
            (96, 157),
            (160, 157),
            (96, 125),
            (160, 125),
            (80, 141),
            (176, 141),
            (120, 173),
            (120, 109),
        ),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (40, 141))),),
    max_frames=16000,
    level=LEVEL4,
)

ROOM_50_SPEC = DungeonRoomSpec(
    spec_id="level4_room50_vires",
    source_room=ROOM_L4_KEESE_KEY_51,
    room_id=ROOM_L4_VIRES_50,
    entry=DoorRoute("LEFT", ((224, 141), (180, 141))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (220, 141))),
        DoorRoute("UP", ((120, 80), (120, 56))),  # scripted north → 0x40
    ),
    max_frames=20000,
    level=LEVEL4,
)

ROOM_62_SPEC = DungeonRoomSpec(
    spec_id="level4_room62_vires_compass",
    source_room=ROOM_L4_VIRES_61,
    room_id=ROOM_L4_COMPASS_62,
    entry=DoorRoute("RIGHT", ((16, 141), (48, 141))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    # Compass is a bitfield — clear-only here; pickup residual (rr-2ysf maze).
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_COMPASS,
    exit_routes=(DoorRoute("LEFT", ((48, 141), (16, 141))),),
    max_frames=20000,
    level=LEVEL4,
)

# Live key pickup after full Zol+Gel clear (rr-q8eq dense BFS): ~(120, 117).
KEY_40_PICKUP_XY = (120, 117)
# North free exit after clear → 0x30 (3× Vire 0x12 + 2× invuln 0x2b residual).
ROOM_L4_NORTH_30 = 0x30
# KEY-RIGHT of cleared 0x30 → 0x31 (5× Vire 0x12; rr-n1wn live).
ROOM_L4_EAST_31 = 0x31
# Invuln movers on 0x30 (slots 1–2); never count as combat clear targets.
INVULN_MOVER_TYPE = 0x2B
# Key-east door 0x30 → 0x31 (live: y≈141 hold RIGHT; keys 1→0).
KEY_30_EAST_Y = 141
KEY_30_EAST_Y_TOL = 4

_PATROL_40: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (96, 117),
    (144, 117),
    (120, 93),
    (80, 157),
    (160, 157),
)

ROOM_40_SPEC = DungeonRoomSpec(
    spec_id="level4_room40_zols_key",
    source_room=ROOM_L4_VIRES_50,
    room_id=ROOM_L4_ZOLS_40,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    # Wooden sword splits Zol 0x13 → Gel 0x14 (HP=0 while alive).
    enemy_types=(ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(GEL_SPLIT_OBJECT_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_40,
        engage_distance=56,
        attack_phase=4,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        # RoomAllDead often stays 0 with gel residuals — settle on live empty.
        settle_all_dead=0,
        target=KEY_40_PICKUP_XY,
        waypoints=(
            (120, 117),
            (112, 117),
            (128, 117),
            (120, 109),
            (120, 125),
            (104, 117),
            (136, 117),
            (120, 101),
            (96, 125),
            (144, 125),
            (80, 117),
            (160, 117),
            (120, 141),
            (120, 93),
        ),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("DOWN", ((120, 189), (120, 205))),
        DoorRoute("UP", ((120, 80), (120, 56))),  # free N → 0x30 after clear
    ),
    max_frames=20000,
    level=LEVEL4,
)

# 0x30: 3× Vire + 2× invuln 0x2b (ignore invuln; rr-n1wn).
# Live geometry: Link walkable band is y∈[128,208] only (solid north wall at
# y≈128). Vires fly above and into the band — clear from north-band patrol
# with Y-first engage (face UP into flyers). RoomAllDead may stay 0/low while
# 0x2b remains — settle on Vire emptiness only.
_PATROL_30: tuple[tuple[int, int], ...] = (
    (40, 133),
    (80, 133),
    (120, 133),
    (160, 133),
    (200, 133),
    (160, 141),
    (120, 141),
    (80, 141),
    (48, 149),
    (192, 149),
    (120, 157),
    (64, 165),
    (176, 165),
)

ROOM_30_SPEC = DungeonRoomSpec(
    spec_id="level4_room30_vires",
    source_room=ROOM_L4_ZOLS_40,
    room_id=ROOM_L4_NORTH_30,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=3,  # Vires; split increases mid-fight; ignore 0x2b
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_30,
        engage_distance=96,  # reach flyers when they dip into north band
        engage_dominant_axis=True,  # face UP/DOWN first (Vires above wall)
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    # Invuln residuals keep RoomAllDead from settling — clear = no live Vire/split.
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("DOWN", ((120, 189), (120, 205))),
        # N/E/W residual — probe after clear (rr-n1wn).
    ),
    max_frames=20000,
    level=LEVEL4,
)

register_room_spec(ROOM_71_SPEC)
register_room_spec(ROOM_61_SPEC)
register_room_spec(ROOM_51_SPEC)
register_room_spec(ROOM_50_SPEC)
register_room_spec(ROOM_62_SPEC)
register_room_spec(ROOM_40_SPEC)
register_room_spec(ROOM_30_SPEC)


class BombWall61North:
    """Geometry stand for ``BombWallController``: 0x61 bomb-UP → 0x51."""

    room = ROOM_L4_VIRES_61
    stand = BOMB_61_NORTH_STAND
    face = BOMB_61_NORTH_FACE
    opens_to = BOMB_61_OPENS_TO


def _need_clear_61(snap: ZeldaSnapshot) -> bool:
    """True while any Vire/split type is present (HP may be 0 on first spawn frames)."""
    return any(
        1 <= o.slot <= 12
        and o.type_id in (VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE)
        for o in snap.objects
    )


def make_bomb_61_north_controller(
    *, clear_vires: bool = True
) -> BombWallController:
    """0x61 → bomb north → 0x51. Optionally clear Vires first."""
    return BombWallController(
        wall=BombWall61North(),
        level=LEVEL4,
        clear_spec=ROOM_61_SPEC if clear_vires else None,
        clear_when=_need_clear_61 if clear_vires else None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=20000,
    )


def make_room_61_clear_controller() -> GenericDungeonRoomController:
    """Fight-only or entry+fight clear of 0x61 Vires."""
    return GenericDungeonRoomController(ROOM_61_SPEC)


def make_room_51_key_controller() -> GenericDungeonRoomController:
    """Clear 0x51 Keese + collect key (FIXED_INVENTORY keys)."""
    return GenericDungeonRoomController(ROOM_51_SPEC)


def make_room_50_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x50 Vires (north exit → 0x40 after clear; rr-xc3x)."""
    return GenericDungeonRoomController(ROOM_50_SPEC)


def make_room_62_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x62 Vires (compass maze; pickup / exits residual)."""
    return GenericDungeonRoomController(ROOM_62_SPEC)


def make_room_40_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x40 Zols+gels (key pickup residual)."""
    return GenericDungeonRoomController(ROOM_40_SPEC)


def level4_room_30_cleared(ram: np.ndarray) -> bool:
    """0x30 Vires+splits cleared (invuln 0x2b residual OK).

    RoomAllDead often stays 0 with invuln movers present, so clear is
    live-Vire emptiness only (settle_all_dead=0).
    """
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_NORTH_30):
        return False
    live = ROOM_30_SPEC.live_enemies(snap)
    return len(live) == 0


# --- 0x30 Vire clear from north band (rr-n1wn) ---

# Link cannot walk north of y≈128 (solid wall). Vires fly above and dip into
# the walkable band — clear by east-west patrol on the north band facing UP.
_NORTH_BAND_Y = 133
_NORTH_BAND_Y_MAX = 148
_CLEAR30_PATROL_X: tuple[int, ...] = (40, 80, 120, 160, 200, 160, 120, 80)


class Clear30Phase(Enum):
    TO_BAND = auto()
    FIGHT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Clear30Controller:
    """Clear 3× Vire on 0x30 from the north walkable band (ignore 0x2b).

    Live (rr-n1wn): walkable cells y∈[128,208]; flyers above wall need UP
    slashes when they share x or dip into the band. Generic mid-room chase
    starves damage.
    """

    max_frames: int = 20000
    phase: Clear30Phase = Clear30Phase.TO_BAND
    frames: int = 0
    phase_frames: int = 0
    combat_frames: int = 0
    patrol_index: int = 0
    max_live_enemies: int = 0
    last_live_enemies: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Clear30Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Clear30Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _live(self, snap: ZeldaSnapshot) -> tuple:
        return ROOM_30_SPEC.live_enemies(snap)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        # period 6 hold 3 — same as ROOM_30_SPEC combat tuning
        if (self.combat_frames % 6) < 3:
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _fight_step(self, snap: ZeldaSnapshot) -> FrameAction:
        from zelda_i.combat import should_swing_at

        self.combat_frames += 1
        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))

        if (
            not live
            and self.max_live_enemies >= ROOM_30_SPEC.expected_enemy_count
        ):
            self.success = True
            self._set_phase(Clear30Phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")

        if not live:
            # Seen empty before expected count — keep patrolling briefly.
            return FrameAction(nes_action("UP"), "wait_spawn")

        # Prefer targets in/near the walkable band; else any Vire for x-align.
        band = [o for o in live if o.y >= 112]
        targets = band if band else list(live)
        nearest = min(
            targets,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y

        # Stay on north band.
        if snap.link_y > _NORTH_BAND_Y_MAX:
            return FrameAction(nes_action("UP"), "return_north_band")

        # Flyer above: face UP and slash when roughly under them.
        above = nearest.y < snap.link_y - 6
        if above and abs(dx) <= 28:
            return self._swing("UP", "slash_up_flyer")
        if above and abs(dx) > 8:
            direction = "RIGHT" if dx > 0 else "LEFT"
            # Keep a light UP bias so we don't drift south while aligning.
            if snap.link_y > _NORTH_BAND_Y + 4:
                return FrameAction(nes_action("UP"), "reband_while_align")
            return FrameAction(nes_action(direction), "align_x_flyer")

        # Target in band: close then slash.
        if abs(dy) > 10 and nearest.y >= 112:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) > 8:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "UP" if above or abs(dy) <= 10 else (
                "DOWN" if dy > 0 else "UP"
            )

        if should_swing_at(
            snap.link_x, snap.link_y, direction, (nearest,)
        ) or (abs(dx) <= 16 and abs(dy) <= 28):
            return self._swing(direction, "engage")

        # No close target — east-west patrol on the band.
        tx = _CLEAR30_PATROL_X[self.patrol_index % len(_CLEAR30_PATROL_X)]
        if abs(snap.link_x - tx) <= 6:
            self.patrol_index += 1
            tx = _CLEAR30_PATROL_X[self.patrol_index % len(_CLEAR30_PATROL_X)]
        if snap.link_y > _NORTH_BAND_Y + 6:
            return FrameAction(nes_action("UP"), "patrol_reband")
        direction = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(direction), "patrol_band")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Clear30Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Clear30Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_NORTH_30:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))
        if (
            not live
            and self.max_live_enemies >= ROOM_30_SPEC.expected_enemy_count
        ):
            self.success = True
            self._set_phase(Clear30Phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")

        if self.phase is Clear30Phase.TO_BAND:
            if snap.link_y <= _NORTH_BAND_Y_MAX and abs(snap.link_x - 120) <= 40:
                self._set_phase(Clear30Phase.FIGHT, "on_north_band")
            else:
                if abs(snap.link_x - 120) > 6 and snap.link_y > 160:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                        "center_x_south",
                    )
                if snap.link_y > _NORTH_BAND_Y:
                    return FrameAction(nes_action("UP"), "walk_north_band")
                self._set_phase(Clear30Phase.FIGHT, "on_north_band")

        return self._fight_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "combat_frames": self.combat_frames,
            "max_live_enemies": self.max_live_enemies,
            "last_live_enemies": self.last_live_enemies,
            "notes": list(self.notes),
            "segment": "level4_clear_0x30",
            "patrol_x": list(_CLEAR30_PATROL_X),
            "north_band_y": _NORTH_BAND_Y,
        }


def make_room_30_clear_controller() -> Level4Clear30Controller:
    """Clear 0x30 Vires from north band (ignore invuln 0x2b; rr-n1wn)."""
    return Level4Clear30Controller()


class KeyRight31Phase(Enum):
    CLEAR = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4KeyRight31Controller:
    """From 0x30 with ≥1 key: optional clear Vires, then KEY-RIGHT into 0x31.

    Live (rr-n1wn): hold RIGHT @ y≈141; keys 1→0; enter west door ~(16,141).
    0x31 has 5× Vire ``0x12``. Free N/E/W sealed; KEY-LEFT none; DOWN→0x40.
    """

    clear_vires: bool = True
    max_frames: int = 25000
    phase: KeyRight31Phase = KeyRight31Phase.CLEAR
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: Level4Clear30Controller | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.clear_vires:
            self._clear = Level4Clear30Controller()
        else:
            self.phase = KeyRight31Phase.ALIGN

    def _set_phase(self, phase: KeyRight31Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(KeyRight31Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is KeyRight31Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is KeyRight31Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_EAST_31
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(KeyRight31Phase.DONE, "entered_0x31")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is KeyRight31Phase.CLEAR:
            if snap.screen != ROOM_L4_NORTH_30:
                return self._fail(f"clear_wrong_room_0x{snap.screen:02x}")
            assert self._clear is not None
            live = ROOM_30_SPEC.live_enemies(snap)
            # Pre-cleared checkpoint (Level4Room30Cleared) or fight just finished.
            if not live and (
                self._clear.max_live_enemies >= 3
                or self._clear.success
                or self.phase_frames <= 2
            ):
                self.keys_before = snap.keys
                note = (
                    "cleared_0x30"
                    if self._clear.max_live_enemies >= 3 or self._clear.success
                    else "precleared_0x30"
                )
                self._set_phase(KeyRight31Phase.ALIGN, note)
            else:
                return self._clear.step(snap)

        if snap.screen not in (ROOM_L4_NORTH_30, ROOM_L4_EAST_31):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.keys_before is None:
            self.keys_before = snap.keys
        if self.keys_before is not None and self.keys_before < 1 and snap.keys < 1:
            return self._fail("no_keys")

        if abs(snap.link_y - KEY_30_EAST_Y) > KEY_30_EAST_Y_TOL:
            self._set_phase(KeyRight31Phase.ALIGN, "align_y")
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_30_EAST_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(KeyRight31Phase.PUSH, "push_key_right")
        return FrameAction(nes_action("RIGHT"), "push_key_right")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_key_right_0x31",
            "target_room": f"0x{ROOM_L4_EAST_31:02x}",
            "key_y": KEY_30_EAST_Y,
            "keys_before": self.keys_before,
        }


def make_key_right_31_controller(
    *, clear_vires: bool = True
) -> Level4KeyRight31Controller:
    """0x30 → KEY-RIGHT @y141 → 0x31 (5× Vire). Optionally clear Vires first."""
    return Level4KeyRight31Controller(clear_vires=clear_vires)


def level4_room_31_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_EAST_31)


# Live scripted key path after combat clear pose ≈(136–140, 164–165).
# East-corridor route (rr-q8eq BFS): UP×2 RIGHT×5 UP×4 LEFT×5 hold6 → key ~(136,117).
MAZE_40_KEY_HOLD = 6
MAZE_40_TO_KEY: tuple[str, ...] = (
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
)


class Key40Phase(Enum):
    FIGHT = auto()
    PATH = auto()
    HUNT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Key40Controller:
    """Clear 0x40 Zols+gels then scripted path to RoomItemId 0x19 key.

    Live (rr-q8eq): wooden sword splits Zol→Gel; after clear, center-band key
    is not reachable via naive mid-room patrol (south pocket walls). Use
    ``MAZE_40_TO_KEY`` hold6 from the common clear pose.
    """

    max_frames: int = 25000
    phase: Key40Phase = Key40Phase.FIGHT
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: GenericDungeonRoomController = field(init=False, repr=False)
    _hunt_i: int = 0

    def __post_init__(self) -> None:
        self._clear = GenericDungeonRoomController(ROOM_40_SPEC)
        self._clear.phase = DungeonPhase.FIGHT

    def _set_phase(self, phase: Key40Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Key40Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Key40Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Key40Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self.keys_before is None and snap.screen == ROOM_L4_ZOLS_40:
            self.keys_before = snap.keys

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_ZOLS_40
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self.keys_before is not None
            and snap.keys > self.keys_before
            and len(ROOM_40_SPEC.live_enemies(snap)) == 0
        ):
            self.success = True
            self._set_phase(Key40Phase.DONE, "key_collected")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_ZOLS_40:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.phase is Key40Phase.FIGHT:
            live = ROOM_40_SPEC.live_enemies(snap)
            if (
                not live
                and self._clear.max_live_enemies >= ROOM_40_SPEC.expected_enemy_count
            ):
                self._set_phase(Key40Phase.PATH, "room_cleared")
                self.path_index = 0
                self.hold_left = 0
            else:
                return self._clear.step(snap)

        if self.phase is Key40Phase.PATH:
            if self.path_index >= len(MAZE_40_TO_KEY):
                self._set_phase(Key40Phase.HUNT, "path_done")
            else:
                direction = MAZE_40_TO_KEY[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_40_KEY_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze40_{direction}")

        if self.phase is Key40Phase.HUNT:
            # Micro orbit around key band if path undershot.
            orbit = ("LEFT", "UP", "RIGHT", "DOWN", "LEFT", "UP", "RIGHT", "DOWN")
            if self.phase_frames >= 400:
                return self._fail("key_hunt_timeout")
            direction = orbit[(self.phase_frames // 8) % len(orbit)]
            return FrameAction(nes_action(direction), "key_hunt")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "path_index": self.path_index,
            "keys_before": self.keys_before,
            "segment": "level4_key_0x40",
            "maze_path": list(MAZE_40_TO_KEY),
            "hold": MAZE_40_KEY_HOLD,
            "pickup_xy": list(KEY_40_PICKUP_XY),
            "clear": self._clear.report(),
        }


def make_room_40_key_controller() -> Level4Key40Controller:
    """Clear 0x40 Zols+gels + collect key via scripted east-corridor path."""
    return Level4Key40Controller()


# --- 0x51 free LEFT → 0x50 (rr-2ysf pocket) ---


class Left50Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Left50Controller:
    """From 0x51 (post-key): align y≈141, push LEFT into 0x50 play-ready."""

    max_frames: int = 2500
    phase: Left50Phase = Left50Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Left50Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Left50Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Left50Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Left50Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_50
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(Left50Phase.DONE, "entered_0x50")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("LEFT"), "scroll_left")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen not in (ROOM_L4_KEESE_KEY_51, ROOM_L4_VIRES_50):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if abs(snap.link_y - LEFT_51_Y) > KEY_61_EAST_Y_TOL:
            return FrameAction(
                nes_action("UP" if snap.link_y > LEFT_51_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(Left50Phase.PUSH, "push_left")
        return FrameAction(nes_action("LEFT"), "push_left")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_left_0x50",
            "target_room": f"0x{ROOM_L4_VIRES_50:02x}",
        }


def make_left_50_controller() -> Level4Left50Controller:
    return Level4Left50Controller()


# --- 0x61 KEY-RIGHT → 0x62 compass maze (rr-2ysf) ---


class KeyRight62Phase(Enum):
    CLEAR = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4KeyRight62Controller:
    """From 0x61 with ≥1 key: optional clear Vires, then KEY-RIGHT into 0x62.

    Live (rr-2ysf): hold RIGHT @ y≈141; keys 1→0; enter vestibule ~(16,141).
    """

    clear_vires: bool = True
    max_frames: int = 25000
    phase: KeyRight62Phase = KeyRight62Phase.CLEAR
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: GenericDungeonRoomController | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.clear_vires:
            self._clear = GenericDungeonRoomController(ROOM_61_SPEC)
            self._clear.phase = DungeonPhase.FIGHT
        else:
            self.phase = KeyRight62Phase.ALIGN

    def _set_phase(self, phase: KeyRight62Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(KeyRight62Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is KeyRight62Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is KeyRight62Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_COMPASS_62
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(KeyRight62Phase.DONE, "entered_0x62")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is KeyRight62Phase.CLEAR:
            if snap.screen != ROOM_L4_VIRES_61:
                return self._fail(f"clear_wrong_room_0x{snap.screen:02x}")
            assert self._clear is not None
            live = any(
                1 <= o.slot <= 12
                and o.type_id in (VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE)
                for o in snap.objects
            )
            if not live and snap.room_all_dead >= 10:
                self.keys_before = snap.keys
                self._set_phase(KeyRight62Phase.ALIGN, "cleared_0x61")
            else:
                return self._clear.step(snap)

        if snap.screen not in (ROOM_L4_VIRES_61, ROOM_L4_COMPASS_62):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.keys_before is None:
            self.keys_before = snap.keys
        # keys may drop to 0 mid-push after the lock consumes; keep holding RIGHT.

        if abs(snap.link_y - KEY_61_EAST_Y) > KEY_61_EAST_Y_TOL:
            self._set_phase(KeyRight62Phase.ALIGN, "align_y")
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(KeyRight62Phase.PUSH, "push_key_right")
        return FrameAction(nes_action("RIGHT"), "push_key_right")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "keys_before": self.keys_before,
            "segment": "level4_key_right_0x62",
            "target_room": f"0x{ROOM_L4_COMPASS_62:02x}",
            "clear": self._clear.report() if self._clear is not None else None,
        }


def make_key_right_62_controller(*, clear_vires: bool = True) -> Level4KeyRight62Controller:
    return Level4KeyRight62Controller(clear_vires=clear_vires)


# --- 0x62 dark maze: compass + return LEFT → 0x61 (rr-9so0) ---


class Compass62Phase(Enum):
    MAZE_IN = auto()
    MAZE_OUT = auto()
    EXIT_LEFT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Compass62Controller:
    """From cleared 0x62: maze to compass pickup, return west, LEFT to 0x61.

    Live (rr-9so0): hold scripted dirs (BFS) — open seek fails on maze walls.
    Success: ``ADDR_COMPASS & 0x08`` and play-ready on 0x61.
    """

    max_frames: int = 12000
    phase: Compass62Phase = Compass62Phase.MAZE_IN
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    compass_at_frame: int | None = None
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Compass62Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.path_index = 0
            self.hold_left = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Compass62Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Compass62Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Compass62Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            bool(snap.compass & LEVEL4_COMPASS_BIT)
            and snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_61
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(Compass62Phase.DONE, "compass_and_0x61")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")

        # Scroll through door transitions while exiting west.
        if snap.transitioning or snap.mode in (4, 6, 7):
            if self.phase is Compass62Phase.EXIT_LEFT or (
                snap.screen in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61)
            ):
                return FrameAction(nes_action("LEFT"), "scroll_left")
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if bool(snap.compass & LEVEL4_COMPASS_BIT) and self.compass_at_frame is None:
            self.compass_at_frame = self.frames
            self.notes.append(f"compass_bit_f{self.frames}")

        if self.phase is Compass62Phase.MAZE_IN:
            if snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_in_wrong_room_0x{snap.screen:02x}")
            if bool(snap.compass & LEVEL4_COMPASS_BIT):
                self._set_phase(Compass62Phase.MAZE_OUT, "got_compass")
                # fall through to MAZE_OUT this frame
            else:
                if self.path_index >= len(MAZE_62_TO_COMPASS):
                    return self._fail("maze_in_path_exhausted_no_compass")
                direction = MAZE_62_TO_COMPASS[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_IN_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_in_{direction}")

        if self.phase is Compass62Phase.MAZE_OUT:
            if snap.screen == ROOM_L4_VIRES_61:
                self._set_phase(Compass62Phase.EXIT_LEFT, "already_0x61")
            elif snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_out_wrong_room_0x{snap.screen:02x}")
            elif self.path_index >= len(MAZE_62_RETURN_WEST):
                self._set_phase(Compass62Phase.EXIT_LEFT, "return_path_done")
            else:
                direction = MAZE_62_RETURN_WEST[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_OUT_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_out_{direction}")

        # EXIT_LEFT: push west door / finish settle on 0x61
        if snap.screen == ROOM_L4_VIRES_61 and snap.mode == PLAY_MODE:
            if bool(snap.compass & LEVEL4_COMPASS_BIT) and not snap.transitioning:
                self.success = True
                self._set_phase(Compass62Phase.DONE, "settled_0x61")
                return FrameAction(nes_idle_action(), "done")
        if snap.screen not in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61):
            return self._fail(f"exit_wrong_room_0x{snap.screen:02x}")
        # Align y≈141 when still in 0x62 vestibule, then LEFT.
        if snap.screen == ROOM_L4_COMPASS_62 and abs(snap.link_y - KEY_61_EAST_Y) > 8:
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN"),
                "align_exit_y",
            )
        return FrameAction(nes_action("LEFT"), "exit_left")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "compass_at_frame": self.compass_at_frame,
            "path_index": self.path_index,
            "segment": "level4_compass_0x62",
            "maze_in": list(MAZE_62_TO_COMPASS),
            "maze_out": list(MAZE_62_RETURN_WEST),
            "pickup_xy": list(COMPASS_PICKUP_XY),
        }


def make_compass_62_controller() -> Level4Compass62Controller:
    return Level4Compass62Controller()


# --- 0x50 cleared → north scripted → 0x40 (rr-xc3x) ---


class North40Phase(Enum):
    WAYPOINTS = auto()
    PUSH_UP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4North40Controller:
    """From cleared 0x50: token path then long UP into 0x40.

    Live (rr-xc3x): ``MAZE_50_TO_NORTH`` hold6 is reliable from the common
    clear_50 end pose ≈(160,149). Interior blocks block center+UP.
    """

    max_frames: int = 10000
    phase: North40Phase = North40Phase.WAYPOINTS  # WAYPOINTS = token path phase
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: North40Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.path_index = 0
            self.hold_left = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(North40Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_40(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_ZOLS_40
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is North40Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is North40Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self._entered_40(snap):
            self.success = True
            self._set_phase(North40Phase.DONE, "entered_0x40")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")

        if snap.transitioning or snap.mode in (4, 6, 7):
            if snap.screen in (ROOM_L4_VIRES_50, ROOM_L4_ZOLS_40):
                return FrameAction(nes_action("UP"), "scroll_up")
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen == ROOM_L4_ZOLS_40:
            self.success = True
            self._set_phase(North40Phase.DONE, "on_0x40")
            return FrameAction(nes_idle_action(), "done")

        if snap.screen != ROOM_L4_VIRES_50:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        # Early north-band boost: if already near door, just push UP.
        if snap.link_y <= 80 and abs(snap.link_x - 120) <= 16:
            self._set_phase(North40Phase.PUSH_UP, "near_north_band")
            return FrameAction(nes_action("UP"), "push_up_north")

        if self.phase is North40Phase.WAYPOINTS:
            if self.path_index >= len(MAZE_50_TO_NORTH):
                self._set_phase(North40Phase.PUSH_UP, "path_done")
                return FrameAction(nes_action("UP"), "push_up_north")
            direction = MAZE_50_TO_NORTH[self.path_index]
            # If stalled on a wall, advance token early and try next.
            if self._stall >= 18:
                self.notes.append(f"stall_skip_{self.path_index}_{direction}")
                self.path_index += 1
                self.hold_left = 0
                self._stall = 0
                if self.path_index >= len(MAZE_50_TO_NORTH):
                    self._set_phase(North40Phase.PUSH_UP, "path_done_stall")
                    return FrameAction(nes_action("UP"), "push_up_north")
                direction = MAZE_50_TO_NORTH[self.path_index]
            self.hold_left += 1
            if self.hold_left >= MAZE_50_HOLD:
                self.path_index += 1
                self.hold_left = 0
            return FrameAction(nes_action(direction), f"maze50_{direction}")

        if self.phase_frames >= MAZE_50_LONG_UP + 120:
            return self._fail("push_up_timeout")
        return FrameAction(nes_action("UP"), "push_up_north")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "path_index": self.path_index,
            "segment": "level4_north_0x40",
            "waypoints": [list(w) for w in MAZE_50_WAYPOINTS],
            "maze_path": list(MAZE_50_TO_NORTH),
            "hold": MAZE_50_HOLD,
            "long_up": MAZE_50_LONG_UP,
        }


def make_north_40_controller() -> Level4North40Controller:
    return Level4North40Controller()


# --- 0x40 cleared+key → free UP → 0x30 (rr-q8eq) ---


class North30Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4North30Controller:
    """From cleared 0x40: center x≈120, push UP into 0x30 play-ready.

    Live (rr-q8eq dense BFS): free north from north-band y≤68 @ x≈120.
    0x30 has 3× Vire ``0x12`` + 2× invuln residual ``0x2b``.
    """

    max_frames: int = 4000
    phase: North30Phase = North30Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: North30Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(North30Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_30(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_NORTH_30
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is North30Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is North30Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self._entered_30(snap):
            self.success = True
            self._set_phase(North30Phase.DONE, "entered_0x30")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll_up")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen == ROOM_L4_NORTH_30:
            self.success = True
            self._set_phase(North30Phase.DONE, "on_0x30")
            return FrameAction(nes_idle_action(), "done")

        if snap.screen != ROOM_L4_ZOLS_40:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if abs(snap.link_x - 120) > 6:
            self._set_phase(North30Phase.ALIGN, "align_x")
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                "align_x",
            )
        self._set_phase(North30Phase.PUSH, "push_up")
        return FrameAction(nes_action("UP"), "push_up_north")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_north_0x30",
            "target_room": f"0x{ROOM_L4_NORTH_30:02x}",
        }


def make_north_30_controller() -> Level4North30Controller:
    return Level4North30Controller()


def level4_room_30_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_NORTH_30)


# --- 0x71 empty entry → UP → 0x61 (rr-zchy) ---


class EntryUpPhase(Enum):
    SETTLE = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4EntryUpController:
    """Empty 0x71 mouth: center x≈120, push UP into 0x61 play-ready."""

    max_frames: int = 2500
    phase: EntryUpPhase = EntryUpPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: EntryUpPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(EntryUpPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is EntryUpPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is EntryUpPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_61
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(EntryUpPhase.DONE, "entered_0x61")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll_up")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is EntryUpPhase.SETTLE:
            if snap.screen != ROOM_L4_ENTRY:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            self._set_phase(EntryUpPhase.ALIGN, "align_x")

        if self.phase is EntryUpPhase.ALIGN:
            if abs(snap.link_x - 120) > 4:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                    "align_x",
                )
            self._set_phase(EntryUpPhase.PUSH, "push_up")
            return FrameAction(nes_action("UP"), "push_up")

        if self.phase is EntryUpPhase.PUSH:
            if abs(snap.link_x - 120) > 6:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                    "re_align_x",
                )
            return FrameAction(nes_action("UP"), "push_up")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_entry_up_0x71",
            "target_room": f"0x{ROOM_L4_VIRES_61:02x}",
        }


def make_entry_up_controller() -> Level4EntryUpController:
    return Level4EntryUpController()


def planning_interior_report() -> dict:
    """Machine-readable live interior facts for probes / docs."""
    return {
        "level": LEVEL4,
        "bead": "rr-5lu",
        "tip": "rr-n1wn",
        "track": "pure",
        "status": "interior_0x30_clear_stepladder_residual",
        "date": "2026-08-10",
        "entry_room": hex(ROOM_L4_ENTRY),
        "live_graph": {
            hex(ROOM_L4_ENTRY): {"UP": hex(ROOM_L4_VIRES_61)},
            hex(ROOM_L4_VIRES_61): {
                "BOMB_UP": hex(ROOM_L4_KEESE_KEY_51),
                "KEY_RIGHT": hex(ROOM_L4_COMPASS_62),
                "RIGHT_reenter": hex(ROOM_L4_COMPASS_62),
                "DOWN": hex(ROOM_L4_ENTRY),
                "enemies": {"0x12": 3, "split": "0x1c"},
            },
            hex(ROOM_L4_KEESE_KEY_51): {
                "LEFT": hex(ROOM_L4_VIRES_50),
                "DOWN": hex(ROOM_L4_VIRES_61),
                "UP": "sealed",
                "RIGHT": "sealed",
                "enemies": {"0x1b": 8},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
                "note": "UP/RIGHT not key doors (poke keys no consume)",
            },
            hex(ROOM_L4_VIRES_50): {
                "enemies": {"0x12": 5},
                "RIGHT": hex(ROOM_L4_KEESE_KEY_51),
                "UP_scripted": hex(ROOM_L4_ZOLS_40),
                "note": "north via MAZE_50_TO_NORTH hold6 + long UP (rr-xc3x)",
            },
            hex(ROOM_L4_COMPASS_62): {
                "enemies": {"0x12": 5},
                "room_item": hex(ROOM_ITEM_COMPASS),
                "LEFT": hex(ROOM_L4_VIRES_61),
                "compass_bit": hex(LEVEL4_COMPASS_BIT),
                "pickup_xy": list(COMPASS_PICKUP_XY),
                "note": "dark_maze_compass_live_return_west_no_bomb_exit",
            },
            hex(ROOM_L4_ZOLS_40): {
                "enemies": {"0x13": 5, "split": "0x14"},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
                "key_pickup_xy": list(KEY_40_PICKUP_XY),
                "DOWN": hex(ROOM_L4_VIRES_50),
                "UP": hex(ROOM_L4_NORTH_30),
                "LEFT": "sealed",
                "RIGHT": "sealed",
                "note": "first outside early component; clear+key then free UP (rr-q8eq)",
            },
            hex(ROOM_L4_NORTH_30): {
                "enemies": {"0x12": 3, "0x2b": 2},
                "DOWN": hex(ROOM_L4_ZOLS_40),
                "KEY_RIGHT": hex(ROOM_L4_EAST_31),
                "UP": "sealed",
                "LEFT": "sealed",
                "RIGHT_free": "sealed",
                "note": (
                    "clear Vires ignore invuln 0x2b (rr-n1wn); walkable y≥128; "
                    "KEY-RIGHT@y141 → 0x31 (5× Vire); stepladder residual"
                ),
            },
            hex(ROOM_L4_EAST_31): {
                "enemies": {"0x12": 5},
                "LEFT": hex(ROOM_L4_NORTH_30),
                "note": "live KEY-RIGHT of 0x30 (rr-n1wn); stepladder residual",
            },
        },
        "post_compass": {
            "bead": "rr-o0nn",
            "expand": "rr-n1wn",
            "start": "Level4Compass",
            "early_component": [
                hex(ROOM_L4_ENTRY),
                hex(ROOM_L4_VIRES_61),
                hex(ROOM_L4_KEESE_KEY_51),
                hex(ROOM_L4_VIRES_50),
                hex(ROOM_L4_COMPASS_62),
            ],
            "first_outside": hex(ROOM_L4_ZOLS_40),
            "next_outside": hex(ROOM_L4_NORTH_30),
            "keys_at_compass": 0,
            "ladder": 0,
            "evidence": [
                "recordings/l4_xc3x_breakthrough.json",
                "recordings/l4_q8eq_40_dense_bfs.json",
                "recordings/l4_q8eq_key40_key_40.json",
                "recordings/l4_n1wn_clear30_clear_30.json",
            ],
            "blocked": [
                "0x51 UP/RIGHT sealed (not key)",
                "0x62 bomb exits none",
                "0x40 LEFT/RIGHT sealed",
                "no Vire key-farm drops (8 cycles)",
            ],
            "opened": [
                "0x50 north scripted → 0x40 (Zols + key 0x19)",
                "0x40 clear+key → free UP → 0x30",
                "0x30 clear Vires (ignore 0x2b)",
                "0x30 KEY-RIGHT @y141 → 0x31 (5× Vire)",
            ],
        },
        "bomb_61_north": {
            "stand": list(BOMB_61_NORTH_STAND),
            "face": BOMB_61_NORTH_FACE,
            "opens_to": hex(BOMB_61_OPENS_TO),
        },
        "key_61_east": {
            "y": KEY_61_EAST_Y,
            "opens_to": hex(KEY_61_OPENS_TO),
            "key_cost": 1,
        },
        "maze_62": {
            "in_hold": MAZE_IN_HOLD,
            "out_hold": MAZE_OUT_HOLD,
            "to_compass": list(MAZE_62_TO_COMPASS),
            "return_west": list(MAZE_62_RETURN_WEST),
            "pickup_xy": list(COMPASS_PICKUP_XY),
        },
        "maze_50_north": {
            "hold": MAZE_50_HOLD,
            "long_up": MAZE_50_LONG_UP,
            "path": list(MAZE_50_TO_NORTH),
            "opens_to": hex(ROOM_L4_ZOLS_40),
        },
        "segments": {
            "entry_up": "rr-zchy",
            "clear_vires_61": "rr-yr77",
            "bomb_up_51": "rr-h278",
            "keese_key_51": "rr-wqdu",
            "clear_50": "rr-2ysf",
            "key_right_62": "rr-2ysf",
            "clear_62": "rr-2ysf",
            "compass_62": "rr-9so0",
            "north_40": "rr-xc3x",
            "key_40": "rr-q8eq",
            "north_30": "rr-q8eq",
            "clear_30": "rr-n1wn",
            "key_right_31": "rr-n1wn",
            "stepladder_path": "rr-o0nn",
        },
        "key_40": {
            "pickup_xy": list(KEY_40_PICKUP_XY),
            "gel_split": hex(GEL_SPLIT_OBJECT_TYPE),
            "opens_north": hex(ROOM_L4_NORTH_30),
        },
        "clear_30": {
            "enemies": {"0x12": 3, "ignore": "0x2b"},
            "settle_all_dead": 0,
            "walkable_y_min": 128,
            "checkpoint": "Level4Room30Cleared",
        },
        "key_right_31": {
            "y": KEY_30_EAST_Y,
            "opens_to": hex(ROOM_L4_EAST_31),
            "key_cost": 1,
            "checkpoint": "Level4Room31",
        },
        "not_yet": [
            "stepladder room / ADDR_LADDER",
            "0x31 clear + further expand toward ladder",
            "Gleeok boss type",
            "TF bit 0x08 natural",
            "Clean promote",
        ],
    }


# Re-export for scripts that want phase types
__all__ = [
    "BOMB_61_NORTH_FACE",
    "BOMB_61_NORTH_STAND",
    "BOMB_61_OPENS_TO",
    "BombWall61North",
    "COMPASS_PICKUP_XY",
    "Compass62Phase",
    "DungeonPhase",
    "EntryUpPhase",
    "KEY_61_EAST_Y",
    "KEY_61_OPENS_TO",
    "KeyRight62Phase",
    "LEVEL4_COMPASS_BIT",
    "Left50Phase",
    "Level4Compass62Controller",
    "Level4EntryUpController",
    "Level4KeyRight62Controller",
    "Level4Left50Controller",
    "KEY_40_PICKUP_XY",
    "Key40Phase",
    "Level4Key40Controller",
    "Level4North30Controller",
    "Level4North40Controller",
    "MAZE_40_KEY_HOLD",
    "MAZE_40_TO_KEY",
    "MAZE_50_HOLD",
    "MAZE_50_LONG_UP",
    "MAZE_50_TO_NORTH",
    "MAZE_62_RETURN_WEST",
    "MAZE_62_TO_COMPASS",
    "MAZE_IN_HOLD",
    "MAZE_OUT_HOLD",
    "North30Phase",
    "North40Phase",
    "KEY_30_EAST_Y",
    "KeyRight31Phase",
    "Level4Clear30Controller",
    "Level4KeyRight31Controller",
    "ROOM_30_SPEC",
    "ROOM_40_SPEC",
    "ROOM_50_SPEC",
    "ROOM_51_SPEC",
    "ROOM_61_SPEC",
    "ROOM_62_SPEC",
    "ROOM_L4_EAST_31",
    "ROOM_L4_NORTH_30",
    "ROOM_L4_ZOLS_40",
    "GEL_SPLIT_OBJECT_TYPE",
    "INVULN_MOVER_TYPE",
    "ZOL_OBJECT_TYPE",
    "level4_room_30_cleared",
    "level4_room_31_ready",
    "make_key_right_31_controller",
    "make_room_30_clear_controller",
    "ROOM_71_SPEC",
    "ROOM_ITEM_COMPASS",
    "ROOM_L4_COMPASS_62",
    "ROOM_L4_ENTRY",
    "ROOM_L4_KEESE_KEY_51",
    "ROOM_L4_VIRES_50",
    "ROOM_L4_VIRES_61",
    "VIRE_OBJECT_TYPE",
    "VIRE_SPLIT_KEESE_TYPE",
    "level4_compass_collected",
    "level4_compass_route_success",
    "level4_entry_ready",
    "level4_room_30_ready",
    "level4_room_40_cleared",
    "level4_room_40_key_success",
    "level4_room_40_ready",
    "level4_room_50_cleared",
    "level4_room_50_ready",
    "level4_room_51_key_success",
    "level4_room_51_ready",
    "level4_room_61_cleared",
    "level4_room_61_ready",
    "level4_room_62_cleared",
    "level4_room_62_ready",
    "level4_room_ready",
    "make_bomb_61_north_controller",
    "make_compass_62_controller",
    "make_entry_up_controller",
    "make_key_right_62_controller",
    "make_left_50_controller",
    "make_north_30_controller",
    "make_north_40_controller",
    "make_room_40_clear_controller",
    "make_room_40_key_controller",
    "make_room_50_clear_controller",
    "make_room_51_key_controller",
    "make_room_61_clear_controller",
    "make_room_62_clear_controller",
    "planning_interior_report",
]
