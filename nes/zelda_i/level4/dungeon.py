"""Level 4 (Snake) dungeon room specs, stop predicates, and live anchors.

Path controllers and maze timing live in ``level4.path``, ``level4.maze_path``,
and ``level4.stepladder``. Assisted geometry; not Clean promote.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.dungeon import ids as _ids
from zelda_i.level4.overworld import LEVEL4, LEVEL4_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live L4 room anchors (rr-5lu / rr-2ysf 2026-08-09/10) ---
ROOM_L4_ENTRY = LEVEL4_ENTRY_ROOM  # 0x71 — empty combat mouth
ROOM_L4_VIRES_61 = 0x61  # north of entry; 3× Vire 0x12
ROOM_L4_KEESE_KEY_51 = 0x51  # bomb-N of 0x61; 8× Keese + key 0x19
ROOM_L4_VIRES_50 = 0x50  # west of 0x51; 5× Vire 0x12 (north exit → 0x40)
ROOM_L4_COMPASS_62 = 0x62  # KEY-RIGHT of 0x61; 5× Vire + compass 0x16 dark maze
ROOM_L4_ZOLS_40 = 0x40  # north of 0x50; 5× Zol 0x13 + key 0x19 (rr-xc3x)
# Free RIGHT of cleared 0x31 → 0x32 (rr-resv). Stairs under left block → 0x60 (rr-tib8).
ROOM_L4_STEPLADDER = 0x60  # mode-9 basement under 0x32; RoomItemId 0x0d → ADDR_LADDER
# Post-ladder map branch (rr-rvae live 2026-08-10): KEY-UP 0x30 → 0x20 → RIGHT → 0x21 map.
ROOM_L4_WATER_NORTH_20 = 0x20  # KEY-UP of 0x30 with ladder; 5× Vire 0x12
ROOM_L4_MAP_21 = 0x21  # free RIGHT of cleared 0x20; 5× Gel 0x15 + map 0x17
# Gleeok approach from map (rr-rvae live recon 2026-08-10)
ROOM_L4_MANHANDLA_10 = 0x10  # free UP of 0x20; Manhandla 0x3c (optional side)
ROOM_L4_BUBBLES_00 = 0x00  # free UP of 0x10; bubbles 0x40 dead-end
ROOM_L4_MID_11 = 0x11  # BOMB_UP of map 0x21; type 0x35 cluster
ROOM_L4_KEY_01 = 0x01  # free UP of 0x11; 8× Keese + key 0x19 (natural key)
ROOM_L4_VIRES_12 = 0x12  # free/bomb RIGHT of 0x11; 5× Vire + block 0x68
ROOM_L4_TRAPS_02 = 0x02  # free UP of 0x12; blade traps 0x49 dead-end
ROOM_L4_GLEEOK_13 = 0x13  # east of 0x12; Gleeok type 0x43 + HC 0x1a

# 0x12 → Gleeok: after Vire clear, push block 0x68 LEFT one tile → doors raw 2→3
# (R bit). Naive hold-RIGHT fails (maze); use hold4 PATH_12_TO_GLEEOK (rr-rvae dual).
PUSH_12_STAND = (112, 144)
PUSH_12_DIR = "LEFT"
PUSH_12_BLOCK_FROM = (96, 144)
PUSH_12_BLOCK_TO = (80, 144)

# Enemy types: dungeon_ids (HYGIENE rule 5). Re-exported for L4 consumers.
VIRE_OBJECT_TYPE = _ids.VIRE_OBJECT_TYPE
VIRE_SPLIT_KEESE_TYPE = _ids.VIRE_SPLIT_KEESE_TYPE
ZOL_OBJECT_TYPE = _ids.ZOL_OBJECT_TYPE
GEL_SPLIT_OBJECT_TYPE = _ids.GEL_SPLIT_OBJECT_TYPE
GEL_OBJECT_TYPE = _ids.GEL_OBJECT_TYPE
LIKE_LIKE_OBJECT_TYPE = _ids.LIKE_LIKE_OBJECT_TYPE
MID_11_OBJECT_TYPE = _ids.L4_MID_11_OBJECT_TYPE
GLEEOK_OBJECT_TYPE = _ids.GLEEOK_OBJECT_TYPE
GLEEOK_HEAD_OBJECT_TYPE = _ids.GLEEOK_HEAD_OBJECT_TYPE
GLEEOK_FIREBALL_TYPE = _ids.MANHANDLA_PROJECTILE_TYPE
ROOM_L4_TRIFORCE = 0x03  # north of Gleeok 0x13 after clear → TF 0x08
BLADE_TRAP_OBJECT_TYPE = _ids.BLADE_TRAP_OBJECT_TYPE
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_COMPASS = 0x16  # live room item on 0x62
ROOM_ITEM_STEPLADDER = 0x0D  # live on 0x60 stairs basement (rr-tib8)
ROOM_ITEM_MAP = 0x17  # live room item on 0x21 (rr-rvae); ADDR_MAP bit 0x08
ROOM_ITEM_HEART_CONTAINER = 0x1A  # live on 0x13 with Gleeok
ROOM_ITEM_NONE = 0x03
LEVEL4_COMPASS_BIT = 0x08  # ADDR_COMPASS bit for dungeon level 4
LEVEL4_MAP_BIT = 0x08  # ADDR_MAP bit for dungeon level 4
LEVEL4_TRIFORCE_BIT = 0x08  # ADDR_TRIFORCE bit for dungeon level 4
# Map bomb-north wall 0x21 → 0x11 (live stand ≈ y105, face UP).
BOMB_21_NORTH_STAND = (120, 105)
BOMB_21_NORTH_FACE = "UP"
BOMB_21_OPENS_TO = ROOM_L4_MID_11

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


COMPASS_PICKUP_XY = (136, 132)

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


def _ram_ready(room: int):
    def pred(ram: np.ndarray) -> bool:
        return level4_room_ready(read_snapshot(ram), room)
    return pred


def _cleared(room: int, spec_attr: str, *, settle: int = 0, keys: int | None = None):
    def pred(ram: np.ndarray) -> bool:
        snap = read_snapshot(ram)
        if not level4_room_ready(snap, room):
            return False
        if globals()[spec_attr].live_enemies(snap):
            return False
        if settle and snap.room_all_dead < settle:
            return False
        return keys is None or snap.keys >= keys
    return pred


level4_entry_ready = _ram_ready(ROOM_L4_ENTRY)
level4_room_61_ready = _ram_ready(ROOM_L4_VIRES_61)
level4_room_51_ready = _ram_ready(ROOM_L4_KEESE_KEY_51)
level4_room_50_ready = _ram_ready(ROOM_L4_VIRES_50)
level4_room_62_ready = _ram_ready(ROOM_L4_COMPASS_62)
level4_room_40_ready = _ram_ready(ROOM_L4_ZOLS_40)
level4_room_61_cleared = _cleared(ROOM_L4_VIRES_61, "ROOM_61_SPEC", settle=20)
level4_room_51_key_success = _cleared(
    ROOM_L4_KEESE_KEY_51, "ROOM_51_SPEC", settle=20, keys=1
)
level4_room_50_cleared = _cleared(ROOM_L4_VIRES_50, "ROOM_50_SPEC", settle=20)
level4_room_62_cleared = _cleared(ROOM_L4_COMPASS_62, "ROOM_62_SPEC", settle=20)
level4_room_40_cleared = _cleared(ROOM_L4_ZOLS_40, "ROOM_40_SPEC")
level4_room_40_key_success = _cleared(ROOM_L4_ZOLS_40, "ROOM_40_SPEC", keys=1)


def level4_compass_collected(ram: np.ndarray) -> bool:
    """L4 compass inventory bit set (ADDR_COMPASS & 0x08)."""
    return bool(read_snapshot(ram).compass & LEVEL4_COMPASS_BIT)


def level4_compass_route_success(ram: np.ndarray) -> bool:
    """Compass bit set and back on 0x61 play-ready (maze return complete)."""
    snap = read_snapshot(ram)
    return bool(snap.compass & LEVEL4_COMPASS_BIT) and level4_room_ready(
        snap, ROOM_L4_VIRES_61
    )


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
# Free RIGHT of cleared 0x31 → 0x32 (rr-resv live; doors open R on clear).
ROOM_L4_EAST_32 = 0x32
# Invuln movers on 0x30/0x32 (slots 1–2); never count as combat clear targets.
INVULN_MOVER_TYPE = _ids.INVULN_MOVER_OBJECT_TYPE
# Pushable block residual object on 0x32 (rr-tib8); not a combat target.
BLOCK_OBJECT_TYPE = 0x68
# Key-east door 0x30 → 0x31 (live: y≈141 hold RIGHT; keys 1→0).
KEY_30_EAST_Y = 141
KEY_30_EAST_Y_TOL = 4

MAZE_31_EAST_X_MIN = 200
MAZE_31_EAST_Y = 136
MAZE_31_EAST_Y_TOL = 16
# 0x32: after clear, push left block LEFT @y≈141 then stairs → 0x60 (rr-tib8).
PUSH_32_STAND = (120, 141)
PUSH_32_DIR = "LEFT"
# After push, walk to NE band then UP into stairs hole (live dual-green).
STAIRS_32_APPROACH = (208, 96)

LADDER_60_PICKUP_XY = (136, 141)

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
        # KEY-RIGHT @y141 → 0x31 after clear (rr-n1wn).
    ),
    max_frames=20000,
    level=LEVEL4,
)

# 0x31: 5× Vire maze (rr-resv). Enter west ~(16,141). Clear opens RIGHT door
# (cur_opened_doors 2→3). Free N/W sealed from interior; free RIGHT → 0x32.
# Maze walkable is non-rectangular (BFS ~79 cells); use hold6 path to east band.
ROOM_31_SPEC = DungeonRoomSpec(
    spec_id="level4_room31_vires",
    source_room=ROOM_L4_NORTH_30,
    room_id=ROOM_L4_EAST_31,
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
        # No contact_backstep: 0x31 maze starves kills (rr-gjey tried).
        occupancy_patrol=True,
        # West-door leftover ~(16,141) sits outside default xmin=40.
        occupancy_bounds=(16, 216, 77, 205),
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("RIGHT", ((200, 141), (224, 141))),  # free after clear → 0x32
    ),
    max_frames=25000,
    level=LEVEL4,
)

# 0x32: 2× Zol + 2× LikeLike (rr-tib8). Enter west ~(16,141).
# Ignore invuln 0x2b + block residual 0x68. Clear enables left-block push → stairs
# mode-9 0x60 Stepladder (ADDR_LADDER). Free LEFT → 0x31; free N/E/W sealed.
_PATROL_32: tuple[tuple[int, int], ...] = (
    (48, 109),
    (96, 109),
    (144, 109),
    (192, 109),
    (192, 141),
    (144, 141),
    (96, 141),
    (48, 141),
    (48, 173),
    (96, 173),
    (144, 173),
    (192, 173),
    (120, 125),
    (80, 157),
    (160, 157),
    (120, 189),
)

ROOM_32_SPEC = DungeonRoomSpec(
    spec_id="level4_room32_zol_likelike",
    source_room=ROOM_L4_EAST_31,
    room_id=ROOM_L4_EAST_32,
    entry=DoorRoute("RIGHT", ((16, 141), (48, 141))),
    enemy_types=(ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, LIKE_LIKE_OBJECT_TYPE),
    expected_enemy_count=4,  # 2 Zol + 2 LikeLike; gels mid-fight
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(GEL_SPLIT_OBJECT_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_32,
        engage_distance=56,
        attack_phase=4,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    # Invuln 0x2b + block 0x68 keep RoomAllDead noisy — clear = no live Zol/gel/LL.
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("LEFT", ((40, 141), (16, 141))),  # free return → 0x31
    ),
    max_frames=25000,
    level=LEVEL4,
)

# 0x12: 5× Vire + push block 0x68 (rr-rvae). Enter west ~(16,141) from 0x11.
# After clear doors raw often 2 (L only / UP free via ODM). Push block LEFT
# (96,144)→(80,144) opens R bit (doors 3). Scripted PATH_12_TO_GLEEOK → 0x13.
ROOM_12_SPEC = DungeonRoomSpec(
    spec_id="level4_room12_vires",
    source_room=ROOM_L4_MID_11,
    room_id=ROOM_L4_VIRES_12,
    entry=DoorRoute("LEFT", ((16, 141), (48, 141))),
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
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("LEFT", ((40, 141), (16, 141))),
        DoorRoute("UP", ((120, 93), (120, 72))),
        # RIGHT after push block → 0x13 Gleeok (maze path, not free corridor).
    ),
    max_frames=25000,
    level=LEVEL4,
)

# 0x20: 5× Vire on H-water (PNG leftover). Walkable south gold y=192–204;
# ignore invuln 0x2b. Clear from south band then east x=208 (no state-BFS).
_PATROL_20: tuple[tuple[int, int], ...] = (
    (48, 197),
    (88, 197),
    (120, 197),
    (160, 197),
    (200, 197),
)

ROOM_20_SPEC = DungeonRoomSpec(
    spec_id="level4_room20_vires",
    source_room=ROOM_L4_NORTH_30,
    room_id=ROOM_L4_WATER_NORTH_20,
    entry=DoorRoute("UP", ((120, 205), (120, 192))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_20,
        engage_distance=64,
        engage_dominant_axis=True,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("RIGHT", ((208, 141), (224, 141))),
    ),
    max_frames=20000,
    level=LEVEL4,
)

# 0x21: 5× Gel 0x15 + RoomItemId 0x17 map. West leftover (16,141). Dark;
# do not grant candle. Ignore invuln 0x2b + block 0x68. Isolated BFS banned.
_PATROL_21: tuple[tuple[int, int], ...] = (
    (48, 141),
    (120, 141),
    (208, 141),
    (208, 181),
    (120, 181),
    (48, 181),
)

ROOM_21_SPEC = DungeonRoomSpec(
    spec_id="level4_room21_gels_map",
    source_room=ROOM_L4_WATER_NORTH_20,
    room_id=ROOM_L4_MAP_21,
    entry=DoorRoute("RIGHT", ((16, 141), (48, 141))),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_21,
        engage_distance=48,
        attack_phase=4,
        engage_attack_period=4,
        engage_attack_hold=2,
        occupancy_patrol=True,
        occupancy_bounds=(16, 216, 77, 205),
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_MAP,
    exit_routes=(DoorRoute("LEFT", ((48, 141), (16, 141))),),
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
register_room_spec(ROOM_31_SPEC)
register_room_spec(ROOM_32_SPEC)
register_room_spec(ROOM_20_SPEC)
register_room_spec(ROOM_21_SPEC)
register_room_spec(ROOM_12_SPEC)


level4_room_30_ready = _ram_ready(ROOM_L4_NORTH_30)
level4_room_31_ready = _ram_ready(ROOM_L4_EAST_31)
level4_room_32_ready = _ram_ready(ROOM_L4_EAST_32)
level4_room_12_ready = _ram_ready(ROOM_L4_VIRES_12)
level4_room_30_cleared = _cleared(ROOM_L4_NORTH_30, "ROOM_30_SPEC")
level4_room_31_cleared = _cleared(ROOM_L4_EAST_31, "ROOM_31_SPEC")
level4_room_32_cleared = _cleared(ROOM_L4_EAST_32, "ROOM_32_SPEC")
level4_room_12_cleared = _cleared(ROOM_L4_VIRES_12, "ROOM_12_SPEC")


def level4_room_12_right_open(ram: np.ndarray) -> bool:
    """0x12 cleared and RIGHT door bit set after block push (doors & 0x01)."""
    return level4_room_12_cleared(ram) and bool(
        read_snapshot(ram).cur_opened_doors & 0x01
    )


def level4_gleeok_enter_success(ram: np.ndarray) -> bool:
    """Play-ready on Gleeok room 0x13 (boss may still be alive)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL4
        and snap.screen == ROOM_L4_GLEEOK_13
        and snap.mode in (PLAY_MODE, 5)
        and not snap.transitioning
    )


def level4_triforce_stop(snap: ZeldaSnapshot) -> bool:
    """Inventory fact: ADDR_TRIFORCE & 0x08 (not a route success claim alone)."""
    return bool(snap.triforce & LEVEL4_TRIFORCE_BIT)


def _ladder_on(ram: np.ndarray) -> bool:
    from zelda_i.ram import ADDR_LADDER, read_u8

    return int(read_u8(ram, ADDR_LADDER)) > 0


def level4_stepladder_success(ram: np.ndarray) -> bool:
    """ADDR_LADDER inventory bit set (stepladder collected)."""
    return _ladder_on(ram)


def level4_post_ladder_success(ram: np.ndarray) -> bool:
    """On 0x32 play with ADDR_LADDER set (exited stepladder basement)."""
    snap = read_snapshot(ram)
    return (
        _ladder_on(ram)
        and snap.level == LEVEL4
        and snap.screen == ROOM_L4_EAST_32
        and snap.mode in (PLAY_MODE, 5)
    )


def level4_west_31_success(ram: np.ndarray) -> bool:
    """On 0x31 play with ADDR_LADDER (post-ladder backtrack west of 0x32)."""
    snap = read_snapshot(ram)
    return (
        _ladder_on(ram)
        and snap.level == LEVEL4
        and snap.screen == ROOM_L4_EAST_31
        and snap.mode in (PLAY_MODE, 5)
    )


KEY_30_NORTH_X = 120
RIGHT_20_STAND = (208, 141)
MAP_21_PICKUP_XY = (208, 181)


def level4_map_success(ram: np.ndarray) -> bool:
    """ADDR_MAP bit 0x08 set (L4 dungeon map collected)."""
    from zelda_i.ram import ADDR_MAP, read_u8

    return bool(int(read_u8(ram, ADDR_MAP)) & LEVEL4_MAP_BIT)


def level4_map_room_success(ram: np.ndarray) -> bool:
    """Map bit set and play-ready on 0x21 (map room)."""
    snap = read_snapshot(ram)
    return (
        level4_map_success(ram)
        and snap.level == LEVEL4
        and snap.screen == ROOM_L4_MAP_21
        and snap.mode in (PLAY_MODE, 5)
    )


_IMPORT_NAMES = frozenset({
    "AliveRule", "CombatTuning", "DoorRoute", "DungeonRoomSpec",
    "KEESE_OBJECT_TYPE", "LEVEL4_ENTRY_ROOM", "PLAY_MODE", "RewardKind",
    "RewardSpec", "ZeldaSnapshot", "annotations", "np", "read_snapshot",
    "register_room_spec",
})


__all__ = sorted(
    n
    for n in globals()
    if not n.startswith("_") and n not in _IMPORT_NAMES
)