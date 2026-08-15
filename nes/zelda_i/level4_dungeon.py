"""Level 4 (Snake) dungeon room specs, stop predicates, and live anchors.

Path controllers and maze timing live in ``level4_path``, ``level4_maze_path``,
and ``level4_stepladder``. ``__getattr__`` keeps ``from zelda_i.level4_dungeon
import make_*`` working.

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
    **Post-compass expand (rr-xc3x / rr-q8eq / rr-n1wn / rr-resv / rr-tib8 live):**
    0x50 is **not** a dead-end. Scripted north → **0x40** (5× Zol ``0x13`` → gel
    ``0x14`` + key ``0x19`` via east-corridor path). Free UP → **0x30**
    (3× Vire + 2× invuln ``0x2b``; ignore invuln for clear). KEY-RIGHT →
    **0x31** (5× Vire; maze interior). Clear opens RIGHT door (doors 2→3) →
    free RIGHT → **0x32** (2× Zol + 2× LikeLike). Clear + push left block →
    stairs **0x60** mode-9 → **ADDR_LADDER** (RoomItemId ``0x0d``). First rooms
    outside early component {0x71, 0x61, 0x51, 0x50, 0x62}. 0x51 UP/RIGHT sealed;
    0x40 L/R sealed.

**Post-ladder (rr-05fz / rr-rvae live):** after ``ADDR_LADDER``, pedestal freezes
    ~100f; clear 4× Keese, then hold4 BFS exits mode-9 **0x60 → 0x32** play. From
    ``Level4PostLadder`` free LEFT (BFS around pushed block) → **0x31**.
    Backtrack **0x31→0x30**; with ladder + **key**, KEY-UP **0x30→0x20** (5× Vire);
    clear 0x20 → free/push RIGHT **0x21** (5× Gel + RoomItemId ``0x17`` map).
    Gel thrash expands maze walkability; hold6 BFS → ``ADDR_MAP & 0x08`` @~(208,181).

**Gleeok approach (rr-rvae live recon 2026-08-10 from Level4Map):** maze BFS LEFT
    **0x21→0x20**; free UP **0x20→0x10** Manhandla ``0x3c``; free UP **0x10→0x00**
    bubbles ``0x40`` (dead-end). Map **BOMB_UP** stand≈(120,105) → **0x11**
    (type ``0x35`` cluster). From cleared 0x11: UP **0x01** Keese+key ``0x19``
    (natural key residual); RIGHT **0x12** 5× Vire + block ``0x68``; LEFT **0x10**.
    From 0x12: UP **0x02** blade traps ``0x49`` (dead-end). **RIGHT→0x13** Gleeok
    type ``0x43`` + HeartContainer ``0x1A`` requires **push block 0x68 LEFT**
    (doors 2→3 opens R bit) then maze hold4 path (not naive y141 hold-RIGHT).
    Dual-green enter live (rr-rvae). Gleeok melee + HC + TF ``0x08`` dual-green
    from ``Level4GleeokEnter`` (``level4_boss_combat``; UP → TF room **0x03**).

Not Clean STATUS (assisted first-pass). Natural continuous residual open.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon import (
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
from zelda_i import dungeon_ids as _ids
from zelda_i.level4_overworld import LEVEL4, LEVEL4_ENTRY_ROOM
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

register_room_spec(ROOM_71_SPEC)
register_room_spec(ROOM_61_SPEC)
register_room_spec(ROOM_51_SPEC)
register_room_spec(ROOM_50_SPEC)
register_room_spec(ROOM_62_SPEC)
register_room_spec(ROOM_40_SPEC)
register_room_spec(ROOM_30_SPEC)
register_room_spec(ROOM_31_SPEC)
register_room_spec(ROOM_32_SPEC)
register_room_spec(ROOM_12_SPEC)


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


def level4_room_31_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_EAST_31)


def level4_room_31_cleared(ram: np.ndarray) -> bool:
    """0x31 Vires+splits cleared (maze residual OK).

    RoomAllDead may lag; clear is live-Vire emptiness only (settle_all_dead=0).
    After clear, RIGHT door opens (doors bit R) for free exit → 0x32.
    """
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_EAST_31):
        return False
    live = ROOM_31_SPEC.live_enemies(snap)
    return len(live) == 0


def level4_room_32_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_EAST_32)


def level4_room_32_cleared(ram: np.ndarray) -> bool:
    """0x32 Zol+LikeLike cleared (invuln 0x2b + block 0x68 residual OK).

    RoomAllDead may lag with invuln movers; clear is live-enemy emptiness only.
    After clear: free LEFT→0x31; push left block → stairs 0x60 Stepladder.
    """
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_EAST_32):
        return False
    live = ROOM_32_SPEC.live_enemies(snap)
    return len(live) == 0


def level4_room_12_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_VIRES_12)


def level4_room_12_cleared(ram: np.ndarray) -> bool:
    """0x12 with no live Vire/split (block 0x68 ignored)."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_12):
        return False
    return len(ROOM_12_SPEC.live_enemies(snap)) == 0


def level4_room_12_right_open(ram: np.ndarray) -> bool:
    """0x12 cleared and RIGHT door bit set after block push (doors & 0x01)."""
    if not level4_room_12_cleared(ram):
        return False
    snap = read_snapshot(ram)
    return bool(snap.cur_opened_doors & 0x01)


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


def level4_stepladder_success(ram: np.ndarray) -> bool:
    """ADDR_LADDER inventory bit set (stepladder collected)."""
    from zelda_i.ram import ADDR_LADDER, read_u8

    return int(read_u8(ram, ADDR_LADDER)) > 0


def level4_post_ladder_success(ram: np.ndarray) -> bool:
    """On 0x32 play with ADDR_LADDER set (exited stepladder basement)."""
    from zelda_i.ram import ADDR_LADDER, read_u8

    snap = read_snapshot(ram)
    return (
        int(read_u8(ram, ADDR_LADDER)) > 0
        and snap.level == LEVEL4
        and snap.screen == ROOM_L4_EAST_32
        and snap.mode in (PLAY_MODE, 5)
    )


def level4_west_31_success(ram: np.ndarray) -> bool:
    """On 0x31 play with ADDR_LADDER (post-ladder backtrack west of 0x32)."""
    from zelda_i.ram import ADDR_LADDER, read_u8

    snap = read_snapshot(ram)
    return (
        int(read_u8(ram, ADDR_LADDER)) > 0
        and snap.level == LEVEL4
        and snap.screen == ROOM_L4_EAST_31
        and snap.mode in (PLAY_MODE, 5)
    )


# KEY-UP 0x30 north (ladder water cross + key door) → 0x20.
KEY_30_NORTH_X = 120
# East push out of cleared 0x20 → 0x21 (door bit R may stay 0; walk x≈208).
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


def level4_room_30_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_NORTH_30)


# Path controllers + timing knobs (canonical: level4_path / maze / stepladder).
_PATH_EXPORTS = frozenset({
    "BombWall61North", "EntryUpPhase", "KeyRight62Phase", "Left50Phase",
    "Level4EntryUpController", "Level4KeyRight62Controller",
    "Level4Left50Controller",
    "make_bomb_61_north_controller", "make_entry_up_controller",
    "make_key_right_62_controller", "make_left_50_controller",
    "make_room_12_clear_controller",
    "make_room_31_clear_controller", "make_room_32_clear_controller",
    "make_room_40_clear_controller", "make_room_50_clear_controller",
    "make_room_51_key_controller", "make_room_61_clear_controller",
    "make_room_62_clear_controller", "planning_interior_report",
})

_MAZE_EXPORTS = frozenset({
    "Compass62Phase", "KEY_40_PATH_ANCHOR", "Key40Phase",
    "Level4Compass62Controller", "Level4Key40Controller",
    "Level4North40Controller", "MAP_21_HOLD", "MAP_21_SAMPLE_PATH",
    "MAZE_40_KEY_HOLD", "MAZE_40_TO_KEY", "MAZE_50_HOLD", "MAZE_50_LONG_UP",
    "MAZE_50_TO_NORTH", "MAZE_50_WAYPOINTS", "MAZE_62_RETURN_WEST",
    "MAZE_62_TO_COMPASS", "MAZE_IN_HOLD", "MAZE_OUT_HOLD", "North40Phase",
    "PATH_12_TO_GLEEOK", "PUSH_12_HOLD", "RIGHT_12_HOLD",
    "make_compass_62_controller", "make_north_40_controller",
    "make_room_40_key_controller",
})

_STEPLADDER_EXPORTS = frozenset({
    "Clear30Phase", "EXIT_60_HOLD", "EXIT_60_SAMPLE_PATH", "KeyRight31Phase",
    "Level4Clear30Controller", "Level4KeyRight31Controller",
    "Level4North30Controller", "Level4StepladderController",
    "MAZE_31_CELL_Q", "MAZE_31_HOLD", "MAZE_60_HOLD", "MAZE_60_SETTLE",
    "MAZE_60_SPAWN_XY", "MAZE_60_TO_LADDER", "North30Phase",
    "POST_LADDER_ITEM_SETTLE", "PUSH_32_HOLD", "STAIRS_32_PUSH",
    "STAIRS_32_PUSH_FRAMES", "StepladderPhase", "WEST_31_HOLD",
    "WEST_31_SAMPLE_PATH", "make_key_right_31_controller",
    "make_north_30_controller", "make_room_30_clear_controller",
    "make_stepladder_controller",
})

_IMPORT_NAMES = frozenset({
    "AliveRule", "CombatTuning", "DoorRoute", "DungeonRoomSpec",
    "KEESE_OBJECT_TYPE", "LEVEL4_ENTRY_ROOM", "PLAY_MODE", "RewardKind",
    "RewardSpec", "ZeldaSnapshot", "annotations", "np", "read_snapshot",
    "register_room_spec",
})


def __getattr__(name: str):
    if name in _PATH_EXPORTS:
        from zelda_i import level4_path as _paths
        return getattr(_paths, name)
    if name in _MAZE_EXPORTS:
        from zelda_i import level4_maze_path as _maze
        return getattr(_maze, name)
    if name in _STEPLADDER_EXPORTS:
        from zelda_i import level4_stepladder as _step
        return getattr(_step, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(_PATH_EXPORTS)
        | set(_MAZE_EXPORTS)
        | set(_STEPLADDER_EXPORTS)
    )


__all__ = sorted(
    {
        n
        for n in globals()
        if not n.startswith("_") and n not in _IMPORT_NAMES
    }
    | _PATH_EXPORTS
    | _MAZE_EXPORTS
    | _STEPLADDER_EXPORTS
)
