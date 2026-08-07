"""Level 2 (Moon) dungeon room specs and stop predicates.

Uses ``dungeon`` engine helpers only. Specs register on import.
Live recon 2026-08-06 — see LEVEL2_ROUTE.md.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    ROPE_OBJECT_TYPE,
    dungeon_room_cleared,
    inventory_reward_success,
    register_room_spec,
)
from zelda_i.ram import PLAY_MODE, read_snapshot

# Level 2 (Moon) room IDs — live recon 2026-08-06 (see LEVEL2_ROUTE.md).
LEVEL_2 = 2
ROOM_L2_ENTRY = 0x7D
ROOM_L2_ROPES = 0x6D
ROOM_L2_WEST_KEY = 0x6C
ROOM_L2_EAST_KEY = 0x7E  # 5× Rope + key 0x19 (entry-east; diamond-nav from 0x7d)
ROOM_L2_EAST_OF_ROPES = 0x6E  # 3× Rope; also N of 0x7e; key-RIGHT → 0x6f
ROOM_L2_COMPASS = 0x6F  # 6× Gel 0x15 (TYPE-only) + compass RoomItemId 0x16
ROOM_L2_BOMB_N = 0x5F  # N of compass via bomb wall @ (120,101); map RoomItemId 0x17 seen
ROOM_L2_GORIYA_WEST = 0x5E  # W of 0x5f via key-LEFT; Goriya 0x06
ROOM_6D_LEFT_DOOR_BIT = 0x02  # cur_opened_doors bit1 after clear
# Magical Boomerang inventory (Data Crystal + live zero-check on L2 states).
# Stop predicate for future pure item room: read_u8(ram, ADDR_MAGIC_BOOMERANG) != 0
# RoomItemId for boomerang drops correlates to 0x1D (L1 wooden boom room).
# Diamond-east (0x7d / 0x6e): band→wall→S2(LEFT×6,vert,RIGHT×10)→pure y=141 RIGHT.
# Door y poke: east wall opens only for y≥137 (y≤133 never). See LEVEL2_ROUTE.
# 0x6f bomb N: stand (120,101) UP+B → 0x5f. See nav_common.diamond_east_phase;
# bands DIAMOND_BAND_7D=157, DIAMOND_BAND_6E=113.

# Level 2 ropes room: open lanes; engage nearest after ~100f spawn settle.
_ROOM_6D_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (112, 109),
    (160, 109),
    (160, 141),
    (160, 173),
    (112, 173),
    (64, 173),
    (64, 141),
)

# --- Level 2 (Moon) rooms (isolated pure from Level2Entrance / Level2RopesCleared)
# Entry 0x7d: no combat types at room-ready; north doorway open without door bit.
# Ropes 0x6d: 5× type 0x28; HP activates ~mode-5 settle; clear sets LEFT bit 0x02.
# Lab (Clean, 12/12): attack_phase=4, engage=64, median ~674f — lab_l2_6d/.
ROOM_7D_SPEC = DungeonRoomSpec(
    spec_id="level2_room7d_entry",
    source_room=ROOM_L2_ENTRY,
    room_id=ROOM_L2_ENTRY,
    entry=DoorRoute("UP", ((120, 205), (120, 93))),
    enemy_types=(),
    expected_enemy_count=0,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(patrol=((120, 141),)),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=2000,
    level=LEVEL_2,
)

ROOM_6D_SPEC = DungeonRoomSpec(
    spec_id="level2_room6d_ropes",
    source_room=ROOM_L2_ENTRY,
    room_id=ROOM_L2_ROPES,
    entry=DoorRoute(
        "UP",
        ((120, 205), (120, 93)),
    ),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_6D_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    required_open_doors=ROOM_6D_LEFT_DOOR_BIT,
    exit_routes=(
        DoorRoute("DOWN", ((120, 189),)),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=6000,
    level=LEVEL_2,
)

# West of 0x6d: 6 Ropes + fixed RoomItemId small key (0x19).
# Key pickup observed near (136, 141) during combat (keys 0→1, 1 rope left).
# Lab clear-only (Clean): phase 2/4 engage 64 → 2/2; phase 0 times out.
ROOM_6C_SPEC = DungeonRoomSpec(
    spec_id="level2_room6c_west_key",
    source_room=ROOM_L2_ROPES,
    room_id=ROOM_L2_WEST_KEY,
    entry=DoorRoute("LEFT", ((120, 141), (32, 141))),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_6D_PATROL,
        engage_distance=64,
        attack_phase=2,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=8000,
    level=LEVEL_2,
)

# East of entry 0x7d: 5 Ropes + fixed RoomItemId small key (0x19).
# Diamond solids block naive y≈141 RIGHT at x≈128. Entry: skirt south band
# y≈157 → wall x≥200 → align y≈141 → RIGHT (see LEVEL2_ROUTE / l2_7d_east_nav).
# Combat seeds from 0x6d 5-rope policy (phase 4, engage 64); key like 0x6c.
ROOM_7E_SPEC = DungeonRoomSpec(
    spec_id="level2_room7e_east_key",
    source_room=ROOM_L2_ENTRY,
    room_id=ROOM_L2_EAST_KEY,
    entry=DoorRoute(
        "RIGHT",
        ((120, 157), (208, 157), (208, 141)),
    ),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_6D_PATROL,
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
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=8000,
    level=LEVEL_2,
)

# East of 0x6d / north of 0x7e: 3 Ropes; key door RIGHT → 0x6f.
# Prefer WEST entry (from 0x6d); south from 0x7e can stick ~y=181.
ROOM_6E_SPEC = DungeonRoomSpec(
    spec_id="level2_room6e_ropes",
    source_room=ROOM_L2_ROPES,
    room_id=ROOM_L2_EAST_OF_ROPES,
    entry=DoorRoute("RIGHT", ((120, 141), (208, 141))),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=((96, 125), (160, 125), (160, 157), (96, 157), (128, 141)),
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        # Key door: wall-first band y≈113 → vertical at x≥200 → pure RIGHT.
        DoorRoute("RIGHT", ((120, 113), (208, 113), (208, 141))),
    ),
    max_frames=8000,
    level=LEVEL_2,
)

# Compass branch east of 0x6e: 6× Gel 0x15 (TYPE-only; hp=0 while alive) +
# RoomItemId 0x16. Compass sits on the **east wall** ~ (200, 101); inventory is
# ADDR_COMPASS bitfield (level 2 → bit1 → value 2). Entry: WEST key-door from
# 0x6e with wall-vertical-push (not door_y LEFT — that re-enters diamond).
_ROOM_6F_PATROL: tuple[tuple[int, int], ...] = (
    (72, 109),
    (120, 109),
    (168, 109),
    (168, 141),
    (168, 173),
    (120, 173),
    (72, 173),
    (72, 141),
    (120, 141),
)

ROOM_6F_SPEC = DungeonRoomSpec(
    spec_id="level2_room6f_compass",
    source_room=ROOM_L2_EAST_OF_ROPES,
    room_id=ROOM_L2_COMPASS,
    entry=DoorRoute(
        "RIGHT",
        ((120, 113), (208, 113), (208, 141)),
    ),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_6F_PATROL,
        engage_distance=56,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="compass",
        # East-wall compass ~ (200,101). Target past the item so the
        # ±5 collect deadband still walks through the sprite (195→208).
        target=(208, 101),
        waypoints=((192, 101), (208, 101), (200, 109), (200, 93), (208, 101)),
    ),
    room_item_id=0x16,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=12000,
    level=LEVEL_2,
)


def level2_room_6d_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x6d 5 Ropes dead, RoomAllDead≥20, left door bit 0x02."""
    return dungeon_room_cleared(ram, ROOM_6D_SPEC)


def level2_room_6c_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x6c with keys≥1 and no live Ropes.

    ``RoomAllDead`` can lag several dozen frames after the last kill (observed
    0→14→43 while idling post-success). Inventory + type/HP liveness is the
    reliable stop; do not require the clear counter for FIXED_INVENTORY rooms.
    """
    return inventory_reward_success(ram, ROOM_6C_SPEC, min_value=1)


def level2_room_7e_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x7e east key room with keys≥1 and no live Ropes.

    Same FIXED_INVENTORY stop as west key: inventory + liveness only. From
    ``Level2Entrance`` (keys=0) this is keys≥1; after west key, controller
    success uses inventory delta while this stop still holds for keys≥1.
    """
    return inventory_reward_success(ram, ROOM_7E_SPEC, min_value=1)


def level2_room_6e_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x6e 3 Ropes dead, RoomAllDead≥20."""
    return dungeon_room_cleared(ram, ROOM_6E_SPEC)


def level2_room_6f_compass_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x6f gels dead and L2 compass bit set.

    ``ADDR_COMPASS`` is one bit per dungeon level (Data Crystal); level 2 sets
    bit 1 (value 2). Stop uses inventory + TYPE-only gel liveness — do not
    require RoomAllDead lag after FIXED_INVENTORY pickup.
    """
    snap = read_snapshot(ram)
    level_bit = 1 << (LEVEL_2 - 1)
    return (
        snap.level == LEVEL_2
        and snap.screen == ROOM_L2_COMPASS
        and snap.mode == PLAY_MODE
        and (snap.compass & level_bit) != 0
        and not ROOM_6F_SPEC.live_enemies(snap)
    )


for _spec in (
    ROOM_7D_SPEC,
    ROOM_6D_SPEC,
    ROOM_6C_SPEC,
    ROOM_7E_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
):
    register_room_spec(_spec)

__all__ = [
    "LEVEL_2",
    "ROOM_L2_ENTRY",
    "ROOM_L2_ROPES",
    "ROOM_L2_WEST_KEY",
    "ROOM_L2_EAST_KEY",
    "ROOM_L2_EAST_OF_ROPES",
    "ROOM_L2_COMPASS",
    "ROOM_L2_BOMB_N",
    "ROOM_L2_GORIYA_WEST",
    "ROOM_6D_LEFT_DOOR_BIT",
    "ROOM_7D_SPEC",
    "ROOM_6D_SPEC",
    "ROOM_6C_SPEC",
    "ROOM_7E_SPEC",
    "ROOM_6E_SPEC",
    "ROOM_6F_SPEC",
    "level2_room_6d_cleared",
    "level2_room_6c_key_success",
    "level2_room_7e_key_success",
    "level2_room_6e_cleared",
    "level2_room_6f_compass_success",
]
