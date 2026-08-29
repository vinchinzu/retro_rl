"""Level 2 (Moon) dungeon room specs and stop predicates.

Uses ``dungeon`` engine helpers only. Specs register on import.
Live recon 2026-08-06 — see LEVEL2_ROUTE.md.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon import (
    AliveRule,
    BLUE_GORIYA_OBJECT_TYPE,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
    MOLDORM_OBJECT_TYPE,
    GORIYA_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    ROPE_OBJECT_TYPE,
    dungeon_room_cleared,
    inventory_reward_success,
    register_room_spec,
)
from zelda_i.ram import (
    ADDR_MAGIC_BOOMERANG,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

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
ROOM_L2_ROPES_NORTH = 0x4E  # N of Goriya free-UP; 5× Rope + key (rr-cjf)
ROOM_L2_BOOM_CANDIDATE = 0x4F  # bomb-N of 0x5f / east of 0x4e; RoomItemId 0x1e
ROOM_L2_NORTH_OF_4E = 0x3E  # free UP from 0x4e; also Moldorm via 0x3f LEFT
ROOM_L2_TRAPS_KEESE = 0x3F  # bomb-N of boom 0x4f; 4× Keese + 4× traps 0x49
ROOM_L2_GELS_NORTH = 0x2F  # bomb-N of 0x3f; 5× Gel; item 0x0f
ROOM_L2_ROPES_UNLOCK = 0x2E  # N of Moldorm / W of 0x2f; 8× Rope kill→UP
ROOM_L2_GORIYA_BOMBS = 0x1E  # N of 0x2e; 5× Goriya 0x06; bomb-N→Dodongo (not walk-UP)
ROOM_L2_DODONGO = 0x0E  # boss; type 0x32; bomb-N of 0x1e @(120,101) LIVE (rr-n5i)
ROOM_L2_WEST_OF_BOSS = 0x0D  # LEFT of 0x0e after kill; north-corridor only; TF residual
ROOM_L2_OLD_MAN = 0x1F  # N of 0x2f / bomb-R of 0x1e; bubbles + NPC 0x4b
# Boss kill: bomb-mouth; HC raises heart_containers; doors often LEFT=2 only.
# Walkthrough "E of boss → TF" not live yet (RIGHT sealed key/bomb/push).
ROOM_6D_LEFT_DOOR_BIT = 0x02  # cur_opened_doors bit1 after clear
# Magical Boomerang: ADDR_MAGIC_BOOMERANG stop via level2_room_4f_magic_boomerang_success.
# Room 0x4f RoomItemId 0x1e pure 2/2 Clean (rr-bsq/rr-ebe); L1 wooden was 0x1D.
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
    # y-first: 6d combat parks on the y=109 statue band (live (48, 109));
    # x-first RIGHT never reaches the west door at y=141.
    entry=DoorRoute("LEFT", ((120, 141), (32, 141)), y_first=True),
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

# West of 0x5f via key-LEFT: 5× Goriya 0x06 (TYPE_AND_HP; spawn HP≈48).
# Entry from Level2_5F: align y≈141 mid-x then pure LEFT (consumes 1 key).
# Clear-only; RoomItemId 0x03; doors bit often RIGHT=0x01 after key entry.
# Live recon: l2_5f_explore / Level2_5F keys≥1 (rr-etl).
_ROOM_5E_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (112, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (112, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)

ROOM_5E_SPEC = DungeonRoomSpec(
    spec_id="level2_room5e_goriya",
    source_room=ROOM_L2_BOMB_N,
    room_id=ROOM_L2_GORIYA_WEST,
    entry=DoorRoute(
        "LEFT",
        ((120, 141), (32, 141)),
    ),
    enemy_types=(GORIYA_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_5E_PATROL,
        engage_distance=72,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=16000,
    level=LEVEL_2,
)

# North of Goriya 0x5e (free UP): 5× Rope + fixed key RoomItemId 0x19.
# Live recon rr-cjf: entry y≈189; RIGHT (key) → 0x4f boom; UP → 0x3e residual.
_ROOM_4E_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (112, 109),
    (160, 109),
    (160, 141),
    (160, 173),
    (112, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)

ROOM_4E_SPEC = DungeonRoomSpec(
    spec_id="level2_room4e_ropes_key",
    source_room=ROOM_L2_GORIYA_WEST,
    room_id=ROOM_L2_ROPES_NORTH,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_4E_PATROL,
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
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=10000,
    level=LEVEL_2,
)

# Boom room 0x4f: 3× type 0x05 (blue Goriya residual, HP≈80) + fireballs 0x55
# (not clear targets). RoomItemId 0x1e Magical Boomerang.
# Paths: 0x5f bomb N @ (120,101) → entry ~(120,189); or 0x4e key-RIGHT → ~(16,141).
# Pickup after kill: dense probe ~(136, 135); also mid-combat near center.
# Stop: ADDR_MAGIC_BOOMERANG != 0 (snapshot.magical_boomerang). rr-bsq/rr-ebe.
_ROOM_4F_PATROL: tuple[tuple[int, int], ...] = (
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

ROOM_4F_SPEC = DungeonRoomSpec(
    spec_id="level2_room4f_magic_boomerang",
    source_room=ROOM_L2_BOMB_N,
    room_id=ROOM_L2_BOOM_CANDIDATE,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(BLUE_GORIYA_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_4F_PATROL,
        engage_distance=80,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="magical_boomerang",
        # Live dense pickup ~(136,135); cover center band through item.
        target=(136, 135),
        waypoints=(
            (120, 141),
            (136, 141),
            (136, 135),
            (128, 125),
            (144, 125),
            (120, 157),
            (136, 135),
        ),
    ),
    room_item_id=0x1E,
    exit_routes=(
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        # bomb-UP @(120,101) → 0x3f (not walk-UP); see BOMB_WALL_4F_NORTH
    ),
    max_frames=20000,
    level=LEVEL_2,
)

# --- Post-boom Dodongo path (rr-n5i, assisted recon 2026-08-06) ---
# 0x4f bomb N @(120,101) → 0x3f traps+Keese → LEFT Moldorm 0x3e → UP ropes 0x2e
# → UP Goriya 0x1e (doors UP|DOWN=12 after clear; physical UP to boss residual).
# Alt: 0x3f bomb N → 0x2f gels → LEFT 0x2e; 0x2f UP → 0x1f old man.

_ROOM_3F_PATROL: tuple[tuple[int, int], ...] = (
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

ROOM_3F_SPEC = DungeonRoomSpec(
    spec_id="level2_room3f_traps_keese",
    source_room=ROOM_L2_BOOM_CANDIDATE,
    room_id=ROOM_L2_TRAPS_KEESE,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=4,
    alive_rule=AliveRule.TYPE,  # Keese TYPE-only (hp=0 alive)
    combat=CombatTuning(
        patrol=_ROOM_3F_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x00,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        # bomb-UP @(120,101) → 0x2f
    ),
    max_frames=12000,
    level=LEVEL_2,
)

_ROOM_3E_MOLDORM_PATROL: tuple[tuple[int, int], ...] = (
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

ROOM_3E_MOLDORM_SPEC = DungeonRoomSpec(
    spec_id="level2_room3e_moldorm_key",
    source_room=ROOM_L2_TRAPS_KEESE,
    room_id=ROOM_L2_NORTH_OF_4E,
    entry=DoorRoute("LEFT", ((224, 141),)),
    enemy_types=(MOLDORM_OBJECT_TYPE,),
    expected_enemy_count=10,  # multi-segment; TYPE clear collapses chain
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_3E_MOLDORM_PATROL,
        engage_distance=80,
        attack_phase=0,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(120, 141),
        waypoints=((120, 141), (136, 125), (104, 125), (120, 157), (120, 141)),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=16000,
    level=LEVEL_2,
)

_ROOM_2E_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (112, 109),
    (160, 109),
    (160, 141),
    (160, 173),
    (112, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)

ROOM_2E_SPEC = DungeonRoomSpec(
    spec_id="level2_room2e_ropes_unlock",
    source_room=ROOM_L2_NORTH_OF_4E,
    room_id=ROOM_L2_ROPES_UNLOCK,
    entry=DoorRoute("UP", ((120, 205),)),
    enemy_types=(ROPE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_2E_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),  # may need clear bit
    ),
    max_frames=16000,
    level=LEVEL_2,
)

_ROOM_1E_PATROL: tuple[tuple[int, int], ...] = (
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

ROOM_1E_SPEC = DungeonRoomSpec(
    spec_id="level2_room1e_goriya_bombs",
    source_room=ROOM_L2_ROPES_UNLOCK,
    room_id=ROOM_L2_GORIYA_BOMBS,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GORIYA_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_1E_PATROL,
        engage_distance=72,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x00,
    exit_routes=(
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        # Walk-UP after clear: doors UP|DOWN=12 but physical solid (min_y≈117).
        # Boss open is bomb-N @(120,101) → 0x0e (LIVE rr-n5i 2026-08-07).
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=20000,
    level=LEVEL_2,
)

_ROOM_2F_PATROL: tuple[tuple[int, int], ...] = (
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

ROOM_2F_SPEC = DungeonRoomSpec(
    spec_id="level2_room2f_gels",
    source_room=ROOM_L2_TRAPS_KEESE,
    room_id=ROOM_L2_GELS_NORTH,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_2F_PATROL,
        engage_distance=56,
        attack_phase=0,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x0F,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=10000,
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


def level2_room_5f_ready(ram: np.ndarray) -> bool:
    """Isolated pure: Level 2 room 0x5f play-ready (bomb-north of compass).

    Stop for ``Level2BombNorthController`` / ``run_level2_bomb_north.py``.
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_2
        and snap.screen == ROOM_L2_BOMB_N
        and snap.mode == PLAY_MODE
    )


def level2_room_5e_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x5e 5× Goriya dead, RoomAllDead≥20.

    Key-LEFT entry from 0x5f (cost 1 key). Stop is CLEAR_ONLY — zero live
    Goriya type 0x06 with TYPE_AND_HP and settle counter; no inventory gate.
    """
    return dungeon_room_cleared(ram, ROOM_5E_SPEC)


def level2_room_4e_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x4e ropes dead and keys increased (fixed key 0x19).

    Free UP from 0x5e. FIXED_INVENTORY stop: inventory + TYPE_AND_HP liveness.
    """
    return inventory_reward_success(ram, ROOM_4E_SPEC, min_value=1)


def level2_room_4f_ready(ram: np.ndarray) -> bool:
    """Play-ready on L2 boom room 0x4f (after 0x5f bomb-north or 0x4e RIGHT)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_2
        and snap.screen == ROOM_L2_BOOM_CANDIDATE
        and snap.mode == PLAY_MODE
    )


def level2_room_4f_magic_boomerang_success(ram: np.ndarray) -> bool:
    """Isolated pure: Magical Boomerang inventory on 0x4f (or any screen).

    Primary stop for ``run_level2_magic_boomerang``: ``ADDR_MAGIC_BOOMERANG
    != 0``. Does not require still standing in 0x4f (pickup may finish mid-
    scroll). Enemies need not be fully settled — item can register mid-clear.
    """
    return read_u8(ram, ADDR_MAGIC_BOOMERANG) != 0

def level2_room_3f_ready(ram: np.ndarray) -> bool:
    """Play-ready on L2 traps+Keese room 0x3f (bomb-north of boom 0x4f)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_2
        and snap.screen == ROOM_L2_TRAPS_KEESE
        and snap.mode == PLAY_MODE
    )


def level2_room_3e_moldorm_key_success(ram: np.ndarray) -> bool:
    """Moldorm dead and keys increased (fixed key 0x19 on 0x3e)."""
    return inventory_reward_success(ram, ROOM_3E_MOLDORM_SPEC, min_value=1)


def level2_room_2e_cleared(ram: np.ndarray) -> bool:
    """0x2e 8× Rope dead (kill opens UP toward Goriya 0x1e)."""
    return dungeon_room_cleared(ram, ROOM_2E_SPEC)


def level2_room_1e_cleared(ram: np.ndarray) -> bool:
    """0x1e 5× Goriya dead (doors often UP|DOWN=12; physical UP residual)."""
    return dungeon_room_cleared(ram, ROOM_1E_SPEC)


def level2_triforce_bit_02(ram: np.ndarray) -> bool:
    """Moon triforce shard collected (ADDR_TRIFORCE & 0x02)."""
    from zelda_i.ram import ADDR_TRIFORCE

    return (read_u8(ram, ADDR_TRIFORCE) & 0x02) != 0



# Register all room specs once (path controllers live in level2_bomb_path).
for _spec in (
    ROOM_7D_SPEC,
    ROOM_6D_SPEC,
    ROOM_6C_SPEC,
    ROOM_7E_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    ROOM_5E_SPEC,
    ROOM_4E_SPEC,
    ROOM_4F_SPEC,
    ROOM_3F_SPEC,
    ROOM_3E_MOLDORM_SPEC,
    ROOM_2E_SPEC,
    ROOM_1E_SPEC,
    ROOM_2F_SPEC,
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
    "ROOM_L2_ROPES_NORTH",
    "ROOM_L2_BOOM_CANDIDATE",
    "ROOM_L2_NORTH_OF_4E",
    "ROOM_L2_TRAPS_KEESE",
    "ROOM_L2_GELS_NORTH",
    "ROOM_L2_ROPES_UNLOCK",
    "ROOM_L2_GORIYA_BOMBS",
    "ROOM_L2_DODONGO",
    "ROOM_L2_WEST_OF_BOSS",
    "ROOM_L2_OLD_MAN",
    "ROOM_6D_LEFT_DOOR_BIT",
    "ROOM_7D_SPEC",
    "ROOM_6D_SPEC",
    "ROOM_6C_SPEC",
    "ROOM_7E_SPEC",
    "ROOM_6E_SPEC",
    "ROOM_6F_SPEC",
    "ROOM_5E_SPEC",
    "ROOM_4E_SPEC",
    "ROOM_4F_SPEC",
    "ROOM_3F_SPEC",
    "ROOM_3E_MOLDORM_SPEC",
    "ROOM_2E_SPEC",
    "ROOM_1E_SPEC",
    "ROOM_2F_SPEC",
    "level2_room_6d_cleared",
    "level2_room_6c_key_success",
    "level2_room_7e_key_success",
    "level2_room_6e_cleared",
    "level2_room_6f_compass_success",
    "level2_room_5f_ready",
    "level2_room_5e_cleared",
    "level2_room_4e_key_success",
    "level2_room_4f_ready",
    "level2_room_4f_magic_boomerang_success",
    "level2_room_3f_ready",
    "level2_room_3e_moldorm_key_success",
    "level2_room_2e_cleared",
    "level2_room_1e_cleared",
    "level2_triforce_bit_02",
]
