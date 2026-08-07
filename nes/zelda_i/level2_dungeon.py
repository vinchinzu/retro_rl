"""Level 2 (Moon) dungeon room specs and stop predicates.

Uses ``dungeon`` engine helpers only. Specs register on import.
Live recon 2026-08-06 — see LEVEL2_ROUTE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import (
    AliveRule,
    BLUE_GORIYA_OBJECT_TYPE,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
    MOLDORM_OBJECT_TYPE,
    GORIYA_OBJECT_TYPE,
    GenericDungeonRoomController,
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
    ZeldaSnapshot,
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


# --- Bomb-north pure: 0x6f stand (120,101) UP+B → 0x5f (rr-lzk) ---

# --- Bomb-north pure: 0x6f stand (120,101) UP+B → 0x5f (rr-lzk) ---
# Recon: l2_past6f_expand.json. Level2Compass often mid-scroll + gels present
# (reload re-spawns TYPE-only gels); clear then west-band to stand. B-slot
# sel is often 0x01 with bombs already owned (places without START menu).
# Documented B-item id for bomb is 0x02 at ADDR 0x0656; no poke required when
# inventory already selects a bomb-capable B item.

BOMB_N_STAND = (120, 101)
BOMB_N_STAND_TOL = 4
BOMB_N_WAIT_BLAST = 100
BOMB_N_STEP_BACK = 6
BOMB_N_MAX_FRAMES = 16000
# Data Crystal / probe_level2_past_6f: B-item bomb select value.
B_ITEM_BOMB = 0x02
ADDR_SELECTED_ITEM = 0x0656


class BombNorthPhase(Enum):
    """Phase machine for 0x6f bomb-north → 0x5f."""

    SETTLE = auto()
    CLEAR = auto()
    TO_STAND = auto()
    FACE = auto()
    PLACE = auto()
    WAIT = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2BombNorthController:
    """From 0x6f: ensure clear/compass, bomb north wall @ (120,101), enter 0x5f.

    Clean geometry policy: no health poke. Uses natural ``snap.bombs``; fails
    cleanly if bombs==0. Does not write selected-item RAM (Level2Compass /
    post-clear states already place bombs with current B selection).
    """

    phase: BombNorthPhase = BombNorthPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_before_place: int | None = None
    bombs_after_place: int | None = None
    clear_controller: GenericDungeonRoomController | None = None
    max_frames: int = BOMB_N_MAX_FRAMES
    stand: tuple[int, int] = BOMB_N_STAND
    stand_tol: int = BOMB_N_STAND_TOL
    wait_blast: int = BOMB_N_WAIT_BLAST
    step_back: int = BOMB_N_STEP_BACK

    def _set_phase(self, phase: BombNorthPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(BombNorthPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _at_stand(self, snap: ZeldaSnapshot) -> bool:
        tx, ty = self.stand
        return abs(snap.link_x - tx) <= self.stand_tol and abs(
            snap.link_y - ty
        ) <= self.stand_tol

    def _goto_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        """Walk to bomb stand. Prefer y-band ~101 when near north (east-wall
        compass finish), else x then y. Unstick wiggle if frozen.
        """
        tx, ty = self.stand
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        # Near north band (post-compass east wall): hold y then walk west.
        if abs(snap.link_y - ty) <= 12 and abs(dx) > self.stand_tol:
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_band_y"
                )
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_band_x"
            )
        if abs(dx) > self.stand_tol and abs(dx) >= abs(dy):
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
            )
        if abs(dy) > self.stand_tol:
            return FrameAction(
                nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
            )
        return FrameAction(nes_idle_action(), "stand_ready")

    def _push_north(self, snap: ZeldaSnapshot) -> FrameAction:
        """Align to door x≈120 then hold UP. North bomb hole rejects wide x."""
        cx = self.stand[0]
        # Near the wall/door line: hard-center before thrusting (stuck at x≈128
        # on y≈93 is the common fail after a good blast).
        x_tol = 3 if snap.link_y <= 110 else 6
        if abs(snap.link_x - cx) > x_tol:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < cx else "LEFT"),
                "push_align_x",
            )
        return FrameAction(nes_action("UP"), "push_north")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is BombNorthPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is BombNorthPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        # Success: play-ready on 0x5f.
        if (
            snap.level == LEVEL_2
            and snap.screen == ROOM_L2_BOMB_N
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self._set_phase(BombNorthPhase.DONE, "entered_0x5f")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL_2:
            return FrameAction(nes_idle_action(), f"wait_level_{LEVEL_2}")

        if snap.transitioning or snap.mode in (4, 6, 7, 16):
            # Hold UP through scroll into 0x5f; else idle settle on 0x6f load.
            if self.phase is BombNorthPhase.PUSH or snap.screen == ROOM_L2_BOMB_N:
                return FrameAction(nes_action("UP"), "scroll_north")
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is BombNorthPhase.SETTLE:
            # Fixture Level2Compass may still have gels (reload) — clear first.
            if snap.screen != ROOM_L2_COMPASS:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            level_bit = 1 << (LEVEL_2 - 1)
            gels = ROOM_6F_SPEC.live_enemies(snap)
            need_clear = bool(gels) or (snap.compass & level_bit) == 0
            if need_clear:
                self.clear_controller = GenericDungeonRoomController(ROOM_6F_SPEC)
                self.clear_controller.phase = DungeonPhase.FIGHT
                self._set_phase(BombNorthPhase.CLEAR, "need_clear_0x6f")
            else:
                if snap.bombs <= 0:
                    return self._fail("no_bombs")
                self._set_phase(BombNorthPhase.TO_STAND, "already_clear")
            # fall through same frame after phase set

        if self.phase is BombNorthPhase.CLEAR:
            assert self.clear_controller is not None
            action = self.clear_controller.step(snap)
            if self.clear_controller.success:
                if snap.bombs <= 0:
                    return self._fail("no_bombs_after_clear")
                self._set_phase(BombNorthPhase.TO_STAND, "cleared_0x6f")
                return FrameAction(nes_idle_action(), "clear_done")
            if self.clear_controller.phase is DungeonPhase.FAILED:
                return self._fail("clear_failed")
            return action

        if self.phase is BombNorthPhase.TO_STAND:
            if snap.bombs <= 0:
                return self._fail("no_bombs")
            if self._at_stand(snap):
                self._set_phase(BombNorthPhase.FACE, "at_bomb_stand")
            elif self.phase_frames > 2500:
                return self._fail("stand_timeout")
            else:
                return self._goto_stand(snap)

        if self.phase is BombNorthPhase.FACE:
            # Hold UP so facing=N before B. ~4 frames is enough live.
            if self.phase_frames < 4:
                return FrameAction(nes_action("UP"), "face_up")
            self._set_phase(BombNorthPhase.PLACE, "faced_up")
            # fall through to place same frame after face

        if self.phase is BombNorthPhase.PLACE:
            if snap.bombs <= 0:
                return self._fail("no_bombs_at_place")
            self.bombs_before_place = int(snap.bombs)
            self._set_phase(BombNorthPhase.WAIT, "placed_bomb")
            return FrameAction(nes_action("UP", "B"), "place_bomb")

        if self.phase is BombNorthPhase.WAIT:
            if self.bombs_after_place is None and self.bombs_before_place is not None:
                if snap.bombs < self.bombs_before_place:
                    self.bombs_after_place = int(snap.bombs)
                    self.notes.append(
                        f"bomb_used_{self.bombs_before_place}->{snap.bombs}"
                    )
            if self.phase_frames < self.step_back:
                return FrameAction(nes_action("DOWN"), "step_back")
            if self.phase_frames < self.wait_blast:
                return FrameAction(nes_idle_action(), "wait_blast")
            if (
                self.bombs_after_place is None
                and self.bombs_before_place is not None
                and snap.bombs >= self.bombs_before_place
            ):
                # Bomb never consumed — wrong B-item or placement failed.
                return self._fail("bomb_not_consumed")
            self._set_phase(BombNorthPhase.PUSH, "blast_done")
            return FrameAction(nes_action("UP"), "push_start")

        if self.phase is BombNorthPhase.PUSH:
            if self.phase_frames > 700:
                return self._fail("push_timeout")
            return self._push_north(snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "bombs_before_place": self.bombs_before_place,
            "bombs_after_place": self.bombs_after_place,
            "stand": list(self.stand),
            "clear": (
                self.clear_controller.report()
                if self.clear_controller is not None
                else None
            ),
        }


def make_bomb_north_controller() -> Level2BombNorthController:
    """Factory for isolated 0x6f → 0x5f bomb-north pure segment."""
    return Level2BombNorthController()


# --- Boom bomb-north: 0x5f stand (120,101) UP+B → 0x4f (rr-bsq / rr-ebe) ---
# Same stand geometry as compass bomb-N. Level2_5F may have gels (TYPE-only);
# optional clear then bomb. Destination doors bit often DOWN after entry.

BOOM_BOMB_N_STAND = BOMB_N_STAND  # (120, 101)
BOOM_BOMB_N_MAX_FRAMES = 16000

# Probe-local gel clear on 0x5f (map gels) before bomb-N — not a STATUS room.
_ROOM_5F_GEL_CLEAR = DungeonRoomSpec(
    spec_id="level2_room5f_gel_clear_for_boom",
    source_room=ROOM_L2_COMPASS,
    room_id=ROOM_L2_BOMB_N,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=(
            (120, 141),
            (168, 141),
            (168, 109),
            (120, 109),
            (72, 109),
            (72, 141),
            (72, 173),
            (120, 173),
            (168, 173),
            (120, 141),
        ),
        engage_distance=56,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=10),
    max_frames=10000,
    level=LEVEL_2,
)


class BoomBombNorthPhase(Enum):
    """Phase machine for 0x5f bomb-north → 0x4f boom room."""

    SETTLE = auto()
    CLEAR = auto()
    TO_STAND = auto()
    FACE = auto()
    PLACE = auto()
    WAIT = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2BoomBombNorthController:
    """From 0x5f: optional gel clear, bomb north wall @ (120,101), enter 0x4f.

    Clean geometry policy: no health/inventory poke. Fails if bombs==0.
    """

    phase: BoomBombNorthPhase = BoomBombNorthPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_before_place: int | None = None
    bombs_after_place: int | None = None
    clear_controller: GenericDungeonRoomController | None = None
    clear_gels: bool = True
    max_frames: int = BOOM_BOMB_N_MAX_FRAMES
    stand: tuple[int, int] = BOOM_BOMB_N_STAND
    stand_tol: int = BOMB_N_STAND_TOL
    wait_blast: int = BOMB_N_WAIT_BLAST
    step_back: int = BOMB_N_STEP_BACK

    def _set_phase(self, phase: BoomBombNorthPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(BoomBombNorthPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _at_stand(self, snap: ZeldaSnapshot) -> bool:
        tx, ty = self.stand
        return abs(snap.link_x - tx) <= self.stand_tol and abs(
            snap.link_y - ty
        ) <= self.stand_tol

    def _goto_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = self.stand
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(snap.link_y - ty) <= 12 and abs(dx) > self.stand_tol:
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_band_y"
                )
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_band_x"
            )
        if abs(dx) > self.stand_tol and abs(dx) >= abs(dy):
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
            )
        if abs(dy) > self.stand_tol:
            return FrameAction(
                nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
            )
        return FrameAction(nes_idle_action(), "stand_ready")

    def _push_north(self, snap: ZeldaSnapshot) -> FrameAction:
        cx = self.stand[0]
        x_tol = 3 if snap.link_y <= 110 else 6
        if abs(snap.link_x - cx) > x_tol:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < cx else "LEFT"),
                "push_align_x",
            )
        return FrameAction(nes_action("UP"), "push_north")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is BoomBombNorthPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is BoomBombNorthPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL_2
            and snap.screen == ROOM_L2_BOOM_CANDIDATE
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self._set_phase(BoomBombNorthPhase.DONE, "entered_0x4f")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL_2:
            return FrameAction(nes_idle_action(), f"wait_level_{LEVEL_2}")

        if snap.transitioning or snap.mode in (4, 6, 7, 16):
            if (
                self.phase is BoomBombNorthPhase.PUSH
                or snap.screen == ROOM_L2_BOOM_CANDIDATE
            ):
                return FrameAction(nes_action("UP"), "scroll_north")
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is BoomBombNorthPhase.SETTLE:
            if snap.screen != ROOM_L2_BOMB_N:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            gels = _ROOM_5F_GEL_CLEAR.live_enemies(snap) if self.clear_gels else ()
            if gels:
                self.clear_controller = GenericDungeonRoomController(
                    _ROOM_5F_GEL_CLEAR
                )
                self.clear_controller.phase = DungeonPhase.FIGHT
                self._set_phase(BoomBombNorthPhase.CLEAR, "need_clear_gels_0x5f")
            else:
                if snap.bombs <= 0:
                    return self._fail("no_bombs")
                self._set_phase(BoomBombNorthPhase.TO_STAND, "already_clear")

        if self.phase is BoomBombNorthPhase.CLEAR:
            assert self.clear_controller is not None
            action = self.clear_controller.step(snap)
            if self.clear_controller.success:
                if snap.bombs <= 0:
                    return self._fail("no_bombs_after_clear")
                self._set_phase(BoomBombNorthPhase.TO_STAND, "cleared_gels")
                return FrameAction(nes_idle_action(), "clear_done")
            if self.clear_controller.phase is DungeonPhase.FAILED:
                return self._fail("clear_failed")
            return action

        if self.phase is BoomBombNorthPhase.TO_STAND:
            if snap.bombs <= 0:
                return self._fail("no_bombs")
            if self._at_stand(snap):
                self._set_phase(BoomBombNorthPhase.FACE, "at_bomb_stand")
            elif self.phase_frames > 2500:
                return self._fail("stand_timeout")
            else:
                return self._goto_stand(snap)

        if self.phase is BoomBombNorthPhase.FACE:
            if self.phase_frames < 4:
                return FrameAction(nes_action("UP"), "face_up")
            self._set_phase(BoomBombNorthPhase.PLACE, "faced_up")

        if self.phase is BoomBombNorthPhase.PLACE:
            if snap.bombs <= 0:
                return self._fail("no_bombs_at_place")
            self.bombs_before_place = int(snap.bombs)
            self._set_phase(BoomBombNorthPhase.WAIT, "placed_bomb")
            return FrameAction(nes_action("UP", "B"), "place_bomb")

        if self.phase is BoomBombNorthPhase.WAIT:
            if self.bombs_after_place is None and self.bombs_before_place is not None:
                if snap.bombs < self.bombs_before_place:
                    self.bombs_after_place = int(snap.bombs)
                    self.notes.append(
                        f"bomb_used_{self.bombs_before_place}->{snap.bombs}"
                    )
            if self.phase_frames < self.step_back:
                return FrameAction(nes_action("DOWN"), "step_back")
            if self.phase_frames < self.wait_blast:
                return FrameAction(nes_idle_action(), "wait_blast")
            if (
                self.bombs_after_place is None
                and self.bombs_before_place is not None
                and snap.bombs >= self.bombs_before_place
            ):
                return self._fail("bomb_not_consumed")
            self._set_phase(BoomBombNorthPhase.PUSH, "blast_done")
            return FrameAction(nes_action("UP"), "push_start")

        if self.phase is BoomBombNorthPhase.PUSH:
            if self.phase_frames > 700:
                return self._fail("push_timeout")
            return self._push_north(snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "bombs_before_place": self.bombs_before_place,
            "bombs_after_place": self.bombs_after_place,
            "stand": list(self.stand),
            "clear": (
                self.clear_controller.report()
                if self.clear_controller is not None
                else None
            ),
        }


def make_boom_bomb_north_controller(
    *, clear_gels: bool = True
) -> Level2BoomBombNorthController:
    """Factory for isolated 0x5f → 0x4f bomb-north segment."""
    return Level2BoomBombNorthController(clear_gels=clear_gels)


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
):
    register_room_spec(_spec)

# --- Post-boom bomb-north: 0x4f stand (120,101) UP+B → 0x3f (rr-n5i) ---
# Same stand as 0x6f/0x5f bomb-N. From Level2Boom (enemies clear, boom owned).


class PostBoomBombNorthPhase(Enum):
    """Phase machine for 0x4f bomb-north → 0x3f traps+Keese."""

    SETTLE = auto()
    TO_STAND = auto()
    FACE = auto()
    PLACE = auto()
    WAIT = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2PostBoomBombNorthController:
    """From 0x4f (boom collected): bomb north wall @ (120,101), enter 0x3f.

    Clean geometry policy: no health/inventory poke. Fails if bombs==0.
    Reuses BOMB_N_STAND / blast wait from compass bomb-N.
    """

    phase: PostBoomBombNorthPhase = PostBoomBombNorthPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_before_place: int | None = None
    bombs_after_place: int | None = None
    max_frames: int = BOOM_BOMB_N_MAX_FRAMES
    stand: tuple[int, int] = BOOM_BOMB_N_STAND
    stand_tol: int = BOMB_N_STAND_TOL
    wait_blast: int = BOMB_N_WAIT_BLAST

    def _set_phase(self, phase: PostBoomBombNorthPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(PostBoomBombNorthPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _at_stand(self, snap: ZeldaSnapshot) -> bool:
        tx, ty = self.stand
        return abs(snap.link_x - tx) <= self.stand_tol and abs(
            snap.link_y - ty
        ) <= self.stand_tol

    def _goto_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = self.stand
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(snap.link_y - ty) <= 12 and abs(dx) > self.stand_tol:
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_band_y"
                )
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_band_x"
            )
        if abs(dx) > self.stand_tol and abs(dx) >= abs(dy):
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
            )
        if abs(dy) > self.stand_tol:
            return FrameAction(
                nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
            )
        return FrameAction(nes_idle_action(), "stand_ready")

    def _push_north(self, snap: ZeldaSnapshot) -> FrameAction:
        cx = self.stand[0]
        x_tol = 3 if snap.link_y <= 110 else 6
        if abs(snap.link_x - cx) > x_tol:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < cx else "LEFT"),
                "push_align_x",
            )
        return FrameAction(nes_action("UP"), "push_north")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is PostBoomBombNorthPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is PostBoomBombNorthPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL_2
            and snap.screen == ROOM_L2_TRAPS_KEESE
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self._set_phase(PostBoomBombNorthPhase.DONE, "entered_0x3f")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL_2:
            return FrameAction(nes_idle_action(), f"wait_level_{LEVEL_2}")

        if snap.transitioning or snap.mode in (4, 6, 7, 16):
            if (
                self.phase is PostBoomBombNorthPhase.PUSH
                or snap.screen == ROOM_L2_TRAPS_KEESE
            ):
                return FrameAction(nes_action("UP"), "scroll_north")
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is PostBoomBombNorthPhase.SETTLE:
            if snap.screen != ROOM_L2_BOOM_CANDIDATE:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            if snap.bombs <= 0:
                return self._fail("no_bombs")
            self._set_phase(PostBoomBombNorthPhase.TO_STAND, "to_bomb_stand")
            return self._goto_stand(snap)

        if self.phase is PostBoomBombNorthPhase.TO_STAND:
            if self._at_stand(snap):
                self._set_phase(PostBoomBombNorthPhase.FACE, "at_bomb_stand")
                return FrameAction(nes_action("UP"), "face_up")
            return self._goto_stand(snap)

        if self.phase is PostBoomBombNorthPhase.FACE:
            if self.phase_frames < 6:
                return FrameAction(nes_action("UP"), "face_up")
            self._set_phase(PostBoomBombNorthPhase.PLACE, "faced_up")
            self.bombs_before_place = int(snap.bombs)
            return FrameAction(nes_action("UP", "B"), "place_bomb")

        if self.phase is PostBoomBombNorthPhase.PLACE:
            self.bombs_after_place = int(snap.bombs)
            if (
                self.bombs_before_place is not None
                and self.bombs_after_place < self.bombs_before_place
            ):
                self.notes.append(
                    f"bomb_used_{self.bombs_before_place}->{self.bombs_after_place}"
                )
            self._set_phase(PostBoomBombNorthPhase.WAIT, "placed_bomb")
            return FrameAction(nes_action("UP"), "wait_blast")

        if self.phase is PostBoomBombNorthPhase.WAIT:
            if self.phase_frames < self.wait_blast:
                return FrameAction(nes_action("UP"), "wait_blast")
            self._set_phase(PostBoomBombNorthPhase.PUSH, "blast_done")
            return self._push_north(snap)

        if self.phase is PostBoomBombNorthPhase.PUSH:
            return self._push_north(snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "frames": self.frames,
            "success": self.success,
            "notes": list(self.notes),
            "stand": list(self.stand),
            "bombs_before_place": self.bombs_before_place,
            "bombs_after_place": self.bombs_after_place,
        }


def make_post_boom_bomb_north_controller() -> Level2PostBoomBombNorthController:
    """Factory for isolated 0x4f → 0x3f bomb-north segment (post Magical Boom)."""
    return Level2PostBoomBombNorthController()


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



# Alias used by run_level2_bomb_north_5f (geometry-only hop; same stand as boom bomb).
Level2BombNorth5FController = Level2BoomBombNorthController


def make_bomb_north_5f_controller() -> Level2BoomBombNorthController:
    return Level2BoomBombNorthController(clear_gels=False)


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
    "BOMB_N_STAND",
    "BOOM_BOMB_N_STAND",
    "B_ITEM_BOMB",
    "BombNorthPhase",
    "BoomBombNorthPhase",
    "PostBoomBombNorthPhase",
    "Level2BombNorthController",
    "Level2BoomBombNorthController",
    "Level2BombNorth5FController",
    "Level2PostBoomBombNorthController",
    "make_bomb_north_5f_controller",
    "make_bomb_north_controller",
    "make_boom_bomb_north_controller",
    "make_post_boom_bomb_north_controller",
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
    "BOMB_N_STAND",
    "BOOM_BOMB_N_STAND",
    "B_ITEM_BOMB",
    "BombNorthPhase",
    "BoomBombNorthPhase",
    "Level2BombNorthController",
    "Level2BoomBombNorthController",
    "make_bomb_north_controller",
    "make_boom_bomb_north_controller",
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
]
