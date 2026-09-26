"""Level 6 (Dragon) dungeon room specs and stop predicates.

Owned by L6 pure wave. Import ``GenericDungeonRoomController`` from ``dungeon``.

The spine enters 0x79 and leaves west through the key door into 0x78.
0x7a is not on the route. Wizzrobe backstep for the old 0x7a and 0x78
clears still lives in ``level6.wizzrobe`` for the tests that pin it.
The spine itself walks 0x78 and bombs 0x28 east.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.dungeon.ids import (
    GEL_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    VIRE_OBJECT_TYPE,
    VIRE_SPLIT_KEESE_TYPE,
    WIZZROBE_BLUE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import PLAY_MODE, read_snapshot

# Re-export for runners / docs.
ROOM_L6_ENTRY = LEVEL6_ENTRY_ROOM  # 0x79
ROOM_L6_EAST_KEY = LEVEL6_EAST_KEY_ROOM  # 0x7a
ROOM_L6_WEST_WIZZROBE = LEVEL6_WEST_WIZZROBE_ROOM  # 0x78
ROOM_L6_COMPASS = LEVEL6_COMPASS_ROOM  # 0x68
ROOM_L6_KEESE = LEVEL6_KEESE_ROOM  # 0x58
ROOM_L6_HARD_38 = LEVEL6_WIZZROBE_38_ROOM  # 0x38
ROOM_L6_WIZZROBE_28 = LEVEL6_WIZZROBE_28_ROOM  # 0x28
ROOM_L6_MAP = LEVEL6_MAP_ROOM  # 0x19, cleared from the south mouth
ROOM_L6_ROD_WIZZ = LEVEL6_ROD_WIZZ_ROOM  # 0x09 north of 0x19
ROOM_L6_DARK_29 = LEVEL6_DARK_29_ROOM  # 0x29 south of Map; dark wizzrobes
ROOM_L6_DARK_39 = LEVEL6_DARK_39_ROOM  # 0x39 south of 0x29; live 5× Vire 0x12
# ADDR_COMPASS bitfield: one bit per dungeon (L6 → bit5 → 0x20).
LEVEL6_COMPASS_BIT = 1 << (LEVEL6 - 1)
_OCC_BOUNDS = (16, 216, 77, 205)


def _occ(
    patrol: tuple[tuple[int, int], ...],
    *,
    occupancy_bounds: tuple[int, int, int, int] = _OCC_BOUNDS,
    occupancy_blocked: tuple[tuple[int, int], ...] = (),
    contact_backstep: int = 0,
    evade: bool = False,
    avoid_wall_bounds: tuple[int, int, int, int] = (56, 200, 109, 173),
    west_mouth_down_to: int = 149,
) -> CombatTuning:
    return CombatTuning(
        patrol=patrol,
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
        occupancy_patrol=True,
        occupancy_bounds=occupancy_bounds,
        occupancy_blocked=occupancy_blocked,
        inland_dash=24,
        avoid_walls=True,
        avoid_wall_bounds=avoid_wall_bounds,
        west_mouth_down_to=west_mouth_down_to,
        contact_backstep=contact_backstep,
        evade=evade,
    )


# Open-floor patrol; wizzrobes teleport — cover mid lanes.
# West column is x=80, not x=64: cart-WRAM block (64,112)-(64,128) traps
# patrol-nearest (64,109) from leftover (64,117) (Clean east-key timeout).
_ROOM_7A_PATROL: tuple[tuple[int, int], ...] = (
    (80, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (80, 173),
    (80, 141),
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
# Key pickup live at room center (120,141) — matches the measured Survival
# leftover (`l6_east_key_continuous_v1.json`, keys 5→6 at (120,141)); the old
# (136,141) target was an unmeasured "~" estimate (LEVEL6_ROUTE.md) and sits
# unreachable from a NW post-combat leftover: cart-WRAM tilemap shows a block
# cell at (64,112)-(64,128) directly south of that leftover, and the plain
# axis-priority `_collect_reward` path (no waypoints) has no stuck-escape, so
# it presses DOWN into the block for the full 12000-frame room timeout
# (root-caused live 2026-09-04). Waypoints reuse the proven-safe combat ring
# so the 24-frame stuck-skip (`_collect_reward` waypoints branch) can route
# around both corner-block pairs from any post-combat leftover.
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
        target=(120, 141),
        waypoints=_ROOM_7A_PATROL,
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
        inland_dash=24,
        evade=True,
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
    combat=_occ(_ROOM_68_PATROL),
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

# North of 0x68: 8× Keese 0x1b + key drop (inventory residual).
# Spine leftover is south mouth (120,205); four corner fires; north sealed
# until clear. Ignore invuln 0x2b / block 0x68. Keese are TYPE-only.
_ROOM_58_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 141),
    (120, 109),
    (160, 141),
    (120, 173),
    (80, 109),
    (160, 109),
    (80, 173),
    (160, 173),
    (120, 141),
)

ROOM_58_SPEC = DungeonRoomSpec(
    spec_id="level6_room58_keese",
    source_room=LEVEL6_COMPASS_ROOM,
    room_id=LEVEL6_KEESE_ROOM,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE,
    combat=_occ(_ROOM_58_PATROL),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=12000,
    level=LEVEL6,
)

register_room_spec(ROOM_58_SPEC)

# North of 0x48: walkthrough 2× orange 0x24 + 2× blue 0x23 + 3× Like-Like
# 0x17 + Bubble 0x40. Spine leftover is south mouth (120,189); two center
# blocks. Ignore invuln 0x2b / block 0x68 / Bubble (sword-immune residual).
# Clear is occupancy-patrol. Left-block UP then west-aisle north is on the spine.
_ROOM_38_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (64, 141),
    (80, 109),
    (120, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (160, 189),
    (120, 173),
    (48, 157),
    (192, 157),
    (120, 93),
)

ROOM_38_SPEC = DungeonRoomSpec(
    spec_id="level6_room38_hard",
    source_room=LEVEL6_TRAPS_ROOM,
    room_id=LEVEL6_WIZZROBE_38_ROOM,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(
        WIZZROBE_ORANGE_TYPE,
        WIZZROBE_BLUE_OBJECT_TYPE,
        LIKE_LIKE_OBJECT_TYPE,
    ),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    # Clean power-on 81 died here in 525f on engage contacts. Peel
    # (contact_backstep) survives the bodies; evade ducks the 0x58 beams
    # that then killed the peel (c8_l6_from_38, shot from 3px).
    combat=_occ(_ROOM_38_PATROL, contact_backstep=16, evade=True),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=25000,
    level=LEVEL6,
)

register_room_spec(ROOM_38_SPEC)

# North of 0x38: leftover RAM max_live=2 orange 0x24. Do not copy 0x38's 7× mix.
# Ignore invuln 0x2b / Bubble 0x40 / block 0x68. Diamond floor is walkable.
_ROOM_28_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (64, 141),
    (80, 109),
    (120, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (160, 189),
    (120, 141),
    (48, 157),
    (192, 157),
    (120, 93),
)

ROOM_28_SPEC = DungeonRoomSpec(
    spec_id="level6_room28_wizzrobes",
    source_room=LEVEL6_WIZZROBE_38_ROOM,
    room_id=LEVEL6_WIZZROBE_28_ROOM,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(WIZZROBE_ORANGE_TYPE,),
    expected_enemy_count=2,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=_occ(
        _ROOM_28_PATROL,
        contact_backstep=16,
        evade=False,
        # The diamond row at y=181 is walkable. Forcing UP from (135,181)
        # hits a block and holds Link in the eastbound beam lane.
        avoid_wall_bounds=(56, 200, 109, 189),
    ),
    # The 5R room item sits across live blue Wizzrobe bodies after the two
    # orange targets die; the spine has enough rupees without this pickup.
    reward=RewardSpec(
        kind=RewardKind.CLEAR_ONLY,
        settle_all_dead=0,
        sweep_room_item=False,
    ),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=12000,
    level=LEVEL6,
)

register_room_spec(ROOM_28_SPEC)

# East of 0x18: live census 2× Zol 0x13 + 2× Like-Like 0x17 + 2× invuln 0x2b.
# Enter PNG beam is a Like-Like. Ignore 0x2b / Bubble 0x40. RoomItemId 0x17 Map.
_ROOM_19_PATROL: tuple[tuple[int, int], ...] = (
    (48, 141),
    (80, 141),
    (120, 141),
    (176, 141),
    (176, 109),
    (120, 109),
    (64, 109),
    (64, 173),
    (120, 173),
    (176, 173),
    (120, 93),
    (120, 189),
)

ROOM_19_SPEC = DungeonRoomSpec(
    spec_id="level6_room19_clear",
    source_room=LEVEL6_GLEEOK_ROOM,
    room_id=LEVEL6_MAP_ROOM,
    entry=DoorRoute("RIGHT", ((208, 141),)),
    enemy_types=(
        ZOL_OBJECT_TYPE,
        LIKE_LIKE_OBJECT_TYPE,
    ),
    expected_enemy_count=4,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=_occ(_ROOM_19_PATROL, west_mouth_down_to=160),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x17,
    exit_routes=(
        DoorRoute("LEFT", ((32, 141),)),
        DoorRoute("RIGHT", ((208, 141),)),
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=15000,
    level=LEVEL6,
    # A full-height water column (x 160..175) splits the room; the exit is
    # the north key door on the west bank. East-bank bodies are off route.
    reachable_only=True,
)

register_room_spec(ROOM_19_SPEC)

# North of 0x19: skip-Map KEY-UP leftover (120,205). Live settle census
# 3× blue 0x23 + 2× orange 0x24 + left 0x68 (96,144). Ignore 0x2b / 0x40 /
# 0x59 shot / block 0x68. Do not push the left block or require Rod.
_ROOM_09_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (64, 141),
    (80, 109),
    (120, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (160, 189),
    (48, 157),
    (192, 157),
    (120, 141),
)

ROOM_09_SPEC = DungeonRoomSpec(
    spec_id="level6_room09_wizzrobes",
    source_room=LEVEL6_MAP_ROOM,
    room_id=LEVEL6_ROD_WIZZ_ROOM,
    entry=DoorRoute("UP", ((120, 141), (120, 93))),
    enemy_types=(WIZZROBE_ORANGE_TYPE, WIZZROBE_BLUE_OBJECT_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=_occ(_ROOM_09_PATROL),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("UP", ((120, 141), (120, 93))),
    ),
    max_frames=15000,
    level=LEVEL6,
)

register_room_spec(ROOM_09_SPEC)

# South of 0x19: dark room. Spine leftover north mouth (120,77).
# Live census (clear29 v1): 3× blue 0x23 + 2× orange 0x24 + 0x59 shots.
# Not Vire 0x12. Ignore 0x2b / Bubble 0x40 / 0x59. Do not grant candle.
# RoomItemId 0x19 key on floor residual. Do not require stairs/Gohma.
# Leftover is the south door. (120,141) is the plus, not a handoff.
_ROOM_29_LEFTOVER = (120, 189)
_ROOM_29_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (64, 141),
    (80, 109),
    (120, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (160, 189),
    (192, 157),
)
# Island minus the y=141 waist (SOUTH29 v4 walked x=120 @ y=141 then DOWN).
_ROOM_29_BLOCKED: tuple[tuple[int, int], ...] = tuple(
    (x, y)
    for x in range(48, 153)
    for y in (*range(117, 141), *range(142, 158))
)

ROOM_29_SPEC = DungeonRoomSpec(
    spec_id="level6_room29_wizzrobes",
    source_room=LEVEL6_MAP_ROOM,
    room_id=LEVEL6_DARK_29_ROOM,
    entry=DoorRoute("DOWN", ((120, 141), (120, 205))),
    enemy_types=(WIZZROBE_ORANGE_TYPE, WIZZROBE_BLUE_OBJECT_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=_occ(
        _ROOM_29_PATROL,
        occupancy_bounds=(16, 216, 77, 141),
        occupancy_blocked=_ROOM_29_BLOCKED,
    ),
    reward=RewardSpec(
        kind=RewardKind.CLEAR_ONLY,
        settle_all_dead=0,
        target=_ROOM_29_LEFTOVER,
        waypoints=((120, 141),),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("RIGHT", ((208, 141),)),
        DoorRoute("LEFT", ((32, 141),)),
    ),
    max_frames=15000,
    level=LEVEL6,
    # A one-cell moat rings the island; the doors are open mouths, not
    # shutters. A Wizzrobe parked on the far bank (48,93) held the fight on
    # the moat ladder for 14500f (Blue Ring power-on 4): clear this bank.
    reachable_only=True,
)

register_room_spec(ROOM_29_SPEC)


# South of 0x29: dark leftover north mouth (120,93). Live census (settle39 v1):
# 5× Vire 0x12 HP64. Ignore 0x2b / Bubble 0x40 / split 0x1c HP0. Do not
# invent Gohma. East PNG lock; dest after clear is RAM.
_ROOM_39_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (80, 189),
    (80, 173),
    (64, 141),
    (80, 109),
    (120, 109),
    (160, 109),
    (176, 141),
    (160, 173),
    (160, 189),
    (48, 157),
    (192, 157),
    (120, 141),
)

ROOM_39_SPEC = DungeonRoomSpec(
    spec_id="level6_room39_vires",
    source_room=LEVEL6_DARK_29_ROOM,
    room_id=LEVEL6_DARK_39_ROOM,
    entry=DoorRoute("DOWN", ((120, 141), (120, 205))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=_occ(_ROOM_39_PATROL),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("RIGHT", ((208, 141),)),
        DoorRoute("LEFT", ((32, 141),)),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=15000,
    level=LEVEL6,
)

register_room_spec(ROOM_39_SPEC)

# East of 0x39: leftover west mouth (16,141). Live census (settle3a v1):
# 3× Like-Like 0x17 + 2× blue 0x23 + 2× orange 0x24 + 0x59 shots + center
# 0x68 (112,144). Ignore 0x2b / 0x40 / 0x59 / block 0x68. Do not push.
_ROOM_3A_PATROL: tuple[tuple[int, int], ...] = (
    (48, 141),
    (80, 189),
    (160, 189),
    (176, 141),
    (160, 109),
    (80, 109),
    (48, 157),
    (192, 157),
    (120, 141),
    (96, 173),
    (144, 109),
)

ROOM_3A_SPEC = DungeonRoomSpec(
    spec_id="level6_room3a_block",
    source_room=LEVEL6_DARK_39_ROOM,
    room_id=LEVEL6_BLOCK_3A_ROOM,
    entry=DoorRoute("RIGHT", ((16, 141),)),
    enemy_types=(
        LIKE_LIKE_OBJECT_TYPE,
        WIZZROBE_BLUE_OBJECT_TYPE,
        WIZZROBE_ORANGE_TYPE,
    ),
    expected_enemy_count=7,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=_occ(_ROOM_3A_PATROL),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("LEFT", ((32, 141),)),
        DoorRoute("RIGHT", ((208, 141),)),
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
    ),
    max_frames=25000,
    level=LEVEL6,
)

register_room_spec(ROOM_3A_SPEC)


def _l6_enemies_dead(ram: np.ndarray, room: int, spec: DungeonRoomSpec) -> bool:
    """Play-mode room with no live spec enemies. Combat: GenericDungeonRoomController(spec)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.screen == room
        and snap.mode == PLAY_MODE
        and not spec.live_enemies(snap)
    )


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


def level6_room_78_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x78 cleared — no live type-0x24, play mode.

    Does not require UP door bit (mask lag) or inventory change.
    """
    return _l6_enemies_dead(ram, LEVEL6_WEST_WIZZROBE_ROOM, ROOM_78_SPEC)


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


def level6_room_58_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x58 no live Keese. Key drop residual."""
    return _l6_enemies_dead(ram, LEVEL6_KEESE_ROOM, ROOM_58_SPEC)


def level6_room_38_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x38 no live wizzrobe/Like-Like. Bubble residual."""
    return _l6_enemies_dead(ram, LEVEL6_WIZZROBE_38_ROOM, ROOM_38_SPEC)


def level6_room_28_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x28 no live orange wizzrobes. Ignore 0x2b/0x40/0x68."""
    return _l6_enemies_dead(ram, LEVEL6_WIZZROBE_28_ROOM, ROOM_28_SPEC)


def level6_room_19_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x19 no live spec enemies. Ignore 0x2b/0x40. No Map."""
    return _l6_enemies_dead(ram, LEVEL6_MAP_ROOM, ROOM_19_SPEC)


def level6_room_09_clear_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x09 no live wizzrobes. Ignore 0x2b/0x40/0x68. No Rod."""
    return _l6_enemies_dead(ram, LEVEL6_ROD_WIZZ_ROOM, ROOM_09_SPEC)


__all__ = [
    "ROOM_L6_ENTRY", "ROOM_L6_EAST_KEY", "ROOM_L6_WEST_WIZZROBE",
    "ROOM_L6_COMPASS", "ROOM_L6_KEESE", "ROOM_L6_HARD_38", "ROOM_L6_WIZZROBE_28",
    "ROOM_L6_MAP", "ROOM_L6_ROD_WIZZ", "ROOM_L6_DARK_29", "ROOM_L6_DARK_39",
    "ROOM_79_SPEC", "ROOM_7A_SPEC", "ROOM_78_SPEC", "ROOM_68_SPEC",
    "ROOM_58_SPEC", "ROOM_38_SPEC", "ROOM_28_SPEC", "ROOM_19_SPEC",
    "ROOM_09_SPEC", "ROOM_29_SPEC", "ROOM_39_SPEC", "ROOM_3A_SPEC",
    "LEVEL6_COMPASS_BIT",
    "level6_room_7a_key_success", "level6_room_78_clear_success",
    "level6_room_68_compass_success", "level6_room_58_clear_success",
    "level6_room_38_clear_success", "level6_room_28_clear_success",
    "level6_room_19_clear_success", "level6_room_09_clear_success",
]
