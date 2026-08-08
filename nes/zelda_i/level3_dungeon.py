"""Level 3 (Manji) dungeon room specs and pure helpers.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Lives outside ``dungeon.py`` so L2 room tables stay untouched.

Live pure (2026-08-06, Clean isolated from ``Level3Entrance``)::

    0x7c entry --(LEFT+UP corner residual)--> 0x7b
    0x7b: 6× Zol type ``0x13`` (HP>0) + fixed key RoomItemId ``0x19``
    Clear + key pickup ~658 combat frames after room-ready (3/3 trials).

Live pure chain from ``Level3WestKey`` (2026-08-06 recon + encode)::

    0x7b --(UP @ x≈120 strict)--> 0x6b
    0x6b: 5× Zol type ``0x13`` on diagonal-block floor; RoomItemId ``0x19``
          (key drop residual — type-0 HP leftovers stall RoomAllDead)
    0x6b --(UP @ x≈120 after type-0x13 clear)--> 0x5b
    0x5b: 3× Darknut type ``0x0b`` (HP 64); north open → 0x4b (3× Zol+key)

Assisted LIVE past Darknuts toward Raft (2026-08-07, ``l3_past_5b``)::

    0x5b --UP open--> 0x4b (3× Zol 0x13, key 0x19)
    0x5b --LEFT open--> 0x5a (4× Keese + traps, Compass 0x16)  **Raft path**
    0x5b --RIGHT walk blocked; BOMB_RIGHT @ (192,141)--> 0x5c (boss shortcut)
    0x5a --LEFT KEY--> 0x59 (5× Darknut; kill opens DOWN)
    0x59 --DOWN--> 0x69 (8× Darknut)
    0x69 --RIGHT @ y≈141--> 0x0f mode-9 passage
    0x0f: DOWN→y189, RIGHT→x≈176, UP channel, LEFT→x≈136 **Raft** (ADDR_RAFT)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.anchors import TF_BIT_L3 as LEVEL3_TRIFORCE_BIT
from zelda_i.door_graph.core import DoorDir
from zelda_i.level3_geometry import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    DOOR_5C_RIGHT_Y,
    KEY_DOOR_Y,
    KEY_DOOR_Y_TOL,
    PASSAGE_EXIT_WAYPOINTS,
    STAIRS_69_RIGHT_Y,
)
from zelda_i.level3_overworld import LEVEL3, SCREEN_LEVEL3_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# Geometry names above are re-exported so existing
# ``from zelda_i.level3_dungeon import BOMB_STAND_*`` imports keep working.

# --- Live L3 room / enemy anchors (isolated pure 2026-08-06 + past-5b 08-07) ---
ROOM_L3_ENTRY = SCREEN_LEVEL3_ENTRY_ROOM  # 0x7C
ROOM_L3_WEST_KEY = 0x7B
ROOM_L3_NORTH_ZOLS = 0x6B  # north of west-key; diagonal blocks
ROOM_L3_DARKNUTS = 0x5B  # north of 0x6b after zol clear
ROOM_L3_ZOL_KEY_4B = 0x4B  # north of darknuts; 3× Zol + key (LIVE)
ROOM_L3_COMPASS = 0x5A  # west of darknuts; 4× Keese + traps + Compass (LIVE)
ROOM_L3_WEST_DARKNUTS = 0x59  # west of compass; 5× Darknut; key door from 0x5a
ROOM_L3_SOUTH_DARKNUTS = 0x69  # south of 0x59; 8× Darknut; stairs RIGHT
ROOM_L3_RAFT_PASSAGE = 0x0F  # mode-9 underworld; Raft pickup (LIVE assisted)
ROOM_L3_MAP_4C = 0x4C  # east of 0x4b via key; map room item 0x17
ROOM_L3_BOMB_SHORTCUT = 0x5C  # bomb-RIGHT of 0x5b (boss shortcut residual)
ROOM_L3_BOSS_PREP = 0x5D  # east of 0x5c after clear; RIGHT@y≈141 (LIVE residual)
ROOM_L3_BOSS = 0x4D  # Manhandla candidate north of 0x5d (assisted glimpse; not pure)
ZOL_OBJECT_TYPE = 0x13  # live type on 0x7b/0x6b/0x4b; wooden sword can leave type-0 HP residual
DARKNUT_OBJECT_TYPE = 0x0B  # live type on 0x5b/0x59/0x69 (red Darknut, HP 64)
KEESE_OBJECT_TYPE = 0x1B
MANHANDLA_OBJECT_TYPE = 0x3C  # LIVE candidate on 0x4d (5 slots + 0x56 projectiles); residual
INVULN_MOVER_0X2B = 0x2B  # 0x49/0x5d HP240 invuln — not Manhandla
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_COMPASS = 0x16
ROOM_ITEM_MAP = 0x17
ROOM_ITEM_RAFT = 0x0C  # live RoomItemId in mode-9 passage 0x0f
ROOM_ITEM_HEART_CONTAINER = 0x1A

# Raft passage geometry (mode 9): enter from 0x69 RIGHT @ y≈141 → spawn ~(48,77).
# Path: DOWN to y≈189 → RIGHT to x≈176 → UP channel to y≈141 → LEFT to x≈136 touch Raft.
RAFT_PASSAGE_MODE = 9
RAFT_CHANNEL_X = 176
RAFT_CHANNEL_X_TOL = 4
RAFT_PICKUP_X = 136
RAFT_PICKUP_Y = 141
RAFT_SOUTH_Y = 189
RAFT_SOUTH_Y_TOL = 6
KEY_DOOR_PUSH_FRAMES = 160  # short push can spend key without room change
SPAWN_SETTLE_FRAMES = 100  # Darknuts lag ~75–100f before clear registers
LEFT_5B_MAX_FRAMES = 1500
KEY_5A_MAX_FRAMES = 2500
CLEAR_59_MAX_FRAMES = 18000
DOWN_69_MAX_FRAMES = 2000
CLEAR_69_MAX_FRAMES = 28000
STAIRS_69_MAX_FRAMES = 2500
PASSAGE_RAFT_MAX_FRAMES = 6000
RAFT_PATH_MAX_FRAMES = 55000

# Darknut sword patrol (side/back hits; assist Survival OK).
_DARKNUT_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (100, 125),
    (140, 157),
    (80, 157),
    (160, 125),
)

# West door residual: pure LEFT sticks at x≈32 (mask==0). LEFT+UP at the west
# wall corner-clips into the scroll (mode 6/7 → room 0x7b). Approach band y≈149
# reaches the wall; y≈141 alone often blocks mid-room at x≈112.
WEST_DOOR_APPROACH_Y = 149
WEST_DOOR_WALL_X = 48
WEST_ENTER_MAX_FRAMES = 1200

# North door residual from 0x7b: UP only works with |x-120|≤4. Threshold 8
# leaves Link at x≈112 and sticks on the north wall (live probe 2026-08-06).
NORTH_DOOR_X = 120
NORTH_DOOR_X_TOL = 4
NORTH_ENTER_MAX_FRAMES = 1500
NORTH_EXIT_6B_MAX_FRAMES = 6000

_ROOM_7B_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 141),
    (160, 181),
    (112, 181),
    (64, 181),
    (64, 141),
    (120, 141),
)

# 0x6b diagonal-block floor: prefer south/mid bands that stay walkable.
_ROOM_6B_PATROL: tuple[tuple[int, int], ...] = (
    (100, 181),
    (140, 181),
    (160, 173),
    (150, 157),
    (120, 157),
    (100, 157),
    (100, 173),
    (128, 141),
    (112, 173),
    (136, 165),
)

# After clear, snake toward north door plane (live free-explore path).
_ROOM_6B_NORTH_EXIT: tuple[tuple[int, int], ...] = (
    (120, 189),
    (144, 181),
    (152, 165),
    (144, 141),
    (136, 125),
    (128, 109),
    (120, 100),
    (120, 93),
)

ROOM_7B_SPEC = DungeonRoomSpec(
    spec_id="level3_room7b_west_key",
    source_room=ROOM_L3_ENTRY,
    room_id=ROOM_L3_WEST_KEY,
    entry=DoorRoute(
        "LEFT",
        ((120, WEST_DOOR_APPROACH_Y), (WEST_DOOR_WALL_X, WEST_DOOR_APPROACH_Y)),
    ),
    enemy_types=(ZOL_OBJECT_TYPE,),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_7B_PATROL,
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
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=10000,
    level=LEVEL3,
)

# Type-0x13 clear only: RoomAllDead often stays 0 (type-0 HP leftovers after
# wooden-sword hits). settle_all_dead=0 so CLEAR_ONLY trips when live Zols==0.
ROOM_6B_SPEC = DungeonRoomSpec(
    spec_id="level3_room6b_north_zols",
    source_room=ROOM_L3_WEST_KEY,
    room_id=ROOM_L3_NORTH_ZOLS,
    entry=DoorRoute(
        "UP",
        ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93)),
    ),
    enemy_types=(ZOL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_6B_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.CLEAR_ONLY,
        settle_all_dead=0,
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("DOWN", ((NORTH_DOOR_X, 205),)),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=12000,
    level=LEVEL3,
)

# Darknut room graph node (combat not pure-encoded yet — side/back hits only).
# LIVE doors (no clear required): UP→0x4b, DOWN→0x6b, LEFT→0x5a; RIGHT walk sealed
# (bomb-RIGHT @ (192,141) → 0x5c boss shortcut; recon poke OK).
ROOM_5B_SPEC = DungeonRoomSpec(
    spec_id="level3_room5b_darknuts",
    source_room=ROOM_L3_NORTH_ZOLS,
    room_id=ROOM_L3_DARKNUTS,
    entry=DoorRoute(
        "UP",
        ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93)),
    ),
    enemy_types=(DARKNUT_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=(
            (80, 141),
            (120, 117),
            (160, 141),
            (160, 173),
            (120, 173),
            (80, 173),
            (120, 141),
        ),
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    exit_routes=(
        DoorRoute("DOWN", ((NORTH_DOOR_X, 205),)),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=15000,
    level=LEVEL3,
)

# 0x4b north of Darknuts: 3× Zol + RoomItemId key (LIVE probe). North door open
# from 0x5b without clear. Key pickup residual (keys inventory may not increment).
_ROOM_4B_PATROL: tuple[tuple[int, int], ...] = (
    (64, 125),
    (120, 125),
    (176, 125),
    (176, 157),
    (120, 173),
    (64, 157),
    (120, 141),
)

ROOM_4B_SPEC = DungeonRoomSpec(
    spec_id="level3_room4b_zol_key",
    source_room=ROOM_L3_DARKNUTS,
    room_id=ROOM_L3_ZOL_KEY_4B,
    entry=DoorRoute(
        "UP",
        ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93)),
    ),
    enemy_types=(ZOL_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_4B_PATROL,
        engage_distance=64,
        attack_phase=4,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.CLEAR_ONLY,
        settle_all_dead=0,
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("DOWN", ((NORTH_DOOR_X, 205),)),
        DoorRoute("LEFT", ((120, 141), (32, 141))),  # KEY → 0x4a
        DoorRoute("RIGHT", ((120, 141), (208, 141))),  # KEY → 0x4c map
    ),
    max_frames=10000,
    level=LEVEL3,
)

# Compass room west of Darknuts (assisted LIVE; Keese type-only clear easy).
ROOM_5A_SPEC = DungeonRoomSpec(
    spec_id="level3_room5a_compass",
    source_room=ROOM_L3_DARKNUTS,
    room_id=ROOM_L3_COMPASS,
    entry=DoorRoute("LEFT", ((120, 141), (32, 141))),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=4,
    alive_rule=AliveRule.TYPE,  # keese often HP residual 0 while typed
    combat=CombatTuning(
        patrol=(
            (64, 117),
            (120, 117),
            (176, 117),
            (176, 173),
            (64, 173),
            (120, 141),
        ),
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_COMPASS,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),  # KEY → 0x59
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),  # free → 0x4a
    ),
    max_frames=8000,
    level=LEVEL3,
)

# 0x59 west of compass: 5× Darknut; kill opens DOWN → 0x69 (assisted LIVE).
ROOM_59_SPEC = DungeonRoomSpec(
    spec_id="level3_room59_west_darknuts",
    source_room=ROOM_L3_COMPASS,
    room_id=ROOM_L3_WEST_DARKNUTS,
    entry=DoorRoute("LEFT", ((120, KEY_DOOR_Y), (32, KEY_DOOR_Y))),
    enemy_types=(DARKNUT_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_DARKNUT_PATROL,
        engage_distance=40,
        attack_phase=2,
        patrol_attack_period=6,
        patrol_attack_hold=3,
        engage_attack_period=5,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    required_open_doors=DoorDir.DOWN,  # kill-clear opens south
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
    ),
    max_frames=CLEAR_59_MAX_FRAMES,
    level=LEVEL3,
)

# 0x69 south of 0x59: 8× Darknut; stairs RIGHT @ y≈141 → 0x0f (assisted LIVE).
ROOM_69_SPEC = DungeonRoomSpec(
    spec_id="level3_room69_south_darknuts",
    source_room=ROOM_L3_WEST_DARKNUTS,
    room_id=ROOM_L3_SOUTH_DARKNUTS,
    entry=DoorRoute("DOWN", ((120, 141), (120, 205))),
    enemy_types=(DARKNUT_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_DARKNUT_PATROL,
        engage_distance=40,
        attack_phase=2,
        patrol_attack_period=6,
        patrol_attack_hold=3,
        engage_attack_period=5,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    exit_routes=(
        DoorRoute("UP", ((NORTH_DOOR_X, 141), (NORTH_DOOR_X, 93))),
        DoorRoute(
            "RIGHT",
            ((120, STAIRS_69_RIGHT_Y), (208, STAIRS_69_RIGHT_Y)),
        ),
    ),
    max_frames=CLEAR_69_MAX_FRAMES,
    level=LEVEL3,
)

register_room_spec(ROOM_7B_SPEC)
register_room_spec(ROOM_6B_SPEC)
register_room_spec(ROOM_5B_SPEC)
register_room_spec(ROOM_4B_SPEC)
register_room_spec(ROOM_5A_SPEC)
register_room_spec(ROOM_59_SPEC)
register_room_spec(ROOM_69_SPEC)


def level3_room_7b_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x7b with keys≥1 and no live Zols.

    FIXED_INVENTORY stop: inventory + TYPE_AND_HP liveness only (RoomAllDead
    may lag after the last kill / key touch).
    """
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_WEST_KEY
        and snap.mode == PLAY_MODE
        and snap.keys >= 1
        and not ROOM_7B_SPEC.live_enemies(snap)
    )


def level3_room_6b_zols_cleared(ram: np.ndarray) -> bool:
    """0x6b with no live type-0x13 Zols (RoomAllDead not required)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_NORTH_ZOLS
        and snap.mode == PLAY_MODE
        and not ROOM_6B_SPEC.live_enemies(snap)
    )


def level3_reached_5b(ram: np.ndarray) -> bool:
    """Isolated pure stop: play mode inside 0x5b (Darknut room)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_DARKNUTS
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_room_4b_zols_cleared(ram: np.ndarray) -> bool:
    """0x4b with no live type-0x13 Zols (RoomAllDead not required)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_ZOL_KEY_4B
        and snap.mode == PLAY_MODE
        and not ROOM_4B_SPEC.live_enemies(snap)
    )


def level3_has_raft(ram: np.ndarray) -> bool:
    """ADDR_RAFT inventory bit set (assisted LIVE pickup in 0x0f passage)."""
    from zelda_i.ram import ADDR_RAFT, read_u8

    return bool(read_u8(ram, ADDR_RAFT))


def level3_reached_boss_prep(ram: np.ndarray) -> bool:
    """Play mode in room 0x5d (Manhandla prep east of bomb-shortcut 0x5c)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_BOSS_PREP
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_reached_boss(ram: np.ndarray) -> bool:
    """Play mode in room 0x4d (Manhandla candidate north of 0x5d)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_BOSS
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_manhandla_live(snap: ZeldaSnapshot) -> list:
    """Live Manhandla heads (type 0x3c, HP>0) on current snapshot."""
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 10
        and o.type_id == MANHANDLA_OBJECT_TYPE
        and o.hp > 0
    ]


def level3_boss_prep_killables(snap: ZeldaSnapshot) -> list:
    """Killable enemies on 0x5d: Zol + Keese only (ignore invuln 0x2b)."""
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id == ZOL_OBJECT_TYPE and o.hp > 0:
            out.append(o)
        elif o.type_id == KEESE_OBJECT_TYPE:
            # Keese often report HP 0 while still "alive" for type liveness
            out.append(o)
    return out



# Register room specs (path controllers in level3_path).
for _spec in (
    ROOM_7B_SPEC,
    ROOM_6B_SPEC,
    ROOM_5B_SPEC,
    ROOM_4B_SPEC,
    ROOM_5A_SPEC,
    ROOM_59_SPEC,
    ROOM_69_SPEC,
):
    register_room_spec(_spec)


_PATH_EXPORTS = frozenset({
    "west_door_step",
    "north_door_7b_step",
    "north_exit_6b_step",
    "Level3WestDoorController",
    "Level3NorthDoor7bController",
    "Level3NorthExit6bController",
    "Level3WestKeyController",
    "Level3NorthChainController",
    "Level3RaftPathController",
    "raft_passage_step",
    "RAFT_PATH_PHASES",
})


def __getattr__(name: str):
    if name in _PATH_EXPORTS:
        if name in {"Level3RaftPathController", "RAFT_PATH_PHASES", "raft_passage_step"}:
            from zelda_i import level3_raft_path as _paths
        else:
            from zelda_i import level3_path as _paths
        return getattr(_paths, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PATH_EXPORTS))
