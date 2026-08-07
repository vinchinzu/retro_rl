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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
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
LEVEL3_TRIFORCE_BIT = 0x04

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


def west_door_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of 0x7c → 0x7b west-door policy (diagonal residual)."""
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3")
    if snap.transitioning:
        return FrameAction(nes_action("LEFT", "UP"), "west_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_L3_WEST_KEY:
        return FrameAction(nes_idle_action(), "west_arrived")
    if snap.screen != ROOM_L3_ENTRY:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    # South mouth → room body
    if snap.link_y > 165:
        return FrameAction(nes_action("UP"), "west_leave_mouth")
    # Horizontal approach on y≈149 band (reaches x≈32 wall)
    if snap.link_x > WEST_DOOR_WALL_X:
        if abs(snap.link_y - WEST_DOOR_APPROACH_Y) > 3:
            direction = "UP" if snap.link_y > WEST_DOOR_APPROACH_Y else "DOWN"
            return FrameAction(nes_action(direction), "west_align_y")
        return FrameAction(nes_action("LEFT"), "west_approach")
    # Door plane: LEFT alone sticks; LEFT+UP corner-clips into 0x7b
    return FrameAction(nes_action("LEFT", "UP"), "west_diagonal_push")


def north_door_7b_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of 0x7b → 0x6b north-door policy (strict x≈120)."""
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_L3_NORTH_ZOLS:
        return FrameAction(nes_idle_action(), "north_arrived_6b")
    if snap.screen != ROOM_L3_WEST_KEY:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "north_align_x")
    return FrameAction(nes_action("UP"), "north_push")


_RIGHT_OF = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
_LEFT_OF = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}


def north_exit_6b_step(
    snap: ZeldaSnapshot,
    *,
    facing: str,
    prefer_goal: bool,
) -> tuple[FrameAction, str]:
    """One frame of 0x6b → 0x5b after Zol clear.

    Live residual: diagonal raised blocks partition the floor. Pure waypoint
    snakes stall after combat. Right-hand wall-follow reaches the north band
    (y≈93); once there, center x≈120 and hold UP into 0x5b.
    """
    if snap.level != LEVEL3:
        return FrameAction(nes_idle_action(), "wait_level3"), facing
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north6b_scroll"), facing
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}"), facing
    if snap.screen == ROOM_L3_DARKNUTS:
        return FrameAction(nes_idle_action(), "north_arrived_5b"), facing
    if snap.screen != ROOM_L3_NORTH_ZOLS:
        return (
            FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"),
            facing,
        )

    # Door plane / north band: center and hold UP.
    if snap.link_y <= 105:
        if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "north6b_align_door"), direction
        return FrameAction(nes_action("UP"), "north6b_push"), "UP"

    # Bias: if south of mid, prefer RIGHT then UP corridor (live free-explore).
    if prefer_goal and snap.link_y > 150:
        if snap.link_x < 140:
            return FrameAction(nes_action("RIGHT"), "north6b_bias_right"), "RIGHT"
        return FrameAction(nes_action("UP"), "north6b_bias_up"), "UP"

    # Right-hand wall follow (facing is last successful move).
    face = facing if facing in _RIGHT_OF else "UP"
    # Order: right of face, face, left, back — caller probes via stuck reset.
    order = (_RIGHT_OF[face], face, _LEFT_OF[face], _RIGHT_OF[_RIGHT_OF[face]])
    # Emit preferred first; Level3NorthExit6bController may override on stuck.
    direction = order[0]
    return FrameAction(nes_action(direction), f"north6b_follow_{direction}"), direction


@dataclass
class Level3WestDoorController:
    """Route 0x7c → 0x7b only (no combat). Success when room-ready on 0x7b."""

    max_frames: int = WEST_ENTER_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        action = west_door_step(snap)
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_WEST_KEY
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x7b")
            return FrameAction(nes_idle_action(), "west_arrived")
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "policy": "LEFT+UP diagonal at west wall; approach y≈149",
        }


@dataclass
class Level3NorthDoor7bController:
    """Route 0x7b → 0x6b only (no combat). Strict x≈120 UP residual."""

    max_frames: int = NORTH_ENTER_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        action = north_door_7b_step(snap)
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_NORTH_ZOLS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x6b")
            return FrameAction(nes_idle_action(), "north_arrived_6b")
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "policy": "UP @ x≈120 (|dx|≤4); wider align sticks at x≈112",
        }


# Free-explore target grid (live 2026-08-06: exits 0x6b→0x5b after combat).
_ROOM_6B_HUNT: tuple[tuple[int, int], ...] = tuple(
    (x, y) for y in range(90, 210, 8) for x in range(72, 200, 8)
) + tuple(
    (x, y) for y in range(90, 112, 4) for x in range(96, 152, 8)
) + (
    (120, 93),
    (120, 93),
    (120, 100),
    (120, 93),
)


@dataclass
class Level3NorthExit6bController:
    """Route 0x6b → 0x5b after Zols cleared (free-explore grid + door push).

    Live: after combat, walk a coarse grid; when blocked try alternate
    directions; on north band hold UP @ x≈120 into 0x5b.
    """

    max_frames: int = NORTH_EXIT_6B_MAX_FRAMES
    frames: int = 0
    hunt_index: int = 0
    target_steps: int = 0
    pending_alt: str | None = None
    success: bool = False
    failed: bool = False
    last_xy: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_DARKNUTS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("entered_0x5b")
            return FrameAction(nes_idle_action(), "north_arrived_5b")

        if snap.transitioning:
            return FrameAction(nes_action("UP"), "north6b_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L3_NORTH_ZOLS:
            return FrameAction(
                nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
            )

        # Consume one-shot alternate direction after a blocked step.
        if self.pending_alt is not None:
            direction = self.pending_alt
            self.pending_alt = None
            return FrameAction(nes_action(direction), "north6b_alt")

        xy = (snap.link_x, snap.link_y)
        blocked = self.last_xy == xy
        self.last_xy = xy

        # North band: center and push door.
        if snap.link_y <= 100 and abs(snap.link_x - NORTH_DOOR_X) <= 8:
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north6b_align_door")
            return FrameAction(nes_action("UP"), "north6b_push")
        if snap.link_y <= 100:
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north6b_align_door")
            return FrameAction(nes_action("UP"), "north6b_push")

        # Advance hunt target after a short attempt window or arrival.
        tx, ty = _ROOM_6B_HUNT[self.hunt_index % len(_ROOM_6B_HUNT)]
        self.target_steps += 1
        if (
            abs(snap.link_x - tx) <= 6 and abs(snap.link_y - ty) <= 6
        ) or self.target_steps >= 45:
            self.hunt_index = (self.hunt_index + 1) % len(_ROOM_6B_HUNT)
            self.target_steps = 0
            tx, ty = _ROOM_6B_HUNT[self.hunt_index % len(_ROOM_6B_HUNT)]

        dx, dy = tx - snap.link_x, ty - snap.link_y
        if abs(dx) > 3 and abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 3:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "UP"

        # If last frame did not move, queue an alternate direction.
        if blocked:
            alts = [d for d in ("UP", "RIGHT", "DOWN", "LEFT") if d != direction]
            self.pending_alt = alts[self.frames % len(alts)]
            if self.frames % 60 == 0:
                self.notes.append(f"block_f{self.frames}_hunt{self.hunt_index}")

        return FrameAction(nes_action(direction), f"north6b_hunt_{self.hunt_index}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "hunt_index": self.hunt_index,
            "notes": list(self.notes),
            "policy": "free-explore grid hunt + UP @ x≈120 on north band",
        }


@dataclass
class Level3WestKeyController:
    """Full isolated pure: Level3Entrance 0x7c → 0x7b clear + key.

    Phase 1: ``Level3WestDoorController`` (diagonal residual).
    Phase 2: ``GenericDungeonRoomController(ROOM_7B_SPEC)`` combat/reward.
    """

    door: Level3WestDoorController = field(default_factory=Level3WestDoorController)
    combat: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_7B_SPEC)
    )
    frames: int = 0
    success: bool = False
    phase: str = "door"
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")

        if self.phase == "door":
            action = self.door.step(snap)
            if self.door.success:
                self.phase = "combat"
                self.notes.append("door_ok")
                # Hand off; combat controller sees room_id and enters FIGHT.
                return self.combat.step(snap)
            if self.door.failed:
                self.phase = "failed"
                self.notes.append("door_failed")
            return action

        if self.phase == "combat":
            action = self.combat.step(snap)
            if self.combat.success:
                self.success = True
                self.phase = "done"
                self.notes.append("key_ok")
            elif self.combat.phase is DungeonPhase.FAILED:
                self.phase = "failed"
                self.notes.append("combat_failed")
            return action

        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "door": self.door.report(),
            "combat": self.combat.report(),
            "spec_id": ROOM_7B_SPEC.spec_id,
            "intervention_class": "clean",
            "track": "clean",
        }


@dataclass
class Level3NorthChainController:
    """Isolated pure from ``Level3WestKey``: 0x7b → 0x6b clear → 0x5b.

    Phase 1: ``Level3NorthDoor7bController`` (UP @ x≈120).
    Phase 2: ``GenericDungeonRoomController(ROOM_6B_SPEC)`` Zol clear.
    Phase 3: ``Level3NorthExit6bController`` north to Darknut room.
    Stop: ``level3_reached_5b``.
    """

    door: Level3NorthDoor7bController = field(
        default_factory=Level3NorthDoor7bController
    )
    combat: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_6B_SPEC)
    )
    north_exit: Level3NorthExit6bController = field(
        default_factory=Level3NorthExit6bController
    )
    frames: int = 0
    success: bool = False
    phase: str = "door"
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")

        # Early success if already in 0x5b (reload / resume).
        if (
            snap.level == LEVEL3
            and snap.screen == ROOM_L3_DARKNUTS
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.phase = "done"
            self.notes.append("already_0x5b")
            return FrameAction(nes_idle_action(), "done")

        if self.phase == "door":
            action = self.door.step(snap)
            if self.door.success:
                self.phase = "combat"
                self.notes.append("door_6b_ok")
                return self.combat.step(snap)
            if self.door.failed:
                self.phase = "failed"
                self.notes.append("door_6b_failed")
            return action

        if self.phase == "combat":
            action = self.combat.step(snap)
            if self.combat.success:
                self.phase = "north_exit"
                self.notes.append("zols_cleared")
                return self.north_exit.step(snap)
            if self.combat.phase is DungeonPhase.FAILED:
                self.phase = "failed"
                self.notes.append("combat_failed")
            return action

        if self.phase == "north_exit":
            action = self.north_exit.step(snap)
            if self.north_exit.success:
                self.success = True
                self.phase = "done"
                self.notes.append("reached_0x5b")
            elif self.north_exit.failed:
                self.phase = "failed"
                self.notes.append("north_exit_failed")
            return action

        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "door": self.door.report(),
            "combat": self.combat.report(),
            "north_exit": self.north_exit.report(),
            "spec_id": ROOM_6B_SPEC.spec_id,
            "stop": "level3_reached_5b",
            "intervention_class": "clean",
            "track": "clean",
        }


# ---------------------------------------------------------------------------
# Assisted Raft path: Level3Darknuts 0x5b → Compass west → 0x59/0x69 → 0x0f Raft
# ---------------------------------------------------------------------------

RAFT_PATH_PHASES: tuple[str, ...] = (
    "settle_5b",
    "left_to_5a",
    "key_to_59",
    "spawn_59",
    "clear_59",
    "down_to_69",
    "spawn_69",
    "clear_69",
    "stairs_to_0f",
    "passage_raft",
    "done",
    "failed",
)


def _align_then_push(
    snap: ZeldaSnapshot,
    *,
    target_x: int | None,
    target_y: int | None,
    push_dir: str,
    y_tol: int = KEY_DOOR_Y_TOL,
    x_tol: int = NORTH_DOOR_X_TOL,
    reason_prefix: str = "door",
    door_plane: int | None = None,
) -> FrameAction:
    """Align to door band then hold push direction (scroll handled by caller).

    For side doors, pass ``door_plane`` (e.g. 48 for west): once Link is at or
    past the plane toward the exit, only push — do not snap back to target_x
    (Link can sit at x≈26 on the west wall while target_x=32).
    """
    if target_y is not None and abs(snap.link_y - target_y) > y_tol:
        direction = "UP" if snap.link_y > target_y else "DOWN"
        return FrameAction(nes_action(direction), f"{reason_prefix}_align_y")
    # Side-door plane: walk until near wall, then hold push.
    if door_plane is not None and push_dir in ("LEFT", "RIGHT"):
        if push_dir == "LEFT" and snap.link_x > door_plane:
            return FrameAction(nes_action("LEFT"), f"{reason_prefix}_approach")
        if push_dir == "RIGHT" and snap.link_x < door_plane:
            return FrameAction(nes_action("RIGHT"), f"{reason_prefix}_approach")
        return FrameAction(nes_action(push_dir), f"{reason_prefix}_push_{push_dir}")
    # North/south or explicit x target.
    if target_x is not None and abs(snap.link_x - target_x) > x_tol:
        direction = "LEFT" if snap.link_x > target_x else "RIGHT"
        return FrameAction(nes_action(direction), f"{reason_prefix}_align_x")
    return FrameAction(nes_action(push_dir), f"{reason_prefix}_push_{push_dir}")


def raft_passage_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of mode-9 0x0f passage geometry to Raft pickup.

    LIVE residual: south band UP is solid except channel at x≈176. Path:
    DOWN y≈189 → RIGHT x≈176 → UP to y≈141 → LEFT x≈136 touch Raft.

    Once on the channel column, prefer vertical align + LEFT (do not re-south).
    """
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    # Scroll / mode settle into underworld.
    if snap.transitioning or snap.mode not in (PLAY_MODE, RAFT_PASSAGE_MODE):
        if snap.mode in (6, 7, 10):
            return FrameAction(nes_action("RIGHT"), "passage_scroll")
        return FrameAction(nes_idle_action(), f"passage_wait_mode_{snap.mode}")
    if snap.screen != ROOM_L3_RAFT_PASSAGE and snap.mode != RAFT_PASSAGE_MODE:
        return FrameAction(
            nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
        )

    at_channel = abs(snap.link_x - RAFT_CHANNEL_X) <= RAFT_CHANNEL_X_TOL
    near_channel = abs(snap.link_x - RAFT_CHANNEL_X) <= 16
    on_south = snap.link_y >= RAFT_SOUTH_Y - RAFT_SOUTH_Y_TOL
    on_pickup_band = abs(snap.link_y - RAFT_PICKUP_Y) <= 8

    # Mid horizontal band (raft corridor): do not re-south — walk to pickup x.
    # Drift off exact channel while walking LEFT is expected (176 → 136).
    if on_pickup_band and (at_channel or near_channel or snap.link_x <= RAFT_CHANNEL_X):
        if snap.link_x > RAFT_PICKUP_X + 2:
            return FrameAction(nes_action("LEFT"), "passage_to_raft")
        if snap.link_x < RAFT_PICKUP_X - 6:
            return FrameAction(nes_action("RIGHT"), "passage_raft_overshoot")
        return FrameAction(nes_action("LEFT"), "passage_raft_touch")

    # Channel column: vertical align to pickup band.
    if at_channel or (near_channel and snap.link_y > RAFT_PICKUP_Y):
        if abs(snap.link_x - RAFT_CHANNEL_X) > RAFT_CHANNEL_X_TOL:
            direction = "RIGHT" if snap.link_x < RAFT_CHANNEL_X else "LEFT"
            return FrameAction(nes_action(direction), "passage_recenter_channel")
        if snap.link_y > RAFT_PICKUP_Y:
            return FrameAction(nes_action("UP"), "passage_channel_up")
        if snap.link_y < RAFT_PICKUP_Y - KEY_DOOR_Y_TOL:
            return FrameAction(nes_action("DOWN"), "passage_channel_down")
        return FrameAction(nes_action("LEFT"), "passage_to_raft")

    # Off channel north: reach south band first (UP solid except channel).
    if not on_south:
        # If already east, prefer re-acquire channel over futile south into wall.
        if snap.link_x >= 140:
            direction = "RIGHT" if snap.link_x < RAFT_CHANNEL_X else "LEFT"
            return FrameAction(nes_action(direction), "passage_seek_channel")
        return FrameAction(nes_action("DOWN"), "passage_to_south")

    # South band: walk to channel x≈176.
    if snap.link_x < RAFT_CHANNEL_X:
        return FrameAction(nes_action("RIGHT"), "passage_to_channel")
    return FrameAction(nes_action("LEFT"), "passage_to_channel")


def _count_live_darknuts(snap: ZeldaSnapshot) -> int:
    return sum(
        1
        for o in snap.objects
        if 1 <= o.slot <= 10
        and o.type_id == DARKNUT_OBJECT_TYPE
        and o.hp > 0
    )


def _is_room_scroll(snap: ZeldaSnapshot) -> bool:
    """True during horizontal/vertical room scroll (modes 4/6/7/16)."""
    return snap.transitioning or snap.mode in (4, 6, 7, 16)


@dataclass
class Level3RaftPathController:
    """Assisted Survival: Level3Darknuts → ADDR_RAFT via Compass west path.

    Phases (see ``RAFT_PATH_PHASES``)::

        settle_5b → left_to_5a → key_to_59 → spawn_59 → clear_59
        → down_to_69 → spawn_69 → clear_69 → stairs_to_0f → passage_raft

    Intervention: Survival (``--infinite-life``). Not Clean STATUS.
    """

    frames: int = 0
    phase_frames: int = 0
    push_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: str = "settle_5b"
    keys_at_key_door: int | None = None
    max_live_59: int = 0
    max_live_69: int = 0
    clear_59: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_59_SPEC)
    )
    clear_69: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_69_SPEC)
    )
    notes: list[str] = field(default_factory=list)
    max_frames: int = RAFT_PATH_MAX_FRAMES

    def _set_phase(self, phase: str, note: str = "") -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.push_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self.failed = True
        self._set_phase("failed", note)
        return FrameAction(nes_idle_action(), "failed")

    def step(self, snap: ZeldaSnapshot, *, has_raft: bool = False) -> FrameAction:
        """One control frame. Pass ``has_raft=level3_has_raft(ram)`` when available."""
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.mode == 17:
            return self._fail("link_death")

        # Global success: Raft inventory bit (may set mid-passage).
        if has_raft:
            self.success = True
            self._set_phase("done", "raft_acquired")
            return FrameAction(nes_idle_action(), "done")

        # --- settle_5b: ignore Darknuts; brief spawn settle then leave west ---
        if self.phase == "settle_5b":
            if (
                snap.screen == ROOM_L3_COMPASS
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                self._set_phase("key_to_59", "already_0x5a")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if (
                snap.screen == ROOM_L3_WEST_DARKNUTS
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                self._set_phase("spawn_59", "already_0x59")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                self._set_phase("passage_raft", "already_passage")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if self.phase_frames < 40:
                return FrameAction(nes_idle_action(), "settle_5b")
            self._set_phase("left_to_5a", "leave_darknuts_west")
            return FrameAction(nes_idle_action(), "phase_handoff")

        # --- left_to_5a: open west door (no clear) @ y≈141 ---
        if self.phase == "left_to_5a":
            if self.phase_frames > LEFT_5B_MAX_FRAMES:
                return self._fail("left_5a_timeout")
            if snap.screen == ROOM_L3_COMPASS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("LEFT"), "left_5a_scroll")
                self._set_phase("key_to_59", "entered_0x5a")
                return FrameAction(nes_idle_action(), "left_5a_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("LEFT"), "left_5a_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            return _align_then_push(
                snap,
                target_x=32,
                target_y=KEY_DOOR_Y,
                push_dir="LEFT",
                reason_prefix="left_5a",
                door_plane=48,
            )

        # --- key_to_59: long LEFT KEY push @ y≈141 (trap: short push wastes key) ---
        if self.phase == "key_to_59":
            if self.phase_frames > KEY_5A_MAX_FRAMES:
                return self._fail("key_59_timeout")
            if snap.screen == ROOM_L3_WEST_DARKNUTS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("LEFT"), "key_59_scroll")
                self._set_phase("spawn_59", "entered_0x59")
                return FrameAction(nes_idle_action(), "key_59_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("LEFT"), "key_59_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_COMPASS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if self.keys_at_key_door is None:
                self.keys_at_key_door = int(snap.keys)
            # Optional compass: if standing near center item and free, step once.
            # Not required — skip if not aligned.
            if (
                abs(snap.link_x - 120) <= 8
                and abs(snap.link_y - 141) <= 8
                and snap.room_item_id == ROOM_ITEM_COMPASS
                and self.phase_frames < 30
            ):
                return FrameAction(nes_idle_action(), "optional_compass_touch")
            # Align y first, approach west wall, long push.
            if abs(snap.link_y - KEY_DOOR_Y) > KEY_DOOR_Y_TOL:
                direction = "UP" if snap.link_y > KEY_DOOR_Y else "DOWN"
                self.push_frames = 0
                return FrameAction(nes_action(direction), "key_59_align_y")
            if snap.link_x > 48:
                self.push_frames = 0
                return FrameAction(nes_action("LEFT"), "key_59_approach")
            # Long hold LEFT at door plane (critical residual).
            self.push_frames += 1
            if self.push_frames > KEY_DOOR_PUSH_FRAMES + 80 and snap.keys == 0:
                # Key spent without scroll — fail honestly.
                if self.keys_at_key_door is not None and snap.keys < self.keys_at_key_door:
                    return self._fail("key_spent_no_scroll")
            return FrameAction(nes_action("LEFT"), "key_59_long_push")

        # --- spawn_59: wait for Darknuts to materialize ---
        if self.phase == "spawn_59":
            live = _count_live_darknuts(snap)
            self.max_live_59 = max(self.max_live_59, live)
            if snap.screen != ROOM_L3_WEST_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if live >= 3 or self.phase_frames >= SPAWN_SETTLE_FRAMES:
                self._set_phase(
                    "clear_59",
                    f"spawn_59_live={live}_f{self.phase_frames}",
                )
                return self.clear_59.step(snap)
            return FrameAction(nes_idle_action(), "spawn_59_wait")

        # --- clear_59: sword patrol until type-0x0b gone AND DOWN opens ---
        if self.phase == "clear_59":
            live = _count_live_darknuts(snap)
            self.max_live_59 = max(self.max_live_59, live)
            if snap.screen != ROOM_L3_WEST_DARKNUTS and snap.mode == PLAY_MODE:
                # Accidental exit — try recover only if still on path.
                if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
                    self._set_phase("spawn_69", "early_0x69")
                    return FrameAction(nes_idle_action(), "phase_handoff")
            action = self.clear_59.step(snap)
            down_open = bool(snap.cur_opened_doors & DoorDir.DOWN)
            # Kill-clear lag: live can hit 0 while doors still lack DOWN for ~40f
            # (room_all_dead ramps after last corpse). Do not leave early.
            if live == 0 and self.max_live_59 >= 3 and not down_open:
                return FrameAction(nes_idle_action(), "clear_59_wait_door")
            if (
                self.clear_59.success
                or (down_open and live == 0 and self.max_live_59 >= 3)
            ):
                self._set_phase(
                    "down_to_69",
                    f"cleared_59_doors={snap.cur_opened_doors}_alldead={snap.room_all_dead}",
                )
                return FrameAction(nes_idle_action(), "clear_59_done")
            if self.clear_59.phase is DungeonPhase.FAILED:
                # Soft fallback: if DOWN open and few live, still try exit.
                if down_open and live <= 1:
                    self._set_phase("down_to_69", "clear_59_partial_down_open")
                    return FrameAction(nes_idle_action(), "clear_59_partial")
                return self._fail("clear_59_failed")
            return action

        # --- down_to_69: south after kill-clear ---
        if self.phase == "down_to_69":
            if self.phase_frames > DOWN_69_MAX_FRAMES:
                return self._fail("down_69_timeout")
            if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("DOWN"), "down_69_scroll")
                self._set_phase("spawn_69", "entered_0x69")
                return FrameAction(nes_idle_action(), "down_69_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("DOWN"), "down_69_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_WEST_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            # Align x≈120 then hold DOWN. Do not chase y=205 — past the door
            # plane Link thrash-oscillates align_y/push and never scrolls.
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "down_69_align_x")
            return FrameAction(nes_action("DOWN"), "down_69_push_DOWN")

        # --- spawn_69 ---
        if self.phase == "spawn_69":
            live = _count_live_darknuts(snap)
            self.max_live_69 = max(self.max_live_69, live)
            if snap.screen != ROOM_L3_SOUTH_DARKNUTS:
                if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                    self._set_phase("passage_raft", "early_passage")
                    return FrameAction(nes_idle_action(), "phase_handoff")
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if live >= 4 or self.phase_frames >= SPAWN_SETTLE_FRAMES:
                self._set_phase(
                    "clear_69",
                    f"spawn_69_live={live}_f{self.phase_frames}",
                )
                return self.clear_69.step(snap)
            return FrameAction(nes_idle_action(), "spawn_69_wait")

        # --- clear_69: 8 Darknuts then stairs ---
        if self.phase == "clear_69":
            live = _count_live_darknuts(snap)
            self.max_live_69 = max(self.max_live_69, live)
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                self._set_phase("passage_raft", "stairs_during_clear")
                return FrameAction(nes_idle_action(), "phase_handoff")
            action = self.clear_69.step(snap)
            if (
                self.clear_69.success
                or (
                    live == 0
                    and self.max_live_69 >= 4
                    and self.phase_frames > 80
                )
            ):
                self._set_phase("stairs_to_0f", f"cleared_69_maxlive={self.max_live_69}")
                return FrameAction(nes_idle_action(), "clear_69_done")
            if self.clear_69.phase is DungeonPhase.FAILED:
                # Try stairs anyway if clear-ish (stairs may need full clear).
                if live == 0:
                    self._set_phase("stairs_to_0f", "clear_69_timeout_try_stairs")
                    return FrameAction(nes_idle_action(), "clear_69_try_stairs")
                return self._fail("clear_69_failed")
            return action

        # --- stairs_to_0f: RIGHT @ y≈141 only ---
        if self.phase == "stairs_to_0f":
            if self.phase_frames > STAIRS_69_MAX_FRAMES:
                return self._fail("stairs_timeout")
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                if snap.mode not in (PLAY_MODE, RAFT_PASSAGE_MODE) and snap.mode != 10:
                    return FrameAction(nes_action("RIGHT"), "stairs_scroll")
                self._set_phase("passage_raft", "entered_passage")
                return FrameAction(nes_idle_action(), "stairs_arrived")
            if _is_room_scroll(snap) or snap.mode in (6, 7, 10):
                return FrameAction(nes_action("RIGHT"), "stairs_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_SOUTH_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            return _align_then_push(
                snap,
                target_x=208,
                target_y=STAIRS_69_RIGHT_Y,
                push_dir="RIGHT",
                y_tol=KEY_DOOR_Y_TOL,
                reason_prefix="stairs",
                door_plane=192,
            )

        # --- passage_raft: mode-9 channel geometry ---
        if self.phase == "passage_raft":
            if self.phase_frames > PASSAGE_RAFT_MAX_FRAMES:
                return self._fail("passage_timeout")
            # level3_has_raft checked at top; keep walking to touch tile.
            return raft_passage_step(snap)

        if self.phase == "done":
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase,
            "frames": self.frames,
            "phase_frames": self.phase_frames,
            "notes": list(self.notes),
            "max_live_59": self.max_live_59,
            "max_live_69": self.max_live_69,
            "keys_at_key_door": self.keys_at_key_door,
            "clear_59": self.clear_59.report(),
            "clear_69": self.clear_69.report(),
            "phases": list(RAFT_PATH_PHASES),
            "stop": "level3_has_raft",
            "path": (
                "0x5b LEFT→0x5a LEFT KEY→0x59 clear DOWN→0x69 clear "
                "RIGHT@y141→0x0f channel→Raft"
            ),
            "intervention_class": "survival",
            "track": "assisted",
            "geometry": {
                "key_door_y": KEY_DOOR_Y,
                "stairs_y": STAIRS_69_RIGHT_Y,
                "channel_x": RAFT_CHANNEL_X,
                "pickup_xy": [RAFT_PICKUP_X, RAFT_PICKUP_Y],
                "south_y": RAFT_SOUTH_Y,
            },
        }
