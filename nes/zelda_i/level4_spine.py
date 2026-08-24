"""Survival-spine L4 stage factories through the 0x11 bomb-UP 0x01 key.

The continuous runner composes these frame controllers. Isolated map_21
state-BFS is not a spine path.
"""

from __future__ import annotations

from zelda_i.dungeon import DungeonPhase
from zelda_i.level4_dungeon import (
    ROOM_31_SPEC,
    ROOM_32_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_NORTH_30,
    ROOM_L4_STEPLADDER,
    ROOM_L4_VIRES_50,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4_exit60 import (
    level4_exit60_stages,
    level4_exit60_success,
)
from zelda_i.level4_keyup20 import (
    level4_keyup20_stages,
    level4_keyup20_success,
)
from zelda_i.level4_map21 import (
    level4_map21_stages,
    level4_map21_success,
)
from zelda_i.level4_bomb11 import (
    level4_bomb11_stages,
    level4_bomb11_success,
)
from zelda_i.level4_key01 import (
    level4_key01_stages,
    level4_key01_success,
)
from zelda_i.level4_mappick import (
    level4_mappick_stages,
    level4_mappick_success,
)
from zelda_i.level4_maze_path import (
    make_maze_31_east_controller,
    make_maze_31_inland_controller,
    make_north_40_controller,
    make_room_40_key_controller,
)
from zelda_i.level4_overworld import (
    POST_L3_PATH_MAX_FRAMES,
    POST_L3_SETTLE_MAX_FRAMES,
    OverworldToLevel4Controller,
    PostL3TriforceSettleController,
)
from zelda_i.level4_path import (
    make_bomb_61_north_controller,
    make_entry_up_controller,
    make_left_50_controller,
    make_room_31_clear_controller,
    make_room_32_clear_controller,
    make_room_50_clear_controller,
    make_room_51_key_controller,
)
from zelda_i.level4_stepladder import (
    make_key_right_31_controller,
    make_north_30_controller,
    make_room_30_clear_controller,
    make_stepladder_controller,
)
from zelda_i.level4_west31 import (
    level4_west31_stages,
    level4_west31_success,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "level4_clear_31_stages",
    "level4_clear_31_success",
    "level4_clear_32_stages",
    "level4_clear_32_success",
    "level4_east_32_stages",
    "level4_east_32_success",
    "level4_entry_stages",
    "level4_exit60_stages",
    "level4_exit60_success",
    "level4_first_key_stages",
    "level4_first_key_success",
    "level4_key_right_31_stages",
    "level4_key_right_31_success",
    "level4_keyup20_stages",
    "level4_keyup20_success",
    "level4_map21_stages",
    "level4_map21_success",
    "level4_bomb11_stages",
    "level4_bomb11_success",
    "level4_key01_stages",
    "level4_key01_success",
    "level4_mappick_stages",
    "level4_mappick_success",
    "level4_north_30_stages",
    "level4_north_30_success",
    "level4_room40_key_stages",
    "level4_room40_key_success",
    "level4_room50_stages",
    "level4_room50_success",
    "level4_stepladder_stages",
    "level4_stepladder_success",
    "level4_west31_stages",
    "level4_west31_success",
]


def level4_entry_stages():
    """After L3 TF: settle on OW 0x74, cross the Raft dock, and enter L4."""
    return (
        (
            "settle_l3_tf",
            PostL3TriforceSettleController(),
            POST_L3_SETTLE_MAX_FRAMES,
        ),
        (
            "enter_level4",
            OverworldToLevel4Controller(require_dungeon=True),
            POST_L3_PATH_MAX_FRAMES,
        ),
    )


def level4_first_key_stages():
    """L4 entry 0x71 → clear 0x61 → bomb north → natural key on 0x51."""
    key = make_room_51_key_controller()
    key.phase = DungeonPhase.FIGHT
    return (
        ("level4_entry_up_0x61", make_entry_up_controller(), 4000),
        (
            "level4_bomb_north_0x61",
            make_bomb_61_north_controller(clear_vires=True),
            20000,
        ),
        ("level4_key_0x51", key, ROOM_51_SPEC.max_frames),
    )


def level4_first_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Exact natural-key stop; RoomAllDead may reset before reward pickup."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_KEESE_KEY_51
        and snap.keys > keys_before
        and not ROOM_51_SPEC.live_enemies(snap)
    )


def level4_room40_key_stages():
    """Natural 0x51 key → west 0x50 → scripted north 0x40 → natural key."""
    return (
        ("level4_north_0x40", make_north_40_controller(), 10000),
        ("level4_key_0x40", make_room_40_key_controller(), 25000),
    )


def level4_room50_stages():
    clear_50 = make_room_50_clear_controller()
    clear_50.phase = DungeonPhase.FIGHT
    return (
        ("level4_left_0x50", make_left_50_controller(), 2500),
        ("level4_clear_0x50", clear_50, ROOM_50_SPEC.max_frames),
    )


def level4_room50_success(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_VIRES_50
        and not ROOM_50_SPEC.live_enemies(snap)
    )


def level4_room40_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_ZOLS_40
        and snap.keys > keys_before
        and not ROOM_40_SPEC.live_enemies(snap)
    )


def level4_north_30_stages():
    """Cleared 0x40 with the natural key → free UP into 0x30 play-ready."""
    return (
        ("level4_north_0x30", make_north_30_controller(), 4000),
    )


def level4_north_30_success(snap: ZeldaSnapshot) -> bool:
    """Exact enter-0x30 stop; do not require the Vire clear or KEY-RIGHT."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_NORTH_30
        and not snap.transitioning
    )


def level4_key_right_31_stages():
    """Enter-0x30 leftover → north-band Vire clear (ignore 0x2b) → KEY-RIGHT."""
    return (
        ("level4_clear_0x30", make_room_30_clear_controller(), 20000),
        (
            "level4_key_right_0x31",
            make_key_right_31_controller(clear_vires=False),
            4000,
        ),
    )


def level4_key_right_31_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Exact enter-0x31 stop; 0x31 Vires stay live. KEY-RIGHT consumes one key."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_31
        and not snap.transitioning
        and snap.keys < keys_before
    )


def level4_clear_31_stages():
    """West-door leftover (16,141) → alcove clip → maze Vire clear."""
    clear_31 = make_room_31_clear_controller()
    clear_31.phase = DungeonPhase.FIGHT
    return (
        ("level4_inland_0x31", make_maze_31_inland_controller(), 4000),
        ("level4_clear_0x31", clear_31, ROOM_31_SPEC.max_frames),
    )


def level4_clear_31_success(snap: ZeldaSnapshot) -> bool:
    """Exact 0x31 maze-clear stop; do not require the free-RIGHT into 0x32."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_31
        and not snap.transitioning
        and not ROOM_31_SPEC.live_enemies(snap)
    )


def level4_east_32_stages():
    """Cleared-0x31 leftover (112,141) → maze thread → free RIGHT into 0x32."""
    return (
        ("level4_east_0x32", make_maze_31_east_controller(), 4000),
    )


def level4_east_32_success(snap: ZeldaSnapshot) -> bool:
    """Exact enter-0x32 stop; Zol/LikeLike stay live."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_32
        and not snap.transitioning
    )


def level4_clear_32_stages():
    """West-door leftover (16,141) → Zol + LikeLike clear (ignore 0x2b/0x68)."""
    clear_32 = make_room_32_clear_controller()
    clear_32.phase = DungeonPhase.FIGHT
    return (
        ("level4_clear_0x32", clear_32, ROOM_32_SPEC.max_frames),
    )


def level4_clear_32_success(snap: ZeldaSnapshot) -> bool:
    """Exact 0x32 empty-room stop; do not require push-block or 0x60 stairs."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_32
        and not snap.transitioning
        and not ROOM_32_SPEC.live_enemies(snap)
    )


def level4_stepladder_stages():
    """Cleared-0x32 leftover (80,109) → push left → 0x60 ADDR_LADDER."""
    ctl = make_stepladder_controller(clear_first=False)
    return (
        ("level4_stepladder", ctl, ctl.max_frames),
    )


def level4_stepladder_success(snap: ZeldaSnapshot) -> bool:
    """Exact ADDR_LADDER stop; do not require 0x32 exit or Keese clear."""
    return (
        snap.level == 4
        and snap.ladder > 0
        and (snap.screen == ROOM_L4_STEPLADDER or snap.mode == 9)
    )
