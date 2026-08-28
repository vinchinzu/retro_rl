"""Post-Gleeok L6 continue: 0x19 through 0x3A stairs.

Extracted from level6_spine so hops can attach without growing a 1.5k file.
Do not poke Rod / doors / keys / bow / arrows. Do not grant Map/Whistle.
Isolated BFS banned. Ignore 0x2b / Bubble. Gohma / TF 0x20 residual.
"""

from __future__ import annotations

from zelda_i.level6_dungeon import (
    LEVEL6_MAP_BIT,
    ROOM_09_SPEC,
    ROOM_19_SPEC,
    ROOM_29_SPEC,
    ROOM_39_SPEC,
    ROOM_3A_SPEC,
    make_clear_09_controller,
    make_clear_19_controller,
    make_clear_29_controller,
    make_clear_39_controller,
    make_clear_3a_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_MAP_ROOM,
)

from zelda_i.level6_room19 import (
    MAP_19_MAX_FRAMES,
    ROOM09_MAX_FRAMES,
    ROOM19_MAX_FRAMES,
    SETTLE_19_MAX_FRAMES,
    make_map19_controller,
    make_room09_controller,
    make_room19_controller,
    make_settle_09_controller,
    make_settle_19_controller,
    make_settle_29_controller,
    make_settle_39_controller,
    make_settle_3a_controller,
)
from zelda_i.level6_exit75 import (
    level6_exit75_stages,
    level6_exit75_success,
)
from zelda_i.level6_south09 import (
    level6_south09_stages,
    level6_south09_success,
)
from zelda_i.level6_south19 import (
    level6_south19_stages,
    level6_south19_success,
)
from zelda_i.level6_east29 import (
    level6_east29_stages,
    level6_east29_success,
)
from zelda_i.level6_south29 import (
    level6_south29_stages,
    level6_south29_success,
)
from zelda_i.level6_east39 import (
    level6_east39_stages,
    level6_east39_success,
)
from zelda_i.level6_rod import (
    ROD_75_MAX_FRAMES,
    make_rod_75_controller,
)
from zelda_i.level6_stairs09 import (
    STAIRS_09_MAX_FRAMES,
    make_stairs_09_controller,
)
from zelda_i.level6_stairs3a_warp import (
    level6_stairs3a_warp_stages,
    level6_stairs3a_warp_success,
)
from zelda_i.level6_cellar08 import (
    level6_cellar08_stages,
    level6_cellar08_success,
)
from zelda_i.level6_south1d import (
    level6_south1d_stages,
    level6_south1d_success,
)
from zelda_i.level6_west2d import (
    level6_west2d_stages,
    level6_west2d_success,
)
from zelda_i.level6_north2c import (
    level6_north2c_stages,
    level6_north2c_success,
)
from zelda_i.level6_east3a import (
    level6_east3a_stages,
    level6_east3a_success,
)
from zelda_i.level6_north39 import (
    level6_north39_stages,
    level6_north39_success,
)
from zelda_i.level6_inland29 import (
    level6_inland29_stages,
    level6_inland29_success,
)
from zelda_i.level6_west19 import (
    level6_west19_stages,
    level6_west19_success,
)
from zelda_i.level6_south18 import (
    level6_south18_stages,
    level6_south18_success,
)
from zelda_i.ram import (
    ADDR_WHISTLE,
    PASSAGE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

__all__ = [
    "continue_level6_spine",
    "level6_room19_stages",
    "level6_room19_success",
    "level6_clear19_stages",
    "level6_clear19_success",
    "level6_map19_stages",
    "level6_map19_success",
    "level6_room09_stages",
    "level6_room09_success",
    "level6_clear09_stages",
    "level6_clear09_success",
    "level6_stairs09_stages",
    "level6_stairs09_success",
    "level6_rod_stages",
    "level6_rod_success",
    "level6_exit75_stages",
    "level6_exit75_success",
    "level6_south09_stages",
    "level6_south09_success",
    "level6_south19_stages",
    "level6_south19_success",
    "level6_clear29_stages",
    "level6_clear29_success",
    "level6_east29_stages",
    "level6_east29_success",
    "level6_south29_stages",
    "level6_south29_success",
    "level6_settle39_stages",
    "level6_settle39_success",
    "level6_clear39_stages",
    "level6_clear39_success",
    "level6_east39_stages",
    "level6_east39_success",
    "level6_settle3a_stages",
    "level6_settle3a_success",
    "level6_clear3a_stages",
    "level6_clear3a_success",
    "level6_stairs3a_warp_stages",
    "level6_stairs3a_warp_success",
    "level6_cellar08_stages",
    "level6_cellar08_success",
    "level6_south1d_stages",
    "level6_south1d_success",
    "level6_west2d_stages",
    "level6_west2d_success",
    "level6_north2c_stages",
    "level6_north2c_success",
    "level6_east3a_stages",
    "level6_east3a_success",
    "level6_north39_stages",
    "level6_north39_success",
    "level6_inland29_stages",
    "level6_inland29_success",
    "level6_west19_stages",
    "level6_west19_success",
    "level6_south18_stages",
    "level6_south18_success",
]


def level6_room19_stages():
    """0x18 leftover → occupancy east (208,141). Stairs stay a dedicated stop."""
    east = make_room19_controller()
    return (
        ("level6_room_0x19", east, ROOM19_MAX_FRAMES),
    )


def level6_room19_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x19. Map pickup residual. Do not require ADDR_MAP bit."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_MAP_ROOM
        and snap.triforce == 0x1F
    )


def level6_clear19_stages():
    """0x19 leftover → idle census then occupancy-patrol. Do not require Map."""
    settle = make_settle_19_controller()
    fight = make_clear_19_controller()
    return (
        ("level6_settle_0x19", settle, SETTLE_19_MAX_FRAMES),
        ("level6_clear_0x19", fight, ROOM_19_SPEC.max_frames),
    )


def level6_clear19_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x19. Ignore 0x2b/0x40. Do not require Map."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_MAP_ROOM
        and not snap.transitioning
        and not ROOM_19_SPEC.live_enemies(snap)
        and snap.triforce == 0x1F
    )


def level6_map19_stages():
    """0x19 leftover → occupancy onto Map drop. Do not poke ADDR_MAP."""
    hunt = make_map19_controller()
    return (
        ("level6_map_0x19", hunt, MAP_19_MAX_FRAMES),
    )


def level6_map19_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x19 with L6 map bit. Do not grant ADDR_MAP."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_MAP_ROOM
        and not snap.transitioning
        and (snap.map & LEVEL6_MAP_BIT) != 0
        and snap.triforce == 0x1F
    )


def level6_room09_stages():
    """0x19 leftover → skip Map, occupancy KEY-UP. Do not poke the door."""
    north = make_room09_controller()
    return (
        ("level6_room_0x09", north, ROOM09_MAX_FRAMES),
    )


def level6_room09_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready dest north of 0x19. Map optional. Do not require 0x09."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_MAP_ROOM
        and snap.triforce == 0x1F
    )


def level6_clear09_stages():
    """0x09 leftover → idle census then occupancy-patrol. Do not push 0x68."""
    settle = make_settle_09_controller()
    fight = make_clear_09_controller()
    return (
        ("level6_settle_0x09", settle, SETTLE_19_MAX_FRAMES),
        ("level6_clear_0x09", fight, ROOM_09_SPEC.max_frames),
    )


def level6_clear09_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x09. Ignore 0x2b/0x40/0x68. Do not require Rod."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_09_SPEC.room_id
        and not snap.transitioning
        and not ROOM_09_SPEC.live_enemies(snap)
        and snap.triforce == 0x1F
    )


def level6_clear29_stages():
    """0x29 leftover → idle census then occupancy-patrol. No candle/Gohma."""
    settle = make_settle_29_controller()
    fight = make_clear_29_controller()
    return (
        ("level6_settle_0x29", settle, SETTLE_19_MAX_FRAMES),
        ("level6_clear_0x29", fight, ROOM_29_SPEC.max_frames),
    )


def level6_clear29_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x29. Ignore 0x2b/0x40. Do not require stairs/Gohma."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_29_SPEC.room_id
        and not snap.transitioning
        and not ROOM_29_SPEC.live_enemies(snap)
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def level6_stairs09_stages():
    """0x09 leftover → left 0x68 then NE 0x68 south-face. Do not grant Rod."""
    stairs = make_stairs_09_controller()
    return (
        ("level6_stairs_0x09", stairs, STAIRS_09_MAX_FRAMES),
    )


def level6_stairs09_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 cellar or a new L6 play room. Do not require ADDR_ROD."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != ROOM_09_SPEC.room_id
    )


def level6_rod_stages():
    """Cellar 0x75 leftover → idle/walk until ADDR_ROD. Do not grant Rod."""
    return (
        ("level6_rod_0x75", make_rod_75_controller(), ROD_75_MAX_FRAMES),
    )


def level6_rod_success(snap: ZeldaSnapshot) -> bool:
    """ADDR_ROD nonzero. Do not write the rod."""
    return (
        snap.level == LEVEL6
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def level6_settle39_stages():
    """0x39 leftover → idle census. Do not invent Vire/Gohma."""
    settle = make_settle_39_controller()
    return (
        ("level6_settle_0x39", settle, SETTLE_19_MAX_FRAMES),
    )


def level6_settle39_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x39 after idle. TF still 0x1F. Types are RAM, not Gohma."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_DARK_39_ROOM
        and not snap.transitioning
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def level6_clear39_stages():
    """0x39 leftover → occupancy-patrol 5× Vire. Do not invent Gohma."""
    fight = make_clear_39_controller()
    return (
        ("level6_clear_0x39", fight, ROOM_39_SPEC.max_frames),
    )


def level6_clear39_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x39. Ignore 0x2b/0x40. Do not require stairs/Gohma."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_39_SPEC.room_id
        and not snap.transitioning
        and not ROOM_39_SPEC.live_enemies(snap)
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def level6_settle3a_stages():
    """0x3A leftover → idle census. Do not push the center block."""
    settle = make_settle_3a_controller()
    return (
        ("level6_settle_0x3a", settle, SETTLE_19_MAX_FRAMES),
    )


def level6_settle3a_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x3A after idle. TF still 0x1F. Types are RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_BLOCK_3A_ROOM
        and not snap.transitioning
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def level6_clear3a_stages():
    """0x3A leftover → occupancy-patrol. Do not push 0x68 / invent Gohma."""
    fight = make_clear_3a_controller()
    return (
        ("level6_clear_0x3a", fight, ROOM_3A_SPEC.max_frames),
    )


def level6_clear3a_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x3A. Ignore 0x2b/0x40/0x68. Do not require stairs."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_3A_SPEC.room_id
        and not snap.transitioning
        and not ROOM_3A_SPEC.live_enemies(snap)
        and snap.triforce == 0x1F
        and snap.rod != 0
    )


def continue_level6_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L6 suffix after L5 TF. Mutates ``run``; caller returns it."""
    from zelda_i.level6_spine import (
        level6_clear28_stages,
        level6_clear28_success,
        level6_clear38_stages,
        level6_clear38_success,
        level6_clear58_stages,
        level6_clear58_success,
        level6_clear68_stages,
        level6_clear68_success,
        level6_compass_stages,
        level6_compass_success,
        level6_east_key_stages,
        level6_east_key_success,
        level6_entry_stages,
        level6_entry_success,
        level6_gleeok18_stages,
        level6_gleeok18_success,
        level6_keese_stages,
        level6_keese_success,
        level6_postgleeok18_stages,
        level6_postgleeok18_success,
        level6_room18_stages,
        level6_room18_success,
        level6_room28_stages,
        level6_room28_success,
        level6_room38_stages,
        level6_room38_success,
        level6_room48_stages,
        level6_room48_success,
        level6_settle18_stages,
        level6_settle18_success,
        level6_stairs18_stages,
        level6_stairs18_success,
        level6_west_stages,
        level6_west_success,
    )

    hop_kw = dict(room_timer=room_timer, assist=assist, on_frame=on_frame)

    def attach(name: str, stop: str, stages_fn, success_fn, *, dedicated: bool = False) -> bool:
        """Run one hop. False stops the spine (fail, through-match, or stage fail)."""
        if dedicated and through != name:
            return True
        if not run_stages(env, run, stages_fn(), **hop_kw):
            return False
        snap = read_snapshot(env.get_ram())
        run.success = bool(success_fn(snap))
        if not run.success:
            run.failed_stage = stop
            return False
        return through != name

    if not attach(
        "level6-entry",
        "level6_entry_0x79",
        level6_entry_stages,
        lambda snap: level6_entry_success(
            snap, whistle=int(read_u8(env.get_ram(), ADDR_WHISTLE))
        ),
    ):
        return
    keys_before = int(read_snapshot(env.get_ram()).keys)
    if not attach(
        "level6-east-key",
        "level6_east_key_0x7a",
        level6_east_key_stages,
        lambda snap: level6_east_key_success(snap, keys_before=keys_before),
    ):
        return
    hops = (
        ("level6-west", "level6_west_0x78", level6_west_stages, level6_west_success, False),
        ("level6-compass", "level6_compass_0x68", level6_compass_stages, level6_compass_success, False),
        ("level6-clear68", "level6_clear_0x68", level6_clear68_stages, level6_clear68_success, False),
        ("level6-keese", "level6_keese_0x58", level6_keese_stages, level6_keese_success, False),
        ("level6-clear58", "level6_clear_0x58", level6_clear58_stages, level6_clear58_success, False),
        ("level6-room48", "level6_room_0x48", level6_room48_stages, level6_room48_success, False),
        ("level6-room38", "level6_room_0x38", level6_room38_stages, level6_room38_success, False),
        ("level6-clear38", "level6_clear_0x38", level6_clear38_stages, level6_clear38_success, False),
        ("level6-room28", "level6_room_0x28", level6_room28_stages, level6_room28_success, False),
        ("level6-clear28", "level6_clear_0x28", level6_clear28_stages, level6_clear28_success, False),
        ("level6-room18", "level6_room_0x18", level6_room18_stages, level6_room18_success, False),
        ("level6-settle18", "level6_settle_0x18", level6_settle18_stages, level6_settle18_success, False),
        ("level6-gleeok18", "level6_gleeok_0x18", level6_gleeok18_stages, level6_gleeok18_success, False),
        ("level6-postgleeok18", "level6_postgleeok_0x18", level6_postgleeok18_stages, level6_postgleeok18_success, False),
        ("level6-stairs18", "level6_stairs_0x18", level6_stairs18_stages, level6_stairs18_success, True),
        ("level6-room19", "level6_room_0x19", level6_room19_stages, level6_room19_success, False),
        ("level6-clear19", "level6_clear_0x19", level6_clear19_stages, level6_clear19_success, False),
        ("level6-map19", "level6_map_0x19", level6_map19_stages, level6_map19_success, True),
        ("level6-room09", "level6_room_0x09", level6_room09_stages, level6_room09_success, False),
        ("level6-clear09", "level6_clear_0x09", level6_clear09_stages, level6_clear09_success, False),
        ("level6-stairs09", "level6_stairs_0x09", level6_stairs09_stages, level6_stairs09_success, False),
        ("level6-rod", "level6_rod_0x75", level6_rod_stages, level6_rod_success, False),
        ("level6-exit75", "level6_exit_0x75", level6_exit75_stages, level6_exit75_success, False),
        ("level6-south09", "level6_south_0x09", level6_south09_stages, level6_south09_success, False),
        ("level6-south19", "level6_south_0x19", level6_south19_stages, level6_south19_success, False),
        ("level6-clear29", "level6_clear_0x29", level6_clear29_stages, level6_clear29_success, False),
        ("level6-east29", "level6_east_0x29", level6_east29_stages, level6_east29_success, True),
        ("level6-south29", "level6_south_0x29", level6_south29_stages, level6_south29_success, False),
        ("level6-settle39", "level6_settle_0x39", level6_settle39_stages, level6_settle39_success, False),
        ("level6-clear39", "level6_clear_0x39", level6_clear39_stages, level6_clear39_success, False),
        ("level6-east39", "level6_east_0x39", level6_east39_stages, level6_east39_success, False),
        ("level6-settle3a", "level6_settle_0x3a", level6_settle3a_stages, level6_settle3a_success, False),
        ("level6-clear3a", "level6_clear_0x3a", level6_clear3a_stages, level6_clear3a_success, False),
        ("level6-stairs3a-warp", "level6_stairs_0x3a_warp", level6_stairs3a_warp_stages, level6_stairs3a_warp_success, True),
        ("level6-cellar08", "level6_cellar_0x08", level6_cellar08_stages, level6_cellar08_success, True),
        ("level6-south1d", "level6_south_0x1d", level6_south1d_stages, level6_south1d_success, True),
        ("level6-west2d", "level6_west_0x2d", level6_west2d_stages, level6_west2d_success, True),
        ("level6-north2c", "level6_north_0x2c", level6_north2c_stages, level6_north2c_success, True),
        ("level6-east3a", "level6_east_0x3a", level6_east3a_stages, level6_east3a_success, True),
        ("level6-north39", "level6_north39_0x29", level6_north39_stages, level6_north39_success, False),
        ("level6-inland29", "level6_inland_0x29", level6_inland29_stages, level6_inland29_success, False),
        ("level6-west19", "level6_west_0x19", level6_west19_stages, level6_west19_success, False),
        ("level6-south18", "level6_south_0x18", level6_south18_stages, level6_south18_success, False),
    )
    for name, stop, stages_fn, success_fn, dedicated in hops:
        if not attach(name, stop, stages_fn, success_fn, dedicated=dedicated):
            return
