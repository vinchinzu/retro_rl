"""Survival-spine L4 hops through Gleeok enter; TF suffix stays a library call."""

from __future__ import annotations

from zelda_i.dungeon.engine import DungeonPhase
from zelda_i.level4.dungeon import (
    LEVEL4,
    LEVEL4_MAP_BIT,
    ROOM_12_SPEC,
    ROOM_31_SPEC,
    ROOM_32_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_GLEEOK_13,
    ROOM_L4_KEY_01,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_MAP_21,
    ROOM_L4_MID_11,
    ROOM_L4_NORTH_30,
    ROOM_L4_STEPLADDER,
    ROOM_L4_VIRES_12,
    ROOM_L4_VIRES_50,
    ROOM_L4_WATER_NORTH_20,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4.bomb11 import level4_bomb11_stages
from zelda_i.level4.clear12 import level4_clear12_stages
from zelda_i.level4.exit60 import level4_exit60_stages
from zelda_i.level4.gleeok13 import attach_level4_tf_suffix, level4_gleeok13_stages
from zelda_i.level4.key01 import level4_key01_stages
from zelda_i.level4.key40 import make_room_40_key_controller
from zelda_i.level4.keyup20 import level4_keyup20_stages
from zelda_i.level4.map21 import level4_map21_stages
from zelda_i.level4.mappick import level4_mappick_stages
from zelda_i.level4.maze_path import (
    make_maze_31_east_controller,
    make_maze_31_inland_controller,
    make_north_40_controller,
)
from zelda_i.level4.overworld import (
    LEVEL4_ENTRY_ROOM,
    POST_L3_PATH_MAX_FRAMES,
    POST_L3_SETTLE_MAX_FRAMES,
    OverworldToLevel4Controller,
    PostL3TriforceSettleController,
)
from zelda_i.level4.path import (
    make_bomb_61_north_controller,
    make_entry_up_controller,
    make_left_50_controller,
    make_room_31_clear_controller,
    make_room_32_clear_controller,
    make_room_50_clear_controller,
    make_room_51_key_controller,
)
from zelda_i.level4.stepladder import (
    make_key_right_31_controller,
    make_north_30_controller,
    make_room_30_clear_controller,
    make_stepladder_controller,
)
from zelda_i.level4.west31 import level4_west31_stages
from zelda_i.ram import PASSAGE_MODE, ZeldaSnapshot
from zelda_i.spine.hops import SpineHop, attach_hops, ready

L4_STOPS: dict[str, str] = {
    "level4": "level4_triforce_0x08",
    "level4-entry": "level4_entry_0x71",
    "level4-key": "level4_natural_key_0x51",
    "level4-clear50": "level4_clear_0x50",
    "level4-room40-key": "level4_natural_key_0x40",
    "level4-room30": "level4_enter_0x30",
    "level4-room31": "level4_enter_0x31",
    "level4-clear31": "level4_clear_0x31",
    "level4-room32": "level4_enter_0x32",
    "level4-clear32": "level4_clear_0x32",
    "level4-stepladder": "level4_stepladder_0x60",
    "level4-exit60": "level4_exit_0x60",
    "level4-west31": "level4_west_0x31",
    "level4-keyup20": "level4_key_up_0x20",
    "level4-room21": "level4_enter_0x21",
    "level4-map": "level4_map_pickup_0x21",
    "level4-bomb11": "level4_enter_0x11",
    "level4-key01": "level4_natural_key_0x01",
    "level4-clear12": "level4_clear_0x12",
    "level4-gleeok13": "level4_enter_0x13",
}

__all__ = [
    "L4_STOPS",
    "attach_level4_tf_suffix",
    "continue_level4_spine",
]


def _as_fight(factory):
    ctl = factory()
    ctl.phase = DungeonPhase.FIGHT
    return ctl


def _first_key_stages():
    return (
        ("level4_entry_up_0x61", make_entry_up_controller(), 4000),
        (
            "level4_bomb_north_0x61",
            make_bomb_61_north_controller(clear_vires=True),
            20000,
        ),
        ("level4_key_0x51", _as_fight(make_room_51_key_controller), ROOM_51_SPEC.max_frames),
    )


def _room50_stages():
    return (
        ("level4_left_0x50", make_left_50_controller(), 2500),
        (
            "level4_clear_0x50",
            _as_fight(make_room_50_clear_controller),
            ROOM_50_SPEC.max_frames,
        ),
    )


def _room40_key_stages():
    return (
        ("level4_north_0x40", make_north_40_controller(), 10000),
        ("level4_key_0x40", make_room_40_key_controller(), 25000),
    )


def _key_right_31_stages():
    return (
        ("level4_clear_0x30", make_room_30_clear_controller(), 20000),
        (
            "level4_key_right_0x31",
            make_key_right_31_controller(clear_vires=False),
            4000,
        ),
    )


def _clear_31_stages():
    return (
        ("level4_inland_0x31", make_maze_31_inland_controller(), 4000),
        (
            "level4_clear_0x31",
            _as_fight(make_room_31_clear_controller),
            ROOM_31_SPEC.max_frames,
        ),
    )


def _clear_32_stages():
    return (
        (
            "level4_clear_0x32",
            _as_fight(make_room_32_clear_controller),
            ROOM_32_SPEC.max_frames,
        ),
    )


def _stepladder_stages():
    ctl = make_stepladder_controller(clear_first=False)
    return (("level4_stepladder", ctl, ctl.max_frames),)


def _stepladder_ok(snap: ZeldaSnapshot, **_) -> bool:
    return snap.level == LEVEL4 and snap.ladder > 0 and (
        snap.screen == ROOM_L4_STEPLADDER or snap.mode == PASSAGE_MODE
    )


def _ok(**kw):
    return ready(level=LEVEL4, **kw)


def l4_hops(*, topup_bombs, spine_fields) -> tuple[SpineHop, ...]:
    def set_entry(env, run, snap):
        if run.success:
            run.l4_entry = spine_fields(snap)

    def bombs(env, run):
        topup_bombs(env, run)

    return (
        SpineHop(
            "level4-entry",
            "level4_entry_0x71",
            (
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
            ),
            _ok(screen=LEVEL4_ENTRY_ROOM),
            after=set_entry,
        ),
        SpineHop(
            "level4-key",
            "level4_natural_key_0x51",
            _first_key_stages,
            _ok(screen=ROOM_L4_KEESE_KEY_51, spec=ROOM_51_SPEC, keys_cmp="gt"),
            capture_keys=True,
            before=bombs,
        ),
        SpineHop(
            "level4-clear50",
            "level4_clear_0x50",
            _room50_stages,
            _ok(screen=ROOM_L4_VIRES_50, spec=ROOM_50_SPEC),
        ),
        SpineHop(
            "level4-room40-key",
            "level4_natural_key_0x40",
            _room40_key_stages,
            _ok(screen=ROOM_L4_ZOLS_40, spec=ROOM_40_SPEC, keys_cmp="gt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-room30",
            "level4_enter_0x30",
            (("level4_north_0x30", make_north_30_controller(), 4000),),
            _ok(screen=ROOM_L4_NORTH_30),
        ),
        SpineHop(
            "level4-room31",
            "level4_enter_0x31",
            _key_right_31_stages,
            _ok(screen=ROOM_L4_EAST_31, keys_cmp="lt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-clear31",
            "level4_clear_0x31",
            _clear_31_stages,
            _ok(screen=ROOM_L4_EAST_31, spec=ROOM_31_SPEC),
        ),
        SpineHop(
            "level4-room32",
            "level4_enter_0x32",
            (("level4_east_0x32", make_maze_31_east_controller(), 4000),),
            _ok(screen=ROOM_L4_EAST_32),
        ),
        SpineHop(
            "level4-clear32",
            "level4_clear_0x32",
            _clear_32_stages,
            _ok(screen=ROOM_L4_EAST_32, spec=ROOM_32_SPEC),
        ),
        SpineHop(
            "level4-stepladder",
            "level4_stepladder_0x60",
            _stepladder_stages,
            _stepladder_ok,
        ),
        SpineHop(
            "level4-exit60",
            "level4_exit_0x60",
            level4_exit60_stages,
            _ok(screen=ROOM_L4_EAST_32, item="ladder"),
        ),
        SpineHop(
            "level4-west31",
            "level4_west_0x31",
            level4_west31_stages,
            _ok(screen=ROOM_L4_EAST_31, item="ladder"),
        ),
        SpineHop(
            "level4-keyup20",
            "level4_key_up_0x20",
            level4_keyup20_stages,
            _ok(screen=ROOM_L4_WATER_NORTH_20, item="ladder"),
        ),
        SpineHop(
            "level4-room21",
            "level4_enter_0x21",
            level4_map21_stages,
            _ok(screen=ROOM_L4_MAP_21, item="ladder"),
        ),
        SpineHop(
            "level4-map",
            "level4_map_pickup_0x21",
            level4_mappick_stages,
            _ok(screen=ROOM_L4_MAP_21, map_bit=LEVEL4_MAP_BIT),
        ),
        SpineHop(
            "level4-bomb11",
            "level4_enter_0x11",
            level4_bomb11_stages,
            _ok(screen=ROOM_L4_MID_11),
            before=bombs,
        ),
        SpineHop(
            "level4-key01",
            "level4_natural_key_0x01",
            level4_key01_stages,
            _ok(screen=ROOM_L4_KEY_01, keys_cmp="gt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-clear12",
            "level4_clear_0x12",
            level4_clear12_stages,
            _ok(screen=ROOM_L4_VIRES_12, spec=ROOM_12_SPEC),
        ),
        SpineHop(
            "level4-gleeok13",
            "level4_enter_0x13",
            level4_gleeok13_stages,
            _ok(screen=ROOM_L4_GLEEOK_13),
        ),
    )


def continue_level4_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    topup_bombs,
    spine_fields,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L4 suffix after L3 TF. Mutates ``run``; caller returns it."""
    attach_hops(
        env,
        run,
        l4_hops(topup_bombs=topup_bombs, spine_fields=spine_fields),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    if not run.success or (through in L4_STOPS and through != "level4"):
        return
    attach_level4_tf_suffix(env, run, assist=assist)
