"""Post-Gleeok L6 hop rows. Factories stay in their modules; no first-half import."""

from __future__ import annotations

from zelda_i.level6_cellar08 import level6_cellar08_success, make_cellar08_controller
from zelda_i.level6_door_hop import (
    EAST29_SPEC,
    EAST39_SPEC,
    NORTH2C_SPEC,
    SOUTH09_SPEC,
    SOUTH18_SPEC,
    SOUTH19_SPEC,
    SOUTH1D_SPEC,
    SOUTH29_SPEC,
    WEST19_SPEC,
    WEST2D_SPEC,
    door_hop_stages,
)
from zelda_i.level6_dungeon import (
    LEVEL6_MAP_BIT,
    ROOM_09_SPEC,
    ROOM_19_SPEC,
    ROOM_29_SPEC,
    ROOM_39_SPEC,
    ROOM_3A_SPEC,
)
from zelda_i.level6_east3a import level6_east3a_success, make_east3a_controller
from zelda_i.level6_exit75 import make_exit75_controller
from zelda_i.level6_gohma import level6_gohma_success, make_gohma_controller
from zelda_i.level6_hops import (
    door_row,
    fight_hop,
    ok6,
    one_hop,
    settle_fight,
    stairs_or_play,
)
from zelda_i.level6_inland29 import level6_inland29_success, make_inland29_controller
from zelda_i.level6_north39 import make_north39_controller
from zelda_i.level6_overworld import (
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.level6_rod import make_rod_75_controller
from zelda_i.level6_room19 import (
    make_map19_controller,
    make_room09_controller,
    make_room19_controller,
    make_settle_09_controller,
    make_settle_19_controller,
    make_settle_29_controller,
    make_settle_39_controller,
    make_settle_3a_controller,
)
from zelda_i.level6_stairs09 import make_stairs_09_controller
from zelda_i.level6_stairs3a_warp import (
    level6_stairs3a_warp_success,
    make_stairs_3a_warp_controller,
)
from zelda_i.spine_hops import SpineHop

__all__ = ["l6_suffix_hops"]


def _gohma_stages():
    ctl = make_gohma_controller()
    return (
        *door_hop_stages(NORTH2C_SPEC),
        ("level6_gohma_0x1c", ctl, ctl.max_frames),
    )


def _cellar08_stages():
    warp = make_stairs_3a_warp_controller()
    cellar = make_cellar08_controller()
    return (
        ("level6_stairs_0x3a_warp", warp, warp.max_frames),
        ("level6_cellar_0x08", cellar, cellar.max_frames),
    )


def l6_suffix_hops() -> tuple[SpineHop, ...]:
    tf1f = dict(tf_eq=0x1F)
    rod1f = dict(tf_eq=0x1F, rod=True)
    return (
        one_hop(
            "level6-room19",
            "level6_room_0x19",
            make_room19_controller,
            ok6(screen=LEVEL6_MAP_ROOM, **tf1f),
        ),
        settle_fight(
            "level6-clear19",
            "level6_clear_0x19",
            make_settle_19_controller,
            "level6_settle_0x19",
            ROOM_19_SPEC,
            **tf1f,
        ),
        one_hop(
            "level6-map19",
            "level6_map_0x19",
            make_map19_controller,
            ok6(screen=LEVEL6_MAP_ROOM, map_bit=LEVEL6_MAP_BIT, **tf1f),
            dedicated=True,
        ),
        one_hop(
            "level6-room09",
            "level6_room_0x09",
            make_room09_controller,
            ok6(not_screen=LEVEL6_MAP_ROOM, **tf1f),
        ),
        settle_fight(
            "level6-clear09",
            "level6_clear_0x09",
            make_settle_09_controller,
            "level6_settle_0x09",
            ROOM_09_SPEC,
            **tf1f,
        ),
        one_hop(
            "level6-stairs09",
            "level6_stairs_0x09",
            make_stairs_09_controller,
            lambda snap, **_: stairs_or_play(snap, not_screen=ROOM_09_SPEC.room_id),
        ),
        one_hop(
            "level6-rod",
            "level6_rod_0x75",
            make_rod_75_controller,
            ok6(rod=True, **tf1f),
        ),
        one_hop(
            "level6-exit75",
            "level6_exit_0x75",
            make_exit75_controller,
            ok6(screen=LEVEL6_ROD_WIZZ_ROOM, **rod1f),
        ),
        door_row("level6-south09", SOUTH09_SPEC),
        door_row("level6-south19", SOUTH19_SPEC),
        settle_fight(
            "level6-clear29",
            "level6_clear_0x29",
            make_settle_29_controller,
            "level6_settle_0x29",
            ROOM_29_SPEC,
            **rod1f,
        ),
        door_row("level6-east29", EAST29_SPEC, dedicated=True),
        door_row("level6-south29", SOUTH29_SPEC),
        one_hop(
            "level6-settle39",
            "level6_settle_0x39",
            make_settle_39_controller,
            ok6(screen=LEVEL6_DARK_39_ROOM, **rod1f),
        ),
        fight_hop("level6-clear39", "level6_clear_0x39", ROOM_39_SPEC, **rod1f),
        door_row("level6-east39", EAST39_SPEC),
        one_hop(
            "level6-settle3a",
            "level6_settle_0x3a",
            make_settle_3a_controller,
            ok6(screen=LEVEL6_BLOCK_3A_ROOM, **rod1f),
        ),
        fight_hop("level6-clear3a", "level6_clear_0x3a", ROOM_3A_SPEC, **rod1f),
        one_hop(
            "level6-stairs3a-warp",
            "level6_stairs_0x3a_warp",
            make_stairs_3a_warp_controller,
            level6_stairs3a_warp_success,
            dedicated=True,
        ),
        SpineHop(
            "level6-cellar08",
            "level6_cellar_0x08",
            _cellar08_stages,
            level6_cellar08_success,
            dedicated=True,
        ),
        door_row("level6-south1d", SOUTH1D_SPEC, dedicated=True),
        door_row("level6-west2d", WEST2D_SPEC, dedicated=True),
        door_row("level6-north2c", NORTH2C_SPEC, dedicated=True),
        SpineHop(
            "level6-gohma",
            "level6_gohma_0x1c",
            _gohma_stages,
            level6_gohma_success,
            dedicated=True,
        ),
        one_hop(
            "level6-east3a",
            "level6_east_0x3a",
            make_east3a_controller,
            level6_east3a_success,
            dedicated=True,
        ),
        one_hop(
            "level6-north39",
            "level6_north39_0x29",
            make_north39_controller,
            ok6(screen=LEVEL6_DARK_29_ROOM, **rod1f),
        ),
        one_hop(
            "level6-inland29",
            "level6_inland_0x29",
            make_inland29_controller,
            level6_inland29_success,
        ),
        door_row("level6-west19", WEST19_SPEC),
        door_row("level6-south18", SOUTH18_SPEC),
    )
