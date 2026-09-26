"""Post-Gleeok L6 hop rows. Factories stay in their modules; no first-half import."""

from __future__ import annotations

from zelda_i.level6.cellar08 import level6_cellar08_success, make_cellar08_controller
from zelda_i.dungeon.door_hop import door_hop_stages, door_hop_success
from zelda_i.level6.door_hop import (
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
)
from zelda_i.level6.dungeon import ROOM_09_SPEC, ROOM_39_SPEC, ROOM_3A_SPEC
from zelda_i.level6.exit75 import make_exit75_controller
from zelda_i.level6.finish import (
    level6_exit_success,
    level6_heart_success,
    level6_north0c_success,
    level6_success,
    make_exit_controller,
    make_heart_controller,
    make_north0c_controller,
    make_shard_controller,
)
from zelda_i.level6.gohma import level6_gohma_success, make_gohma_controller
from zelda_i.level6.hops import (
    door_row,
    fight_hop,
    ok6,
    one_hop,
    rod_cellar_ok,
    settle_fight,
    stairs_or_play,
)
from zelda_i.level6.inland29 import level6_inland29_success, make_inland29_controller
from zelda_i.level6.north39 import make_north39_controller
from zelda_i.level6.overworld import (
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.level6.rod import make_rod_75_controller
from zelda_i.level6.room19 import (
    make_settle_09_controller,
    make_settle_39_controller,
    make_settle_3a_controller,
)
from zelda_i.level6.stairs09 import make_stairs_09_controller
from zelda_i.level6.stairs3a_warp import (
    level6_stairs3a_warp_success,
    make_stairs_3a_warp_controller,
)
from zelda_i.spine.hops import SpineHop

__all__ = ["l6_suffix_hops"]


def _warp_cellar_stages():
    """clear 0x3A → position-warp → cross cellar 0x08 to play 0x1D."""
    warp = make_stairs_3a_warp_controller()
    cellar = make_cellar08_controller()
    return (
        ("level6_stairs_0x3a_warp", warp, warp.max_frames),
        ("level6_cellar_0x08", cellar, cellar.max_frames),
    )


def _cellar08_stages():
    return _warp_cellar_stages()


def _south1d_stages():
    return (*_warp_cellar_stages(), *door_hop_stages(SOUTH1D_SPEC))


def _west2d_stages():
    return (*_south1d_stages(), *door_hop_stages(WEST2D_SPEC))


def _north2c_stages():
    return (*_west2d_stages(), *door_hop_stages(NORTH2C_SPEC))


def _gohma_stages(*, poke_arrows: bool = False):
    from zelda_i.dungeon.pause_select import B_SLOT_ARROWS, PauseSelectController

    select = PauseSelectController(want=B_SLOT_ARROWS, name="arrows")
    ctl = make_gohma_controller(poke_arrows=poke_arrows)
    return (
        *_north2c_stages(),
        ("level6_select_arrows", select, select.max_frames),
        ("level6_gohma_0x1c", ctl, ctl.max_frames),
    )


def _heart_stages(*, poke_arrows: bool = False):
    ctl = make_heart_controller()
    return (*_gohma_stages(poke_arrows=poke_arrows), (ctl.spec_id, ctl, ctl.max_frames))


def _north0c_stages(*, poke_arrows: bool = False):
    ctl = make_north0c_controller()
    return (*_heart_stages(poke_arrows=poke_arrows), (ctl.spec_id, ctl, ctl.max_frames))


def _level6_stages(*, poke_arrows: bool = False):
    ctl = make_shard_controller()
    return (*_north0c_stages(poke_arrows=poke_arrows), (ctl.spec_id, ctl, ctl.max_frames))


def _exit_stages(*, poke_arrows: bool = False):
    ctl = make_exit_controller()
    return (*_level6_stages(poke_arrows=poke_arrows), (ctl.spec_id, ctl, ctl.max_frames))


def _door_success(spec):
    return lambda snap, s=spec, **_: door_hop_success(s, snap)


def l6_suffix_hops(
    *, poke_arrows: bool = False, require_prior_tf: bool = True
) -> tuple[SpineHop, ...]:
    tf1f = dict(tf_eq=0x1F) if require_prior_tf else {}
    rod1f = dict(rod=True, **(dict(tf_eq=0x1F) if require_prior_tf else {}))
    gohma = dict(poke_arrows=poke_arrows)
    return (
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
            lambda snap, r=require_prior_tf, **_: stairs_or_play(
                snap, not_screen=ROOM_09_SPEC.room_id, require_prior_tf=r
            ),
        ),
        one_hop(
            "level6-rod",
            "level6_rod_0x75",
            make_rod_75_controller,
            lambda snap, r=require_prior_tf, **_: rod_cellar_ok(
                snap, require_prior_tf=r
            ),
        ),
        one_hop(
            "level6-exit75",
            "level6_exit_0x75",
            make_exit75_controller,
            ok6(screen=LEVEL6_ROD_WIZZ_ROOM, **rod1f),
        ),
        door_row("level6-south09", SOUTH09_SPEC),
        door_row("level6-south19", SOUTH19_SPEC),
        door_row("level6-east29", EAST29_SPEC, dedicated=True),
        # 0x29's S door is open in the ROM door table: walk it, no clear.
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
        SpineHop(
            "level6-south1d",
            SOUTH1D_SPEC.spec_id,
            _south1d_stages,
            _door_success(SOUTH1D_SPEC),
            dedicated=True,
        ),
        SpineHop(
            "level6-west2d",
            WEST2D_SPEC.spec_id,
            _west2d_stages,
            _door_success(WEST2D_SPEC),
            dedicated=True,
        ),
        SpineHop(
            "level6-north2c",
            NORTH2C_SPEC.spec_id,
            _north2c_stages,
            _door_success(NORTH2C_SPEC),
            dedicated=True,
        ),
        SpineHop(
            "level6-gohma",
            "level6_gohma_0x1c",
            lambda: _gohma_stages(**gohma),
            level6_gohma_success,
            dedicated=True,
        ),
        SpineHop(
            "level6-heart",
            "level6_heart_0x1c",
            lambda: _heart_stages(**gohma),
            level6_heart_success,
            dedicated=True,
        ),
        SpineHop(
            "level6-north0c",
            "level6_north_0x0c",
            lambda: _north0c_stages(**gohma),
            level6_north0c_success,
            dedicated=True,
        ),
        SpineHop(
            "level6",
            "level6_triforce_0x20",
            lambda: _level6_stages(**gohma),
            level6_success,
            dedicated=True,
        ),
        SpineHop(
            "level6-exit",
            "level6_exit_ow",
            lambda: _exit_stages(**gohma),
            level6_exit_success,
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
