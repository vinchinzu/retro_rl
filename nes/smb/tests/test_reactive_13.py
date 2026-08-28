"""No-ROM gates for isolated 1-3 (dash-level completion, not AreaNumber)."""

from __future__ import annotations

from retro_harness.platformer.level_config import get_level_config

import smb.platformer_levels  # noqa: F401
from smb.ram import SmbSnapshot
from smb.reactive_13 import BunnyHopPolicy, JumpWindowPolicy, is_1_4_control
from smb.routes import ROUTE_ALL_EXITS, ROUTE_WARP_ANY_PERCENT


def _snap(**kwargs: object) -> SmbSnapshot:
    player_x = int(kwargs.get("player_x", 40))
    timer = int(kwargs.get("timer", 400))
    return SmbSnapshot(
        frame=0,
        player_state=int(kwargs.get("player_state", 8)),
        player_x=player_x,
        player_y=int(kwargs.get("player_y", 176)),
        x_page=player_x // 256,
        x_offset=player_x % 256,
        lives=2,
        world=int(kwargs.get("world", 0)),
        level=int(kwargs.get("level", 2)),
        level_id=int(kwargs.get("world", 0)) * 4 + int(kwargs.get("level", 2)),
        oper_mode=1,
        player_power=0,
        timer_hundreds=timer // 100,
        timer=timer,
        area_pointer=int(kwargs.get("area_pointer", 38)),
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
        level_number=int(kwargs.get("level_number", kwargs.get("dash_level", 2))),
    )


def test_smb_1_3_levelconfig_uses_dash_not_area_number() -> None:
    cfg = get_level_config("smb_1_3")
    assert cfg.start_state == "Level1_3"
    assert cfg.target_level_id == 2
    assert cfg.completion_level_ids == [3]
    ug = {
        "world": 0,
        "level": 2,  # 1-2 UG AreaNumber
        "dash_level": 1,
        "x_page": 0,
        "x_offset": 40,
        "screen_page": 0,
        "screen_x_off": 0,
        "timer_hundreds": 3,
        "timer_tens": 5,
        "timer_ones": 5,
    }
    out = dict(ug)
    cfg.apply_computed(out)
    assert out["level_id"] == 1  # still 1-2, not a 1-3 complete
    start = dict(ug)
    start["dash_level"] = 2
    start["level"] = 3  # real 1-3 AreaNumber
    out13 = dict(start)
    cfg.apply_computed(out13)
    assert out13["level_id"] == 2
    done = dict(start)
    done["dash_level"] = 3
    out14 = dict(done)
    cfg.apply_computed(out14)
    assert out14["level_id"] == 3
    assert 3 in cfg.completion_level_ids


def test_smb_1_2_warp_config_still_uses_area_number() -> None:
    cfg = get_level_config("smb_1_2")
    assert 2 in cfg.level_id_aliases
    assert cfg.completion_level_ids == [12]
    values = {
        "world": 0,
        "level": 2,
        "dash_level": 1,
        "x_page": 0,
        "x_offset": 40,
        "screen_page": 0,
        "screen_x_off": 0,
        "timer_hundreds": 3,
        "timer_tens": 0,
        "timer_ones": 0,
    }
    out = dict(values)
    cfg.apply_computed(out)
    assert out["level_id"] == 2  # AreaNumber alias for UG


def test_1_4_control_is_dash_3() -> None:
    assert is_1_4_control(_snap(dash_level=3, level_number=3, player_x=40))
    assert not is_1_4_control(_snap(dash_level=2, level_number=2, player_x=40))
    assert not is_1_4_control(_snap(dash_level=3, level_number=3, player_x=200))


def test_bunny_hop_jumps_when_physics_grounded() -> None:
    pol = BunnyHopPolicy(hold_a=4)
    air = pol.step(_snap(player_x=80, dash_level=2), on_ground=False)
    assert int(air.action[8]) == 0
    hop = pol.step(_snap(player_x=80, dash_level=2), on_ground=True)
    assert int(hop.action[8]) == 1
    assert pol.jump_held == 3


def test_jump_window_fires_rba_inside_window() -> None:
    pol = JumpWindowPolicy(jump_xs=[400], hold_a=4)
    before = pol.step(_snap(player_x=300, dash_level=2), on_ground=True)
    assert int(before.action[8]) == 0
    hop = pol.step(_snap(player_x=380, dash_level=2), on_ground=True)
    assert int(hop.action[8]) == 1
    assert int(hop.action[7]) == 1
    assert 400 in pol.used


def test_all_exits_1_2_policy_is_flag_not_warp() -> None:
    normal = ROUTE_ALL_EXITS.exits[1]
    warp = ROUTE_WARP_ANY_PERCENT.exits[1]
    assert normal.policy_id == "smb_1_2_flag"
    assert warp.policy_id == "smb_1_2_warp"
    assert normal.accepts_successor(_snap(world=0, level=2, dash_level=2, level_number=2))
    assert not normal.accepts_successor(_snap(world=3, level=0, dash_level=0, level_number=0))
