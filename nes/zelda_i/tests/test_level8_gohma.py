"""Unit tests for Level 8 Blue Gohma 0x1E climb dodge (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon.ids import (
    FIREBALL_OBJECT_TYPE,
    GOHMA_OBJECT_TYPE,
    MANHANDLA_PROJECTILE_TYPE,
)
from zelda_i.level6.gohma import EYE_ADDR, FACE_NORTH
from zelda_i.level6.gohma import STAND_Y as L6_STAND_Y
from zelda_i.level8.magic_key import (
    COLUMN_X_MAX,
    COLUMN_X_MIN,
    GOHMA_ROOM_1E,
    STAND_Y,
    make_blue_gohma_1e_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_RUPEES,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": GOHMA_ROOM_1E,
    "x": 120,
    "y": 205,
    "triforce": 0x7F,
    "sword": 3,
    "health": 0x21,
    "keys": 8,
    "bombs": 2,
    "bow": 1,
    "arrows": 1,
    "rupees": 254,
    "selected": 2,
    "facing": FACE_NORTH,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _put_obj(
    ram: np.ndarray,
    slot: int,
    type_id: int,
    hp: int,
    x: int,
    y: int,
) -> None:
    ram[ADDR_OBJ_TYPE + slot] = type_id
    ram[ADDR_OBJ_HP + slot] = hp
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def _plant_gohma(ram: np.ndarray, *, x: int = 119, y: int = 112) -> None:
    _put_obj(ram, 1, GOHMA_OBJECT_TYPE, 96, x, y)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "gohma controller must not write RAM"
    return act


def test_doorway_climbs_up_even_with_fireball() -> None:
    """LEFT/RIGHT are no-ops on the south lip — climb off the door first."""
    ram = _ram(x=120, y=205)
    _plant_gohma(ram)
    _put_obj(ram, 2, MANHANDLA_PROJECTILE_TYPE, 0, 120, 190)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))


def test_climb_dodges_0x56_in_up_band() -> None:
    """Inland climb must not walk UP into a type-0x56 on the column."""
    ram = _ram(x=120, y=181)
    _plant_gohma(ram)
    _put_obj(ram, 2, MANHANDLA_PROJECTILE_TYPE, 0, 120, 170)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb_dodge_fb"
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) in (list(nes_action("LEFT")), list(nes_action("RIGHT")))


def test_climb_dodges_gohma_body_contact() -> None:
    """Do not walk UP into the body — leftover died on the L6 STAND_Y line."""
    ram = _ram(x=120, y=175)
    _plant_gohma(ram, x=120, y=160)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb_dodge_body"
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) in (list(nes_action("LEFT")), list(nes_action("RIGHT")))


def test_south_stand_is_inland_of_l6_fire_band() -> None:
    assert STAND_Y > L6_STAND_Y
    assert STAND_Y <= 189


def test_clear_climb_still_up() -> None:
    ram = _ram(x=120, y=197)
    _plant_gohma(ram, x=119, y=112)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))


def test_on_south_stand_does_not_climb_to_l6_line() -> None:
    ram = _ram(x=120, y=STAND_Y)
    _plant_gohma(ram, x=119, y=112)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "climb"
    assert list(act.action) != list(nes_action("UP"))


def test_open_eye_edge_faces_north_then_fires() -> None:
    ram = _ram(x=128, y=STAND_Y, facing=0x02)
    ram[EYE_ADDR] = 0x70
    _plant_gohma(ram, x=128, y=112)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    turn = _step(ctl, ram)
    assert turn.reason == "face_up"
    assert list(turn.action) == list(nes_action("UP"))
    assert ctl.shots == 0

    ram[ADDR_LINK_FACING] = FACE_NORTH
    shot = _step(ctl, ram)
    assert not ctl.failed
    assert shot.reason == "arrow_shot"
    assert list(shot.action) == list(nes_action("UP", "B"))
    assert ctl.shots == 1


def test_south_lip_inbound_55_does_not_pause_select() -> None:
    """Leftover (120,205): do not START-pause while a statue 0x55 is inbound."""
    ram = _ram(x=120, y=205, selected=1)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 120, 190)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "pause_open"
    assert "pause" not in act.reason
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert ctl._select is not None
    assert ctl._select.frames == 0


def test_lip_does_not_strafe_right_into_se_fireball() -> None:
    """y=189 + Gohma east + SE 0x55 must not emit strafe RIGHT into the stream."""
    ram = _ram(x=120, y=189)
    _plant_gohma(ram, x=160, y=112)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 160, 200)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("RIGHT"))
    assert act.reason != "strafe"
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))


def test_stand_dodges_0x55_like_0x56() -> None:
    ram = _ram(x=120, y=STAND_Y)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 120, 170)
    ctl = make_blue_gohma_1e_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb_dodge_fb"
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) in (list(nes_action("LEFT")), list(nes_action("RIGHT")))


def test_column_hold_does_not_left_dodge_inbound_55() -> None:
    """Leftover (120,181) + 0x55 at x≈120: peel RIGHT toward 128, not LEFT/idle."""
    ram = _ram(x=120, y=STAND_Y, selected=2)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 120, 170)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_idle_action())
    assert act.reason == "column_peel"
    assert act.reason != "pause_open"
    assert COLUMN_X_MIN <= 120 <= COLUMN_X_MAX


def test_column_peel_right_when_55_west_or_north() -> None:
    """(120,181) + 0x55 at (110,181) or (120,160): always RIGHT, never idle."""
    for fx, fy in ((110, 181), (120, 160)):
        ram = _ram(x=120, y=STAND_Y, selected=2)
        _plant_gohma(ram)
        _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, fx, fy)
        ctl = make_blue_gohma_1e_controller()
        ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
        act = _step(ctl, ram)
        assert not ctl.failed, (fx, fy)
        assert list(act.action) == list(nes_action("RIGHT")), (fx, fy, act.reason)
        assert list(act.action) != list(nes_action("LEFT"))
        assert list(act.action) != list(nes_idle_action())
        assert act.reason == "column_peel"


def test_east_edge_shoots_not_idle_on_stand_55() -> None:
    """Leftover (128,181) + 0x55 on/near stand + eye open: UP/UP+B, never idle/LEFT."""
    for fx, fy, facing in (
        (128, 181, FACE_NORTH),
        (120, 181, FACE_NORTH),
        (128, 160, 0x02),
    ):
        ram = _ram(x=128, y=STAND_Y, selected=2, facing=facing)
        ram[EYE_ADDR] = 0x70
        _plant_gohma(ram, x=128, y=112)
        _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, fx, fy)
        ctl = make_blue_gohma_1e_controller()
        ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
        act = _step(ctl, ram)
        assert not ctl.failed, (fx, fy, facing)
        assert list(act.action) != list(nes_idle_action()), (fx, fy, act.reason)
        assert list(act.action) != list(nes_action("LEFT"))
        assert act.reason != "column_hold"
        assert act.reason in ("face_up", "arrow_shot"), (fx, fy, act.reason)
        assert list(act.action) in (
            list(nes_action("UP")),
            list(nes_action("UP", "B")),
        )


def test_east_edge_peels_left_when_eye_closed_on_stand_55() -> None:
    """Leftover (128,181) + 0x55 on/near stand + eye closed: peel LEFT, do not waste arrows."""
    ram = _ram(x=128, y=STAND_Y, selected=2, facing=FACE_NORTH)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 128, 181)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("LEFT"))
    assert act.reason == "column_peel"


def test_east_edge_shot_is_one_frame_then_not_up() -> None:
    """(128,181) eye-open: one-frame UP+B, next frame must not hold UP."""
    ram = _ram(x=128, y=STAND_Y, selected=2, facing=FACE_NORTH)
    ram[EYE_ADDR] = 0x70
    _plant_gohma(ram, x=128, y=112)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 128, 181)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    shot = _step(ctl, ram)
    assert not ctl.failed
    assert shot.reason == "arrow_shot"
    assert list(shot.action) == list(nes_action("UP", "B"))
    nxt = _step(ctl, ram)
    assert not ctl.failed
    assert list(nxt.action) != list(nes_action("UP"))
    assert list(nxt.action) != list(nes_action("UP", "B"))
    assert nxt.reason not in ("arrow_shot", "face_up", "climb")


def test_cooldown_peels_in_band_not_idle() -> None:
    """Leftover (128,181) cooldown + inbound 0x55: LEFT toward 112, not RIGHT."""
    ram = _ram(x=128, y=STAND_Y, selected=2, facing=FACE_NORTH)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 128, 181)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    ctl.cooldown = 20
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("UP", "B"))
    assert act.reason == "column_peel"
    assert COLUMN_X_MIN <= 128 <= COLUMN_X_MAX

    ram[ADDR_LINK_X] = 120
    nxt = _step(ctl, ram)
    assert not ctl.failed
    assert list(nxt.action) == list(nes_action("LEFT"))
    assert list(nxt.action) != list(nes_action("RIGHT"))
    assert list(nxt.action) != list(nes_idle_action())
    assert list(nxt.action) != list(nes_action("UP"))
    assert nxt.reason == "column_peel"
    assert COLUMN_X_MIN <= 120 < COLUMN_X_MAX


def test_north_of_stand_settles_down() -> None:
    """Leftover (128,119): DOWN settle, never hold UP into the body."""
    ram = _ram(x=128, y=119, selected=2, facing=FACE_NORTH)
    _plant_gohma(ram, x=128, y=112)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "settle"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("UP", "B"))
    assert list(act.action) != list(nes_action("LEFT"))


def test_x88_is_not_a_legal_dodge_target() -> None:
    """B=arrows at leftover x=88: recover RIGHT toward the column, never LEFT."""
    ram = _ram(x=88, y=STAND_Y, selected=2)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 88, 170)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) == list(nes_action("RIGHT"))
    assert act.reason == "column_recover"


def test_lip_y189_only_climbs_up() -> None:
    """y=189 is still the lip: inbound 0x55 must not LEFT-dodge off entry x."""
    ram = _ram(x=120, y=189, selected=1)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 120, 175)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert ctl._select.frames == 0


def test_west_box_edge_after_select_does_not_cross_min() -> None:
    """(88,181) inbound 0x55 may dodge after arrows are on B, but not LEFT."""
    ram = _ram(x=88, y=STAND_Y, selected=2)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 88, 170)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("LEFT"))
    assert act.reason != "pause_open"


def test_west_wall_leftover_does_not_sidestep_left() -> None:
    """(50,149) + inbound 0x55 must not walk LEFT into the west statue."""
    ram = _ram(x=50, y=149, selected=1)
    _plant_gohma(ram, x=160, y=112)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 50, 149)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("LEFT"))
    assert act.reason != "pause_open"
    assert "pause" not in act.reason


def test_inland_stand_may_pause_select_with_inbound_55() -> None:
    """y=181 x=120 is not the lip: pause-select may start even in 0x55 fire."""
    ram = _ram(x=120, y=STAND_Y, selected=1)
    _plant_gohma(ram)
    _put_obj(ram, 2, FIREBALL_OBJECT_TYPE, 0, 120, 170)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "pause_open"
    assert list(act.action) == list(nes_action("START"))


def test_gohma_skips_pause_when_arrows_already_on_b() -> None:
    """Arriving 0x1E with B=arrows: no pause_open; lip action is climb."""
    ram = _ram(x=120, y=205, selected=2)
    _plant_gohma(ram)
    ctl = make_blue_gohma_1e_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "pause_open"
    assert "pause" not in act.reason
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))


def test_darknut_does_not_pause_select_on_live_manhandla() -> None:
    """0x2E entry with a live body must fight, not START (h6 leftover)."""
    from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
    from zelda_i.level8.north_column import ROOM_MAP_MANHANDLA
    from zelda_i.level8.path import make_darknut_key_controller

    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=189, selected=1)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 128, 120, 141)
    ctl = make_darknut_key_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "pause_open"
    assert "pause" not in act.reason


def test_darknut_holds_north_door_until_arrows() -> None:
    """(120,93) B=bombs + Manhandla off-corridor: pause-select, do not leave."""
    from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
    from zelda_i.level8.north_column import ROOM_MAP_MANHANDLA
    from zelda_i.level8.path import make_darknut_key_controller

    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=93, selected=1)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 64, 180, 120)
    ctl = make_darknut_key_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "pause_open"
    assert list(act.action) == list(nes_action("START"))
    assert act.reason != "combat_north_door"


def test_darknut_fights_south_when_manhandla_on_corridor() -> None:
    """Inland 0x2E with a live body in range: slash, bombs stay on B."""
    from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
    from zelda_i.level8.north_column import ROOM_MAP_MANHANDLA
    from zelda_i.level8.path import make_darknut_key_controller

    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=140, selected=1)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 64, 135, 140)
    ctl = make_darknut_key_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "pause_open"
    assert "pause" not in act.reason
    assert act.reason == "combat_slash"


def test_darknut_does_not_pause_select_on_0x2e_south_lip() -> None:
    """Cleared 0x2E at y=189: climb inland, do not START on the lip."""
    from zelda_i.level8.north_column import ROOM_MAP_MANHANDLA
    from zelda_i.level8.path import make_darknut_key_controller

    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=189, selected=1)
    ctl = make_darknut_key_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "pause_open"
    assert "pause" not in act.reason
    assert act.reason == "climb"
    assert list(act.action) == list(nes_action("UP"))


def test_darknut_selects_arrows_in_0x2e_before_gohma() -> None:
    """0x2E cleared + y<=181 + B!=arrows: pause-select before the 0x1E walk."""
    from zelda_i.level8.north_column import ROOM_MAP_MANHANDLA
    from zelda_i.level8.path import make_darknut_key_controller

    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=STAND_Y, selected=1)
    ctl = make_darknut_key_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "pause_open"
    assert list(act.action) == list(nes_action("START"))


def test_controller_does_not_poke_arrows() -> None:
    ram = _ram(bow=1, arrows=1)
    _plant_gohma(ram)
    ctl = make_blue_gohma_1e_controller()
    _step(ctl, ram)
    assert ctl.writes == 0
    assert ctl.report()["poked_arrows"] is False
    assert ctl.report()["route_eligible"] is False
    assert int(ram[ADDR_BOW]) == 1
    assert int(ram[ADDR_ARROWS]) == 1
    assert int(ram[ADDR_RUPEES]) == 254
