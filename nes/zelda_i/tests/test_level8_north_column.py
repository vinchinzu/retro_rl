"""Durable L8 north-column controllers: fixture-live, fail-closed, no RAM writes.

No emulator.  Fake snapshots in 0x7E/0x6E/0x5E/… must emit a non-idle action
with a semantic reason.  Wrong level/room fail closed.  Controllers never
write RAM.  The Gohma body in 0x1E is a dest, not a fight.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
from zelda_i.level8.hops import l8_hops
from zelda_i.level8.north_column import (
    DARKNUT_KEY_ROOMS,
    NORTH_MANHANDLA_ROOMS,
    ROOM_BLUE_DARKNUTS,
    ROOM_DARKNUT_KEY,
    ROOM_ENTRY,
    ROOM_GOHMA,
    ROOM_MANHANDLA,
    ROOM_MAP_MANHANDLA,
    ROOM_SHUTTER,
    TYPE_0C,
    make_darknut_key_controller,
    make_north_manhandla_controller,
)
from zelda_i.level8.path import (
    Level8BlueGohma1EController,
    Level8DarknutKeyController,
    Level8MagicKeyStairsController,
    Level8NorthManhandlaController,
    UnverifiedLevel8PathController,
    make_blue_gohma_controller,
    make_darknut_key_controller as path_make_darknut,
    make_north_manhandla_controller as path_make_manhandla,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": ROOM_ENTRY,
    "x": 120,
    "y": 205,
    "triforce": 0x7F,
    "sword": 3,
    "health": 0xBB,
    "keys": 9,
    "bombs": 8,
    "selected": 4,
    "room_item": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _put_obj(ram: np.ndarray, slot: int, type_id: int, hp: int, x: int, y: int) -> None:
    ram[ADDR_OBJ_TYPE + slot] = type_id
    ram[ADDR_OBJ_HP + slot] = hp
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "north-column controllers must not write RAM"
    return act


def _busy(act) -> None:
    assert list(act.action) != list(nes_idle_action()), act.reason
    assert act.reason
    assert "blocked_unverified" not in act.reason


def test_factories_are_live_policies_not_unverified() -> None:
    man = path_make_manhandla()
    dark = path_make_darknut()
    assert isinstance(man, Level8NorthManhandlaController)
    assert isinstance(dark, Level8DarknutKeyController)
    assert not isinstance(man, UnverifiedLevel8PathController)
    assert man.route_eligible is False
    assert dark.route_eligible is False
    assert man.report()["evidence"] == "fixture-live"
    assert dark.report()["natural_entry"] is False
    assert man.report()["writes"] == 0
    assert NORTH_MANHANDLA_ROOMS == frozenset({0x7E, 0x6E, 0x5E})
    assert DARKNUT_KEY_ROOMS == frozenset({0x5E, 0x4E, 0x3E, 0x2E, 0x1E})


def test_spine_magic_key_stages_wire_the_live_gohma_kill() -> None:
    hops = l8_hops(SimpleNamespace(get_ram=lambda: _ram()))
    stages = hops[1].stages()
    names = [name for name, _, _ in stages]
    assert names[:3] == [
        "level8_north_manhandla_bomb",
        "level8_darknut_key_up",
        "level8_blue_gohma",
    ]
    assert isinstance(stages[0][1], Level8NorthManhandlaController)
    assert isinstance(stages[1][1], Level8DarknutKeyController)
    # rr-6o7.2: the 0x1E arrow kill is promoted from probe_l8_1e_gohma.
    assert isinstance(stages[2][1], Level8BlueGohma1EController)
    assert not isinstance(stages[2][1], UnverifiedLevel8PathController)
    assert stages[2][1].report()["route_eligible"] is False
    assert stages[0][2] >= 10_000
    assert stages[1][2] >= 10_000
    # rr-6o7.2: level8_magic_key_stairs is now the live clear -> 0x68 slide ->
    # cellar 0x0F key -> two-ladder return controller (spine-green from the
    # power-on 0x1F frontier).
    assert names[3] == "level8_magic_key_stairs"
    assert isinstance(stages[3][1], Level8MagicKeyStairsController)
    assert not isinstance(stages[3][1], UnverifiedLevel8PathController)
    assert stages[3][1].report()["route_eligible"] is False


def test_wrong_level_and_unknown_room_fail_closed() -> None:
    man = make_north_manhandla_controller()
    act = _step(man, _ram(level=0, screen=0x6D))
    assert man.failed and not man.success
    assert list(act.action) == list(nes_idle_action())

    man = make_north_manhandla_controller()
    act = _step(man, _ram(screen=0x6D))
    assert man.failed and not man.success
    assert "l8_north_unknown_room_0x6d" in man.notes

    dark = make_darknut_key_controller()
    act = _step(dark, _ram(screen=ROOM_ENTRY))
    assert dark.failed and not dark.success
    assert "l8_north_unknown_room_0x7e" in dark.notes


def test_0x7e_emits_north_door_action() -> None:
    ctl = make_north_manhandla_controller()
    act = _step(ctl, _ram(screen=ROOM_ENTRY, x=120, y=205))
    assert not ctl.failed
    _busy(act)
    assert "free_north_0x7e" in act.reason


def test_0x7e_door_overshoot_keeps_pushing_north() -> None:
    """Past the door plane (y=87) must hold UP, not walk back to y=93."""
    ctl = make_north_manhandla_controller()
    act = _step(ctl, _ram(screen=ROOM_ENTRY, x=120, y=87))
    assert not ctl.failed
    _busy(act)
    assert act.reason == "free_north_0x7e_push"
    assert list(act.action) == list(nes_action("UP"))


def test_0x6e_with_manhandla_emits_combat() -> None:
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=205)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 64, 160, 120)
    ctl = make_north_manhandla_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "combat" in act.reason or "leave_wall" in act.reason


def test_0x6e_cleared_emits_bomb_stand_walk() -> None:
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=141, bombs=8)
    ctl = make_north_manhandla_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert act.reason.startswith("stand_") or act.reason.startswith("face_")


def test_manhandla_arriving_0x5e_succeeds() -> None:
    ctl = make_north_manhandla_controller()
    act = _step(ctl, _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189))
    assert ctl.success and not ctl.failed
    assert list(act.action) == list(nes_idle_action())
    assert act.reason == "arrived_0x5e"


def test_0x5e_with_0x0c_emits_combat() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "combat" in act.reason or "leave_wall" in act.reason


def test_0x5e_center_key_walks_to_stand() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, room_item=0x19)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "center_key" in act.reason


def test_0x5e_does_not_walk_back_after_key_count_rises() -> None:
    ctl = make_darknut_key_controller()
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=141, keys=9, room_item=0x19)
    _step(ctl, ram)
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=141, keys=10, room_item=0x19)
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "shutter_north_0x5e" in act.reason
    assert "center_key" not in act.reason


def test_0x4e_skips_mixed_census_and_takes_north_key() -> None:
    ram = _ram(screen=ROOM_SHUTTER, x=120, y=205, keys=10)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    _put_obj(ram, 2, 0x0B, 64, 160, 141)
    _put_obj(ram, 3, 0x30, 112, 120, 120)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "key_north_0x4e" in act.reason
    assert "combat" not in act.reason


def test_0x4e_without_keys_fails_closed() -> None:
    ctl = make_darknut_key_controller()
    act = _step(ctl, _ram(screen=ROOM_SHUTTER, keys=0))
    assert ctl.failed and not ctl.success
    assert "no_keys_0x4e" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_0x3e_cleared_emits_bomb_approach() -> None:
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=205, bombs=7)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert act.reason in {
        "approach_y",
        "approach_x",
        "approach_hold",
        "stand_y",
        "stand_x",
        "south_band",
        "to_bomb_stand",
    } or act.reason.startswith("stand_") or act.reason.startswith("face_")


def test_0x2e_map_skip_peels_off_centre_column() -> None:
    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=189, room_item=0x17)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert "map_skip" in act.reason


def test_0x1e_is_dest_even_with_live_0x33_body() -> None:
    ram = _ram(screen=ROOM_GOHMA, x=120, y=205)
    _put_obj(ram, 1, 0x33, 96, 119, 112)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert ctl.success and not ctl.failed
    assert list(act.action) == list(nes_idle_action())
    assert act.reason == "arrived_0x1e"
    gohma = make_blue_gohma_controller()
    gohma.step(read_snapshot(ram))
    assert gohma.failed and not gohma.success


def test_unit_walk_records_semantic_reasons() -> None:
    """Controller walk 0x7E → 0x1E with recorded reasons. Not a fixture replay."""
    reasons: list[tuple[str, str]] = []

    man = make_north_manhandla_controller()
    ram = _ram(screen=ROOM_ENTRY, x=120, y=205)
    reasons.append(("0x7e", _step(man, ram).reason))
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=205)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 64, 160, 120)
    reasons.append(("0x6e_live", _step(man, ram).reason))
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=141, bombs=8)
    reasons.append(("0x6e_clear", _step(man, ram).reason))
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189)
    reasons.append(("0x5e_arrive", _step(man, ram).reason))
    assert man.success

    dark = make_darknut_key_controller()
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    reasons.append(("0x5e_live", _step(dark, ram).reason))
    ram = _ram(screen=ROOM_SHUTTER, x=120, y=205, keys=10)
    _put_obj(ram, 1, TYPE_0C, 128, 90, 141)
    reasons.append(("0x4e", _step(dark, ram).reason))
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=205, bombs=7)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    reasons.append(("0x3e_live", _step(dark, ram).reason))
    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=189, room_item=0x17)
    _put_obj(ram, 1, MANHANDLA_OBJECT_TYPE, 64, 160, 120)
    reasons.append(("0x2e_live", _step(dark, ram).reason))
    ram = _ram(screen=ROOM_MAP_MANHANDLA, x=120, y=189, room_item=0x17)
    reasons.append(("0x2e_skip", _step(dark, ram).reason))
    ram = _ram(screen=ROOM_GOHMA, x=120, y=205)
    _put_obj(ram, 1, 0x33, 96, 119, 112)
    reasons.append(("0x1e", _step(dark, ram).reason))
    assert dark.success

    by_room = dict(reasons)
    assert "free_north_0x7e" in by_room["0x7e"]
    assert "combat" in by_room["0x6e_live"] or "leave_wall" in by_room["0x6e_live"]
    assert by_room["0x6e_clear"].startswith("stand_") or by_room["0x6e_clear"].startswith(
        "face_"
    )
    assert by_room["0x5e_arrive"] == "arrived_0x5e"
    assert "combat" in by_room["0x5e_live"] or "leave_wall" in by_room["0x5e_live"]
    assert "key_north_0x4e" in by_room["0x4e"]
    assert "combat" in by_room["0x3e_live"] or "leave_wall" in by_room["0x3e_live"]
    assert "combat" in by_room["0x2e_live"] or "leave_wall" in by_room["0x2e_live"]
    assert "map_skip" in by_room["0x2e_skip"]
    assert by_room["0x1e"] == "arrived_0x1e"


def test_candle_leftover_pause_selects_bombs_before_place() -> None:
    """L8 leftover after the bush burn is candle=4. Isolated pins hid that."""
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=105, bombs=8, selected=4)
    ctl = make_north_manhandla_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    reasons: list[str] = []
    for _ in range(20):
        act = _step(ctl, ram)
        reasons.append(act.reason)
        if act.reason in {"place_bomb", "pause_open"}:
            break
    assert "pause_open" in reasons
    assert "place_bomb" not in reasons
    assert ctl._wall is not None
    assert ctl._wall.select_item == 1
