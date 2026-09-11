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
    ROOM_3E_STATUE_BLOCKS,
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
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
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


def _put_obj(
    ram: np.ndarray,
    slot: int,
    type_id: int,
    hp: int,
    x: int,
    y: int,
    facing: int = 0,
) -> None:
    ram[ADDR_OBJ_TYPE + slot] = type_id
    ram[ADDR_OBJ_HP + slot] = hp
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y
    ram[ADDR_LINK_FACING + slot] = facing


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


def test_0x7e_off_column_leftover_does_not_up_into_wall() -> None:
    """Knockback at (208,157) must LEFT to the door column, never UP."""
    ctl = make_north_manhandla_controller()
    act = _step(ctl, _ram(screen=ROOM_ENTRY, x=208, y=157))
    assert not ctl.failed
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert act.reason == "free_north_0x7e_x"


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
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    _busy(act)
    assert act.reason in {
        "combat_approach",
        "combat_flank",
        "combat_slash",
        "combat_backstep",
        "inland_leave",
        "column_peel",
    } or act.reason.startswith("combat_")


def test_0x5e_shield_front_flanks_not_slash() -> None:
    """South leftover into a south-facing 0x0C: circle, never UP into the shield."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason != "combat_slash"
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("UP", "A"))
    assert act.reason in {
        "column_peel",
        "combat_flank",
        "combat_approach",
        "combat_backstep",
    }


def test_0x5e_rear_in_reach_slashes() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=157, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 141, facing=0x08)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "combat_slash"
    assert "A" in str(act.action) or list(act.action) == list(nes_action("UP", "A"))


def test_0x5e_contact_backsteps() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=149, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 141, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason in {"combat_backstep", "combat_flank"}
    if act.reason == "combat_backstep":
        assert list(act.action) == list(nes_action("DOWN"))
    else:
        assert list(act.action) in (list(nes_action("LEFT")), list(nes_action("RIGHT")))


def test_0x5e_contact_up_does_not_cross_waist() -> None:
    """Link (104,150) vs 0x0C (104,160): backstep UP would hunt to (104,117). Slash rear or stand."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=104, y=150, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 104, 160, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("UP", "A"))
    assert act.reason in {"occupancy_stand", "combat_slash"}
    if act.reason == "occupancy_stand":
        assert list(act.action) == list(nes_idle_action())
    else:
        assert list(act.action) == list(nes_action("DOWN", "A"))


def test_0x5e_occupancy_miss_blocks_and_replans() -> None:
    """Same leftover pose twice: first dir misses, that cell is blocked, next dir differs."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141, facing=0x01)
    ctl = make_darknut_key_controller()
    first = _step(ctl, ram)
    _busy(first)
    second = _step(ctl, ram)
    assert not ctl.failed
    assert ctl._walker.misses >= 1
    if second.reason == "occupancy_stand":
        assert list(second.action) == list(nes_idle_action())
    else:
        assert list(second.action) != list(first.action)


def test_0x5e_occupancy_nopath_stands() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141, facing=0x01)
    ctl = make_darknut_key_controller()
    _step(ctl, ram)
    xy = (120, 189)
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        ctl._walker.grid.mark_blocked_ahead(*xy, direction)
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "occupancy_stand"
    assert list(act.action) == list(nes_idle_action())


def test_0x5e_south_mouth_does_not_exit_to_0x6e() -> None:
    """Leftover (120,189): contact north of Link must not DOWN through the bomb hole."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 177, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("DOWN", "A"))
    assert act.reason in {
        "column_peel",
        "combat_backstep",
        "combat_flank",
        "occupancy_stand",
    }


def test_0x5e_south_band_contact_holds_column_peel() -> None:
    """Contact on y=189 must not 1px-backstep; hold LEFT until |x-120|>=16."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 177, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert act.reason != "combat_backstep"
    ram2 = _ram(screen=ROOM_DARKNUT_KEY, x=116, y=189, health=0x22)
    _put_obj(ram2, 1, TYPE_0C, 128, 120, 177, facing=0x04)
    act2 = _step(ctl, ram2)
    assert act2.reason == "column_peel"
    assert list(act2.action) == list(nes_action("LEFT"))


def test_0x5e_rom_death_pose_stays_on_south_band() -> None:
    """ROM l8_npv4_5e_hold leftover (116,189) mode-17: still no DOWN."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=116, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 177, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("DOWN", "A"))


def test_0x5e_south_band_peels_column_before_inland() -> None:
    """Leftover (120,189) peels off x=120 first even if the facing axis is free."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141, facing=0x01)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x5e_death_pose_holds_peel_on_column() -> None:
    """ROM death (116,189) still |x-120|<16: hold LEFT, not inland UP."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=116, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141, facing=0x01)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x5e_on_column_peels_then_inland() -> None:
    """Entry column + south-facing 0x0C: LEFT off x=120, never UP into the shield."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x5e_peeled_column_then_inland_up() -> None:
    """After peel to x=104, south-facing 0x0C still on x=120: axis free, UP inland."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=104, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "inland_leave"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x5e_off_column_south_band_inland_not_west() -> None:
    """ROM death leftover (80,181): off-column, UP inland, never more LEFT."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=80, y=181, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "inland_leave"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x5e_off_column_contact_stands_not_west() -> None:
    """(80,181) contact with 0x0C on the UP cell: stand, not LEFT into the west wall."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=80, y=181, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 170, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    if act.reason == "occupancy_stand":
        assert list(act.action) == list(nes_idle_action())
    else:
        assert act.reason == "inland_leave"
        assert list(act.action) == list(nes_action("UP"))


def test_0x5e_inland_north_of_waist_does_not_hunt_up() -> None:
    """ROM death leftover (104,117): DOWN to waist or stand, never UP into the pack."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=104, y=117, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 141, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("UP", "A"))
    assert act.reason in {"waist_return", "occupancy_stand", "combat_slash", "combat_approach"}
    if act.reason == "waist_return":
        assert list(act.action) == list(nes_action("DOWN"))


def test_0x5e_inland_occupancy_nopath_stands() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=104, y=117, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    _step(ctl, ram)
    xy = (104, 117)
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        ctl._walker.grid.mark_blocked_ahead(*xy, direction)
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "occupancy_stand"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("UP"))


def test_0x5e_off_column_occupancy_nopath_stands() -> None:
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=80, y=181, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 163, facing=0x04)
    ctl = make_darknut_key_controller()
    _step(ctl, ram)
    xy = (80, 181)
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        ctl._walker.grid.mark_blocked_ahead(*xy, direction)
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "occupancy_stand"
    assert list(act.action) == list(nes_idle_action())
    assert list(act.action) != list(nes_action("LEFT"))


def test_0x5e_death_pose_on_column_keeps_peeling() -> None:
    """ROM death (116,189) still inside the entry column: peel LEFT, not UP into shield."""
    ram = _ram(screen=ROOM_DARKNUT_KEY, x=116, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 120, 177, facing=0x04)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_0x6e_spill_reenters_north_not_fail_closed() -> None:
    """ROM l8_npv4_5e leftover 0x6E (120,93): UP the open hole, do not fail."""
    ram = _ram(screen=ROOM_MANHANDLA, x=120, y=93, health=0x26, bombs=7)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason.startswith("reenter_north_0x6e")
    assert list(act.action) == list(nes_action("UP"))


def test_0x5e_hc3_does_not_write_health_keys_bombs_mk() -> None:
    ram = _ram(
        screen=ROOM_DARKNUT_KEY,
        x=120,
        y=189,
        health=0x22,
        keys=9,
        bombs=7,
        magic_key=0,
    )
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141, facing=0x01)
    ctl = make_darknut_key_controller()
    _step(ctl, ram)
    snap = read_snapshot(ram)
    assert snap.heart_containers == 3
    assert snap.health_is_full
    assert ram[ADDR_HEALTH] == 0x22
    assert ram[ADDR_KEYS] == 9
    assert ram[ADDR_BOMBS] == 7
    assert ram[ADDR_MAGIC_KEY] == 0


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
    assert by_room["0x5e_live"] in {
        "column_peel",
        "inland_leave",
        "combat_approach",
        "combat_flank",
    } or ("combat" in by_room["0x5e_live"] or "leave_wall" in by_room["0x5e_live"])
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


def test_0x3e_door_entry_steps_up() -> None:
    """Link at south door (120, 205) steps UP into room 0x3E."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=205, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "combat_door_enter"
    assert list(act.action) == list(nes_action("UP"))


def test_0x3e_south_band_peels_west() -> None:
    """On south band (120, 189), Link peels LEFT off center column toward west aisle."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    ctl = make_darknut_key_controller()
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "column_peel"
    assert list(act.action) == list(nes_action("LEFT"))


def test_0x3e_statues_are_blocked_in_grid() -> None:
    """0x3E statue locations (96, 141) and (144, 141) are marked impassable."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=189, health=0x22)
    _put_obj(ram, 1, TYPE_0C, 128, 80, 141)
    ctl = make_darknut_key_controller()
    _step(ctl, ram)
    assert not ctl._walker.grid.passable(96, 141)
    assert not ctl._walker.grid.passable(144, 141)
    assert ctl._walker.grid.passable(120, 141)  # center aisle is passable
    assert ctl._walker.grid.passable(64, 141)   # west aisle is passable


def test_0x3e_tactical_bomb_against_approaching_darknut() -> None:
    """Link drops a bomb facing UP when a Darknut approaches south down the west aisle."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=64, y=165, health=0x22, bombs=7)
    _put_obj(ram, 1, TYPE_0C, 128, 64, 135, facing=0x04)
    ctl = make_darknut_key_controller()
    ctl._3e_peeled = True
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "threat_bomb_up"
    assert list(act.action) == list(nes_action("UP", "B"))


def test_0x3e_flank_slash() -> None:
    """Link slashes into the flank of a south-facing Darknut."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=50, y=140, health=0x22, bombs=7)
    _put_obj(ram, 1, TYPE_0C, 128, 64, 140, facing=0x04)
    ctl = make_darknut_key_controller()
    ctl._3e_peeled = True
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "combat_slash"
    assert list(act.action) == list(nes_action("RIGHT", "A"))


def test_0x3e_advance_north_when_clear() -> None:
    """Link advances UP the west corridor toward the north wall when corridor is clear."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=64, y=165, health=0x22, bombs=7)
    # Darknut is far off on the east side
    _put_obj(ram, 1, TYPE_0C, 128, 180, 140, facing=0x01)
    ctl = make_darknut_key_controller()
    ctl._3e_peeled = True
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "advance_north"
    assert list(act.action) == list(nes_action("UP"))


def test_0x3e_bomb_north_wall_at_stand() -> None:
    """Link places a bomb facing UP when at the north bomb stand (120, 105)."""
    ram = _ram(screen=ROOM_BLUE_DARKNUTS, x=120, y=105, health=0x22, bombs=7)
    _put_obj(ram, 1, TYPE_0C, 128, 180, 140, facing=0x01)
    ctl = make_darknut_key_controller()
    ctl._3e_peeled = True
    act = _step(ctl, ram)
    assert not ctl.failed
    assert act.reason == "bomb_north_wall"
    assert list(act.action) == list(nes_action("UP", "B"))

