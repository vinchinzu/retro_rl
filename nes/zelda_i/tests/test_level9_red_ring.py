"""Red Ring predecessor, item ownership, and natural return contracts."""
from dataclasses import replace

import pytest
from retro_harness.nes import nes_action
from zelda_i.level9.stairs import RED_RING_DOORS, make_red_ring_controller
from zelda_i.ram import read_snapshot
from zelda_i.tests.ram_helpers import make_ram


def snap(**fields):
    ring = fields.pop("ring", 1)
    base = make_ram({"mode": 5, "level": 9, "screen": 0x16, "x": 120, "y": 205,
                     "health": 0xEE, "bombs": 6, "keys": 3, "triforce": 0xFF}, **fields)
    return replace(read_snapshot(base), ring=ring)


@pytest.mark.parametrize("fields", [{"screen": 0x06}, {"triforce": 0x7F}, {"bombs": 2}])
def test_detour_rejects_wrong_predecessor_or_short_bag(fields):
    ctl = make_red_ring_controller()
    ctl.step(snap(**fields))
    assert ctl.failed and not ctl.success
    assert ctl.report()["inventory_writes"] == 0


def test_owned_red_ring_skips_without_spending_bombs():
    ctl = make_red_ring_controller()
    ctl.step(snap(ring=2, bombs=0))
    assert ctl.success and not ctl.failed
    assert ctl.bomb is None


def test_ring_is_required_at_the_return_room():
    ctl = make_red_ring_controller()
    ctl.door_i = len(RED_RING_DOORS)
    ctl.ring_acquired = True
    assert not ctl.arrived(snap(ring=1))
    assert not ctl.arrived(snap(ring=2, screen=0x07))
    assert not ctl.arrived(snap(ring=2, mode=7))
    assert ctl.arrived(snap(ring=2))


def test_empty_cellar_exit_does_not_claim_ring():
    ctl = make_red_ring_controller()
    ctl.start_checked = True
    ctl.phase = "cellar"
    ctl.door_i = 4
    ctl.step(snap(screen=0x07))
    assert ctl.failed and not ctl.success
    assert "red_ring_cellar_return_without_ring" in ctl.notes


def test_cellar_chamber_uses_floor_and_shaft_before_item():
    ctl = make_red_ring_controller()
    assert ctl._cellar_step(snap(mode=9, screen=0, x=48, y=93)).action == nes_action("DOWN")
    assert ctl._cellar_step(snap(mode=9, screen=0, x=48, y=189)).action == nes_action("RIGHT")
    assert ctl._cellar_step(snap(mode=9, screen=0, x=176, y=181)).action == nes_action("UP")
    assert ctl._cellar_step(snap(mode=9, screen=0, x=176, y=141)).action == nes_action("LEFT")
    assert not ctl.ring_acquired
    ctl._cellar_step(snap(mode=9, screen=0, x=128, y=141, ring=2))
    assert ctl.ring_acquired


def test_clear_census_excludes_invulnerable_movers():
    ctl = make_red_ring_controller()
    assert ctl.fight07.spec.expected_enemy_count == 5
    assert ctl.fight17.spec.expected_enemy_count == 7
    assert 0x2B not in ctl.fight07.spec.enemy_types
    assert 0x2B not in ctl.fight17.spec.enemy_types


def test_probe_labels_keep_original_pins_valid():
    from zelda_i.scratch.l9_probe import steps

    assert "s14r_level9_red_ring" not in [name for name, _, _ in steps()]
    names = [name for name, _, _ in steps(red_ring=True)]
    assert "s14_level9_east_15" in names
    assert names.index("s14b_level9_patra_16") < names.index("s14r_level9_red_ring")
    assert "s14r_level9_red_ring" in names
    assert "s15_level9_north_16" in names
    assert "s24_level9_room10_silver_arrows" in names
    assert "s25_level9_natural_patra_join" in names
