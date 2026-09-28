"""Trap charge lanes and orange magic escapes around dungeon pillars."""

import numpy as np
import pytest

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.shot_guard import (
    Forecast, LinkModel, ShotGuard, Trap, read_forecast, simulate,
)
from zelda_i.ram import read_snapshot


def room_ram(*, x=112, y=93):
    ram = np.zeros(0x10000, dtype=np.uint8)
    ram[0x12], ram[0x10], ram[0xEB] = 5, 9, 0x41
    ram[0x70], ram[0x84], ram[0x98] = x, y, 8
    return ram


def put_object(ram, slot, type_id, x, y, facing=1, state=0):
    ram[0x34F + slot] = type_id
    ram[0x70 + slot], ram[0x84 + slot] = x, y
    ram[0x98 + slot], ram[0xAC + slot], ram[0x485 + slot] = facing, state, 0x40


def test_invulnerable_traps_are_relevant_and_forecast_with_bodies_disabled():
    ram = room_ram()
    put_object(ram, 1, 0x49, 32, 93)
    assert ShotGuard.relevant(ram)
    forecast = read_forecast(ram, 64, bodies=False, blue_bodies=False)
    assert not forecast.empty
    assert len(forecast.traps) == 1
    assert simulate(forecast, LinkModel(None), (112, 93), "UP", [None] * 64).first_hit is not None
    assert simulate(forecast, LinkModel(None), (112, 93), "DOWN", ["DOWN"] * 20 + [None] * 44).first_hit is None


@pytest.mark.parametrize("dy, armed", [(13, True), (14, False), (-13, True), (-14, False)])
def test_trap_sense_band_is_strict(dy, armed):
    trap = Trap(1, 32, 93)
    trap.step(128, 93 + dy)
    assert bool(trap.state) is armed
    assert (trap.x, trap.y) == (32, 93)  # Arming does not move yet.


def test_trap_respects_slot_direction_and_row_priority():
    trap = Trap(1, 32, 93)
    trap.step(24, 100)  # Close column too, but disallowed west wins row check.
    assert trap.state == 0
    trap.step(32, 128)
    assert (trap.state, trap.dir, trap.limit) == (1, 4, 93)


def test_trap_charge_reverses_near_center_and_returns_to_rest():
    trap = Trap(1, 32, 93)
    trap.step(128, 93)
    for _ in range(100):
        trap.step(128, 141)
        if trap.state == 2:
            break
    assert (trap.x, trap.dir, trap.q) == (117, 2, 0x20)
    for _ in range(200):
        trap.step(128, 141)
        if trap.state == 0:
            break
    assert (trap.x, trap.y, trap.state) == (32, 93, 0)


def test_plan_simulation_never_mutates_the_sensed_trap():
    ram = room_ram()
    put_object(ram, 1, 0x49, 32, 93)
    forecast = read_forecast(ram, 64, bodies=False)
    first = simulate(forecast, LinkModel(None), (128, 93), "UP", [None] * 64)
    assert forecast.traps[0].state == 0
    assert simulate(forecast, LinkModel(None), (128, 93), "UP", [None] * 64) == first


def test_orange_guard_rejects_a_swing_before_escape_around_pillar_closes():
    # L9 0x20 f72: a shot due in 9f reaches the stand in 28f; the next
    # 13f swing leaves no time for west -> north around the pillar.
    ram = room_ram(x=96, y=149)
    put_object(ram, 1, 0x23, 83, 173)
    put_object(ram, 2, 0x23, 128, 143, facing=4)
    put_object(ram, 4, 0x23, 157, 173, facing=2)
    put_object(ram, 5, 0x24, 32, 157, state=0xB9)
    nodes = frozenset((x, y) for x in range(32, 209, 8) for y in range(93, 190, 8)
                      if not (88 <= x <= 120 and 133 <= y <= 141)
                      and not (104 <= x <= 120 and 117 <= y <= 157))
    guard = ShotGuard()
    guard._nodes_room = (9, 0x41)
    guard._nodes = nodes
    inner = FrameAction(nes_action("UP", a=True), "approach_lattice_slash")
    dodge = guard.filter(read_snapshot(ram), ram, inner)
    assert dodge.reason.startswith("guard_")
    assert dodge.action[8] == 0 and dodge.action != inner.action
    assert guard._escape  # A safe escape includes its bend, not just the first press.


def test_trap_guard_dodges_before_corner_charge_reaches_far_stand():
    ram = room_ram()
    put_object(ram, 1, 0x49, 96, 93, state=1)
    ram[0x381], ram[0x3BD] = 32, 0x70
    guard = ShotGuard()
    inner = FrameAction(nes_idle_action(), "stand_in_trap_lane")
    dodge = guard.filter(read_snapshot(ram), ram, inner)
    assert dodge.reason.startswith("guard_")
    assert dodge.action != inner.action
    assert ram[0xAD] == 1 and (ram[0x71], ram[0x85]) == (96, 93)


def test_trap_does_not_replace_menu_buttons_while_world_is_paused():
    ram = room_ram()
    put_object(ram, 1, 0x49, 104, 93, state=1)
    ram[0x381], ram[0x3BD], ram[0xE1] = 32, 0x70, 1
    inner = FrameAction(nes_action("LEFT"), "select_bombs")
    assert ShotGuard().filter(read_snapshot(ram), ram, inner) is inner
