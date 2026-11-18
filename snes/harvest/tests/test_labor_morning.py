"""The child-naming morning (kid age 60): a person in the doorway that only the wife clears.

e2e power-on run ``v1n`` ended at Y2 Winter D24 with 8 sleep rescues: the house held the wife in
bed (281,99) and three villagers, one at (128,200) in the middle of the doorway, and the house
clock does not tick, so nothing moved her. No ROM here; the ROM check is ``test_labor_morning_rom``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import pytest

from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from harvest.core.ram_catalog import field_spec, live_wram_base
from harvest.planner.tasks import labor
from harvest.tasks.nav import make_action
from retro_harness import TaskStatus

from day_plan_test_helpers import set_player_pos


def _set(ram: np.ndarray, key: str, value: int) -> None:
    spec = field_spec(key)
    base = spec.address + (live_wram_base(ram) if spec.live_offset else 0)
    ram[base] = value & 0xFF


def _put(ram: np.ndarray, slot: int, sprite: int, px: int, py: int) -> None:
    base = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    for offset, value in ((0, GOBJ_INITIALIZED), (2, sprite), (8, px), (10, py)):
        ram[base + offset] = value & 0xFF
        ram[base + offset + 1] = value >> 8


def _house(tilemap: int = 0x17, *, doorway: bool = True, wife: bool = True, crowd: bool = True) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    _set(ram, "tilemap", tilemap)
    _set(ram, "input_lock", 1)
    set_player_pos(ram, 136, 120)
    if wife:
        _put(ram, 1, 0x02E9, 281, 99)
    if crowd:
        _put(ram, 2, 0x028D, 296, 120)
    if doorway:
        _put(ram, 3, 0x02A7, 128, 200)
    return ram


def test_the_morning_is_due_with_a_person_in_the_doorway_and_the_wife_in_bed():
    assert labor.labor_morning_due(_house())
    assert labor.people_in_doorway(_house())
    assert labor.wife_in_bed(_house()) == (281, 99)


@pytest.mark.parametrize(
    "ram",
    [
        _house(doorway=False, crowd=False),  # an ordinary morning: the wife stands at (328,120)
        _house(wife=False),
        _house(tilemap=0x00),  # on the farm
    ],
)
def test_an_ordinary_morning_is_not_the_labor_morning(ram):
    assert not labor.labor_morning_due(ram)


def test_an_ordinary_morning_is_no_work():
    ram = _house(doorway=False, crowd=False)
    task = labor.LaborMorningTask()
    task.reset(type("W", (), {"ram": ram})())
    result = task.step(type("W", (), {"ram": ram})())
    assert result.status == TaskStatus.SUCCESS and result.reason.startswith("no_work")


def test_a_text_box_is_advanced_with_a_pulse_never_a_hold():
    ram = _house()
    _set(ram, "input_lock", 2)
    task = labor.LaborMorningTask()
    world = type("W", (), {"ram": ram})()
    task.reset(world)
    a_index = int(np.flatnonzero(make_action(a=True))[0])
    pressed = [bool(task.step(world).action.action[a_index]) for _ in range(60)]
    run = longest = 0
    for flag in pressed:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    assert any(pressed) and not all(pressed) and longest <= labor.TALK_PULSE[0]


def test_exit_to_farm_runs_the_labor_task_first_on_the_crowded_morning():
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask

    ram = _house()
    world = type("W", (), {"ram": ram})()
    task = ExitToFarmTask()
    task.reset(world)
    result = task.step(world)
    assert task._labor is not None and result.status == TaskStatus.RUNNING


def test_exit_to_farm_is_unchanged_on_an_ordinary_morning():
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask

    ram = _house(doorway=False, crowd=False)
    world = type("W", (), {"ram": ram})()
    task = ExitToFarmTask()
    task.reset(world)
    task.step(world)
    assert task._labor is None


def test_the_baby_the_east_villager_and_the_crowd_are_read_from_the_objects():
    ram = _house()
    assert not labor.baby_born(ram) and labor.blocker_present(ram) is False
    _put(ram, 4, 0x0301, 264, 96)
    _put(ram, 5, 0x028C, 312, 104)
    assert labor.baby_born(ram) and labor.blocker_present(ram)
    assert labor.bed_crowded(ram)
    ram2 = _house(doorway=False, crowd=False)
    _put(ram2, 4, 0x02E2, 328, 120)  # the wife, up after the naming, is not the crowd
    assert not labor.bed_crowded(ram2)


def test_a_baby_in_the_crib_starts_at_the_east_villager_or_the_wife():
    ram = _house()
    _put(ram, 4, 0x0301, 264, 96)
    task = labor.LaborMorningTask()
    task.reset(type("W", (), {"ram": ram})())
    assert task._stage == "walk_wife"  # no villager at (312,104) yet
    _put(ram, 5, 0x028C, 312, 104)
    task.reset(type("W", (), {"ram": ram})())
    assert task._stage == "walk_b"
