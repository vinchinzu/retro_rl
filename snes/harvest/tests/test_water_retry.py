"""Year 3 watering: a failed shed pickup of the can is tried again before the night is lost.

Across the r41..r47 batch and its previews 17 of 592 Year-3 Summer ``ENSURE_WATERING_CAN`` calls failed ("tool 0x15 missing
after shelf", "enter failed: directional transition timeout pos=(426,4xx)", a wedged route), 16 of them right after the evening
hay cut. Only the wedged walks are worth a second try (3 of 42 retries worked, all of them walks); the other two repeat exactly. The phase is optional and ``CROP_WATER`` then fails the same way, so every plant missed a watered night (a plant only
grows on a watered night: one lost night is about 25 crops of the Summer's 380). Each test goes red without the retry.
"""
from __future__ import annotations

import pytest

from harvest.planner.day_phase_catalog import CROP_WATER_PHASE, ENSURE_WATERING_CAN_PHASE, EXIT_TO_FARM_PHASE
from harvest.planner.day_phase_stamina import MAX_WATER_RETRIES, WATER_RETRY_ENV, splice_water_retry
from harvest.planner.day_plan_orchestrator import PhaseSchedule
from tests.test_run_strategy import set_field, world

SHELF = "route failed: nav wedged 600f route_shed_0x15 wp=3"


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv(WATER_RETRY_ENV, "1")


def _ram(year=2):
    w = world(season=1, day=10)
    set_field(w.ram, "year", year)
    return w.ram


def _schedule():
    return PhaseSchedule.from_sequence([EXIT_TO_FARM_PHASE, ENSURE_WATERING_CAN_PHASE, CROP_WATER_PHASE], [])


def test_a_failed_pickup_leaves_the_building_and_asks_for_the_can_again():
    schedule, done = _schedule(), set()
    assert splice_water_retry(schedule, 1, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(), done)
    assert [p.phase for p in schedule.active] == ["EXIT_TO_FARM", "EXIT_TO_FARM", "ENSURE_WATERING_CAN", "CROP_WATER"]


def test_a_failed_crop_water_gets_the_pickup_back_in_front_of_it():
    schedule, done = _schedule(), set()
    assert splice_water_retry(schedule, 2, CROP_WATER_PHASE, SHELF, _ram(), done)
    assert [p.phase for p in schedule.active] == ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "EXIT_TO_FARM", "ENSURE_WATERING_CAN", "CROP_WATER"]


def test_two_retries_a_day_then_the_day_fails_as_before():
    schedule, done = _schedule(), set()
    assert MAX_WATER_RETRIES == 2
    assert splice_water_retry(schedule, 1, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(), done)
    assert splice_water_retry(schedule, 2, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(), done)
    assert not splice_water_retry(schedule, 3, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(), done)


@pytest.mark.parametrize("reason", [SHELF, "route failed: nav wedged 600f route_shed", "exit_to_farm failed: nav no progress 2400f return_grape_downhill",
                                    "exit_to_farm failed: pixel_stuck pos=(27"])
def test_the_walking_failures_that_worked_on_a_second_try_are_retried(reason):
    assert splice_water_retry(_schedule(), 1, ENSURE_WATERING_CAN_PHASE, reason, _ram(), set())


@pytest.mark.parametrize("reason", ["stamina_low watered=61 unfunded=16", "empty_can", "map_mismatch:have=0x58:need=0x00",
                                    "enter failed: directional transition timeout pos=(424,486) tile=(26, 30) tilemap=0x00 held=0x00 lock=1 near=[]",
                                    "inside failed: tool 0x15 missing after shelf (selected=0x01 backpack=0x0E held=0x00"])
def test_other_failures_keep_their_own_handling(reason):
    assert not splice_water_retry(_schedule(), 2, CROP_WATER_PHASE, reason, _ram(), set())


def test_only_year_three_and_only_the_water_phases(monkeypatch):
    assert not splice_water_retry(_schedule(), 1, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(year=1), set())      # Year 2 keeps its timeline
    assert not splice_water_retry(_schedule(), 0, EXIT_TO_FARM_PHASE, SHELF, _ram(), set())
    monkeypatch.setenv(WATER_RETRY_ENV, "0")
    assert not splice_water_retry(_schedule(), 1, ENSURE_WATERING_CAN_PHASE, SHELF, _ram(), set())
