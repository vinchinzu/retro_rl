"""PR-999 animals lane. F3/F5 rules and the owned flock/herd controllers.

Pin windows are not invented here. A what-if encodes the night rules from
F3 and fails when those anchors move.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from cow_test_helpers import _make_barn_ram, _make_world, set_cow_daily, set_cow_slot
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
    COW_DAILY_TALKED_FLAG,
    read_cow_happiness,
)
from harvest.tasks.coop_layout import (
    egg_route,
    flock_after_storm,
    flock_drop_alarm,
    flock_growth_matches_f3,
    flock_must_stay_in,
    flock_sale_allowed,
    grow_flock,
    hen_unfed_days,
)
from harvest.core.animal_probe import stray_hen_px
from harvest.core.animal_status import CHICKEN_SLOT_BASE
from harvest.planner.day_phase_catalog import animal_purchase_phases
from harvest.tasks.coop_task import CoopChoresTask, HenDeliveryTask, hen_stand, yard_free, yard_step
from harvest.planner.livestock_econ import (
    COW_ADULT_FLAG,
    COW_BABY_FLAG,
    COW_EXISTS_FLAG,
    COW_SICK_FLAG,
    HerdCow,
    animal_shop_errand,
    apply_cow_medicine,
    born_happiness,
    bought_happiness,
    cannot_reach_200,
    controller_day,
    cow_tools_ready,
    herd_count_after_night,
    winter_unfed_days,
)
from harvest.tasks.cow_task import CowChoresTask

_CARE_FLAGS = COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG | COW_DAILY_MILKED_FLAG
_ADULT = COW_EXISTS_FLAG | COW_ADULT_FLAG
_BABY = COW_EXISTS_FLAG | COW_BABY_FLAG


class TestFlockRules(unittest.TestCase):
    def test_growth_matches_f3_hatch_clock(self):
        # Egg placed at timer 0 hatches on night 3. One incubator, so
        # nights 3, 6 and 9 each add one bird: 1 → 4.
        self.assertEqual(grow_flock(1, 2), 1)
        self.assertEqual(grow_flock(1, 3), 2)
        self.assertEqual(grow_flock(1, 9), 4)
        self.assertEqual(grow_flock(12, 9), 12)

    def test_morning_series_keeps_the_three_night_hatch_interval(self):
        series = [1] + [grow_flock(1, night) for night in range(1, 10)]
        self.assertEqual(series, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        self.assertTrue(flock_growth_matches_f3(series))
        # Sick nights before the first egg stay flat. Rises still land 3 apart.
        self.assertTrue(flock_growth_matches_f3([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]))
        self.assertFalse(flock_growth_matches_f3([1, 1, 2, 2, 2, 3, 3, 3, 4]))
        self.assertFalse(flock_growth_matches_f3([1, 1, 1, 2, 2, 2, 4]))
        self.assertFalse(flock_growth_matches_f3([1, 1, 1, 2, 2, 1, 2, 3, 4]))

    def test_full_flock_ships_and_a_drop_alarms(self):
        self.assertEqual(
            egg_route("auto", chicks_n=12, slot_total=12, incubating=False),
            "ship",
        )
        self.assertEqual(
            egg_route("incubate", chicks_n=12, slot_total=12, incubating=False),
            "ship",
        )
        self.assertEqual(
            egg_route("auto", chicks_n=3, slot_total=3, incubating=False),
            "incubate",
        )
        self.assertIsNone(flock_drop_alarm(12, 12))
        task = CoopChoresTask()
        task._flock_before = 12
        alarm = task.note_flock_count(11)
        self.assertEqual(alarm, "flock drop 12->11")
        self.assertEqual(task.flock_alarms, ["flock drop 12->11"])

    def test_storm_indoors_keeps_12_and_one_outdoor_bird_is_stripped(self):
        # CODE_83C0E5 writes $01 for one map<4 adult per night (bit $0020).
        # The next night CODE_83C05E deletes it. Coop map $28 is not < 4.
        self.assertTrue(flock_must_stay_in(0x0010))
        self.assertFalse(flock_must_stay_in(0))
        self.assertEqual(flock_after_storm(indoor=True, weather_flags=0x0010), 12)
        self.assertEqual(flock_after_storm(indoor=False, weather_flags=0x0010), 11)
        self.assertFalse(flock_sale_allowed(12))
        self.assertTrue(flock_sale_allowed(13))

    def test_seven_days_of_hay_for_12_hens(self):
        self.assertEqual(hen_unfed_days(84, 12, 7), 0)
        self.assertEqual(hen_unfed_days(83, 12, 7), 1)


class TestHerdRules(unittest.TestCase):
    def test_f3_happiness_anchors(self):
        self.assertEqual(bought_happiness(22, feedable=True, after_brush=False), 84)
        self.assertEqual(bought_happiness(50, feedable=True, after_brush=False), 196)
        self.assertEqual(bought_happiness(50, feedable=True, after_brush=True), 200)
        self.assertEqual(bought_happiness(22, feedable=False, after_brush=True), 88)
        peak_unfed = max(
            bought_happiness(day, feedable=False, after_brush=True) for day in range(1, 80)
        )
        self.assertEqual(peak_unfed, 88)
        self.assertEqual(born_happiness(25, 100, feedable=False, after_brush=True), 200)
        self.assertLess(born_happiness(25, 20, feedable=False, after_brush=True), 200)
        peak_born20 = max(
            born_happiness(day, 20, feedable=False, after_brush=True) for day in range(1, 80)
        )
        self.assertLess(peak_born20, 200)
        self.assertGreaterEqual(
            born_happiness(45, 50, feedable=True, after_brush=True),
            200,
        )

    def test_controller_closes_a_20_point_gap(self):
        cows = [HerdCow(1, 200, status=_ADULT), HerdCow(2, 180, status=_ADULT)]
        alarms: list[str] = []
        for _ in range(5):
            cows, day_alarms, deltas = controller_day(cows)
            alarms.extend(day_alarms)
            self.assertEqual(sorted(row["slot"] for row in deltas), [1, 2])
        self.assertEqual([cow.happiness for cow in cows], [200, 200])
        self.assertEqual(alarms, [])

    def test_bought_slot_0_has_a_native_stall_and_reaches_200(self):
        bought = HerdCow(0, 80, origin="bought", status=_ADULT)
        self.assertFalse(cannot_reach_200(bought, 60))
        rows = [bought]
        for _ in range(30):
            rows, alarms, _ = controller_day(rows)
            self.assertEqual(alarms, [])
        self.assertEqual(rows[0].happiness, 200)
        for birth in (20, 50, 100):
            calf = HerdCow(0, birth, origin="born", status=_BABY)
            self.assertFalse(cannot_reach_200(calf, 60))

    def test_twelve_slots_targeted_once_by_happiness(self):
        happiness = [40, 10, 200, 5, 80, 15, 90, 30, 70, 50, 60, 20]
        ram = _make_barn_ram(cows=12, hay=50, tool_selected=0x0F, tool_backpack=0x0E)
        for slot, happy in enumerate(happiness):
            set_cow_slot(ram, slot, (6 + (slot % 4) * 2, 8 + (slot // 4) * 3), status=_ADULT)
            set_cow_daily(ram, slot, flags=0, happiness=happy)
        task = CowChoresTask(talk=True, brush=True, milk=True, feed=False)
        task.reset(_make_world(ram))
        order: list[int] = []
        deltas: list[dict[str, int]] = []
        for _ in range(12):
            slots = task._care_needed_cow_slots(ram)
            self.assertTrue(slots)
            slot = slots[0]
            before = read_cow_happiness(ram, slot)
            after = min(255, before + 4)
            set_cow_daily(ram, slot, flags=_CARE_FLAGS, happiness=after)
            deltas.append(
                {"slot": slot, "before": before, "after": after, "delta": after - before}
            )
            order.append(slot)
        self.assertEqual(order, sorted(range(12), key=lambda slot: (happiness[slot], slot)))
        self.assertEqual(sorted(row["slot"] for row in deltas), list(range(12)))
        self.assertEqual(len({row["slot"] for row in deltas}), 12)

    def test_equal_happiness_keeps_slot_order(self):
        ram = _make_barn_ram(cows=2, hay=10, tool_selected=0x0F, tool_backpack=0x0E)
        set_cow_slot(ram, 0, (6, 8), status=_ADULT)
        set_cow_slot(ram, 1, (8, 8), status=_ADULT)
        set_cow_daily(ram, 0, flags=0, happiness=40)
        set_cow_daily(ram, 1, flags=0, happiness=40)
        task = CowChoresTask(talk=True, brush=True, milk=True, feed=False)
        task.reset(_make_world(ram))
        self.assertEqual(task._care_slots, [0, 1])

    def test_care_delta_records_the_byte(self):
        ram = _make_barn_ram(cows=1, hay=10, tool_selected=0x0F, tool_backpack=0x0E)
        set_cow_daily(ram, 0, flags=0, happiness=40)
        task = CowChoresTask(feed=False)
        task.reset(_make_world(ram))
        task._note_care_delta(ram, "milk", 40)
        set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=41)
        task._note_care_delta(ram, "talk", 40)
        self.assertEqual(task.happiness_deltas[0]["delta"], 0)
        self.assertEqual(task.happiness_deltas[1]["delta"], 1)
        self.assertEqual(task.happiness_deltas[1]["slot"], 0)

    def test_cow_tools_ready_carried_or_on_the_shelf(self):
        self.assertTrue(cow_tools_ready(0, 0x0F, 0x0E))
        self.assertTrue(cow_tools_ready(0x60, 0, 0))
        self.assertFalse(cow_tools_ready(0x40, 0, 0))
        self.assertFalse(cow_tools_ready(0, 0, 0))

    def test_medicine_clears_sick_and_a_storm_keeps_12(self):
        status, timer = apply_cow_medicine(_ADULT | COW_SICK_FLAG, 7)
        self.assertEqual(status & COW_SICK_FLAG, 0)
        self.assertEqual(timer, 0)
        self.assertEqual(
            herd_count_after_night(
                cows=12,
                in_barn=True,
                sick_timer=timer,
                weather_flags=0x0010,
                sick=False,
            ),
            12,
        )
        self.assertEqual(
            herd_count_after_night(
                cows=12,
                in_barn=True,
                sick_timer=1,
                weather_flags=0,
                sick=True,
            ),
            11,
        )
        self.assertEqual(
            herd_count_after_night(
                cows=12,
                in_barn=True,
                sick_timer=1,
                weather_flags=0x0010,
                sick=True,
            ),
            12,
        )

    def test_winter_hay_consumption_is_one_per_cow(self):
        # Stock already in the barn. Cutting the grass is not this function.
        self.assertEqual(winter_unfed_days(360, 12, 30), 0)
        self.assertEqual(winter_unfed_days(359, 12, 30), 1)


class TestHenDelivery(unittest.TestCase):
    """The shop spawns the hen in a fenced yard behind it. Only the back door reaches it."""

    @staticmethod
    def _ram_with_hen(map_byte: int, pixel=(568, 792)):
        ram = np.zeros(0x20000, dtype=np.uint8)
        off = CHICKEN_SLOT_BASE
        ram[off] = 0x09
        ram[off + 1] = map_byte
        ram[off + 4], ram[off + 5] = pixel[0] & 0xFF, pixel[0] >> 8
        ram[off + 6], ram[off + 7] = pixel[1] & 0xFF, pixel[1] >> 8
        return ram

    def test_stray_hen_is_an_adult_whose_map_byte_is_not_the_coop(self):
        self.assertEqual(stray_hen_px(self._ram_with_hen(0x05)), (568, 792))
        self.assertIsNone(stray_hen_px(self._ram_with_hen(0x28)))
        self.assertIsNone(stray_hen_px(np.zeros(0x20000, dtype=np.uint8)))

    def test_purchase_trip_ends_with_delivery_not_a_bare_shop_exit(self):
        names = [p.phase for p in animal_purchase_phases("chicken")]
        self.assertEqual(names, ["NAV_TO_ANIMAL_SHOP", "BUY_COW_VENDOR", "HEN_DELIVERY", "COOP_CHORES", "EXIT_COOP"])
        self.assertEqual([p.phase for p in animal_purchase_phases("hen_pickup")][:2],
                         ["NAV_TO_ANIMAL_SHOP", "HEN_DELIVERY"])
        self.assertTrue(all(p.failure_policy == "optional" for p in animal_purchase_phases("chicken")[1:]))
        self.assertIn("EXIT_ANIMAL_SHOP", [p.phase for p in animal_purchase_phases("cow")])

    def test_delivery_does_not_bypass_the_flock_growth_policy(self):
        for species in ("chicken", "hen_pickup"):
            coop = next(p for p in animal_purchase_phases(species) if p.phase == "COOP_CHORES")
            self.assertEqual(coop.params["egg_mode"], "ship")

    def test_an_undelivered_hen_is_picked_up_before_any_new_purchase(self):
        self.assertEqual(animal_shop_errand(self._ram_with_hen(0x05)), "hen_pickup")
        self.assertNotEqual(animal_shop_errand(self._ram_with_hen(0x28)), "hen_pickup")

    def test_yard_stands_stay_on_free_ground(self):
        # Hen on the strip above the roof: the player's side is chosen.
        self.assertEqual(hen_stand((568, 792), (584, 808)), ((584, 792), "left"))
        # Hen on the top row: the stand above would be the fence.
        stand, _ = hen_stand((568, 780), (584, 808))
        self.assertTrue(yard_free(*stand))
        # Hen pressed into the top-left corner (a measured fail): stand to her right.
        self.assertEqual(hen_stand((517, 774), (584, 808)), ((533, 774), "left"))
        # Hen in the right corridor beside the roof: only above/below are free.
        self.assertIn(hen_stand((646, 830), (584, 808))[1], ("up", "down"))
        self.assertTrue(yard_free(584, 808))
        self.assertFalse(yard_free(600, 840))  # the roof

    def test_yard_walk_runs_the_lane_above_the_roof(self):
        # From the back door to a stand in the left strip: up to the lane,
        # across, then down. Never straight through the roof.
        self.assertEqual(yard_step((584, 808), (540, 840)), "up")
        self.assertEqual(yard_step((584, 792), (540, 840)), "left")
        self.assertEqual(yard_step((540, 792), (540, 840)), "down")
        self.assertIsNone(yard_step((540, 840), (540, 840)))
        self.assertIsNone(yard_step((584, 795), (584, 792)))

    def _shop_world(self, minute: int):
        from day_plan_test_helpers import make_date_world

        world = make_date_world(0x24, season=1, day=5, hour=16, minute=minute)
        off = 0x4000 + CHICKEN_SLOT_BASE
        world.ram[off], world.ram[off + 1] = 0x09, 0x05
        world.ram[off + 4], world.ram[off + 5] = 568 & 0xFF, 568 >> 8
        world.ram[off + 6], world.ram[off + 7] = 792 & 0xFF, 792 >> 8
        return world

    def test_shop_closing_time_seals_the_yard_so_a_late_visit_walks_home(self):
        late = HenDeliveryTask()
        world = self._shop_world(minute=40)
        late.reset(world)
        late.step(world)
        self.assertTrue(late.deferred)
        self.assertEqual(late._phase, "to_farm")
        early = HenDeliveryTask()
        world = self._shop_world(minute=5)
        early.reset(world)
        early.step(world)
        self.assertFalse(early.deferred)
        self.assertEqual(early._phase, "to_yard")

    def test_delivery_task_names_no_cow_counter_pose(self):
        source = Path(__file__).resolve().parents[1] / "harvest" / "tasks" / "coop_task.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("class HenDeliveryTask", text)
        self.assertNotIn("farm_to_animal_shop_counter", text)


if __name__ == "__main__":
    unittest.main()
