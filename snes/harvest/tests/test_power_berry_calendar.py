"""Which ROM power berry source a morning's plan spends its trip on (RAM only, no emulator)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from harvest.core.ram_catalog import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE, field_spec
from harvest.planner import power_berry_calendar as pbc
from harvest.planner import run_strategy

SPRING, SUMMER, FALL, WINTER = 0, 1, 2, 3
ADULT_HEN = 0x09          # alive bit + age field 4


def ram(*, year=0, season=SPRING, day=4, hour=6, berries=0, f64=0, f66=0, f6a=0, f6c=0, weather=0, hens=2):
    out = np.zeros(0x20000, dtype=np.uint8)
    for key, value in dict(year=year, season=season, day=day, hour=hour, power_berry_count=berries).items():
        spec = field_spec(key)
        out[spec.address] = spec.to_storage(value) & 0xFF
    for address, word in ((pbc.UPGRADE_FLAGS, f64), (pbc.FAMILY_66, f66), (pbc.HOUSE_6A, f6a),
                          (pbc.FESTIVAL_6C, f6c), (pbc.WEATHER, weather)):
        out[address], out[address + 1] = word & 0xFF, word >> 8
    for slot in range(hens):
        out[CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE] = ADULT_HEN
    return out


def due(**kw):
    shops_closed = kw.pop("shops_closed", False)
    return pbc.due_source(ram(**kw), shops_closed=shops_closed, force=True)


def test_receipts_expose_a_lost_spawn_and_do_not_count_the_turtle_as_a_berry():
    lost = ram(berries=6, f64=0x1E00, f66=0x6000, f6a=0x0040,
               f6c=pbc.EGG_TURTLE_BIT | pbc.EGG_HOUSE_ARM_BIT)
    maximum = field_spec("max_stamina")
    lost[maximum.address] = maximum.to_storage(160)
    for snapshot in (lost, np.pad(lost, (0x4000, 0))):
        receipt = pbc.berry_receipts(snapshot)
        assert receipt["count"] == 6
        assert receipt["max_stamina"] == 160
        assert receipt["spent_sources"] == 7
        assert receipt["spent_sources"] > receipt["count"]


class SwitchTests(unittest.TestCase):
    def test_cash_path_schedules_nothing(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pbc.ENV_VAR, None)
            os.environ.pop(run_strategy.ENV_VAR, None)
            self.assertFalse(pbc.enabled())
            self.assertIsNone(pbc.due_source(ram()))

    def test_score_family_collects_and_cash_does_not(self) -> None:
        self.assertFalse(run_strategy.CASH.collect_power_berries)
        for strategy in (run_strategy.SCORE, run_strategy.SCORE_HERD, run_strategy.COVER, run_strategy.PERFECT):
            self.assertTrue(strategy.collect_power_berries, strategy.name)

    def test_environment_override_wins_both_ways(self) -> None:
        with patch.dict(os.environ, {run_strategy.ENV_VAR: "cash", pbc.ENV_VAR: "1"}):
            self.assertTrue(pbc.enabled())
            self.assertEqual(pbc.due_source(ram()), "statue")
        with patch.dict(os.environ, {run_strategy.ENV_VAR: "score", pbc.ENV_VAR: "0"}):
            self.assertFalse(pbc.enabled())
            self.assertIsNone(pbc.due_source(ram()))
        with patch.dict(os.environ, {run_strategy.ENV_VAR: "score"}):
            os.environ.pop(pbc.ENV_VAR, None)
            self.assertTrue(pbc.enabled())


class SpringTests(unittest.TestCase):
    def test_missing_stump_is_scheduled_after_the_four_early_berries(self):
        early = pbc.STATUE_BIT | 0x1800
        self.assertEqual(due(season=SUMMER, day=23, berries=4, f64=early, f66=pbc.FISH_BIT), "stump")
        self.assertIsNone(due(season=SUMMER, day=23, berries=5, f64=early | pbc.STUMP_BIT, f66=pbc.FISH_BIT))
        self.assertEqual(due(season=FALL, day=12, berries=4, f64=early, f66=pbc.FISH_BIT), "harvest_festival")
        self.assertEqual(due(season=FALL, day=20, berries=4, f64=early, f66=pbc.FISH_BIT), "egg_festival")

    def test_statue_then_fish_from_the_first_free_day(self) -> None:
        self.assertIsNone(due(day=3))
        self.assertEqual(due(day=4), "statue")
        self.assertEqual(due(day=4, f64=pbc.STATUE_BIT), "fish")
        self.assertIsNone(due(day=5, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT))

    def test_statue_and_fish_refuse_rain_snow_and_late_starts(self) -> None:
        for weather in (0x0002, 0x0008, 0x0010):
            self.assertIsNone(due(day=4, weather=weather), weather)
        self.assertIsNone(due(day=4, hour=11))
        self.assertEqual(due(day=4, hour=10), "statue")

    def test_later_years_and_seasons_are_free_from_day_one(self) -> None:
        self.assertEqual(due(year=1, day=1), "statue")
        self.assertEqual(due(season=SUMMER, day=1), "statue")

    def test_fish_is_not_scheduled_in_winter(self) -> None:
        self.assertIsNone(due(season=WINTER, day=3, f64=pbc.STATUE_BIT))


class GoldHammerTests(unittest.TestCase):
    def test_live_emulator_ram_reads_the_shifted_gold_hammer_shelf(self):
        live = np.pad(ram(year=2), (0x4000, 0))
        live[pbc.SHED_ROW_2 + 0x4000] = pbc.GOLD_HAMMER_SHELF_BIT
        with patch.dict(os.environ, {pbc.GOLD_STATUE_ENV: "0"}):
            self.assertEqual(pbc.due_source(live, force=True), "fish")

    def test_statue_is_skipped_once_the_shelf_holds_only_the_gold_hammer(self) -> None:
        gold = ram(year=2, season=SPRING, day=2)
        gold[pbc.SHED_ROW_2] = pbc.GOLD_HAMMER_SHELF_BIT
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pbc.GOLD_STATUE_ENV, None)
            self.assertEqual(pbc.due_source(gold, force=True), "fish", "the dead statue trip hands the slot to the fish")
        with patch.dict(os.environ, {pbc.GOLD_STATUE_ENV: "1"}):
            self.assertEqual(pbc.due_source(gold, force=True), "statue", "the opt-in keeps the gold route reachable")
        self.assertEqual(due(year=2, day=2), "statue", "a shelf with the plain hammer still sends the farmer")


class WinterAndFallTests(unittest.TestCase):
    def test_cave_waits_for_the_boulder_flag_and_stops_after_the_eat(self) -> None:
        base = dict(season=WINTER, day=8, f64=pbc.STATUE_BIT | pbc.STUMP_BIT)
        self.assertIsNone(due(**base))
        self.assertEqual(due(**{**base, "f64": base["f64"] | pbc.BOULDER_OPEN_BIT}), "cave")
        self.assertIsNone(due(**{**base, "f64": base["f64"] | pbc.BOULDER_OPEN_BIT, "f6a": pbc.CAVE_BIT}))

    def test_cave_is_still_due_in_any_later_season(self) -> None:
        """A cave berry missed in Winter (Year 3 pins carry the boulder flag and no berry) is collected in Spring or Summer."""
        flags = pbc.STATUE_BIT | pbc.BOULDER_OPEN_BIT
        for season in (SPRING, SUMMER):
            self.assertEqual(due(year=2, season=season, day=2, f64=flags), "cave", season)
        self.assertEqual(due(year=2, season=SUMMER, day=2, f64=flags | 0, f6a=pbc.CAVE_BIT), "fish")

    def test_cave_does_not_need_clear_weather(self) -> None:
        self.assertEqual(due(season=WINTER, day=8, f64=pbc.STATUE_BIT | pbc.BOULDER_OPEN_BIT, weather=0x0008), "cave")

    def test_hawker_trade_needs_a_spare_hen_an_open_dry_shop_and_an_unspent_offer(self) -> None:
        base = dict(season=FALL, day=3, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT)
        self.assertEqual(due(**base), "chicken_trade")
        self.assertIsNone(due(**base, hens=1), "the route keeps its two laying hens")
        self.assertIsNone(due(**{**base, "f66": base["f66"] | pbc.HAWKER_BIT}))
        self.assertIsNone(due(**base, shops_closed=True))
        self.assertIsNone(due(**{**base, "weather": 0x0002}))
        self.assertIsNone(due(**{**base, "hour": 10}))

    def test_harvest_festival_day_offers_once(self) -> None:
        base = dict(season=FALL, day=12, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT | pbc.HAWKER_BIT)
        with patch.dict(os.environ, {pbc.HARVEST_FESTIVAL_ENV: "1"}):
            self.assertEqual(due(**base), "harvest_festival")
            self.assertIsNone(due(**base, f6c=pbc.HARVEST_FESTIVAL_BIT))

    def test_harvest_festival_trip_is_enabled_and_can_be_disabled_for_comparison(self) -> None:
        base = dict(season=FALL, day=12, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT | pbc.HAWKER_BIT)
        with patch.dict(os.environ, {pbc.HARVEST_FESTIVAL_ENV: ""}):
            self.assertEqual(due(**base), "harvest_festival")
        with patch.dict(os.environ, {pbc.HARVEST_FESTIVAL_ENV: "0"}):
            self.assertIsNone(due(**base))

    def test_egg_festival_returns_until_both_prize_bits_are_in(self) -> None:
        base = dict(season=FALL, day=20, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT | pbc.HAWKER_BIT)
        self.assertEqual(due(**base), "egg_festival")
        self.assertEqual(due(**base, f6c=pbc.EGG_TURTLE_BIT), "egg_festival", "turtle won: the berry arm is still to win")
        self.assertEqual(due(**base, f6c=pbc.EGG_HOUSE_ARM_BIT), "egg_festival", "berry armed: the turtle is still to win")
        self.assertIsNone(due(**base, f6c=pbc.EGG_PRIZE_BITS))
        self.assertIsNone(due(**base, hour=10), "the hunt cannot finish before the festival closes")

    def test_festival_days_never_fall_through_to_the_hen_trade(self) -> None:
        for day in (12, 20):
            self.assertIsNone(due(season=FALL, day=day, f64=pbc.STATUE_BIT, f66=pbc.FISH_BIT,
                                  f6c=pbc.HARVEST_FESTIVAL_BIT | pbc.EGG_PRIZE_BITS))


class StopTests(unittest.TestCase):
    def test_ten_berries_still_schedule_the_missing_turtle(self) -> None:
        base = dict(year=1, season=FALL, day=20, berries=10,
                    f6c=pbc.EGG_HOUSE_ARM_BIT | 0x4000)
        self.assertEqual(due(**base), "egg_festival")
        self.assertIsNone(due(**{**base, "f6c": base["f6c"] | pbc.EGG_TURTLE_BIT}))
        self.assertIsNone(due(**{**base, "hour": 10}))

    def test_ten_berries_end_the_calendar(self) -> None:
        self.assertIsNone(due(berries=10))
        self.assertEqual(due(berries=9), "statue")

    def test_phase_is_optional_and_carries_the_source(self) -> None:
        for name, spec in pbc.SOURCES.items():
            (phase,) = pbc.power_berry_phases(name)
            self.assertEqual(phase.phase, f"COLLECT_POWER_BERRY_{name.upper()}")
            self.assertEqual(phase.params["source"], name)
            self.assertEqual(phase.params["timeout"], spec.timeout)
            self.assertEqual(phase.failure_policy, "optional")


class SnowFlowerTests(unittest.TestCase):
    def test_winter_schedules_the_seed_from_shelf_or_backpack(self):
        wram = ram(season=WINTER, day=1)
        wram[pbc.SHED_ROW_2] = pbc.SNOW_SEED_SHELF_BIT
        self.assertEqual(pbc.due_source(wram, force=True), "snow_flower")
        wram[pbc.SHED_ROW_2] = 0
        wram[field_spec("tool_backpack").address] = pbc.SNOW_SEED_TOOL
        self.assertEqual(pbc.due_source(wram, force=True), "snow_flower")
        wram[field_spec("season").address] = SPRING
        self.assertNotEqual(pbc.due_source(wram, force=True), "snow_flower")

    def test_festival_purchase_is_once_and_requires_2000_gold(self):
        from harvest.tasks.proposal import eve_festival_spec
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
        from retro_harness import WorldState
        wram = ram(day=23)
        for key, val in (("money", 2000), ("eve_hearts", 500), ("marriage_flags", 0x10)):
            spec = field_spec(key)
            raw = spec.to_storage(val)
            for i in range(spec.width):
                wram[spec.address+i] = (raw >> (8*i)) & 255
        with patch.dict(os.environ, {pbc.ENV_VAR: "1"}):
            self.assertTrue(pbc.snow_seed_due(wram))
            phase = eve_festival_spec(wram)
            self.assertTrue(phase.params["buy_snow_seed"])
            task = build_phase_task(TaskBuildContext(), phase, WorldState(frame=0, ram=wram, info={}, obs=None))
            self.assertTrue(task.buy_snow_seed)
            wram[pbc.SNOW_SEED_BOUGHT_ADDR+1] |= pbc.SNOW_SEED_BOUGHT_BIT >> 8
            self.assertFalse(pbc.snow_seed_due(wram))
            self.assertIsNone(eve_festival_spec(wram))

    def test_empty_winter_farm_still_plans_the_snow_flower(self):
        from harvest.planner.day_plan_phases import build_day_phases_from_ram
        wram = ram(season=WINTER, day=2, hens=0)
        wram[field_spec("tilemap").address] = 0x15
        wram[pbc.SHED_ROW_2] = pbc.SNOW_SEED_SHELF_BIT
        with patch.dict(os.environ, {pbc.ENV_VAR: "1"}):
            names = [p.phase for p in build_day_phases_from_ram(wram)]
        self.assertIn("COLLECT_POWER_BERRY_SNOW_FLOWER", names)


if __name__ == "__main__":
    unittest.main()


class FestivalGiveUpTests(unittest.TestCase):
    """A festival trip that can no longer reach the 15:00 close ends at once instead of waiting out the day."""

    def _task(self, source, phase, hour, tilemap=0x10):
        from retro_harness import WorldState
        from harvest.tasks.power_berry import PowerBerryTask

        world = WorldState(frame=0, ram=ram(season=FALL, day=12, hour=hour), info={}, obs=None)
        world.ram[field_spec("tilemap").address] = tilemap
        task = PowerBerryTask(source=source)
        task.reset(world)
        task._phase, task._child = phase, None
        return task, world

    def test_fishing_at_two_in_the_afternoon_gives_up_the_harvest_festival(self) -> None:
        task, world = self._task("harvest_festival", "source", 14)
        result = task.step(world)
        self.assertEqual(result.status.value, "failure")
        self.assertIn("too late for the festival", result.reason)

    def test_the_same_trip_continues_before_that_hour(self) -> None:
        task, world = self._task("harvest_festival", "source", 13)
        self.assertNotIn("too late", task.step(world).reason or "")

    def test_a_cast_line_is_reeled_in_before_the_give_up_ends_the_task(self) -> None:
        """fdet_20261115 Y1 Fall D12: gave up at 14:00 with the rod cast (action 17), which ignores every button but Y and
        pinned the farmer at (633,371) through eight rescues. Wait for the bite, reel it once, never recast, then fail."""
        from harvest.tasks.nav import make_action
        task, world = self._task("harvest_festival", "source", 14)
        action_addr = field_spec("player_action").address
        world.ram[action_addr] = 17                       # cast line, waiting for a bite
        for _ in range(30):
            result = task.step(world)
            self.assertEqual(result.status.value, "running")
            self.assertFalse(result.action.action[make_action(y=True).nonzero()[0][0]])  # no Y: it would recast
        world.ram[action_addr] = 18                       # the bite: one Y reels it
        pressed = [bool(task.step(world).action.action[make_action(y=True).nonzero()[0][0]]) for _ in range(12)]
        self.assertTrue(any(pressed))
        world.ram[action_addr] = 19                       # the reel animation: hands off
        self.assertEqual(task.step(world).status.value, "running")
        world.ram[action_addr] = 15                       # rod rested: the farmer can walk, so the give-up lands
        result = task.step(world)
        self.assertEqual(result.status.value, "failure")
        self.assertIn("too late for the festival", result.reason)

    def test_camp_exit_walks_down_then_left_along_the_gap_row_then_down(self) -> None:
        from harvest.tasks.nav import make_action
        from harvest.tasks.power_berry import camp_exit_pad

        def keys(pad):
            return None if pad is None else tuple(int(v) for v in pad.nonzero()[0])

        down, left = keys(make_action(down=True)), keys(make_action(left=True))
        self.assertEqual(keys(camp_exit_pad(0x10, 633, 371)), down)   # the cast stand: down to the gap row
        self.assertEqual(keys(camp_exit_pad(0x10, 633, 395)), left)   # along it to the gap
        self.assertEqual(keys(camp_exit_pad(0x10, 584, 400)), down)   # through the gap
        self.assertIsNone(camp_exit_pad(0x10, 584, 430))              # out
        self.assertIsNone(camp_exit_pad(0x10, 400, 371))              # elsewhere on the mountain
        self.assertIsNone(camp_exit_pad(0x09, 633, 371))              # inside the festival map

    def test_a_farmer_who_never_cast_gives_up_at_once(self) -> None:
        task, world = self._task("harvest_festival", "source", 14)
        world.ram[field_spec("player_action").address] = 0
        self.assertEqual(task.step(world).status.value, "failure")

    def test_inside_the_festival_map_the_clock_no_longer_matters(self) -> None:
        task, world = self._task("egg_festival", "festival", 15, tilemap=0x0B)
        self.assertNotIn("too late", task.step(world).reason or "")


def test_egg_prize_pickup_owns_the_festivals_overnight_transition():
    assert pbc.power_berry_phases("egg_festival")[0].params["ends_day"] is True
    assert not pbc.power_berry_phases("harvest_festival")[0].params.get("ends_day")
