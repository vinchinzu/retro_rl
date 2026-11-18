"""Unit tests for the authoritative Harvest Moon SNES ROM RNG catalog."""
from __future__ import annotations

import unittest
import numpy as np

from harvest.core.rng_catalog import (
    AXE_STUMP_BERRY_ROLL_DENOMINATOR,
    COW_CRANKY_HAPPINESS_PENALTY,
    COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR,
    COW_CRANKY_START_TIMER,
    COW_SICK_HAPPINESS_PENALTY,
    COW_SICK_ROLL_DENOMINATOR,
    COW_SICKNESS_START_TIMER,
    COW_UNFED_HAPPINESS_PENALTY,
    COW_UNFED_SICK_OVERNIGHT_DELTA,
    DAILY_LOCK_CAVE_STATUE,
    DAILY_LOCK_FARM_BERRY,
    DAILY_LOCK_FARM_GOLD,
    DAMAGE_PROBABILITY_TABLE,
    EGG_FESTIVAL_INITIAL_SLOTS,
    FISHING_CATCH_VARIANTS,
    FLAG5C_FALL_ANIMAL,
    FLAG5C_SPRING_ANIMAL,
    FLAG5C_WINTER_ANIMAL,
    FLAG64_BLIZZARD_SEEN,
    FLAG64_THUNDER_SEEN,
    FLAG6E_BLIZZARD_EVENT,
    FLAG6E_THUNDER_EVENT,
    HAMMER_CAVE_BERRY_ROLL_DENOMINATOR,
    HOE_COIN_ROLL_DENOMINATOR,
    HOE_MOLE_ROLL_DENOMINATOR,
    HOE_MONEY_BAG_ROLL_DENOMINATOR,
    HOE_POWER_BERRY_ROLL_DENOMINATOR,
    HURRICANE_CHANCE_TABLE,
    HURRICANE_DENOMINATOR_BASE,
    HURRICANE_DENOMINATOR_TURTLE,
    MARRIED_BREAKFAST_VARIANTS,
    MORNING_BIRD_HAPPINESS_BONUS,
    MORNING_BIRD_ROLL_DENOMINATOR,
    MOUNTAIN_ANIMAL_ROLL_DENOMINATOR,
    POWER_BERRY_CAVE_STATUE,
    POWER_BERRY_FARM_HOE_1,
    POWER_BERRY_FARM_HOE_2,
    POWER_BERRY_MOUNTAIN_STUMP,
    RAIN_CHANCE_TABLE,
    SICKLE_FROG_ROLL_DENOMINATOR,
    SNOW_CHANCE_TABLE,
    SNOWSTORM_CHANCE_TABLE,
    THUNDER_CHANCE_TABLE,
    TURTLE_SHELL_FLAG_BIT,
    WEATHER_CODE_BLIZZARD,
    WEATHER_CODE_EGG_FESTIVAL,
    WEATHER_CODE_FLOWER_FESTIVAL,
    WEATHER_CODE_HARVEST_FESTIVAL,
    WEATHER_CODE_HURRICANE,
    WEATHER_CODE_NEW_YEAR,
    WEATHER_CODE_RAIN,
    WEATHER_CODE_SNOW,
    WEATHER_CODE_STAR_NIGHT,
    WEATHER_CODE_SUNNY,
    WEATHER_CODE_THANKSGIVING,
    WEATHER_CODE_THUNDER,
    WEATHER_FLAG_BLIZZARD,
    WEATHER_FLAG_HURRICANE,
    WEATHER_FLAG_RAIN,
    WEATHER_FLAG_SNOW,
    WEATHER_FLAG_SUNNY_SUMMER,
    WEATHER_FLAG_THUNDER,
    WEATHER_FLAG_WILD_DOG_BREACH,
    WEED_SPROUT_ROLL_DENOMINATOR,
    WILD_DOG_BREACH_ROLL_DENOMINATOR,
    YEAR1_BLIZZARD_DAY,
    YEAR1_THUNDER_DAY,
    can_axe_mountain_stump_berry,
    can_dig_farm_power_berry,
    can_hammer_cave_statue_berry,
    cow_outside_cranky_probability,
    expected_farm_damage,
    expected_weed_sprouts,
    forecast_weather_distribution,
    hoe_expected_yield_per_till,
    shuffle_egg_festival_slots,
    storm_damage_probabilities,
    weather_code_to_today_flags,
    wild_dog_breach_probability,
)


class WeatherRNGTests(unittest.TestCase):
    def test_weather_chance_tables_match_rom_decomp(self) -> None:
        """Verify exact ROM table arrays from bank_82.asm:1580-1585."""
        self.assertEqual(HURRICANE_CHANCE_TABLE, (0, 0x1E, 0, 0))  # Summer 30
        self.assertEqual(RAIN_CHANCE_TABLE, (6, 10, 10, 0))        # db $06,$0A,$0A,$00
        self.assertEqual(SNOW_CHANCE_TABLE, (0, 0, 0, 3))          # db $00,$00,$00,$03
        self.assertEqual(THUNDER_CHANCE_TABLE, (0, 0x1E, 0, 0))    # db $00,$1E,$00,$00
        self.assertEqual(SNOWSTORM_CHANCE_TABLE, (0, 0, 0, 8))     # db $00,$00,$00,$08

        # Turtle shell doubling
        self.assertEqual(HURRICANE_DENOMINATOR_BASE, 30)
        self.assertEqual(HURRICANE_DENOMINATOR_TURTLE, 60)
        self.assertEqual(TURTLE_SHELL_FLAG_BIT, 0x1000)

        # Guaranteed dates
        self.assertEqual(YEAR1_BLIZZARD_DAY, 7)
        self.assertEqual(YEAR1_THUNDER_DAY, 29)

    def test_forecast_weather_distribution_festivals(self) -> None:
        """Festival days have 100% deterministic weather forecast."""
        self.assertEqual(forecast_weather_distribution(0, 22), {WEATHER_CODE_FLOWER_FESTIVAL: 1.0})
        self.assertEqual(forecast_weather_distribution(2, 11), {WEATHER_CODE_HARVEST_FESTIVAL: 1.0})
        self.assertEqual(forecast_weather_distribution(2, 19), {WEATHER_CODE_EGG_FESTIVAL: 1.0})
        self.assertEqual(forecast_weather_distribution(3, 9), {WEATHER_CODE_THANKSGIVING: 1.0})
        self.assertEqual(forecast_weather_distribution(3, 23), {WEATHER_CODE_STAR_NIGHT: 1.0})
        self.assertEqual(forecast_weather_distribution(3, 30), {WEATHER_CODE_NEW_YEAR: 1.0})

    def test_forecast_weather_distribution_seasons(self) -> None:
        # Spring: 1/6 rain, 5/6 sunny
        dist_spring = forecast_weather_distribution(0, 10)
        self.assertAlmostEqual(dist_spring[WEATHER_CODE_RAIN], 1.0 / 6)
        self.assertAlmostEqual(dist_spring[WEATHER_CODE_SUNNY], 5.0 / 6)

        # Winter D7 in Year 1: guaranteed Blizzard
        dist_w7 = forecast_weather_distribution(3, 7, year=0, blizzard_seen=False)
        self.assertEqual(dist_w7, {WEATHER_CODE_BLIZZARD: 1.0})

        # Winter after blizzard seen: 1/3 snow, 2/3 sunny
        dist_winter = forecast_weather_distribution(3, 15, year=0, blizzard_seen=True)
        self.assertAlmostEqual(dist_winter[WEATHER_CODE_SNOW], 1.0 / 3)
        self.assertAlmostEqual(dist_winter[WEATHER_CODE_SUNNY], 2.0 / 3)

        # Summer hurricane probability comparison: without vs with turtle shell
        dist_summer_base = forecast_weather_distribution(1, 15, year=0, thunder_seen=True, turtle_shell_owned=False)
        dist_summer_turtle = forecast_weather_distribution(1, 15, year=0, thunder_seen=True, turtle_shell_owned=True)
        self.assertAlmostEqual(dist_summer_base[WEATHER_CODE_HURRICANE], 1.0 / 30)
        self.assertAlmostEqual(dist_summer_turtle[WEATHER_CODE_HURRICANE], 1.0 / 60)

    def test_weather_code_to_today_flags(self) -> None:
        """Verify SetWeatherFlags ($828C09) flag settings."""
        flags, f64, f6e = weather_code_to_today_flags(WEATHER_CODE_RAIN, season=0)
        self.assertEqual(flags, WEATHER_FLAG_RAIN)
        self.assertEqual(f64, 0)
        self.assertEqual(f6e, 0)

        # Blizzard sets 0x0200, sets $7F1F64 bit 0x0002, and sets $7F1F6E bit 0x0100
        flags, f64, f6e = weather_code_to_today_flags(WEATHER_CODE_BLIZZARD, season=3)
        self.assertEqual(flags, WEATHER_FLAG_BLIZZARD)
        self.assertEqual(f64, FLAG64_BLIZZARD_SEEN)
        self.assertEqual(f6e, FLAG6E_BLIZZARD_EVENT)

        # Thunderstorm sets 0x0100, sets $7F1F64 bit 0x0001, and sets $7F1F6E bit 0x0080
        flags, f64, f6e = weather_code_to_today_flags(WEATHER_CODE_THUNDER, season=1)
        self.assertEqual(flags, WEATHER_FLAG_THUNDER)
        self.assertEqual(f64, FLAG64_THUNDER_SEEN)
        self.assertEqual(f6e, FLAG6E_THUNDER_EVENT)


class StormDamageTests(unittest.TestCase):
    def test_damage_probability_table_rates(self) -> None:
        """Verify exact ROM denominators from DamageProbabilityTable (bank_82.asm:344)."""
        self.assertEqual(DAMAGE_PROBABILITY_TABLE["rain"], (96, 0, 0))
        self.assertEqual(DAMAGE_PROBABILITY_TABLE["snow"], (64, 0, 0))
        self.assertEqual(DAMAGE_PROBABILITY_TABLE["hurricane"], (8, 16, 4))
        self.assertEqual(DAMAGE_PROBABILITY_TABLE["blizzard"], (8, 0, 4))
        self.assertEqual(DAMAGE_PROBABILITY_TABLE["new_year"], (0, 64, 32))

    def test_storm_damage_probabilities_and_expectations(self) -> None:
        # Hurricane: 1/8 fence, 1/16 grass, 1/4 crop
        pf, pg, pc = storm_damage_probabilities(WEATHER_FLAG_HURRICANE)
        self.assertEqual(pf, 1.0 / 8)
        self.assertEqual(pg, 1.0 / 16)
        self.assertEqual(pc, 1.0 / 4)

        # Out of 80 fences, 80 grass, 80 crops: expect 10 fences, 5 grass, 20 crops damaged
        ef, eg, ec = expected_farm_damage(80, 80, 80, WEATHER_FLAG_HURRICANE)
        self.assertAlmostEqual(ef, 10.0)
        self.assertAlmostEqual(eg, 5.0)
        self.assertAlmostEqual(ec, 20.0)

        # Rain: only fences damaged at 1/96 (~1.04%)
        pf, pg, pc = storm_damage_probabilities(WEATHER_FLAG_RAIN)
        self.assertEqual(pf, 1.0 / 96)
        self.assertEqual(pg, 0.0)
        self.assertEqual(pc, 0.0)


class FarmDecayAndWildDogsTests(unittest.TestCase):
    def test_wild_dog_breach_probability(self) -> None:
        # 0 broken fences -> 0% breach risk
        self.assertEqual(wild_dog_breach_probability(0), 0.0)
        # 1 broken fence -> 1/8 (12.5%) breach risk
        self.assertEqual(wild_dog_breach_probability(1), 1.0 / 8)
        # 4 broken fences -> 1 - (7/8)^4
        self.assertAlmostEqual(wild_dog_breach_probability(4), 1.0 - (7.0 / 8.0) ** 4)

    def test_weed_sprouting(self) -> None:
        # Spring D4 (day % 4 == 0): 256 empty tiles -> expect 1.0 weed sprout
        self.assertEqual(expected_weed_sprouts(256, season=0, day=4), 1.0)
        # Spring D3 (day % 4 != 0): 0
        self.assertEqual(expected_weed_sprouts(256, season=0, day=3), 0.0)
        # Fall (season 2): 0
        self.assertEqual(expected_weed_sprouts(256, season=2, day=4), 0.0)


class ToolDropsAndPowerBerryTests(unittest.TestCase):
    def test_tool_roll_denominators(self) -> None:
        """Verify exact tool roll denominators from bank_82_toolused_subrutines.asm."""
        self.assertEqual(HOE_POWER_BERRY_ROLL_DENOMINATOR, 64)   # LDA.B #$40
        self.assertEqual(HOE_MOLE_ROLL_DENOMINATOR, 32)          # LDA.B #$20
        self.assertEqual(HOE_COIN_ROLL_DENOMINATOR, 16)          # LDA.B #$10
        self.assertEqual(HOE_MONEY_BAG_ROLL_DENOMINATOR, 16)     # LDA.B #$10
        self.assertEqual(HAMMER_CAVE_BERRY_ROLL_DENOMINATOR, 4)  # LDA.B #$04
        self.assertEqual(AXE_STUMP_BERRY_ROLL_DENOMINATOR, 16)   # LDA.B #$10
        self.assertEqual(SICKLE_FROG_ROLL_DENOMINATOR, 16)       # LDA.B #$10

    def test_power_berry_eligibility(self) -> None:
        # Farm hoe berry: eligible if on farm, daily flag clear, and at least one berry not yet owned
        self.assertTrue(can_dig_farm_power_berry(0, 0, on_farm=True))
        self.assertFalse(can_dig_farm_power_berry(0, 0, on_farm=False))
        # Daily lock bit set -> ineligible today
        self.assertFalse(can_dig_farm_power_berry(0, DAILY_LOCK_FARM_BERRY, on_farm=True))
        # Both berries owned (0x0800 | 0x1000 = 0x1800) -> ineligible permanently
        self.assertFalse(can_dig_farm_power_berry(POWER_BERRY_FARM_HOE_1 | POWER_BERRY_FARM_HOE_2, 0, on_farm=True))
        # One berry owned -> still eligible for second
        self.assertTrue(can_dig_farm_power_berry(POWER_BERRY_FARM_HOE_1, 0, on_farm=True))

        # Cave statue hammer berry
        self.assertTrue(can_hammer_cave_statue_berry(0, 0, in_cave=True))
        self.assertFalse(can_hammer_cave_statue_berry(0, DAILY_LOCK_CAVE_STATUE, in_cave=True))
        self.assertFalse(can_hammer_cave_statue_berry(POWER_BERRY_CAVE_STATUE, 0, in_cave=True))

        # Mountain stump axe berry
        self.assertTrue(can_axe_mountain_stump_berry(0, in_mountain=True))
        self.assertFalse(can_axe_mountain_stump_berry(POWER_BERRY_MOUNTAIN_STUMP, in_mountain=True))

    def test_hoe_expected_yield_per_till(self) -> None:
        # On fresh day in Spring with all drops available
        yields = hoe_expected_yield_per_till(upgrade_flags=0, daily_flags_1f5c=0, on_farm=True, season=0)
        self.assertAlmostEqual(yields["mole"], 1.0 / 32)
        # Gold expectation should be strictly positive (from coin & money bag)
        self.assertGreater(yields["expected_gold"], 0.0)
        self.assertGreater(yields["power_berry"], 0.0)


class MiscellaneousRNGMechanicsTests(unittest.TestCase):
    def test_mountain_animal_constants(self) -> None:
        self.assertEqual(MOUNTAIN_ANIMAL_ROLL_DENOMINATOR, 16)
        self.assertEqual(FLAG5C_SPRING_ANIMAL, 0x2000)
        self.assertEqual(FLAG5C_FALL_ANIMAL, 0x4000)
        self.assertEqual(FLAG5C_WINTER_ANIMAL, 0x8000)

    def test_morning_bird_event(self) -> None:
        self.assertEqual(MORNING_BIRD_ROLL_DENOMINATOR, 8)
        self.assertEqual(MORNING_BIRD_HAPPINESS_BONUS, 2)

    def test_egg_festival_shuffle(self) -> None:
        self.assertEqual(EGG_FESTIVAL_INITIAL_SLOTS, (0, 1, 2, 3, 4, 5, 0, 0))
        # Deterministic swap sequence
        swaps = [7, 6, 5, 4, 3, 2, 1]
        shuffled = shuffle_egg_festival_slots(rng_sequence=swaps)
        self.assertEqual(len(shuffled), 8)
        # Initial multiset of values must be preserved
        self.assertEqual(sorted(shuffled), sorted(EGG_FESTIVAL_INITIAL_SLOTS))

    def test_fishing_and_breakfast_variants(self) -> None:
        self.assertEqual(FISHING_CATCH_VARIANTS, 3)
        self.assertEqual(MARRIED_BREAKFAST_VARIANTS, 3)


if __name__ == "__main__":
    unittest.main()
