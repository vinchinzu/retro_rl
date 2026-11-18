"""Forage by season: the ROM table, the stand registry, the routes and where the trip sits in the day."""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import patch

from harvest.maps.map_config import ROUTES, SEGMENTS
from harvest.planner import forage_calendar as fc
from harvest.planner.day_plan import DayPlannerPolicy, build_day_phases
from harvest.tasks.mountain_berry import FORAGE_STANDS

DECOMP = Path(__file__).resolve().parents[1] / "HM-Decomp" / "src" / "code_banks"
SPRING, SUMMER, FALL, WINTER = 0, 1, 2, 3


def _db_bytes(path: Path, label: str, lines: int) -> list[int]:
    rows = path.read_text().splitlines()
    start = next(i for i, row in enumerate(rows) if row.startswith(f"{label}:"))
    out: list[int] = []
    for row in rows[start:start + lines]:
        match = re.search(r"db ([^;]*);", row)
        if not match:
            break
        out += [int(tok.strip().lstrip("$"), 16) for tok in match.group(1).split(",") if tok.strip()]
    return out


@unittest.skipUnless((DECOMP / "bank_82.asm").exists(), "HM-Decomp not checked out")
class ForageRomTableTests(unittest.TestCase):
    """``ForageKind`` is the ROM's ``DATA8_82CFB4[tile * 4 + season]`` and ``Items_Price_Table``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.items = _db_bytes(DECOMP / "bank_82.asm", "DATA8_82CFB4", 34)
        cls.prices = _db_bytes(DECOMP / "bank_81.asm", "Items_Price_Table", 12)

    def held(self, tile: int, season: int) -> int:
        return self.items[tile * 4 + season]

    def price(self, item: int) -> int:
        return self.prices[item * 2 + 1] * 10

    def test_each_kind_matches_its_rom_tile_season_and_price(self) -> None:
        tile_of = {"grape": 0x30, "green_fruit": 0x32, "mushroom": 0x33, "poison_mushroom": 0x34}
        for kind in fc.ALL_KINDS:
            tile = tile_of[kind.name]
            self.assertEqual(self.held(tile, kind.season), kind.item_id, kind.name)
            self.assertEqual(self.price(kind.item_id), kind.price_g, kind.name)

    def test_a_kind_gives_nothing_outside_its_own_season(self) -> None:
        tile_of = {"grape": 0x30, "green_fruit": 0x32, "mushroom": 0x33, "poison_mushroom": 0x34}
        for kind in fc.ALL_KINDS:
            for season in (SPRING, SUMMER, FALL, WINTER):
                if season != kind.season:
                    self.assertEqual(self.held(tile_of[kind.name], season), 0, (kind.name, season))

    def test_the_poison_mushroom_outsells_the_mushroom(self) -> None:
        self.assertEqual(self.held(0x34, FALL), 2)
        self.assertGreater(self.price(2), self.price(1))


class ForageRegistryTests(unittest.TestCase):
    def test_one_kind_per_season_and_winter_has_none(self) -> None:
        self.assertEqual(fc.forage_for_season(SPRING).name, "grape")
        self.assertEqual(fc.forage_for_season(SUMMER).name, "green_fruit")
        self.assertEqual(fc.forage_for_season(FALL).name, "poison_mushroom")
        self.assertIsNone(fc.forage_for_season(WINTER))

    def test_every_kind_has_a_stand_a_climb_and_a_bin_route(self) -> None:
        for kind in fc.ALL_KINDS:
            stand = FORAGE_STANDS[kind.name]
            self.assertIn(kind.item_id, stand.items, kind.name)
            self.assertTrue(ROUTES.get(stand.ship_route), kind.name)
            self.assertTrue(all(SEGMENTS.get(name) for name in stand.segments), kind.name)

    def test_green_fruit_stand_is_the_tile_below_the_fruit_facing_up(self) -> None:
        stand = FORAGE_STANDS["green_fruit"]
        (fx, fy), = fc.GREEN_FRUIT.tiles
        self.assertEqual((stand.stand_px[0] // 16, stand.stand_px[1] // 16), (fx, fy + 1))
        self.assertEqual(stand.face, "up")
        self.assertGreater(stand.push_frames, 1)
        last = ROUTES["green_fruit_to_shipping_bin"][-1]
        self.assertEqual(last.tilemap, 0x00)  # ends on the farm at the bin

    def test_loops_that_fit_before_the_bin_closes(self) -> None:
        self.assertEqual(fc.loops_before_bin_closes(6, 0, 3_300), 2)
        self.assertEqual(fc.loops_before_bin_closes(13, 0, 3_300), 0)
        self.assertEqual(fc.trips_today(fc.GREEN_FRUIT, 6), 2)  # the tile gives again the same day
        self.assertEqual(fc.trips_today(fc.GRAPE, 6), 2)
        self.assertEqual(fc.trips_today(fc.GRAPE, 15), 0)


def _names(phases) -> list[str]:
    return [phase.phase for phase in phases]


class ForageOrderTests(unittest.TestCase):
    """The trip goes ahead of crop work only when nothing is ripe to ship."""

    def plan(self, **kw):
        base = dict(
            weekday=2, hour=7, season=SUMMER, day=10, year=0, has_waterable=True,
            has_harvest=False, has_seeds=True, policy=DayPlannerPolicy(),
        )
        base.update(kw)
        return build_day_phases(None, **base)

    def test_summer_without_a_harvest_forages_the_green_fruit_before_watering(self) -> None:
        phases = self.plan()
        names = _names(phases)
        self.assertLess(names.index("MOUNTAIN_BERRY"), names.index("CROP_WATER"))
        berry = phases[names.index("MOUNTAIN_BERRY")]
        self.assertEqual(berry.params["forage"], "green_fruit")
        self.assertEqual(berry.params["count"], 2)

    def test_a_harvest_day_keeps_the_morning_and_forages_after_the_crop_work(self) -> None:
        names = _names(self.plan(has_harvest=True))
        self.assertLess(names.index("HARVEST_ROUTE"), names.index("CROP_WATER"))
        if "MOUNTAIN_BERRY" in names:
            self.assertLess(names.index("CROP_WATER"), names.index("MOUNTAIN_BERRY"))

    def test_spring_grape_days_ask_for_two_grapes_ahead_of_the_crop_work(self) -> None:
        phases = self.plan(season=SPRING, day=12)
        names = _names(phases)
        self.assertLess(names.index("MOUNTAIN_BERRY"), names.index("CROP_WATER"))
        berry = phases[names.index("MOUNTAIN_BERRY")]
        self.assertEqual(berry.params["count"], 2)
        self.assertNotIn("forage", berry.params)  # the grape is the task default

    def test_year_two_forages_too(self) -> None:
        for season in (SPRING, SUMMER):
            self.assertIn("MOUNTAIN_BERRY", _names(self.plan(season=season, year=1)))

    def test_fall_and_winter_plan_no_berry_run(self) -> None:
        for season in (FALL, WINTER):
            self.assertNotIn("MOUNTAIN_BERRY", _names(self.plan(season=season)))

    def test_a_late_start_or_a_festival_skips_the_morning_slot(self) -> None:
        names = _names(self.plan(hour=13))
        if "MOUNTAIN_BERRY" in names:
            self.assertGreater(names.index("MOUNTAIN_BERRY"), names.index("CROP_WATER"))
        fest = _names(self.plan(is_festival=True))  # the festival keeps its morning: crop work first
        self.assertLess(fest.index("CROP_WATER"), fest.index("MOUNTAIN_BERRY"))

    def test_forage_first_can_be_switched_off_for_an_ab(self) -> None:
        with patch.object(fc, "FORAGE_FIRST", False):
            names = _names(self.plan())
        self.assertLess(names.index("CROP_WATER"), names.index("MOUNTAIN_BERRY"))


if __name__ == "__main__":
    unittest.main()


class SpaRefreshRuleTests(unittest.TestCase):
    """Watering may soak again while its last pass watered tiles and left some unfunded."""

    def allowed(self, phase, done, seen, reason=""):
        from harvest.planner.day_phase_stamina import spa_refresh_allowed

        return spa_refresh_allowed(phase, done, set(seen), reason)

    def test_first_refill_of_a_phase_is_always_allowed(self) -> None:
        self.assertTrue(self.allowed("CROP_WATER", 0, []))
        self.assertTrue(self.allowed("CLEAR_FIELD", 0, []))

    def test_watering_soaks_again_only_after_a_pass_that_watered_and_left_tiles(self) -> None:
        low = "planted=0 watered=45 skipped=2 completed=45 refills=2 passes=1 stamina_low unfunded=26"
        self.assertTrue(self.allowed("CROP_WATER", 1, ["CROP_WATER"], low))
        self.assertFalse(self.allowed("CROP_WATER", 1, ["CROP_WATER"], "watered=0 stamina_low unfunded=26"))
        self.assertFalse(self.allowed("CROP_WATER", 1, ["CROP_WATER"], "watered=45 passes=1"))
        self.assertFalse(self.allowed("CROP_WATER", 1, ["CROP_WATER"], ""))

    def test_other_phases_keep_one_refill_each_and_two_a_day(self) -> None:
        low = "cleared=3 stamina_low watered=9 unfunded=4"
        self.assertFalse(self.allowed("CLEAR_FIELD", 1, ["CLEAR_FIELD"], low))
        self.assertFalse(self.allowed("TILL_FIELD", 2, [], low))

    def test_watering_stops_at_four_soaks_a_day(self) -> None:
        low = "watered=40 stamina_low unfunded=30"
        self.assertTrue(self.allowed("CROP_WATER", 3, ["CROP_WATER"], low))
        self.assertFalse(self.allowed("CROP_WATER", 4, ["CROP_WATER"], low))
