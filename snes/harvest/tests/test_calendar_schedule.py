"""PR-021 calendar: festival dates, door hours, measured girl cells."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from harvest.calendar_schedule import (
    CARPENTER_INTRO_TEXT_ID,
    DOOR_FLAGS,
    GIRL_SIGHTINGS,
    SHOP_NAMES,
    carpenter_availability,
    door_refuses,
    festival_on,
    girl_location,
    shop_open,
    venue_open,
)
from harvest.core.npc_catalog import ROMANCE_NPCS

_ASM = (
    Path(__file__).resolve().parents[1]
    / "HM-Decomp"
    / "src"
    / "code_banks"
    / "bank_80.asm"
)


def _unk_table13() -> bytes:
    text = _ASM.read_text(encoding="utf-8")
    chunk = text[text.index("Unk_Table13:") : text.index("UNK_AudioTable:")]
    return bytes(int(part, 16) for part in re.findall(r"\$([0-9A-Fa-f]{2})", chunk))


class CalendarScheduleTests(unittest.TestCase):
    def test_door_flags_match_unk_table13(self) -> None:
        data = _unk_table13()
        self.assertEqual(len(data) % 8, 0)
        for name, (index, dest, flags) in DOOR_FLAGS.items():
            base = index * 8
            self.assertEqual(data[base], dest, name)
            self.assertEqual(data[base + 3], flags, name)

    def test_festival_dates_and_forced(self) -> None:
        flower = festival_on(0, 0, 23)
        assert flower is not None
        self.assertEqual(flower.name, "Flower Festival")
        self.assertEqual(flower.forecast_code, 0x06)
        self.assertFalse(flower.forced)
        star = festival_on(0, 3, 24)
        assert star is not None
        self.assertEqual(star.name, "Star Night")
        self.assertEqual(star.forecast_code, 0x09)
        self.assertTrue(star.forced)
        self.assertIsNone(festival_on(0, 0, 1))
        new_year = festival_on(1, 0, 1)
        assert new_year is not None
        self.assertEqual(new_year.name, "New Year")
        self.assertEqual(new_year.forecast_code, 0x0A)
        self.assertTrue(new_year.forced)
        self.assertIsNone(festival_on(0, 1, 8))
        self.assertIsNone(festival_on(0, 3, 30))
        harvest = festival_on(0, 2, 12)
        egg = festival_on(0, 2, 20)
        thanks = festival_on(0, 3, 10)
        assert harvest and egg and thanks
        self.assertEqual(
            (harvest.forecast_code, egg.forecast_code, thanks.forecast_code),
            (0x07, 0x0B, 0x08),
        )
        self.assertFalse(harvest.forced or egg.forced or thanks.forced)

    def test_shops_shut_saturday_sunday_and_from_17(self) -> None:
        open_day = dict(weekday=3, hour=10, season=0, day=4, weather_flags=0)
        for name in SHOP_NAMES:
            self.assertTrue(shop_open(name, **open_day), name)
            self.assertFalse(shop_open(name, **{**open_day, "weekday": 6}), name)
            self.assertFalse(shop_open(name, **{**open_day, "weekday": 0}), name)
            self.assertFalse(shop_open(name, **{**open_day, "hour": 17}), name)
            # Bit 0x10 also refuses Winter D10 (Thanksgiving), any weekday.
            self.assertFalse(
                shop_open(name, weekday=3, hour=10, season=3, day=10, weather_flags=0),
                name,
            )
        self.assertTrue(venue_open("bar", weekday=3, hour=17, season=0, day=4))
        self.assertFalse(venue_open("bar", weekday=3, hour=10, season=0, day=4))
        self.assertFalse(venue_open("bar", weekday=0, hour=18, season=0, day=5))
        self.assertTrue(venue_open("church", weekday=0, hour=10, season=0, day=5))
        self.assertFalse(venue_open("church", weekday=3, hour=17, season=0, day=4))
        self.assertTrue(
            door_refuses(0x81, weekday=3, hour=10, season=0, day=4, weather_flags=0x0010)
        )
        self.assertFalse(
            door_refuses(0x32, weekday=3, hour=10, season=0, day=4, weather_flags=0x0010)
        )

    def test_carpenter_is_not_a_labeled_shop_door(self) -> None:
        info = carpenter_availability()
        self.assertEqual(info["intro_text_id"], CARPENTER_INTRO_TEXT_ID)
        self.assertIsNone(info["shop_door"])
        self.assertFalse(info["map_2b_is_carpenter"])
        self.assertEqual(DOOR_FLAGS["map_2b"][1:], (0x2B, 0x32))

    def test_three_sample_days_locate_five_girls(self) -> None:
        samples = {row.sample for row in GIRL_SIGHTINGS}
        self.assertEqual(samples, {"weekday", "saturday", "sunday"})
        for sample in samples:
            names = {row.name for row in GIRL_SIGHTINGS if row.sample == sample}
            self.assertEqual(names, set(ROMANCE_NPCS), sample)
            for row in GIRL_SIGHTINGS:
                if row.sample != sample:
                    continue
                hit = girl_location(
                    row.name,
                    season=row.season,
                    weekday=row.weekday,
                    hour=row.hour,
                    weather_flags=row.weather_flags,
                )
                self.assertEqual(hit, row)
