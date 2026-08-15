"""In-game clock timeline, lunch mark, and berry-run frame benches."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.game_clock import (
    BERRY_SHIP_BENCH,
    ClockTime,
    ClockTimeline,
    LUNCH_TIME,
    compare_frame_benches,
    format_segment_time,
    mark_from_mapping,
    path_waste,
)
from harvest.planner.day_plan import DayPlannerPolicy, WorldProbe, build_day_phases
from harvest.planner.day_plan_decision import _planning_notes, build_day_plan_decision

from day_plan_test_helpers import make_date_world, set_player_pos


def _row(frame, hour, minute, tilemap, x, y, **extra):
    row = {
        "frame": frame,
        "hour": hour,
        "minute": minute,
        "tilemap": tilemap,
        "x": x,
        "y": y,
    }
    row.update(extra)
    return row


# Hour-by-hour stands from recordings/mountain_segments_after.json
# (Y1_Inside_House ship, 3224f, 06:08→10:11). Lunch is after the ship.
_BERRY_HOUR_SAMPLES = [
    _row(0, 6, 8, 0x15, 128, 200, map="house", phase="pick"),
    _row(18, 6, 8, 0x00, 132, 212, map="farm", phase="pick"),
    _row(356, 6, 10, 0x0C, 10, 422, map="path", phase="pick"),
    _row(662, 7, 12, 0x10, 137, 10, map="mountain_spring", phase="pick"),
    _row(1670, 8, 0, 0x10, 326, 409, map="mountain_spring", phase="pick"),
    _row(1941, 8, 12, 0x10, 326, 409, map="mountain_spring", phase="return_to_bin"),
    _row(2373, 9, 4, 0x0C, 314, 740, map="path", phase="return_to_bin"),
    _row(2857, 10, 9, 0x00, 244, 118, map="farm", phase="return_to_bin"),
    _row(3224, 10, 11, 0x00, 135, 456, map="farm", phase="done"),
]


class ClockTimeTests(unittest.TestCase):
    def test_order_and_lunch_slack(self) -> None:
        dawn = ClockTime(6, 8)
        lunch = LUNCH_TIME
        self.assertLess(dawn, lunch)
        self.assertEqual(dawn.minutes_until(lunch), 352)
        self.assertEqual(str(dawn), "06:08")
        self.assertFalse(dawn >= lunch)
        self.assertTrue(ClockTime(12, 0) >= lunch)

    def test_format_segment_time_matches_play_clock(self) -> None:
        clock = format_segment_time(1016)
        self.assertEqual(clock["seconds"], 16.933)
        self.assertEqual(clock["clock"], "00:16.93")


class ClockTimelineTests(unittest.TestCase):
    def test_berry_run_hour_by_hour_locations(self) -> None:
        timeline = ClockTimeline.from_samples(_BERRY_HOUR_SAMPLES)
        hours = {mark.clock.hour: mark for mark in timeline.hour_marks()}
        self.assertEqual(str(timeline.start.clock), "06:08")
        self.assertEqual(hours[6].map_name, "house")
        self.assertEqual((hours[6].x, hours[6].y), (128, 200))
        self.assertEqual(hours[7].map_name, "mountain_spring")
        self.assertEqual((hours[7].x, hours[7].y), (137, 10))
        self.assertEqual(hours[8].map_name, "mountain_spring")
        self.assertEqual((hours[8].x, hours[8].y), (326, 409))
        self.assertEqual(hours[9].map_name, "path")
        self.assertEqual(hours[10].map_name, "farm")
        self.assertIsNone(timeline.lunch_mark())

    def test_live_bench_locks_frames_lunch_and_delta(self) -> None:
        self.assertEqual(BERRY_SHIP_BENCH["frames"], 3154)
        self.assertEqual(BERRY_SHIP_BENCH["previous_frames"], 3224)
        self.assertEqual(BERRY_SHIP_BENCH["end_clock"], "10:10")
        self.assertEqual(BERRY_SHIP_BENCH["lunch_clock"], "12:00")
        self.assertEqual(BERRY_SHIP_BENCH["lunch_pixel"], (135, 456))
        delta = compare_frame_benches(
            BERRY_SHIP_BENCH["previous_frames"], BERRY_SHIP_BENCH["frames"]
        )
        self.assertTrue(delta["faster"])
        self.assertEqual(delta["delta_frames"], -70)
        hours = BERRY_SHIP_BENCH["hour_locations"]
        self.assertEqual(hours[0], (6, "house", 128, 200))
        self.assertEqual(hours[-1], (12, "farm", 135, 456))

    def test_lunch_mark_is_first_sample_at_or_after_noon(self) -> None:
        rows = list(_BERRY_HOUR_SAMPLES) + [
            _row(4600, 11, 50, 0x00, 135, 456, map="farm", stamina=80, phase="wait_lunch"),
            _row(4680, 12, 0, 0x00, 135, 456, map="farm", stamina=100, phase="lunch"),
        ]
        timeline = ClockTimeline.from_samples(rows)
        lunch = timeline.lunch_mark()
        self.assertIsNotNone(lunch)
        self.assertEqual(lunch.clock, LUNCH_TIME)
        self.assertEqual(lunch.map_name, "farm")
        self.assertEqual((lunch.x, lunch.y), (135, 456))
        self.assertEqual(lunch.stamina, 100)
        self.assertEqual(lunch.phase, "lunch")

    def test_minute_marks_are_clock_resolution(self) -> None:
        timeline = ClockTimeline.from_samples(_BERRY_HOUR_SAMPLES)
        minutes = [(m.clock.hour, m.clock.minute) for m in timeline.minute_marks()]
        self.assertEqual(minutes[0], (6, 8))
        self.assertIn((7, 12), minutes)
        self.assertIn((10, 11), minutes)
        self.assertEqual(len(minutes), len(set(minutes)))

    def test_path_waste_counts_wall_hug_and_turns(self) -> None:
        # 50f pinned on a corner, then a 3-turn jog.
        samples = [
            _row(0, 6, 8, 0x00, 40, 424),
            _row(50, 6, 8, 0x00, 40, 424),
            _row(60, 6, 8, 0x00, 80, 424),
            _row(70, 6, 8, 0x00, 80, 456),
            _row(80, 6, 8, 0x00, 136, 456),
        ]
        waste = path_waste(samples, stasis_min_frames=45)
        self.assertEqual(waste["stasis_frames"], 50)
        self.assertEqual(len(waste["stasis_windows"]), 1)
        self.assertEqual(waste["stasis_windows"][0]["pixel"], [40, 424])
        self.assertEqual(waste["turns"], 2)
        self.assertEqual(waste["moves"], 3)

    def test_faster_bench_is_negative_delta(self) -> None:
        delta = compare_frame_benches(3224, 3000)
        self.assertTrue(delta["faster"])
        self.assertEqual(delta["delta_frames"], -224)
        self.assertEqual(delta["before"]["frames"], BERRY_SHIP_BENCH["previous_frames"])


class PlannerClockTests(unittest.TestCase):
    def test_world_probe_clock_and_pixel(self) -> None:
        world = make_date_world(0x00, season=0, day=2, hour=6, minute=8)
        set_player_pos(world.ram, 136, 424)
        probe = WorldProbe(ram=world.ram)
        self.assertEqual(str(probe.clock()), "06:08")
        self.assertEqual(probe.player_pixel(), (136, 424))

    def test_planning_facts_expose_lunch_window(self) -> None:
        morning = make_date_world(0x15, season=0, day=2, hour=6, minute=8)
        set_player_pos(morning.ram, 128, 200)
        decision = build_day_plan_decision(ram=morning.ram)
        self.assertEqual(decision.facts.hour, 6)
        self.assertEqual(decision.facts.minute, 8)
        self.assertEqual(decision.facts.lunch_hour, 12)
        self.assertFalse(decision.facts.lunch_reached)
        self.assertEqual(decision.facts.minutes_to_lunch, 352)
        self.assertEqual(decision.facts.player_x, 128)
        self.assertEqual(decision.facts.player_y, 200)
        phases = build_day_phases(None, weekday=1, hour=6, season=0, day=2)
        self.assertIn("MOUNTAIN_BERRY", [phase.phase for phase in phases])
        notes = _planning_notes(decision.facts, phases)
        self.assertTrue(
            any(note.startswith("mountain berry before lunch") for note in notes)
        )
        payload = decision.to_jsonable()
        self.assertEqual(payload["facts"]["clock"], "06:08")
        self.assertEqual(payload["facts"]["minutes_to_lunch"], 352)

        noon = make_date_world(0x00, season=0, day=2, hour=12, minute=0)
        noon_facts = build_day_plan_decision(ram=noon.ram).facts
        self.assertTrue(noon_facts.lunch_reached)
        self.assertEqual(noon_facts.minutes_to_lunch, 0)

    def test_policy_lunch_clock_is_have_lunch(self) -> None:
        policy = DayPlannerPolicy()
        self.assertEqual(policy.lunch_clock(), LUNCH_TIME)


class MarkFromMappingTests(unittest.TestCase):
    def test_missing_map_name_uses_tilemap_registry(self) -> None:
        mark = mark_from_mapping(_row(10, 8, 0, 0x10, 326, 409))
        self.assertEqual(mark.map_name, "mountain_spring")
        self.assertEqual(mark.tile, (20, 25))


if __name__ == "__main__":
    unittest.main()
