from __future__ import annotations

import time
import unittest

from harvest.runtime.watch_display import (
    SPEED_LEVELS,
    bot_speed_timing,
    default_speed_index,
    pace_present,
)


class WatchDisplayTimingTests(unittest.TestCase):
    def test_speed_ladder_includes_2x_and_4x(self) -> None:
        self.assertIn(2.0, SPEED_LEVELS)
        self.assertIn(4.0, SPEED_LEVELS)
        self.assertEqual(SPEED_LEVELS[default_speed_index()], 4.0)

    def test_bot_2x_repeats_two_emu_steps_at_60hz(self) -> None:
        repeat, tick, skip = bot_speed_timing(2.0, bot=True)
        self.assertEqual(repeat, 2)
        self.assertEqual(tick, 60)
        self.assertFalse(skip)

    def test_bot_4x_repeats_four_emu_steps_at_60hz(self) -> None:
        repeat, tick, skip = bot_speed_timing(4.0, bot=True)
        self.assertEqual(repeat, 4)
        self.assertEqual(tick, 60)
        self.assertFalse(skip)

    def test_bot_1x_is_one_step_at_60hz(self) -> None:
        repeat, tick, skip = bot_speed_timing(1.0, bot=True)
        self.assertEqual(repeat, 1)
        self.assertEqual(tick, 60)
        self.assertFalse(skip)

    def test_bot_8x_is_unthrottled(self) -> None:
        repeat, tick, skip = bot_speed_timing(8.0, bot=True)
        self.assertEqual(repeat, 1)
        self.assertEqual(tick, 0)
        self.assertTrue(skip)

    def test_tab_turbo_unthrottles_4x(self) -> None:
        repeat, tick, skip = bot_speed_timing(4.0, turbo=True, bot=True)
        self.assertEqual(repeat, 1)
        self.assertEqual(tick, 0)
        self.assertTrue(skip)

    def test_human_4x_does_not_frame_repeat(self) -> None:
        repeat, tick, skip = bot_speed_timing(4.0, bot=False)
        self.assertEqual(repeat, 1)
        self.assertEqual(tick, 240)
        self.assertFalse(skip)

    def test_pace_present_does_not_double_wait_when_behind(self) -> None:
        holder = type("H", (), {})()
        holder._next_present = time.perf_counter() - 1.0
        t0 = time.perf_counter()
        pace_present(60, holder)
        self.assertLess(time.perf_counter() - t0, 0.05)


if __name__ == "__main__":
    unittest.main()
