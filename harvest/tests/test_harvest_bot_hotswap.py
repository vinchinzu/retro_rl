from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harvest.runtime.harvest_bot import AutoClearBot, PlaySession, ADDR_INPUT_LOCK
from harvest.runtime.rom_tools import parse_save_state, resolve_state_path
from harvest.tasks.crop_planter import ADDR_TOOL_BACKPACK, CropWaterTask, SEED_ITEM
from harvest.planner.day_plan import ADDR_DAY, ADDR_HOUR, ADDR_SEASON
from harvest.tasks.farm_clearer import ADDR_MAP, ADDR_TILEMAP, ADDR_TOOL, ADDR_X, ADDR_Y, MAP_WIDTH, Tool
from retro_harness import TaskResult, TaskStatus
from harvest.core.tile_catalog import FRESH_TILLED


class _DummyBot:
    def __init__(self, first_phase_kind: str):
        self.day_plan_enabled = True
        self.day_plan_started = False
        self.day_plan_task = SimpleNamespace(phases=(SimpleNamespace(kind=first_phase_kind),))
        self.enabled = True
        self.disable_reason = None
        self.crop_seed_hack = False

    def get_action(self, game_state, obs):
        action = np.zeros(12, dtype=np.int32)
        action[7] = 1
        return action


class _DummyEnv:
    def __init__(self, input_lock: int):
        self._ram = np.zeros(0x20000, dtype=np.uint8)
        self._ram[ADDR_INPUT_LOCK] = input_lock

    def get_ram(self):
        return self._ram


class _HudEnv:
    def __init__(self, ram):
        self._ram = ram

    def get_ram(self):
        return self._ram


def _set_pos(ram, x: int, y: int) -> None:
    ram[ADDR_X] = x & 0xFF
    ram[ADDR_X + 1] = x >> 8
    ram[ADDR_Y] = y & 0xFF
    ram[ADDR_Y + 1] = y >> 8


class _SuccessfulDayPlan:
    phase_text = "DONE"
    progress_text = "phase=1/1 step=1"

    def __init__(self, *, advance_day: bool) -> None:
        self.advance_day = advance_day
        self._phases = []

    @property
    def phases(self):
        return tuple(self._phases)

    @property
    def current_task(self):
        return None

    def can_start(self, world) -> bool:
        return True

    def reset(self, world) -> None:
        return None

    def step(self, world) -> TaskResult:
        if self.advance_day:
            world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
        return TaskResult(status=TaskStatus.SUCCESS, reason="done")


class _WatchdogBot:
    def __init__(self) -> None:
        self.enabled = True
        self.disable_reason = None
        self.day_plan_enabled = True
        self.crop_enabled = False
        self.grass_enabled = False
        self.day_plan_task = SimpleNamespace(
            current_task=SimpleNamespace(_phase="approach", _target_tile=(6, 38)),
        )
        self.force_end_calls = []

    def force_end_day(self, reason, world):
        self.force_end_calls.append((reason, world.frame))
        return True

    def disable(self, reason):
        self.enabled = False
        self.disable_reason = reason


class HotswapWarmupTests(unittest.TestCase):
    def test_recorded_transition_hotswap_clears_lock_then_runs_bot(self) -> None:
        session = PlaySession(state="pinned_fixture", bot=_DummyBot("recorded_transition"))
        session._start_hotswap_cancel()

        self.assertTrue(session.hotswap_cancel_until_clear)
        self.assertEqual(session.hotswap_cancel_frames, 0)

        locked_action = session._bot_mode_action(_DummyEnv(0), None, None)
        self.assertEqual(int(locked_action[0] + locked_action[8]), 1)

        unlocked_action = session._bot_mode_action(_DummyEnv(1), None, None)
        self.assertFalse(session.hotswap_cancel_until_clear)
        self.assertEqual(int(unlocked_action[7]), 1)

    def test_standard_hotswap_keeps_b_warmup_after_unlock(self) -> None:
        session = PlaySession(state="pinned_fixture", bot=_DummyBot("exit"))
        session._start_hotswap_cancel()

        self.assertTrue(session.hotswap_cancel_until_clear)
        self.assertEqual(session.hotswap_cancel_frames, 90)

        action = session._bot_mode_action(_DummyEnv(1), None, None)
        self.assertEqual(int(action[0]), 1)
        self.assertFalse(session.hotswap_cancel_until_clear)
        self.assertEqual(session.hotswap_cancel_frames, 89)

    def test_crop_ram_shortcuts_are_off_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            bot = AutoClearBot(crop_enabled=True, day_plan_enabled=True)
        self.assertFalse(bot.crop_seed_hack)

    def test_sync_active_item_selects_watering_can_for_crop_water(self) -> None:
        bot = _DummyBot("exit")
        bot.crop_enabled = True
        bot.crop_task_started = True
        bot.crop_task_done = False
        bot.crop_task = SimpleNamespace(_plot_phase="water", seed_type="potato")
        bot.crop_seed_hack = True
        session = PlaySession(state="pinned_fixture", bot=bot)

        calls = []
        session._set_live_value = lambda env, key, value, addr: calls.append((key, value, addr))

        session._sync_active_item(env=None)
        self.assertEqual(calls, [("item_in_hand", int(Tool.WATERING_CAN), 0x4921)])

    def test_sync_active_item_is_noop_without_crop_ram_shortcuts(self) -> None:
        bot = _DummyBot("exit")
        bot.crop_enabled = True
        bot.crop_task_started = True
        bot.crop_task_done = False
        bot.crop_task = SimpleNamespace(_plot_phase="plant", seed_type="potato")
        session = PlaySession(state="pinned_fixture", bot=bot)

        calls = []
        session._set_live_value = lambda env, key, value, addr: calls.append((key, value, addr))

        session._sync_active_item(env=None)
        self.assertEqual(calls, [])

    def test_sync_active_item_does_not_select_seed_for_crop_plant(self) -> None:
        bot = _DummyBot("exit")
        bot.crop_enabled = True
        bot.crop_task_started = True
        bot.crop_task_done = False
        bot.crop_task = SimpleNamespace(_plot_phase="plant", seed_type="potato")
        bot.crop_seed_hack = True
        session = PlaySession(state="pinned_fixture", bot=bot)

        calls = []
        session._set_live_value = lambda env, key, value, addr: calls.append((key, value, addr))

        session._sync_active_item(env=None)
        self.assertEqual(calls, [])

    def test_day_plan_seed_stock_does_not_force_seed_selection(self) -> None:
        bot = AutoClearBot(day_plan_enabled=True)
        bot.day_plan_started = True
        bot.day_plan_done = False
        bot.crop_seed_hack = False
        crop_task = CropWaterTask(seed_type="potato")
        crop_task._plot_phase = "plant"
        bot.day_plan_task._current_task = crop_task
        session = PlaySession(state="pinned_fixture", bot=bot)
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[0x092A] = 4
        env = _HudEnv(ram)

        calls = []
        session._set_live_value = lambda env, key, value, addr: calls.append((key, value, addr))

        session._sync_active_item(env=env)

        self.assertEqual(calls, [])

    def test_day_plan_missing_seed_stock_does_not_force_seed_selection(self) -> None:
        bot = AutoClearBot(day_plan_enabled=True)
        bot.day_plan_started = True
        bot.day_plan_done = False
        bot.crop_seed_hack = False
        crop_task = CropWaterTask(seed_type="potato")
        crop_task._plot_phase = "plant"
        bot.day_plan_task._current_task = crop_task
        session = PlaySession(state="pinned_fixture", bot=bot)
        env = _HudEnv(np.zeros(0x20000, dtype=np.uint8))

        calls = []
        session._set_live_value = lambda env, key, value, addr: calls.append((key, value, addr))

        session._sync_active_item(env=env)

        self.assertEqual(calls, [])

    def test_bot_day_plan_seed_sync_does_not_edit_inventory_with_stock(self) -> None:
        bot = AutoClearBot(day_plan_enabled=True)
        crop_task = CropWaterTask(seed_type="potato")
        crop_task._plot_phase = "plant"
        bot.day_plan_task._current_task = crop_task
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[0x092A] = 4
        bot.env = _HudEnv(ram)

        calls = []

        class Editor:
            def __init__(self, env) -> None:
                self.env = env

            def set_field(self, key, value) -> None:
                calls.append((key, value))

        with patch("harvest.runtime.harvest_bot.LiveRamEditor", Editor):
            bot._sync_day_plan_seed_item(ram)

        self.assertEqual(calls, [])

    def test_bot_day_plan_seed_sync_does_not_create_seed_stock(self) -> None:
        bot = AutoClearBot(day_plan_enabled=True)
        crop_task = CropWaterTask(seed_type="potato")
        crop_task._plot_phase = "plant"
        bot.day_plan_task._current_task = crop_task
        ram = np.zeros(0x20000, dtype=np.uint8)
        bot.env = _HudEnv(ram)

        calls = []

        class Editor:
            def __init__(self, env) -> None:
                self.env = env

            def set_field(self, key, value) -> None:
                calls.append((key, value))

        with patch("harvest.runtime.harvest_bot.LiveRamEditor", Editor):
            bot._sync_day_plan_seed_item(ram)

        self.assertEqual(calls, [])

    def test_bot_seed_sync_preserves_ready_carry_pair(self) -> None:
        bot = AutoClearBot(day_plan_enabled=True)
        crop_task = CropWaterTask(seed_type="potato")
        crop_task._plot_phase = "plant"
        bot.day_plan_task._current_task = crop_task
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[0x092A] = 4
        ram[ADDR_TOOL] = SEED_ITEM["potato"]
        ram[ADDR_TOOL_BACKPACK] = int(Tool.WATERING_CAN)
        bot.env = _HudEnv(ram)

        calls = []

        class Editor:
            def __init__(self, env) -> None:
                self.env = env

            def set_field(self, key, value) -> None:
                calls.append((key, value))

        with patch("harvest.runtime.harvest_bot.LiveRamEditor", Editor):
            bot._sync_day_plan_seed_item(ram)

        self.assertEqual(calls, [])

    def test_hud_shows_location_and_unwatered_crop_count_on_farm(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_MAP + 25 * MAP_WIDTH + 18] = 0x54
        bot = AutoClearBot(day_plan_enabled=False)
        session = PlaySession(state="pinned_fixture", bot=bot)
        game_state = SimpleNamespace(
            date_str="Y1 Spring 14 (Sun)",
            time_str="6:00 AM",
            money=0,
            item_name="Empty",
        )

        lines = session._build_hud_lines(_HudEnv(ram), game_state, np.zeros(12, dtype=np.int32))

        self.assertIn("Loc farm (0x00)", lines)
        self.assertIn("Unwatered 1", lines)

    def test_hud_does_not_show_stale_crop_counts_off_farm(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x0C
        ram[ADDR_MAP + 25 * MAP_WIDTH + 18] = FRESH_TILLED
        bot = AutoClearBot(day_plan_enabled=False)
        session = PlaySession(state="pinned_fixture", bot=bot)
        game_state = SimpleNamespace(
            date_str="Y1 Spring 14 (Sun)",
            time_str="6:00 AM",
            money=0,
            item_name="Empty",
        )

        lines = session._build_hud_lines(_HudEnv(ram), game_state, np.zeros(12, dtype=np.int32))

        self.assertIn("Loc path (0x0C)", lines)
        self.assertIn("Ready --", lines)
        self.assertIn("Unwatered --", lines)

    def test_auto_day_plan_rebuilds_after_sleep_advances_day(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x15
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_SEASON + 0x4000] = 0
        ram[ADDR_DAY + 0x4000] = 14
        _set_pos(ram, 49, 86)
        bot = AutoClearBot(day_plan_enabled=True)
        bot.env = _HudEnv(ram)
        bot.enabled = True
        bot.day_plan_task = _SuccessfulDayPlan(advance_day=True)

        action = bot.get_action(SimpleNamespace(), np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertFalse(action.any())
        self.assertTrue(bot.enabled)
        self.assertFalse(bot.day_plan_done)
        self.assertFalse(bot.day_plan_started)
        self.assertTrue(bot._pending_auto_day_plan_rebuild)
        self.assertIsNone(bot.disable_reason)
        self.assertEqual(int(ram[ADDR_DAY + 0x4000]), 15)

        ram[ADDR_HOUR + 0x4000] = 6
        _set_pos(ram, 136, 120)
        for _ in range(10):
            action = bot.get_action(SimpleNamespace(), np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertFalse(action.any())
        self.assertTrue(bot.enabled)
        self.assertFalse(bot._pending_auto_day_plan_rebuild)
        self.assertFalse(bot.day_plan_done)
        self.assertFalse(bot.day_plan_started)

    def test_auto_day_plan_rebuild_waits_for_normal_scene(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x15
        ram[ADDR_INPUT_LOCK] = 0
        ram[ADDR_HOUR + 0x4000] = 6
        _set_pos(ram, 136, 120)
        bot = AutoClearBot(day_plan_enabled=True)

        self.assertFalse(bot._auto_day_plan_rebuild_ready(ram))

        ram[ADDR_INPUT_LOCK] = 1
        _set_pos(ram, 0, 0)
        self.assertFalse(bot._auto_day_plan_rebuild_ready(ram))

        _set_pos(ram, 136, 120)
        for _ in range(9):
            self.assertFalse(bot._auto_day_plan_rebuild_ready(ram))
        self.assertTrue(bot._auto_day_plan_rebuild_ready(ram))

    def test_auto_day_plan_rebuild_accepts_pinned_morning_after_sleep_fixture(self) -> None:
        state_path = resolve_state_path("Y1_After_Sleep")
        ram = parse_save_state(state_path).ram
        bot = AutoClearBot(day_plan_enabled=True)

        for _ in range(9):
            self.assertFalse(bot._auto_day_plan_rebuild_ready(ram))
        self.assertTrue(bot._auto_day_plan_rebuild_ready(ram))

    def test_hotswap_to_bot_rebuilds_auto_day_plan_when_dry_crops_remain(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_SEASON + 0x4000] = 0
        ram[ADDR_DAY + 0x4000] = 29
        ram[ADDR_HOUR + 0x4000] = 18
        ram[ADDR_MAP + 34 * MAP_WIDTH + 7] = 0x5A
        bot = AutoClearBot(day_plan_enabled=True)
        bot.env = _HudEnv(ram)
        bot.day_plan_started = True
        bot.day_plan_done = False
        bot.day_plan_task = SimpleNamespace(
            phase_text="RETURN_HOME",
            progress_text="phase=4/5 step=100",
            _phases=[],
            resume_after_hotswap=lambda world: None,
        )

        bot.prepare_for_enable()

        names = [phase.phase for phase in bot.day_plan_task.phases]
        self.assertFalse(bot.day_plan_started)
        self.assertFalse(bot.day_plan_done)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("CROP_WATER"), names.index("RETURN_HOME"))

    def test_hotswap_to_bot_resumes_active_day_plan_when_no_dry_crops_remain(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        bot = AutoClearBot(day_plan_enabled=True)
        bot.env = _HudEnv(ram)
        bot.day_plan_started = True
        bot.day_plan_done = False
        calls = []
        bot.day_plan_task = SimpleNamespace(
            phase_text="RETURN_HOME",
            progress_text="phase=4/5 step=100",
            _phases=[],
            resume_after_hotswap=lambda world: calls.append(world.ram),
        )

        bot.prepare_for_enable()

        self.assertEqual(calls, [ram])
        self.assertTrue(bot.day_plan_started)
        self.assertFalse(bot.day_plan_done)

    def test_hotswap_to_bot_resumes_active_crop_mode_task(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        bot = AutoClearBot(crop_enabled=True, day_plan_enabled=False)
        bot.env = _HudEnv(ram)
        bot.crop_task_started = True
        bot.crop_task_done = False
        calls = []
        bot.crop_task = SimpleNamespace(
            resume_after_hotswap=lambda world: calls.append(world.ram),
        )

        bot.prepare_for_enable()

        self.assertEqual(calls, [ram])

    def test_auto_day_plan_still_stops_when_same_day_plan_completes(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_SEASON + 0x4000] = 0
        ram[ADDR_DAY + 0x4000] = 14
        bot = AutoClearBot(day_plan_enabled=True)
        bot.env = _HudEnv(ram)
        bot.enabled = True
        bot.day_plan_task = _SuccessfulDayPlan(advance_day=False)

        action = bot.get_action(SimpleNamespace(), np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertFalse(action.any())
        self.assertFalse(bot.enabled)
        self.assertTrue(bot.day_plan_done)
        self.assertIn("Day plan complete", bot.disable_reason or "")

    def test_days_mode_does_not_auto_rebuild_after_target_completion(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x15
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_SEASON + 0x4000] = 0
        ram[ADDR_DAY + 0x4000] = 14
        bot = AutoClearBot(day_plan_enabled=True, multi_day_count=1)
        bot.env = _HudEnv(ram)
        bot.enabled = True
        bot.day_plan_task = _SuccessfulDayPlan(advance_day=True)

        action = bot.get_action(SimpleNamespace(), np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertFalse(action.any())
        self.assertFalse(bot.enabled)
        self.assertTrue(bot.day_plan_done)
        self.assertFalse(bot._pending_auto_day_plan_rebuild)
        self.assertIn("Day plan complete", bot.disable_reason or "")

    def test_force_end_day_delegates_to_multi_day_planner(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        world = SimpleNamespace(frame=1, ram=ram, info={}, obs=None)
        calls = []

        class Planner:
            def force_end_day(self, world, reason):
                calls.append((world.ram, reason))
                return True

        bot = AutoClearBot(day_plan_enabled=True, multi_day_count=1)
        bot.day_plan_task = Planner()
        bot.day_plan_started = True
        bot.day_plan_done = True
        bot.enabled = False
        bot.disable_reason = "old failure"

        self.assertTrue(bot.force_end_day("stall", world))
        self.assertEqual(calls, [(ram, "stall")])
        self.assertTrue(bot.enabled)
        self.assertFalse(bot.day_plan_done)
        self.assertIsNone(bot.disable_reason)

    def test_watchdog_signature_ignores_clock_and_small_position_changes(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_SEASON + 0x4000] = 1
        ram[ADDR_DAY + 0x4000] = 15
        ram[ADDR_HOUR + 0x4000] = 12
        _set_pos(ram, 100, 100)
        bot = _WatchdogBot()
        session = PlaySession(state="pinned_fixture", bot=bot, autoplay=True)
        env = _HudEnv(ram)
        game_state = SimpleNamespace(season=1)

        first = session._progress_signature(env, game_state)
        ram[ADDR_HOUR + 0x4000] = 16
        _set_pos(ram, 103, 97)
        second = session._progress_signature(env, game_state)

        self.assertEqual(first, second)

    def test_watchdog_catches_small_oscillation_without_semantic_progress(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_SEASON + 0x4000] = 1
        ram[ADDR_DAY + 0x4000] = 15
        _set_pos(ram, 100, 100)
        bot = _WatchdogBot()
        session = PlaySession(state="pinned_fixture", bot=bot, autoplay=True, watchdog_frames=3)
        env = _HudEnv(ram)
        game_state = SimpleNamespace(season=1, date_str="Y1 Summer 15", time_str="12:00 PM")
        action = np.zeros(12, dtype=np.int32)
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        captures = []
        session._write_diagnostic_artifacts = (
            lambda env, game_state, action, obs, event, reason: captures.append((event, reason))
        )

        for frame, x in [(1, 100), (2, 101), (3, 99), (4, 100)]:
            session.frame_count = frame
            _set_pos(ram, x, 100)
            self.assertFalse(session._check_autoplay_watchdog(env, game_state, action, obs))

        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0][0], "stall_watchdog")
        self.assertEqual(bot.force_end_calls, [("no progress for 3 frames", 4)])

    def test_watchdog_treats_large_position_change_as_progress(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_SEASON + 0x4000] = 1
        ram[ADDR_DAY + 0x4000] = 15
        _set_pos(ram, 100, 100)
        bot = _WatchdogBot()
        session = PlaySession(state="pinned_fixture", bot=bot, autoplay=True, watchdog_frames=3)
        env = _HudEnv(ram)
        game_state = SimpleNamespace(season=1, date_str="Y1 Summer 15", time_str="12:00 PM")
        action = np.zeros(12, dtype=np.int32)
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        captures = []
        session._write_diagnostic_artifacts = (
            lambda env, game_state, action, obs, event, reason: captures.append((event, reason))
        )

        for frame, x in [(1, 100), (2, 180), (3, 181), (4, 180)]:
            session.frame_count = frame
            _set_pos(ram, x, 100)
            self.assertFalse(session._check_autoplay_watchdog(env, game_state, action, obs))

        self.assertEqual(captures, [])
        self.assertEqual(bot.force_end_calls, [])


if __name__ == "__main__":
    unittest.main()
