from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_X, ADDR_Y, MAP_WIDTH
from harvest.tasks.harvest_task import (
    ACTION_CARRYING_BIT,
    ADDR_PLAYER_STATE,
    ADDR_SHIPPING_MONEY,
    HarvestStep,
    HarvestTask,
    build_harvest_steps,
    crop_nav_target_px,
    is_ripe_crop_tile,
    live_harvestable_crop_tiles,
    state_harvestable_crop_tiles,
)
from retro_harness import TaskStatus


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


class _FakeDocument:
    def __init__(self, tiles):
        self._tiles = tiles

    def farm_tile(self, x: int, y: int):
        persistent, visible = self._tiles.get((x, y), (0x00, 0x00))
        return SimpleNamespace(persistent_value=persistent, visible_value=visible)


class _FakeDocumentWithRam(_FakeDocument):
    def __init__(self, tiles, ram):
        super().__init__(tiles)
        self._ram = ram

    def ram_array(self):
        return self._ram


class HarvestTaskTests(unittest.TestCase):
    def test_state_harvestable_crop_tiles_uses_exact_mature_ids(self) -> None:
        fake_doc = _FakeDocument(
            {
                (3, 34): (88, 160),
                (4, 34): (92, 160),
                (5, 34): (96, 160),
                (6, 34): (96, 161),
                (7, 34): (96, 7),  # already harvested; visible tile should exclude it
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(state_harvestable_crop_tiles("fake"), [(5, 34), (6, 34)])

    def test_state_harvestable_crop_tiles_rejects_mid_growth_potatoes(self) -> None:
        fake_doc = _FakeDocument(
            {
                (10, 34): (0x5E, 0xA0),
                (11, 34): (0x5E, 0xA0),
                (12, 34): (0x5E, 0xA1),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(state_harvestable_crop_tiles("fake"), [])

    def test_state_harvestable_crop_tiles_falls_back_to_live_map_when_document_has_no_targets(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 12, 28, 0x66)
        _set_tile(ram, 13, 28, 0x60)
        fake_doc = _FakeDocumentWithRam({}, ram)

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(state_harvestable_crop_tiles("fake"), [(13, 28)])

    def test_live_harvestable_crop_tiles_ignores_stale_state_tiles_when_live_map_is_not_ready(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 5, 34, 0x5E)
        _set_tile(ram, 6, 34, 0x5E)
        fake_doc = _FakeDocument(
            {
                (5, 34): (96, 160),
                (6, 34): (96, 161),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(live_harvestable_crop_tiles(ram, "fake"), [])

    def test_live_harvestable_crop_tiles_falls_back_to_live_map_when_document_has_no_targets(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 12, 28, 0x66)
        _set_tile(ram, 13, 28, 0x60)
        fake_doc = _FakeDocument({})

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(live_harvestable_crop_tiles(ram, "fake"), [(13, 28)])

    def test_live_harvestable_crop_tiles_accepts_loaded_mature_crop_stage(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 5, 34, 0x60)
        _set_tile(ram, 6, 34, 0x61)
        _set_tile(ram, 7, 34, 0xA0)
        fake_doc = _FakeDocument(
            {
                (5, 34): (96, 160),
                (6, 34): (96, 161),
                (7, 34): (96, 160),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(live_harvestable_crop_tiles(ram, "fake"), [(5, 34), (6, 34)])

    def test_live_harvestable_crop_tiles_requires_ready_visible_state(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 5, 34, 0x60)
        _set_tile(ram, 6, 34, 0x60)
        fake_doc = _FakeDocument(
            {
                (5, 34): (96, 0x60),
                (6, 34): (96, 0xA0),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(live_harvestable_crop_tiles(ram, "fake"), [(6, 34)])

    def test_live_harvestable_crop_tiles_default_bounds_include_left_edge_column(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 2, 34, 0x60)
        _set_tile(ram, 3, 34, 0x60)
        fake_doc = _FakeDocument(
            {
                (2, 34): (96, 0xA0),
                (3, 34): (96, 0xA0),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(live_harvestable_crop_tiles(ram, "fake"), [(2, 34), (3, 34)])

    def test_live_harvestable_crop_tiles_without_state_uses_exact_mature_ids(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 5, 34, 0x58)
        _set_tile(ram, 6, 34, 0x5C)
        _set_tile(ram, 7, 34, 0x60)
        _set_tile(ram, 8, 34, 0x68)
        _set_tile(ram, 9, 34, 0xA6)

        self.assertEqual(live_harvestable_crop_tiles(ram, None), [(7, 34)])

    def test_live_harvestable_crop_tiles_without_state_requires_ripe_stage(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 10, 34, 0x5E)
        _set_tile(ram, 11, 34, 0x5F)
        _set_tile(ram, 12, 34, 0x68)

        self.assertFalse(is_ripe_crop_tile(0x5E))
        self.assertFalse(is_ripe_crop_tile(0x5F))
        self.assertFalse(is_ripe_crop_tile(0x68))
        self.assertEqual(live_harvestable_crop_tiles(ram, None), [])

    def test_build_harvest_steps_prefers_shipping_side_adjacent_stands(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 8, 33, 0xA1)
        _set_tile(ram, 8, 34, 0xA0)
        _set_tile(ram, 8, 35, 0xA0)

        steps = build_harvest_steps(ram, [(8, 34), (8, 35)])

        self.assertEqual(steps[0].target, (8, 34))
        self.assertEqual(steps[0].stand, (8, 33))
        self.assertEqual(steps[0].face, "down")

    def test_build_harvest_steps_does_not_stand_on_crop_tiles(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        for ty in range(MAP_WIDTH):
            for tx in range(MAP_WIDTH):
                _set_tile(ram, tx, ty, 0xFF)
        _set_tile(ram, 11, 35, 0x60)
        _set_tile(ram, 12, 35, 0x60)
        _set_tile(ram, 11, 36, 0xA0)

        steps = build_harvest_steps(ram, [(11, 35), (12, 35)])
        by_target = {step.target: step for step in steps}

        self.assertEqual(by_target[(11, 35)].stand, (11, 36))
        self.assertNotEqual(by_target[(11, 35)].stand, (12, 35))

    def test_ship_options_prefer_right_side_bin_stand(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 11, 30, 0x01)
        _set_tile(ram, 8, 32, 0x01)
        x = 12 * 16 + 8
        y = 31 * 16 + 8
        ram[ADDR_X] = x & 0xFF
        ram[ADDR_X + 1] = x >> 8
        ram[ADDR_Y] = y & 0xFF
        ram[ADDR_Y + 1] = y >> 8

        task = HarvestTask()
        task._navigator.update(ram)

        self.assertEqual(task._current_ship_option(ram), ((11, 30), "left"))

    def test_ship_options_prefer_below_bin_stand_when_closer(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 11, 30, 0x01)
        _set_tile(ram, 8, 32, 0x01)
        x = 4 * 16 + 8
        y = 34 * 16 + 8
        ram[ADDR_X] = x & 0xFF
        ram[ADDR_X + 1] = x >> 8
        ram[ADDR_Y] = y & 0xFF
        ram[ADDR_Y + 1] = y >> 8

        task = HarvestTask()
        task._navigator.update(ram)

        self.assertEqual(task._current_ship_option(ram), ((8, 32), "up"))

    def test_harvest_verify_does_not_accept_unchanged_mature_tile(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_tile(ram, 5, 34, 0x60)
        task = HarvestTask(state_name="fake")
        task._phase = "target_verify"
        task._current = HarvestStep(target=(5, 34), stand=(5, 33), face="down")
        task._target_live_before = 0x60
        task._verify_count = 12

        result = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.harvested_count, 0)
        self.assertEqual(task.skipped_count, 1)
        self.assertEqual(task._phase, "select")

    def test_pick_success_clears_stale_nav_blocks_before_shipping(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        task = HarvestTask(state_name="fake")
        task._phase = "target_verify"
        task._current = HarvestStep(target=(7, 34), stand=(7, 33), face="down")
        task._pathfinder.temp_blocked.add((11, 30))
        task._navigator.path = [(7, 33), (8, 33)]
        task._navigator.stasis = 99

        result = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertEqual(task._navigator.path, [])
        self.assertEqual(task._navigator.stasis, 0)
        self.assertEqual(task._pathfinder.temp_blocked, set())

    def test_ship_verify_pulses_a_while_shipping_dialog_is_locked(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        ram[ADDR_INPUT_LOCK] = 0
        task = HarvestTask()
        task._phase = "ship_verify"
        task._step_count = 1

        result = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "dialog")
        self.assertEqual(int(result.action.action[8]), 1)

    def test_ship_verify_waits_for_shipping_dialog_to_clear_before_success(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_SHIPPING_MONEY] = 8
        ram[ADDR_INPUT_LOCK] = 0
        task = HarvestTask()
        task._phase = "ship_verify"
        task._ship_money_before = 0

        locked = task.step(SimpleNamespace(ram=ram, info={}, obs=None))
        ram[ADDR_INPUT_LOCK] = 1
        unlocked = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(locked.status, TaskStatus.RUNNING)
        self.assertEqual(locked.reason, "dialog")
        self.assertEqual(unlocked.status, TaskStatus.RUNNING)
        self.assertEqual(task.shipped_count, 1)
        self.assertEqual(task._phase, "select")

    def test_ship_verify_counts_bin_drop_when_shipping_money_unchanged(self) -> None:
        """Crop no longer carried means successful bin drop; money may settle at 5pm."""
        ram = np.zeros(0x20000, dtype=np.uint8)
        # Not carrying (player_state bit clear), input unlocked, money flat.
        ram[ADDR_INPUT_LOCK] = 1
        task = HarvestTask()
        task._phase = "ship_verify"
        task._ship_money_before = 0
        task.shipped_count = 0

        result = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("crop cleared without shipping money", result.reason or "")
        self.assertEqual(task.shipped_count, 1)
        self.assertEqual(task._phase, "select")

    def test_ship_verify_counts_money_delta_when_shipping_money_increases(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_SHIPPING_MONEY] = 8  # read_shipping_money => 80
        ram[ADDR_INPUT_LOCK] = 1
        task = HarvestTask()
        task._phase = "ship_verify"
        task._ship_money_before = 0
        task.shipped_count = 0

        result = task.step(SimpleNamespace(ram=ram, info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.shipped_count, 1)  # max(1, 80//80)
        self.assertEqual(task._phase, "select")

    def test_harvest_completion_fails_when_any_target_skipped(self) -> None:
        task = HarvestTask()
        task._initial_target_count = 1
        task.skipped_count = 1

        result = task.step(SimpleNamespace(ram=np.zeros(0x20000, dtype=np.uint8), info={}, obs=None))

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("incomplete harvest", result.reason)

    def test_harvest_choose_next_finishes_active_plot_group_first(self) -> None:
        task = HarvestTask()
        task._navigator.current_pos.x = 8 * 16 + 8
        task._navigator.current_pos.y = 32 * 16 + 8
        task._steps = [
            HarvestStep(target=(6, 38), stand=(6, 37), face="down", group=1),
            HarvestStep(target=(18, 24), stand=(18, 23), face="down", group=2),
            HarvestStep(target=(7, 38), stand=(7, 37), face="down", group=1),
        ]

        first = task._choose_next_step()
        second = task._choose_next_step()

        self.assertEqual(first.group, 1)
        self.assertEqual(second.group, 1)

    def test_crop_nav_target_uses_nearest_harvest_stand(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        for ty in range(MAP_WIDTH):
            for tx in range(MAP_WIDTH):
                _set_tile(ram, tx, ty, 0xA0)
        x = 18 * 16 + 8
        y = 24 * 16 + 8
        ram[ADDR_X] = x & 0xFF
        ram[ADDR_X + 1] = x >> 8
        ram[ADDR_Y] = y & 0xFF
        ram[ADDR_Y + 1] = y >> 8
        fake_doc = _FakeDocument(
            {
                (2, 15): (96, 0xA0),
                (4, 17): (96, 0xA0),
            }
        )

        with patch("harvest.tasks.harvest_task.HarvestStateDocument.load", return_value=fake_doc):
            self.assertEqual(crop_nav_target_px(ram, "fake"), (4 * 16 + 8, 18 * 16 + 8))

    def test_crop_nav_target_virgin_field_uses_preferred_plant_anchor(self) -> None:
        """No harvest tiles / plots → land at preferred plant field, not ship area."""
        from harvest.tasks.harvest_task import PREFERRED_PLANT_PX

        ram = np.zeros(0x20000, dtype=np.uint8)
        for ty in range(MAP_WIDTH):
            for tx in range(MAP_WIDTH):
                _set_tile(ram, tx, ty, 0xA0)  # path; no crop/tilled plots

        with patch(
            "harvest.tasks.harvest_task.HarvestStateDocument.load",
            side_effect=FileNotFoundError,
        ):
            self.assertEqual(crop_nav_target_px(ram, None), PREFERRED_PLANT_PX)
            # Explicit shipping-area override still honored when passed.
            self.assertEqual(
                crop_nav_target_px(ram, None, fallback_px=(136, 520)),
                (136, 520),
            )


if __name__ == "__main__":
    unittest.main()
