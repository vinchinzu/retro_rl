"""Unit tests for nav-to-door seed shop buy and false CrossMap returns."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import field_spec
from harvest.maps.map_config import ROUTES, SEGMENTS, find_landmark
from harvest.planner.day_phase_catalog import BUY_SEEDS_PHASE, buy_seeds_phase
from harvest.planner.day_phase_types import PhaseKind
from harvest.planner.tasks.navigation import CrossMapRecordedTask
from harvest.tasks.buy_seeds import (
    SEED_CLERK_PX,
    BuySeedsTask,
    first_shop_nav_segment,
    first_shop_return_segment,
    purchase_closed,
    shop_door_px,
    town_coords_settled,
)
from retro_harness import TaskStatus

from day_plan_test_helpers import make_transition_world, make_world, set_money, set_player_pos

ADDR_POTATO = field_spec("potato_seeds").address


def _set_stock(ram, count: int) -> None:
    # 0x20000 fixtures have no live WRAM mirror; field reads use raw offsets.
    ram[ADDR_POTATO] = count


class ShopRouteTests(unittest.TestCase):
    def test_shop_door_route_uses_landmark_open_face(self) -> None:
        found = find_landmark("shop_door", tilemap_id=0x04)
        self.assertIsNotNone(found)
        _tm, landmark = found
        self.assertEqual(landmark.tile, (37, 13))
        self.assertEqual(landmark.face, "up")
        self.assertEqual(shop_door_px(), landmark.target_px)
        approach = SEGMENTS["town_to_shop_door"][-1]
        # Plaza south of the door (37,17). Doorframe (37,14) is a hug.
        self.assertEqual(approach.target_px, (602, 274))
        self.assertTrue(approach.is_exit)
        self.assertEqual(approach.exit_direction, "up")
        self.assertNotEqual(approach.target_px, (37 * 16 + 8, 14 * 16 + 8))

    def test_seed_counter_matches_d2_clerk_stand(self) -> None:
        found = find_landmark("seed_counter", tilemap_id=0x1C)
        self.assertIsNotNone(found)
        _tm, landmark = found
        self.assertEqual(landmark.tile, (11, 21))
        self.assertEqual(SEGMENTS["shop_to_counter"][-1].target_px, SEED_CLERK_PX)
        self.assertEqual(SEED_CLERK_PX, (182, 342))

    def test_shop_exit_stands_on_open_face_not_doorframe(self) -> None:
        exit_wp = SEGMENTS["shop_to_town"][-1]
        self.assertEqual(exit_wp.target_px, (8 * 16 + 8, 28 * 16 + 8))
        self.assertNotEqual(exit_wp.target_px, (8 * 16 + 8, 29 * 16 + 8))
        self.assertEqual(exit_wp.exit_direction, "down")

    def test_farm_to_shop_door_composes_named_hops(self) -> None:
        composed = (
            list(SEGMENTS["farm_to_path"])
            + list(SEGMENTS["path_to_town_shop"])
            + list(SEGMENTS["town_to_shop_door"])
        )
        self.assertEqual(composed, ROUTES["farm_to_shop_door"])

    def test_path_leak_town_pixels_are_not_settled(self) -> None:
        world = make_transition_world(0x04, current_tile=(0, 8))
        set_player_pos(world.ram, 10, 134)
        self.assertFalse(town_coords_settled(world.ram))
        set_player_pos(world.ram, 756, 422)
        self.assertTrue(town_coords_settled(world.ram))

    def test_segment_choice_follows_live_tilemap(self) -> None:
        self.assertEqual(first_shop_nav_segment(0x00), "farm_to_path")
        self.assertEqual(first_shop_nav_segment(0x0C), "path_to_town_shop")
        self.assertEqual(first_shop_nav_segment(0x04), "town_to_shop_door")
        self.assertIsNone(first_shop_nav_segment(0x1C))
        self.assertIsNone(first_shop_nav_segment(0x15))
        self.assertEqual(first_shop_return_segment(0x1C), "shop_to_town")
        self.assertEqual(first_shop_return_segment(0x04), "town_shop_to_path")
        self.assertEqual(first_shop_return_segment(0x0C), "path_to_farm")
        self.assertEqual(SEGMENTS["path_to_farm"][0].target_px, (132, 128))
        self.assertEqual(SEGMENTS["path_to_farm"][-1].exit_direction, "right")


class PurchaseCloseTests(unittest.TestCase):
    def test_purchase_requires_stock_up_and_money_down(self) -> None:
        self.assertTrue(purchase_closed(stock_before=0, stock_after=1, money_before=300, money_after=100))
        self.assertFalse(purchase_closed(stock_before=0, stock_after=0, money_before=300, money_after=300))
        self.assertFalse(purchase_closed(stock_before=0, stock_after=1, money_before=300, money_after=300))
        self.assertFalse(purchase_closed(stock_before=0, stock_after=0, money_before=300, money_after=100))

    def test_bought_on_farm_succeeds(self) -> None:
        world = make_transition_world(0x00, current_tile=(2, 26))
        _set_stock(world.ram, 0)
        set_money(world.ram, 300, live_offset=False)
        task = BuySeedsTask()
        task.reset(world)
        _set_stock(world.ram, 1)
        set_money(world.ram, 100, live_offset=False)
        task._seen_shop = True
        task._bought = True
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0->1", result.reason or "")

    def test_spring_buy_phase_is_shop_buy(self) -> None:
        self.assertEqual(BUY_SEEDS_PHASE.kind, PhaseKind.SHOP_BUY)
        spring = buy_seeds_phase(recording_name="buy_potato_seeds_d2")
        self.assertEqual(spring.kind, PhaseKind.SHOP_BUY)
        summer = buy_seeds_phase(recording_name="buy_summer")
        self.assertEqual(summer.kind, "cross_map")
        self.assertEqual(summer.params["recording_start"], 0)

    def test_day_plan_factory_builds_buy_seeds_task(self) -> None:
        """rr-zmss: D2 BUY_SEEDS must be nav+RAM, not CrossMap origin-return."""
        from harvest.planner.day_task_factory import DayTaskFactory

        world = make_world(0x00)
        task = DayTaskFactory().make_task(BUY_SEEDS_PHASE, world)
        self.assertIsInstance(task, BuySeedsTask)
        self.assertNotIsInstance(task, CrossMapRecordedTask)


class CrossMapShopRejectTests(unittest.TestCase):
    def test_origin_return_without_shop_is_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buy_potato_seeds.json"
            path.write_text(json.dumps({"name": "buy_potato_seeds", "frames": [[0] * 12] * 4}))
            world = make_world(0x00)
            task = CrossMapRecordedTask(
                recording_name="buy_potato_seeds",
                tasks_dir=tmpdir,
                origin_tilemap=0x00,
                min_replay_before_return=0,
                continue_after_return=0,
                stock_field="potato_seeds",
                require_purchase=True,
            )
            task.reset(world)
            # Walk off farm.
            world.ram[field_spec("tilemap").address] = 0x0C
            off = task.step(world)
            self.assertEqual(off.status, TaskStatus.RUNNING)
            # Come back without 0x1C or stock delta. First farm step plays a
            # tape frame; the next close-check rejects the false return.
            world.ram[field_spec("tilemap").address] = 0x00
            playing = task.step(world)
            self.assertEqual(playing.status, TaskStatus.RUNNING)
            done = task.step(world)
            self.assertEqual(done.status, TaskStatus.FAILURE)
            self.assertIn("shop miss", done.reason or "")


if __name__ == "__main__":
    unittest.main()
