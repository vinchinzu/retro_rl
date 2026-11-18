"""Unit tests for nav-to-door seed shop buy and false CrossMap returns."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import field_spec, live_wram_base
from harvest.maps.map_config import ROUTES, SEGMENTS, find_landmark
from harvest.planner.day_phase_catalog import (
    BUY_SEEDS_PHASE,
    buy_seeds_phase,
    cow_day_phases,
    tool_purchase_phases,
)
from harvest.planner.livestock_econ import missing_cow_tools, tool_shop_open
from harvest.planner.day_phase_types import PhaseKind
from harvest.planner.tasks.navigation import CrossMapRecordedTask
from harvest.tasks.buy_seeds import (
    SEED_BAG_PRICE,
    SEED_CLERK_PX,
    TOOL_SHELVES,
    TOWN_BOUNCE_RETRIES,
    TOWN_SETTLE_FRAMES,
    TOOL_SHOP,
    BuySeedsTask,
    BuyShelfItemTask,
    at_seed_clerk,
    seed_stand_px,
    first_shop_nav_segment,
    first_shop_return_segment,
    plan_tool_buys,
    purchase_closed,
    shop_door_px,
    tool_owned,
    town_coords_settled,
)
from retro_harness import TaskStatus

from day_plan_test_helpers import (
    make_date_world,
    make_transition_world,
    make_world,
    set_money,
    set_player_pos,
)

ADDR_POTATO = field_spec("potato_seeds").address


def _set_stock(ram, count: int) -> None:
    # 0x20000 fixtures have no live WRAM mirror; field reads use raw offsets.
    ram[ADDR_POTATO] = count


@pytest.mark.rom
def test_d3_seed_buy_from_the_natural_morning_returns_with_a_receipt():
    from retro_harness import AuditCapabilities, AuditedEnv, WorldState
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, NAV_FARM_EXIT_PHASE
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.farm_clear_quota import farm_map_loaded
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.d3_seed_buy"))
    try:
        env.reset()
        money = read_ram_value(env.get_ram(), "money")
        stock = read_ram_value(env.get_ram(), "potato_seeds")
        task = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE, NAV_FARM_EXIT_PHASE, BUY_SEEDS_PHASE])
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(8000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            bought = next((row for row in task.phase_results if row["phase"] == "BUY_SEEDS"), None)
            if bought is not None:
                break
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        assert bought and bought["status"] == "success", (frame, result.reason, task.phase_results)
        assert read_ram_value(ram, "potato_seeds") == stock + 1, task.phase_results
        assert read_ram_value(ram, "money") == money - 200, task.phase_results
        assert farm_map_loaded(ram)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


class ShopRouteTests(unittest.TestCase):
    def test_east_gate_coordinates_do_not_prove_input_is_ready(self):
        from harvest.core.ram_catalog import live_wram_base
        world = make_transition_world(0x04)
        set_player_pos(world.ram, 744, 424)
        brightness = live_wram_base(world.ram) + 0x25
        world.ram[brightness] = 1
        self.assertFalse(town_coords_settled(world.ram))
        world.ram[brightness] = 15
        self.assertTrue(town_coords_settled(world.ram))

    def test_east_gate_shop_corridor_force_runs(self) -> None:
        """D2 town east-gate approach stalls on walk; force-run on the verified corridor."""
        force = [
            wp.target_px
            for wp in SEGMENTS["town_to_shop_door"]
            if wp.force_run
        ]
        self.assertEqual(force, [(692, 424), (684, 400), (684, 340)])
        self.assertFalse(SEGMENTS["town_to_shop_door"][0].force_run)

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
        self.assertEqual(SEGMENTS["path_to_town_shop"][0].target_px, (132, 128))
        self.assertTrue(SEGMENTS["path_to_town_shop"][0].force_run)

    def test_path_leak_town_pixels_are_not_settled(self) -> None:
        world = make_transition_world(0x04, current_tile=(0, 8))
        world.ram[live_wram_base(world.ram) + 0x25] = 15
        set_player_pos(world.ram, 10, 134)
        self.assertFalse(town_coords_settled(world.ram))
        set_player_pos(world.ram, 756, 422)
        self.assertTrue(town_coords_settled(world.ram))

    def test_transition_scene_is_not_settled(self) -> None:
        world = make_transition_world(0x04, current_tile=(0, 8))
        world.ram[live_wram_base(world.ram) + 0x25] = 15
        set_player_pos(world.ram, 744, 424)
        world.ram[0xD3] = 0xC4  # game_state 0xC4xx: 0x8000 scene running
        self.assertFalse(town_coords_settled(world.ram))
        world.ram[0xD3] = 0x44
        self.assertTrue(town_coords_settled(world.ram))

    def test_segment_choice_follows_live_tilemap(self) -> None:
        self.assertEqual(first_shop_nav_segment(0x00), "farm_to_path")
        self.assertIsNone(
            first_shop_nav_segment(0x00, ("path_to_town_shop", "town_to_shop_door"))
        )
        self.assertEqual(first_shop_nav_segment(0x0C), "path_to_town_shop")
        self.assertEqual(first_shop_nav_segment(0x04), "town_to_shop_door")
        self.assertIsNone(first_shop_nav_segment(0x1C))
        self.assertIsNone(first_shop_nav_segment(0x15))
        self.assertEqual(first_shop_return_segment(0x1C), "shop_to_town")
        self.assertEqual(first_shop_return_segment(0x04), "town_shop_to_path")
        self.assertEqual(first_shop_return_segment(0x0C), "path_to_farm")
        self.assertEqual(SEGMENTS["path_to_farm"][0].target_px, (132, 128))
        self.assertEqual(SEGMENTS["path_to_farm"][-1].exit_direction, "right")


class TownBounceTests(unittest.TestCase):
    """Summer D27: a villager on the east gate pushed the shop walk back out."""

    def _bounced(self, task: BuySeedsTask, world) -> object:
        class _Fail:
            def step(self, _world):
                from retro_harness import TaskResult

                return TaskResult(status=TaskStatus.FAILURE, reason="expected tilemap 0x04, got 0x0C")

        task._child = _Fail()
        task._child_name = "town_to_shop_door"
        task._done_nav.update({"farm_to_path", "path_to_town_shop"})
        world.ram[field_spec("tilemap").address] = 0x0C
        return task.step(world)

    def test_inbound_town_hop_back_on_the_path_walks_in_again(self) -> None:
        world = make_transition_world(0x0C, current_tile=(2, 8))
        task = BuySeedsTask()
        task.reset(world)
        for attempt in range(TOWN_BOUNCE_RETRIES):
            result = self._bounced(task, world)
            self.assertEqual(result.status, TaskStatus.RUNNING, attempt)
            self.assertEqual(task._child_name, "path_to_town_shop")
        result = self._bounced(task, world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("town_to_shop_door failed", result.reason or "")


class ShopExitFadeTests(unittest.TestCase):
    """The seed-shop exit lands on the door apron and the town fades in for ~190 frames."""

    def test_slow_town_fade_after_the_buy_is_waited_out(self) -> None:
        # e2e base1/base0 (power-on, Y1 Spring D2): the apron (600,232) read settled at
        # fade 2 but not faded in; the 180-frame cap fired at fade 14, BUY_SEEDS failed and
        # the farmer stood in town for D2_FARM_CLEAR ("stale_farm_map", whole run lost).
        world = make_transition_world(0x04, current_tile=(0, 8))
        task = BuySeedsTask()
        task.reset(world)
        task._bought = True
        task._seen_shop = True
        task._child = None
        task._child_name = "town_settle"
        set_player_pos(world.ram, 600, 232)
        fade = live_wram_base(world.ram) + 0x25
        for frame in range(200):
            world.ram[fade] = 1 if frame < 190 else 15
            result = task.step(world)
            self.assertNotEqual(result.status, TaskStatus.FAILURE, f"frame {frame}: {result.reason}")

    def test_a_town_that_never_fades_in_still_fails(self) -> None:
        world = make_transition_world(0x04, current_tile=(0, 8))
        task = BuySeedsTask()
        task.reset(world)
        task._bought = True
        task._seen_shop = True
        task._child = None
        task._child_name = "town_settle"
        set_player_pos(world.ram, 600, 232)
        world.ram[live_wram_base(world.ram) + 0x25] = 1
        statuses = [task.step(world).status for _ in range(TOWN_SETTLE_FRAMES + 5)]
        self.assertIn(TaskStatus.FAILURE, statuses)


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
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task

        world = make_world(0x00)
        task = build_phase_task(TaskBuildContext(), BUY_SEEDS_PHASE, world)
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


class SeedStandTests(unittest.TestCase):
    """Summer shelf bags: each seed is bought from its own stand (ROM-measured)."""

    def test_stand_and_price_follow_the_seed(self) -> None:
        self.assertEqual(seed_stand_px("potato_seeds"), SEED_CLERK_PX)
        self.assertEqual(seed_stand_px("corn_seeds"), (152, 342))
        self.assertEqual(seed_stand_px("tomato_seeds"), (168, 342))
        corn = BuySeedsTask(stock_field="corn_seeds")
        self.assertEqual(corn.bag_price, 300)
        self.assertEqual(corn.stand_px, (152, 342))
        self.assertEqual(BuySeedsTask().bag_price, SEED_BAG_PRICE["potato_seeds"])
        self.assertEqual(BuySeedsTask(bag_price=123).bag_price, 123)

    def test_at_seed_clerk_is_relative_to_the_stand(self) -> None:
        import numpy as np
        from harvest.core.tile_catalog import ADDR_X, ADDR_Y

        ram = np.zeros(0x2000, dtype=np.uint8)
        ram[ADDR_X], ram[ADDR_X + 1] = 152, 0
        ram[ADDR_Y], ram[ADDR_Y + 1] = 342 & 0xFF, 342 >> 8
        self.assertFalse(at_seed_clerk(ram))  # potato stand is 30px away
        self.assertTrue(at_seed_clerk(ram, stand=seed_stand_px("corn_seeds")))

    def test_summer_seed_phases_use_the_live_shop_task(self) -> None:
        for crop in ("corn", "tomato"):
            phase = buy_seeds_phase(recording_name="buy_summer", seed_type=crop)
            self.assertEqual(phase.kind, BUY_SEEDS_PHASE.kind)
            self.assertEqual(phase.params["stock_field"], f"{crop}_seeds")


SHED_ROW_2 = 0x11F01 + 0x4000  # live WRAM mirror of $7F1F01
BRUSH, MILKER = 0x0F, 0x0E


def _store_world(*, money: int, shed: int = 0, selected: int = 0, backpack: int = 0, weekday: int = 1):
    from harvest.core.ram_catalog import field_spec

    world = make_date_world(0x15, season=1, day=3)
    set_money(world.ram, money)
    world.ram[SHED_ROW_2] = shed
    world.ram[field_spec("tool_selected").address] = selected
    world.ram[field_spec("tool_backpack").address] = backpack
    world.ram[field_spec("weekday").address + 0x4000] = weekday
    return world


class ToolShelfTests(unittest.TestCase):
    """Brush and milker are general-store (0x22) shelf objects, not seed-shop stock."""

    def test_shelf_table_matches_rom_and_cow_geometry(self) -> None:
        from harvest.planner.livestock_econ import SHED_ROW_2_BRUSH_BIT, SHED_ROW_2_MILKER_BIT

        BRUSH_TOOL_ID, MILKER_TOOL_ID = 0x0F, 0x0E

        brush, milker = TOOL_SHELVES[BRUSH], TOOL_SHELVES[MILKER]
        self.assertEqual((brush.tool_id, brush.price, brush.shed_bit), (BRUSH_TOOL_ID, 800, SHED_ROW_2_BRUSH_BIT))
        self.assertEqual((milker.tool_id, milker.price, milker.shed_bit), (MILKER_TOOL_ID, 1800, SHED_ROW_2_MILKER_BIT))
        # Shelf F6 (14,22) and F4 (14,21): stand one px past the wall stop, facing right.
        self.assertEqual(brush.stand, (217, 358))
        self.assertEqual(milker.stand, (217, 344))
        self.assertEqual(brush.stand[1] // 16, 22)
        self.assertEqual(milker.stand[1] // 16, 21)

    def test_owned_by_shed_bit_or_carry_pair(self) -> None:
        self.assertFalse(tool_owned(_store_world(money=0).ram, BRUSH))
        self.assertTrue(tool_owned(_store_world(money=0, shed=0x40).ram, BRUSH))
        self.assertFalse(tool_owned(_store_world(money=0, shed=0x40).ram, MILKER))
        self.assertTrue(tool_owned(_store_world(money=0, selected=MILKER).ram, MILKER))
        self.assertTrue(tool_owned(_store_world(money=0, backpack=BRUSH).ram, BRUSH))

    def test_plan_spends_the_wallet_down_in_order(self) -> None:
        ids = (BRUSH, MILKER)
        self.assertEqual(plan_tool_buys(_store_world(money=2600).ram, ids), [BRUSH, MILKER])
        self.assertEqual(plan_tool_buys(_store_world(money=2599).ram, ids), [BRUSH])
        self.assertEqual(plan_tool_buys(_store_world(money=799).ram, ids), [])
        # Brush first leaves 1,000: the milker is never planned on a short wallet.
        self.assertEqual(plan_tool_buys(_store_world(money=1800).ram, ids), [BRUSH])
        self.assertEqual(plan_tool_buys(_store_world(money=1800).ram, (MILKER, BRUSH)), [MILKER])
        self.assertEqual(plan_tool_buys(_store_world(money=5000, shed=0x40).ram, ids), [MILKER])
        self.assertEqual(plan_tool_buys(_store_world(money=5000).ram, (BRUSH, BRUSH)), [BRUSH])

    def test_task_with_nothing_to_buy_succeeds_without_walking(self) -> None:
        world = _store_world(money=100)
        task = BuyShelfItemTask(tool_ids=(BRUSH, MILKER))
        task.reset(world)
        self.assertEqual(task.shop, TOOL_SHOP)
        self.assertEqual(task.step(world).status, TaskStatus.SUCCESS)
        with self.assertRaises(ValueError):
            BuyShelfItemTask(tool_ids=(0x10,))

    def test_task_arms_the_first_planned_shelf(self) -> None:
        world = _store_world(money=2600)
        task = BuyShelfItemTask(tool_ids=(BRUSH, MILKER))
        task.reset(world)
        self.assertEqual(task.stand_px, (217, 358))
        self.assertEqual(task.face, "right")
        self.assertEqual(task.counter_segment, "tool_shop_to_brush")
        self.assertEqual(task.bag_price, 800)
        self.assertEqual(task.name, "buy_tools")

    def test_receipt_needs_the_exact_shelf_price(self) -> None:
        world = _store_world(money=2600)
        task = BuyShelfItemTask(tool_ids=(BRUSH,))
        task.reset(world)
        self.assertFalse(task._purchase_ok(world.ram))
        set_money(world.ram, 1800)
        self.assertTrue(task._purchase_ok(world.ram))
        set_money(world.ram, 800)  # paid the milker's price at the brush shelf
        self.assertFalse(task._purchase_ok(world.ram))

    def test_second_shelf_is_a_row_change_off_the_wall(self) -> None:
        """Hugging x=217 blocks the row change: back off first, then up or down."""
        world = _store_world(money=2600)
        task = BuyShelfItemTask(tool_ids=(BRUSH, MILKER))
        task.reset(world)
        task._next_tool(world.ram)  # brush done, milker armed
        self.assertEqual(task.stand_px, (217, 344))
        set_player_pos(world.ram, 217, 358)
        world.ram[0x22] = 0x22
        action = task._align_to_stand(world).action.action
        self.assertTrue(action[6])  # LEFT (button order B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,...)
        set_player_pos(world.ram, 213, 358)
        up = task._align_to_stand(world).action.action
        self.assertTrue(up[4])

    def test_routes_reach_the_general_store_door(self) -> None:
        door = SEGMENTS["town_to_tool_door"][-1]
        self.assertEqual((door.tilemap, door.is_exit, door.exit_direction), (0x04, True, "up"))
        # Door metatile 0xC6 is town tile (25,53); the walkway is x=408.
        self.assertEqual(door.target_px[0] // 16, 25)
        for name in TOOL_SHOP.nav_segments + TOOL_SHOP.return_segments:
            self.assertIn(name, SEGMENTS)
        self.assertEqual(SEGMENTS["tool_shop_to_town"][-1].exit_direction, "down")
        self.assertEqual(
            first_shop_nav_segment(0x04, ["town_to_tool_door"], TOOL_SHOP.tilemap),
            "town_to_tool_door",
        )
        self.assertIsNone(first_shop_nav_segment(0x22, list(TOOL_SHOP.nav_segments), TOOL_SHOP.tilemap))
        self.assertEqual(
            [first_shop_return_segment(tm, TOOL_SHOP.return_segments, TOOL_SHOP.tilemap) for tm in (0x22, 0x04, 0x0C)],
            list(TOOL_SHOP.return_segments),
        )

    def test_exit_flip_coordinates_are_not_settled_for_the_tool_shop(self) -> None:
        """The 0x22 -> 0x04 flip passes through (408,408): x>200 but not the apron."""
        import numpy as np

        task = BuyShelfItemTask(tool_ids=(BRUSH,))
        ram = np.zeros(0x2000, dtype=np.uint8)
        ram[0x25] = 15
        set_player_pos(ram, 408, 408)
        self.assertTrue(task._town_ready(ram))  # inbound: east-gate coords are fine
        task._bought = True
        self.assertFalse(task._town_ready(ram))
        set_player_pos(ram, 408, 872)
        self.assertTrue(task._town_ready(ram))

    def test_phase_wiring(self) -> None:
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task

        (phase,) = tool_purchase_phases()
        self.assertEqual((phase.phase, phase.kind), ("BUY_COW_TOOLS", PhaseKind.SHOP_BUY))
        self.assertEqual(phase.params["tool_ids"], [BRUSH, MILKER])
        self.assertEqual(phase.failure_policy, "optional")
        self.assertEqual(tool_purchase_phases(["milker"])[0].params["tool_ids"], [MILKER])
        task = build_phase_task(TaskBuildContext(), phase, make_world(0x00))
        self.assertIsInstance(task, BuyShelfItemTask)
        self.assertEqual(task.tool_ids, (BRUSH, MILKER))
        # The seed phases are untouched.
        self.assertEqual(build_phase_task(TaskBuildContext(), BUY_SEEDS_PHASE, make_world(0x00)).tool_ids, ())

    def test_missing_cow_tools_and_cow_day_plan(self) -> None:
        none = _store_world(money=0).ram
        self.assertEqual(missing_cow_tools(none), (BRUSH, MILKER))
        both = _store_world(money=0, shed=0x60).ram
        self.assertEqual(missing_cow_tools(both), ())
        self.assertEqual(missing_cow_tools(_store_world(money=0, shed=0x40).ram), (MILKER,))
        self.assertEqual(missing_cow_tools(_store_world(money=0, selected=BRUSH, backpack=MILKER).ram), ())
        full = [p.phase for p in cow_day_phases(both)]
        self.assertEqual(full[0], "ENSURE_ANIMAL_TOOLS")
        brush_only = cow_day_phases(_store_world(money=0, shed=0x40).ram)
        self.assertEqual(brush_only[0].phase, "ENSURE_BRUSH")
        chores = next(p for p in brush_only if p.phase == "COW_CHORES")
        self.assertEqual((chores.params["brush"], chores.params["milk"]), (True, False))
        bare = cow_day_phases(none)
        self.assertNotIn("ENSURE_BRUSH", [p.phase for p in bare])
        chores = next(p for p in bare if p.phase == "COW_CHORES")
        self.assertEqual((chores.params["brush"], chores.params["milk"]), (False, False))

    def test_tool_shop_hours_follow_the_door_flags(self) -> None:
        self.assertTrue(tool_shop_open(_store_world(money=0, weekday=1).ram))
        self.assertFalse(tool_shop_open(_store_world(money=0, weekday=0).ram))  # Sunday
        self.assertFalse(tool_shop_open(_store_world(money=0, weekday=6).ram))  # Saturday


if __name__ == "__main__":
    unittest.main()


def test_unrelated_wallet_debit_before_the_shelf_is_not_a_seed_receipt():
    from unittest.mock import patch

    world = _store_world(money=5000)
    task = BuySeedsTask(stock_field="turnip_seeds")
    with patch.object(task, "_arm_nav"):
        task.reset(world)
        set_money(world.ram, 4000)  # unrelated event on the way to the shop
        world.ram[field_spec("tilemap").address] = task.shop.tilemap
        task.step(world)
    assert not task._bought
    assert task._bags_seen == 0


def test_medicine_has_a_separate_shop_trip_and_a_shed_receipt():
    world = _store_world(money=5000)
    task = BuyShelfItemTask(tool_ids=(9,))
    assert task.shop.tilemap == 0x24
    assert TOOL_SHELVES[9].price == 1000
    assert TOOL_SHELVES[9].shed_bit == 1
    assert not tool_owned(world.ram, 9)
    world.ram[SHED_ROW_2] |= 1
    assert tool_owned(world.ram, 9)
    with pytest.raises(ValueError, match="own animal-shop trip"):
        BuyShelfItemTask(tool_ids=(9, BRUSH))


# Ellen's weekday stand in front of the animal shop door (calendar_schedule: ellen 0x023C-0x0241 at town (648,904)).
ELLEN_STAND_PX = (648, 904)


@pytest.mark.parametrize("route", ["town_animal_shop_to_path", "animal_shop_town_to_farm"])
def test_animal_shop_exit_routes_keep_clear_of_ellens_stand(route):
    """A tight waypoint on her pixel never arrives; the replan then walks back into the shop door.

    Y2 Winter D19 (BUY_MEDICINE, power-on run ``perfect999b``): the medicine trip's exit route started on her
    pixel with radius 4, the day was lost, four sick cows were left uncured and died. No town waypoint of an
    animal-shop exit route may be centred within one tile (16 px) of her pixel; the working routes stop 16 px short
    at (632,904) and swing south of her.
    """
    from harvest.maps.map_routes import ROUTES as MAP_ROUTES, SEGMENTS

    ex, ey = ELLEN_STAND_PX
    for wp in {**SEGMENTS, **MAP_ROUTES}[route]:
        if wp.tilemap != 0x04:
            continue
        dx, dy = wp.target_px[0] - ex, wp.target_px[1] - ey
        assert (dx * dx + dy * dy) ** 0.5 >= 16, (route, wp)
