from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harvest.core.animal_status import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE, COW_DAILY_TALKED_FLAG
from harvest.planner.day_plan import (
    ADDR_CHICKEN_COUNT,
    ADDR_CORN_SEEDS,
    ADDR_COW_COUNT,
    ADDR_DAY,
    ADDR_EGG_AVAILABLE,
    ADDR_FED_CHICKENS_N,
    ADDR_FED_COWS_N,
    ADDR_SEASON,
    ADDR_TILEMAP,
    ADDR_HAY_COUNT,
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_MONEY,
    ADDR_WEEKDAY,
    ADDR_WEATHER,
    ADDR_WEATHER_FLAGS,
    ActionResult,
    CHICKEN_PHASES,
    ChickenSaleEventTask,
    ChickenSaleFollowupTask,
    ChickenSaleRequestTask,
    COOP_TILEMAP,
    CoopPickupChickenTask,
    COW_PHASES,
    COW_PURCHASE_COST,
    CrossMapRecordedTask,
    DayPlannerPolicy,
    DayPlanTask,
    DeadlineCheckTask,
    DirectionalTransitionTask,
    DropCarriedChickenTask,
    EnsureAnimalToolsTask,
    EnsureCropSeedsTask,
    EnsureCarryToolTask,
    ExitBuildingTask,
    ExitToFarmTask,
    ShedFetchItemTask,
    EveTalkLoopTask,
    EVE_TALK_LOOP_PHASES,
    GoToSleepTask,
    HARVEST_ROUTE_PHASE,
    ReturnHomeTask,
    ShedShelfToolTask,
    SwapCarrySlotsTask,
    MultiMapNavTask,
    NavTask,
    MultiDayPlannerTask,
    PHASE_SEQUENCES,
    PhaseSpec,
    RecordedTransitionTask,
    SHED_TOOL_SPECS,
    SHED_SEED_SPECS,
    TaskResult,
    TaskStatus,
    WaitUntilTimeTask,
    auto_day_phases,
    auto_day_plan_name_for_ram,
    auto_day_plan_name_for_state,
    auto_day_plan_name_for_weekday,
    build_day_phases,
    build_day_phases_from_ram,
    build_outdoor_day_phases,
    build_outdoor_day_phases_from_ram,
    is_farm_tilemap,
    is_house_tilemap,
    is_rainy_weather,
    make_action,
    romance_points_for_hearts,
    ram_has_waterable_crops,
    state_has_chickens,
    state_has_cows,
)
from harvest.planner.day_task_factory import DayTaskFactory
from harvest.planner.day_plan_decision import DayPlanDecision, DeferredPlan, PlanningFacts
from harvest.planner.world_probe import WorldProbe
from harvest.tasks.crop_planter import is_rainy_weather as crop_task_is_rainy_weather
from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_X, ADDR_Y, Navigator, Pathfinder, Point, TileScanner, Tool
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec


def make_world(tilemap: int):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    return SimpleNamespace(ram=ram, info={}, obs=None)


def make_time_world(tilemap: int, *, day: int, hour: int, minute: int, live_offset: bool = False):
    ram = np.zeros(0x24000, dtype=np.uint8) if live_offset else np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    base = 0x4000 if live_offset else 0
    ram[ADDR_DAY + base] = day
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    return SimpleNamespace(ram=ram, info={}, obs=None)


def make_date_world(tilemap: int, *, season: int, day: int, hour: int = 6, minute: int = 0):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    base = 0x4000
    ram[ADDR_SEASON + base] = season
    ram[ADDR_DAY + base] = day
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    # Settled stand points so scene classification is not wake/invalid.
    if is_house_tilemap(tilemap):
        set_player_pos(ram, 136, 120)
    else:
        set_player_pos(ram, 136, 424)
    return SimpleNamespace(ram=ram, info={}, obs=None)


def set_player_pos(ram: np.ndarray, x: int, y: int) -> None:
    ram[ADDR_X] = x & 0xFF
    ram[ADDR_X + 1] = (x >> 8) & 0xFF
    ram[ADDR_Y] = y & 0xFF
    ram[ADDR_Y + 1] = (y >> 8) & 0xFF


def set_money(ram: np.ndarray, amount: int, *, live_offset: bool = True) -> None:
    base = 0x4000 if live_offset else 0
    raw = amount // 10
    ram[ADDR_MONEY + base] = raw & 0xFF
    ram[ADDR_MONEY + base + 1] = (raw >> 8) & 0xFF
    ram[ADDR_MONEY + base + 2] = (raw >> 16) & 0xFF


def set_live_u16(ram: np.ndarray, field: str, value: int) -> None:
    addr = field_spec(field).address
    base = 0x4000 if len(ram) > 0x20000 and addr + 0x4001 < len(ram) else 0
    ram[addr + base] = value & 0xFF
    ram[addr + base + 1] = (value >> 8) & 0xFF


def set_live_cow_slot(
    ram: np.ndarray,
    slot: int,
    *,
    flags: int = 0,
    happiness: int = 0,
    tile: tuple[int, int] = (9, 17),
) -> None:
    offset = 0x4000 + COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = 0x05
    ram[offset + 1] = flags
    ram[offset + 2] = 0x27
    ram[offset + 4] = happiness
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 8] = px & 0xFF
    ram[offset + 9] = px >> 8
    ram[offset + 10] = py & 0xFF
    ram[offset + 11] = py >> 8


def set_live_chicken_slots(ram: np.ndarray, *, adults: int = 0, chicks: int = 0, eggs: int = 0) -> None:
    base = 0x4000 if len(ram) > 0x20000 else 0
    slot = 0
    for count, status in ((adults, 0x09), (chicks, 0x05), (eggs, 0x01)):
        for _ in range(count):
            offset = base + CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE
            ram[offset] = status
            slot += 1


def make_navigation_ram(*, current_tile=(13, 8), blocked_tile=(14, 8), blocked_id=0x76):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    set_player_pos(ram, current_tile[0] * 16 + 8, current_tile[1] * 16 + 8)
    for ty in range(64):
        for tx in range(64):
            ram[ADDR_MAP + ty * 64 + tx] = 0xA1
    ram[ADDR_MAP + blocked_tile[1] * 64 + blocked_tile[0]] = blocked_id
    return ram


def make_transition_world(tilemap: int, *, current_tile=(13, 8)):
    ram = make_navigation_ram(current_tile=current_tile, blocked_tile=(63, 63), blocked_id=0xA1)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    return SimpleNamespace(ram=ram, info={}, obs=None)


class DayPlanSequenceTests(unittest.TestCase):
    def test_berries_water_sequence_keeps_recorded_berry_route_disabled(self) -> None:
        phases = PHASE_SEQUENCES["berries_water"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])
        self.assertNotIn("BUY_SEEDS", [phase.phase for phase in phases])

    def test_sunday_sequence_keeps_recorded_berry_route_disabled(self) -> None:
        phases = PHASE_SEQUENCES["sunday"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])
        self.assertNotIn("BUY_SEEDS", [phase.phase for phase in phases])

    def test_weekday_sequence_keeps_town_first_route(self) -> None:
        phases = PHASE_SEQUENCES["day1"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "CLEAR_FIELD",
                "NAV_FARM_EXIT",
                "BUY_SEEDS",
                "ENSURE_CROP_SEEDS",
                "NAV_CROP",
                "CROP_ESTABLISH",
                "ENSURE_WATERING_CAN",
                "NAV_CROP",
                "CROP_WATER",
                "TOWN_EXPLORE",
                "READY_TO_GO_HOME",
                "RETURN_HOME",
                "GO_TO_SLEEP",
            ],
        )
        self.assertIn("BUY_SEEDS", [phase.phase for phase in phases])
        self.assertIn("TOWN_EXPLORE", [phase.phase for phase in phases])
        self.assertIn("RETURN_HOME", [phase.phase for phase in phases])
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])

    def test_boot_to_day2_sequence_chains_macros_and_sleep(self) -> None:
        phases = PHASE_SEQUENCES["boot_to_day2"]
        names = [phase.phase for phase in phases]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        self.assertIn("GET_HAMMER", names)
        self.assertIn("BUY_SEEDS", names)
        self.assertIn("TOWN_EXPLORE", names)
        self.assertIn("READY_TO_GO_HOME", names)
        self.assertEqual(names[-2:], ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_resume_water_sequence_exits_house_then_routes_to_crop_watering(self) -> None:
        phases = PHASE_SEQUENCES["resume_water"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertEqual(phases[2].params["target_px"], (248, 472))
        self.assertEqual(phases[3].params["refill_bounds"], (3, 10, 62, 60))

    def test_harvest_sequence_uses_trimmed_recording_only(self) -> None:
        phases = PHASE_SEQUENCES["harvest"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "NAV_CROP", "HARVEST_ROUTE"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "nav", "harvest"],
        )

    def test_buy_cow_sequence_splits_nav_menu_name_and_barn_chores(self) -> None:
        phases = PHASE_SEQUENCES["buy_cow"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "NAV_TO_ANIMAL_SHOP",
                "BUY_COW_VENDOR",
                "EXIT_ANIMAL_SHOP",
                "RETURN_FARM_AFTER_COW_PURCHASE",
                "NAME_COW",
                "ENSURE_ANIMAL_TOOLS",
                "NAV_TO_BARN",
                "ENTER_BARN",
                "COW_CHORES",
            ],
        )
        self.assertEqual(phases[2].kind, "cow_purchase")
        self.assertEqual(phases[2].params["start_frame"], 1631)
        self.assertEqual(phases[5].params["end_frame"], 5001)

    def test_eve_loop_sequence_replays_recording_from_bar_exterior(self) -> None:
        phases = PHASE_SEQUENCES["eve_loop"]

        self.assertIs(phases, EVE_TALK_LOOP_PHASES)
        self.assertEqual(
            [phase.phase for phase in phases],
            ["EVE_TALK_LOOP"],
        )
        self.assertEqual(phases[0].kind, "eve_talk_loop")
        self.assertEqual(phases[0].params["target_hearts"], 10)

    def test_sell_chicken_test_sequence_is_only_required_sale_work(self) -> None:
        phases = PHASE_SEQUENCES["sell_chicken_test"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "NAV_TO_COOP_FOR_SALE_TEST",
                "ENTER_COOP_FOR_SALE_TEST",
                "PICKUP_CHICKEN_FOR_SALE_TEST",
                "EXIT_COOP_FOR_SALE_TEST",
                "DROP_CHICKEN_FOR_SALE_TEST",
                "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST",
                "REQUEST_CHICKEN_SALE_TEST",
                "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST",
                "RETURN_FARM_AFTER_CHICKEN_SALE_TEST",
                "SELL_CHICKEN_TEST",
            ],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            [
                "multi_nav",
                "directional_transition",
                "pickup_chicken",
                "directional_transition",
                "drop_chicken",
                "multi_nav",
                "chicken_sale_request",
                "multi_nav",
                "multi_nav",
                "chicken_sale_event",
            ],
        )
        self.assertTrue(all(phase.failure_policy == "required" for phase in phases))

    def test_sell_three_chickens_test_repeats_required_sale_cycle(self) -> None:
        phases = PHASE_SEQUENCES["sell_three_chickens_test"]

        self.assertEqual(len(phases), 30)
        self.assertEqual([phase.phase for phase in phases[0:10]], [
            "NAV_TO_COOP_FOR_SALE_TEST_1",
            "ENTER_COOP_FOR_SALE_TEST_1",
            "PICKUP_CHICKEN_FOR_SALE_TEST_1",
            "EXIT_COOP_FOR_SALE_TEST_1",
            "DROP_CHICKEN_FOR_SALE_TEST_1",
            "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST_1",
            "REQUEST_CHICKEN_SALE_TEST_1",
            "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST_1",
            "RETURN_FARM_AFTER_CHICKEN_SALE_TEST_1",
            "SELL_CHICKEN_TEST_1",
        ])
        self.assertEqual([phase.phase for phase in phases[-10:]], [
            "NAV_TO_COOP_FOR_SALE_TEST_3",
            "ENTER_COOP_FOR_SALE_TEST_3",
            "PICKUP_CHICKEN_FOR_SALE_TEST_3",
            "EXIT_COOP_FOR_SALE_TEST_3",
            "DROP_CHICKEN_FOR_SALE_TEST_3",
            "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST_3",
            "REQUEST_CHICKEN_SALE_TEST_3",
            "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST_3",
            "RETURN_FARM_AFTER_CHICKEN_SALE_TEST_3",
            "SELL_CHICKEN_TEST_3",
        ])

    def test_sell_three_chickens_batch_test_stages_before_shop(self) -> None:
        phases = PHASE_SEQUENCES["sell_three_chickens_batch_test"]

        self.assertEqual(len(phases), 22)
        self.assertEqual([phase.phase for phase in phases[:15:5]], [
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_1",
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_2",
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_3",
        ])
        self.assertEqual(
            [phase.phase for phase in phases[15:]],
            [
                "NAV_TO_ANIMAL_SHOP_FOR_SALE_BATCH_TEST",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_1",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_2",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_3",
                "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_BATCH_TEST",
                "RETURN_FARM_AFTER_CHICKEN_SALE_BATCH_TEST",
                "SELL_CHICKEN_BATCH_TEST",
            ],
        )
        self.assertEqual(phases[-1].params["target_sales"], 3)
        self.assertEqual(phases[-1].params["event_hour"], 0)
        self.assertTrue(all(phase.failure_policy == "required" for phase in phases))

    def test_chicken_sale_phase_factory_builds_subtasks(self) -> None:
        factory = DayTaskFactory()
        world = make_world(0x00)
        phases = PHASE_SEQUENCES["sell_chicken_test"]

        self.assertIsInstance(factory.make_task(phases[0], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[2], world), CoopPickupChickenTask)
        self.assertIsInstance(factory.make_task(phases[4], world), DropCarriedChickenTask)
        self.assertIsInstance(factory.make_task(phases[5], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[6], world), ChickenSaleRequestTask)
        self.assertIsInstance(factory.make_task(phases[8], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[9], world), ChickenSaleEventTask)

        batch_event = factory.make_task(PHASE_SEQUENCES["sell_three_chickens_batch_test"][-1], world)
        self.assertIsInstance(batch_event, ChickenSaleEventTask)
        self.assertEqual(batch_event.target_sales, 3)

    def test_chicken_sale_request_ignores_stale_request_text(self) -> None:
        world = make_world(0x24)
        set_player_pos(world.ram, 201, 158)
        text_addr = field_spec("dialog_text_id").address
        world.ram[text_addr] = 0x0B
        world.ram[text_addr + 1] = 0x03

        task = ChickenSaleRequestTask(timeout=40)
        task.reset(world)

        result = None
        for _ in range(20):
            result = task.step(world)

        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)

    def test_chicken_sale_followup_stops_after_verified_sale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/sell_chicken.json", "w") as f:
                json.dump({"name": "sell_chicken", "frames": [[0] * 12 for _ in range(5)]}, f)

            world = make_date_world(0x00, season=0, day=10)
            set_player_pos(world.ram, 60, 480)
            base = 0x4000
            world.ram[ADDR_CHICKEN_COUNT + base] = 6
            world.ram[ADDR_SHIPPING_MONEY + base] = 65

            task = ChickenSaleFollowupTask(tasks_dir=tmp, start_frame=0, success_settle_frames=2)
            task.reset(world)

            world.ram[ADDR_CHICKEN_COUNT + base] = 5
            world.ram[ADDR_SHIPPING_MONEY + base] = 80
            first = task.step(world)
            second = task.step(world)

            self.assertEqual(first.status, TaskStatus.RUNNING)
            self.assertEqual(second.status, TaskStatus.SUCCESS)
            self.assertIn("chickens 6->5", second.reason)

    def test_chicken_sale_event_stops_after_verified_sale_money(self) -> None:
        world = make_date_world(0x00, season=0, day=25, hour=15, minute=4)
        base = 0x4000
        world.ram[ADDR_CHICKEN_COUNT + base] = 6
        set_money(world.ram, 7830)

        task = ChickenSaleEventTask(success_settle_frames=2)
        task.reset(world)

        world.ram[ADDR_CHICKEN_COUNT + base] = 5
        set_money(world.ram, 8330)
        first = task.step(world)
        second = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(second.status, TaskStatus.SUCCESS)
        self.assertIn("chickens 6->5", second.reason)
        self.assertIn("money 7830->8330", second.reason)

    def test_auto_day_plan_name_uses_sunday_for_sunday_weekday(self) -> None:
        self.assertEqual(auto_day_plan_name_for_weekday(0), "sunday")
        self.assertEqual(auto_day_plan_name_for_weekday(4), "day1")

    def test_auto_day_plan_name_prefers_berry_route_when_seed_stock_exists(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY] = 1
        ram[ADDR_HOUR] = 6
        ram[ADDR_MINUTE] = 0
        ram[0x092A] = 3
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "berries_water")

    def test_auto_day_plan_name_uses_resume_water_for_non_morning_state(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY] = 1
        ram[ADDR_HOUR] = 15
        ram[ADDR_MINUTE] = 0
        ram[0x092A] = 3
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "resume_water")

    def test_auto_day_plan_name_prefers_harvest_when_mature_crops_exist(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 6, 0)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=False):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_prefers_harvest_over_resume_water(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 6, 0)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=True), patch.object(
            WorldProbe, "is_rainy", return_value=False
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_keeps_harvest_priority_later_in_day_before_cutoff(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 8, 30)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=False):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_for_ram_prefers_resume_water_when_tiles_need_water(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY + 0x4000] = 2
        ram[ADDR_HOUR + 0x4000] = 18
        ram[ADDR_MINUTE + 0x4000] = 5
        ram[ADDR_MAP + 35 * 64 + 11] = 0x58

        self.assertEqual(auto_day_plan_name_for_ram(ram, fallback_state_name="fake"), "resume_water")

    def test_deadline_check_reads_live_time_from_offset_ram(self) -> None:
        task = DeadlineCheckTask(latest_hour=17, latest_minute=0)
        world = make_time_world(0x00, day=9, hour=11, minute=10, live_offset=True)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "before cutoff at 11:10")

    def test_deadline_check_fails_at_shipping_cutoff(self) -> None:
        task = DeadlineCheckTask(latest_hour=17, latest_minute=0)
        world = make_time_world(0x00, day=9, hour=17, minute=0)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(result.reason, "cutoff reached at 17:00")

    def test_recorded_transition_requires_target_tilemap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/leave_farm_west.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "name": "leave_farm_west",
                        "frames": [[0] * 12, [0] * 12],
                        "start_state": "pinned_fixture",
                    },
                    f,
                )

            task = RecordedTransitionTask(
                task_name="leave_farm_west",
                tasks_dir=tmpdir,
                origin_tilemap=0x00,
                target_tilemap=0x0C,
                min_frames_before_success=1,
            )
            task.reset(make_world(0x00))

            first = task.step(make_world(0x00))
            self.assertEqual(first.status, TaskStatus.RUNNING)

            second = task.step(make_world(0x0C))
            self.assertEqual(second.status, TaskStatus.SUCCESS)

    def test_recorded_transition_accepts_seasonal_farm_target(self) -> None:
        world = make_world(0x01)
        task = RecordedTransitionTask(target_tilemap=0x00, min_frames_before_success=0)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x01")

    def test_cross_map_exit_waits_on_seasonal_farm_origin(self) -> None:
        world = make_world(0x01)
        task = CrossMapRecordedTask(origin_tilemap=0x00, exit_direction="left")

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)

    def test_eve_talk_loop_replays_talk_chunk_until_heart_gain_then_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "name": "talk_eve_loop",
                        "frames": [[0] * 12, [1] + [0] * 11],
                    },
                    f,
                )

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2, max_loops=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            first = task.step(world)
            self.assertEqual(first.status, TaskStatus.RUNNING)
            self.assertEqual(first.reason, "talk")

            set_live_u16(world.ram, "eve_hearts", 50)
            second = task.step(world)
            self.assertEqual(second.status, TaskStatus.RUNNING)
            self.assertEqual(second.reason, "heart_gain")
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_ignores_small_yes_gain_and_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            set_live_u16(world.ram, "eve_hearts", 44)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "small_heart_gain")
            self.assertEqual(task._loop_count, 0)
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_overrides_locked_choice_to_no(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            world.ram[ADDR_INPUT_LOCK] = 2
            world.ram[field_spec("player_action").address] = 9
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "choose_no")
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[5]), 1)

    def test_eve_talk_loop_resets_entry_when_talk_stops_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            task.step(world)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "reset_after_missed_talk")
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_is_noop_when_target_already_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_live_u16(world.ram, "eve_hearts", 999)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=10)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.SUCCESS)
            self.assertIn("eve hearts 999/999", result.reason)

    def test_eve_talk_loop_stops_at_daily_question_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_live_u16(world.ram, "eve_hearts", 120)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=10)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.SUCCESS)
            self.assertIn("eve daily question cap 120/999", result.reason)

    def test_eve_talk_loop_exits_bar_between_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1)
            task.reset(world)
            task._phase = "exit_bar"

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "exit_bar")
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[7]), 1)
            self.assertEqual(int(result.action.action[0]), 1)

    def test_eve_talk_loop_rejects_wrong_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x00)
            set_player_pos(world.ram, 152, 872)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1, align_timeout=2)
            task.reset(world)

            task.step(world)
            task.step(world)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.FAILURE)
            self.assertIn("expected Eve loop origin", result.reason)

    def test_eve_talk_loop_aligns_to_recording_origin_on_same_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x04)
            set_player_pos(world.ram, 152, 953)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[4]), 1)

    def test_day_plan_advances_to_next_phase_without_idle_frame(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("FIRST", "custom"),
                        PhaseSpec("SECOND", "custom"),
                    ]
                )
                self._tasks = [ImmediateSuccessTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(plan.phase_text, "SECOND")

    def test_day_plan_skips_berry_route_on_harvest_failure(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="incomplete harvest")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        HARVEST_ROUTE_PHASE,
                        PhaseSpec("BERRY_RUN_WINDOW", "deadline"),
                        PhaseSpec("EXIT_FARM_WEST", "directional_transition"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self._tasks = [FailTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(
            [item.phase for item in plan.deferred_plans],
            ["HARVEST_ROUTE", "BERRY_RUN_WINDOW", "EXIT_FARM_WEST"],
        )
        self.assertTrue(all(item.reason == "incomplete harvest" for item in plan.deferred_plans))

    def test_day_plan_skips_optional_route_on_deadline_failure(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="missed cutoff")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("BERRY_RUN_WINDOW", "deadline"),
                        PhaseSpec("EXIT_FARM_WEST", "directional_transition"),
                        PhaseSpec("GET_BERRIES_AND_SHIP", "recorded"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self._tasks = [FailTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(
            [item.phase for item in plan.deferred_plans],
            ["BERRY_RUN_WINDOW", "EXIT_FARM_WEST", "GET_BERRIES_AND_SHIP"],
        )
        self.assertTrue(all(item.reason == "missed cutoff" for item in plan.deferred_plans))

    def test_day_plan_appends_return_home_after_late_optional_route_skip(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="cutoff reached at 18:06")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("BERRY_RUN_WINDOW", "deadline"),
                        PhaseSpec("EXIT_FARM_WEST", "directional_transition"),
                        PhaseSpec("BERRY_RECORDING_WINDOW", "deadline"),
                        PhaseSpec("GET_BERRIES_AND_SHIP", "recorded"),
                    ]
                )
                self._tasks = [FailTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_date_world(0x00, season=0, day=13, hour=18, minute=6)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "RETURN_HOME")
        self.assertEqual(
            [phase.phase for phase in plan.phases],
            ["BERRY_RUN_WINDOW", "EXIT_FARM_WEST", "BERRY_RECORDING_WINDOW", "GET_BERRIES_AND_SHIP"],
        )
        self.assertEqual([phase.phase for phase in plan.runtime_phases[-2:]], ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_day_plan_skips_optional_blocked_phase(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("OPTIONAL_SETUP", "test", failure_policy="optional"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self._tasks = [BlockedTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual([item.phase for item in plan.deferred_plans], ["OPTIONAL_SETUP"])
        self.assertEqual(plan.deferred_plans[0].reason, "precondition missing")

    def test_day_plan_recovers_required_blocked_phase_then_retries(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

        class SuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class RecoverySuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="recovered")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self.phase_attempts = 0
                self.recovery_attempts = 0

            def _make_task(self, spec, world):
                if spec.phase == "EXIT_TO_FARM":
                    self.phase_attempts += 1
                    if self.phase_attempts == 1:
                        return BlockedTask()
                    return SuccessTask()
                return ActionTask()

            def _make_recovery_task(self, spec, status, reason, world):
                self.recovery_attempts += 1
                return RecoverySuccessTask()

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(plan.phase_attempts, 2)
        self.assertEqual(plan.recovery_attempts, 1)

    def test_day_plan_advances_exit_phase_when_recovery_reaches_target_map(self) -> None:
        class FailingExitTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="exit blocked")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class RecoveryToFarmTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_TILEMAP] = 0x01
                set_player_pos(world.ram, 328, 360)
                return TaskResult(status=TaskStatus.SUCCESS, reason="recovered to farm")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec(
                            "EXIT_BARN",
                            "directional_transition",
                            {"target_tilemap": 0x00},
                        ),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self.exit_attempts = 0

            def _make_task(self, spec, world):
                if spec.phase == "EXIT_BARN":
                    self.exit_attempts += 1
                    return FailingExitTask()
                return ActionTask()

            def _make_recovery_task(self, spec, status, reason, world):
                return RecoveryToFarmTask()

        plan = TestPlan()
        world = make_world(0x27)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(plan.exit_attempts, 1)

    def test_day_plan_retries_crop_water_with_targeted_watering_can_recovery(self) -> None:
        class MissingWateringCanTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="watering can not in carry pair")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(phase_sequence=[PhaseSpec("CROP_WATER", "crop")])

            def _make_task(self, spec, world):
                return MissingWateringCanTask()

        plan = TestPlan()
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.MILKER)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(plan._recovery_task, EnsureCarryToolTask)
        self.assertEqual(plan.phase_text, "CROP_WATER")

    def test_day_plan_aborts_required_blocked_phase_after_recovery_blocks(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

        class RecoveryBlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="recovery stuck")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )

            def _make_task(self, spec, world):
                return BlockedTask()

            def _make_recovery_task(self, spec, status, reason, world):
                return RecoveryBlockedTask()

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertIn("required phase EXIT_TO_FARM failed after recovery: precondition missing", result.reason)
        self.assertIn("recovery blocked: recovery stuck", result.reason)
        self.assertEqual(plan.phase_text, "EXIT_TO_FARM")

    def test_day_plan_aborts_required_missing_task(self) -> None:
        plan = DayPlanTask(phase_sequence=[PhaseSpec("ENTER_BARN", "unknown_kind")])
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("required phase ENTER_BARN failed: no task", result.reason)
        self.assertEqual(plan.phase_text, "ENTER_BARN")

    def test_exit_to_farm_uses_return_route_from_path(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x0C)

        task.reset(world)

        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.name, "return_path_to_farm")
        self.assertEqual(task._task.waypoints[-1].target_px, (244, 128))

    def test_exit_to_farm_uses_event_town_return_route(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x05)
        set_player_pos(world.ram, 440, 160)

        task.reset(world)

        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.name, "return_event_town_to_farm")
        self.assertEqual(task._task.waypoints[0].tilemap, 0x05)
        self.assertEqual(task._task.waypoints[-1].target_px, (244, 128))

    def test_exit_to_farm_succeeds_on_seasonal_farm_tilemaps(self) -> None:
        for tilemap in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap=tilemap):
                task = ExitToFarmTask()
                world = make_world(tilemap)

                task.reset(world)
                result = task.step(world)

                self.assertTrue(is_farm_tilemap(tilemap))
                self.assertIsNone(task._task)
                self.assertEqual(result.status, TaskStatus.SUCCESS)
                self.assertEqual(result.reason, f"tilemap=0x{tilemap:02X}")

    def test_exit_to_farm_uses_scripted_house_exit(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x15)

        task.reset(world)

        self.assertIsInstance(task._task, ExitBuildingTask)
        self.assertEqual(task._task.target_tilemap, 0x00)
        self.assertEqual(task._task.timeout, 2200)

    def test_exit_to_farm_accepts_remodeled_house_tilemaps(self) -> None:
        for tilemap in (0x16, 0x17):
            with self.subTest(tilemap=tilemap):
                task = ExitToFarmTask()
                world = make_world(tilemap)

                task.reset(world)

                self.assertIsInstance(task._task, ExitBuildingTask)
                self.assertTrue(is_house_tilemap(tilemap))

    def test_exit_to_farm_mashes_then_blocks_stuck_cutscene(self) -> None:
        task = ExitToFarmTask(cutscene_mash_limit=2)
        world = make_world(0xFE)
        set_player_pos(world.ram, 136, 424)

        task.reset(world)
        first = task.step(world)
        second = task.step(world)
        blocked = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIn("cutscene_event", first.reason or "")
        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertEqual(blocked.status, TaskStatus.BLOCKED)
        self.assertIn("cannot exit to farm from cutscene_event", blocked.reason or "")

    def test_exit_to_farm_step_mashes_cutscene_after_lazy_reset(self) -> None:
        task = ExitToFarmTask(cutscene_mash_limit=1)
        world = make_world(0xFE)
        set_player_pos(world.ram, 136, 424)

        first = task.step(world)
        blocked = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIn("cutscene_event", first.reason or "")
        self.assertEqual(blocked.status, TaskStatus.BLOCKED)
        self.assertIn("cannot exit to farm from cutscene_event", blocked.reason or "")

    def test_house_exit_relocalizes_from_live_position(self) -> None:
        task = ExitBuildingTask(target_tilemap=0x00)
        world = make_transition_world(0x15, current_tile=(8, 7))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[5]), 0)

    def test_house_exit_recovers_from_doorway_offset(self) -> None:
        task = ExitBuildingTask(target_tilemap=0x00)
        world = make_transition_world(0x15, current_tile=(10, 12))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)

    def test_house_exit_waits_for_stable_farm_coordinates(self) -> None:
        player_state_addr = field_spec("player_state").address
        task = ExitBuildingTask(target_tilemap=0x00, settle_frames=1)
        world = make_transition_world(0x00, current_tile=(8, 13))
        set_player_pos(world.ram, 137, 212)
        world.ram[player_state_addr] = 0x81

        task.reset(world)
        task._step_count = 30
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        # Mid-warp outdoor (y < 330): hold down+B — neutral settle freezes control (rr-bhr).
        self.assertEqual(result.reason, "exit mid-warp push")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)  # down

        set_player_pos(world.ram, 137, 344)
        world.ram[player_state_addr] = 0x01
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_house_exit_accepts_seasonal_farm_tilemap(self) -> None:
        player_state_addr = field_spec("player_state").address
        task = ExitBuildingTask(target_tilemap=0x00, settle_frames=1)
        world = make_world(0x01)
        set_player_pos(world.ram, 137, 344)
        world.ram[player_state_addr] = 0x01

        task.reset(world)
        task._step_count = 30
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x01")

    def test_navigator_treats_transition_tile_0x76_as_stale_and_repaths(self) -> None:
        ram = make_navigation_ram()
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])

    def test_navigator_clears_path_for_known_nonwalkable_tile(self) -> None:
        ram = make_navigation_ram(blocked_tile=(14, 8), blocked_id=0xA6)
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])
        self.assertIn((14, 8), navigator.pathfinder.temp_blocked)

    def test_farm_free_move_ready_and_house_front_softlock(self) -> None:
        """Gate B: free-move bit 0x4000 distinguishes good outdoor from soft-lock."""
        from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, live_wram_base
        from harvest.planner.tasks.inventory import (
            EVENT_1F68_OUTDOOR_INTRO_MASK,
            farm_free_move_ready,
            farm_house_front_softlock,
            outdoor_intro_flags_ready,
        )

        world = make_world(0x00)
        set_player_pos(world.ram, 136, 344)
        # game_state u16 @ 0x00D2; base 0 for 0x20000 ram fixtures
        lo = live_wram_base(world.ram) + 0x00D2
        world.ram[lo] = 0x01
        world.ram[lo + 1] = 0x40  # 0x4001 free-move
        self.assertTrue(farm_free_move_ready(world.ram))
        self.assertFalse(farm_house_front_softlock(world.ram))

        world.ram[lo + 1] = 0x00  # 0x0001 control lost
        set_player_pos(world.ram, 133, 425)
        self.assertFalse(farm_free_move_ready(world.ram))
        self.assertTrue(farm_house_front_softlock(world.ram))

        # Live-size snapshot still resolves free-move via WRAM mirror.
        live = make_time_world(0x00, day=2, hour=6, minute=0, live_offset=True)
        set_player_pos(live.ram, 136, 344)
        lo2 = LIVE_RAM_WRAM_OFFSET + 0x00D2
        live.ram[lo2] = 0x01
        live.ram[lo2 + 1] = 0x40
        self.assertTrue(farm_free_move_ready(live.ram))

        # outdoor intro flags: truck 0x0011 incomplete; 0x00A1 / 0x00B1 ready
        f68 = live_wram_base(world.ram) + 0x11F68
        world.ram[f68] = 0x11
        world.ram[f68 + 1] = 0x00
        self.assertFalse(outdoor_intro_flags_ready(world.ram))
        world.ram[f68] = EVENT_1F68_OUTDOOR_INTRO_MASK & 0xFF  # 0xA1
        world.ram[f68 + 1] = 0x00
        self.assertTrue(outdoor_intro_flags_ready(world.ram))
        world.ram[f68] = 0xB1
        self.assertTrue(outdoor_intro_flags_ready(world.ram))

    def test_complete_outdoor_morning_intro_already_ready(self) -> None:
        """Gate B: CompleteOutdoorMorningIntroTask no-ops when flags+free ready."""
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.tasks.inventory import CompleteOutdoorMorningIntroTask

        world = make_world(0x00)
        set_player_pos(world.ram, 136, 424)
        base = live_wram_base(world.ram)
        world.ram[base + 0x00D2] = 0x01
        world.ram[base + 0x00D3] = 0x40  # free-move
        world.ram[base + 0x11F68] = 0xB1
        world.ram[base + 0x11F69] = 0x00
        task = CompleteOutdoorMorningIntroTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already ready", result.reason or "")

    def test_complete_outdoor_morning_intro_name_entry_presses(self) -> None:
        """Gate B: dog name tilemap 0x5F uses PowerOn-style AAAA + OK path."""
        from harvest.planner.tasks.inventory import CompleteOutdoorMorningIntroTask

        world = make_world(0x5F)
        # input_lock @ ADDR_INPUT_LOCK
        world.ram[ADDR_INPUT_LOCK] = 5
        world.ram[0x099F] = 3  # dog name kind
        world.ram[0x0994] = 0  # name length
        world.ram[0x0991] = 0  # cursor
        # free-move off, intro incomplete
        from harvest.core.ram_catalog import live_wram_base

        base = live_wram_base(world.ram)
        world.ram[base + 0x00D2] = 0x01
        world.ram[base + 0x00D3] = 0x00
        world.ram[base + 0x11F68] = 0x31
        world.ram[base + 0x11F69] = 0x00

        task = CompleteOutdoorMorningIntroTask()
        task.reset(world)
        r1 = task.step(world)
        self.assertEqual(r1.status, TaskStatus.RUNNING)
        self.assertIsNotNone(r1.action)
        self.assertIn("dog name char", r1.reason or "")

        # After 4 chars, cursor 0 → LEFT toward OK
        world.ram[0x0994] = 4
        world.ram[0x0991] = 0
        task._last_input_step = -100
        r2 = task.step(world)
        self.assertEqual(r2.status, TaskStatus.RUNNING)
        self.assertIn("move to OK", r2.reason or "")

        world.ram[0x0991] = 40
        task._last_input_step = -100
        r3 = task.step(world)
        self.assertIn("move to OK", r3.reason or "")

        world.ram[0x0991] = 70
        task._last_input_step = -100
        r4 = task.step(world)
        self.assertIn("confirm dog name", r4.reason or "")
        self.assertTrue(task._name_submitted)

    def test_exit_to_farm_task_exits_shed_with_downward_transition(self) -> None:
        world = make_world(0x26)
        set_player_pos(world.ram, 8 * 16 + 8, 12 * 16 + 8)
        task = ExitToFarmTask()
        task.reset(world)

        first = task.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIsNotNone(first.action)
        self.assertEqual(int(first.action.action[5]), 1)

        for _ in range(13):
            task.step(make_world(0x26))
        target = make_world(0x00)
        set_player_pos(target.ram, 422, 489)
        for _ in range(4):
            self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        result = task.step(target)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_exit_to_farm_task_routes_to_shed_door_before_exiting(self) -> None:
        world = make_world(0x26)
        set_player_pos(world.ram, 4 * 16 + 8, 11 * 16 + 8)
        world.ram[ADDR_MAP + 11 * 64 + 4] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 5] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 6] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 7] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 8] = 0xA1
        world.ram[ADDR_MAP + 12 * 64 + 8] = 0xA1
        task = ExitToFarmTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(task._task.stand_tile, (8, 12))

    def test_directional_transition_uses_stand_tile_before_holding_direction(self) -> None:
        world = make_transition_world(0x28, current_tile=(8, 11))
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)

    def test_directional_transition_routes_to_nonwalkable_exit_tile(self) -> None:
        ram = make_navigation_ram(current_tile=(2, 1), blocked_tile=(1, 1), blocked_id=0xA6)
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_MAP + 0 * 64 + 0] = 0xC0
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = DirectionalTransitionTask(
            direction="left",
            origin_tilemap=0x00,
            target_tilemap=0x0C,
            stand_tile=(0, 0),
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_directional_transition_can_settle_after_tilemap_change(self) -> None:
        origin = make_transition_world(0x28, current_tile=(8, 12))
        target = make_transition_world(0x00, current_tile=(8, 13))
        set_player_pos(target.ram, 456, 360)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            min_frames_before_success=1,
            settle_frames=3,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        result = task.step(target)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x00")

    def test_directional_transition_waits_for_target_stand_tile(self) -> None:
        origin = make_transition_world(0x00)
        transitional = make_transition_world(0x27)
        landed = make_transition_world(0x27)
        set_player_pos(origin.ram, 20 * 16 + 8, 22 * 16 + 8)
        set_player_pos(transitional.ram, 20 * 16 + 8, 21 * 16 + 8)
        set_player_pos(landed.ram, 8 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
            settle_frames=2,
            target_stand_tile=(8, 22),
            target_stand_tolerance=1,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        waiting = task.step(transitional)
        self.assertEqual(waiting.status, TaskStatus.RUNNING)
        self.assertEqual(waiting.reason, "transition target settle")
        self.assertEqual(int(waiting.action.action[4]), 1)
        self.assertEqual(task.step(landed).status, TaskStatus.RUNNING)
        result = task.step(landed)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x27")

    def test_directional_transition_factory_passes_target_stand_tile(self) -> None:
        spec = PhaseSpec(
            "ENTER_BARN",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x27,
                "target_stand_tile": (8, 22),
                "target_stand_tolerance": 1,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.target_stand_tile, (8, 22))
        self.assertEqual(task.target_stand_tolerance, 1)

    def test_directional_transition_factory_passes_door_entry_params(self) -> None:
        spec = PhaseSpec(
            "ENTER_COOP",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x28,
                "stand_tile": (28, 22),
                "door_align_px": 456,
                "overshoot_limit_px": 330,
                "require_empty_hands": True,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.door_align_px, 456)
        self.assertEqual(task.overshoot_limit_px, 330)
        self.assertTrue(task.require_empty_hands)

    def test_directional_transition_clears_hands_before_door(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_transition_world(0x00, current_tile=(8, 26))
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            require_empty_hands=True,
            door_align_px=136,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "clear hands for door")
        self.assertEqual(task._hands_attempts, 1)
        self.assertTrue(task._action_queue)

    def test_directional_transition_overshoot_restands(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 20))
        set_player_pos(world.ram, 136, 310)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            overshoot_limit_px=328,
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task._stand_reached)
        self.assertEqual(int(result.action.action[5]), 1)

    def test_directional_transition_aligns_door_column_before_push(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 26))
        set_player_pos(world.ram, 120, 424)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[4]), 0)

    def test_nav_crop_factory_uses_dynamic_crop_target(self) -> None:
        spec = PhaseSpec("NAV_CROP", "nav", {"target_px": (136, 520)})

        with patch(
            "harvest.planner.day_phase_registry.crop_nav_target_px",
            return_value=(72, 296),
        ):
            task = DayTaskFactory(state_name="latest").make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, NavTask)
        self.assertEqual((task.target_px.x, task.target_px.y), (72, 296))

    def test_nav_task_soft_arrives_when_stuck_near_target(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 121, 519)
        task = NavTask(
            target_px=Point(136, 520),
            radius=12,
            soft_radius=48,
            soft_stasis=5,
            timeout=200,
        )
        task.reset(world)
        task._navigator.stasis = 10

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soft arrived", result.reason or "")

    def test_directional_transition_accepts_seasonal_farm_origin(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 20 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("expected origin", result.reason or "")

    def test_multi_nav_treats_seasonal_farm_as_farm_waypoint(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 137, 344)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 344), radius=12)],
            initial_settle_frames=0,
        )

        task.reset(world)
        first = task.step(world)
        result = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_directional_transition_does_not_recenter_after_stand_reached(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 12))
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x28,
            stand_tile=(8, 12),
        )

        task.reset(world)
        first = task.step(world)
        self.assertEqual(int(first.action.action[4]), 1)

        set_player_pos(world.ram, 8 * 16 + 8, 12 * 16 + 6)
        second = task.step(world)

        self.assertEqual(int(second.action.action[4]), 1)
        self.assertEqual(int(second.action.action[5]), 0)

    def test_barn_exit_uses_right_aisle_from_upper_stalls(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 163, 153)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[6]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_moves_from_lower_cow_lane_to_door(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 163, 344)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_turns_left_from_right_aisle_crossing(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 201, 329)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_coop_exit_climbs_above_false_open_before_crossing_east(self) -> None:
        world = make_transition_world(0x28)
        set_player_pos(world.ram, 57, 184)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # From left service lane, climb above the false-open band first.
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_coop_exit_climbs_out_of_shipping_bin_alcove(self) -> None:
        world = make_transition_world(0x28)
        set_player_pos(world.ram, 22, 169)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # First regain the left lane at x=38, then climb.
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_uses_bypass_when_right_aisle_is_blocked(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 201, 267)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        task._navigator.current_pos = Point(201, 267)
        task._navigator.stasis = 46
        first = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIsNotNone(first.action)
        self.assertEqual(int(first.action.action[7]), 1)
        self.assertEqual(int(first.action.action[5]), 0)
        self.assertTrue(task._barn_exit_bypass)

        set_player_pos(world.ram, 216, 267)
        second = task.step(world)

        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertIsNotNone(second.action)
        self.assertEqual(int(second.action.action[5]), 1)
        self.assertEqual(int(second.action.action[7]), 0)

    def test_multi_nav_blocks_wrong_tilemap_before_farm_route(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 375), run_direction="down")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x15, current_tile=(8, 7))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("expected tilemap 0x00", result.reason or "")

    def test_multi_nav_run_direction_corrects_off_lane_before_running(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(200, 200), radius=12, run_direction="right")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x00, current_tile=(10, 15))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[7]), 0)

    def test_farm_to_barn_route_starts_south_of_house_exit(self) -> None:
        route = ROUTES["farm_to_barn"]

        self.assertEqual(route[0].target_px, (137, 375))
        self.assertNotIn((137, 300), [waypoint.target_px for waypoint in route])
        self.assertTrue(all(px != 137 or py >= 344 for px, py in [waypoint.target_px for waypoint in route]))

    def test_day_plan_resume_after_hotswap_clears_active_task_state(self) -> None:
        plan = DayPlanTask(phase_sequence=[PhaseSpec("NAV_TO_COOP", "multi_nav", {"route": "farm_to_coop"})])
        world = make_transition_world(0x00, current_tile=(8, 13))
        plan.reset(world)
        plan.step(world)
        task = plan.current_task
        self.assertIsInstance(task, MultiMapNavTask)
        task._action_queue.append(make_action(a=True))
        task._phase = "action_drain"
        task._navigator.path = [(9, 13)]

        plan.resume_after_hotswap(world)

        self.assertEqual(task._phase, "nav")
        self.assertEqual(len(task._action_queue), 0)
        self.assertEqual(task._navigator.path, [])

    def test_barn_to_coop_phase_skips_initial_settle_backtrack(self) -> None:
        plan = DayPlanTask(
            phase_sequence=[
                PhaseSpec(
                    "NAV_TO_COOP",
                    "multi_nav",
                    {"route": "barn_to_coop", "initial_settle_frames": 0},
                )
            ]
        )
        world = make_world(0x00)
        set_player_pos(world.ram, 20 * 16 + 8, 22 * 16 + 8)

        plan.reset(world)
        plan.step(world)

        task = plan.current_task
        self.assertIsInstance(task, MultiMapNavTask)
        self.assertEqual(task.initial_settle_frames, 0)
        self.assertEqual(task.waypoints, ROUTES["barn_to_coop"])

    def test_ensure_carry_tool_is_noop_when_tool_already_held(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x10
        task = EnsureCarryToolTask(tool_id=0x10)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tool 0x10 ready")

    def test_ensure_carry_tool_uses_farmside_shed_approach_target(self) -> None:
        world = make_world(0x00)
        task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "route")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.initial_settle_frames, 0)
        self.assertEqual(task._task.waypoints[0].target_px, (137, 375))
        self.assertEqual(task._task.waypoints[1].target_px, (244, 375))
        self.assertEqual(task._task.waypoints[-1].target_px, (424, 489))

    def test_watering_can_shed_spec_uses_shelf_coordinates(self) -> None:
        spec = SHED_TOOL_SPECS[int(Tool.WATERING_CAN)]

        self.assertEqual(spec.farm_route, "farm_to_shed")
        self.assertEqual(spec.nav_target_px, (422, 474))
        self.assertEqual(spec.inside_stand_px, (96, 168))
        self.assertEqual(spec.inside_face, "up")
        self.assertIsNone(spec.inside_recording)

    def test_animal_tool_shed_specs_use_shelf_coordinates(self) -> None:
        milker = SHED_TOOL_SPECS[int(Tool.MILKER)]
        brush = SHED_TOOL_SPECS[int(Tool.BRUSH)]

        self.assertEqual(int(Tool.MILKER), 0x0E)
        self.assertEqual(milker.inside_stand_px, (64, 168))
        self.assertEqual(milker.inside_face, "up")
        self.assertIsNone(milker.inside_recording)
        self.assertEqual(brush.inside_stand_px, (80, 168))
        self.assertEqual(brush.inside_face, "up")
        self.assertIsNone(brush.inside_recording)

    def test_ensure_carry_tool_uses_dynamic_shed_shelf_task_for_brush(self) -> None:
        world = make_world(0x26)
        task = EnsureCarryToolTask(tool_id=int(Tool.BRUSH))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "inside")
        self.assertIsInstance(task._task, ShedShelfToolTask)
        self.assertEqual(task._task.stand_px, (80, 168))
        self.assertEqual(task._task.radius, 2)

    def test_ensure_animal_tools_is_noop_when_milker_and_brush_carried(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.MILKER)
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "animal tools ready")

    def test_ensure_animal_tools_swaps_selected_target_before_getting_missing_tool(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_animal_tools_gets_brush_first_when_neither_target_carried(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = 0x07
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_0x0F")
        self.assertIsInstance(task._task, EnsureCarryToolTask)
        self.assertEqual(task._task.tool_id, int(Tool.BRUSH))

    def test_ensure_crop_seeds_is_noop_when_seed_and_hoe_held(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x07
        world.ram[0x0923] = int(Tool.HOE)
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "seed tool 0x07 ready")

    def test_ensure_crop_seeds_grabs_hoe_first_when_stock_exists(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x00
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_hoe")
        self.assertIsInstance(task._task, EnsureCarryToolTask)
        self.assertEqual(task._task.tool_id, int(Tool.HOE))

    def test_ensure_crop_seeds_swaps_hoe_off_selected_before_seed_fetch(self) -> None:
        """Shelf A replaces selected; hoe must be backpack before seed grab (rr-6byj)."""
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap_preserve_hoe")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_crop_seeds_swaps_seed_off_selected_before_hoe_fetch(self) -> None:
        """If seeds are already held and selected, preserve them before hoe shelf."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x07  # potato selected
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap_preserve_seed")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_crop_seeds_fetch_when_hoe_in_backpack(self) -> None:
        """Hoe already in backpack → selected is disposable → go straight to seed route."""
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.HOE)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)

    def test_ensure_crop_seeds_caps_shed_trips_against_carry_thrash(self) -> None:
        """Re-fetch loop must fail cleanly instead of multi_nav hang (rr-6byj)."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x00
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato", max_shed_trips=2)
        task.reset(world)
        # Simulate prior trips without completing plant tools.
        task._shed_trips = 2
        result = task._start_next_phase(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("carry thrash", result.reason or "")

    def test_ensure_crop_seeds_uses_tool_shed_route_when_hoe_ready(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        # Hoe selected + empty backpack: no X-swap (empty cannot be selected);
        # open backpack slot absorbs shelf item → go straight to seed route.
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)

    def test_ensure_crop_seeds_uses_field_route_after_harvest_shipping(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 11 * 16 + 8, 30 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["field_to_shed"])

    def test_ensure_crop_seeds_uses_upper_route_from_coop_frontage(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 28 * 16 + 8, 22 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["upper_farm_to_shed"])

    def test_ensure_crop_seeds_uses_field_route_from_lower_farm_edge(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 1 * 16 + 8, 29 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["field_to_shed"])

    def test_shed_routes_stop_below_doorway_for_transition(self) -> None:
        self.assertEqual(ROUTES["farm_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["upper_farm_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["field_to_shed"][-1].target_px, (424, 489))
        self.assertNotIn((422, 478), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertIn((354, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 424), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])

    def test_potato_seed_shed_spec_uses_upper_shelf_stand(self) -> None:
        spec = SHED_SEED_SPECS["potato"]

        self.assertEqual(spec.farm_route, "farm_to_shed")
        self.assertEqual(spec.nav_target_px, (422, 474))
        self.assertEqual(spec.inside_stand_px, (190, 118))
        self.assertIsNone(spec.inside_recording)

    def test_shed_fetch_exits_shed_when_item_missing_after_shelf(self) -> None:
        class DoneTask:
            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        world = make_world(0x26)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.BRUSH)
        task = ShedFetchItemTask(
            item_id=0x07,
            shelf=SHED_SEED_SPECS["potato"],
        )
        task._phase = "inside"
        task._task = DoneTask()

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_after_failure")
        self.assertIsInstance(task._task, ExitToFarmTask)

    def test_shed_fetch_fails_after_shed_exit_without_item(self) -> None:
        class DoneTask:
            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.BRUSH)
        task = ShedFetchItemTask(item_id=0x07, shelf=SHED_SEED_SPECS["potato"])
        task._phase = "exit_after_failure"
        task._task = DoneTask()
        task._failed_reason = "tool 0x07 missing after shed task"

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(result.reason, "tool 0x07 missing after shed task")

    def test_ensure_carry_tool_exits_shed_when_already_holding_tool(self) -> None:
        """Watering-can ensure must leave the shed so NAV_CROP can run outdoors."""
        world = make_world(0x26)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = 0x00
        task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_after_ready")
        self.assertIsInstance(task._task, ExitToFarmTask)

    def test_ensure_crop_seeds_does_not_restore_watering_can(self) -> None:
        """Seeds must stay in the carry pair for the following plant phase."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x07
        world.ram[0x0923] = int(Tool.HOE)
        task = EnsureCropSeedsTask(seed_type="potato")
        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotEqual(getattr(task, "_phase", ""), "restore_watering_can")
        self.assertEqual(int(world.ram[0x0921]), 0x07)


class BuildDayPhasesTests(unittest.TestCase):
    """Tests for the dynamic day plan builder."""

    def _phase_names(self, phases):
        return [p.phase for p in phases]

    def test_minimal_day_has_only_exit(self) -> None:
        phases = build_day_phases(None, hour=16)
        self.assertEqual(self._phase_names(phases), ["EXIT_TO_FARM"])

    def test_chickens_come_first(self) -> None:
        phases = build_day_phases(None, has_chickens=True)
        names = self._phase_names(phases)
        chicken_names = [p.phase for p in CHICKEN_PHASES]
        chicken_start = names.index("NAV_TO_COOP")
        self.assertEqual(names[chicken_start:chicken_start + len(chicken_names)], chicken_names)

    def test_oversupplied_chickens_schedule_sale_and_limited_chores(self) -> None:
        phases = build_day_phases(None, hour=6, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertLess(names.index("SELL_CHICKEN"), names.index("NAV_TO_COOP"))
        coop_phase = phases[names.index("COOP_CHORES")]
        self.assertEqual(coop_phase.params["egg_mode"], "ship")
        self.assertEqual(coop_phase.params["max_feed_adults"], 2)

    def test_chicken_sale_skips_after_cutoff(self) -> None:
        phases = build_day_phases(None, hour=14, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertNotIn("SELL_CHICKEN", names)
        self.assertIn("COOP_CHORES", names)

    def test_chicken_sale_skips_sunday(self) -> None:
        phases = build_day_phases(None, weekday=0, hour=6, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertNotIn("SELL_CHICKEN", names)
        self.assertIn("COOP_CHORES", names)

    def test_cows_run_before_chickens_when_both_need_chores(self) -> None:
        phases = build_day_phases(None, has_chickens=True, has_cows=True)
        names = self._phase_names(phases)
        chicken_names = [p.phase for p in CHICKEN_PHASES]
        cow_names = [p.phase for p in COW_PHASES]
        cow_start = names.index("ENSURE_ANIMAL_TOOLS")
        chicken_start = names.index("NAV_TO_COOP")

        self.assertEqual(names[cow_start:cow_start + len(cow_names)], cow_names)
        self.assertEqual(names[chicken_start:chicken_start + len(chicken_names)], chicken_names)
        self.assertEqual(phases[chicken_start].params["route"], "barn_to_coop")
        self.assertLess(cow_start, chicken_start)
        self.assertEqual(phases[chicken_start].params["initial_settle_frames"], 0)

    def test_affordable_first_cow_purchase_runs_before_chickens(self) -> None:
        phases = build_day_phases(
            None,
            weekday=2,
            hour=6,
            has_chickens=True,
            has_harvest=True,
            has_seeds=True,
            should_buy_cow=True,
        )
        names = self._phase_names(phases)

        self.assertLess(names.index("BUY_COW_VENDOR"), names.index("NAV_TO_COOP"))
        self.assertLess(names.index("EXIT_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("NAV_TO_COOP")].params["route"], "barn_to_coop")
        self.assertEqual(names[0], "EXIT_TO_FARM")

    def test_harvest_before_water(self) -> None:
        phases = build_day_phases(None, hour=16, has_harvest=True, has_waterable=True)
        names = self._phase_names(phases)
        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        harvest_idx = names.index("HARVEST_ROUTE")
        water_idx = names.index("CROP_WATER")
        self.assertLess(harvest_idx, water_idx)

    def test_harvest_and_water_share_nav_crop(self) -> None:
        phases = build_day_phases(None, hour=16, has_harvest=True, has_waterable=True)
        names = self._phase_names(phases)
        # NAV_CROP should appear once (harvest needs it, water reuses position)
        self.assertEqual(names.count("NAV_CROP"), 1)

    def test_water_only_gets_nav_crop(self) -> None:
        phases = build_day_phases(None, hour=16, has_waterable=True)
        names = self._phase_names(phases)
        self.assertIn("NAV_CROP", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)

    def test_morning_plant_day_orders_establish_ensure_can_then_water(self) -> None:
        """A3: after hoe+seeds plant, re-fetch can before same-day water.

        Carry holds only two tools. Establish uses hoe+seeds; seed bag frees a
        slot once spent, so ENSURE_WATERING_CAN must sit between establish and
        water (no RAM can poke).
        """
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            has_seeds=True,
            has_waterable=False,
            has_harvest=False,
            is_rainy=False,
        )
        names = self._phase_names(phases)

        self.assertIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("CROP_ESTABLISH", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)

        establish_idx = names.index("CROP_ESTABLISH")
        ensure_can_idx = names.index("ENSURE_WATERING_CAN")
        water_idx = names.index("CROP_WATER")
        self.assertLess(names.index("ENSURE_CROP_SEEDS"), establish_idx)
        self.assertLess(establish_idx, ensure_can_idx)
        self.assertLess(ensure_can_idx, water_idx)

        # Contiguous plant→water crop block (nav may sit between ensure can and water).
        crop_block = names[establish_idx : water_idx + 1]
        self.assertEqual(crop_block[0], "CROP_ESTABLISH")
        self.assertEqual(crop_block[1], "ENSURE_WATERING_CAN")
        self.assertEqual(crop_block[-1], "CROP_WATER")
        self.assertNotIn("CROP_ESTABLISH", crop_block[1:])

        water = phases[water_idx]
        self.assertEqual(water.params.get("work_mode"), "water")
        self.assertEqual(water.params.get("refill_bounds"), (3, 10, 62, 60))

        establish = phases[establish_idx]
        self.assertEqual(establish.params.get("work_mode"), "establish")

    def test_plant_with_existing_waterable_still_re_ensures_can_after_establish(self) -> None:
        """Plant pass still steals carry slots; water pass must re-ensure can."""
        phases = build_day_phases(
            None,
            weekday=3,
            hour=8,
            has_seeds=True,
            has_waterable=True,
            has_harvest=False,
            is_rainy=False,
        )
        names = self._phase_names(phases)
        establish_idx = names.index("CROP_ESTABLISH")
        ensure_can_idx = names.index("ENSURE_WATERING_CAN")
        water_idx = names.index("CROP_WATER")
        self.assertLess(establish_idx, ensure_can_idx)
        self.assertLess(ensure_can_idx, water_idx)
        # Only one ensure-can in the crop block (always before water, after plant).
        self.assertEqual(names.count("ENSURE_WATERING_CAN"), 1)

    def test_water_phase_scheduled_for_dry_plots_regardless_of_can_fill(self) -> None:
        """Empty can must not drop CROP_WATER — refill lives inside CropWaterTask."""
        phases = build_day_phases(
            None,
            hour=10,
            has_seeds=False,
            has_waterable=True,
            is_rainy=False,
        )
        names = self._phase_names(phases)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)
        water = phases[names.index("CROP_WATER")]
        self.assertEqual(water.params.get("work_mode"), "water")
        self.assertEqual(water.params.get("refill_bounds"), (3, 10, 62, 60))

    def test_rainy_day_skips_water_only_crop_phase(self) -> None:
        phases = build_day_phases(None, hour=16, has_waterable=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertNotIn("ENSURE_WATERING_CAN", names)
        self.assertNotIn("CROP_WATER", names)

    def test_rainy_day_can_still_start_seed_crop_phase_without_watering_can(self) -> None:
        phases = build_day_phases(None, hour=16, has_seeds=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("NAV_CROP", names)
        self.assertNotIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_ESTABLISH", names)
        self.assertNotIn("CROP_WATER", names)

    def test_seeded_crop_phase_checks_tool_shed_before_field(self) -> None:
        phases = build_day_phases(None, hour=16, has_seeds=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertLess(names.index("ENSURE_CROP_SEEDS"), names.index("NAV_CROP"))
        self.assertLess(names.index("NAV_CROP"), names.index("CROP_ESTABLISH"))

    def test_rainy_weather_ignores_forecast_weather_field(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEATHER + 0x4000] = 1

        self.assertFalse(is_rainy_weather(world.ram))
        self.assertFalse(crop_task_is_rainy_weather(world.ram))

    def test_rainy_weather_reads_current_weather_flags(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=16)
        world.ram[ADDR_WEATHER + 0x4000] = 0
        world.ram[ADDR_WEATHER_FLAGS] = 0x02

        self.assertTrue(is_rainy_weather(world.ram))
        self.assertTrue(crop_task_is_rainy_weather(world.ram))

    def test_berries_disabled_by_default_on_sunday_with_seeds(self) -> None:
        phases = build_day_phases(None, weekday=0, hour=8, has_seeds=True)
        names = self._phase_names(phases)
        self.assertNotIn("BERRY_RUN_WINDOW", names)
        self.assertNotIn("EXIT_FARM_WEST", names)
        self.assertNotIn("GET_BERRIES_AND_SHIP", names)
        self.assertNotIn("BUY_SEEDS", names)

    def test_berries_can_be_enabled_by_policy_on_sunday_with_seeds(self) -> None:
        phases = build_day_phases(
            None,
            weekday=0,
            hour=8,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=True),
        )
        names = self._phase_names(phases)
        self.assertIn("BERRY_RUN_WINDOW", names)
        self.assertIn("EXIT_FARM_WEST", names)
        self.assertLess(names.index("EXIT_FARM_WEST"), names.index("BERRY_RECORDING_WINDOW"))
        self.assertLess(names.index("BERRY_RECORDING_WINDOW"), names.index("GET_BERRIES_AND_SHIP"))
        self.assertIn("GET_BERRIES_AND_SHIP", names)
        self.assertNotIn("BUY_SEEDS", names)
        self.assertEqual(phases[names.index("BERRY_RUN_WINDOW")].params["latest_hour"], 14)
        self.assertEqual(phases[names.index("BERRY_RECORDING_WINDOW")].params["latest_hour"], 15)

    def test_berries_skipped_when_late(self) -> None:
        phases = build_day_phases(None, weekday=0, hour=16, has_seeds=True)
        names = self._phase_names(phases)
        self.assertNotIn("GET_BERRIES_AND_SHIP", names)

    def test_weekday_morning_no_seeds_buys_seeds(self) -> None:
        phases = build_day_phases(None, weekday=3, hour=6, has_seeds=False)
        names = self._phase_names(phases)
        self.assertIn("BUY_SEEDS_WINDOW", names)
        self.assertIn("NAV_FARM_EXIT", names)
        self.assertIn("BUY_SEEDS", names)

    def test_summer_morning_buys_summer_seed_recording(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            season=1,
            day=7,
            has_seeds=False,
        )
        names = self._phase_names(phases)
        self.assertIn("BUY_SEEDS", names)
        buy = phases[names.index("BUY_SEEDS")]
        self.assertEqual(buy.params["recording_name"], "buy_summer")
        self.assertEqual(buy.params["recording_start"], 0)

    def test_fall_morning_skips_seed_shop_and_planting(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            season=2,
            day=5,
            has_seeds=True,
            has_waterable=False,
            has_harvest=False,
        )
        names = self._phase_names(phases)
        self.assertNotIn("BUY_SEEDS", names)
        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertNotIn("CROP_WATER", names)

    def test_winter_policy_disables_planting(self) -> None:
        from harvest.planner.day_phase_types import day_planner_policy_for_season

        policy = day_planner_policy_for_season(3)
        self.assertFalse(policy.include_planting)
        self.assertIsNone(policy.seed_purchase_recording)

    def test_seed_buy_runs_before_field_clear(self) -> None:
        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            has_debris=True,
        )
        names = [phase.phase for phase in phases]
        self.assertLess(names.index("BUY_SEEDS"), names.index("CLEAR_FIELD"))
        self.assertLess(names.index("BUY_SEEDS_WINDOW"), names.index("CLEAR_FIELD"))

    def test_full_day_priority_order(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=8,
            has_chickens=True,
            has_cows=True,
            has_harvest=True,
            has_waterable=True,
            has_seeds=True,
        )
        names = self._phase_names(phases)
        # Verify ordering: exit -> cows -> chickens -> harvest -> crop work.
        exit_idx = names.index("EXIT_TO_FARM")
        cow_idx = names.index("NAV_TO_BARN")
        coop_idx = names.index("NAV_TO_COOP")
        harvest_idx = names.index("HARVEST_ROUTE")
        water_idx = names.index("CROP_WATER")
        self.assertLess(exit_idx, cow_idx)
        self.assertLess(cow_idx, coop_idx)
        self.assertLess(coop_idx, harvest_idx)
        self.assertLess(harvest_idx, water_idx)
        self.assertIn("CROP_WATER", names)
        self.assertNotIn("GET_BERRIES_AND_SHIP", names)

    def test_early_berry_window_runs_after_crop_work_when_enabled(self) -> None:
        phases = build_day_phases(
            None,
            weekday=6,
            hour=6,
            has_chickens=True,
            has_harvest=True,
            has_waterable=True,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=True),
        )
        names = self._phase_names(phases)

        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("GET_BERRIES_AND_SHIP", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("CROP_WATER"), names.index("GET_BERRIES_AND_SHIP"))

    def test_auto_day_phases_uses_ram_priority_instead_of_resume_water_shortcut(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        names = self._phase_names(auto_day_phases("fake", ram=world.ram))

        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        self.assertNotIn("GET_BERRIES_AND_SHIP", names)

    def test_eve_loop_target_uses_ten_heart_threshold(self) -> None:
        self.assertEqual(romance_points_for_hearts(10), 999)
        self.assertEqual(romance_points_for_hearts(1), 49)

    def test_state_has_chickens_detects_chicken_count(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_CHICKEN_COUNT] = 2
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertTrue(state_has_chickens("fake"))

    def test_state_has_chickens_false_when_zero(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertFalse(state_has_chickens("fake"))

    def test_state_has_cows_detects_cow_count(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_COW_COUNT] = 1
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertTrue(state_has_cows("fake"))

    def test_chicken_phases_structure(self) -> None:
        names = [p.phase for p in CHICKEN_PHASES]
        kinds = [p.kind for p in CHICKEN_PHASES]
        self.assertEqual(names, ["NAV_TO_COOP", "ENTER_COOP", "COOP_CHORES", "EXIT_COOP"])
        self.assertEqual(kinds, ["multi_nav", "directional_transition", "coop_chores", "directional_transition"])
        self.assertEqual(CHICKEN_PHASES[-1].params["stand_tile"], (8, 12))
        self.assertEqual(CHICKEN_PHASES[-1].params["settle_frames"], 5)
        self.assertEqual(len(ROUTES["barn_to_coop"]), 1)
        self.assertEqual(ROUTES["barn_to_coop"][0].target_px, (454, 360))
        self.assertIsNone(ROUTES["barn_to_coop"][0].run_direction)

    def test_cow_phases_structure(self) -> None:
        names = [p.phase for p in COW_PHASES]
        kinds = [p.kind for p in COW_PHASES]
        self.assertEqual(names, ["ENSURE_ANIMAL_TOOLS", "EXIT_TO_FARM", "NAV_TO_BARN", "ENTER_BARN", "COW_CHORES", "EXIT_BARN"])
        self.assertEqual(kinds, ["ensure_animal_tools", "farm_building_exit", "multi_nav", "directional_transition", "cow_chores", "directional_transition"])
        self.assertEqual(COW_PHASES[2].params["route"], "farm_to_barn")
        self.assertTrue(COW_PHASES[4].params.get("milk", True))
        self.assertTrue(COW_PHASES[4].params.get("feed", True))
        self.assertTrue(COW_PHASES[4].params.get("talk", True))
        self.assertTrue(COW_PHASES[4].params.get("brush", True))
        self.assertEqual(COW_PHASES[-1].params["stand_tile"], (8, 22))

    def test_build_day_phases_from_ram_defers_outdoor_work_until_after_exit(self) -> None:
        world = make_date_world(0x15, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "NAV_TO_COOP",
                "ENTER_COOP",
                "COOP_CHORES",
                "EXIT_COOP",
                "DYNAMIC_OUTDOOR_PLAN",
            ],
        )

    def test_build_day_phases_from_ram_sells_when_adult_chickens_exceed_target(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 6
        world.ram[ADDR_HAY_COUNT + 0x4000] = 50
        set_live_chicken_slots(world.ram, adults=6)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("SELL_CHICKEN", names)
        self.assertLess(names.index("SELL_CHICKEN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("COOP_CHORES")].params["max_feed_adults"], 2)

    def test_next_morning_ram_plan_runs_cow_then_coop_chores(self) -> None:
        world = make_date_world(0x15, season=0, day=14, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20
        set_live_cow_slot(world.ram, 0)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertLess(names.index("NAV_TO_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("NAV_TO_COOP")].params["route"], "barn_to_coop")
        self.assertIn("COOP_CHORES", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_includes_cow_chores_when_unfed(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("NAV_TO_BARN", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_includes_cow_chores_when_brush_missing(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        set_live_cow_slot(world.ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("NAV_TO_BARN", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_starts_cows_in_barn_when_tools_ready(self) -> None:
        world = make_date_world(0x27, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20
        world.ram[field_spec("tool_selected").address + 0x4000] = int(Tool.BRUSH)
        world.ram[field_spec("tool_backpack").address + 0x4000] = int(Tool.MILKER)
        set_live_cow_slot(world.ram, 0)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(
            self._phase_names(phases)[:3],
            ["COW_CHORES", "EXIT_BARN", "DYNAMIC_OUTDOOR_PLAN"],
        )
        self.assertNotIn("NAV_TO_BARN", self._phase_names(phases))

    def test_build_day_phases_from_ram_starts_chickens_in_coop(self) -> None:
        world = make_date_world(COOP_TILEMAP, season=0, day=13, hour=6)
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 4
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 0
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertEqual(names[:2], ["COOP_CHORES", "EXIT_COOP"])
        self.assertNotIn("NAV_TO_COOP", names)
        self.assertLess(names.index("EXIT_COOP"), names.index("NAV_TO_BARN"))
        self.assertEqual(names[-1], "DYNAMIC_OUTDOOR_PLAN")

    def test_build_day_phases_from_ram_uses_coop_exit_when_coop_chores_done(self) -> None:
        world = make_date_world(COOP_TILEMAP, season=0, day=13, hour=6)
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 0

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(self._phase_names(phases), ["EXIT_COOP", "DYNAMIC_OUTDOOR_PLAN"])

    def test_build_day_phases_from_ram_buys_first_cow_before_coop(self) -> None:
        world = make_date_world(0x15, season=0, day=16, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        set_money(world.ram, COW_PURCHASE_COST)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertLess(names.index("BUY_COW_VENDOR"), names.index("NAV_TO_COOP"))
        self.assertLess(names.index("EXIT_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(names[-1], "DYNAMIC_OUTDOOR_PLAN")

    def test_build_day_phases_from_ram_skips_cow_purchase_when_already_owned(self) -> None:
        world = make_date_world(0x15, season=0, day=16, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        set_money(world.ram, COW_PURCHASE_COST)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("BUY_COW_VENDOR", names)

    def test_late_day_ram_plan_skips_done_coop_waters_then_sleeps(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 0
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("NAV_TO_COOP", names)
        self.assertNotIn("NAV_TO_BARN", names)
        self.assertNotIn("HARVEST_ROUTE", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("CROP_WATER", names)
        self.assertEqual(names[-2:], ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_late_day_forecast_rain_still_waters_dry_crops(self) -> None:
        world = make_date_world(0x00, season=0, day=29, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 1
        world.ram[ADDR_WEATHER + 0x4000] = 1
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x5A

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("CROP_WATER"), names.index("RETURN_HOME"))

    def test_late_day_seed_stock_only_goes_home_to_sleep(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertNotIn("CROP_WATER", names)
        self.assertEqual(names, ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_late_day_house_state_goes_directly_to_sleep(self) -> None:
        world = make_date_world(0x15, season=0, day=14, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["GO_TO_SLEEP"])

    def test_late_day_remodeled_house_state_goes_directly_to_sleep(self) -> None:
        world = make_date_world(0x16, season=0, day=14, hour=18)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["GO_TO_SLEEP"])

    def test_waterable_detection_skips_known_harvestable_tiles(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        with patch("harvest.planner.day_plan_status.live_harvestable_crop_tiles", return_value=[(7, 34)]):
            self.assertFalse(ram_has_waterable_crops(world.ram, state_name="fake"))

    def test_waterable_detection_ignores_unplanted_and_mature_tiles(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x07
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 36 * 64 + 7] = 0x61

        self.assertFalse(ram_has_waterable_crops(world.ram))

    def test_return_home_enters_when_already_at_house_front(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 136, 424)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)
        self.assertEqual(task._task.stand_tile, (8, 26))
        self.assertEqual(task._task.overshoot_limit_px, 328)
        self.assertTrue(task._task.require_empty_hands)

    def test_return_home_timeout_fails_cleanly(self) -> None:
        """Outer budget prevents multi-day hang when enter/nav never terminates."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        # Child nav keeps RUNNING; outer timeout must still fire.
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="stuck nav")
        )
        task._phase = "nav_house_front"

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")

    def test_return_home_renavs_when_stuck_north_of_door_stand(self) -> None:
        """Mid-door tiles (~y=389) must walk south before pushing up."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 137, 389)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, NavTask)
        self.assertEqual(
            (task._task.target_px.x, task._task.target_px.y),
            (136, 424),
        )

    def test_return_home_enter_uses_catalog_stand_not_overshoot_tile(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 136, 424)
        task = ReturnHomeTask()
        enter = task._house_enter_task(world)

        self.assertEqual(enter.stand_tile, (8, 26))
        self.assertEqual(enter.door_align_px, 136)
        self.assertEqual(enter.overshoot_limit_px, 328)

    def test_return_home_navs_when_far_from_house_front(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 424))

    def test_return_home_uses_remodeled_house_waypoint_from_upgrade_flags(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[field_spec("upgrade_flags").address + 0x4000] = 0x40
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 344))

    def test_return_home_enter_house_accepts_remodel_tilemap(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        task = ReturnHomeTask()
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(step=lambda _world: TaskResult(status=TaskStatus.SUCCESS, reason="arrived"))

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)
        self.assertIn(0x16, task._task.target_tilemaps)
        self.assertEqual(task._task.stand_tile, (8, 26))
        self.assertEqual(task._task.door_align_px, 136)
        self.assertTrue(task._task.require_empty_hands)

    def test_day_plan_expands_dynamic_outdoor_phase_from_live_farm_state(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        plan = DayPlanTask(phase_sequence=[PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")])
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "NAV_CROP")
        self.assertEqual([phase.phase for phase in plan.phases], ["DYNAMIC_OUTDOOR_PLAN"])
        self.assertNotEqual([phase.phase for phase in plan.runtime_phases], ["DYNAMIC_OUTDOOR_PLAN"])

    def test_day_plan_expands_dynamic_outdoor_phase_from_seasonal_farm_state(self) -> None:
        world = make_date_world(0x01, season=1, day=1, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 3
        # Summer needs corn/tomato stock; leftover potato seeds are ignored.
        world.ram[ADDR_CORN_SEEDS + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        plan = DayPlanTask(phase_sequence=[PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")])
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotEqual(plan.phase_text, "EXIT_TO_FARM")
        self.assertEqual(plan.phase_text, "NAV_CROP")
        self.assertEqual([phase.phase for phase in plan.phases], ["DYNAMIC_OUTDOOR_PLAN"])

    def test_dynamic_outdoor_phase_uses_state_harvest_fallback(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 3
        world.ram[0x092A + 0x4000] = 5

        with patch("harvest.planner.world_probe.ram_has_harvestable_crops", return_value=False), patch(
            "harvest.planner.world_probe.state_has_harvestable_crops",
            return_value=True,
        ):
            phases = build_outdoor_day_phases_from_ram(world.ram, state_name="pinned_fixture")

        names = self._phase_names(phases)
        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("HARVEST_ROUTE"), names.index("CROP_WATER"))


class SleepAndPlannerTests(unittest.TestCase):
    def test_go_to_sleep_succeeds_once_date_advances(self) -> None:
        task = GoToSleepTask(morning_ready_frames=3)
        start = make_date_world(0x15, season=0, day=12)
        task.reset(start)

        advanced = make_date_world(0x15, season=0, day=13)
        # Morning settle: wait for house-ready frames after day roll (Gate B).
        result = None
        for _ in range(5):
            result = task.step(advanced)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("day advanced", result.reason or "")

    def test_go_to_sleep_starts_return_home_when_outside_house(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x00, season=0, day=12)
        set_player_pos(world.ram, 200, 500)
        task.reset(world)

        self.assertEqual(task._phase, "ensure_house")
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_house")
        self.assertIsNotNone(task._return_home)

    def test_go_to_sleep_succeeds_on_ending_credits(self) -> None:
        task = GoToSleepTask()
        start = make_date_world(0x15, season=3, day=30)
        task.reset(start)

        ending = make_date_world(0x15, season=3, day=30)
        set_live_u16(ending.ram, "ending_scene_index", 0x04)
        result = task.step(ending)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "ending reached")

    def test_go_to_sleep_dismisses_during_sleep_transition(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x0F, season=0, day=12)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "sleep transition")
        self.assertIsNotNone(result.action)

    def test_go_to_sleep_routes_stuck_left_wall_back_to_lower_lane(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12)
        set_player_pos(world.ram, 77, 182)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(action[5], 1)  # down
        self.assertEqual(action[0], 1)  # run
        self.assertEqual(action[4], 0)  # not up into the bed/wall edge
        self.assertEqual(action[6], 0)  # not left into the bed/wall edge

    def test_go_to_sleep_requires_tight_bed_column_before_interacting(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12)
        set_player_pos(world.ram, 75, 86)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[6], 1)  # left toward the proven x=70 stand point
        self.assertEqual(action[8], 0)  # do not press A from the loose column

    def test_go_to_sleep_confirmation_faces_up_then_plain_a(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        # Face up at the bed stand, then A alone (not up+A).
        self.assertTrue(any(action[4] == 1 and action[8] == 0 for action in task._action_queue))
        self.assertTrue(
            any(action[8] == 1 and action[4] == 0 for action in task._action_queue)
        )
        self.assertFalse(
            any(action[8] == 1 and action[4] == 1 for action in task._action_queue)
        )
        # Recording taps B before the sleep confirmation.
        self.assertTrue(any(action[0] == 1 for action in task._action_queue))
        # Never face-left + A (walks into mattress / misses prompt — rr-m0wq).
        self.assertFalse(
            any(action[8] == 1 and action[6] == 1 for action in task._action_queue)
        )

    def test_go_to_sleep_later_attempts_stay_face_up_no_post_a_b(self) -> None:
        """rr-m0wq: left-face A and B-after-A cancel Yes; stay face-up + A-only."""
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)
        task._sleep_attempts = 6
        task._phase = "sleep_attempt"
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        queue = list(task._action_queue)
        # Still face up somewhere in the attempt.
        self.assertTrue(any(action[4] == 1 for action in queue))
        # No left+A combo on the confirm press.
        self.assertFalse(any(action[8] == 1 and action[6] == 1 for action in queue))
        # After the first A appears, no B should follow (B = No on sleep confirm).
        first_a = next(i for i, a in enumerate(queue) if a[8] == 1)
        self.assertFalse(any(a[0] == 1 for a in queue[first_a:]))

    def test_go_to_sleep_level2_uses_recorded_wife_bed_position(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6)
        set_player_pos(world.ram, 294, 102)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        # Face up toward the wife bed, then A alone (not up+A).
        self.assertTrue(any(action[4] == 1 and action[8] == 0 for action in task._action_queue))
        self.assertTrue(any(action[8] == 1 and action[4] == 0 for action in task._action_queue))
        self.assertFalse(
            any(action[8] == 1 and action[4] == 1 for action in task._action_queue)
        )

    def test_go_to_sleep_level2_does_not_use_base_bed_position(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[8], 0)
        self.assertEqual(action[6], 1)

    def test_go_to_sleep_level2_routes_entry_toward_recorded_bed_lane(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6)
        set_player_pos(world.ram, 136, 330)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[4], 1)
        self.assertEqual(action[0], 1)

    def test_multi_day_planner_rebuilds_day_task_after_sleep(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class WaitForNextDayTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                day = int(world.ram[ADDR_DAY + 0x4000])
                if day >= 2:
                    return TaskResult(status=TaskStatus.SUCCESS, reason="slept")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(until_season=0, until_day=2, max_days=3)
                self.day_builds = 0

            def _build_day_task(self, world):
                self.day_builds += 1
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return WaitForNextDayTask()

        planner = TestPlanner()
        planner.morning_settle_frames = 2
        day1 = make_date_world(0x15, season=0, day=1)
        planner.reset(day1)

        planner.step(day1)
        day2 = make_date_world(0x15, season=0, day=2)
        for _ in range(16):
            planner.step(day2)
            if planner.day_builds >= 2 and planner.phase_text == "PLAN_DAY":
                break

        self.assertEqual(planner.day_builds, 2)
        self.assertEqual(planner.phase_text, "PLAN_DAY")

    def test_multi_day_planner_settles_morning_before_rebuild(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class AdvancingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                return TaskResult(status=TaskStatus.SUCCESS, reason="day advanced")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=3, max_days=3, morning_settle_frames=3)
                self.day_builds = 0

            def _build_day_task(self, world):
                self.day_builds += 1
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return AdvancingSleepTask()

        planner = TestPlanner()
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        # Drive through first day plan → return_home → sleep.
        for _ in range(6):
            planner.step(world)
            if planner.phase_text == "SETTLE_MORNING":
                break

        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        builds_before = planner.day_builds

        for _ in range(3):
            planner.step(world)

        self.assertEqual(planner.phase_text, "PLAN_DAY")
        self.assertEqual(planner.day_builds, builds_before)

    def test_multi_day_planner_stops_on_ending_after_sleep(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class EndingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                set_live_u16(world.ram, "ending_scene_index", 0x25)
                return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")

        class TestPlanner(MultiDayPlannerTask):
            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return EndingSleepTask()

        planner = TestPlanner(target_days=10, max_days=10)
        world = make_date_world(0x15, season=3, day=30)
        planner.reset(world)

        result = TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        for _ in range(8):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "ending reached")

    def test_multi_day_planner_detects_overnight_while_return_home_running(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class OvernightWhileRunningTask:
            def __init__(self) -> None:
                self.steps = 0

            def reset(self, world) -> None:
                self.steps = 0

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                self.steps += 1
                if self.steps >= 2:
                    world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                    world.ram[ADDR_TILEMAP] = 0x0F
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="still exiting",
                )

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=1, max_days=1, morning_settle_frames=1)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return OvernightWhileRunningTask()

        planner = TestPlanner()
        world = make_date_world(0x05, season=1, day=10)
        planner.reset(world)
        planner.step(world)  # plan_day success → return_home

        result = planner.step(world)  # return_home running day 10
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "RETURN_HOME")

        result = planner.step(world)  # overnight while RUNNING → settle morning
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner.day_failures[0]["phase"], "return_home")

        # Fake overnight leaves sleep-transition tilemap; land in morning house.
        world.ram[ADDR_TILEMAP] = 0x15
        set_player_pos(world.ram, 136, 120)
        world.ram[ADDR_HOUR + 0x4000] = 6
        for _ in range(4):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")

    def test_multi_day_planner_forces_sleep_route_when_day_plan_blocks(self) -> None:
        class BlockedDayTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="required phase EXIT_TO_FARM failed")

        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class TestPlanner(MultiDayPlannerTask):
            def _build_day_task(self, world):
                return BlockedDayTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

        planner = TestPlanner(until_season=0, until_day=2, max_days=3)
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        result = planner.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "RETURN_HOME")
        self.assertEqual(len(planner.day_failures), 1)
        self.assertEqual(planner.day_failures[0]["reason"], "required phase EXIT_TO_FARM failed")

        result = planner.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SLEEP")

    def test_multi_day_planner_days_target_completes_after_requested_sleeps(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class AdvancingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                return TaskResult(status=TaskStatus.SUCCESS, reason="slept")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=2, max_days=2)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return AdvancingSleepTask()

        planner = TestPlanner()
        planner.morning_settle_frames = 1
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        result = TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        for _ in range(32):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")
        # After two overnights from day 1, morning is day 3; settle updates active day.
        self.assertEqual(planner.progress_text, "date=0:3 completed=2/2")

    def test_multi_day_journal_keeps_planning_and_runtime_deferrals(self) -> None:
        planner = MultiDayPlannerTask(target_days=1, max_days=1)
        world = make_date_world(0x15, season=0, day=4)
        planner.reset(world)
        planner._last_day_decision = DayPlanDecision(
            phases=(),
            facts=PlanningFacts(source="test", weekday=1, hour=6),
            deferred=(
                DeferredPlan(
                    phase="CROP_WATER",
                    kind="crop",
                    reason="rainy_day",
                    params={"work_mode": "water"},
                ),
            ),
        )
        # The runtime can report the same deferral as the initial plan plus a
        # later route miss.  Preserve decision metadata and do not duplicate it.
        planner._last_day_deferred = [
            {"phase": "CROP_WATER", "reason": "rainy_day", "retry": "tomorrow"},
            {"phase": "BUY_SEEDS", "reason": "seed_shop_cutoff", "retry": "tomorrow"},
        ]

        planner._journal_day_complete(world, sleep_reason="test")

        deferred = planner.day_journal[-1]["deferred"]
        self.assertEqual([item["phase"] for item in deferred], ["CROP_WATER", "BUY_SEEDS"])
        self.assertEqual(deferred[0]["params"], {"work_mode": "water"})

    def test_multi_day_planner_counts_event_transition_during_return_home(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class EventTransitionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                world.ram[ADDR_TILEMAP] = 0x0F
                return TaskResult(status=TaskStatus.FAILURE, reason="expected tilemap 0x05, got 0x0F")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=1, max_days=1, morning_settle_frames=1)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return EventTransitionTask()

        planner = TestPlanner()
        world = make_date_world(0x05, season=1, day=10)
        planner.reset(world)
        planner.step(world)

        result = planner.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner.day_failures[0]["phase"], "return_home")

        world.ram[ADDR_TILEMAP] = 0x15
        set_player_pos(world.ram, 136, 120)
        world.ram[ADDR_HOUR + 0x4000] = 6
        for _ in range(4):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")
        self.assertEqual(planner.progress_text, "date=1:11 completed=1/1")


if __name__ == "__main__":
    unittest.main()
