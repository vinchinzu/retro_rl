"""Return-home / late-day sleep day-plan tests.

Split from test_day_plan_home monofile (return-home behavior + late-day builder).
Exit-to-farm / house-exit sequences live in ``test_day_plan_home_sequences``.
Discovered via pytest; also aggregated by ``test_day_plan_home`` shim.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import (
    DayPlanPhaseHelpers,
    make_date_world,
    set_player_pos,
)

import unittest
from types import SimpleNamespace

from harvest.planner.day_plan import (
    ADDR_CHICKEN_COUNT,
    ADDR_EGG_AVAILABLE,
    ADDR_WEEKDAY,
    DirectionalTransitionTask,
    MultiMapNavTask,
    NavTask,
    ReturnHomeTask,
    TaskResult,
    TaskStatus,
    build_day_phases_from_ram,
)
from harvest.core.tile_catalog import ADDR_MAP
from harvest.core.ram_catalog import field_spec


class BuildDayPhasesHomeTests(DayPlanPhaseHelpers):
    """Tests for the dynamic day plan builder."""

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

    def test_return_home_succeeds_when_already_house_mid_exit_to_farm(self) -> None:
        """rr-ws8h: house tilemap short-circuits even if phase is exit_to_farm."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertIn("exit_to_farm", result.reason or "")

    def test_return_home_timeout_succeeds_when_already_house(self) -> None:
        """rr-ws8h: hard timeout must not FAIL if player is already inside."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )
        # Past timeout with house tilemap — short-circuit wins (via=step).
        task._total_steps = 5

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertNotIn("timeout", result.reason or "")

    def test_return_home_remodel_tilemap_short_circuits(self) -> None:
        """Remodeled house tilemaps (0x16/0x17) also count as arrival."""
        world = make_date_world(0x16, season=0, day=10, hour=17)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="nav")
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0x16", result.reason or "")

    def test_return_home_level2_tilemap_short_circuits_exit_to_farm(self) -> None:
        """House L2 (0x17) mid exit_to_farm must SUCCESS (not run child forever)."""
        world = make_date_world(0x17, season=0, day=10, hour=17)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0x17", result.reason or "")
        self.assertIn("exit_to_farm", result.reason or "")
        self.assertIn("via=step", result.reason or "")

    def test_return_home_start_next_phase_short_circuits_when_house(self) -> None:
        """_start_next_phase must not spawn exit_to_farm when already house."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask()
        task.reset(world)
        # Fresh task: phase=start, no child — same path as reset → first phase pick.
        self.assertIsNone(task._task)
        self.assertEqual(task._phase, "start")

        result = task._start_next_phase(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertIn("via=start_next_phase", result.reason or "")
        self.assertIsNone(task._task)

    def test_return_home_timeout_exit_to_farm_on_non_house_fails_with_phase(
        self,
    ) -> None:
        """Soak residual: stuck exit_to_farm off-house must FAIL with phase=."""
        # D23 power_on end: dialogue@unknown tilemap=0x08 — not house/farm.
        world = make_date_world(0x08, season=0, day=23, hour=7)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")
        self.assertIn("phase=exit_to_farm", result.reason or "")

    def test_return_home_timeout_on_farm_fails_with_phase(self) -> None:
        """Hard timeout on farm (never entered house) stays FAILURE + phase."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="stuck nav")
        )

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")
        self.assertIn("phase=nav_house_front", result.reason or "")

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

    def test_return_home_navs_to_drop_spot_when_hands_full_in_field(self) -> None:
        """rr-6g7g: CLEAR_FIELD may finish holding a stone far from the house."""
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 89, 726)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_drop_spot")
        self.assertEqual(task._drop_spot_navs, 1)
        # Deep south: densified MultiMapNav ending at drop spot ~(136,480).
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 480))
        self.assertGreaterEqual(len(task._task.waypoints), 2)

    def test_return_home_densifies_south_field_approach(self) -> None:
        """rr-5in: mid-wall south of y=31 → east of pond (not x≈248 or pond)."""
        world = make_date_world(0x00, season=0, day=9, hour=14)
        # Mid-wall x (under fence body). Exhaust pre-escape budget so densify
        # multi_nav is exercised (pre-escape now covers mid-south of fence).
        set_player_pos(world.ram, 280, 620)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertGreaterEqual(len(wps), 3)
        self.assertEqual(wps[-1].target_px, (136, 424))
        # East of pond free lane (tile x≥36 → px≥576); never pond column 512.
        self.assertGreaterEqual(wps[0].target_px[0], 576)
        self.assertEqual(wps[0].run_direction, "right")
        for wp in wps:
            if wp.run_direction == "up":
                self.assertGreaterEqual(wp.target_px[0], 576)
                # Must not lateral-align through pond body (x≈512).
                self.assertNotEqual(wp.target_px[0], 512)

    def test_return_home_far_east_pond_pre_escapes_before_approach(self) -> None:
        """rr-5in D12: ~(854,527) after water — west+north pre-escape, not pond crawl."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 854, 527)
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("far-east", result.reason or "")

    def test_return_home_far_east_densifies_north_of_pond_lane(self) -> None:
        """After pre-escape budget spent, east free lane is east-of-pond then west above wall."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 700, 520)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # Northbound stages stay east of pond (x≥576), never 512.
        up_xs = [wp.target_px[0] for wp in wps if wp.run_direction == "up"]
        self.assertTrue(up_xs)
        for x in up_xs:
            self.assertGreaterEqual(x, 576)
            self.assertLessEqual(x, 640)
        # Eventually slides west above fence.
        self.assertTrue(any(wp.run_direction == "left" for wp in wps))

    def test_return_home_south_escape_on_fence_latitude_timeout(self) -> None:
        """South-of-fence but not deep_south (y=527) still escapes on multi_nav fail."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 774, 521)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("south escape", result.reason or "")

    def test_return_home_forces_enter_when_mid_yard_south_of_door(self) -> None:
        """D12 residual (118,486): force enter instead of hard multi_nav fail."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 118, 486)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)

    def test_return_home_west_of_fence_keeps_near_x_lane(self) -> None:
        """West free side densify uses current x, not forced SW pocket x=96."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 122, 518)
        task = ReturnHomeTask()
        task.reset(world)
        # Skip pre-escape so densify multi_nav is exercised.
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # Northbound corridor stays near player x (not yanked to 96).
        up_xs = [wp.target_px[0] for wp in wps if wp.run_direction == "up"]
        self.assertTrue(up_xs)
        for x in up_xs:
            self.assertGreaterEqual(x, 110)
            self.assertLessEqual(x, 160)

    def test_return_home_west_of_fence_runs_north(self) -> None:
        """West of fence wall (px x<176): densify north on free side, not east."""
        world = make_date_world(0x00, season=0, day=8, hour=15)
        set_player_pos(world.ram, 120, 620)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # West corridor x≈96, not east 512.
        self.assertLessEqual(wps[0].target_px[0], 160)
        self.assertEqual(wps[0].run_direction, "up")

    def test_return_home_pre_escapes_sw_pocket_before_approach(self) -> None:
        """Deep SW after CLEAR: B-run east first so multi_nav is not born stuck."""
        world = make_date_world(0x00, season=0, day=8, hour=15)
        set_player_pos(world.ram, 37, 715)
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("pre-escape", result.reason or "")

    def test_return_home_south_escape_on_multi_nav_timeout(self) -> None:
        """Far-south multi_nav fail queues B-run escape instead of hard-fail."""
        world = make_date_world(0x00, season=0, day=9, hour=18)
        set_player_pos(world.ram, 102, 726)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertTrue(task._action_queue)
        self.assertIn("south escape", result.reason or "")

    def test_return_home_uses_fence_gap_when_wall_confirmed(self) -> None:
        """Open y=31 gap after water: approach through gap, not only east end."""
        from harvest.core.tile_catalog import ADDR_MAP, FENCE, UNTILLED

        world = make_date_world(0x00, season=0, day=9, hour=14)
        set_player_pos(world.ram, 280, 560)
        # Solid fence wall x=11–29 with gap at x=14 (water refill opened it).
        for x in range(11, 30):
            world.ram[ADDR_MAP + 31 * 64 + x] = FENCE
        world.ram[ADDR_MAP + 31 * 64 + 14] = UNTILLED
        task = ReturnHomeTask()
        task.reset(world)
        # Skip mid-south pre-escape so gap densify multi_nav is exercised.
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        gap_px = 14 * 16 + 8  # 232
        self.assertTrue(any(abs(wp.target_px[0] - gap_px) <= 8 for wp in wps[:3]))
        self.assertEqual(wps[-1].target_px, (136, 424))

    def test_return_home_tosses_at_drop_spot_when_hands_full(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "drop carried before house")
        self.assertEqual(task._phase, "drop_carried")
        self.assertEqual(task._drop_attempts, 1)
        self.assertTrue(task._action_queue)

    def test_return_home_fails_after_drop_budget_with_held_item(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_attempt_limit=2)
        task.reset(world)
        task._drop_spot_navs = 3  # skip relocate
        task._drop_attempts = 2

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not clear hands before house entry", result.reason or "")
        self.assertIn("held=0x0D", result.reason or "")

    def test_return_home_low_budget_south_of_fence_short_charge(self) -> None:
        """D19 residual: after drop thrash, ~1k frames left at (153,518).

        Must queue compact east→north rather than hard-fail immediately.
        """
        world = make_date_world(0x00, season=0, day=19, hour=18)
        set_player_pos(world.ram, 153, 518)
        task = ReturnHomeTask(timeout=11000)
        task.reset(world)
        task._total_steps = 10000  # remaining ~999f
        task._drop_attempts = 3

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertIn("low-budget", result.reason or "")
        self.assertTrue(task._action_queue)

    def test_return_home_fails_early_on_stuck_same_held(self) -> None:
        """Power-on D19 residual: held=0x0F rock fragment never clears.

        Same held id across drop_stuck_held_limit observations must hard-fail
        before burning the outer 11k timeout in phase=drop_carried.
        """
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=19, hour=18)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0F
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_stuck_held_limit=3, drop_attempt_limit=10)
        task.reset(world)
        task._drop_spot_navs = 3
        task._drop_deep_relocated = True  # deep south already tried
        task._drop_last_held = 0x0F
        task._drop_same_held = 2  # one more observation trips the gate
        task._drop_attempts = 2

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not clear hands before house entry", result.reason or "")
        self.assertIn("held=0x0F", result.reason or "")
        self.assertIn("same_held=", result.reason or "")


if __name__ == "__main__":
    unittest.main()
