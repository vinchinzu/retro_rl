"""Return-home / late-day sleep day-plan tests.

Split from test_day_plan_home monofile (return-home behavior + late-day builder).
Exit-to-farm / house-exit sequences live in ``test_day_plan_home_sequences``.
Discovered via pytest; also aggregated by ``test_day_plan_home`` shim.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pytest

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


@pytest.mark.rom
def test_grown_corn_east_return_clears_the_well_and_reaches_home():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action
    from harvest.planner.day_plan_status import is_house_tilemap
    from retro_harness import AuditedEnv, AuditCapabilities, WorldState

    env = AuditedEnv(make_harvest_env("LN_y23_east_natural", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.east_home_test"))
    try:
        env.reset()
        ram = env.get_ram()
        pos = get_pos_from_ram(ram)
        assert read_ram_value(ram, "season") == 1
        assert read_ram_value(ram, "day") == 21
        assert pos.x > 560 and pos.y >= 504
        assert get_tile_at(ram, 12, 28) == 0x52  # grown west-ring corn
        task = ReturnHomeTask()
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        for frame in range(6000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert is_house_tilemap(read_ram_value(env.get_ram(), "tilemap"))
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("state,tilemap", [("LN_marriage_after_propose_FAIL", 0x16), ("LN_marriage_propose_candidate_FAIL", 0x16), ("LN_marriage_propose_verified_candidate_FAIL", 0x17)])
def test_storm_remodeled_house_sleeps_from_diary_or_near_bed(state, tilemap):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.tasks.home import GoToSleepTask
    from harvest.tasks.nav import make_action
    from retro_harness import WorldState, AuditedEnv, AuditCapabilities

    env = AuditedEnv(make_harvest_env(state, lean_step=True), capabilities=AuditCapabilities(
        provider="harvest.storm_sleep_test", observes_ram_writes=True,
        observes_mid_run_loads=True, observes_assists=True))
    try:
        env.reset()
        assert read_ram_value(env.get_ram(), "tilemap") == tilemap
        start_day = read_ram_value(env.get_ram(), "day")
        task = GoToSleepTask(earliest_sleep_hour=0)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(1600):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "day") == start_day + 1
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


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

    def test_return_home_carpenter_pose_walks_downhill_not_spa(self) -> None:
        world = make_date_world(0x10, season=0, day=15, hour=12)
        set_player_pos(world.ram, 474, 630)
        task = ReturnHomeTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, MultiMapNavTask)
        hops = [wp.target_px for wp in task._task.waypoints]
        self.assertEqual(hops[0], (520, 712))
        self.assertNotIn((569, 201), hops)
        self.assertIn((312, 744), hops)
        self.assertNotIn((132, 128), hops)
        exit_i = hops.index((312, 744))
        self.assertEqual(hops[exit_i + 1], (137, 40))

    def test_return_home_jumps_out_of_the_spa_pond_before_the_mountain_route(self) -> None:
        # Y2 Spring D19: the day floor fired mid-soak, the farmer stood in the
        # water at (655,198) and the route's first waypoint (west of the lip)
        # was unreachable, so return_home timed out 11k frames, three rescues.
        world = make_date_world(0x10, season=0, day=19, hour=18)
        set_player_pos(world.ram, 655, 198)
        task = ReturnHomeTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "spa_pond_exit")
        self.assertIsNone(task._task)
        # The queue holds the Left+A re-crossing, then re-evaluates from the lip.
        self.assertGreater(len(task._action_queue), 100)
        task._action_queue.clear()
        set_player_pos(world.ram, 590, 201)
        after = task.step(world)
        self.assertEqual(after.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, MultiMapNavTask)

    def test_return_home_walks_the_mountain_route_from_the_dry_lip(self) -> None:
        world = make_date_world(0x10, season=0, day=19, hour=18)
        set_player_pos(world.ram, 619, 201)
        task = ReturnHomeTask()
        task.reset(world)
        task.step(world)
        self.assertNotEqual(task._phase, "spa_pond_exit")
        self.assertIsInstance(task._task, MultiMapNavTask)

    def test_a_blind_charge_never_steps_onto_the_well_pit(self) -> None:
        # Y2 Summer D4: the far-east escape's "up 80" from x=805 walked onto the
        # well hole (50,55), the Harvest Sprites' cave, and never came back.
        from harvest.core.tile_catalog import set_tile_at
        from harvest.planner.tasks.home_recover import FARM_PIT_TILE, hold_back_from_pit, sanitize_action
        from harvest.tasks.nav import make_action

        world = make_date_world(0x00, season=1, day=4, hour=18)
        set_tile_at(world.ram, 50, 55, FARM_PIT_TILE)
        up = make_action(up=True, b=True)
        set_player_pos(world.ram, 806, 930)  # far below: free to walk
        self.assertIs(hold_back_from_pit(up, world.ram), up)
        set_player_pos(world.ram, 806, 902)  # the next 12px reach the pit tile
        held = hold_back_from_pit(up, world.ram)
        self.assertFalse(held.any())
        self.assertFalse(sanitize_action(up, world.ram).any())
        # Walking away from it, sideways around it and standing still are untouched.
        down = make_action(down=True)
        self.assertIs(hold_back_from_pit(down, world.ram), down)
        left = make_action(left=True)
        self.assertIs(hold_back_from_pit(left, world.ram), left)
        idle = make_action()
        self.assertIs(hold_back_from_pit(idle, world.ram), idle)
        # The second hole, (37,48) tile 0xC8, is held the same way (Y2 Summer D16).
        set_tile_at(world.ram, 37, 48, 0xC8)
        set_player_pos(world.ram, 598, 790)
        self.assertFalse(hold_back_from_pit(up, world.ram).any())
        # Off the farm map the filter does nothing.
        mountain = make_date_world(0x10, season=1, day=4, hour=18)
        set_tile_at(mountain.ram, 50, 55, FARM_PIT_TILE)
        set_player_pos(mountain.ram, 806, 902)
        self.assertIs(hold_back_from_pit(up, mountain.ram), up)

    def test_a_rider_dismounts_with_an_a_pulse_only_at_the_door_stand(self) -> None:
        # Y2 Spring D19 (jitter run): on the horse at the stand (134,425), the enter push
        # timed out three times. A held A anywhere else is a regression: it froze the walk
        # (Y2 Fall D23) and starved the 17:00 shipping scene of its A edge (Y3 Spring D20).
        from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
        from harvest.planner.tasks.home_recover import PLAYER_STATE_RIDING, dismount_horse, sanitize_action
        from harvest.tasks.nav import make_action

        world = make_date_world(0x00, season=0, day=19, hour=18)
        live = LIVE_RAM_WRAM_OFFSET if len(world.ram) > 0x20000 else 0
        a_idx = int(make_action(a=True).argmax())
        up = make_action(up=True)

        def ride(x: int, y: int, *, lock: int = 1, action: int = 0) -> None:
            set_player_pos(world.ram, x, y)
            world.ram[field_spec("player_state").address + live] = PLAYER_STATE_RIDING
            world.ram[field_spec("input_lock").address] = lock
            world.ram[field_spec("player_action").address + live] = action

        set_player_pos(world.ram, 134, 425)
        self.assertIs(dismount_horse(up, world.ram), up)  # walking: untouched
        ride(134, 425)
        self.assertTrue(dismount_horse(up, world.ram)[a_idx])
        self.assertTrue(sanitize_action(up, world.ram)[a_idx])
        ride(134, 425, action=5)  # the dismount animation plays on neutral frames
        self.assertFalse(dismount_horse(up, world.ram).any())
        ride(134, 425, lock=2)  # a dialogue or the 5pm scene owns the pad
        self.assertIs(dismount_horse(up, world.ram), up)
        ride(259, 376)  # mounted elsewhere: walking works, leave the pad alone
        self.assertIs(dismount_horse(up, world.ram), up)
        mountain = make_date_world(0x10, season=0, day=19, hour=18)
        set_player_pos(mountain.ram, 134, 425)
        mountain.ram[field_spec("player_state").address + live] = PLAYER_STATE_RIDING
        self.assertIs(dismount_horse(up, mountain.ram), up)

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

    def test_return_home_from_inside_the_animal_shop_walks_out_through_town(self) -> None:
        """Y2 Spring D16 (compose7): a failed walk home from the animal shop re-entered the shop, and the generic
        ExitToFarm (farm buildings only) timed out three times per rescue until the run wedged after 8 rescues."""
        for tilemap, pos, first_wp in ((0x24, (62, 201), 0x24), (0x04, (600, 888), 0x04)):
            world = make_date_world(tilemap, season=0, day=16, hour=8)
            set_player_pos(world.ram, *pos)
            task = ReturnHomeTask()
            task.reset(world)

            task.step(world)

            self.assertIsInstance(task._task, MultiMapNavTask, hex(tilemap))
            self.assertEqual(task._task.waypoints[0].tilemap, first_wp)
            self.assertEqual(task._task.waypoints[-1].tilemap, 0x00)   # ends back on the farm
            self.assertEqual(task._phase, "exit_to_farm")

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
        """North of the stand but off the door's walkway column: walk south first."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 155, 389)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, NavTask)
        self.assertEqual(
            (task._task.target_px.x, task._task.target_px.y),
            (136, 424),
        )

    def _horse(self, world, x: int, y: int, *, sprite: int = 0x1FB, slot: int = 2) -> None:
        from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE

        offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
        for at, value in ((0, GOBJ_INITIALIZED), (2, sprite), (8, x), (10, y)):
            world.ram[offset + at] = value & 0xFF
            world.ram[offset + at + 1] = (value >> 8) & 0xFF

    def test_a_horse_beside_the_door_line_shifts_the_push_away_from_it(self) -> None:
        """e2e s2 Y3 Summer D27: horse 0x1FB at (126,358), farmer at (136,371); x=141 entered, x=136 never did."""
        world = make_date_world(0x00, season=1, day=27, hour=18)
        set_player_pos(world.ram, 136, 371)
        task = ReturnHomeTask()
        self.assertEqual(task._house_enter_task(world).door_align_px, 136)
        self.assertEqual(task._house_enter_task(world).door_align_tolerance, 4)

        self._horse(world, 126, 358)
        enter = task._house_enter_task(world)
        self.assertEqual((enter.door_align_px, enter.door_align_tolerance), (141, 1))

        world = make_date_world(0x00, season=1, day=27, hour=18)
        set_player_pos(world.ram, 136, 371)
        self._horse(world, 146, 358)                         # on the right: shift left
        self.assertEqual(task._house_enter_task(world).door_align_px, 131)

    def test_the_farm_dog_in_the_corridor_is_sidestepped_too(self) -> None:
        """e2e s3 Y2 Summer D22: the same spot (126,358) held by a dog frame (0x1E3)."""
        world = make_date_world(0x00, season=1, day=22, hour=18)
        set_player_pos(world.ram, 136, 371)
        self._horse(world, 126, 358, sprite=0x1E3)
        enter = ReturnHomeTask()._house_enter_task(world)
        self.assertEqual((enter.door_align_px, enter.door_align_tolerance), (141, 1))

    def test_an_animal_away_from_the_door_line_or_a_non_animal_changes_nothing(self) -> None:
        world = make_date_world(0x00, season=1, day=27, hour=18)
        set_player_pos(world.ram, 136, 371)
        task = ReturnHomeTask()
        self._horse(world, 200, 358)                         # beside, not in, the corridor
        self.assertEqual(task._house_enter_task(world).door_align_px, 136)
        self._horse(world, 126, 358, sprite=0x0100)          # not a farm animal sprite
        self.assertEqual(task._house_enter_task(world).door_align_px, 136)
        self._horse(world, 126, 250)                         # an animal far above the door
        self.assertEqual(task._house_enter_task(world).door_align_px, 136)

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
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 376))

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
        self.assertTrue(task._drop_spot_visited)
        self.assertFalse(task._drop_wandered)
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
        # Cross west above the well's north edge (y=416), with room for the
        # waypoint radius. The old y=440 slide crossed the well and corn.
        west = [wp for wp in wps if wp.run_direction == "left"]
        self.assertTrue(west)
        self.assertTrue(all(wp.target_px[1] + wp.radius < 416 for wp in west))

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

    def test_return_home_carries_debris_to_the_pond_when_drop_spot_refuses(self) -> None:
        """Summer D27: the spot and every face refused a stone; the pond takes it."""
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.tasks.home_recover import PondTossTask, WanderDropTask

        world = make_date_world(0x00, season=1, day=27, hour=18)
        set_player_pos(world.ram, 121, 486)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_attempt_limit=2)
        task.reset(world)
        task._drop_spot_visited = True
        task._drop_attempts = 1

        task.step(world)
        self.assertIsInstance(task._task, PondTossTask)
        self.assertTrue(task._pond_tossed)
        self.assertFalse(task._drop_wandered)

        # The pond walk failed too: one wander, then the existing failure.
        task._task = None
        task._phase = "start"
        result = task.step(world)
        self.assertIsInstance(task._task, WanderDropTask)
        self.assertTrue(task._drop_wandered)
        self.assertIn("wander", (result.reason or "").lower())

    def test_return_home_tosses_at_the_pond_first_when_it_is_nearer(self) -> None:
        """D27: the far-east escape lifted a stone beside the pond, then walked it home."""
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.tasks.home_recover import PondTossTask

        world = make_date_world(0x00, season=1, day=27, hour=18)
        set_player_pos(world.ram, 582, 502)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "start"

        task._start_next_phase(world)

        self.assertIsInstance(task._task, PondTossTask)
        self.assertFalse(task._drop_spot_visited)

    def test_house_approach_may_use_the_bin_lane(self) -> None:
        """Summer D17/D24: the strip east of the grown west ring had no other way home."""
        from harvest.maps.farm_pond import SHIP_BIN_LANE_TILES

        world = make_date_world(0x00, season=1, day=24, hour=18)
        set_player_pos(world.ram, 250, 454)
        task = ReturnHomeTask()
        task.reset(world)

        task._start_house_approach(world, task._house_front_px(world.ram))

        self.assertEqual(task._task.name, "nav_house_front")
        self.assertTrue(SHIP_BIN_LANE_TILES <= set(task._task.allow_tiles))

    def test_return_home_pushes_up_from_the_door_walkway(self) -> None:
        """Y2 Spring D8: a sprite parked on the stand (8,26); the walkway leads to the door."""
        from harvest.planner.tasks.home_approach import in_house_walkway
        from harvest.tasks.nav import Point

        front = Point(136, 424)
        self.assertTrue(in_house_walkway(Point(135, 409), front))
        self.assertFalse(in_house_walkway(Point(160, 409), front))
        self.assertFalse(in_house_walkway(Point(136, 340), front))
        world = make_date_world(0x00, season=0, day=8, hour=18)
        set_player_pos(world.ram, 135, 409)
        task = ReturnHomeTask()
        task.reset(world)

        task._start_next_phase(world)

        self.assertEqual(task._phase, "enter_house")
        self.assertEqual(task._task.stand_tile, (8, 25))

    def test_return_home_never_takes_a_hen_to_the_pond(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.tasks.home_recover import WanderDropTask

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x25
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_attempt_limit=2)
        task.reset(world)
        task._drop_spot_visited = True
        task._drop_attempts = 1

        task.step(world)

        self.assertIsInstance(task._task, WanderDropTask)

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
        task._drop_spot_visited = True
        task._drop_wandered = True
        task._drop_attempts = 1

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

    def test_return_home_fails_after_one_wander_still_holding(self) -> None:
        """Power-on residual: held=0x0F fence/rock the spot and wander refused.

        One wander, then fail. No deep-south third strategy and no same-held
        counter burning the outer timeout in phase=drop_carried.
        """
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=19, hour=18)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0F
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)
        task._drop_spot_visited = True
        task._drop_wandered = True
        task._drop_attempts = 1

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not clear hands before house entry", result.reason or "")
        self.assertIn("held=0x0F", result.reason or "")
        self.assertNotIn("same_held=", result.reason or "")


if __name__ == "__main__":
    unittest.main()


class UnknownHeldItemTests(unittest.TestCase):
    """Sweep t5f_s54, Y2 Summer D18: held 0x57 after a forced-failure day; every yard tile
    (0xA8) refused the toss and RETURN_HOME ended in 8 rescues. The pond takes anything but a
    crop, an egg, milk, fodder or a hen."""

    def _home(self, held: int):
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=1, day=18, hour=18)
        set_player_pos(world.ram, 121, 486)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = held
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_attempt_limit=2)
        task.reset(world)
        task._drop_spot_visited = True
        task._drop_attempts = 1
        return task, world

    def test_an_unrecognised_item_goes_to_the_pond_when_the_yard_refuses_it(self) -> None:
        from harvest.planner.tasks.home_recover import PondTossTask

        task, world = self._home(0x57)
        task.step(world)
        self.assertIsInstance(task._task, PondTossTask)
        self.assertTrue(task._pond_tossed)

    def test_a_hen_or_a_crop_is_never_thrown_in_the_pond(self) -> None:
        from harvest.planner.tasks.home_recover import PondTossTask, WanderDropTask

        for held in (0x25, 0x10, 0x14, 0x1A):
            task, world = self._home(held)
            task.step(world)
            self.assertNotIsInstance(task._task, PondTossTask, hex(held))
            self.assertIsInstance(task._task, WanderDropTask, hex(held))
