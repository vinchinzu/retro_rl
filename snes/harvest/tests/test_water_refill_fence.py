"""Unit tests for fence local-drop and corridor_only pond access."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from water_refill_helpers import (
    _blank_ram,
    _set_player_tile,
    _set_tile,
)

from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    DebrisType,
    FENCE,
    MAP_WIDTH,
    STONE,
)
from retro_harness import TaskStatus


def _fill_farm_map(ram, tile_id: int) -> None:
    for i in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + i] = tile_id


class FenceLocalDropTests(unittest.TestCase):
    """FenceClearLoopTask must not hard-fail when pond BFS is viewport-blocked."""

    def test_navigate_pond_falls_back_to_local_drop(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        # Wall of solid tiles — no path to pond stands.
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0x05)
        # Tiny open cell around player.
        for ty in range(28, 31):
            for tx in range(14, 17):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (15, 29))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=1, max_steps_per_fence=200)
        # Avoid loading recorded toss task from disk.
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(15, 31), tile_id=0x05)
        task._navigator.update(ram)

        # Hop may improve manhattan inside the pocket once; then local_drop.
        final_state = None
        for _ in range(8):
            result = task.step(world)
            final_state = task._state
            if final_state == "local_drop":
                break
            # Simulate walk along hop without leaving the pocket.
            task._navigator.update(ram)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(final_state, "local_drop")

    def test_local_drop_clears_when_hands_empty(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(20, 40):
            for tx in range(10, 40):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (15, 29))
        # Not carrying — local_drop should count as cleared.
        ram[ADDR_PLAYER_STATE] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=2, max_steps_per_fence=200)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "local_drop"
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.cleared_count, 1)
        self.assertEqual(task._state, "scan")

    def test_corridor_drop_does_not_succeed_with_stale_held_item(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.tasks.fence_flow import ADDR_PLAYER_STATE, FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        _set_player_tile(ram, (15, 32))
        # Carry animation can clear a frame before held-item RAM. The corridor
        # is not usable for berry pickup until both signals agree.
        ram[ADDR_PLAYER_STATE] = 0
        ram[ADDR_HELD_ITEM] = 0x0D

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=1, corridor_only=True)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "local_drop"
        task._navigator.update(ram)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.cleared_count, 0)
        self.assertGreater(len(task._action_queue), 0)


class FenceCorridorOnlyTests(unittest.TestCase):
    """corridor_only FenceClearLoop must local-drop instead of pond thrash."""

    def test_corridor_only_targets_y31_wall_not_nearest_decorative_fence(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 2, 21, 0x05)   # closer, but not pond-access wall
        _set_tile(ram, 11, 31, 0x05)  # intended corridor fence
        _set_player_tile(ram, (3, 21))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=1, max_steps_per_fence=500, corridor_only=True
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)

        result = task.step(world)
        # Corridor mode first stages west; simulate arrival, then scan.
        if task._state == "stage_corridor":
            self.assertIsNotNone(task._corridor_stage)
            _set_player_tile(ram, task._corridor_stage)
            task.step(world)
            result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._current)
        self.assertEqual(task._current.tile, (11, 31))

    def test_corridor_only_carry_south_from_y30(self) -> None:
        """ROM: after lift player is often on y=30; charge must still fire."""
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_player_tile(ram, (13, 30))  # approach tile after lift
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=1, max_steps_per_fence=500, corridor_only=True
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(13, 31), tile_id=0x05)
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False),
            msg="must carry-south charge from y=30, not immediate local_drop",
        )
        self.assertNotEqual(
            task._state,
            "local_drop",
            msg="must not local_drop before carry-south charge on y=30",
        )

    def test_corridor_only_skips_navigate_pond_to_local_drop(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_player_tile(ram, (13, 31))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=1, max_steps_per_fence=500, corridor_only=True
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(13, 31), tile_id=0x05)
        task._navigator.update(ram)

        # First step: carry-south charge from y<=31 (not only y==31).
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False)
            or task._state == "local_drop"
            or len(task._action_queue) > 0
            or result.reason
            in (
                "corridor_only south charge",
                "corridor_only local drop",
                "corridor_only drop south of wall",
            ),
            msg=f"state={task._state} reason={result.reason}",
        )
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False)
            or result.reason == "corridor_only south charge"
            or len(task._action_queue) > 0,
            msg="corridor_only must attempt carry-south before local_drop",
        )
        # Drain charge queue then expect local_drop arm (still north in unit).
        for _ in range(700):
            result = task.step(world)
            if task._state == "local_drop":
                break
        self.assertEqual(
            task._state,
            "local_drop",
            msg=f"after charge must local_drop, got {task._state}",
        )


class LeftoverPondDumpTests(unittest.TestCase):
    def test_cleared_pond_perimeter_post_becomes_next_approach(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        # Live frontier after the first perimeter post at (30,30) is gone.
        _set_tile(ram, 31, 30, FENCE)
        _set_tile(ram, 32, 30, FENCE)
        _set_player_tile(ram, (29, 30))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=None, pond_dump=True)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._current)
        self.assertEqual(tuple(task._current.tile), (31, 30))
        self.assertEqual(task._approach_tile, (30, 30))

    def test_toss_completion_counts_before_input_lock_mash(self) -> None:
        from harvest.tasks.carry_toss import POND_WEST_EGRESS_STAND
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 0
        _set_player_tile(ram, (32, 34))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=None, pond_dump=True)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(29, 31), tile_id=FENCE)
        task._pond_carry._toss_started = True
        task._pond_carry._egress_goal = POND_WEST_EGRESS_STAND

        result = task.step(world)

        self.assertEqual(result.reason, "input_lock")
        self.assertEqual(task.cleared_count, 0)

        ram[ADDR_INPUT_LOCK] = 1
        _set_player_tile(ram, (33, 34))
        result = task.step(world)

        self.assertIn("egress boxed pond stand", result.reason)
        self.assertEqual(task.cleared_count, 0)

        _set_player_tile(ram, (33, 35))
        result = task.step(world)

        self.assertIn("egress boxed pond stand", result.reason)
        self.assertEqual(task.cleared_count, 0)

        _set_player_tile(ram, POND_WEST_EGRESS_STAND)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "pond dump complete cleared=1")
        self.assertEqual(task.cleared_count, 1)
        self.assertEqual(task._state, "scan")

    def test_stone_dump_scans_stones_not_only_fences(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 20, 20, FENCE)
        _set_tile(ram, 14, 28, STONE)
        _set_player_tile(ram, (13, 28))

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=10,
            corridor_only=False,
            pond_dump=True,
            debris_types=(DebrisType.STONE,),
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._current)
        self.assertEqual(task._current.debris_type, DebrisType.STONE)
        self.assertEqual(task._current.tile, (14, 28))

    def test_pond_dump_picks_y31_wall_before_house_row(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 2, 24, FENCE)
        _set_tile(ram, 29, 31, FENCE)
        _set_player_tile(ram, (5, 28))

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            corridor_only=False,
            pond_dump=True,
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._current)
        self.assertEqual(tuple(task._current.tile), (29, 31))

    def test_pond_dump_hops_to_nearest_wall_post(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 11, 31, FENCE)
        _set_tile(ram, 29, 31, FENCE)
        _set_player_tile(ram, (12, 28))

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            corridor_only=False,
            pond_dump=True,
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._current)
        self.assertEqual(tuple(task._current.tile), (11, 31))

    def test_pond_dump_does_not_south_charge_from_north_farm(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (18, 20))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(18, 21), tile_id=0x05)
        task._navigator.update(ram)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._state, "navigate_pond")
        self.assertFalse(task._corridor_charge_done)
        self.assertLess(len(task._action_queue), 40)

    def test_pond_dump_timeout_keeps_carrying(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (18, 20))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=10,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate"
        task._current = SimpleNamespace(tile=(18, 21), tile_id=0x05)
        task._approach_tile = (18, 20)
        task._steps_on_fence = 10
        task._navigator.update(ram)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._state, "navigate_pond")
        self.assertNotEqual(task._state, "local_drop")
        self.assertIn("keep carrying", result.reason)

    def test_pond_dump_boxed_does_not_local_drop(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0x05)
        for ty in range(18, 22):
            for tx in range(16, 20):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (18, 20))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(18, 21), tile_id=0x05)
        task._navigator.update(ram)
        for _ in range(8):
            result = task.step(world)
            task._navigator.update(ram)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._state, "navigate_pond")

    def test_pond_dump_local_drop_does_not_count_as_cleared(self) -> None:
        from harvest.tasks.fence_flow import ADDR_PLAYER_STATE, FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(20, 40):
            for tx in range(10, 40):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (15, 29))
        ram[ADDR_PLAYER_STATE] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=200,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "local_drop"
        task._current = SimpleNamespace(tile=(15, 31), tile_id=0x05)
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.cleared_count, 0)
        self.assertEqual(task._state, "scan")
        self.assertIn((15, 31), task._skip_tiles)

    def test_navigate_idle_skips_stuck_target(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0x05)
        for ty in range(28, 31):
            for tx in range(14, 17):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 32, 26, FENCE)
        _set_player_tile(ram, (15, 29))

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=400,
            stasis_repath=2,
            max_failures=20,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate"
        task._current = SimpleNamespace(tile=(32, 26), tile_id=0x05)
        task._approach_tile = (32, 27)
        task._navigator.update(ram)
        task._navigator.path = []
        task._navigator.stasis = 50

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._state, "scan")
        self.assertIn((32, 26), task._skip_tiles)

    def test_skip_clears_temp_blocked_so_wall_is_not_sealed(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 29, 31, FENCE)
        _set_player_tile(ram, (15, 29))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=400,
            max_failures=20,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._current = SimpleNamespace(tile=(29, 31), tile_id=0x05)
        task._pathfinder.temp_blocked.update({(28, 30), (29, 30), (30, 30)})
        result = task._skip_current("stasis")
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._pathfinder.temp_blocked, set())
        self.assertIn((29, 31), task._skip_tiles)

    def test_pond_dump_hops_around_stump_inside_viewport(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )
        from harvest.tasks.nav import VIEWPORT_HOP_TILES

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        stump = ((27, 30), (28, 30), (27, 31), (28, 31))
        for (tx, ty), tid in zip(stump, (0x09, 0x0A, 0x0B, 0x0C)):
            _set_tile(ram, tx, ty, tid)
        _set_player_tile(ram, (24, 28))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=None,
            pond_dump=True,
            max_steps_per_fence=400,
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._corridor_charge_done = True
        task._current = SimpleNamespace(tile=(24, 29), tile_id=0x05)
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        # Pond-dump travel is owned by the focused carry composer, not the
        # fence-selection state machine.
        path = list(task._pond_carry._navigator.path or [])
        self.assertTrue(path)
        self.assertLessEqual(len(path), VIEWPORT_HOP_TILES)
        self.assertTrue(set(stump).isdisjoint(path))


class FenceClearTerminationTests(unittest.TestCase):
    """Global termination: timeout, carry retries, input lock, loaded-map."""

    def _pond_dump_world(self, *, tile_id: int = 0xA1, player=(15, 29)):
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        _fill_farm_map(ram, tile_id)
        _set_player_tile(ram, player)
        return SimpleNamespace(ram=ram, info={}, obs=None)

    def _pond_dump_task(self, world, **kwargs):
        from harvest.tasks.fence_flow import FenceClearLoopTask

        params = dict(max_fences=None, pond_dump=True, corridor_only=False)
        params.update(kwargs)
        task = FenceClearLoopTask(**params)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        return task

    def test_build_fence_clear_passes_timeout(self) -> None:
        import numpy as np
        from harvest.planner.day_phase_registry import (
            TaskBuildContext,
            _build_fence_clear,
        )
        from harvest.planner.day_phase_types import PhaseSpec
        from retro_harness import WorldState

        spec = PhaseSpec("CLEAR_FENCES", "fence_clear", {"timeout": 12345})
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={}, obs=None)
        task = _build_fence_clear(TaskBuildContext(), spec, world)
        self.assertEqual(task.timeout, 12345)

    def test_pond_dump_carry_timeout_is_bounded(self) -> None:
        from harvest.tasks.fence_flow import ACTION_CARRYING_BIT, ADDR_PLAYER_STATE
        from retro_harness import TaskResult

        world = self._pond_dump_world(player=(18, 20))
        world.ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT
        task = self._pond_dump_task(world, max_steps_per_fence=10, max_failures=20)
        task._state = "navigate"
        task._current = SimpleNamespace(tile=(18, 21), tile_id=0x05)
        task._approach_tile = (18, 20)
        task._navigator.update(world.ram)
        task._pond_carry.step = lambda w: TaskResult(
            status=TaskStatus.RUNNING, reason="stub carry"
        )

        keep_carrying = 0
        result = None
        for _ in range(80):
            result = task.step(world)
            if result.reason == "pond_dump keep carrying":
                keep_carrying += 1
            if result.status != TaskStatus.RUNNING:
                break
            if result.reason != "pond_dump keep carrying" and (
                (18, 21) in task._skip_tiles
                or task._state in ("scan", "local_drop")
                or "retry cap" in (result.reason or "")
            ):
                break
        self.assertIsNotNone(result)
        self.assertLessEqual(keep_carrying, task.max_pond_carry_retries)
        self.assertNotEqual(result.reason, "pond_dump keep carrying")
        self.assertTrue(
            (18, 21) in task._skip_tiles
            or result.status == TaskStatus.FAILURE
            or task._state == "local_drop",
            msg=f"state={task._state} reason={result.reason} skip={task._skip_tiles}",
        )

    def test_input_lock_ab_is_bounded(self) -> None:
        world = self._pond_dump_world()
        world.ram[ADDR_INPUT_LOCK] = 0
        task = self._pond_dump_task(world, max_failures=20)
        task._state = "navigate"
        task._current = SimpleNamespace(tile=(15, 31), tile_id=FENCE)
        task._approach_tile = (15, 30)
        task._navigator.update(world.ram)

        ab_presses = 0
        result = None
        for _ in range(40):
            result = task.step(world)
            if result.reason == "input_lock":
                ab_presses += 1
                continue
            break
        self.assertIsNotNone(result)
        self.assertGreater(ab_presses, 0)
        self.assertLessEqual(ab_presses, task.max_input_lock_presses)
        self.assertNotEqual(result.reason, "input_lock")

    def test_pond_dump_stale_farm_map_is_not_success(self) -> None:
        world = self._pond_dump_world(tile_id=0xFF, player=(26, 30))
        task = self._pond_dump_task(world)
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("stale_farm_map", result.reason)

    def test_pond_dump_loaded_farm_zero_is_success(self) -> None:
        world = self._pond_dump_world(tile_id=0xA1, player=(15, 29))
        task = self._pond_dump_task(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("loaded-farm zero", result.reason)

    def test_skip_tiles_survive_other_dump_success(self) -> None:
        world = self._pond_dump_world()
        task = self._pond_dump_task(world)
        task._current = SimpleNamespace(tile=(10, 31), tile_id=FENCE)
        skipped = task._skip_current("stuck on A")
        self.assertEqual(skipped.status, TaskStatus.RUNNING)
        self.assertIn((10, 31), task._skip_tiles)
        task._current = SimpleNamespace(tile=(20, 31), tile_id=FENCE)
        dumped = task._finish_pond_carry(world)
        self.assertEqual(dumped.status, TaskStatus.RUNNING)
        self.assertIn((10, 31), task._skip_tiles)

    def test_corridor_only_open_gap_is_success(self) -> None:
        from harvest.tasks.fence_flow import FenceClearLoopTask

        world = self._pond_dump_world(tile_id=0xA1, player=(15, 32))
        task = FenceClearLoopTask(max_fences=1, corridor_only=True)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "corridor already open")


if __name__ == "__main__":
    unittest.main()
