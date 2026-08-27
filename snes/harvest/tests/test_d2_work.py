"""D2 field-work composition — bounded quotas and exhaustive fences."""

from __future__ import annotations

import unittest

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import Tool
from harvest.planner.d2_work import (
    D2_TARGETS,
    bush_clear_phase,
    d2_leftover_phases,
    d2_post_shop_work_phases,
    ensure_axe_phase,
    ensure_hammer_phase,
    fence_dump_phase,
    leftover_already_queued,
    needs_spa_before_next_smash,
    pocket_water_phase,
    rock_clear_phase,
    should_spa_retry,
    stone_pond_phase,
    stump_clear_phase,
)
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind
from harvest.planner.day_plan_phases import pocket_plant_phases


class D2WholeFarmContractTests(unittest.TestCase):
    def test_crop_targets_are_not_debris_quotas(self) -> None:
        self.assertEqual(D2_TARGETS, {"plant": 8, "water": 8})

    def test_bush_phase_is_ten_lifts_not_plot_ring(self) -> None:
        spec = bush_clear_phase()
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.kind, PhaseKind.CLEAR_FIELD)
        self.assertEqual(spec.params["handoff"], "quota")
        self.assertEqual(spec.params["quota"], {"weeds": 10})
        self.assertFalse(spec.params["fetch_tools"])
        self.assertEqual(spec.params["priority"], ["weed"])
        self.assertNotIn("farm_bounds", spec.params)
        self.assertEqual(spec.params["timeout"], 0)

    def test_fence_dump_is_all_posts_to_pond(self) -> None:
        spec = fence_dump_phase()
        self.assertEqual(spec.phase, "CLEAR_FENCES")
        self.assertEqual(spec.kind, PhaseKind.FENCE_CLEAR)
        self.assertIsNone(spec.params["max_fences"])
        self.assertFalse(spec.params["corridor_only"])
        self.assertTrue(spec.params["pond_dump"])
        self.assertEqual(spec.params["max_steps_per_fence"], 2800)
        self.assertEqual(spec.params["debris_types"], ["fence"])
        self.assertEqual(spec.params["timeout"], 0)

    def test_stone_pond_phase_dumps_all_not_hammer(self) -> None:
        spec = stone_pond_phase()
        self.assertEqual(spec.phase, "CLEAR_STONES")
        self.assertEqual(spec.kind, PhaseKind.FENCE_CLEAR)
        self.assertIsNone(spec.params["max_fences"])
        self.assertEqual(spec.params["timeout"], 0)
        self.assertEqual(spec.params["max_failures"], 60)
        self.assertFalse(spec.params["corridor_only"])
        self.assertEqual(spec.params["debris_types"], ["stone"])

    def test_rock_phase_needs_hammer_for_large_only(self) -> None:
        spec = rock_clear_phase()
        self.assertEqual(spec.phase, "CLEAR_ROCKS")
        self.assertEqual(spec.params["handoff"], "quota")
        self.assertEqual(spec.params["quota"], {"large_rocks": 10_000})
        self.assertEqual(spec.params["timeout"], 0)
        self.assertEqual(spec.params["priority"], ["rock"])
        self.assertFalse(spec.params["prefer_lift_for_stones"])
        self.assertEqual(spec.contract.required_tools, ("hammer",))
        self.assertFalse(spec.params["fetch_tools"])

    def test_stump_phase_needs_axe(self) -> None:
        spec = stump_clear_phase()
        self.assertEqual(spec.phase, "CLEAR_STUMPS")
        self.assertEqual(spec.params["handoff"], "quota")
        self.assertEqual(spec.params["quota"], {"stumps": 2})
        self.assertEqual(spec.params["priority"], ["stump"])
        self.assertEqual(spec.contract.required_tools, ("axe",))

    def test_ensure_hammer_and_axe_are_ram_shelf_not_recorded(self) -> None:
        hammer = ensure_hammer_phase()
        axe = ensure_axe_phase()
        self.assertEqual(hammer.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(axe.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(hammer.params["tool_id"], int(Tool.HAMMER))
        self.assertEqual(axe.params["tool_id"], int(Tool.AXE))
        self.assertNotEqual(hammer.kind, PhaseKind.RECORDED)


class D2LeftoverOrderTests(unittest.TestCase):
    def test_low_stam_inserts_spa_before_hammer_work(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=8, maximum=100))
        names = [p.phase for p in phases]
        self.assertEqual(names[0], "HOT_SPRING_STAMINA")
        self.assertLess(names.index("CLEAR_BUSHES"), names.index("CLEAR_FENCES"))
        self.assertLess(names.index("CLEAR_FENCES"), names.index("CLEAR_STONES"))
        self.assertLess(names.index("CLEAR_STONES"), names.index("ENSURE_HAMMER"))
        self.assertLess(names.index("ENSURE_HAMMER"), names.index("CLEAR_ROCKS"))
        self.assertLess(names.index("CLEAR_ROCKS"), names.index("ENSURE_AXE"))
        self.assertLess(names.index("ENSURE_AXE"), names.index("CLEAR_STUMPS"))
        self.assertNotIn("CLEAR_FIELD", names)

    def test_full_stam_skips_spa_but_keeps_smash_order(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=100, maximum=100))
        names = [p.phase for p in phases]
        self.assertNotIn("HOT_SPRING_STAMINA", names)
        self.assertEqual(
            names,
            [
                "CLEAR_BUSHES",
                "CLEAR_FENCES",
                "CLEAR_STONES",
                "ENSURE_HAMMER",
                "CLEAR_ROCKS",
                "ENSURE_AXE",
                "CLEAR_STUMPS",
            ],
        )

    def test_policy_can_drop_leftover(self) -> None:
        phases = d2_leftover_phases(
            stamina=Stamina(current=4, maximum=100),
            policy=DayPlannerPolicy(include_field_clear=False),
        )
        self.assertEqual(phases, [])

    def test_hammer_and_axe_are_sequential_not_same_carry(self) -> None:
        names = [p.phase for p in d2_leftover_phases()]
        self.assertLess(names.index("CLEAR_ROCKS"), names.index("ENSURE_AXE"))
        self.assertEqual(names.count("ENSURE_HAMMER"), 1)
        self.assertEqual(names.count("ENSURE_AXE"), 1)

    def test_stamina_low_rocks_retry_inserts_spa(self) -> None:
        low = Stamina(current=8, maximum=100)
        self.assertTrue(
            should_spa_retry("CLEAR_ROCKS", "stamina_low cleared=2", low, include_spa=True)
        )
        self.assertFalse(
            should_spa_retry("CLEAR_ROCKS", "stamina_low cleared=2", low, include_spa=False)
        )
        self.assertFalse(
            should_spa_retry("CLEAR_STONES", "stamina_low", low, include_spa=True)
        )
        self.assertFalse(
            should_spa_retry(
                "CLEAR_ROCKS",
                "partial_clear remaining=2",
                low,
                include_spa=True,
            )
        )
        self.assertFalse(
            should_spa_retry(
                "CLEAR_STUMPS",
                "stamina_low",
                Stamina(current=40, maximum=100),
                include_spa=True,
            )
        )

    def test_after_rocks_spa_when_stumps_remain(self) -> None:
        low = Stamina(current=10, maximum=100)
        self.assertTrue(
            needs_spa_before_next_smash(
                "CLEAR_ROCKS",
                low,
                include_spa=True,
                remaining_phases=("ENSURE_AXE", "CLEAR_STUMPS"),
            )
        )
        self.assertFalse(
            needs_spa_before_next_smash(
                "CLEAR_ROCKS",
                Stamina(current=40, maximum=100),
                include_spa=True,
                remaining_phases=("ENSURE_AXE", "CLEAR_STUMPS"),
            )
        )
        self.assertFalse(
            needs_spa_before_next_smash(
                "CLEAR_ROCKS",
                low,
                include_spa=True,
                remaining_phases=(),
            )
        )


class D2PostShopComposeTests(unittest.TestCase):
    def test_post_shop_is_plant_water_then_leftover(self) -> None:
        names = [p.phase for p in d2_post_shop_work_phases()]
        self.assertEqual(
            names[:6],
            [
                "ENSURE_CROP_SEEDS",
                "CLEAR_PLOT",
                "NAV_CROP",
                "CROP_ESTABLISH",
                "ENSURE_WATERING_CAN",
                "CROP_WATER",
            ],
        )
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_BUSHES"))
        self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
        self.assertNotIn("CLEAR_FIELD", names)
        self.assertEqual(pocket_water_phase().params["work_mode"], "pocket")
        self.assertEqual(pocket_water_phase().params["min_wet"], 8)

    def test_pocket_plant_phases_delegate_to_d2_work(self) -> None:
        plant = [p.phase for p in pocket_plant_phases()]
        composed = [p.phase for p in d2_post_shop_work_phases()]
        self.assertEqual(plant, composed)

    def test_leftover_already_queued(self) -> None:
        self.assertTrue(leftover_already_queued(["CROP_WATER", "CLEAR_ROCKS"]))
        self.assertTrue(leftover_already_queued(["CLEAR_FENCES", "RETURN_HOME"]))
        self.assertTrue(leftover_already_queued(["CLEAR_STONES"]))
        self.assertFalse(leftover_already_queued(["CROP_WATER", "RETURN_HOME"]))
        self.assertFalse(leftover_already_queued(["HOT_SPRING_STAMINA"]))

    def test_fence_dump_builder_dumps_all_posts(self) -> None:
        import numpy as np
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
        from harvest.tasks.fence_flow import FenceClearLoopTask
        from retro_harness import WorldState

        ram = np.zeros(0x20000, dtype=np.uint8)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = build_phase_task(TaskBuildContext(), fence_dump_phase(), world)
        self.assertIsInstance(task, FenceClearLoopTask)
        self.assertIsNone(task.max_fences)
        self.assertFalse(task.corridor_only)
        self.assertTrue(task.pond_dump)
        self.assertEqual(task.max_steps_per_fence, 2800)
        self.assertEqual(task.max_failures, 20)
        self.assertEqual(task.debris_types[0].name, "FENCE")

        stones = build_phase_task(TaskBuildContext(), stone_pond_phase(), world)
        self.assertIsInstance(stones, FenceClearLoopTask)
        self.assertIsNone(stones.max_fences)
        self.assertTrue(stones.pond_dump)
        self.assertEqual(stones.max_steps_per_fence, 2800)
        self.assertEqual(stones.max_failures, 60)
        self.assertEqual(stones.debris_types[0].name, "STONE")


class LeftoverSkipClearTests(unittest.TestCase):
    def test_skip_bushes_when_weeds_already_gone(self) -> None:
        from harvest.scripts.leftover_exec import phase_already_clear
        from harvest.tasks.farm_clear_quota import DebrisCounts

        empty = DebrisCounts(stones=45, large_rocks=47, stumps=36)
        self.assertTrue(phase_already_clear("CLEAR_BUSHES", empty))
        self.assertTrue(phase_already_clear("CLEAR_FENCES", empty))
        self.assertFalse(phase_already_clear("CLEAR_STONES", empty))
        self.assertFalse(phase_already_clear("CLEAR_ROCKS", empty))
        self.assertFalse(phase_already_clear("ENSURE_HAMMER", empty))


class LeftoverProbeBudgetTests(unittest.TestCase):
    def test_probe_section_uses_day2_quotas_not_whole_farm_empty(self) -> None:
        from harvest.scripts.d2_leftover_probe import _section_complete
        from harvest.tasks.farm_clear_quota import DebrisCounts

        start = DebrisCounts(
            weeds=100, stones=185, large_rocks=51, stumps=38, fences=80
        )
        enough = DebrisCounts(
            weeds=90, stones=0, large_rocks=0, stumps=36, fences=0
        )
        short = DebrisCounts(
            weeds=90, stones=1, large_rocks=0, stumps=36, fences=0
        )

        self.assertTrue(_section_complete("all", start, enough))
        self.assertFalse(_section_complete("all", start, short))

    def test_probe_fence_quota_is_exhaustive(self) -> None:
        from harvest.scripts.d2_leftover_probe import _section_complete
        from harvest.tasks.farm_clear_quota import DebrisCounts

        start = DebrisCounts(fences=80)
        self.assertTrue(_section_complete("fences", start, DebrisCounts()))
        self.assertFalse(
            _section_complete("fences", start, DebrisCounts(fences=1))
        )

    def test_probe_stone_quota_is_exhaustive(self) -> None:
        from harvest.scripts.d2_leftover_probe import _section_complete
        from harvest.tasks.farm_clear_quota import DebrisCounts

        start = DebrisCounts(stones=175)
        self.assertTrue(_section_complete("stones", start, DebrisCounts()))
        self.assertFalse(
            _section_complete("stones", start, DebrisCounts(stones=1))
        )

    def test_zero_phase_timeout_spends_remaining_budget(self) -> None:
        from harvest.scripts.d2_leftover_probe import _phase_timeout

        remaining = 200_000
        self.assertEqual(_phase_timeout(bush_clear_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(fence_dump_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(stone_pond_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(rock_clear_phase(), remaining), remaining)

    def test_positive_phase_timeout_is_capped_by_remaining(self) -> None:
        from harvest.scripts.d2_leftover_probe import _phase_timeout

        self.assertEqual(_phase_timeout(stump_clear_phase(), 50_000), 50_000)
        self.assertEqual(_phase_timeout(stump_clear_phase(), 200_000), 120_000)

    def test_leftover_probe_uses_repo_headed(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "harvest"
            / "scripts"
            / "d2_leftover_probe.py"
        )
        text = src.read_text(encoding="utf-8")
        self.assertIn("from retro_harness.headed import", text)
        self.assertIn("add_headed_flag", text)
        self.assertIn("attach_headed", text)
        self.assertIn("idle_headed", text)
        exec_src = src.parent / "leftover_exec.py"
        self.assertIn("headed_emu_repeat", exec_src.read_text(encoding="utf-8"))
        self.assertNotIn("WatchDisplay", text)
        self.assertNotIn("--watch", text)


class LeftoverProbePayloadTests(unittest.TestCase):
    def test_fail_payload_always_has_leftover_and_glance_misses(self) -> None:
        from harvest.clock_glance import FENCE_STAND, leftover_json
        from harvest.scripts.d2_leftover_probe import leftover_json as probe_leftover_json

        self.assertIs(probe_leftover_json, leftover_json)
        snap = {
            "tilemap": "0x0",
            "pos": [86, 69],
            "tile": [5, 4],
            "clock": {"hour": 18, "minute": 6, "clock": "18:06"},
            "carry": {"selected": 16, "backpack": 2},
            "debris": {
                "weeds": 0,
                "stones": 185,
                "small_rocks": 0,
                "large_rocks": 51,
                "stumps": 38,
                "fences": 80,
            },
        }
        fail = leftover_json(
            snap,
            FENCE_STAND,
            ok=False,
            journal=[{"phase": "CLEAR_FENCES", "status": "failed"}],
            partial=True,
            section="fences",
        )
        self.assertFalse(fail["ok"])
        self.assertIn("leftover", fail)
        self.assertIn("final", fail)
        self.assertIn("glance_misses", fail)
        self.assertEqual(fail["leftover"]["tilemap"], 0)
        self.assertEqual(fail["leftover"]["hour"], 18)
        self.assertEqual(fail["leftover"]["debris"]["fences"], 80)
        self.assertEqual(fail["glance_misses"], [])
        exit_fail = leftover_json(
            {"tilemap": "0x15", "clock": {"hour": 6, "minute": 8, "clock": "06:08"}},
            FENCE_STAND,
            ok=False,
            journal=[{"phase": "exit_to_farm"}],
        )
        self.assertIn("leftover", exit_fail)
        self.assertIn("glance_misses", exit_fail)
        self.assertTrue(exit_fail["glance_misses"])
        self.assertEqual(exit_fail["leftover"]["tilemap"], 0x15)


class LeftoverStallAbortTests(unittest.TestCase):
    def test_stall_aborts_after_unchanged_window(self) -> None:
        from harvest.scripts.leftover_exec import _should_abort_stall

        self.assertTrue(_should_abort_stall(24_000, 0, 24_000))
        self.assertFalse(_should_abort_stall(23_999, 0, 24_000))

    def test_progress_resets_the_stall_timer(self) -> None:
        from harvest.scripts.leftover_exec import _should_abort_stall

        self.assertFalse(_should_abort_stall(24_000, 1_000, 24_000))
        self.assertTrue(_should_abort_stall(25_000, 1_000, 24_000))

    def test_nonpositive_stall_frames_never_aborts(self) -> None:
        from harvest.scripts.leftover_exec import _should_abort_stall

        self.assertFalse(_should_abort_stall(400_001, 0, 0))
        self.assertFalse(_should_abort_stall(400_001, 0, -1))

    def _idle_run(self, *, stall_frames, timeout, key_fn=None, checkpoint_state=None):
        from unittest.mock import patch

        import numpy as np
        from retro_harness import TaskResult, TaskStatus

        from harvest.scripts.leftover_exec import run_leftover_task

        class Env:
            def __init__(self):
                self._ram = np.zeros(8, dtype=np.uint8)
                self.n_steps = 0

            def get_ram(self):
                return self._ram

            def step(self, action):
                self.n_steps += 1
                return None, 0.0, False, False, {}

        class Task:
            def step(self, world):
                return TaskResult(status=TaskStatus.RUNNING)

        env = Env()
        keys = (lambda _ram: key_fn(env)) if key_fn else (lambda _ram: (0,))
        with (
            patch("harvest.scripts.leftover_exec._debris_key", keys),
            patch(
                "harvest.scripts.leftover_exec.shipping_scene_needs_dismiss",
                return_value=False,
            ),
            patch("harvest.scripts.leftover_exec.save_emulator_state") as save,
        ):
            frame, result, _ram = run_leftover_task(
                env,
                Task(),
                timeout=timeout,
                start_frame=0,
                checkpoint_state=checkpoint_state,
                stall_frames=stall_frames,
            )
        return frame, result, env, save

    def test_run_leftover_task_stops_on_stall(self) -> None:
        from retro_harness import TaskStatus

        frame, result, env, save = self._idle_run(
            stall_frames=60,
            timeout=400,
            checkpoint_state="Y1_D2_Leftover_Checkpoint",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("no debris progress", result.reason)
        self.assertIn("60", result.reason)
        self.assertLess(frame, 400)
        self.assertLessEqual(env.n_steps, 120)
        save.assert_called_once()

    def test_run_leftover_task_progress_delays_abort(self) -> None:
        from retro_harness import TaskStatus

        frame, result, env, _save = self._idle_run(
            stall_frames=100,
            timeout=250,
            key_fn=lambda e: (1,) if e.n_steps >= 60 else (0,),
        )
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("no debris progress", result.reason)
        self.assertGreaterEqual(frame, 160)
        self.assertLess(frame, 250)
        self.assertGreater(env.n_steps, 100)

    def test_run_leftover_task_disabled_stall_runs_to_timeout(self) -> None:
        from retro_harness import TaskStatus

        frame, result, env, save = self._idle_run(stall_frames=0, timeout=80)
        self.assertNotEqual(result.status, TaskStatus.FAILURE)
        self.assertGreater(env.n_steps, 60)
        self.assertGreater(frame, 80)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
