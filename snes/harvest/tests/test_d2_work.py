"""D2 field-work composition — bounded quotas and exhaustive fences."""

from __future__ import annotations

import unittest

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import Tool
from harvest.planner.d2_farm_chunks import EXHAUSTIVE, FARM_CHUNK_ORDER
from harvest.planner.d2_work import (
    D2_TARGETS,
    bush_clear_phase,
    d2_leftover_phases,
    d2_post_shop_work_phases,
    ensure_axe_phase,
    ensure_hammer_phase,
    fence_dump_phase,
    leftover_already_queued,
    leftover_section_phases,
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

    def test_bush_phase_is_exhaustive_quota_not_plot_ring(self) -> None:
        spec = bush_clear_phase()
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.kind, PhaseKind.CLEAR_FIELD)
        self.assertEqual(spec.params["handoff"], "quota")
        self.assertEqual(spec.params["quota"], {"weeds": EXHAUSTIVE})
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
        self.assertEqual(spec.params["quota"], {"stumps": EXHAUSTIVE})
        self.assertEqual(spec.params["timeout"], 0)
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
                *["CLEAR_STONES"] * 4,
                "ENSURE_HAMMER",
                *["CLEAR_ROCKS"] * 4,
                "ENSURE_AXE",
                *["CLEAR_STUMPS"] * 4,
            ],
        )
        stones = [p for p in phases if p.phase == "CLEAR_STONES"]
        self.assertEqual([p.params["chunk"] for p in stones], list(FARM_CHUNK_ORDER))

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
        self.assertTrue(
            needs_spa_before_next_smash(
                "CLEAR_STUMPS",
                low,
                include_spa=True,
                remaining_phases=("CLEAR_STUMPS",),
            )
        )
        self.assertFalse(
            needs_spa_before_next_smash(
                "CLEAR_STUMPS",
                Stamina(current=40, maximum=100),
                include_spa=True,
                remaining_phases=("CLEAR_STUMPS",),
            )
        )
        self.assertFalse(
            needs_spa_before_next_smash(
                "CLEAR_STUMPS",
                low,
                include_spa=True,
                remaining_phases=(),
            )
        )
        from harvest.planner.d2_work import leftover_chain_decision
        from retro_harness import TaskStatus

        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_STUMPS",
                TaskStatus.SUCCESS,
                "quota met",
                low,
                (),
            ),
            "continue",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_STUMPS",
                TaskStatus.SUCCESS,
                "quota met",
                low,
                ("CLEAR_STUMPS",),
            ),
            "insert_spa",
        )


class D2PostShopComposeTests(unittest.TestCase):
    def test_post_shop_is_plant_water_then_leftover(self) -> None:
        phases = d2_post_shop_work_phases()
        names = [p.phase for p in phases]
        self.assertEqual(names, ["D2_FARM_CLEAR"])
        self.assertEqual(phases[0].kind, PhaseKind.CLEAR_FIELD)
        self.assertEqual(phases[0].failure_policy, "required")
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
        self.assertTrue(leftover_already_queued(["D2_FARM_CLEAR"]))
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
        self.assertIsNone(stones.farm_bounds)

        sw = leftover_section_phases("stones", chunk="sw")[0]
        sw_task = build_phase_task(TaskBuildContext(), sw, world)
        self.assertEqual(sw.params["chunk"], "sw")
        self.assertEqual(sw_task.farm_bounds, (0, 32, 31, 63))


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
    def test_probe_section_all_requires_weeds_gone(self) -> None:
        from harvest.scripts.d2_leftover_probe import _section_complete
        from harvest.tasks.farm_clear_quota import DebrisCounts

        start = DebrisCounts(
            weeds=100, stones=185, large_rocks=51, stumps=38, fences=80
        )
        leftover_weeds = DebrisCounts(
            weeds=90, stones=0, large_rocks=0, stumps=0, fences=0
        )
        short = DebrisCounts(
            weeds=90, stones=1, large_rocks=0, stumps=0, fences=0
        )
        leftover_stumps = DebrisCounts(
            weeds=90, stones=0, large_rocks=0, stumps=36, fences=0
        )

        self.assertTrue(_section_complete("all", start, DebrisCounts()))
        self.assertFalse(_section_complete("all", start, leftover_weeds))
        self.assertFalse(_section_complete("all", start, short))
        self.assertFalse(_section_complete("all", start, leftover_stumps))

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
        self.assertEqual(_phase_timeout(stump_clear_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(stump_clear_phase(), 50_000), 50_000)

    def test_probe_stump_quota_is_exhaustive(self) -> None:
        from harvest.scripts.d2_leftover_probe import _section_complete
        from harvest.tasks.farm_clear_quota import DebrisCounts

        start = DebrisCounts(stumps=38)
        self.assertTrue(_section_complete("stumps", start, DebrisCounts()))
        self.assertFalse(
            _section_complete("stumps", start, DebrisCounts(stumps=1))
        )

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
        self.assertNotIn("spa_retried", text)
        self.assertIn("D2FarmClearTactic", text)
        exec_text = exec_src.read_text(encoding="utf-8")
        self.assertIn("leftover_chain_decision", exec_text)
        d2_src = (
            Path(__file__).resolve().parents[1] / "harvest" / "planner" / "d2_work.py"
        ).read_text(encoding="utf-8")
        self.assertIn("should_spa_retry", d2_src)
        self.assertIn("D2FarmClearTactic", d2_src)
        self.assertIn("--chunk", text)
        self.assertIn("--no-spa", text)


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
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(result.reason, "phase timeout 80f")
        self.assertGreater(env.n_steps, 60)
        self.assertGreater(frame, 80)
        save.assert_not_called()


_SHIP_OK = [{"phase": "WAIT_FARM_SHIPPING", "status": "success"}]


def _farm_ram(*, stamina=100, lock=1, hour=12, player=(10, 10)):
    import numpy as np
    from harvest.core.ram_catalog import field_spec
    from harvest.core.tile_catalog import (
        ADDR_INPUT_LOCK,
        ADDR_MAP,
        ADDR_STAMINA,
        ADDR_TILEMAP,
        ADDR_X,
        ADDR_Y,
        MAP_WIDTH,
        TILE_SIZE,
    )

    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = lock
    ram[ADDR_STAMINA] = stamina
    ram[field_spec("max_stamina").address] = 100
    ram[field_spec("hour").address] = hour
    ram[field_spec("day").address] = 2
    for i in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + i] = 0xA1
    px, py = player[0] * TILE_SIZE + 8, player[1] * TILE_SIZE + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF
    return ram


def _set_tile(ram, tx, ty, tile_id):
    from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH

    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_large_rock(ram, tx, ty, *, damage=False):
    ids = (0x11, 0x12, 0x13, 0x14) if damage else (0x0D, 0x0E, 0x0F, 0x10)
    for (dx, dy), tid in zip(((0, 0), (1, 0), (0, 1), (1, 1)), ids):
        _set_tile(ram, tx + dx, ty + dy, tid)


def _place_stump(ram, tx, ty):
    for (dx, dy), tid in zip(((0, 0), (1, 0), (0, 1), (1, 1)), (0x09, 0x0A, 0x0B, 0x0C)):
        _set_tile(ram, tx + dx, ty + dy, tid)


def _clear_2x2(ram, tx, ty):
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        _set_tile(ram, tx + dx, ty + dy, 0xA1)


# Wood_Progress leftover: 5 stumps, last live chunks (ne empty).
_LAST_STUMPS = ((4, 20), (12, 8), (20, 24), (8, 48), (52, 44))


def _plant_eight_wet(ram):
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_geometry import plot_tiles
    from harvest.tasks.crop_skills import PLANTED_WET

    cx, cy = WEST_POCKET_PLANT_CENTER
    for tx, ty in plot_tiles((cx, cy), include_center=False):
        _set_tile(ram, tx, ty, PLANTED_WET)


class D2ObserveTruthTableTests(unittest.TestCase):
    def test_complete_only_when_all_adr_clauses_hold(self) -> None:
        from harvest.planner.d2_work import (
            D2FarmOutcome,
            confirm_d2_complete,
            observe_d2_farm,
        )

        ram = _farm_ram()
        _plant_eight_wet(ram)
        status = observe_d2_farm(ram, _SHIP_OK)
        self.assertEqual(status.outcome, D2FarmOutcome.COMPLETE)
        self.assertTrue(status.is_complete)
        self.assertEqual(status.planted, 8)
        self.assertEqual(status.wet, 8)
        self.assertFalse(status.damaged_boulder)
        self.assertTrue(status.hands_clear)
        self.assertTrue(status.farm_map_loaded)
        self.assertFalse(status.animating)
        self.assertTrue(status.shipped_before_17)

        dry = _farm_ram()
        from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
        from harvest.tasks.crop_geometry import plot_tiles
        from harvest.tasks.crop_skills import PLANTED_DRY

        cx, cy = WEST_POCKET_PLANT_CENTER
        for tx, ty in plot_tiles((cx, cy), include_center=False):
            _set_tile(dry, tx, ty, PLANTED_DRY)
        self.assertFalse(observe_d2_farm(dry, _SHIP_OK).is_complete)

        weed = _farm_ram()
        _plant_eight_wet(weed)
        _set_tile(weed, 20, 20, 0x03)
        self.assertFalse(observe_d2_farm(weed, _SHIP_OK).is_complete)

        dmg = _farm_ram()
        _plant_eight_wet(dmg)
        _place_large_rock(dmg, 50, 50, damage=True)
        hit = observe_d2_farm(dmg, _SHIP_OK)
        self.assertTrue(hit.damaged_boulder)
        self.assertNotEqual(hit.outcome, D2FarmOutcome.COMPLETE)

        stale = _farm_ram()
        _plant_eight_wet(stale)
        from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH

        for i in range(MAP_WIDTH * MAP_WIDTH):
            stale[ADDR_MAP + i] = 0xFF
        self.assertEqual(
            observe_d2_farm(stale, _SHIP_OK).outcome,
            D2FarmOutcome.TEMPORARILY_UNOBSERVABLE,
        )
        self.assertFalse(observe_d2_farm(stale, _SHIP_OK).is_complete)

        swinging = _farm_ram(lock=0)
        _plant_eight_wet(swinging)
        self.assertEqual(
            observe_d2_farm(swinging, _SHIP_OK).outcome,
            D2FarmOutcome.TEMPORARILY_UNOBSERVABLE,
        )

        hands = _farm_ram()
        _plant_eight_wet(hands)
        from harvest.core.ram_catalog import field_spec, live_wram_base
        from harvest.planner.tasks.transitions import PLAYER_STATE_CARRYING_BIT

        idx = field_spec("player_state").address + live_wram_base(hands)
        hands[idx] = PLAYER_STATE_CARRYING_BIT
        self.assertFalse(observe_d2_farm(hands, _SHIP_OK).is_complete)

        noship = _farm_ram()
        _plant_eight_wet(noship)
        self.assertFalse(observe_d2_farm(noship).is_complete)
        self.assertFalse(observe_d2_farm(noship, []).shipped_before_17)

        late = _farm_ram(hour=18)
        _plant_eight_wet(late)
        self.assertFalse(observe_d2_farm(late).shipped_before_17)
        self.assertFalse(observe_d2_farm(late).is_complete)
        from harvest.core.ram_catalog import field_spec

        late[field_spec("shipping_money_raw").address] = 1
        self.assertTrue(observe_d2_farm(late).shipped_before_17)
        self.assertTrue(observe_d2_farm(late).is_complete)

        done = observe_d2_farm(ram, _SHIP_OK)
        self.assertTrue(confirm_d2_complete(done, done))
        self.assertFalse(confirm_d2_complete(None, done))
        self.assertFalse(confirm_d2_complete(observe_d2_farm(weed, _SHIP_OK), done))

        last = _farm_ram(hour=18)
        _plant_eight_wet(last)
        last[field_spec("shipping_money_raw").address] = 1
        _place_stump(last, 52, 44)
        leftover = observe_d2_farm(last)
        self.assertTrue(leftover.shipped_before_17)
        self.assertEqual(leftover.stumps, 1)
        self.assertEqual(leftover.stumps_by_chunk, (0, 0, 0, 1))
        self.assertFalse(leftover.is_complete)

        for tx, ty in _LAST_STUMPS:
            _place_stump(last, tx, ty)
        five = observe_d2_farm(last)
        self.assertEqual(five.stumps, 5)
        self.assertEqual(five.stumps_by_chunk, (3, 0, 1, 1))
        self.assertFalse(five.is_complete)


class D2NextSpecTests(unittest.TestCase):
    def test_empty_rock_chunk_is_omitted(self) -> None:
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        ram = _farm_ram()
        _place_large_rock(ram, 60, 51)
        status = observe_d2_farm(ram)
        spec = next_d2_spec(status, section="rocks", last_phase="ENSURE_HAMMER")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.phase, "CLEAR_ROCKS")
        self.assertEqual(spec.params["chunk"], "se")

        empty_se = _farm_ram()
        _place_large_rock(empty_se, 8, 18)
        empty_status = observe_d2_farm(empty_se)
        skipped = next_d2_spec(
            empty_status, section="rocks", chunk="se", last_phase="ENSURE_HAMMER"
        )
        self.assertIsNone(skipped)
        nw = next_d2_spec(
            empty_status, section="rocks", chunk="nw", last_phase="ENSURE_HAMMER"
        )
        self.assertEqual(nw.phase, "CLEAR_ROCKS")
        self.assertEqual(nw.params["chunk"], "nw")

    def test_live_stamina_inserts_spa_before_rocks(self) -> None:
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        low = _farm_ram(stamina=8)
        _place_large_rock(low, 50, 50)
        low_status = observe_d2_farm(low)
        self.assertEqual(low_status.stamina.current, 8)
        spec = next_d2_spec(low_status, section="rocks")
        self.assertEqual(spec.phase, "HOT_SPRING_STAMINA")

        full = _farm_ram(stamina=100)
        _place_large_rock(full, 50, 50)
        full_status = observe_d2_farm(full)
        spec = next_d2_spec(full_status, section="rocks")
        self.assertEqual(spec.phase, "ENSURE_HAMMER")
        spec = next_d2_spec(full_status, section="rocks", last_phase="ENSURE_HAMMER")
        self.assertEqual(spec.phase, "CLEAR_ROCKS")

    def test_empty_stump_chunk_is_omitted(self) -> None:
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        ram = _farm_ram()
        _place_stump(ram, 52, 44)
        status = observe_d2_farm(ram)
        spec = next_d2_spec(status, section="stumps", last_phase="ENSURE_AXE")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.phase, "CLEAR_STUMPS")
        self.assertEqual(spec.params["chunk"], "se")

        empty_se = _farm_ram()
        _place_stump(empty_se, 4, 20)
        empty_status = observe_d2_farm(empty_se)
        skipped = next_d2_spec(
            empty_status, section="stumps", chunk="se", last_phase="ENSURE_AXE"
        )
        self.assertIsNone(skipped)
        nw = next_d2_spec(
            empty_status, section="stumps", chunk="nw", last_phase="ENSURE_AXE"
        )
        self.assertEqual(nw.phase, "CLEAR_STUMPS")
        self.assertEqual(nw.params["chunk"], "nw")

    def test_five_remaining_stumps_select_only_live_chunks(self) -> None:
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        ram = _farm_ram()
        _plant_eight_wet(ram)
        for tx, ty in _LAST_STUMPS:
            _place_stump(ram, tx, ty)
        status = observe_d2_farm(ram, _SHIP_OK)
        self.assertFalse(status.is_complete)
        self.assertEqual(status.stumps_by_chunk, (3, 0, 1, 1))

        first = next_d2_spec(status, last_phase="ENSURE_AXE")
        self.assertEqual(first.phase, "CLEAR_STUMPS")
        self.assertEqual(first.params["chunk"], "nw")
        skipped_ne = next_d2_spec(
            status, last_phase="CLEAR_STUMPS", skip_chunks=("nw",)
        )
        self.assertEqual(skipped_ne.phase, "CLEAR_STUMPS")
        self.assertEqual(skipped_ne.params["chunk"], "sw")
        last = next_d2_spec(
            status, last_phase="CLEAR_STUMPS", skip_chunks=("nw", "sw")
        )
        self.assertEqual(last.phase, "CLEAR_STUMPS")
        self.assertEqual(last.params["chunk"], "se")
        none = next_d2_spec(
            status, last_phase="CLEAR_STUMPS", skip_chunks=("nw", "sw", "se")
        )
        self.assertIsNone(none)
        se_only = next_d2_spec(
            status, section="stumps", chunk="se", last_phase="ENSURE_AXE"
        )
        self.assertEqual(se_only.params["chunk"], "se")
        ne_only = next_d2_spec(
            status, section="stumps", chunk="ne", last_phase="ENSURE_AXE"
        )
        self.assertIsNone(ne_only)


class D2FarmClearTacticTests(unittest.TestCase):
    def test_complete_ram_succeeds_after_settle(self) -> None:
        from retro_harness import TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram()
        _plant_eight_wet(ram)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        tactic = D2FarmClearTactic(evidence=_SHIP_OK)
        tactic.reset(world)
        first = tactic.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        second = tactic.step(world)
        self.assertEqual(second.status, TaskStatus.SUCCESS)
        self.assertTrue(tactic.farm_status.is_complete)

    def test_skips_empty_se_rock_chunk(self) -> None:
        from unittest.mock import patch

        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram()
        _place_large_rock(ram, 8, 18)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        seen = []

        class Instant:
            def reset(self, _world) -> None:
                return None

            def step(self, _world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="ok")

        def fake_build(_ctx, spec, _world):
            seen.append((spec.phase, (spec.params or {}).get("chunk")))
            if spec.phase == "CLEAR_ROCKS" and spec.params.get("chunk") == "se":
                return None
            return Instant()

        tactic = D2FarmClearTactic(section="rocks", chunk="all", include_spa=False)
        tactic.reset(world)
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(12):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break
        self.assertNotIn(("CLEAR_ROCKS", "se"), seen)
        self.assertIn(("CLEAR_ROCKS", "nw"), seen)
        self.assertNotEqual(result.status, TaskStatus.FAILURE)

    def test_stale_west_gate_walks_into_yard_not_idle(self) -> None:
        from retro_harness import TaskStatus, WorldState

        from harvest.core.tile_catalog import ADDR_INPUT_LOCK
        from harvest.planner.d2_work import D2FarmClearTactic, D2FarmOutcome
        from harvest.tasks.farm_clear_quota import farm_map_loaded, yard_load_action
        from harvest.tasks.nav import make_action

        ram = _farm_ram(player=(1, 28))
        ram[ADDR_INPUT_LOCK] = 1
        _set_tile(ram, 1, 28, 0xFF)
        self.assertFalse(farm_map_loaded(ram))
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        tactic = D2FarmClearTactic(section="stumps", chunk="se", include_spa=False)
        tactic.reset(world)
        result = tactic.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(tactic.farm_status.outcome, D2FarmOutcome.TEMPORARILY_UNOBSERVABLE)
        self.assertEqual(tactic.farm_status.reason, "stale_farm_map")
        self.assertIsNotNone(result.action)
        self.assertTrue((result.action.action == yard_load_action(ram)).all())
        self.assertFalse((result.action.action == make_action()).all())

    def test_leftover_exec_still_exports_spa_retry(self) -> None:
        from retro_harness import TaskStatus

        from harvest.scripts.leftover_exec import leftover_chain_decision

        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.FAILURE,
                "stamina_low cleared=2",
                Stamina(current=8, maximum=100),
                ("ENSURE_AXE", "CLEAR_STUMPS"),
            ),
            "spa_retry",
        )

    def test_skips_empty_stump_chunks_and_clears_last_live(self) -> None:
        from unittest.mock import patch

        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram()
        _place_stump(ram, 52, 44)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        seen = []

        class Instant:
            def reset(self, _world) -> None:
                return None

            def step(self, _world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="ok")

        def fake_build(_ctx, spec, _world):
            seen.append((spec.phase, (spec.params or {}).get("chunk")))
            if spec.phase == "CLEAR_STUMPS" and spec.params.get("chunk") != "se":
                return None
            return Instant()

        tactic = D2FarmClearTactic(section="stumps", chunk="all", include_spa=False)
        tactic.reset(world)
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(16):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break
        self.assertNotIn(("CLEAR_STUMPS", "nw"), seen)
        self.assertNotIn(("CLEAR_STUMPS", "ne"), seen)
        self.assertNotIn(("CLEAR_STUMPS", "sw"), seen)
        self.assertIn(("CLEAR_STUMPS", "se"), seen)
        self.assertNotEqual(result.status, TaskStatus.FAILURE)

    def test_last_stump_success_settles_complete_without_spa(self) -> None:
        from unittest.mock import patch

        from harvest.core.ram_catalog import field_spec
        from harvest.core.tile_catalog import ADDR_STAMINA
        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram(stamina=100)
        _plant_eight_wet(ram)
        ram[field_spec("shipping_money_raw").address] = 1
        _place_stump(ram, 52, 44)
        seen = []

        class Instant:
            def __init__(self, spec) -> None:
                self.spec = spec

            def reset(self, _world) -> None:
                return None

            def step(self, world):
                if self.spec.phase == "CLEAR_STUMPS":
                    _clear_2x2(world.ram, 52, 44)
                    world.ram[ADDR_STAMINA] = 8
                return TaskResult(status=TaskStatus.SUCCESS, reason="quota met")

        def fake_build(_ctx, spec, _world):
            seen.append((spec.phase, (spec.params or {}).get("chunk")))
            return Instant(spec)

        tactic = D2FarmClearTactic(
            section="all", chunk="all", include_spa=True, evidence=_SHIP_OK
        )
        tactic.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        result = None
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(20):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break
        self.assertEqual([phase for phase, _chunk in seen], ["ENSURE_AXE", "CLEAR_STUMPS"])
        self.assertEqual(seen[-1], ("CLEAR_STUMPS", "se"))
        self.assertNotIn("HOT_SPRING_STAMINA", [phase for phase, _ in seen])
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertTrue(tactic.farm_status.is_complete)
        self.assertEqual(tactic.farm_status.stumps, 0)

    def test_se_stump_chunk_success_is_not_whole_farm_complete(self) -> None:
        from unittest.mock import patch

        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic, observe_d2_farm

        ram = _farm_ram()
        _plant_eight_wet(ram)
        for tx, ty in _LAST_STUMPS:
            _place_stump(ram, tx, ty)
        seen = []

        class Instant:
            def __init__(self, spec) -> None:
                self.spec = spec

            def reset(self, _world) -> None:
                return None

            def step(self, world):
                if self.spec.phase == "CLEAR_STUMPS":
                    _clear_2x2(world.ram, 52, 44)
                return TaskResult(status=TaskStatus.SUCCESS, reason="quota met")

        def fake_build(_ctx, spec, _world):
            seen.append((spec.phase, (spec.params or {}).get("chunk")))
            return Instant(spec)

        tactic = D2FarmClearTactic(section="stumps", chunk="se", include_spa=False)
        tactic.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        result = None
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(16):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn(("CLEAR_STUMPS", "se"), seen)
        self.assertNotIn(("CLEAR_STUMPS", "nw"), seen)
        farm = observe_d2_farm(ram, _SHIP_OK)
        self.assertEqual(farm.stumps, 4)
        self.assertFalse(farm.is_complete)

    def test_mid_stump_chunks_still_spa_when_more_remain(self) -> None:
        from unittest.mock import patch

        from harvest.core.tile_catalog import ADDR_STAMINA
        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram(stamina=100)
        _place_stump(ram, 4, 20)
        _place_stump(ram, 52, 44)
        seen = []

        class Instant:
            def __init__(self, spec) -> None:
                self.spec = spec

            def reset(self, _world) -> None:
                return None

            def step(self, world):
                if self.spec.phase == "CLEAR_STUMPS" and self.spec.params.get("chunk") == "nw":
                    _clear_2x2(world.ram, 4, 20)
                    world.ram[ADDR_STAMINA] = 8
                return TaskResult(status=TaskStatus.SUCCESS, reason="quota met")

        def fake_build(_ctx, spec, _world):
            seen.append((spec.phase, (spec.params or {}).get("chunk")))
            return Instant(spec)

        tactic = D2FarmClearTactic(section="stumps", chunk="all", include_spa=True)
        tactic.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(12):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break
                if ("HOT_SPRING_STAMINA", None) in seen:
                    break
        self.assertIn(("CLEAR_STUMPS", "nw"), seen)
        self.assertIn(("HOT_SPRING_STAMINA", None), seen)
        self.assertNotIn(("CLEAR_STUMPS", "se"), seen)
        self.assertEqual(result.status, TaskStatus.RUNNING)


class D2RunnerFlagTests(unittest.TestCase):
    def test_argparse_has_stop_after_d2_clear(self) -> None:
        from harvest.scripts.run_to_day2 import _parse_args

        args = _parse_args(["--stop-after-d2-clear", "--power-on"])
        self.assertTrue(args.stop_after_d2_clear)
        self.assertFalse(args.stop_after_d2_shipping)
        shipping = _parse_args(["--stop-after-d2-shipping"])
        self.assertTrue(shipping.stop_after_d2_shipping)
        self.assertFalse(getattr(shipping, "stop_after_d2_clear", False))


if __name__ == "__main__":
    unittest.main()
