"""D2 field-work composition — bounded quotas and exhaustive fences."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from retro_harness import WorldState

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import Tool
from harvest.planner.d2_farm_chunks import EXHAUSTIVE, FARM_CHUNK_BOUNDS
from harvest.planner.d2_work import (
    D2_TARGETS,
    bush_clear_phase,
    d2_post_shop_work_phases,
    ensure_axe_phase,
    ensure_hammer_phase,
    fence_dump_phase,
    leftover_already_queued,
    needs_spa_before_next_smash,
    next_d2_spec,
    observe_d2_farm,
    pocket_water_phase,
    rock_clear_phase,
    should_spa_retry,
    stone_pond_phase,
    stump_clear_phase,
)
from harvest.planner.day_phase_types import PhaseKind
from harvest.planner.day_plan_phases import pocket_plant_phases


class D2WholeFarmContractTests(unittest.TestCase):
    def test_crop_targets_are_not_debris_quotas(self) -> None:
        self.assertEqual(D2_TARGETS, {"plant": 8, "water": 8})

    def test_smash_builders_are_exhaustive_required_quota(self) -> None:
        bushes = bush_clear_phase()
        fences = fence_dump_phase()
        stones = stone_pond_phase()
        rocks = rock_clear_phase()
        stumps = stump_clear_phase()
        self.assertEqual(bushes.params["quota"], {"weeds": EXHAUSTIVE})
        self.assertEqual(bushes.params["handoff"], "quota")
        self.assertTrue(fences.params["pond_dump"])
        self.assertIsNone(fences.params["max_fences"])
        self.assertEqual(fences.params["debris_types"], ["fence"])
        self.assertEqual(stones.params["debris_types"], ["stone"])
        self.assertTrue(stones.params["pond_dump"])
        self.assertEqual(rocks.params["quota"], {"large_rocks": EXHAUSTIVE})
        self.assertEqual(rocks.contract.required_tools, ("hammer",))
        self.assertEqual(stumps.params["quota"], {"stumps": EXHAUSTIVE})
        self.assertEqual(stumps.contract.required_tools, ("axe",))
        for spec in (bushes, fences, stones, rocks, stumps):
            self.assertEqual(spec.failure_policy, "required")
            self.assertEqual(spec.params["timeout"], 0)

    def test_ensure_hammer_and_axe_are_ram_shelf_not_recorded(self) -> None:
        hammer = ensure_hammer_phase()
        axe = ensure_axe_phase()
        self.assertEqual(hammer.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(axe.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(hammer.params["tool_id"], int(Tool.HAMMER))
        self.assertEqual(axe.params["tool_id"], int(Tool.AXE))
        self.assertNotEqual(hammer.kind, PhaseKind.RECORDED)


class D2LeftoverOrderTests(unittest.TestCase):
    def test_low_stam_spas_before_rocks_not_before_bushes(self) -> None:
        ram = _farm_ram(stamina=8)
        _set_tile(ram, 40, 40, 0x03)
        _place_large_rock(ram, 50, 50)
        status = observe_d2_farm(ram)
        self.assertEqual(next_d2_spec(status).phase, "CLEAR_BUSHES")
        self.assertEqual(next_d2_spec(status, section="rocks").phase, "HOT_SPRING_STAMINA")

    def test_full_stam_skips_spa_but_keeps_smash_order(self) -> None:
        ram = _farm_ram(stamina=100)
        _plant_eight_wet(ram)
        _set_tile(ram, 40, 40, 0x03)
        _set_tile(ram, 42, 42, 0x05)
        _set_tile(ram, 40, 36, 0x04)
        _place_large_rock(ram, 50, 50)
        _place_stump(ram, 52, 44)
        status = observe_d2_farm(ram, _SHIP_OK)
        bushes = next_d2_spec(status)
        self.assertEqual(bushes.phase, "CLEAR_BUSHES")
        self.assertEqual(bushes.params["chunk"], "se")
        self.assertEqual(next_d2_spec(status, section="rocks").phase, "ENSURE_HAMMER")

    def test_hammer_and_axe_are_sequential_not_same_carry(self) -> None:
        ram = _farm_ram(stamina=100)
        _place_large_rock(ram, 50, 50)
        _place_stump(ram, 52, 44)
        status = observe_d2_farm(ram)
        self.assertEqual(next_d2_spec(status, section="rocks").phase, "ENSURE_HAMMER")
        self.assertEqual(
            next_d2_spec(status, section="rocks", last_phase="ENSURE_HAMMER").phase,
            "CLEAR_ROCKS",
        )
        self.assertEqual(next_d2_spec(status, section="stumps").phase, "ENSURE_AXE")

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
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_BUSHES",
                TaskStatus.FAILURE,
                "partial_clear cleared=503 remaining=1 lift_only",
                Stamina(current=100, maximum=100),
                ("CLEAR_FENCES",),
            ),
            "continue",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.FAILURE,
                "field_clear cleared=0 lift_only",
                Stamina(current=100, maximum=100),
                ("CLEAR_STUMPS",),
            ),
            "continue",
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
        from retro_harness import TaskStatus, WorldState

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

        sw = stone_pond_phase(farm_bounds=FARM_CHUNK_BOUNDS["sw"], chunk="sw")
        sw_task = build_phase_task(TaskBuildContext(), sw, world)
        self.assertEqual(sw.params["chunk"], "sw")
        self.assertEqual(sw_task.farm_bounds, (0, 32, 31, 63))


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


_SHIP_OK = [{
    "phase": "MOUNTAIN_BERRY",
    "status": "success",
    "shipping_deposit": {
        "season": 0,
        "day": 2,
        "hour": 12,
        "minute": 34,
        "shipping_money_before": 0,
        "shipping_money_after": 30,
    },
}]


def _farm_ram(*, stamina=100, lock=1, season=0, day=2, hour=12, player=(10, 10)):
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
    ram[field_spec("season").address] = season
    ram[field_spec("day").address] = day
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
        self.assertTrue(status.input_stable)
        self.assertTrue(status.settled)
        self.assertTrue(status.shipped_before_17)
        self.assertEqual(status.trees_or_stumps, 0)
        self.assertIn("trees_or_stumps", status.to_record())

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

        small = _farm_ram()
        _plant_eight_wet(small)
        _set_tile(small, 30, 30, 0x06)
        small_status = observe_d2_farm(small, _SHIP_OK)
        self.assertEqual(small_status.small_rocks, 1)
        self.assertFalse(small_status.is_complete)

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
        self.assertFalse(observe_d2_farm(late).shipped_before_17)
        self.assertTrue(observe_d2_farm(late, _SHIP_OK).shipped_before_17)
        self.assertTrue(observe_d2_farm(late, _SHIP_OK).is_complete)

        wrong_date = _farm_ram(day=3)
        _plant_eight_wet(wrong_date)
        self.assertFalse(observe_d2_farm(wrong_date, _SHIP_OK).is_complete)

        done = observe_d2_farm(ram, _SHIP_OK)
        self.assertTrue(confirm_d2_complete(done, done))
        self.assertFalse(confirm_d2_complete(None, done))
        self.assertFalse(confirm_d2_complete(observe_d2_farm(weed, _SHIP_OK), done))

        last = _farm_ram(hour=18)
        _plant_eight_wet(last)
        last[field_spec("shipping_money_raw").address] = 1
        _place_stump(last, 52, 44)
        leftover = observe_d2_farm(last, _SHIP_OK)
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

    def test_deposit_requires_timestamped_d2_event_before_five(self) -> None:
        from harvest.planner.d2_work import observe_d2_farm

        ram = _farm_ram()
        _plant_eight_wet(ram)
        late_deposit = [{
            "shipping_deposit": {
                "season": 0,
                "day": 2,
                "hour": 17,
                "minute": 0,
                "shipping_money_before": 0,
                "shipping_money_after": 30,
            }
        }]
        wrong_day = [{
            "shipping_deposit": {
                **_SHIP_OK[0]["shipping_deposit"],
                "day": 3,
            }
        }]
        self.assertFalse(observe_d2_farm(ram, late_deposit).shipped_before_17)
        self.assertFalse(observe_d2_farm(ram, wrong_day).shipped_before_17)


class D2ShippingJournalTests(unittest.TestCase):
    def test_mountain_berry_phase_records_timestamped_bin_deposit(self) -> None:
        from harvest.planner.day_phase_types import PhaseSpec
        from harvest.planner.day_plan_orchestrator import DayPlanTask

        class ShippedGrape:
            shipped_count = 1
            harvested_count = 0
            _shipping_before = 0
            _shipping_after = 30

        ram = _farm_ram(hour=12)
        plan = DayPlanTask(phase_sequence=[])
        plan._current_task = ShippedGrape()
        plan._record_phase_result(
            PhaseSpec("MOUNTAIN_BERRY", "mountain_berry"),
            "success",
            world=WorldState(frame=0, ram=ram, info={}, obs=None),
        )
        deposit = plan.phase_results[0]["shipping_deposit"]
        self.assertEqual(deposit["day"], 2)
        self.assertEqual(deposit["hour"], 12)
        self.assertEqual(deposit["shipping_money_after"], 30)

    def test_d2_tactic_receives_prior_deposit_evidence_from_day_plan(self) -> None:
        from harvest.planner.d2_work import D2FarmClearTactic, d2_farm_clear_phase
        from harvest.planner.day_plan_orchestrator import DayPlanTask

        ram = _farm_ram()
        _plant_eight_wet(ram)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        plan = DayPlanTask(phase_sequence=[d2_farm_clear_phase()])
        plan.reset(world)
        plan._phase_results.append(dict(_SHIP_OK[0]))

        plan.step(world)

        self.assertIsInstance(plan._current_task, D2FarmClearTactic)
        self.assertEqual(plan._current_task.journal, _SHIP_OK)
        self.assertTrue(plan._current_task.farm_status.is_complete)


class D2NextSpecTests(unittest.TestCase):
    def test_shop_miss_does_not_loop_pocket_or_plant(self) -> None:
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        ram = _farm_ram()
        _set_tile(ram, 13, 28, 0x03)
        _set_tile(ram, 40, 40, 0x03)
        status = observe_d2_farm(ram)
        self.assertTrue(status.pocket_needs_clear)
        self.assertEqual(status.potato_seeds, 0)
        self.assertEqual(next_d2_spec(status).phase, "CLEAR_PLOT")
        self.assertEqual(
            next_d2_spec(status, last_phase="CLEAR_PLOT").phase,
            "CLEAR_BUSHES",
        )

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
        for tx, ty in ((4, 20), (12, 8), (20, 24)):
            _clear_2x2(ram, tx, ty)
        after_nw = observe_d2_farm(ram, _SHIP_OK)
        self.assertEqual(after_nw.stumps_by_chunk, (0, 0, 1, 1))
        skipped_ne = next_d2_spec(after_nw, last_phase="CLEAR_STUMPS")
        self.assertEqual(skipped_ne.phase, "CLEAR_STUMPS")
        self.assertEqual(skipped_ne.params["chunk"], "sw")
        _clear_2x2(ram, 8, 48)
        after_sw = observe_d2_farm(ram, _SHIP_OK)
        last = next_d2_spec(after_sw, last_phase="CLEAR_STUMPS")
        self.assertEqual(last.phase, "CLEAR_STUMPS")
        self.assertEqual(last.params["chunk"], "se")
        _clear_2x2(ram, 52, 44)
        none = next_d2_spec(observe_d2_farm(ram, _SHIP_OK), last_phase="CLEAR_STUMPS")
        self.assertIsNone(none)
        se_only = next_d2_spec(
            status, section="stumps", chunk="se", last_phase="ENSURE_AXE"
        )
        self.assertEqual(se_only.params["chunk"], "se")
        ne_only = next_d2_spec(
            status, section="stumps", chunk="ne", last_phase="ENSURE_AXE"
        )
        self.assertIsNone(ne_only)

    def test_next_spec_walks_plot_then_crops_then_leftover_order(self) -> None:
        from harvest.core.ram_catalog import field_spec
        from harvest.planner.d2_work import next_d2_spec, observe_d2_farm

        ram = _farm_ram(stamina=100)
        ram[field_spec("potato_seeds").address] = 1
        _set_tile(ram, 13, 28, 0x03)
        _set_tile(ram, 40, 40, 0x03)
        _set_tile(ram, 42, 42, 0x05)
        _set_tile(ram, 8, 40, 0x04)
        _place_large_rock(ram, 50, 50)
        _place_stump(ram, 52, 44)
        status = observe_d2_farm(ram)
        self.assertEqual(next_d2_spec(status).phase, "CLEAR_PLOT")
        self.assertEqual(
            next_d2_spec(status, last_phase="CLEAR_PLOT").phase,
            "ENSURE_CROP_SEEDS",
        )

        _set_tile(ram, 13, 28, 0xA1)
        ram[field_spec("potato_seeds").address] = 0
        _plant_eight_wet(ram)
        leftover = observe_d2_farm(ram, _SHIP_OK)
        stones = next_d2_spec(leftover)
        self.assertEqual(stones.phase, "CLEAR_STONES")
        self.assertEqual(stones.params["chunk"], "sw")
        _set_tile(ram, 8, 40, 0xA1)
        leftover = observe_d2_farm(ram, _SHIP_OK)
        bushes = next_d2_spec(leftover)
        self.assertEqual(bushes.phase, "CLEAR_BUSHES")
        self.assertEqual(bushes.params["chunk"], "se")
        _set_tile(ram, 40, 40, 0xA1)
        leftover = observe_d2_farm(ram, _SHIP_OK)
        fences = next_d2_spec(leftover)
        self.assertEqual(fences.phase, "CLEAR_FENCES")
        self.assertEqual(fences.params["chunk"], "se")
        _set_tile(ram, 42, 42, 0xA1)
        leftover = observe_d2_farm(ram, _SHIP_OK)
        self.assertEqual(next_d2_spec(leftover).phase, "ENSURE_HAMMER")
        rocks = next_d2_spec(leftover, last_phase="ENSURE_HAMMER")
        self.assertEqual(rocks.phase, "CLEAR_ROCKS")
        self.assertEqual(rocks.params["chunk"], "se")
        _clear_2x2(ram, 50, 50)
        leftover = observe_d2_farm(ram, _SHIP_OK)
        self.assertEqual(next_d2_spec(leftover).phase, "ENSURE_AXE")
        stumps = next_d2_spec(leftover, last_phase="ENSURE_AXE")
        self.assertEqual(stumps.phase, "CLEAR_STUMPS")
        self.assertEqual(stumps.params["chunk"], "se")
        _clear_2x2(ram, 52, 44)
        done = observe_d2_farm(ram, _SHIP_OK)
        self.assertTrue(done.is_complete)
        self.assertIsNone(next_d2_spec(done))


class D2FarmClearTacticTests(unittest.TestCase):
    def test_post_shop_plot_clear_advances_to_crop_while_pocket_stays_dirty(self) -> None:
        """Broad-pocket weeds must not restart CLEAR_PLOT after seed/tool setup."""
        from unittest.mock import patch

        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.core.ram_catalog import field_spec
        from harvest.planner.d2_work import D2FarmClearTactic

        ram = _farm_ram(hour=18)
        ram[field_spec("potato_seeds").address] = 1
        for tile in ((12, 27), (13, 27), (14, 27)):
            _set_tile(ram, *tile, 0x03)
        _set_tile(ram, 20, 20, 0x03)
        seen = []

        class Instant:
            def __init__(self, spec) -> None:
                self.spec = spec

            def reset(self, _world) -> None:
                return None

            def step(self, world):
                if self.spec.phase == "CLEAR_PLOT":
                    for tile in ((12, 27), (13, 27), (14, 27)):
                        _set_tile(world.ram, *tile, 0xA1)
                return TaskResult(status=TaskStatus.SUCCESS, reason="ok")

        def fake_build(_ctx, spec, _world):
            seen.append(spec.phase)
            return Instant(spec)

        tactic = D2FarmClearTactic()
        tactic.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            for frame in range(3):
                tactic.step(WorldState(frame=frame, ram=ram, info={}, obs=None))

        self.assertEqual(seen, ["CLEAR_PLOT", "ENSURE_CROP_SEEDS", "NAV_CROP"])

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

    def test_unobservable_map_still_steps_an_active_child(self) -> None:
        """ENSURE_HAMMER in the shed must run; yard idle froze the fetch."""
        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.core.tile_catalog import ADDR_TILEMAP
        from harvest.planner.d2_work import D2FarmClearTactic, D2FarmOutcome

        ram = _farm_ram(player=(9, 12))
        ram[ADDR_TILEMAP] = 0x26
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        tactic = D2FarmClearTactic(section="rocks", chunk="nw", include_spa=False)
        tactic.reset(world)

        class Fetch:
            name = "ensure_tool"
            calls = 0

            def step(self, _world):
                self.calls += 1
                return TaskResult(status=TaskStatus.RUNNING, reason="picking hammer")

        child = Fetch()
        tactic._child = child
        result = tactic.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(child.calls, 1)
        self.assertEqual(tactic.farm_status.outcome, D2FarmOutcome.TEMPORARILY_UNOBSERVABLE)

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

    def test_stationary_six_hit_stump_clear_outlives_motion_window(self) -> None:
        """Planted axe work is progress, not a 360-frame navigation stall."""
        from unittest.mock import patch

        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.core.tile_catalog import DebrisType
        from harvest.planner.d2_work import D2FarmClearTactic
        from harvest.tasks.farm_clear_task import FarmClearTask
        from harvest.tasks.farm_clearer import Target
        from harvest.tasks.nav import Point

        ram = _farm_ram(player=(51, 44))
        _place_stump(ram, 52, 44)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        child = FarmClearTask(fetch_tools=False, handoff="quota", quota={"stumps": EXHAUSTIVE})
        reset_child = child.reset

        def reset_at_stump(active_world):
            reset_child(active_world)
            child.clearer.state = "clearing"
            child.clearer.current_phase = DebrisType.STUMP
            child.clearer.current_target = Target(
                (52, 44), Point(52 * 16 + 8, 44 * 16 + 8), DebrisType.STUMP, 0x09
            )
            child.clearer.approach_tile = (51, 44)

        def stationary_six_hit_step(active_world):
            child._step_count += 1
            child.clearer.target_hits = min(6, child._step_count // 61)
            if child.clearer.target_hits < 6:
                return TaskResult(status=TaskStatus.RUNNING)
            _clear_2x2(active_world.ram, 52, 44)
            return TaskResult(status=TaskStatus.SUCCESS, reason="sixth axe hit destroyed stump")

        child.reset = reset_at_stump
        child.step = stationary_six_hit_step

        class Instant:
            def reset(self, _world) -> None:
                return None

            def step(self, _world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="ready")

        def fake_build(_ctx, spec, _world):
            return child if spec.phase == "CLEAR_STUMPS" else Instant()

        tactic = D2FarmClearTactic(section="stumps", chunk="se", include_spa=False)
        tactic.reset(world)
        with patch("harvest.planner.day_phase_registry.build_phase_task", fake_build):
            result = None
            for frame in range(400):
                world = WorldState(frame=frame, ram=ram, info={}, obs=None)
                result = tactic.step(world)
                if result.status != TaskStatus.RUNNING:
                    break

        self.assertGreater(child._step_count, 360)
        self.assertEqual(child.clearer.target_hits, 6)
        progress = dict(child.progress_snapshot().details)
        self.assertEqual(progress["clearing_phase"], "STUMP")
        self.assertEqual(progress["target"], (52, 44))
        self.assertEqual(progress["hits"], 6)
        self.assertEqual(progress["approach_position"], (824, 712))
        self.assertEqual(tactic.farm_status.stumps, 0)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertFalse(any(row["status"] == TaskStatus.BLOCKED.value for row in tactic.journal))

    def test_navigation_stall_is_recorded_without_skipping_required_chunk(self) -> None:
        from retro_harness import TaskStatus, WorldState

        from harvest.core.task_progress import ProgressSnapshot
        from harvest.planner.d2_work import D2FarmClearTactic, observe_d2_farm

        ram = _farm_ram(player=(51, 44))
        _place_stump(ram, 52, 44)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)

        class StuckNav:
            name = "farm_clear"

            def progress_snapshot(self):
                return ProgressSnapshot(
                    task_name=self.name,
                    phase_text="navigating",
                    details=(("target", (52, 44)), ("approach", (51, 44))),
                )

        tactic = D2FarmClearTactic(section="stumps", chunk="se", include_spa=False)
        tactic.reset(world)
        tactic._child = StuckNav()
        tactic._spec = stump_clear_phase(
            farm_bounds=FARM_CHUNK_BOUNDS["se"], chunk="se"
        )
        status = observe_d2_farm(ram)
        tactic._step = 0
        self.assertIsNone(tactic._watchdogs(world, status))
        tactic._step = 360
        result = tactic._watchdogs(world, status)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertEqual(tactic.journal[-1]["chunk"], "se")
        self.assertEqual(tactic.journal[-1]["watchdog"], "navigation_motion_stall")

    def test_clock_hour_does_not_reset_goal_stall(self) -> None:
        from harvest.core.ram_catalog import field_spec
        from harvest.core.task_progress import GOAL_STALL_FRAMES
        from harvest.planner.d2_work import D2FarmClearTactic, observe_d2_farm

        ram = _farm_ram(player=(51, 44), hour=18)
        _place_stump(ram, 52, 44)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        tactic = D2FarmClearTactic(section="stumps", chunk="se", include_spa=False)
        tactic.reset(world)
        status = observe_d2_farm(ram)
        tactic._step = 0
        self.assertIsNone(tactic._watchdogs(world, status))
        hour_addr = field_spec("hour").address
        ram[hour_addr] = 19
        from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET

        if hour_addr + LIVE_RAM_WRAM_OFFSET < len(ram):
            ram[hour_addr + LIVE_RAM_WRAM_OFFSET] = 19
        later = observe_d2_farm(ram)
        self.assertEqual(later.hour, 19)
        self.assertEqual(later.stumps, status.stumps)
        tactic._step = GOAL_STALL_FRAMES
        result = tactic._watchdogs(world, later)
        self.assertIsNotNone(result)
        self.assertIn("goal stall", result.reason)

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


class D2DaytimeClearPhaseTests(unittest.TestCase):
    def test_d2_morning_uses_farm_clear_tactic_not_quota_field(self) -> None:
        from harvest.planner.day_plan_phases import _daytime_clear_phase

        d2 = _daytime_clear_phase(0, 2)
        self.assertEqual(d2.phase, "D2_FARM_CLEAR")
        self.assertEqual(d2.failure_policy, "required")
        other = _daytime_clear_phase(0, 3)
        self.assertEqual(other.phase, "CLEAR_FIELD")

    def test_stop_after_d2_clear_does_not_idle_when_debris_remains(self) -> None:
        from retro_harness import TaskResult, TaskStatus, WorldState

        from harvest.planner.d2_work import D2FarmClearTactic
        from harvest.planner.day_phase_types import DayPlannerPolicy
        from harvest.planner.multi_day_planner import MultiDayPlannerTask

        ram = _farm_ram()
        _place_stump(ram, 52, 44)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        planner = MultiDayPlannerTask(policy=DayPlannerPolicy(include_end_day=False))
        planner.reset(world)
        planner._phase = "plan_day"
        planner._current_task = object()
        result = planner._handle_result(
            world, TaskResult(status=TaskStatus.SUCCESS, reason="day plan complete")
        )
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(planner._current_task, D2FarmClearTactic)


class D2RunnerFlagTests(unittest.TestCase):
    def test_argparse_has_stop_after_d2_clear(self) -> None:
        from harvest.scripts.run_to_day2 import _parse_args

        args = _parse_args(["--stop-after-d2-clear", "--power-on"])
        self.assertTrue(args.stop_after_d2_clear)
        self.assertFalse(args.stop_after_d2_shipping)
        shipping = _parse_args(["--stop-after-d2-shipping"])
        self.assertTrue(shipping.stop_after_d2_shipping)
        self.assertFalse(getattr(shipping, "stop_after_d2_clear", False))

    def test_power_on_d2_clear_defaults_a_progress_sidecar(self) -> None:
        from harvest.scripts.run_to_day2 import _checkpoint_dir, _parse_args, _progress_sidecar_path

        args = _parse_args(
            [
                "--power-on",
                "--stop-after-d2-clear",
                "--checkpoint-on-progress",
                "--out",
                "recordings/power_on_d2_farm_clear.json",
            ]
        )
        sidecar = _progress_sidecar_path(args)
        self.assertIsNotNone(sidecar)
        self.assertTrue(str(sidecar).endswith("power_on_d2_farm_clear.progress.json"))
        self.assertTrue(args.checkpoint_on_progress)
        self.assertIsNotNone(_checkpoint_dir(args))
        off = _parse_args(["--power-on", "--progress-sidecar-every", "0"])
        self.assertIsNone(_progress_sidecar_path(off))


_FARM_CLEAR_REPORT = (
    Path(__file__).resolve().parents[1] / "recordings" / "power_on_d2_farm_clear.json"
)


@unittest.skipUnless(_FARM_CLEAR_REPORT.is_file(), "power-on D2 farm-clear recording not on disk")
class PowerOnD2FarmClearReportTests(unittest.TestCase):
    """Lock the closed rr-20w.2.3 evidence. Overwrite with a red run and this fails."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(_FARM_CLEAR_REPORT.read_text(encoding="utf-8"))

    def test_clean_power_on_farm_clear_is_complete(self) -> None:
        report = self.report
        self.assertTrue(report["success"])
        self.assertTrue(report["terminal"])
        self.assertEqual(report["reason"], "d2 farm clear complete")
        self.assertIsNone(report["state"])
        self.assertTrue(report["power_on"]["completed"])
        self.assertEqual(report["frames"], 393223)
        self.assertEqual(report["planner_frames"], 371309)
        self.assertEqual(report["end"]["day"], 2)
        self.assertEqual(report["end"]["hour"], 18)
        self.assertEqual(report["end"]["minute"], 1)
        self.assertEqual(report["end"]["money"], 100)
        self.assertEqual(report["end"]["stamina"], 52)
        farm = report["d2_farm"]
        self.assertEqual(farm["end"]["weeds"], 0)
        self.assertEqual(farm["end"]["fences"], 0)
        self.assertEqual(farm["end"]["stones"], 0)
        self.assertEqual(farm["end"]["large_rocks"], 0)
        self.assertEqual(farm["end"]["stumps"], 0)
        self.assertEqual(farm["end"]["planted"], 8)
        self.assertEqual(farm["end"]["wet"], 8)
        status = farm["final_status"]
        self.assertEqual(status["outcome"], "complete")
        self.assertTrue(status["shipped_before_17"])
        self.assertTrue(status["settled"])
        self.assertTrue(farm["two_consecutive_settled_observations"])
        clean = report["clean_run"]
        self.assertEqual(clean["intervention_class"], "Clean")
        self.assertEqual(clean["ram_writes"], 0)
        self.assertEqual(clean["mid_run_state_loads"], 0)
        self.assertEqual(clean["initial_state_loads"], 0)
        self.assertIsNone(report["video"])


if __name__ == "__main__":
    unittest.main()
