"""D2 leftover smash chunks: four farm quadrants plus full-chain empty."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_STAMINA,
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    MAP_WIDTH,
    STONE,
    TILE_SIZE,
    DebrisType,
    Tool,
)
from harvest.planner.d2_farm_chunks import (
    CHUNK_PIN_TILES,
    EXHAUSTIVE,
    FARM_CHUNK_BOUNDS,
    FARM_CHUNK_ORDER,
    chunk_of_tile,
    chunks_cover_farm,
    resolve_chunks,
    section_complete,
    smash_is_clear,
    wanted_quota,
)
from harvest.planner.d2_work import (
    d2_leftover_phases,
    leftover_section_phases,
    rock_clear_phase,
    stump_clear_phase,
)
from harvest.scripts.leftover_exec import leftover_chain_decision, phase_already_clear
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.tasks.farm_clear_quota import DebrisCounts, count_debris
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.farm_ops import scan_typed_targets
from retro_harness import TaskStatus, WorldState


def _set_player(ram: np.ndarray, tile: tuple[int, int]) -> None:
    px = tile[0] * TILE_SIZE + 8
    py = tile[1] * TILE_SIZE + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_stump(ram: np.ndarray, tx: int, ty: int) -> None:
    _set_tile(ram, tx, ty, 0x09)
    _set_tile(ram, tx + 1, ty, 0x0A)
    _set_tile(ram, tx, ty + 1, 0x0B)
    _set_tile(ram, tx + 1, ty + 1, 0x0C)


def _place_large_rock(ram: np.ndarray, tx: int, ty: int) -> None:
    _set_tile(ram, tx, ty, 0x0D)
    _set_tile(ram, tx + 1, ty, 0x0E)
    _set_tile(ram, tx, ty + 1, 0x0F)
    _set_tile(ram, tx + 1, ty + 1, 0x10)


def _make_farm_ram(*, player_tile=(10, 10), stamina=100, tool=int(Tool.HAMMER)):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = 1
    ram[ADDR_STAMINA] = stamina
    ram[ADDR_TOOL] = tool
    for i in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + i] = 0xA1
    _set_player(ram, player_tile)
    return ram


def _world(ram) -> WorldState:
    return WorldState(frame=0, ram=ram, info={}, obs=None)


class FarmChunkGeometryTests(unittest.TestCase):
    def test_four_chunks_partition_the_64_farm(self) -> None:
        self.assertTrue(chunks_cover_farm())
        self.assertEqual(resolve_chunks("all"), FARM_CHUNK_ORDER)
        self.assertEqual(resolve_chunks("sw"), ("sw",))

    def test_live_stall_tiles_land_in_the_named_chunk(self) -> None:
        self.assertEqual(chunk_of_tile(11, 29), "nw")
        self.assertEqual(chunk_of_tile(48, 13), "ne")
        self.assertEqual(chunk_of_tile(12, 55), "sw")
        self.assertEqual(chunk_of_tile(60, 51), "se")
        for name, tile in CHUNK_PIN_TILES.items():
            self.assertEqual(chunk_of_tile(*tile), name)
            x0, y0, x1, y1 = FARM_CHUNK_BOUNDS[name]
            self.assertTrue(x0 <= tile[0] <= x1 and y0 <= tile[1] <= y1)

    def test_split_line_goes_east_and_south(self) -> None:
        self.assertEqual(chunk_of_tile(31, 31), "nw")
        self.assertEqual(chunk_of_tile(32, 31), "ne")
        self.assertEqual(chunk_of_tile(31, 32), "sw")
        self.assertEqual(chunk_of_tile(32, 32), "se")


class ChunkedCountIsolationTests(unittest.TestCase):
    def test_one_smash_object_per_chunk_is_invisible_to_the_others(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 11, 29, STONE)
        _place_large_rock(ram, 48, 12)
        _place_stump(ram, 12, 54)
        _set_tile(ram, 60, 51, STONE)

        nw = count_debris(ram, FARM_CHUNK_BOUNDS["nw"])
        ne = count_debris(ram, FARM_CHUNK_BOUNDS["ne"])
        sw = count_debris(ram, FARM_CHUNK_BOUNDS["sw"])
        se = count_debris(ram, FARM_CHUNK_BOUNDS["se"])
        whole = count_debris(ram)

        self.assertEqual(nw.stones, 1)
        self.assertEqual(nw.large_rocks, 0)
        self.assertEqual(ne.large_rocks, 1)
        self.assertEqual(ne.stones, 0)
        self.assertEqual(sw.stumps, 1)
        self.assertEqual(sw.stones, 0)
        self.assertEqual(se.stones, 1)
        self.assertEqual(se.stumps, 0)
        self.assertEqual(whole.stones, 2)
        self.assertEqual(whole.large_rocks, 1)
        self.assertEqual(whole.stumps, 1)
        self.assertFalse(smash_is_clear(whole))
        self.assertTrue(smash_is_clear(DebrisCounts()))

    def test_scan_typed_targets_clips_to_chunk(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 11, 29, STONE)
        _set_tile(ram, 12, 55, STONE)
        nw = scan_typed_targets(ram, (DebrisType.STONE,), FARM_CHUNK_BOUNDS["nw"])
        sw = scan_typed_targets(ram, (DebrisType.STONE,), FARM_CHUNK_BOUNDS["sw"])
        self.assertEqual([t.tile for t in nw], [(11, 29)])
        self.assertEqual([t.tile for t in sw], [(12, 55)])


class ChunkedPhaseChainTests(unittest.TestCase):
    def test_section_stones_is_four_bounded_phases(self) -> None:
        phases = leftover_section_phases(
            "stones", stamina=Stamina(current=100, maximum=100)
        )
        self.assertEqual([p.phase for p in phases], ["CLEAR_STONES"] * 4)
        self.assertEqual([p.params["chunk"] for p in phases], list(FARM_CHUNK_ORDER))
        for spec, name in zip(phases, FARM_CHUNK_ORDER):
            self.assertEqual(spec.params["farm_bounds"], FARM_CHUNK_BOUNDS[name])

    def test_one_chunk_section_is_a_single_bounded_phase(self) -> None:
        phases = leftover_section_phases("rocks", chunk="se")
        self.assertEqual(phases[0].phase, "ENSURE_HAMMER")
        rocks = [p for p in phases if p.phase == "CLEAR_ROCKS"]
        self.assertEqual(len(rocks), 1)
        self.assertEqual(rocks[0].params["chunk"], "se")
        self.assertEqual(rocks[0].params["farm_bounds"], FARM_CHUNK_BOUNDS["se"])
        self.assertEqual(rocks[0].params["quota"], {"large_rocks": EXHAUSTIVE})

    def test_full_leftover_chains_four_smash_chunks_without_getting_stuck(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=100, maximum=100))
        names = [p.phase for p in phases]
        self.assertEqual(names.count("CLEAR_STONES"), 4)
        self.assertEqual(names.count("CLEAR_ROCKS"), 4)
        self.assertEqual(names.count("CLEAR_STUMPS"), 4)
        last_stone = max(i for i, n in enumerate(names) if n == "CLEAR_STONES")
        first_hammer = names.index("ENSURE_HAMMER")
        last_rock = max(i for i, n in enumerate(names) if n == "CLEAR_ROCKS")
        first_axe = names.index("ENSURE_AXE")
        first_stump = names.index("CLEAR_STUMPS")
        self.assertLess(last_stone, first_hammer)
        self.assertLess(first_hammer, names.index("CLEAR_ROCKS"))
        self.assertLess(last_rock, first_axe)
        self.assertLess(first_axe, first_stump)
        for spec in phases:
            if spec.phase in {"CLEAR_STONES", "CLEAR_ROCKS", "CLEAR_STUMPS"}:
                self.assertIn(spec.params["chunk"], FARM_CHUNK_ORDER)
                self.assertEqual(
                    spec.params["farm_bounds"],
                    FARM_CHUNK_BOUNDS[spec.params["chunk"]],
                )


class FullChainEmptyTests(unittest.TestCase):
    def test_clearing_each_chunk_empties_the_farm(self) -> None:
        ram = _make_farm_ram()
        stones = {"nw": (11, 29), "ne": (40, 16), "sw": (12, 55), "se": (60, 51)}
        rocks = {"nw": (8, 18), "ne": (36, 10), "sw": (6, 40), "se": (50, 50)}
        stumps = {"nw": (4, 20), "ne": (44, 8), "sw": (8, 48), "se": (52, 44)}
        for tile in stones.values():
            _set_tile(ram, *tile, STONE)
        for tile in rocks.values():
            _place_large_rock(ram, *tile)
        for tile in stumps.values():
            _place_stump(ram, *tile)

        start = count_debris(ram)
        self.assertEqual(start.stones, 4)
        self.assertEqual(start.large_rocks, 4)
        self.assertEqual(start.stumps, 4)
        self.assertFalse(smash_is_clear(start))
        self.assertFalse(section_complete("all", start, start))

        for name, bounds in FARM_CHUNK_BOUNDS.items():
            chunk_start = count_debris(ram, bounds)
            self.assertFalse(smash_is_clear(chunk_start))
            _set_tile(ram, *stones[name], 0xA1)
            rx, ry = rocks[name]
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                _set_tile(ram, rx + dx, ry + dy, 0xA1)
            sx, sy = stumps[name]
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                _set_tile(ram, sx + dx, sy + dy, 0xA1)
            chunk_end = count_debris(ram, bounds)
            self.assertTrue(smash_is_clear(chunk_end))
            self.assertTrue(section_complete("stones", chunk_start, chunk_end))
            self.assertTrue(section_complete("rocks", chunk_start, chunk_end))
            self.assertTrue(section_complete("stumps", chunk_start, chunk_end))
            if name != "se":
                self.assertFalse(smash_is_clear(count_debris(ram)))

        end = count_debris(ram)
        self.assertTrue(smash_is_clear(end))
        self.assertTrue(section_complete("all", start, end))
        self.assertEqual(wanted_quota("stumps").stumps, EXHAUSTIVE)

    def test_skipping_one_chunk_keeps_the_full_chain_red(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 12, 55, STONE)
        _place_large_rock(ram, 50, 50)
        _place_stump(ram, 52, 44)
        start = count_debris(ram)
        _set_tile(ram, 12, 55, 0xA1)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _set_tile(ram, 50 + dx, 50 + dy, 0xA1)
        end = count_debris(ram)
        self.assertEqual(end.stumps, 1)
        self.assertFalse(smash_is_clear(end))
        self.assertFalse(section_complete("all", start, end))
        self.assertTrue(section_complete("stones", start, end))
        self.assertTrue(section_complete("rocks", start, end))
        self.assertFalse(section_complete("stumps", start, end))

    def test_last_five_stumps_skip_empty_chunks(self) -> None:
        ram = _make_farm_ram()
        last = ((4, 20), (12, 8), (20, 24), (8, 48), (52, 44))
        for tile in last:
            _place_stump(ram, *tile)
        start = count_debris(ram)
        self.assertEqual(start.stumps, 5)
        self.assertEqual(count_debris(ram, FARM_CHUNK_BOUNDS["ne"]).stumps, 0)
        self.assertFalse(section_complete("stumps", start, start))

        phases = leftover_section_phases("stumps")
        run = []
        skipped = []
        for spec in phases:
            counts = count_debris(ram, (spec.params or {}).get("farm_bounds"))
            row = (spec.phase, (spec.params or {}).get("chunk"))
            if phase_already_clear(spec.phase, counts):
                skipped.append(row)
            else:
                run.append(row)
        self.assertEqual(skipped, [("CLEAR_STUMPS", "ne")])
        self.assertEqual(
            run,
            [
                ("ENSURE_AXE", None),
                ("CLEAR_STUMPS", "nw"),
                ("CLEAR_STUMPS", "sw"),
                ("CLEAR_STUMPS", "se"),
            ],
        )
        se_bounds = FARM_CHUNK_BOUNDS["se"]
        se_start = count_debris(ram, se_bounds)
        sx, sy = 52, 44
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _set_tile(ram, sx + dx, sy + dy, 0xA1)
        se_end = count_debris(ram, se_bounds)
        self.assertTrue(section_complete("stumps", se_start, se_end))
        self.assertFalse(section_complete("stumps", start, count_debris(ram)))
        self.assertFalse(smash_is_clear(count_debris(ram)))


class QuotaChunkDoesNotPocketApproachTests(unittest.TestCase):
    def test_quota_farm_bounds_do_not_walk_to_the_plant_notch(self) -> None:
        ram = _make_farm_ram(player_tile=(54, 42), tool=int(Tool.HAMMER))
        _place_large_rock(ram, 50, 50)
        world = _world(ram)
        task = FarmClearTask(
            fetch_tools=False,
            handoff="quota",
            quota={"large_rocks": EXHAUSTIVE},
            farm_bounds=FARM_CHUNK_BOUNDS["se"],
            priority=[DebrisType.ROCK],
            timeout=200,
        )
        task.reset(world)
        self.assertFalse(task._uses_pocket_approach())
        self.assertIsNone(task._step_pocket_approach(world))
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("plant pocket", result.reason or "")

    def test_chunked_rock_builder_keeps_quota_handoff(self) -> None:
        spec = rock_clear_phase(farm_bounds=FARM_CHUNK_BOUNDS["se"], chunk="se")
        ram = _make_farm_ram()
        task = build_phase_task(TaskBuildContext(), spec, _world(ram))
        self.assertIsInstance(task, FarmClearTask)
        self.assertEqual(task.farm_bounds, FARM_CHUNK_BOUNDS["se"])
        self.assertEqual(task.handoff, "quota")
        self.assertFalse(task._uses_pocket_approach())
        stumps = stump_clear_phase(farm_bounds=FARM_CHUNK_BOUNDS["sw"], chunk="sw")
        self.assertEqual(stumps.params["quota"], {"stumps": EXHAUSTIVE})
        self.assertEqual(stumps.params["timeout"], 0)


class LeftoverChainReadinessTests(unittest.TestCase):
    """Edges that would cut a full D2 leftover movie before the farm is empty."""

    def test_stall_on_se_rock_aborts_and_never_reaches_stumps(self) -> None:
        remaining = ("CLEAR_ROCKS", "ENSURE_AXE", "CLEAR_STUMPS")
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.FAILURE,
                "no debris progress 24000f (last_progress=1000)",
                Stamina(current=40, maximum=100),
                remaining,
            ),
            "abort",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.SUCCESS,
                "quota met",
                Stamina(current=10, maximum=100),
                remaining,
            ),
            "insert_spa",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.FAILURE,
                "stamina_low cleared=2",
                Stamina(current=8, maximum=100),
                remaining,
            ),
            "spa_retry",
        )
        self.assertEqual(
            leftover_chain_decision(
                "HOT_SPRING_STAMINA",
                TaskStatus.RUNNING,
                None,
                Stamina(current=4, maximum=100),
                ("ENSURE_HAMMER", "CLEAR_ROCKS"),
            ),
            "abort",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_STUMPS",
                TaskStatus.SUCCESS,
                "quota met",
                Stamina(current=4, maximum=100),
                ("CLEAR_STUMPS",),
            ),
            "insert_spa",
        )

    def test_partial_pin_skips_empty_chunks_and_keeps_se_boulder(self) -> None:
        ram = _make_farm_ram()
        _place_large_rock(ram, 60, 51)
        phases = leftover_section_phases(
            "all", stamina=Stamina(current=100, maximum=100)
        )
        run = []
        skipped = []
        for spec in phases:
            counts = count_debris(ram, (spec.params or {}).get("farm_bounds"))
            row = (spec.phase, (spec.params or {}).get("chunk"))
            if phase_already_clear(spec.phase, counts):
                skipped.append(row)
            else:
                run.append(row)
        self.assertEqual(
            skipped,
            [
                ("CLEAR_BUSHES", None),
                ("CLEAR_FENCES", None),
                ("CLEAR_STONES", "nw"),
                ("CLEAR_STONES", "ne"),
                ("CLEAR_STONES", "sw"),
                ("CLEAR_STONES", "se"),
                ("CLEAR_ROCKS", "nw"),
                ("CLEAR_ROCKS", "ne"),
                ("CLEAR_ROCKS", "sw"),
                ("CLEAR_STUMPS", "nw"),
                ("CLEAR_STUMPS", "ne"),
                ("CLEAR_STUMPS", "sw"),
                ("CLEAR_STUMPS", "se"),
            ],
        )
        self.assertEqual(
            run,
            [
                ("ENSURE_HAMMER", None),
                ("CLEAR_ROCKS", "se"),
                ("ENSURE_AXE", None),
            ],
        )

    def test_section_all_green_requires_empty_weeds(self) -> None:
        from harvest.planner.d2_farm_chunks import smash_done_empty, wanted_quota

        self.assertIn("weeds", smash_done_empty("all"))
        self.assertEqual(smash_done_empty("all"), ("weeds", "fences", "stones", "large_rocks", "stumps"))
        self.assertEqual(smash_done_empty("bushes"), ("weeds",))
        self.assertEqual(wanted_quota("all").weeds, EXHAUSTIVE)
        self.assertEqual(wanted_quota("bushes").weeds, EXHAUSTIVE)

    def test_leftover_smash_is_required_so_a_day_plan_cannot_skip_a_stall(self) -> None:
        phases = d2_leftover_phases(stamina=Stamina(current=100, maximum=100))
        smash = [
            p
            for p in phases
            if p.phase
            in {"CLEAR_BUSHES", "CLEAR_FENCES", "CLEAR_STONES", "CLEAR_ROCKS", "CLEAR_STUMPS"}
        ]
        self.assertTrue(smash)
        for spec in smash:
            self.assertEqual(spec.failure_policy, "required")

    def test_default_day_budgets_cover_multi_section_leftover(self) -> None:
        from pathlib import Path

        harvest_dir = Path(__file__).resolve().parents[1] / "harvest"
        leftover_src = (harvest_dir / "scripts" / "d2_leftover_probe.py").read_text(
            encoding="utf-8"
        )
        run_src = (harvest_dir / "scripts" / "run_to_day2.py").read_text(encoding="utf-8")
        self.assertIn("default=2_000_000", leftover_src)
        self.assertNotIn("default=400_000", leftover_src)
        self.assertIn("2_000_000 * max(1, overnights_budget)", run_src)
        self.assertNotIn("200_000 * max(1, overnights_budget)", run_src)


if __name__ == "__main__":
    unittest.main()
