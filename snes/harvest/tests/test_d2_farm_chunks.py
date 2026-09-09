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
    FENCE,
    MAP_WIDTH,
    STONE,
    TILE_SIZE,
    WEED,
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
    bush_clear_phase,
    leftover_chain_decision,
    next_d2_spec,
    observe_d2_farm,
    rock_clear_phase,
    stump_clear_phase,
)
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
    def test_next_spec_walks_live_stone_chunks_in_order(self) -> None:
        ram = _make_farm_ram()
        stones = {"nw": (11, 29), "ne": (40, 16), "sw": (12, 55), "se": (60, 51)}
        for tile in stones.values():
            _set_tile(ram, *tile, STONE)
        seen = []
        for _ in FARM_CHUNK_ORDER:
            spec = next_d2_spec(observe_d2_farm(ram), section="stones")
            self.assertIsNotNone(spec)
            self.assertEqual(spec.phase, "CLEAR_STONES")
            name = spec.params["chunk"]
            seen.append(name)
            self.assertEqual(spec.params["farm_bounds"], FARM_CHUNK_BOUNDS[name])
            _set_tile(ram, *stones[name], 0xA1)
        self.assertEqual(seen, list(FARM_CHUNK_ORDER))
        self.assertIsNone(next_d2_spec(observe_d2_farm(ram), section="stones"))

    def test_one_chunk_section_is_a_single_bounded_phase(self) -> None:
        ram = _make_farm_ram()
        _place_large_rock(ram, 50, 50)
        spec = next_d2_spec(
            observe_d2_farm(ram), section="rocks", chunk="se", last_phase="ENSURE_HAMMER"
        )
        self.assertEqual(spec.phase, "CLEAR_ROCKS")
        self.assertEqual(spec.params["chunk"], "se")
        self.assertEqual(spec.params["farm_bounds"], FARM_CHUNK_BOUNDS["se"])
        self.assertEqual(spec.params["quota"], {"large_rocks": EXHAUSTIVE})
        self.assertIsNone(
            next_d2_spec(
                observe_d2_farm(ram),
                section="rocks",
                chunk="nw",
                last_phase="ENSURE_HAMMER",
            )
        )


def _leftover(ram, **kwargs):
    return next_d2_spec(observe_d2_farm(ram), plot_attempted=True, **kwargs)


class SectionFirstLiftTests(unittest.TestCase):
    def test_local_stones_beat_distant_bushes(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 11, 10, STONE)
        _set_tile(ram, 40, 40, WEED)
        spec = _leftover(ram)
        self.assertEqual(spec.phase, "CLEAR_STONES")
        self.assertEqual(spec.params["chunk"], "nw")
        self.assertEqual(spec.params["farm_bounds"], FARM_CHUNK_BOUNDS["nw"])

    def test_same_chunk_weeds_before_stones_and_fences(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 10, 8, WEED)
        _set_tile(ram, 11, 10, STONE)
        _set_tile(ram, 8, 6, FENCE)
        spec = _leftover(ram)
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.params["chunk"], "nw")
        _set_tile(ram, 10, 8, 0xA1)
        spec = _leftover(ram)
        self.assertEqual(spec.phase, "CLEAR_FENCES")
        self.assertEqual(spec.params["chunk"], "nw")
        _set_tile(ram, 8, 6, 0xA1)
        spec = _leftover(ram)
        self.assertEqual(spec.phase, "CLEAR_STONES")
        self.assertEqual(spec.params["chunk"], "nw")

    def test_bushes_walk_live_chunks_in_order(self) -> None:
        ram = _make_farm_ram()
        bushes = {"nw": (10, 8), "ne": (40, 16), "sw": (12, 40), "se": (40, 40)}
        for tile in bushes.values():
            _set_tile(ram, *tile, WEED)
        seen = []
        for _ in FARM_CHUNK_ORDER:
            spec = _leftover(ram, section="bushes")
            self.assertEqual(spec.phase, "CLEAR_BUSHES")
            name = spec.params["chunk"]
            seen.append(name)
            self.assertEqual(spec.params["farm_bounds"], FARM_CHUNK_BOUNDS[name])
            _set_tile(ram, *bushes[name], 0xA1)
        self.assertEqual(seen, list(FARM_CHUNK_ORDER))
        self.assertIsNone(_leftover(ram, section="bushes"))

    def test_section_bushes_one_chunk_ignores_other_quadrants(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 10, 8, WEED)
        _set_tile(ram, 40, 40, WEED)
        spec = _leftover(ram, section="bushes", chunk="se")
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.params["chunk"], "se")
        self.assertIsNone(_leftover(ram, section="bushes", chunk="ne"))


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

        status = observe_d2_farm(ram)
        self.assertEqual(next_d2_spec(status, section="stumps").phase, "ENSURE_AXE")
        seen = []
        while True:
            spec = next_d2_spec(observe_d2_farm(ram), section="stumps", last_phase="ENSURE_AXE")
            if spec is None:
                break
            self.assertEqual(spec.phase, "CLEAR_STUMPS")
            name = spec.params["chunk"]
            seen.append(name)
            x0, y0, x1, y1 = FARM_CHUNK_BOUNDS[name]
            for tx, ty in last:
                if x0 <= tx <= x1 and y0 <= ty <= y1:
                    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        _set_tile(ram, tx + dx, ty + dy, 0xA1)
        self.assertEqual(seen, ["nw", "sw", "se"])
        self.assertEqual(count_debris(ram).stumps, 0)
        self.assertTrue(smash_is_clear(count_debris(ram)))


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
        status = observe_d2_farm(ram)
        self.assertEqual(next_d2_spec(status).phase, "ENSURE_HAMMER")
        rocks = next_d2_spec(status, last_phase="ENSURE_HAMMER")
        self.assertEqual(rocks.phase, "CLEAR_ROCKS")
        self.assertEqual(rocks.params["chunk"], "se")
        self.assertIsNone(
            next_d2_spec(status, section="stumps", last_phase="ENSURE_AXE")
        )

    def test_section_all_green_requires_empty_weeds(self) -> None:
        from harvest.planner.d2_farm_chunks import smash_done_empty, wanted_quota

        self.assertIn("weeds", smash_done_empty("all"))
        self.assertEqual(
            smash_done_empty("all"),
            ("weeds", "fences", "stones", "large_rocks", "stumps"),
        )
        self.assertEqual(smash_done_empty("bushes"), ("weeds",))
        self.assertEqual(wanted_quota("all").weeds, EXHAUSTIVE)
        self.assertEqual(wanted_quota("bushes").weeds, EXHAUSTIVE)

    def test_leftover_smash_is_required_so_a_day_plan_cannot_skip_a_stall(self) -> None:
        ram = _make_farm_ram()
        _set_tile(ram, 40, 40, 0x03)
        spec = next_d2_spec(observe_d2_farm(ram))
        self.assertEqual(spec.phase, "CLEAR_BUSHES")
        self.assertEqual(spec.params["chunk"], "se")
        self.assertEqual(spec.failure_policy, "required")
        self.assertEqual(bush_clear_phase().failure_policy, "required")
        self.assertEqual(rock_clear_phase().failure_policy, "required")
        self.assertEqual(stump_clear_phase().failure_policy, "required")


if __name__ == "__main__":
    unittest.main()
