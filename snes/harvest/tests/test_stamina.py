"""Stamina RAM object, rock 8-swing budget, spa full-restore (no ROM)."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import field_spec
from harvest.core.stamina import (
    MULTI_HIT_SWING_BUDGET,
    ROM_MULTI_HITS,
    SWING_STAMINA_COST,
    Stamina,
    stamina_cost_for_hits,
    swings_to_finish_multi_hit,
)
from harvest.core.tile_catalog import (
    ADDR_MAP,
    DebrisType,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    WEED,
    Tool,
)
from harvest.core.world_context import WorldContext
from harvest.core.world_snapshot import WorldSnapshot
from harvest.planner.day_phase_stamina import evening_clear_phases, full_restore_spa_phase
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind
from harvest.planner.day_plan_phases import build_day_phases
from harvest.tasks.farm_clearer import FarmClearer
from harvest.tasks.farm_ops import Target
from harvest.tasks.hot_spring import HotSpringStaminaTask, read_max_stamina, read_stamina
from harvest.tasks.nav import Point
from retro_harness import TaskStatus, WorldState


ADDR_STAMINA = field_spec("stamina").address
ADDR_MAX_STAMINA = field_spec("max_stamina").address
ADDR_EXHAUSTION = field_spec("exhaustion_level").address
ADDR_TOOL_HITS = field_spec("tool_hit_counter").address
ADDR_TILEMAP = field_spec("tilemap").address


def _blank_ram() -> np.ndarray:
    return np.zeros(0x20000, dtype=np.uint8)


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_large_rock(ram: np.ndarray, tx: int, ty: int) -> None:
    for dx, dy, tid in ((0, 0, 0x0D), (1, 0, 0x0E), (0, 1, 0x0F), (1, 1, 0x10)):
        _set_tile(ram, tx + dx, ty + dy, tid)


class StaminaObjectTests(unittest.TestCase):
    def test_from_ram_direct_wram_on_live_sized_buffer(self) -> None:
        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[ADDR_STAMINA] = 42
        ram[ADDR_MAX_STAMINA] = 100
        stam = Stamina.from_ram(ram)
        self.assertEqual(stam.current, 42)
        self.assertEqual(stam.maximum, 100)

    def test_from_ram_live_mirror_when_max_is_there(self) -> None:
        from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET

        ram = np.zeros(0x24000, dtype=np.uint8)
        ram[LIVE_RAM_WRAM_OFFSET + ADDR_STAMINA] = 77
        ram[LIVE_RAM_WRAM_OFFSET + ADDR_MAX_STAMINA] = 130
        stam = Stamina.from_ram(ram)
        self.assertEqual(stam.current, 77)
        self.assertEqual(stam.maximum, 130)

    def test_from_ram_object_fields(self) -> None:
        ram = _blank_ram()
        ram[ADDR_STAMINA] = 42
        ram[ADDR_MAX_STAMINA] = 130
        ram[ADDR_EXHAUSTION] = 1
        ram[ADDR_TOOL_HITS] = 3
        stam = Stamina.from_ram(ram)
        self.assertEqual(stam.current, 42)
        self.assertEqual(stam.maximum, 130)
        self.assertEqual(stam.exhaustion, 1)
        self.assertEqual(stam.tool_hits, 3)
        self.assertEqual(int(stam), 42)
        self.assertEqual(stam, 42)
        self.assertFalse(stam.is_full)
        self.assertEqual(stam.deficit, 88)
        data = stam.to_dict()
        self.assertEqual(data["current"], 42)
        self.assertEqual(data["maximum"], 130)
        self.assertEqual(data["tool_hits"], 3)
        self.assertFalse(data["is_full"])

    def test_zero_max_floors_to_100(self) -> None:
        ram = _blank_ram()
        ram[ADDR_STAMINA] = 80
        stam = Stamina.from_ram(ram)
        self.assertEqual(stam.maximum, 100)
        self.assertFalse(stam.is_full)

    def test_eight_swing_budget_is_16_stamina(self) -> None:
        self.assertEqual(ROM_MULTI_HITS, 6)
        self.assertEqual(MULTI_HIT_SWING_BUDGET, 8)
        self.assertEqual(swings_to_finish_multi_hit(0), 8)
        self.assertEqual(stamina_cost_for_hits(6), 16)
        self.assertEqual(stamina_cost_for_hits(1), SWING_STAMINA_COST)
        short = Stamina(current=15, maximum=100)
        self.assertFalse(short.can_finish_multi_hit())
        enough = Stamina(current=16, maximum=100)
        self.assertTrue(enough.can_finish_multi_hit())
        # Four registered hits already: 2 remaining + 2 miss = 4 swings / 8 stam.
        mid = Stamina(current=8, maximum=100, tool_hits=4)
        self.assertTrue(mid.can_finish_multi_hit())
        self.assertFalse(Stamina(current=7, maximum=100, tool_hits=4).can_finish_multi_hit())

    def test_world_snapshot_player_stamina_is_object(self) -> None:
        ram = _blank_ram()
        ram[field_spec("player_x").address] = 8
        ram[field_spec("player_y").address] = 8
        ram[ADDR_STAMINA] = 88
        ram[ADDR_MAX_STAMINA] = 130
        ram[ADDR_TOOL_HITS] = 2
        snap = WorldSnapshot.from_ram(ram)
        self.assertIsInstance(snap.player.stamina, Stamina)
        self.assertEqual(snap.player.stamina.current, 88)
        self.assertEqual(snap.player.stamina.maximum, 130)
        self.assertEqual(snap.player.stamina.tool_hits, 2)
        data = snap.player.to_dict()
        self.assertEqual(data["stamina"]["current"], 88)
        self.assertEqual(data["stamina"]["maximum"], 130)
        self.assertEqual(data["stamina"]["tool_hits"], 2)

    def test_world_context_returns_stamina_object(self) -> None:
        ram = _blank_ram()
        ram[ADDR_STAMINA] = 42
        ram[ADDR_MAX_STAMINA] = 100
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        ctx = WorldContext().bind(world)
        stam = ctx.stamina(ram)
        self.assertIsInstance(stam, Stamina)
        self.assertEqual(stam, 42)
        snap = ctx.snapshot_dict(ram)
        self.assertEqual(snap["stamina"], 42)
        self.assertEqual(snap["stamina_state"]["current"], 42)
        self.assertEqual(snap["stamina_state"]["maximum"], 100)


class RockStaminaGateTests(unittest.TestCase):
    def test_target_stamina_to_clear_uses_eight_swings(self) -> None:
        large = Target((5, 5), Point(88, 88), DebrisType.ROCK, LARGE_ROCK_TL)
        self.assertEqual(large.required_hits, 6)
        self.assertEqual(large.stamina_to_clear(lifting=False), 16)
        self.assertEqual(large.stamina_to_clear(lifting=False, tool_hits=5), 6)

    def test_twelve_stamina_does_not_start_large_rock(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 12
        ram[ADDR_MAX_STAMINA] = 100
        ram[field_spec("tool_selected").address] = int(Tool.HAMMER)
        ram[field_spec("input_lock").address] = 1
        _place_large_rock(ram, 11, 10)
        ram[field_spec("player_x").address] = 10 * 16 + 8
        ram[field_spec("player_y").address] = 10 * 16 + 8
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.navigator.update(ram)
        clearer.tool_manager.update(ram)
        nxt = clearer._handle_scanning(ram)
        self.assertEqual(nxt, "complete")
        self.assertTrue(clearer.stamina_exhausted)
        self.assertIsNone(clearer.current_target)
        self.assertEqual(ram[ADDR_MAP + 10 * MAP_WIDTH + 11], LARGE_ROCK_TL)

    def test_sixteen_stamina_starts_large_rock(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 16
        ram[ADDR_MAX_STAMINA] = 100
        ram[field_spec("tool_selected").address] = int(Tool.HAMMER)
        ram[field_spec("input_lock").address] = 1
        _place_large_rock(ram, 11, 10)
        ram[field_spec("player_x").address] = 10 * 16 + 8
        ram[field_spec("player_y").address] = 10 * 16 + 8
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.navigator.update(ram)
        clearer.tool_manager.update(ram)
        nxt = clearer._handle_scanning(ram)
        self.assertIn(nxt, {"navigating", "clearing"})
        assert clearer.current_target is not None
        self.assertEqual(clearer.current_target.tile, (11, 10))

    def test_low_stamina_still_lifts_weeds(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 2
        ram[ADDR_MAX_STAMINA] = 100
        ram[field_spec("input_lock").address] = 1
        ram[field_spec("player_x").address] = 10 * 16 + 8
        ram[field_spec("player_y").address] = 10 * 16 + 8
        _set_tile(ram, 11, 10, WEED)
        clearer = FarmClearer()
        clearer.startup_done = True
        clearer.navigator.update(ram)
        action = clearer.tick(ram)
        self.assertIsNotNone(action)
        self.assertFalse(clearer.stamina_exhausted)


class SpaFullRestoreTests(unittest.TestCase):
    def test_default_task_fills_to_max(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 90
        ram[ADDR_MAX_STAMINA] = 100
        ram[field_spec("input_lock").address] = 1
        task = HotSpringStaminaTask()
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        result = task.step(WorldState(frame=0, ram=ram, info={}, obs=None))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "route_mountain")

    def test_full_on_farm_succeeds(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_STAMINA] = 100
        ram[ADDR_MAX_STAMINA] = 100
        ram[field_spec("input_lock").address] = 1
        task = HotSpringStaminaTask()
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        result = task.step(WorldState(frame=0, ram=ram, info={}, obs=None))
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already sufficient", result.reason or "")

    def test_helpers_match_object(self) -> None:
        ram = _blank_ram()
        ram[ADDR_STAMINA] = 55
        ram[ADDR_MAX_STAMINA] = 130
        self.assertEqual(read_stamina(ram), 55)
        self.assertEqual(read_max_stamina(ram), 130)

    def test_evening_low_stam_inserts_spa_then_clear(self) -> None:
        phases = evening_clear_phases(
            has_debris=True,
            late_day=True,
            policy=DayPlannerPolicy(include_end_day=False),
            stamina=Stamina(current=10, maximum=100),
        )
        names = [p.phase for p in phases]
        self.assertEqual(names, ["HOT_SPRING_STAMINA", "CLEAR_FIELD"])
        self.assertEqual(phases[0].params.get("min_stamina"), "full")

    def test_evening_enough_stam_skips_spa(self) -> None:
        phases = evening_clear_phases(
            has_debris=True,
            late_day=True,
            policy=DayPlannerPolicy(include_end_day=False),
            stamina=Stamina(current=16, maximum=100),
        )
        self.assertEqual([p.phase for p in phases], ["CLEAR_FIELD"])

    def test_morning_never_inserts_spa(self) -> None:
        phases = evening_clear_phases(
            has_debris=True,
            late_day=False,
            policy=DayPlannerPolicy(),
            stamina=Stamina(current=4, maximum=100),
        )
        self.assertEqual(phases, [])

    def test_build_day_phases_evening_low_stam_inserts_spa(self) -> None:
        phases = build_day_phases(
            hour=17,
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            stamina=Stamina(current=8, maximum=100),
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        names = [p.phase for p in phases]
        self.assertIn("HOT_SPRING_STAMINA", names)
        self.assertLess(names.index("HOT_SPRING_STAMINA"), names.index("CLEAR_FIELD"))

    def test_full_restore_spa_phase_contract(self) -> None:
        spec = full_restore_spa_phase()
        self.assertEqual(spec.kind, PhaseKind.HOT_SPRING)
        self.assertEqual(spec.params["min_stamina"], "full")
        self.assertTrue(spec.params["return_to_farm"])


if __name__ == "__main__":
    unittest.main()
