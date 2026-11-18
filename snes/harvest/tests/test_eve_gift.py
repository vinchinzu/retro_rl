"""Married Eve's morning egg gift (group $43 script 4: an egg is +4 hearts)."""
from __future__ import annotations

import unittest

import numpy as np
import pytest

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, OPTIONAL_MONEY_PHASES
from harvest.planner.day_phase_chicken import WIFE_GIFT_PHASES
from harvest.tasks.eve_gift import EVE_GIFT_TARGET, EVE_SPAWN_WAIT_FRAMES as GIFT_SPAWN_FRAMES
from harvest.tasks.eve_gift import GiveWifeTask, apply_wife_gift, wife_gift_wanted, wife_settle_cap
from harvest.tasks.nav import make_action
from retro_harness import TaskStatus, WorldState


def _write(ram: np.ndarray, key: str, value: int) -> None:
    spec = field_spec(key)
    addr = ram_index(ram, spec.address, live_offset=spec.live_offset)
    for i in range(2 if spec.kind == "u16" else 1):
        ram[addr + i] = (value >> (8 * i)) & 0xFF


def _house(*, hearts: int = 300, married: bool = True, egg: bool = True, tilemap: int = 0x16,
           hour: int = 6, held: int = 0x14) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    _write(ram, "tilemap", tilemap)
    _write(ram, "marriage_flags", 0x4010 if married else 0x4000)
    _write(ram, "eve_hearts", hearts)
    _write(ram, "egg_available", 0x0001 if egg else 0)
    _write(ram, "hour", hour)
    _write(ram, "held_item", held)
    _write(ram, "input_lock", 1)
    return ram


class WifeGiftPlanTests(unittest.TestCase):
    def test_only_egg_rebuilds_the_laying_pair_before_wife_gifts(self):
        from harvest.core.animal_status import CHICKEN_SLOT_BASE
        ram = _house()
        _write(ram, "num_chickens", 1)
        _write(ram, "planted_grass", 160)
        ram[0x4000 + CHICKEN_SLOT_BASE] = 0x09
        phases = [EXIT_TO_FARM_PHASE]
        self.assertEqual(apply_wife_gift(phases, ram), phases)
        ram[0x4000 + CHICKEN_SLOT_BASE + 8] = 0x03  # incubating replacement
        self.assertIn("GIVE_WIFE", [p.phase for p in apply_wife_gift(phases, ram)])

    def test_score_family_keeps_both_eggs_above_the_old_cap_until_child_two(self):
        from unittest.mock import patch

        ram = _house(hearts=502, tilemap=0x17)
        _write(ram, "egg_available", 3)
        with patch.dict("os.environ", {"HARVEST_STRATEGY": "score_herd"}):
            self.assertTrue(wife_gift_wanted(ram))
            self.assertEqual(sum(p.phase == "GIVE_WIFE" for p in apply_wife_gift([EXIT_TO_FARM_PHASE], ram)), 2)
            _write(ram, "incubator_flags", 0x000C)
            self.assertEqual(sum(p.phase == "GIVE_WIFE" for p in apply_wife_gift([EXIT_TO_FARM_PHASE], ram)), 1)
            _write(ram, "eve_hearts", 654)
            self.assertFalse(wife_gift_wanted(ram))

    def test_final_morning_tops_up_the_494_ending_failure(self):
        ram = _house(hearts=502)
        _write(ram, "year", 2)
        _write(ram, "season", 1)
        _write(ram, "day", 29)
        _write(ram, "incubator_flags", 0x000C)
        self.assertTrue(wife_gift_wanted(ram))

    def test_late_family_gifts_reach_650_before_the_dated_downward_correction(self):
        from unittest.mock import patch

        ram = _house(hearts=646, tilemap=0x17)
        for name, value in (("year", 2), ("season", 0), ("day", 3),
                            ("kid1_age", 90), ("egg_available", 3), ("incubator_flags", 4)):
            _write(ram, name, value)
        with patch.dict("os.environ", {"HARVEST_STRATEGY": "score_herd"}):
            phases = apply_wife_gift([EXIT_TO_FARM_PHASE], ram)
            self.assertEqual(sum(p.phase == "GIVE_WIFE" for p in phases), 2)
            _write(ram, "incubator_flags", 12)
            phases = apply_wife_gift([EXIT_TO_FARM_PHASE], ram)
            # child 2 flagged and Eve far above the settle cap: the excess is shed by evening house entries
            # (-2 each at hour 18, no game time), not by a fish trip that costs the morning.
            self.assertFalse(any(p.phase == "GIVE_WIFE" for p in phases))
            self.assertIn("HEART_DRAIN_WAIT", [p.phase for p in phases])
            with patch.dict("os.environ", {"HARVEST_HEART_DRAIN": "0"}):
                phases = apply_wife_gift([EXIT_TO_FARM_PHASE], ram)
                self.assertEqual(next(p.params for p in phases if p.phase == "GIVE_WIFE"), {"fish": True})
            # The child2 age-60 +50 scene still fits before the ending; reserve
            # that and Spring D23's +45 before relying on the nightly -2.
            _write(ram, "kid2_age", 5)
            self.assertEqual(wife_settle_cap(ram), 511 + 2 * 57 - 45 - 50)

    def test_fish_correction_preserves_the_last_ripe_crop_mornings(self):
        from unittest.mock import patch
        from harvest.tasks.eve_gift import wife_fish_gift_wanted
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.field_cover import FARM_MAP_ARRAY_ADDR
        from harvest.runtime.rom_tools import wram_offset

        ram = _house(hearts=654, tilemap=0x17)
        for name, value in (("year", 2), ("season", 1), ("day", 29), ("incubator_flags", 12)):
            _write(ram, name, value)
        # The farmer is indoors: only the persistent farm array has this crop.
        base = live_wram_base(ram) + wram_offset(FARM_MAP_ARRAY_ADDR)
        ram[base + 40 * 64 + 20] = 0x38  # ripe tomato, whose counter is still 0
        with patch.dict("os.environ", {"HARVEST_STRATEGY": "score_herd"}):
            self.assertFalse(wife_fish_gift_wanted(ram))
            _write(ram, "shipped_tomatoes", 496)
            self.assertTrue(wife_fish_gift_wanted(ram))

    def test_evening_drain_sheds_the_excess_two_hearts_an_entry_after_the_work(self):
        from unittest.mock import patch
        from harvest.tasks.eve_gift import (HEART_DRAIN_MARGIN, heart_drain_phases, wife_heart_drain_entries,
                                            wife_settle_cap)

        ram = _house(hearts=586, tilemap=0x17)
        for name, value in (("year", 2), ("season", 1), ("day", 2), ("incubator_flags", 12)):
            _write(ram, name, value)
        with patch.dict("os.environ", {"HARVEST_STRATEGY": "score_herd"}):
            cap = wife_settle_cap(ram)
            entries = wife_heart_drain_entries(ram)
            # ends at or just under cap - margin: each entry is exactly -2
            self.assertGreater(entries, 0)
            self.assertLessEqual(586 - 2 * entries, cap - HEART_DRAIN_MARGIN)
            self.assertGreater(586 - 2 * (entries - 1), cap - HEART_DRAIN_MARGIN)
            # nothing to shed at or under the margin, and never before child 2 is flagged (650 is needed first)
            _write(ram, "eve_hearts", cap - HEART_DRAIN_MARGIN)
            self.assertEqual(wife_heart_drain_entries(ram), 0)
            _write(ram, "eve_hearts", 586)
            _write(ram, "incubator_flags", 4)
            self.assertEqual(wife_heart_drain_entries(ram), 0)
            # also on the Summer planting mornings: the drain is an evening job and costs no morning
            _write(ram, "incubator_flags", 12)
            phases = apply_wife_gift([EXIT_TO_FARM_PHASE], ram)
            self.assertEqual(phases[0].phase, "EXIT_TO_FARM")
            drain = phases[1:]
            self.assertEqual(sum(p.phase == "ENTER_HOUSE_FOR_GIFT" for p in drain), entries)
            self.assertEqual(drain[0].phase, "EXIT_TO_FARM")
            self.assertEqual(drain[1].phase, "HEART_DRAIN_WAIT")
            self.assertEqual(drain[1].params["target_hour"], 18)
            self.assertEqual(drain[-1].phase, "ENTER_HOUSE_FOR_GIFT")  # the last entry stays home for the night
            self.assertTrue(all(p.params["target_tilemap"] == 0x17 for p in drain if p.phase == "ENTER_HOUSE_FOR_GIFT"))
        self.assertEqual(len(heart_drain_phases(1, 0x17)), 4)

    def test_fish_correction_leaves_the_summer_planting_mornings_alone(self):
        """Y3 Summer D1-D3 and D5 were fish trips: no corn or tomato bag was bought and the season shipped none."""
        from unittest.mock import patch
        from harvest.tasks.eve_gift import wife_fish_gift_wanted

        ram = _house(hearts=586, tilemap=0x17)
        for name, value in (("year", 2), ("season", 1), ("incubator_flags", 12), ("hour", 6)):
            _write(ram, name, value)
        with patch.dict("os.environ", {"HARVEST_STRATEGY": "score_herd"}):
            for day in range(1, 6):
                _write(ram, "day", day)
                self.assertFalse(wife_fish_gift_wanted(ram), day)
            _write(ram, "day", 6)
            self.assertTrue(wife_fish_gift_wanted(ram))
            # Spring keeps its corrections: they start the day the second child is flagged.
            _write(ram, "season", 0)
            _write(ram, "day", 20)
            self.assertTrue(wife_fish_gift_wanted(ram))

    def test_gift_mornings_need_eve_home_short_and_an_egg(self) -> None:
        self.assertTrue(wife_gift_wanted(_house()))
        self.assertFalse(wife_gift_wanted(_house(hearts=EVE_GIFT_TARGET)))
        self.assertFalse(wife_gift_wanted(_house(married=False)))
        self.assertFalse(wife_gift_wanted(_house(egg=False)))
        self.assertTrue(wife_gift_wanted(_house(tilemap=0x17)))

    def test_upgraded_house_gift_chain_returns_to_the_current_house(self) -> None:
        phases = apply_wife_gift([EXIT_TO_FARM_PHASE], _house(tilemap=0x17))
        entry = next(p for p in phases if p.phase == "ENTER_HOUSE_FOR_GIFT")
        self.assertEqual(entry.params["target_tilemap"], 0x17)
        # Building a second-house plan must not mutate the shared first-house recipe.
        original = next(p for p in WIFE_GIFT_PHASES if p.phase == "ENTER_HOUSE_FOR_GIFT")
        self.assertEqual(original.params["target_tilemap"], 0x16)

    def test_chain_runs_after_the_exit_and_before_the_day(self) -> None:
        day = [EXIT_TO_FARM_PHASE]
        planned = [p.phase for p in apply_wife_gift(day, _house())]
        self.assertEqual(planned[0], "EXIT_TO_FARM")
        self.assertEqual(planned[1:-1], [p.phase for p in WIFE_GIFT_PHASES])
        self.assertEqual(planned[-1], "EXIT_TO_FARM")
        self.assertEqual(apply_wife_gift(day, _house(hearts=510)), day)
        # Two eggs and Eve short of child 1's 450: two rounds.
        two = _house(hearts=300)
        _write(two, "egg_available", 0x0003)
        rounds = [p.phase for p in apply_wife_gift(day, two)]
        self.assertEqual(len(rounds), 3 + 2 * len(WIFE_GIFT_PHASES))
        self.assertEqual(rounds[len(WIFE_GIFT_PHASES) + 1], "EXIT_TO_FARM")  # out of the house again
        _write(two, "eve_hearts", 460)
        self.assertEqual(len(apply_wife_gift(day, two)), 2 + len(WIFE_GIFT_PHASES))
        # A failed step skips the rest of the chain, not the day.
        self.assertTrue({p.phase for p in WIFE_GIFT_PHASES} <= OPTIONAL_MONEY_PHASES)

    def test_failed_first_gift_round_skips_second_round_and_preserves_barn_route(self) -> None:
        from harvest.planner.day_plan import DayPlanTask, PhaseSpec
        from harvest.planner.day_phase_cow import EXIT_BARN_PHASE, NAV_TO_BARN_PHASE

        two = _house(hearts=300)
        _write(two, "egg_available", 0x0003)
        base = [EXIT_TO_FARM_PHASE, NAV_TO_BARN_PHASE, EXIT_BARN_PHASE]
        planned = apply_wife_gift(base, two)
        # Force the first gift phase after the initial EXIT_TO_FARM to fail immediately.
        planned[1] = PhaseSpec("NAV_TO_COOP_FOR_GIFT", "unknown_kind", failure_policy="optional")
        plan = DayPlanTask(phase_sequence=planned[1:])
        world = WorldState(frame=0, ram=_house(tilemap=0x00), info={}, obs=None)
        plan.reset(world)
        plan.step(world)
        remaining = [p.phase for p in plan._schedule.active[plan._phase_index :]]
        self.assertEqual(remaining, ["EXIT_TO_FARM", "NAV_TO_BARN", "EXIT_BARN"])
        self.assertEqual(plan._failure_policy(EXIT_BARN_PHASE), "required")


class GiveWifeTaskTests(unittest.TestCase):
    def _run(self, ram: np.ndarray, frames: int = 50):
        task = GiveWifeTask()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task.reset(world)
        return task, [task.step(world) for _ in range(frames)]

    def test_closed_hours_fail_and_drop_the_egg(self) -> None:
        task, results = self._run(_house(hour=12), frames=3)
        self.assertEqual(results[0].status, TaskStatus.RUNNING)
        self.assertIn("closed", results[0].reason)
        self.assertTrue(results[0].action.action[8])  # A drops the egg

    def test_waits_for_her_sprite_after_the_door(self) -> None:
        task, results = self._run(_house(), frames=GIFT_SPAWN_FRAMES + 2)
        self.assertEqual(results[0].status, TaskStatus.RUNNING)
        self.assertFalse(results[0].action.action.any())
        self.assertIn("wife not home", task._failed or "")

    def test_nothing_in_hand_fails_at_once(self) -> None:
        _task, results = self._run(_house(held=0), frames=1)
        self.assertEqual(results[0].status, TaskStatus.FAILURE)
        self.assertEqual(results[0].reason, "nothing in hand")


class LargeHouseBlockedStandTests(unittest.TestCase):
    """r47 Summer D21-D29: an NPC 13 px east of the farmer blocked the walker's last 2 px sidestep for nine mornings."""

    UP, DOWN, LEFT, RIGHT, A = 4, 5, 6, 7, 8

    def _task(self, pos):
        from unittest.mock import patch

        from harvest.tasks.nav import Point
        from tests.day_plan_test_helpers import set_player_pos

        ram = _house(tilemap=0x17)
        set_player_pos(ram, *pos)
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = GiveWifeTask()
        task.reset(world)
        patches = [patch("harvest.tasks.eve_gift.wife_sprite", return_value=Point(328, 120)),
                   patch("harvest.tasks.eve_gift.screen_faded_in", return_value=True),
                   patch("harvest.planner.tasks.inventory_exit.room_walk_action",
                         side_effect=lambda ram, target: make_action(right=True))]   # the sidestep the NPC blocks
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return task, world, ram

    def test_a_blocked_sidestep_goes_up_first_then_presses_from_a_near_column(self) -> None:
        from tests.day_plan_test_helpers import set_player_pos

        task, world, ram = self._task((326, 152))
        early = [task.step(world) for _ in range(30)]
        self.assertTrue(all(r.action.action[self.RIGHT] for r in early))        # still the room walker's sidestep
        results = [task.step(world) for _ in range(20)]
        self.assertTrue(any(r.action.action[self.UP] and not r.action.action[self.RIGHT] for r in results))
        self.assertNotEqual(results[-1].status, TaskStatus.FAILURE)             # 90 still frames never reached
        set_player_pos(ram, 326, 144)                                           # up under her, 2 px off her column
        pressed = []
        for _ in range(140):
            pressed.append(task.step(world))
        self.assertTrue(any(r.action.action[self.A] for r in pressed), "never pressed A under her")
        self.assertFalse(any(r.status == TaskStatus.FAILURE for r in pressed))

    def test_a_walker_that_is_moving_keeps_the_room_walker(self) -> None:
        from tests.day_plan_test_helpers import set_player_pos

        task, world, ram = self._task((326, 200))
        for x in range(326, 326 + 80):                                          # a free walk: the position changes every frame
            set_player_pos(ram, x % 400, 200)
            result = task.step(world)
            self.assertTrue(result.action.action[self.RIGHT])


@pytest.mark.rom
def test_egg_gift_raises_married_eve_by_four():
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from retro_harness import AuditedEnv, AuditCapabilities

    env = AuditedEnv(make_harvest_env("LN_y23_egg_home_probe", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.eve_gift_test"))
    try:
        env.reset()
        assert read_ram_value(env.get_ram(), "held_item") == 0x14
        task = GiveWifeTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(1500):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "eve_hearts") == 292
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


if __name__ == "__main__":
    unittest.main()


@pytest.mark.rom
def test_two_egg_gifts_in_the_larger_house_return_to_farm_with_empty_hands():
    from pathlib import Path
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.tasks.home_recover import sanitize_action
    from retro_harness import AuditedEnv, AuditCapabilities

    pin = "PR_e2es4_Y3_Spring_D3"
    if not (Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes" / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is absent")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.large_house_gift"))
    try:
        env.reset()
        before = int(read_ram_value(env.get_ram(), "eve_hearts"))
        phases = apply_wife_gift([EXIT_TO_FARM_PHASE], env.get_ram())
        assert sum(p.phase == "GIVE_WIFE" for p in phases) == 2
        task = DayPlanTask(phase_sequence=phases)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(30_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            action = result.action.action if result.action else make_action()
            env.step(sanitize_action(action, env.get_ram()))
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "eve_hearts") == before + 8
        assert read_ram_value(env.get_ram(), "tilemap") == 0
        assert read_ram_value(env.get_ram(), "held_item") == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_native_catch_can_lower_eves_hearts_and_leave_empty_hands():
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask, PhaseSpec
    from harvest.planner.tasks.home_recover import sanitize_action
    from retro_harness import AuditedEnv, AuditCapabilities

    env = AuditedEnv(make_harvest_env("PR_e2es4_Y3_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.wife_fish_gift"))
    try:
        env.reset()
        before = int(read_ram_value(env.get_ram(), "eve_hearts"))
        task = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE,
            PhaseSpec("GIVE_WIFE", "give_wife", {"fish": True}), EXIT_TO_FARM_PHASE])
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(32_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(sanitize_action(result.action.action if result.action else make_action(), env.get_ram()))
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "eve_hearts") == before - 8
        assert read_ram_value(env.get_ram(), "held_item") == 0
        assert read_ram_value(env.get_ram(), "tilemap") == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("WIFE_FISH_GIFT", frame, before, read_ram_value(env.get_ram(), "eve_hearts"),
              read_ram_value(env.get_ram(), "hour"), env.audit())
    finally:
        env.close()


@pytest.mark.rom
def test_research_two_eggs_cross_650_and_set_the_second_child_after_entry_loss(monkeypatch):
    """Declared research starting hearts/age; gifts and the overnight are native."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.tasks.home_sleep import GoToSleepTask
    from harvest.planner.tasks.home_recover import sanitize_action
    from retro_harness import AuditedEnv, AuditCapabilities

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    env = AuditedEnv(make_harvest_env("PR_e2es4_Y3_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.family_threshold_research"))
    try:
        env.reset()
        for name, value in (("eve_hearts", 646), ("kid1_age", 90)):
            spec = field_spec(name)
            for offset, byte in enumerate(value.to_bytes(2, "little")):
                env.data.memory.assign(0x7E0000 + spec.address + offset, "|u1", byte)
            env.record_ram_write(2)
        assert env.audit().ram_writes == 4
        env.begin_attempt()  # only the declared research setup above writes RAM
        phases = apply_wife_gift([EXIT_TO_FARM_PHASE], env.get_ram())
        assert sum(p.phase == "GIVE_WIFE" for p in phases) == 2
        frame = 0
        for task, limit in ((DayPlanTask(phase_sequence=phases), 24_000), (GoToSleepTask(), 24_000)):
            task.reset(WorldState(frame, env.get_ram(), {}, None))
            for frame in range(frame, frame + limit):
                result = task.step(WorldState(frame, env.get_ram(), {}, None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(sanitize_action(result.action.action if result.action else make_action(), env.get_ram()))
            assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "eve_hearts") == 652
        assert int(read_ram_value(env.get_ram(), "incubator_flags")) & 0x000C == 0x000C
        assert read_ram_value(env.get_ram(), "kid2_age") == 1
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("FAMILY_THRESHOLD_RESEARCH", frame, "setup_writes=4", "646->654->652",
              read_ram_value(env.get_ram(), "kid2_age"), env.audit())
    finally:
        env.close()
