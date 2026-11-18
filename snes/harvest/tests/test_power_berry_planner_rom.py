"""Power berries through the day planner, not the bare task (ROM, ``-m rom``).

``test_power_berries_rom.py`` drives ``PowerBerryTask`` on its own. A campaign runs it inside ``DayPlanTask``, whose
step-level berry hook once took the Kappa's cutscene sprite for a stray berry and failed the whole day
(``no power berry ground sprite``, Y1 Spring D5). These tests build the morning's plan from the pin's RAM with
``HARVEST_STRATEGY=score``, require the calendar to pick the expected source, and run that plan until the berry is eaten.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from harvest.core.ram_catalog import read_ram_u16, read_ram_value
from harvest.tasks.nav import make_action
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

PINS = Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes"


def _plan_and_run(pin: str, source: str, *, limit: int = 60_000, farm_idle: int = 0, done_bits=()):
    """Build the morning plan from ``pin``, check it leads with ``source``, run until the berry count rises."""
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.runtime.retro_setup import make_harvest_env

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.berry_planner"))
    old = os.environ.get("HARVEST_STRATEGY")
    os.environ["HARVEST_STRATEGY"] = "score"
    try:
        env.reset()
        for address, bit in done_bits:
            # Research poke (declared): mark an earlier source done, as a score run would have by this date.
            word = read_ram_u16(env.get_ram(), address) | bit
            for offset, byte in enumerate(word.to_bytes(2, "little")):
                env.data.memory.assign(0x7E0000 + address + offset, "|u1", byte)
            env.record_ram_write(2)
        if done_bits:
            env.begin_attempt()
        ram = env.get_ram()
        before = int(read_ram_value(ram, "power_berry_count"))
        maximum = int(read_ram_value(ram, "max_stamina"))
        phases = build_day_phases_from_ram(ram, policy=DayPlannerPolicy(), state_name=pin)
        names = [p.phase for p in phases]
        assert f"COLLECT_POWER_BERRY_{source.upper()}" in names, names
        berry_at = names.index(f"COLLECT_POWER_BERRY_{source.upper()}")
        assert "DYNAMIC_OUTDOOR_PLAN" in names[berry_at:], "the trip must be followed by a plan from live RAM"
        # Run only up to and including the berry phase (the rest of the day is the planner's ordinary work).
        task = DayPlanTask(phase_sequence=phases[: berry_at + 1], state_name=pin)
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        left = farm_idle
        result = None
        for frame in range(limit):
            ram = env.get_ram()
            if left > 0 and int(read_ram_value(ram, "tilemap")) == 0:
                left -= 1
                env.step(make_action())
                continue
            result = task.step(WorldState(frame=frame, ram=ram, info={}, obs=None))
            if int(read_ram_value(ram, "power_berry_count")) > before and result.status != TaskStatus.RUNNING:
                break
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        return (int(read_ram_value(ram, "power_berry_count")) - before,
                int(read_ram_value(ram, "max_stamina")) - maximum, result.status, result.reason, read_ram_u16(ram, 0x11F6C))
    finally:
        env.close()
        if old is None:
            os.environ.pop("HARVEST_STRATEGY", None)
        else:
            os.environ["HARVEST_STRATEGY"] = old


@pytest.mark.rom
@pytest.mark.parametrize("pin", ["PR_spring_final_Y1_Spring_D4", "PR_tenberries_fixed_20261111_Y1_Spring_D5"])
def test_planner_sends_the_farmer_to_the_statue_on_the_first_free_spring_day(pin):
    count, stamina, status, reason, _ = _plan_and_run(pin, "statue")
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_planner_runs_the_fish_trip_without_the_plan_hook_stealing_the_kappas_berry():
    # Day 4 itself schedules the statue first, so the fish trip is planned on a pin where the statue is already done.
    pin = "LN_berries_20261103_statue_Y1_Spring_D4_after"
    count, stamina, status, reason, _ = _plan_and_run(pin, "fish", limit=80_000)
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_planner_runs_the_cave_in_winter_once_the_pocket_is_open():
    count, stamina, status, reason, _ = _plan_and_run("PR_y1new_Y1_Winter_D8", "cave", limit=60_000)
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_late_statue_recipe_fetches_the_native_gold_hammer(monkeypatch):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.power_berry import statue_recipe
    from harvest.core.carry import carry_pair_items
    from harvest.planner.day_plan import DayPlanTask, PhaseSpec

    pin = "PR_e2es4_Y3_Spring_D10"
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is absent")
    monkeypatch.setenv("HARVEST_GOLD_STATUE", "1")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.statue_hammer"))
    try:
        env.reset()
        recipe = statue_recipe(env.get_ram())
        hammer = recipe.tasks[0].tool_id
        assert hammer == 0x13
        plan = DayPlanTask(phase_sequence=[PhaseSpec("ENSURE_STATUE_HAMMER", "ensure_tool", {"tool_id": hammer})])
        plan.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(8000):
            result = plan.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert hammer in carry_pair_items(env.get_ram())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_planner_visits_the_missing_stump_after_the_early_berries_are_done():
    count, stamina, status, reason, _ = _plan_and_run(
        "PR_tenberries_guarded_20261111_Y1_Summer_D23", "stump", limit=80_000)
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_planner_takes_the_hawkers_hen_trade_on_a_fall_morning():
    # The pin comes from a cash run (no berries); a score run has finished the statue and the fish by Fall.
    count, stamina, status, reason, _ = _plan_and_run(
        "PR_y1new_Y1_Fall_D13", "chicken_trade", limit=60_000, done_bits=((0x11F64, 0x0200), (0x11F66, 0x2000)))
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_fresh_fall_hawker_return_reaches_the_berry_inside_the_west_gate():
    count, stamina, status, reason, _ = _plan_and_run(
        "PR_tenberries_guarded_20261111_Y1_Fall_D1", "chicken_trade", limit=60_000)
    assert (count, stamina) == (1, 10), (status, reason)


@pytest.mark.rom
def test_hawker_gate_route_also_returns_a_normally_bought_cow():
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.day_phase_catalog import animal_purchase_phases, EXIT_TO_FARM_PHASE
    from harvest.planner.day_plan_status import is_farm_tilemap
    from harvest.runtime.retro_setup import make_harvest_env

    pin = "PR_tenberries_guarded_20261111_Y1_Summer_D16"
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.animal_return"))
    try:
        env.reset()
        before = {key: int(read_ram_value(env.get_ram(), key)) for key in ("money", "num_cows")}
        task = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE, *animal_purchase_phases("cow")])
        task.reset(WorldState(0, env.get_ram(), {}, None))
        for frame in range(30_000):
            result = task.step(WorldState(frame, env.get_ram(), {}, None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
        assert int(read_ram_value(env.get_ram(), "num_cows")) == before["num_cows"] + 1
        assert int(read_ram_value(env.get_ram(), "money")) == before["money"] - 5000
        assert is_farm_tilemap(int(read_ram_value(env.get_ram(), "tilemap")))
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_planner_offers_the_harvest_festival_fish(monkeypatch):
    monkeypatch.setenv("HARVEST_HARVEST_FESTIVAL_BERRY", "1")   # off by default: the festival map traps the day
    # The optional feather trip is deferred so feeding and the dated fish trip fit.
    count, stamina, status, reason, flags = _plan_and_run("PR_y1new_Y1_Fall_D12", "harvest_festival", limit=60_000,
                                                            farm_idle=0)
    assert (count, stamina) == (1, 10), (status, reason)
    assert flags & 0x8000


@pytest.mark.rom
@pytest.mark.parametrize("farm_idle", [100, 900])
def test_planner_egg_festival_receipts_follow_the_native_prize(farm_idle):
    count, stamina, status, reason, flags = _plan_and_run("PR_e2ec1_Y2_Fall_D20", "egg_festival", limit=60_000,
                                                            farm_idle=farm_idle)
    # Chores change RNG timing. Both native prizes are valid; the bare-task
    # tests exercise each face separately, including its exact eat receipt.
    assert (flags & 0x7000, count, stamina) in {(0x1000, 0, 0), (0x6000, 1, 10)}, (status, reason)


@pytest.mark.rom
@pytest.mark.parametrize("pin,season,day,expected_count", [
    ("PR_tenberries_guarded_20261111_Y1_Fall_D1", 2, 1, 5),
    ("PR_tenberries_gate_20261111_Y1_Fall_D12", 2, 12, 7),
    ("PR_tenberries_gate_20261111_Y1_Fall_D20", 2, 20, None),
    ("PR_tenberries_egg_20261111_Y1_Winter_D1", 3, 1, 9),
    ("PR_tenberries_egg_20261111_Y1_Winter_D8", 3, 8, 10),
])
def test_herd_morning_retains_scheduled_berry_receipts_overnight(pin, season, day, expected_count):
    """Run the real campaign, including chores, festival arbitration and sleep."""
    import json
    from harvest.planner import run_strategy
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from harvest.planner.campaign_composer import _resync_loaded_pin, _restore_planner
    from harvest.runtime.retro_setup import make_harvest_env

    if not (PINS / f"{pin}.lineage.json").exists():
        pytest.skip(f"local pin {pin} is not present")
    old = run_strategy.active().name
    run_strategy.select("score_herd")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.berry_overnight"))
    try:
        env.reset()
        lineage = json.loads((PINS / f"{pin}.lineage.json").read_text())
        lineage["path"] = str(PINS / f"{pin}.state")
        _resync_loaded_pin(env, lineage, env.get_ram())
        env.begin_attempt()
        task = MultiDayPlannerTask(target_days=1)
        task.reset(WorldState(0, env.get_ram(), {}, None))
        _restore_planner(task, lineage["planner"])
        for frame in range(60_000):
            result = task.step(WorldState(frame, env.get_ram(), {}, None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        assert result.status == TaskStatus.SUCCESS, result.reason
        count, maximum = (int(read_ram_value(ram, key)) for key in ("power_berry_count", "max_stamina"))
        assert read_ram_value(ram, "season") == season
        assert read_ram_value(ram, "day") == day + 1
        assert maximum == 100 + 10 * count
        if expected_count is not None:
            assert count == expected_count
        else:
            assert (read_ram_u16(ram, 0x11F6C) & 0x7000, count) in {(0x1000, 7), (0x6000, 8)}
        if season == 3:
            from harvest.planner.power_berry_calendar import snow_seed_owned

            assert not snow_seed_owned(ram)
            assert bool(read_ram_u16(ram, 0x11F6A) & 0x0040) == (day == 8)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
        run_strategy.select(old)
