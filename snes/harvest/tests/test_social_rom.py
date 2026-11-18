"""ROM receipts for the courting calendar: Sunday church, a weekday town question, Star Night for the wife.

Natural pins of one score replay from the Fall D1 power-on pin (local, gitignored); controller input only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harvest.core.ram_catalog import read_ram_value as value
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

PINS = Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes"
GIRLS = ("maria", "ann", "nina", "ellen", "eve")


@pytest.mark.rom
def test_opening_day_two_plants_before_any_courting_detour(monkeypatch):
    from harvest.planner import run_strategy, social_calendar
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import count_ring_planted
    from harvest.tasks.nav import make_action

    monkeypatch.setenv(run_strategy.ENV_VAR, "score_herd")
    monkeypatch.setenv(social_calendar.ENV_VAR, "1")
    env = _env("PR_tenberries_20261111_Y1_Spring_D2", "harvest.opening_courting")
    try:
        task = MultiDayPlannerTask(target_days=1)
        world = WorldState(0, env.get_ram(), {}, None)
        task.reset(world)
        phases = task._build_day_task(world).phases
        assert all(spec.phase != "SOCIAL_TOWN" for spec in phases)
        for frame in range(15_000):
            world = WorldState(frame, env.get_ram(), {}, None)
            result = task.step(world)
            if count_ring_planted(world.ram, WEST_POCKET_PLANT_CENTER) >= 8:
                break
            assert result.status == TaskStatus.RUNNING, (frame, result.reason)
            env.step(result.action.action if result.action else make_action())
        assert count_ring_planted(env.get_ram(), WEST_POCKET_PLANT_CENTER) >= 8
        assert value(env.get_ram(), "money") < 300
        assert value(env.get_ram(), "day") == 2
        assert not task._day_failures
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("OPENING_PLANTED", frame, value(env.get_ram(), "hour"), env.audit())
    finally:
        env.close()


@pytest.mark.rom
def test_harvest_festival_leaves_after_the_berry_without_losing_the_day():
    from harvest.planner.tasks.festival_escape import FestivalEscapeTask
    from harvest.planner.tasks.home_sleep import GoToSleepTask

    env = _env("LN_berries_20261103_harvest_festival_Y1_Fall_D12_after", "harvest.harvest_festival_exit")
    try:
        before = {key: value(env.get_ram(), key) for key in ("day", "power_berry_count", "max_stamina")}
        before_hearts = _hearts(env)
        result, frame = _run(env, FestivalEscapeTask(), 20_000)
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
        assert value(env.get_ram(), "tilemap") != 0x09
        assert {key: value(env.get_ram(), key) for key in before} == before
        assert value(env.get_ram(), "eve_hearts") >= before_hearts["eve"] + 30
        print("HARVEST_EXIT", frame, result.reason, _hearts(env))
        result, frame = _run(env, GoToSleepTask(), 20_000, frame + 1)
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
        assert value(env.get_ram(), "day") == before["day"] + 1
        assert all(value(env.get_ram(), key) == before[key] for key in ("power_berry_count", "max_stamina"))
        print("HARVEST_NEXT_MORNING", frame, env.audit())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def _env(pin, name):
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    from harvest.runtime.retro_setup import make_harvest_env

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all(name))
    env.reset()
    return env


def _run(env, task, limit, frame=0):
    from harvest.tasks.nav import make_action

    task.reset(WorldState(frame, env.get_ram(), {}, None))
    result = None
    for frame in range(frame, frame + limit):
        result = task.step(WorldState(frame, env.get_ram(), {}, None))
        if result.status != TaskStatus.RUNNING:
            break
        env.step(result.action.action if result.action else make_action())
    return result, frame


def _exit_house(env):
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_plan import DayPlanTask

    result, frame = _run(env, DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE]), 6_000)
    assert result.status == TaskStatus.SUCCESS, result.reason
    return frame


def _hearts(env):
    return {girl: value(env.get_ram(), f"{girl}_hearts") for girl in GIRLS}


@pytest.mark.rom
def test_sunday_church_pays_maria_and_ann_four_a_reentry_and_walks_home():
    from harvest.tasks.relationships import ChurchCourtTask

    env = _env("PR_f1ctl_s1_Y1_Fall_D3", "harvest.church_court")
    try:
        before = _hearts(env)
        frame = _exit_house(env)
        result, _ = _run(env, ChurchCourtTask(targets={"maria": 199, "ann": 199}, max_entries=3), 30_000, frame)
        after = _hearts(env)
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert after["maria"] - before["maria"] == 12 and after["ann"] - before["ann"] == 12
        assert all(after[g] == before[g] for g in ("nina", "ellen", "eve"))
        assert value(env.get_ram(), "tilemap") == 0x00
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_weekday_fall_ellen_question_pays_eight_a_town_reload():
    from harvest.tasks.relationships import TownCourtTask

    env = _env("PR_f1ctl_s1_Y1_Fall_D4", "harvest.town_court")
    try:
        before = _hearts(env)
        frame = _exit_house(env)
        result, _ = _run(env, TownCourtTask(girl="ellen", cap=199, loops=2), 30_000, frame)
        after = _hearts(env)
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert after["ellen"] - before["ellen"] == 16
        assert all(after[g] == before[g] for g in ("maria", "ann", "nina", "eve"))
        assert value(env.get_ram(), "tilemap") == 0x00
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_star_night_picks_the_wifes_scene_and_pays_her_instead_of_ellen(monkeypatch):
    from harvest.planner import social_calendar
    from harvest.planner.tasks.festival_escape import StarNightSceneTask, star_night_menu_press, star_night_scene_map
    from harvest.tasks.nav import make_action

    monkeypatch.setenv(social_calendar.ENV_VAR, "1")
    env = _env("PR_e2ec1_Y1_Winter_D24", "harvest.star_night")
    try:
        before = _hearts(env)
        state, frame = {}, 0
        while frame < 4_000 and not star_night_scene_map(env.get_ram()):
            press = star_night_menu_press(env.get_ram(), state, frame)
            env.step(press.action.action if press else make_action())
            frame += 1
        assert value(env.get_ram(), "tilemap") == 0x1E, state
        result, _ = _run(env, StarNightSceneTask(), 40_000, frame)
        after = _hearts(env)
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert after["eve"] - before["eve"] == 30          # the house entry's -2 lands later, at 18:xx
        assert all(after[g] == before[g] for g in ("maria", "ann", "nina", "ellen"))
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_weekday_winter_nina_question_pays_eight_a_flower_shop_reload():
    from harvest.tasks.relationships import TownCourtTask

    env = _env("PR_e2ec1_Y1_Winter_D20", "harvest.town_court")
    try:
        before = _hearts(env)
        frame = _exit_house(env)
        result, _ = _run(env, TownCourtTask(girl="nina", cap=300, loops=2), 30_000, frame)
        after = _hearts(env)
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert after["nina"] - before["nina"] == 16
        assert all(after[g] == before[g] for g in ("maria", "ann", "ellen", "eve"))
        assert value(env.get_ram(), "tilemap") == 0x00
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
