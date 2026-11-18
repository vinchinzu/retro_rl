"""Crop quotas reserve both seed kinds and the morning shipping window."""
from pathlib import Path
from unittest.mock import patch

import pytest

from harvest.maps import farm_pond
from harvest.planner import run_strategy as rs
from harvest.planner.day_plan_status import resolve_seed_type_from_ram
from harvest.planner.day_phase_registry import _shop_bag_count
from harvest.planner.day_phase_catalog import buy_seeds_phase
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec
from harvest.planner.day_plan_phases import build_day_phases_from_ram, build_outdoor_day_phases_from_ram
from tests.layout_markers import legacy_rings_only
from tests.test_run_strategy import world, set_tile, set_field


@legacy_rings_only
def test_summer_buys_half_the_empty_plots_for_each_needed_crop(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, 'score')
    w = world(season=1, day=1)
    buy = buy_seeds_phase(seed_type='corn')
    assert _shop_bag_count(buy, w, 'corn_seeds') == 5


@legacy_rings_only
def test_summer_reserves_standing_plants_before_picking_the_next_bag(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, 'score')
    w = world(season=1, day=2)
    for center in farm_pond.POCKET_PLANT_CENTERS[:5]:
        for x, y in farm_pond.plot_crop_tiles(center):
            set_tile(w.ram, x, y, 0x3A)
    assert resolve_seed_type_from_ram(w.ram) == 'tomato'
    assert _shop_bag_count(buy_seeds_phase(seed_type='tomato'), w, 'tomato_seeds') == 4


def test_one_last_spring_bag_can_close_an_uneven_shipment_deficit(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, 'score')
    w = world(season=0, day=14, shipped={'potato': 490, 'turnip': 496})
    assert rs.one_shot_sowing_ok(w.ram)
    assert resolve_seed_type_from_ram(w.ram) == 'potato'


def test_score_expands_the_legacy_west_strip_in_year_two(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    if farm_pond.ring_layout_name() != 'legacy':
        return
    assert len(farm_pond.active_plot_centers(0)) == 9
    assert len(farm_pond.active_plot_centers(1)) == 12
    reserved = farm_pond.crop_working_tiles()
    for x, y in farm_pond.active_plot_centers(1):
        assert {(x+dx,y+dy) for dx in range(-2,3) for dy in range(-2,3)} <= reserved
    monkeypatch.setenv(rs.ENV_VAR, 'cash')
    assert len(farm_pond.active_plot_centers(1)) == 9


@pytest.mark.parametrize('inside', [False, True])
def test_score_harvests_before_courting_and_replans_from_live_ram(monkeypatch, inside):
    monkeypatch.setenv(rs.ENV_VAR, 'score')
    w = world(season=1, day=17)
    set_tile(w.ram, 13, 29, 0x52)
    if inside:
        from harvest.core.harvest_state import FARM_MAP_ARRAY_ADDR
        from harvest.core.ram_catalog import live_wram_base
        from harvest.runtime.rom_tools import wram_offset

        set_field(w.ram, 'tilemap', 0x17)
        base = live_wram_base(w.ram) + wram_offset(FARM_MAP_ARRAY_ADDR)
        w.ram[base + 29 * 64 + 13] = 0x52
    policy = DayPlannerPolicy(include_cows=False, include_chickens=False, include_pasture=False)
    with (patch('harvest.planner.social_calendar.social_phase', return_value=PhaseSpec('SOCIAL_CHURCH','church_court')),
          patch('harvest.planner.power_berry_calendar.due_source', return_value=None)):
        names = [p.phase for p in build_day_phases_from_ram(w.ram, policy=policy)]
    assert names.index('HARVEST_ROUTE') < names.index('SOCIAL_CHURCH')
    assert names[names.index('SOCIAL_CHURCH') + 1:] == ['EXIT_TO_FARM', 'DYNAMIC_OUTDOOR_PLAN']


def test_score_harvests_before_the_pasture_errand(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, 'score')
    w = world(season=1, day=17)
    set_tile(w.ram, 13, 29, 0x52)
    with patch('harvest.planner.day_plan_phases.pasture_day_phases', return_value=[PhaseSpec('CUT_FODDER','grass')]):
        names = [p.phase for p in build_outdoor_day_phases_from_ram(w.ram)]
    assert names.index('HARVEST_ROUTE') < names.index('CUT_FODDER')


# These are local component/composition checks; only a fresh campaign can prove all four quotas.


def _rom_env(pin, label):
    from retro_harness import AuditCapabilities, AuditedEnv
    from harvest.runtime.retro_setup import make_harvest_env

    pins = Path(__file__).resolve().parents[1] / 'custom_integrations/HarvestMoon-Snes'
    if not (pins / f'{pin}.state').exists():
        pytest.skip(f'missing natural pin {pin}')
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all(label))
    env.reset()
    return env


def _run_native(env, task, limit=30000):
    from retro_harness import WorldState, TaskStatus
    from harvest.tasks.nav import make_action

    task.reset(WorldState(0, env.get_ram(), {}, None))
    for frame in range(limit):
        result = task.step(WorldState(frame, env.get_ram(), {}, None))
        if result.status != TaskStatus.RUNNING:
            assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
            return frame
        env.step(result.action.action if result.action else make_action())
    pytest.fail(f'{task.name} did not finish: {result.reason}')


@pytest.mark.rom
def test_extra_west_strip_plots_plant_from_a_natural_year_two_morning(monkeypatch):
    from harvest.core.tile_catalog import get_tile_at
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask, EnsureCropSeedsTask
    from harvest.tasks.buy_seeds import BuyShelfItemTask
    from harvest.tasks.skills import farm_pocket_plant_skill

    if farm_pond.ring_layout_name() != 'legacy':
        pytest.skip('legacy extension geometry')
    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    env = _rom_env('PR_f6final_s1_Y2_Spring_D2', 'harvest.crop_extension')
    try:
        frames = _run_native(env, BuyShelfItemTask(stock_field='turnip_seeds', bags=3))
        frames += _run_native(env, EnsureCarryToolTask(tool_id=2, exit_when_done=True))
        frames += _run_native(env, EnsureCropSeedsTask(seed_type='turnip'))
        for center in farm_pond.SCORE_WEST_STRIP_CENTERS:
            frames += _run_native(env, farm_pocket_plant_skill(seed_type='turnip', center=center,
                                                               ram=env.get_ram(), include_water=False))
            planted = [t for t in farm_pond.plot_crop_tiles(center) if 0x62 <= get_tile_at(env.get_ram(), *t) <= 0x6F]
            assert len(planted) == 8, (center, planted)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print('WEST_STRIP_24_PLANTED', frames, env.audit())
    finally:
        env.close()


@pytest.mark.rom
def test_natural_ending_route_harvest_ships_before_optional_church(monkeypatch):
    from retro_harness import WorldState, TaskStatus
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from harvest.tasks.nav import make_action

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    env = _rom_env('PR_ending12_20261111_s1_Y3_Summer_D17', 'harvest.crop_priority')
    try:
        world = WorldState(0, env.get_ram(), {}, None)
        planner = MultiDayPlannerTask()
        task = planner._build_day_task(world)
        names = [spec.phase for spec in task.phases]
        assert names.index('HARVEST_ROUTE') < names.index('SOCIAL_CHURCH'), names
        before = rs.shipped_totals(world.ram)
        task.reset(world)
        for frame in range(160000):
            result = task.step(WorldState(frame, env.get_ram(), {}, None))
            rows = task.phase_results
            if any(row['phase'] == 'HARVEST_ROUTE' for row in rows):
                break
            assert result.status == TaskStatus.RUNNING, (frame, result.reason)
            env.step(result.action.action if result.action else make_action())
        after = rs.shipped_totals(env.get_ram())
        assert sum(after.values()) > sum(before.values()), (frame, before, after, rows)
        assert not any(row['phase'] == 'SOCIAL_CHURCH' for row in rows)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print('HARVEST_BEFORE_CHURCH', frame, before, after, env.audit())
    finally:
        env.close()


@legacy_rings_only
def test_final_summer_sows_the_short_crop_once_the_big_one_has_its_share(monkeypatch):
    """perfect999b Y3 Summer: corn 11 bags on D1, tomatoes (26 short of 496) never picked, ended 470 for 2 points."""
    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    w = world(season=1, day=2, shipped={'corn': 281, 'tomato': 470})
    set_field(w.ram, 'year', 2)
    for center in farm_pond.active_plot_centers(2)[:11]:
        for x, y in farm_pond.plot_crop_tiles(center):
            set_tile(w.ram, x, y, 0x3A)
    assert resolve_seed_type_from_ram(w.ram) == 'tomato'
    # earlier Summers keep the plain "most room left" pick: the power-on timeline up to Year 3 must not move
    set_field(w.ram, 'year', 1)
    assert resolve_seed_type_from_ram(w.ram) == 'corn'
