"""Regressions from native hawker and summit berry traces."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from retro_harness import TaskStatus, WorldState
from harvest.core.ram_catalog import field_spec
from harvest.tasks import power_berry as pb


def world(**values):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[0xD3] = 0x40  # player control
    for key, value in values.items():
        spec = field_spec(key)
        raw = spec.to_storage(value)
        for i in range(spec.width):
            ram[spec.address + i] = (raw >> (8*i)) & 255
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def test_hawker_berry_23_pixels_below_is_approached_before_pressing():
    w = world(power_berry_count=6, max_stamina=160, input_lock=1, player_direction=0)
    task = pb.PowerBerryPickupTask()
    task.reset(w)
    with patch.object(pb, "get_pos_from_ram", return_value=SimpleNamespace(x=70, y=482)), \
         patch.object(pb, "berry_objects", return_value=[SimpleNamespace(pixel=(72, 505))]):
        result = task.step(w)
    assert result.status == TaskStatus.RUNNING
    assert np.array_equal(result.action.action, pb.make_action(down=True))


def test_snow_flower_uses_the_live_reward_position_after_planting():
    w = world(season=3, hour=12, input_lock=1)
    task = pb.PowerBerryTask(source="snow_flower")
    task.reset(w)
    task._phase, task._steps, task._phase_start = "plant", 901, 0
    task.step(w)
    assert isinstance(task._child, pb.PowerBerryPickupTask)
    assert task._child.stand is None


def test_ground_pickup_pushes_into_the_sprite_before_the_next_press():
    w = world(input_lock=1, player_direction=0)
    task = pb.PowerBerryPickupTask()
    task.reset(w)
    task._wait = 6
    with patch.object(pb, "get_pos_from_ram", return_value=SimpleNamespace(x=70, y=492)), \
         patch.object(pb, "berry_objects", return_value=[SimpleNamespace(pixel=(72, 505))]):
        result = task.step(w)
    assert np.array_equal(result.action.action, pb.make_action(down=True))


def test_berry_eat_menu_does_not_confirm_a_previous_no_selection():
    w = world(input_lock=2, held_item=8, dialog_choice_result=1)
    w.ram[0x19B] = 0x10
    task = pb.PowerBerryPickupTask()
    task.reset(w)
    assert np.array_equal(task.step(w).action.action, pb.make_action(up=True))


def test_egg_hunt_accepts_the_same_colour_on_consecutive_requests():
    w = world(tilemap=0x0B, input_lock=2)
    task = pb.EggHuntTask()
    task.reset(w)
    task._phase, task._target, task._score = "give", 3, 0
    w.ram[0x09A2] = 1  # mayor accepted this egg
    task.step(w)
    assert task._phase == "question"
    task.step(w)  # next question's dialogue
    w.ram[field_spec("input_lock").address] = 1
    task.step(w)  # box closed; the old colour is still in RAM
    assert task._phase == "question_ready"
    w.ram[0x09AC] = w.ram[0x09A5] = 3  # a repeated colour, now a new request
    with patch.object(task, "_walk"):
        task.step(w)
    assert task._phase == "pick" and task._target == 3


def test_second_egg_win_does_not_chase_last_years_eaten_house_berry():
    from unittest.mock import Mock
    from retro_harness import TaskResult
    w = world(tilemap=0x17)
    w.ram[0x11F6D] = 0x70  # house arm, eaten, and this year's turtle
    task = pb.PowerBerryTask(source="egg_festival")
    task.reset(w)
    task._phase = "hunt"
    task._child = Mock(step=lambda w: TaskResult(TaskStatus.SUCCESS))
    assert task.step(w).status == TaskStatus.SUCCESS
    assert task._child is None


def test_ten_berry_owner_can_play_the_second_hunt_for_the_turtle():
    from unittest.mock import Mock
    from retro_harness import TaskResult
    w = world(tilemap=0x17, power_berry_count=10, max_stamina=200)
    w.ram[0x11F6D] = 0x60  # first prize armed and eaten; turtle still missing
    task = pb.PowerBerryTask(source="egg_festival")
    task.reset(w)
    assert task.can_start(w)
    task._phase = "hunt"
    task._child = Mock(step=lambda w: TaskResult(TaskStatus.SUCCESS))
    w.ram[0x11F6D] |= 0x10  # native second-prize receipt
    assert task.step(w).status == TaskStatus.SUCCESS
    assert task._child is None


def test_old_egg_prize_does_not_make_a_lost_hunt_look_like_a_win():
    w = world(tilemap=0x0B)
    w.ram[0x11F6D] = 0x10
    task = pb.EggHuntTask()
    task.reset(w)
    task._phase = "award"
    w.ram[field_spec("tilemap").address] = 0x17
    w.ram[0x25] = 15
    assert task.step(w).status == TaskStatus.FAILURE


def test_dated_berry_trip_defers_the_optional_feather_fetch():
    from harvest.tasks.proposal import apply_eve_phases
    from harvest.planner.day_phase_types import PhaseSpec
    phases = [PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
              PhaseSpec("COLLECT_POWER_BERRY_HARVEST_FESTIVAL", "power_berry", {"source": "harvest_festival"})]
    feather = PhaseSpec("FETCH_FEATHER", "ensure_tool", {"tool_id": 0x18})
    with patch("harvest.tasks.proposal.eve_phases_for_ram", return_value=[feather]), \
         patch("harvest.planner.tasks.home.apply_house_phases", side_effect=lambda p, r: p):
        assert apply_eve_phases(phases, world().ram) == phases


def test_festival_water_deadline_does_not_take_a_spa_detour():
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.day_phase_types import PhaseSpec
    spec = PhaseSpec("CROP_WATER", "crop", {"stop_hour": 12}, failure_policy="optional")
    w = world(hour=12)
    task = DayPlanTask(phase_sequence=[spec])
    task.reset(w)
    assert not task._try_stamina_spa_refresh(spec, w, result_reason="stamina_low unfunded=40")


def test_hen_pickup_retargets_when_a_hen_blocks_centering():
    from harvest.planner.tasks import chicken_sale as sale
    w = world(tilemap=0x28, input_lock=1)
    task = sale.CoopPickupChickenTask()
    task.reset(w)
    task._target = ((11, 3), "right", (12, 3))
    task._navigator.stasis = 100
    with patch.object(task._navigator, "update"), \
         patch.object(sale, "chicken_stage_tiles", return_value=[(12, 3)]), \
         patch.object(sale, "select_adjacent_pickup_target", return_value=None) as choose:
        task.step(w)
    choose.assert_called_once()
    assert (11, 3) in task._pathfinder.temp_blocked


def test_unregistered_hen_pickup_retires_that_stand():
    from harvest.planner.tasks.chicken_sale import CoopPickupChickenTask
    w = world(tilemap=0x28, input_lock=1)
    task = CoopPickupChickenTask()
    task.reset(w)
    task._target = ((11, 3), "right", (12, 3))
    task._phase, task._verify_count, task._attempts = "verify", 24, 1
    assert task.step(w).status == TaskStatus.RUNNING
    assert (11, 3) in task._pathfinder.temp_blocked
    assert task._phase == "nav"


def _drive(env, task, limit=60000):
    task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
    for frame in range(limit):
        result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
        if result.status != TaskStatus.RUNNING:
            assert result.status == TaskStatus.SUCCESS, result.reason
            return frame
        env.step(result.action.action if result.action else pb.make_action())
    pytest.fail(f"{task.name} did not finish: {result.reason}")


@pytest.mark.rom
def test_snow_flower_is_eaten_and_kept_through_the_return_and_night():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask
    from harvest.planner.tasks.home_sleep import GoToSleepTask
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.power_berry_calendar import snow_seed_owned
    from retro_harness import AuditedEnv, AuditCapabilities
    env = AuditedEnv(make_harvest_env("LN_berries_snow_Winter_D1", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.snow_flower"))
    try:
        env.reset()
        before = int(read_ram_value(env.get_ram(), "power_berry_count"))
        maximum = int(read_ram_value(env.get_ram(), "max_stamina"))
        _drive(env, pb.PowerBerryTask(source="snow_flower"))
        _drive(env, ExitToFarmTask())
        _drive(env, GoToSleepTask())
        assert read_ram_value(env.get_ram(), "power_berry_count") == before + 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        assert not snow_seed_owned(env.get_ram())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_flower_festival_buys_the_seed_once_and_keeps_it_overnight(monkeypatch):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner import power_berry_calendar as calendar
    from harvest.core.ram_catalog import read_ram_u16
    from retro_harness import AuditedEnv, AuditCapabilities
    monkeypatch.setenv(calendar.ENV_VAR, "1")
    env = AuditedEnv(make_harvest_env("PR_y1new_Y1_Spring_D23", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.snow_seed"))
    try:
        env.reset()
        assert calendar.snow_seed_due(env.get_ram())
        phases = build_day_phases_from_ram(env.get_ram())
        at = next(i for i, p in enumerate(phases) if p.phase == "ATTEND_FESTIVAL")
        assert phases[at].params["buy_snow_seed"]
        _drive(env, DayPlanTask(phase_sequence=phases[:at+1]), limit=100000)
        assert read_ram_u16(env.get_ram(), calendar.SNOW_SEED_BOUGHT_ADDR) & calendar.SNOW_SEED_BOUGHT_BIT
        assert calendar.snow_seed_owned(env.get_ram())
        assert not calendar.snow_seed_due(env.get_ram())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("prior,berries", [(0x1000, None), (0x6000, None), (0x6000, 10)])
def test_second_egg_win_collects_the_other_prize_from_a_declared_fixture(prior, berries):
    """Research fixture for last year's prize; the hunt itself uses native inputs."""
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_u16, read_ram_value
    from retro_harness import AuditedEnv, AuditCapabilities
    env = AuditedEnv(make_harvest_env("PR_e2ec1_Y2_Fall_D20", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.second_egg_prize"))
    try:
        env.reset()
        for offset, byte in enumerate((read_ram_u16(env.get_ram(), 0x11F6C) | prior).to_bytes(2, "little")):
            env.data.memory.assign(0x7F1F6C + offset, "|u1", byte)
        env.record_ram_write(2)
        if berries is not None:
            for name, value in (("power_berry_count", berries), ("max_stamina", 100 + berries * 10)):
                spec = field_spec(name)
                for offset, byte in enumerate(spec.to_storage(value).to_bytes(spec.width, "little")):
                    env.data.memory.assign(0x7E0000 + spec.address + offset, "|u1", byte)
                env.record_ram_write(spec.width)
        env.begin_attempt()
        before = int(read_ram_value(env.get_ram(), "power_berry_count"))
        from harvest.planner.power_berry_calendar import due_source

        assert due_source(env.get_ram(), force=True) == "egg_festival"
        _drive(env, pb.PowerBerryTask(source="egg_festival"))
        assert read_ram_u16(env.get_ram(), 0x11F6C) & 0x7000 == 0x7000
        assert read_ram_value(env.get_ram(), "power_berry_count") - before == (1 if prior == 0x1000 else 0)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
