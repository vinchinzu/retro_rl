"""ROM replays of the Year 2-3 wedges that ended or cost a run (2026-11-01).

Each state is a local research pin captured from the failing run (see
docs/plan.md, "Years 2-3 repeatability sweep"); a test skips when its pin is absent.
The tasks run through ``sanitize_action``, the filter the planner applies to every action.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harvest.core.ram_catalog import read_ram_value
from harvest.planner.day_plan_status import is_house_tilemap
from harvest.planner.tasks.home_recover import sanitize_action
from harvest.planner.tasks.home_return import ReturnHomeTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

PINS = Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes"


@pytest.mark.rom
@pytest.mark.parametrize("pin,limit", [
    ("LN_handoff_20261112_settle_morning_wedge_Y2_Fall_D20", 70000),
    ("LN_handoff_20261112_refill_loop_Y1_Summer_D14", 48000),
])
def test_handoff_day_reaches_a_settled_morning_with_planner_lineage(pin, limit, monkeypatch):
    from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
    from harvest.planner.campaign_composer import SeasonPinSession
    from harvest.planner.multi_day_planner import MultiDayPlannerTask

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    env = _env(pin)
    try:
        task = MultiDayPlannerTask(target_days=1)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        pins = SeasonPinSession(save=False, power_on=False, source_sha256=None,
            start_key=(0, 0, 0), state_name=pin, save_dir=PINS)
        for frame in range(limit):
            ram = env.get_ram()
            pins.before_step(task, ram, frame, lambda path: None, env)
            result = task.step(WorldState(frame=frame, ram=ram, info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason, task.phase_text)
        ram = env.get_ram()
        assert morning_scene_ready(classify_scene_from_ram(ram), int(read_ram_value(ram, "hour")))
        assert task.floor.rescues == 0
        assert not any("watchdog" in str(row.get("reason", ""))
                       for day in task.day_journal for row in day["phase_results"])
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_outdoor_handoff_hen_returns_to_coop_and_lays_after_feeding():
    from harvest.core.animal_status import egg_available_today
    from harvest.core.animal_probe import stray_hen_px
    from harvest.planner.day_plan import DayPlanTask, PhaseSpec
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_phase_chicken import COOP_CHORES_PHASE, EXIT_COOP_PHASE

    env = _env("LN_handoff_20261112_Y2_Spring_D25")
    try:
        assert stray_hen_px(env.get_ram()) is not None
        task = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE,
            PhaseSpec("RECOVER_FARM_HEN", "hen_delivery"), COOP_CHORES_PHASE,
            EXIT_COOP_PHASE, PhaseSpec("RETURN_HOME", "return_home"),
            PhaseSpec("GO_TO_SLEEP", "sleep", {"earliest_sleep_hour": 0})])
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(20000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
        assert stray_hen_px(env.get_ram()) is None
        assert egg_available_today(env.get_ram())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def _env(pin: str):
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    from harvest.runtime.retro_setup import make_harvest_env

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.y23_wedges"))
    env.reset()
    return env


def _return_home(pin: str, frames: int = 9000) -> tuple[TaskStatus, int, int]:
    env = _env(pin)
    try:
        task = ReturnHomeTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(frames):
            world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
            action = result.action.action if result.action else make_action()
            env.step(sanitize_action(action, env.get_ram()))
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        return result.status, int(read_ram_value(env.get_ram(), "tilemap")), frame
    finally:
        env.close()


@pytest.mark.rom
def test_return_home_jumps_out_of_the_spa_pond_and_reaches_the_house():
    # Y2 Spring D19: stranded at (651,198) in the water after the day floor fired
    # mid-soak; the mountain route shuffled on the 0xF7 edge for 3 x 11k frames.
    status, tilemap, frame = _return_home("LN_y23_spa_pond_d19")
    assert status == TaskStatus.SUCCESS and is_house_tilemap(tilemap), (status, hex(tilemap), frame)
    assert frame < 9000


@pytest.mark.rom
def test_return_home_leaves_the_berry_cave_without_losing_the_berry():
    status, tilemap, frame = _return_home("LN_berries_20261103_cave_Y1_Winter_D8_after")
    assert status == TaskStatus.SUCCESS and is_house_tilemap(tilemap), (status, hex(tilemap), frame)
    assert frame < 9000


@pytest.mark.rom
def test_return_home_dismounts_the_horse_at_the_door_stand():
    # Y2 Spring D19 (jitter run): on the horse at (134,425); the door refuses a rider.
    status, tilemap, frame = _return_home("LN_y23_door_p150", frames=6000)
    assert status == TaskStatus.SUCCESS and is_house_tilemap(tilemap), (status, hex(tilemap), frame)


@pytest.mark.rom
def test_a_blind_up_charge_never_enters_the_well_pit():
    # Y2 Summer D4: "up" from (805,930) walked onto the well hole (50,55), map 0x2A.
    env = _env("LN_y23_m2a_pre")
    try:
        for _ in range(240):
            env.step(sanitize_action(make_action(up=True, b=True), env.get_ram()))
        assert int(read_ram_value(env.get_ram(), "tilemap")) == 0x00
        pos = get_pos_from_ram(env.get_ram())
        assert pos.y >= 896, pos  # held on the south side of the pit tile row
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()



@pytest.mark.rom
def test_coop_chores_never_leave_the_coop_carrying_a_hen():
    # Y2 Spring D28-D29: a hen lifted at the feed bin stayed in the hands (every egg
    # "held_item_block"), EXIT_COOP carried her onto the farm, and `stray_hen_px` sent
    # the next day to the shop yard for a hen that was not there.
    from harvest.core.animal_status import read_item_on_hand
    from harvest.tasks.coop_task import CoopChoresTask

    env = _env("LN_y23_coop_holding_hen")
    try:
        assert int(read_item_on_hand(env.get_ram())) == 0x25
        task = CoopChoresTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(7000):
            world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (result.status, result.reason, frame)
        assert int(read_item_on_hand(env.get_ram())) != 0x25, "still carrying the hen"
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_return_home_walks_a_tired_farmer_home_on_a_festival_eve():
    # Y2 Fall D19: town (744,424), player_state 0x09 (tired), the eve of the Harvest
    # Festival. The scene reader called it a cutscene, so return_home mashed A/B
    # 241 frames at a time for 8 rescues.
    status, tilemap, frame = _return_home("LN_y23_festeve_carry", frames=20000)
    assert status == TaskStatus.SUCCESS and is_house_tilemap(tilemap), (status, hex(tilemap), frame)


@pytest.mark.rom
def test_barn_exit_steps_around_a_calf_parked_in_the_doorway():
    # Y2 Spring D30: a calf (sprite 0x19f) at (150,361) overlaps the door lane; the push
    # down at x=139 stood still for the 1,800-frame timeout and cost a rescue.
    from harvest.planner.tasks.transitions import DirectionalTransitionTask

    env = _env("LN_y23_barn_exit_fail")
    try:
        task = DirectionalTransitionTask(
            name="exit_barn", direction="down", origin_tilemap=0x27, target_tilemap=0x00,
            timeout=1800, stand_tile=(8, 22), stand_tolerance=1, door_align_px=136, settle_frames=5,
        )
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(1800):
            world = WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, (result.status, result.reason, frame)
        assert frame < 600 and int(read_ram_value(env.get_ram(), "tilemap")) == 0x00
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_a_walk_up_the_south_side_never_enters_the_second_well_hole():
    # Y2 Summer D16: down the east side, west along y=857, then up at x=598 entered the
    # hole at (37,48), tile 0xC8, map 0x2A; 8 rescues ended the run.
    env = _env("LN_y23_m2b_approach")
    try:
        for action in [make_action(left=True, b=True)] * 23 + [make_action(up=True, b=True)] * 240:
            env.step(sanitize_action(action, env.get_ram()))
        assert int(read_ram_value(env.get_ram(), "tilemap")) == 0x00
        assert get_pos_from_ram(env.get_ram()).y >= 784, get_pos_from_ram(env.get_ram())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_the_5pm_shipping_scene_clears_for_a_rider_through_the_action_filters():
    # Y3 Spring D20: on the horse at (255,362) when the 17:00 shipping box opened. The
    # dismount filter held A every frame, so the scene never saw its A edge and the
    # clock stood at 17:00 for 1.6M frames. The scene press must reach the pad as pulses.
    from harvest.core.scene import SceneGuard, unowned_scene_press
    from harvest.tasks.primitives import dismiss_dialogue_result

    env = _env("LN_y3a_frozen17")
    try:
        guard = SceneGuard()
        start_minute = int(read_ram_value(env.get_ram(), "minute"))
        assert int(read_ram_value(env.get_ram(), "hour")) == 17 and start_minute == 0
        for step in range(1500):
            ram = env.get_ram()
            choice = unowned_scene_press(ram, guard, step)
            if choice is not None:
                action = dismiss_dialogue_result(step, buttons=choice[0], pulse_every=2, reason=choice[1]).action.action
            else:
                action = make_action(right=True)  # a nav that never stops pressing
            env.step(sanitize_action(action, ram))
            if int(read_ram_value(env.get_ram(), "minute")) != start_minute:
                break
        assert int(read_ram_value(env.get_ram(), "minute")) != start_minute, "the 17:00 clock never restarted"
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_the_5pm_shipping_scene_does_not_leave_the_0x0800_flag_that_kills_the_bed():
    # Y2 Spring D10: the planner pulsed A while input_lock was 0 (the scene still starting).
    # That set game_state & 0x0800 for the rest of the day; the bed ignored every press and
    # eight sleep rescues ended the run. A pulses only once the box is open.
    from harvest.core.ram_catalog import read_ram_value as rv
    from harvest.core.scene import SceneGuard, unowned_scene_press
    from harvest.tasks.primitives import dismiss_dialogue_result

    env = _env("LN_y23_dogrise_d10")
    try:
        guard = SceneGuard()
        assert int(rv(env.get_ram(), "game_state", raw=True)) & 0x0800 == 0
        for step in range(900):
            ram = env.get_ram()
            choice = unowned_scene_press(ram, guard, step)
            action = dismiss_dialogue_result(step, buttons=choice[0], pulse_every=2, reason=choice[1]).action.action if choice else make_action()
            env.step(sanitize_action(action, ram))
            assert int(rv(env.get_ram(), "game_state", raw=True)) & 0x0800 == 0, f"0x0800 set at step {step}"
        assert int(rv(env.get_ram(), "minute")) >= 1 or int(rv(env.get_ram(), "hour")) > 17, "the scene never finished"
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
