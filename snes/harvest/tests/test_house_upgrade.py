"""First-house prerequisites and native lumber/request receipts."""

import json
from pathlib import Path

import pytest

from harvest.core.ram_catalog import read_ram_value
from harvest.planner.tasks.home import GatherHouseWoodTask, HouseUpgradeTask
from harvest.tasks.nav import make_action
from retro_harness import TaskResult, TaskStatus, WorldState


def save_native_state(env, name):
    import gzip
    from harvest.paths import GAME_DIR
    path = GAME_DIR / f"{name}.state"
    path.write_bytes(gzip.compress(env.unwrapped.em.get_state(), mtime=0))


def house_ram(**fields):
    import numpy as np
    from harvest.core.ram_catalog import field_spec
    ram = np.zeros(0x24000, dtype=np.uint8)
    defaults = dict(season=1, day=9, hour=6, weekday=4, eve_hearts=250, stamina=100,
                    tilemap=0, input_lock=1, money=6000, stored_wood=250)
    for key, value in (defaults | fields).items():
        spec = field_spec(key)
        address = spec.address + (0x4000 if spec.live_offset else 0)
        raw = value // 10 if key == "money" else value
        ram[address] = raw & 255
        if key not in ("season", "day", "hour", "weekday", "stamina", "tilemap", "input_lock"):
            ram[address + 1] = (raw >> 8) & 255
    return ram


def test_funded_quote_waits_until_a_cow_exists():
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], house_ram()) == []


@pytest.mark.parametrize("season", [1, 2, 3])
def test_ready_house_plan_builds_a_native_task_after_the_cow(season):
    from harvest.planner.tasks.home import apply_house_phases
    from harvest.planner.day_phase_registry import build_phase_task, TaskBuildContext
    from harvest.planner.day_phase_types import PhaseSpec
    ram = house_ram(season=season, num_cows=2)
    phases = apply_house_phases([PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")], ram)
    assert phases[0].phase == "HOUSE_UPGRADE"
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    assert isinstance(build_phase_task(TaskBuildContext(), phases[0], world), HouseUpgradeTask)


def test_missing_wood_builds_a_native_lumber_task():
    from harvest.planner.tasks.home import apply_house_phases
    from harvest.planner.day_phase_registry import build_phase_task, TaskBuildContext
    ram = house_ram(stored_wood=0, num_cows=2)
    phase = apply_house_phases([], ram)[0]
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    assert isinstance(build_phase_task(TaskBuildContext(), phase, world), GatherHouseWoodTask)


@pytest.mark.parametrize("fields", [dict(money=5990), dict(upgrade_flags=0x40),
                                   dict(marriage_flags=0x80), dict(weekday=0), dict(weekday=6)])
def test_house_waits_for_reserve_completion_or_cabin_hours(fields):
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], house_ram(num_cows=1, **fields)) == []


def test_spring_leaves_the_first_house_for_crops():
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], house_ram(season=0, num_cows=1)) == []


def test_later_year_spring_can_still_order_the_first_house():
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], house_ram(year=1, season=0, num_cows=1)) == []
    phases = apply_house_phases([], house_ram(year=1, season=0, num_cows=2))
    assert phases[0].phase == "HOUSE_UPGRADE"


def married_ram(**fields):
    base = dict(marriage_flags=0x10, upgrade_flags=0x41, eve_hearts=234,
                stored_wood=2, money=680, season=2, day=18)
    return house_ram(**(base | fields))


def test_married_short_wood_gathers_toward_500():
    from harvest.planner.tasks.home import apply_house_phases
    from harvest.planner.day_phase_registry import build_phase_task, TaskBuildContext
    ram = married_ram()
    phases = apply_house_phases([], ram)
    assert [phase.phase for phase in phases] == ["HOUSE_WOOD"]
    assert phases[0].params["target"] == 500
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    task = build_phase_task(TaskBuildContext(), phases[0], world)
    assert isinstance(task, GatherHouseWoodTask)
    task.reset(world)
    assert task.target == 500


def test_married_winter_still_gathers_house_two_wood():
    from harvest.planner.tasks.home import apply_house_phases
    phases = apply_house_phases([], married_ram(season=3, day=5))
    assert [phase.phase for phase in phases] == ["HOUSE_WOOD"]
    assert phases[0].params["target"] == 500


def test_married_second_year_still_gathers_house_two_wood():
    from harvest.planner.tasks.home import apply_house_phases
    phases = apply_house_phases([], married_ram(year=1, season=0, day=3))
    assert [phase.phase for phase in phases] == ["HOUSE_WOOD"]


def test_married_weekday_morning_orders_the_second_house():
    from harvest.planner.day_phase_registry import build_phase_task, TaskBuildContext
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.tasks.home import apply_house_phases
    ram = married_ram(stored_wood=500, money=10000, weekday=3, hour=6, num_cows=1)
    phases = apply_house_phases([PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")], ram)
    assert phases[0].phase == "HOUSE_UPGRADE"
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    task = build_phase_task(TaskBuildContext(), phases[0], world)
    assert isinstance(task, HouseUpgradeTask)
    assert task.goal == "house2"


def test_married_short_of_10000_leaves_the_day_for_crops():
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], married_ram(stored_wood=500, money=9990, num_cows=1)) == []


def test_second_house_payment_waits_for_a_cow():
    from harvest.planner.tasks.home import apply_house_phases
    ram = married_ram(stored_wood=500, money=10000, weekday=3, hour=6)
    assert apply_house_phases([], ram) == []


@pytest.mark.parametrize("weekday", [0, 6])
def test_married_weekend_does_not_order_house_two(weekday):
    from harvest.planner.tasks.home import apply_house_phases
    ram = married_ram(stored_wood=500, money=10000, weekday=weekday, hour=6, num_cows=1)
    assert apply_house_phases([], ram) == []


def test_married_pay_morning_still_prefixes_the_order():
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.tasks.home import apply_house_phases
    ram = married_ram(marriage_flags=0x50, stored_wood=2, money=680)
    phases = apply_house_phases([PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")], ram)
    assert [phase.phase for phase in phases[:2]] == ["HOUSE_UPGRADE", "DYNAMIC_OUTDOOR_PLAN"]
    assert phases[0].params["goal"] == "house2"


def test_married_accepted_order_does_not_schedule_more_lumber():
    from harvest.planner.tasks.home import apply_house_phases
    assert apply_house_phases([], married_ram(marriage_flags=0x90)) == []


def test_tired_married_farmer_does_not_block_the_day_on_wood():
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.tasks.home import apply_house_phases
    crop = PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")
    assert apply_house_phases([crop], married_ram(stamina=10)) == [crop]


def test_default_lumber_target_stays_one_farm_session():
    task = GatherHouseWoodTask()
    start = house_ram(stored_wood=0)
    task.reset(WorldState(frame=0, ram=start, info={}, obs=None))
    assert task.target == 250
    gained = house_ram(stored_wood=24)
    result = task.step(WorldState(frame=1, ram=gained, info={}, obs=None))
    assert result.status == TaskStatus.SUCCESS
    assert result.meta["end_wood"] - result.meta["start_wood"] == 24


def test_house_two_lumber_succeeds_on_one_session_not_on_a_single_log_indoors():
    task = GatherHouseWoodTask(target=500)
    start = married_ram(stored_wood=2)
    task.reset(WorldState(frame=0, ram=start, info={}, obs=None))
    assert task.target == 500

    class _Hold:
        def step(self, world):
            return TaskResult(TaskStatus.RUNNING, reason="holding")

    task._child = _Hold()
    early = task.step(WorldState(frame=1, ram=married_ram(stored_wood=8), info={}, obs=None))
    assert early.status == TaskStatus.RUNNING
    assert early.reason == "holding"
    task._phase = "chop"
    log = task.step(WorldState(frame=2, ram=married_ram(stored_wood=8), info={}, obs=None))
    assert log.status == TaskStatus.SUCCESS
    assert log.meta["end_wood"] - log.meta["start_wood"] == 6
    task._phase = "axe"
    task._start = 2
    done = task.step(WorldState(frame=3, ram=married_ram(stored_wood=26), info={}, obs=None))
    assert done.status == TaskStatus.SUCCESS
    assert done.meta["end_wood"] - done.meta["start_wood"] == 24


def test_finished_first_house_still_closes_the_order_task():
    task = HouseUpgradeTask()
    world = WorldState(frame=0, ram=house_ram(upgrade_flags=0x40), info={}, obs=None)
    task.reset(world)
    result = task.step(world)
    assert result.status == TaskStatus.SUCCESS
    assert result.reason == "first house complete"


def test_second_order_does_not_treat_the_first_house_as_done():
    task = HouseUpgradeTask(goal="house2")
    world = WorldState(frame=0, ram=married_ram(stored_wood=500, money=10000), info={}, obs=None)
    task.reset(world)
    result = task.step(world)
    assert result.status == TaskStatus.RUNNING
    assert result.reason != "first house complete"


def _paid_world(start_money, start_wood, money, wood, flags, upgrades):
    start = house_ram(money=start_money, stored_wood=start_wood, marriage_flags=0x10,
                      upgrade_flags=upgrades, eve_hearts=234, season=2)
    paid = house_ram(money=money, stored_wood=wood, marriage_flags=flags,
                     upgrade_flags=upgrades, eve_hearts=234, season=2)
    return start, paid


def test_second_order_accepts_only_the_10000_debit():
    task = HouseUpgradeTask(goal="house2")
    start, paid = _paid_world(20000, 500, 10000, 0, 0x90, 0x41)
    task.reset(WorldState(frame=0, ram=start, info={}, obs=None))
    result = task.step(WorldState(frame=1, ram=paid, info={}, obs=None))
    assert result.status == TaskStatus.SUCCESS
    assert result.meta["money_spent"] == 10000
    assert result.meta["wood_spent"] == 500


def test_second_order_rejects_the_first_house_debit():
    task = HouseUpgradeTask(goal="house2")
    start, paid = _paid_world(20000, 500, 15000, 250, 0x90, 0x41)
    task.reset(WorldState(frame=0, ram=start, info={}, obs=None))
    result = task.step(WorldState(frame=1, ram=paid, info={}, obs=None))
    assert result.status == TaskStatus.FAILURE
    assert result.reason == "unverified carpenter debit 5000G/250 wood"


def test_first_order_still_requires_the_5000_debit():
    task = HouseUpgradeTask()
    start = house_ram(money=10000, stored_wood=250)
    paid = house_ram(money=5000, stored_wood=0, marriage_flags=0x80)
    task.reset(WorldState(frame=0, ram=start, info={}, obs=None))
    result = task.step(WorldState(frame=1, ram=paid, info={}, obs=None))
    assert result.status == TaskStatus.SUCCESS
    assert result.reason == "house ordered: 5000G and 250 wood"
    assert result.meta["money_spent"] == 5000
    assert result.meta["wood_spent"] == 250


def test_empty_ranch_plants_pasture_and_keeps_the_seed_and_hen_funds():
    from harvest.planner.day_phase_catalog import pasture_day_phases
    phases = pasture_day_phases(house_ram(day=8, stored_wood=0, grass_seeds=1))
    assert [phase.phase for phase in phases] == ["BUY_GRASS_SEEDS", "PLANT_PASTURE"]
    # 6000G holds two corn bags (600G), the hen (1000G) and the 300G floor,
    # which leaves the three 500G bags that fill 32 tiles (stock 1+3).
    assert phases[0].params["target_stock"] == 4


def test_saturday_fetches_shelved_feather_and_proposes_before_crop_tools_replace_it():
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    ram = house_ram(tilemap=0x15, weekday=6, upgrade_flags=0x40)
    ram[0x15F02] = 0x80
    phases = build_day_phases_from_ram(ram)
    assert [phase.phase for phase in phases] == ["EXIT_TO_FARM", "FETCH_FEATHER", "PROPOSE_EVE"]


def test_wedding_keeps_the_date_roll_until_its_house_transition_finishes():
    from types import SimpleNamespace
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from retro_harness import TaskResult
    task = MultiDayPlannerTask()
    task._phase, task._active_day = "plan_day", (2, 9)
    task._current_task = SimpleNamespace(phases=[PhaseSpec("PROPOSE_EVE", "propose_eve", {"ends_day": True})])
    world = WorldState(frame=0, ram=house_ram(tilemap=0x04, season=2, day=10), info={}, obs=None)
    running = TaskResult(TaskStatus.RUNNING)
    assert task._observe_running_side_effects(world, running) is running
    assert task._days_completed == 0


def test_completed_wedding_records_its_overnight_without_an_extra_return_route():
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from retro_harness import TaskResult
    task = MultiDayPlannerTask()
    task._phase, task._active_day = "plan_day", (2, 9)
    world = WorldState(frame=0, ram=house_ram(tilemap=0x16, season=2, day=10), info={}, obs=None)
    task._handle_result(world, TaskResult(TaskStatus.SUCCESS))
    assert task._days_completed == 1 and task._phase == "settle_morning"


def test_remodeled_bed_route_does_not_cache_leaked_outdoor_coordinates():
    from harvest.planner.tasks.home import GoToSleepTask
    from harvest.tasks.nav import Point
    route = GoToSleepTask._route_for_position(Point(136, 331), 0x16)
    assert all(point.y <= 224 for point, *_ in route)


def drive(env, task, budget):
    task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
    for frame in range(budget):
        result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
        if result.status != TaskStatus.RUNNING:
            if result.status != TaskStatus.SUCCESS:
                from PIL import Image
                from harvest.paths import GAME_DIR, PROJECT_DIR
                import gzip
                import numpy as np
                Image.fromarray(env.render()).save(PROJECT_DIR / "recordings/house_failure.png")
                np.save(PROJECT_DIR / "recordings/house_failure_ram.npy", env.get_ram())
                with gzip.open(GAME_DIR / "LN_house_request_failure.state", "wb") as handle:
                    handle.write(env.unwrapped.em.get_state())
            assert result.status == TaskStatus.SUCCESS, result.reason
            return frame, result
        env.step(result.action.action if result.action else make_action())
    pytest.fail(f"{task.name} exceeded {budget} frames")


@pytest.mark.rom
def test_natural_summer_predecessor_collects_lumber():
    from harvest.runtime.retro_setup import make_harvest_env
    env = make_harvest_env("LN_eve_chain_summer", lean_step=True)
    try:
        env.reset()
        before = {key: read_ram_value(env.get_ram(), key) for key in
                  ("stored_wood", "money", "season", "day", "upgrade_flags")}
        assert before["stored_wood"] == 0
        wood = GatherHouseWoodTask()
        wood_frames, _ = drive(env, wood, 22000)
        assert wood.receipt["end_wood"] - before["stored_wood"] == 24
        after = {key: read_ram_value(env.get_ram(), key) for key in
                 ("stored_wood", "money", "season", "day", "marriage_flags")}
        assert after["money"] == before["money"]
        assert after["stored_wood"] == 24
        report = {"source_state": "LN_eve_chain_summer", "start": before,
                  "end": after, "wood_frames": wood_frames,
                  "wood_receipt": wood.receipt, "ram_writes": 0, "mid_run_state_loads": 0}
        out = Path(__file__).resolve().parents[1] / "recordings/house_prerequisites.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


@pytest.mark.rom
def test_natural_summer_predecessor_requests_carpenter():
    from harvest.runtime.retro_setup import make_harvest_env
    env = make_harvest_env("LN_eve_chain_summer", lean_step=True)
    try:
        env.reset()
        request = HouseUpgradeTask(goal="clock_arm")
        drive(env, request, 22000)
        assert read_ram_value(env.get_ram(), "marriage_flags") & 0x20
    finally:
        env.close()


@pytest.mark.rom
def test_natural_summer_predecessor_establishes_income_crop():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.inventory import EnsureCropSeedsTask, ExitToFarmTask
    from harvest.tasks.skills import farm_pocket_plant_skill
    from harvest.maps.farm_pond import next_unplanted_pocket_center
    env = make_harvest_env("LN_eve_chain_summer", lean_step=True)
    try:
        env.reset()
        drive(env, ExitToFarmTask(), 4000)
        drive(env, EnsureCropSeedsTask(seed_type="corn"), 15000)
        center = next_unplanted_pocket_center(env.get_ram())
        drive(env, farm_pocket_plant_skill(seed_type="corn", center=center, ram=env.get_ram()), 18000)
    finally:
        env.close()


@pytest.mark.rom
def test_natural_paid_order_takes_four_nights():
    import hashlib
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.home import GoToSleepTask
    from retro_harness import AuditedEnv, AuditCapabilities

    source = GAME_DIR / "LN_house_paid_natural.state"
    if not source.is_file():
        pytest.skip("Run the natural house campaign through its paid order first")
    env = AuditedEnv(make_harvest_env(source.stem, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.house_upgrade_test"))
    fields = ("season", "day", "money", "stored_wood", "upgrade_flags",
              "marriage_flags", "development_rate")
    try:
        env.reset()
        snapshots = [{key: read_ram_value(env.get_ram(), key) for key in fields}]
        assert snapshots[0]["marriage_flags"] & 0x80
        assert snapshots[0]["development_rate"] == 0
        assert not snapshots[0]["upgrade_flags"] & 0x40
        for night in range(1, 5):
            drive(env, GoToSleepTask(earliest_sleep_hour=0), 18000)
            snap = {key: read_ram_value(env.get_ram(), key) for key in fields}
            snapshots.append(snap)
            assert snap["development_rate"] == night
            assert bool(snap["upgrade_flags"] & 0x40) == (night == 4)
        assert not snapshots[-1]["marriage_flags"] & 0x80
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        save_native_state(env, "LN_house_built_natural")
        report = {"source_state": source.name,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "nights": snapshots, "ram_writes": env.audit().ram_writes,
                  "mid_run_state_loads": env.audit().mid_run_loads}
        (PROJECT_DIR / "recordings/house_build_nights.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


@pytest.mark.rom
def test_natural_upgraded_house_buys_the_sunday_feather():
    import hashlib
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.home import GoToSleepTask
    from harvest.tasks.proposal import FeatherBuyTask
    from harvest.planner.tasks.inventory import ExitToFarmTask
    if not (GAME_DIR / "LN_house_built_natural.state").is_file():
        pytest.skip("Run the native four-night construction test first")
    from retro_harness import AuditedEnv, AuditCapabilities
    env = AuditedEnv(make_harvest_env("LN_house_built_natural", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.feather_test"))
    try:
        env.reset()
        drive(env, ExitToFarmTask(), 6000)
        drive(env, GoToSleepTask(earliest_sleep_hour=0), 18000)
        assert read_ram_value(env.get_ram(), "weekday") == 0
        task = FeatherBuyTask()
        drive(env, task, 18000)
        assert task.purchase_receipt["wallet_delta"] == -1000
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        save_native_state(env, "LN_house_feather_natural")
        receipt = dict(task.purchase_receipt, ram_writes=env.audit().ram_writes,
                       mid_run_state_loads=env.audit().mid_run_loads,
                       source_state="LN_house_built_natural",
                       source_sha256=hashlib.sha256((GAME_DIR / "LN_house_built_natural.state").read_bytes()).hexdigest())
        (PROJECT_DIR / "recordings/house_feather.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        env.close()


@pytest.mark.rom
def test_natural_feather_reaches_eves_wedding():
    import hashlib
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.home import GoToSleepTask
    from harvest.planner.tasks.inventory import EnsureCarryToolTask
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    from harvest.tasks.eve_proposal import proposal_blockers
    from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
    from retro_harness import AuditedEnv, AuditCapabilities
    source = GAME_DIR / "LN_house_feather_natural.state"
    if not source.is_file():
        pytest.skip("Run the natural Sunday feather test first")
    env = AuditedEnv(make_harvest_env(source.stem, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.wedding_test"))
    try:
        env.reset()
        drive(env, EnsureCarryToolTask(tool_id=0x18), 16000)
        for night in range(21):
            if proposal_blockers(env.get_ram()) is None:
                break
            drive(env, GoToSleepTask(earliest_sleep_hour=0), 24000)
        assert proposal_blockers(env.get_ram()) is None
        before = {key: read_ram_value(env.get_ram(), key) for key in
                  ("season", "day", "weekday", "money", "eve_hearts", "marriage_flags")}
        phases = build_day_phases_from_ram(env.get_ram())
        assert phases[-1].phase == "PROPOSE_EVE"
        frames, _ = drive(env, DayPlanTask(phase_sequence=phases), 70000)
        after = {key: read_ram_value(env.get_ram(), key) for key in
                 ("season", "day", "hour", "money", "eve_hearts", "marriage_flags", "upgrade_flags")}
        assert after["marriage_flags"] & 0x10
        assert after["upgrade_flags"] & 0x40
        assert after["day"] != before["day"]
        assert morning_scene_ready(classify_scene_from_ram(env.get_ram()), after["hour"])
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        save_native_state(env, "LN_house_wedding_natural")
        mornings = []
        for _ in range(2):
            drive(env, GoToSleepTask(earliest_sleep_hour=0), 18000)
            mornings.append({key: read_ram_value(env.get_ram(), key) for key in
                             ("day", "hour", "marriage_flags")})
            assert mornings[-1]["marriage_flags"] & 0x10
        save_native_state(env, "LN_house_married_next_morning")
        report = {"source_state": source.name,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "before": before, "after": after, "proposal_frames": frames,
                  "post_wedding_mornings": mornings,
                  "ram_writes": env.audit().ram_writes, "mid_run_state_loads": env.audit().mid_run_loads}
        (PROJECT_DIR / "recordings/house_wedding.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


@pytest.mark.rom
def test_married_pin_gains_lumber_for_house_two():
    import hashlib
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.runtime.retro_setup import make_harvest_env
    from retro_harness import AuditedEnv, AuditCapabilities

    source = GAME_DIR / "LN_house_married_natural.state"
    env = AuditedEnv(
        make_harvest_env(source.stem, lean_step=True),
        capabilities=AuditCapabilities.all("harvest.house2_wood"),
    )
    fields = ("season", "day", "hour", "minute", "stored_wood", "money", "stamina",
              "tilemap", "upgrade_flags", "marriage_flags")
    try:
        env.reset()
        start = {key: read_ram_value(env.get_ram(), key) for key in fields}
        task = GatherHouseWoodTask(target=500)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        result = None
        frames = 0
        for frame in range(22000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            frames = frame + 1
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        else:
            wood = read_ram_value(env.get_ram(), "stored_wood")
            pytest.fail(f"gather_house_wood exceeded 22000 frames phase={task._phase} wood={wood}")
        end = {key: read_ram_value(env.get_ram(), key) for key in fields}
        audit = env.audit()
        assert result is not None and result.status == TaskStatus.SUCCESS, result.reason
        assert end["stored_wood"] > start["stored_wood"]
        assert end["money"] == start["money"]
        assert (end["upgrade_flags"] & 0xC0) == (start["upgrade_flags"] & 0xC0)
        assert end["marriage_flags"] & 0x10
        assert end["marriage_flags"] == start["marriage_flags"]
        assert audit.ram_writes == 0
        assert audit.mid_run_loads == 0
        report = {
            "source_state": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "start": start,
            "end": end,
            "frames": frames,
            "wood_receipt": task.receipt,
            "ram_writes": audit.ram_writes,
            "mid_run_state_loads": audit.mid_run_loads,
        }
        out = PROJECT_DIR / "recordings" / "house2_wood.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        save_native_state(env, "LN_house2_wood_natural")
    finally:
        env.close()
