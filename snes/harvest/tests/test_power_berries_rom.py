"""Power berries require an eaten count and max-stamina gain on the ROM."""
from pathlib import Path
import gzip
import hashlib
import json
import pytest
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState
from harvest.core.ram_catalog import read_ram_value, read_ram_u16
from harvest.tasks.nav import make_action
from harvest.tasks.power_berry import PowerBerryTask, PowerBerryPickupTask


def berry_pin(env, source, stage, predecessor):
    """Keep collection-day states beside the ROM, with replay and audit receipts."""
    from harvest.paths import GAME_DIR, PROJECT_DIR
    ram = env.get_ram()
    values = {k: int(read_ram_value(ram, k)) for k in
              ("year", "season", "day", "hour", "minute", "power_berry_count", "max_stamina")}
    season = ("Spring", "Summer", "Fall", "Winter")[values["season"]]
    name = f"LN_berries_20261103_{source}_Y{values['year']+1}_{season}_D{values['day']}_{stage}"
    state = gzip.compress(env.em.get_state(), mtime=0)
    (GAME_DIR / f"{name}.state").write_bytes(state)
    receipt = {"pin": name, "predecessor": predecessor, **values,
               "ram_writes": env.audit().ram_writes, "mid_run_loads": env.audit().mid_run_loads,
               "state_sha256": hashlib.sha256(state).hexdigest(),
               "skill_sha256": hashlib.sha256((PROJECT_DIR / "harvest/tasks/power_berry.py").read_bytes()).hexdigest()}
    out = PROJECT_DIR / "recordings/berries"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{source}_{stage}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return name


def run(env, task, limit=180000, *, pin_source=None, predecessor=None, farm_idle=0):
    """Drive one task. ``farm_idle`` neutral frames are spent outdoors on the farm before it walks: idle indoors does
    not move the ROM's RNG, outdoors it does (the Egg Festival prize is an RNG coin flip)."""
    before_pin = berry_pin(env, pin_source, "before", predecessor) if pin_source else None
    held = []
    if isinstance(task, PowerBerryTask):
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
        from harvest.planner.day_phase_types import PhaseSpec
        task = build_phase_task(TaskBuildContext(), PhaseSpec("COLLECT_POWER_BERRY", "power_berry",
                                {"source": task.source, "timeout": task.timeout}),
                                WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
    task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
    for frame in range(limit):
        if farm_idle > 0 and int(read_ram_value(env.get_ram(), "tilemap")) == 0:
            farm_idle -= 1
            env.step(make_action())
            continue
        result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
        item = int(read_ram_value(env.get_ram(), "held_item"))
        if not held or held[-1] != item:
            held.append(item)
        if result.status != TaskStatus.RUNNING:
            break
        env.step(result.action.action if result.action else make_action())
    assert result.status == TaskStatus.SUCCESS, (task.name, frame, result.reason)
    if pin_source:
        berry_pin(env, pin_source, "after", before_pin)
    if pin_source == "fish":
        assert held == [0, 7, 0, 8, 0], held
    return frame


def test_collection_refuses_an_eleventh_berry_and_bad_weather():
    import numpy as np
    from harvest.core.ram_catalog import field_spec
    ram = np.zeros(0x20000, dtype=np.uint8)
    for source in ("statue", "fish"):
        ram[0x196] = 2
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        task = PowerBerryTask(source=source)
        task.reset(world)
        assert not task.can_start(world)
        assert task.step(world).status == TaskStatus.BLOCKED
    ram[0x196] = 0
    ram[field_spec("power_berry_count").address] = 10
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    for source in ("field_1", "stump", "statue", "fish", "chicken_trade", "harvest_festival", "snow_flower"):
        task = PowerBerryTask(source=source)
        task.reset(world)
        assert not task.can_start(world)
        assert task.step(world).status == TaskStatus.BLOCKED


@pytest.mark.rom
@pytest.mark.parametrize("source, bit", [("field_1", 0x0800), ("field_2", 0x1000)])
def test_field_berry_is_dug_picked_and_eaten(source, bit):
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.power_berries"))
    try:
        env.reset()
        if source == "field_2":
            from harvest.planner.tasks.home_sleep import GoToSleepTask
            run(env, PowerBerryTask(source="field_1"))
            run(env, GoToSleepTask(), limit=20000)
        before = read_ram_value(env.get_ram(), "power_berry_count")
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        frames = run(env, PowerBerryTask(source=source), pin_source=source,
                     predecessor="PR_spring_final_Y1_Spring_D3" + (" -> field_1 -> sleep" if source == "field_2" else ""))
        assert read_ram_u16(env.get_ram(), 0x11F64) & bit
        assert read_ram_value(env.get_ram(), "power_berry_count") == before + 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        print(source, frames)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_statue_berry_is_hammered_picked_and_eaten():
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D4", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.statue_berry"))
    try:
        env.reset()
        run(env, PowerBerryTask(source="statue"), pin_source="statue", predecessor="PR_spring_final_Y1_Spring_D4")
        assert read_ram_u16(env.get_ram(), 0x11F64) & 0x0200
        assert read_ram_value(env.get_ram(), "power_berry_count") == 1
        assert read_ram_value(env.get_ram(), "max_stamina") == 110
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_stump_berry_is_acquired_and_eaten():
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.mountain_power_berry"))
    try:
        env.reset()
        run(env, PowerBerryTask(source="stump"), pin_source="stump", predecessor="PR_spring_final_Y1_Spring_D3")
        assert read_ram_u16(env.get_ram(), 0x11F64) & 0x0400
        assert read_ram_value(env.get_ram(), "power_berry_count") >= 1
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def _closed_pocket_ram(flag):
    import numpy as np
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[0x11F64], ram[0x11F65] = flag & 0xFF, flag >> 8
    return ram


def test_cave_source_stays_closed_until_the_boulder_flag():
    """Offline: ``$7F1F64`` bit 2 (Year 1 Winter's first snowstorm) is the only thing that opens the pocket."""
    closed, opened = (WorldState(frame=0, ram=_closed_pocket_ram(f), info={}, obs=None) for f in (0, 0x0002))
    assert not PowerBerryTask(source="cave").can_start(closed)
    assert PowerBerryTask(source="cave").can_start(opened)
    for source in ("stump", "field_1", "statue"):
        assert PowerBerryTask(source=source).can_start(closed), source


def _declare_research_poke(env, wram_address, value):
    """RAM write on purpose (research), then restart the audited attempt so the recipe's own audit stays clean."""
    for offset, byte in enumerate(int(value).to_bytes(2, "little")):
        env.data.memory.assign(0x7E0000 + wram_address + offset, "|u1", byte)
    # AuditedEnv only counts ``data.set_value``; a raw ``memory.assign`` is invisible to it, so declare the write.
    env.record_ram_write(2)
    assert env.audit().ram_writes == 2
    env.begin_attempt()


@pytest.mark.rom
def test_cave_berry_is_refused_while_the_boulder_pocket_is_closed():
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.cave_berry_gate"))
    try:
        env.reset()
        task = PowerBerryTask(source="cave")
        world = WorldState(frame=0, ram=env.get_ram(), info={}, obs=None)
        task.reset(world)
        assert not task.can_start(world)
        result = task.step(world)
        assert result.status == TaskStatus.BLOCKED and "boulder" in result.reason
    finally:
        env.close()


@pytest.mark.rom
def test_cave_berry_is_chopped_dropped_picked_and_eaten_once_the_pocket_is_open():
    """Farm -> mountain landing -> BFS walk to the pocket stump -> six axe chops -> drop into the sealed
    chamber -> pick and eat. The pocket's boulder flag is the one research poke: the game sets
    ``$7F1F64`` bit 2 itself on Year 1 Winter's first snowstorm (no later than D8, see the module notes)."""
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("PR_spring_final_Y1_Spring_D3", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.cave_berry"))
    try:
        env.reset()
        _declare_research_poke(env, 0x11F64, read_ram_u16(env.get_ram(), 0x11F64) | 0x0002)
        before_max = read_ram_value(env.get_ram(), "max_stamina")
        frames = run(env, PowerBerryTask(source="cave"), limit=60000)
        ram = env.get_ram()
        assert read_ram_u16(ram, 0x11F6A) & 0x0040, "eating on map 0x29 sets the cave berry's own flag"
        assert read_ram_value(ram, "power_berry_count") == 1
        assert read_ram_value(ram, "max_stamina") == before_max + 10
        print("cave berry frames", frames)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def _winter_night(*pokes):
    """One native night from the Year 1 Winter D1 morning after declared calendar/forecast pokes (research).

    Returns the RAM at the next morning. The night pass applies the forecast made the night before
    (``SetWeatherFlags``) and only then makes the next one (``WeatherTomorrow``, after the day rolls over)."""
    from harvest.core.ram_catalog import field_spec
    from harvest.planner.tasks.home_sleep import GoToSleepTask
    from harvest.runtime.retro_setup import make_harvest_env
    env = AuditedEnv(make_harvest_env("LN_calendar_Y1_Winter_D1", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.boulder_flag"))
    try:
        env.reset()
        ram = env.get_ram()
        assert [read_ram_value(ram, k) for k in ("year", "season", "day")] == [0, 3, 1]
        assert not read_ram_u16(ram, 0x11F64) & 0x0002
        for key, value in pokes:
            env.data.memory.assign(0x7E0000 + field_spec(key).address, "|u1", value)
            env.record_ram_write()
        env.begin_attempt()
        run(env, GoToSleepTask(), limit=40000)
        for _ in range(60):
            env.step(make_action())
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        return env.get_ram().copy()
    finally:
        env.close()


@pytest.mark.rom
def test_weather_4_forecast_sets_the_boulder_flag_when_the_night_applies_it():
    ram = _winter_night(("weather_tomorrow", 4))
    assert read_ram_u16(ram, 0x11F64) & 0x0002, "SetWeatherFlags .fair must set $7F1F64 bit 2"


@pytest.mark.rom
def test_year1_winter_night_into_day7_makes_the_weather_4_forecast():
    """``WeatherTomorrow`` ``.specialdays``: Year 1 Winter, flag clear, day 7 (read after the rollover) -> weather 4.
    The flag itself lands the next night, so the pocket opens on the D8 morning at the latest (a random 1-in-8
    snowstorm on an earlier Winter night can open it sooner)."""
    ram = _winter_night(("day", 6))
    assert read_ram_value(ram, "day") == 7
    assert read_ram_value(ram, "weather_tomorrow") == 4
    assert not read_ram_u16(ram, 0x11F64) & 0x0002, "applied only on the following night"


@pytest.mark.rom
def test_fish_is_caught_returned_and_its_berry_eaten():
    from harvest.runtime.retro_setup import make_harvest_env
    pin = "PR_spring_final_Y1_Spring_D4"
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.fish_berry"))
    try:
        env.reset()
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        run(env, PowerBerryTask(source="fish"), limit=24000, pin_source="fish", predecessor=pin)
        assert read_ram_u16(env.get_ram(), 0x11F66) & 0x2000
        assert read_ram_value(env.get_ram(), "power_berry_count") == 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_cave_berry_from_the_natural_winter_collection_day():
    from harvest.runtime.retro_setup import make_harvest_env
    pin = "PR_y1new_Y1_Winter_D8"
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.winter_cave_berry"))
    try:
        env.reset()
        assert read_ram_u16(env.get_ram(), 0x11F64) & 2
        before = read_ram_value(env.get_ram(), "power_berry_count")
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        run(env, PowerBerryTask(source="cave"), limit=60000, pin_source="cave", predecessor=pin)
        assert read_ram_u16(env.get_ram(), 0x11F6A) & 0x40
        assert read_ram_value(env.get_ram(), "power_berry_count") == before + 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_hawker_chicken_trade_berry_is_eaten():
    from harvest.runtime.retro_setup import make_harvest_env
    pin = "PR_y1new_Y1_Fall_D1"
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.chicken_trade_berry"))
    try:
        env.reset()
        hens = read_ram_value(env.get_ram(), "num_chickens")
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        run(env, PowerBerryTask(source="chicken_trade"), limit=30000,
            pin_source="chicken_trade", predecessor=pin)
        assert read_ram_value(env.get_ram(), "num_chickens") == hens - 1
        assert read_ram_value(env.get_ram(), "power_berry_count") == 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
def test_harvest_festival_fish_offering_berry_is_eaten():
    from harvest.runtime.retro_setup import make_harvest_env
    pin = "PR_y1new_Y1_Fall_D12"
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.harvest_festival_berry"))
    try:
        env.reset()
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        run(env, PowerBerryTask(source="harvest_festival"), limit=40000,
            pin_source="harvest_festival", predecessor=pin)
        assert read_ram_u16(env.get_ram(), 0x11F6C) & 0x8000
        assert read_ram_value(env.get_ram(), "power_berry_count") == 1
        assert read_ram_value(env.get_ram(), "max_stamina") == maximum + 10
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def _egg_festival(farm_idle):
    from harvest.runtime.retro_setup import make_harvest_env
    pin = "PR_e2ec1_Y2_Fall_D20"
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.egg_festival_berry"))
    try:
        env.reset()
        before = read_ram_value(env.get_ram(), "power_berry_count")
        maximum = read_ram_value(env.get_ram(), "max_stamina")
        run(env, PowerBerryTask(source="egg_festival"), limit=40000, farm_idle=farm_idle,
            pin_source=f"egg_festival_idle{farm_idle}", predecessor=pin)
        ram = env.get_ram()
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        return (read_ram_u16(ram, 0x11F6C) & 0x7000, read_ram_value(ram, "power_berry_count") - before,
                read_ram_value(ram, "max_stamina") - maximum)
    finally:
        env.close()


@pytest.mark.rom
def test_egg_festival_win_can_pay_the_turtle_shell():
    """The prize roll is a coin flip: bit 0x1000 is the turtle shell (+21 ranch points), no berry on this win."""
    assert _egg_festival(farm_idle=0) == (0x1000, 0, 0)


@pytest.mark.rom
def test_egg_festival_win_can_arm_the_house_berry_which_is_picked_and_eaten():
    """The other coin face: bit 0x2000 arms the house berry, the prize scene seats the farmer, and the berry (on the
    table's edge) is lifted from its east side and eaten (bit 0x4000, count +1, max stamina +10)."""
    assert _egg_festival(farm_idle=509) == (0x6000, 1, 10)
