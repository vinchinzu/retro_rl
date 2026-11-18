"""Receipt checks from natural morning pins, bounded to a single shopping trip."""
from pathlib import Path

import pytest

from harvest.core.ram_catalog import read_ram_value, read_ram_u8
from harvest.tasks.buy_seeds import BuyShelfItemTask, tool_owned
from harvest.tasks.nav import make_action
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState


@pytest.mark.rom
def test_sprinkler_waters_sparse_crops_from_the_final_spring_failure():
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.sprinkler import SprinklerWaterTask, dry_tiles

    pin = "PR_fycandidate_Y3_Spring_D13"
    pins = Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes"
    if not (pins / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is absent")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.sparse_sprinkler"))
    try:
        env.reset()
        # Morning pins are indoors; the watering phase reads the live farm
        # map after the composer's EXIT_TO_FARM, not the house's tile buffer.
        from harvest.planner.tasks.inventory_exit import ExitToFarmTask
        leave = ExitToFarmTask()
        leave.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(2000):
            result = leave.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        before = dry_tiles(env.get_ram(), (2, 3, 62, 60))
        assert before, "pin must contain dry crops after the native farm exit"
        task = SprinklerWaterTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(24_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert task._watered > 0, result.reason
        assert len(dry_tiles(env.get_ram(), task.bounds)) < len(before), result.reason
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("item", ["medicine", "turnip", "sprinkler"])
def test_care_supply_trip_buys_the_requested_item_and_returns_home(item):
    from harvest.runtime.retro_setup import make_harvest_env

    pin = "PR_e2es4_Y2_Spring_D4"
    pins = Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes"
    if not (pins / f"{pin}.state").exists():
        pytest.skip(f"missing local pin {pin}")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.care_shopping"))
    try:
        env.reset()
        before = read_ram_value(env.get_ram(), "money")
        stock = read_ram_value(env.get_ram(), "turnip_seeds")
        task = (BuyShelfItemTask(tool_ids=(9,)) if item == "medicine" else BuyShelfItemTask(tool_ids=(0x15,)) if item == "sprinkler"
                else BuyShelfItemTask(stock_field="turnip_seeds"))
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(18000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason, task.phase_text, {k: read_ram_value(ram, k) for k in ("player_x", "player_y", "dialog_text_id", "input_lock")})
        assert read_ram_value(ram, "tilemap") == 0
        assert read_ram_value(ram, "money") == before - ({"medicine": 1000, "turnip": 200, "sprinkler": 2000}[item])
        if item == "medicine":
            assert tool_owned(ram, 9) and read_ram_u8(ram, 0x11F01) & 1
        elif item == "sprinkler":
            assert tool_owned(ram, 0x15) and read_ram_u8(ram, 0x11F02) & 0x10
        else:
            assert read_ram_value(ram, "turnip_seeds") == stock + 1
        if item in {"medicine", "sprinkler"}:
            from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
            from harvest.core.carry import tool_in_carry_pair

            wanted = 9 if item == "medicine" else 0x15
            pickup = EnsureCarryToolTask(tool_id=wanted, exit_when_done=True)
            pickup.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
            for frame in range(6000):
                result = pickup.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
            assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
            assert tool_in_carry_pair(env.get_ram(), wanted)
        if item == "sprinkler":
            from harvest.tasks.sprinkler import SprinklerWaterTask, dry_tiles
            bounds = (2, 3, 62, 60)
            dry_before = dry_tiles(env.get_ram(), bounds)
            water_before = read_ram_value(env.get_ram(), "watering_can")
            watering = SprinklerWaterTask()
            watering.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
            for frame in range(24000):
                result = watering.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
            print("SPRINKLER", len(dry_before), len(dry_tiles(env.get_ram(), bounds)), result.reason)
            assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
            assert watering._watered > 0
            assert len(dry_tiles(env.get_ram(), bounds)) < len(dry_before)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
