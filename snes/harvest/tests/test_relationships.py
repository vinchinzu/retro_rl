"""Affection receipts, spending reserves, and social stops through the composer."""
from pathlib import Path

import pytest

from harvest.core.ram_catalog import field_spec, ram_index, read_ram_value as value
from harvest.tasks.relationships import FLOWER_PRICE, RelationshipTask, flower_affordable, social_mode
from retro_harness import TaskStatus, WorldState
from tests.day_plan_test_helpers import make_date_world, set_money


def write(ram, name, amount):
    spec = field_spec(name)
    index = ram_index(ram, spec.address, live_offset=spec.live_offset)
    stored = spec.to_storage(amount)
    for byte in range(spec.width):
        ram[index + byte] = stored >> (8 * byte) & 255


def funded_world():
    world = make_date_world(0, season=0, day=4, hour=8)
    write(world.ram, "weekday", 4)
    set_money(world.ram, 5_300)
    return world


def test_flower_spending_preserves_farm_and_final_cash(monkeypatch):
    monkeypatch.setenv("HARVEST_SOCIAL", "gifts")
    world = funded_world()
    assert flower_affordable(world.ram)
    set_money(world.ram, 5_290)
    assert not flower_affordable(world.ram)
    write(world.ram, "year", 2)
    set_money(world.ram, 99_840 + FLOWER_PRICE)
    assert flower_affordable(world.ram)
    set_money(world.ram, 99_840 + FLOWER_PRICE - 10)
    assert not flower_affordable(world.ram)


@pytest.mark.parametrize("field,amount", [
    ("nina_hearts", 496), ("nina_hearts", 512), ("marriage_flags", 4),
    ("held_item", 0x14), ("season", 1), ("weekday", 0), ("weekday", 6),
    ("hour", 14), ("day", 23),
])
def test_optional_flower_needs_a_safe_scheduled_recipient(monkeypatch, field, amount):
    monkeypatch.setenv("HARVEST_SOCIAL", "gifts")
    world = funded_world()
    write(world.ram, field, amount)
    assert not flower_affordable(world.ram)


def test_social_switch_and_unsafe_gifts(monkeypatch):
    monkeypatch.delenv("HARVEST_SOCIAL", raising=False)
    assert social_mode() == "off"
    monkeypatch.setenv("HARVEST_SOCIAL", "talk")
    assert not flower_affordable(funded_world().ram)
    monkeypatch.setenv("HARVEST_SOCIAL", "typo")
    with pytest.raises(ValueError):
        social_mode()
    world = funded_world()
    write(world.ram, "held_item", 2)  # poison mushroom
    task = RelationshipTask(girl="nina")
    task.reset(world)
    assert task.step(world).status == TaskStatus.FAILURE


def test_flower_still_fits_after_morning_chores(monkeypatch):
    monkeypatch.setenv("HARVEST_SOCIAL", "gifts")
    world = funded_world()
    write(world.ram, "hour", 13)
    assert flower_affordable(world.ram)


def test_daily_social_ledger_is_shared_between_shop_builds():
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    ctx = TaskBuildContext()
    spec = PhaseSpec("SHOP", PhaseKind.SHOP_BUY, {"max_bags": 1})
    first = build_phase_task(ctx, spec, funded_world())
    second = build_phase_task(ctx, spec, funded_world())
    first.social_visits.add("nina")
    assert second.social_visits == {"nina"}
    assert TaskBuildContext().social_visits == set()  # next day starts fresh


def test_bench_reports_each_measured_counter_and_keeps_missing_old_fields_unknown():
    from harvest.tools.y1_bench import relationships
    rows = relationships({"start": {"nina_hearts": 8}, "end": {"nina_hearts": 14, "eve_hearts": 512}})
    assert rows["nina"]["delta"] == 6 and rows["nina"]["points"] == 0
    assert rows["eve"]["points"] == 0 and not rows["eve"]["in_window"]
    assert rows["ann"]["after"] is None
    assert relationships({"end": {"maria_hearts": 496}})["maria"]["points"] == 31


@pytest.mark.rom
@pytest.mark.parametrize("mode", ["off", "talk", "gifts"])
def test_seed_errand_adds_one_funded_flower_without_rebuying_seeds(monkeypatch, mode):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_phase_types import PhaseSpec, PhaseKind
    from harvest.tasks.nav import make_action
    from retro_harness import AuditedEnv, AuditCapabilities

    pin = "PR_e2es4_Y2_Spring_D4"
    if not (Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes" / f"{pin}.state").exists():
        pytest.skip(f"missing local pin {pin}")
    monkeypatch.setenv("HARVEST_SOCIAL", mode)
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.errand_relationships"))
    try:
        env.reset()
        before = {k: value(env.get_ram(), k) for k in
                  ("money", "potato_seeds", "maria_hearts", "ann_hearts", "nina_hearts", "ellen_hearts", "eve_hearts")}
        task = DayPlanTask(phase_sequence=[PhaseSpec("BUY_SEEDS", PhaseKind.SHOP_BUY, {"max_bags": 1})])
        task.reset(WorldState(0, env.get_ram(), {}, None))
        for frame in range(14_000):
            result = task.step(WorldState(frame, env.get_ram(), {}, None))
            if task.phase_results:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        row = next(row for row in task.phase_results if row["phase"] == "BUY_SEEDS")
        assert row["status"] == "success", row
        assert value(ram, "potato_seeds") == before["potato_seeds"] + 1
        assert value(ram, "money") == before["money"] - 200 - (300 if mode == "gifts" else 0)
        assert value(ram, "nina_hearts") == before["nina_hearts"] + (6 if mode == "gifts" else 0)
        assert before["maria_hearts"] <= value(ram, "maria_hearts") <= before["maria_hearts"] + 8
        for girl in ("ann", "ellen", "eve"):
            assert value(ram, f"{girl}_hearts") == before[f"{girl}_hearts"]
        assert value(ram, "tilemap") == 0 and value(ram, "held_item") == 0
        assert "bags=1/1" in row["reason"]
        if mode == "gifts":
            assert task._build_ctx.social_visits == {"nina", "maria"}
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("SOCIAL_ERRAND", mode, frame, row)
    finally:
        env.close()


def drive(env, task, limit=8_000):
    from harvest.tasks.nav import make_action
    task.reset(WorldState(0, env.get_ram(), {}, None))
    for frame in range(limit):
        result = task.step(WorldState(frame, env.get_ram(), {}, None))
        if result.status != TaskStatus.RUNNING:
            assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
            return frame
        env.step(result.action.action if result.action else make_action())
    pytest.fail(f"task budget: {task.name}")


@pytest.mark.rom
@pytest.mark.parametrize("girl,delta", [("maria", 4), ("ann", 2), ("nina", 6), ("ellen", 5), ("eve", 8)])
def test_native_bought_flower_is_given_to_the_named_woman(monkeypatch, girl, delta):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.buy_seeds import BuyShelfItemTask, town_coords_settled
    from harvest.tasks.relationships import BuyFlowerTask, TOWN_SPRITES
    from harvest.core.npc_catalog import game_objects
    from harvest.maps.map_routes import segment_waypoints
    from harvest.maps.map_types import Waypoint
    from harvest.planner.tasks.multi_nav import MultiMapNavTask
    from harvest.tasks.nav import make_action
    from retro_harness import AuditedEnv, AuditCapabilities

    monkeypatch.setenv("HARVEST_SOCIAL", "off")
    pin = "PR_spring_check_Y1_Spring_D15"  # unmarried; enough native cash for seeds + flower
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.isolated_relationship"))
    try:
        env.reset()
        money = value(env.get_ram(), "money")
        shop = BuyShelfItemTask()
        shop.reset(WorldState(0, env.get_ram(), {}, None))
        for frame in range(8_000):
            ram = env.get_ram()
            if (shop._bought and value(ram, "tilemap") == 0x1C and value(ram, "input_lock") == 1
                    and not value(ram, "game_state") & 0x8000):
                break
            result = shop.step(WorldState(frame, ram, {}, None))
            env.step(result.action.action if result.action else make_action())
        else:
            pytest.fail("seed shelf not reached")
        drive(env, BuyFlowerTask())
        # Native shop exit, preserving the paid-for flower.
        drive(env, MultiMapNavTask(waypoints=segment_waypoints("shop_to_town"),
                                  allow_opportunistic_clear=False))
        for _ in range(480):
            if town_coords_settled(env.get_ram()) and value(env.get_ram(), "player_y") >= 268:
                break
            env.step(make_action(down=True))
        if girl in {"ann", "ellen", "eve"}:
            upper = segment_waypoints("town_shop_to_path")[:-1]
            lower = segment_waypoints("town_to_tool_door")[:2]
            drive(env, MultiMapNavTask(waypoints=upper + lower, allow_opportunistic_clear=False))
        if girl == "ann":
            while value(env.get_ram(), "hour") < 9:
                env.step(make_action())
            drive(env, MultiMapNavTask(waypoints=segment_waypoints("town_to_tool_door")[2:],
                                      allow_opportunistic_clear=False))
            for _ in range(240):
                env.step(make_action())
        elif girl == "eve":
            while value(env.get_ram(), "hour") < 17:
                env.step(make_action())
            drive(env, MultiMapNavTask(waypoints=[
                Waypoint(4, (600, 890), radius=8), Waypoint(4, (440, 890), radius=8),
                Waypoint(4, (280, 890), radius=8), Waypoint(4, (152, 888), radius=8),
                Waypoint(4, (152, 872), radius=6, is_exit=True, exit_direction="up"),
            ], allow_opportunistic_clear=False))
            for _ in range(240):
                env.step(make_action())
        ram = env.get_ram()
        actor = next((obj for obj in game_objects(ram) if obj.is_npc_candidate
                      and obj.sprite_table_idx in TOWN_SPRITES[girl]), None)
        assert actor is not None, (girl, [(obj.sprite_table_idx, obj.pixel) for obj in game_objects(ram)])
        task = RelationshipTask(girl=girl, slot=actor.slot, timeout=2_400)
        drive(env, task, 3_000)
        assert task.receipt["delta"] == delta, task.receipt
        assert value(env.get_ram(), "money") == money - 500
        assert value(env.get_ram(), "held_item") == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("ISOLATED_GIFT", task.receipt)
    finally:
        env.close()


@pytest.mark.rom
def test_three_native_mornings_grow_nina_without_abandoning_wife_gifts(monkeypatch):
    """Isolated shopping/family loop, not a whole-farm economy benchmark."""
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, GO_TO_SLEEP_PHASE
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec
    from harvest.tasks.eve_gift import apply_wife_gift
    from retro_harness import AuditedEnv, AuditCapabilities

    monkeypatch.setenv("HARVEST_SOCIAL", "gifts")
    env = AuditedEnv(make_harvest_env("PR_e2es4_Y2_Spring_D14", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.gradual_relationships"))
    try:
        env.reset()
        for expected_day, expected_nina in [(15, 14), (16, 20), (17, 26)]:
            ram = env.get_ram()
            before_eve = value(ram, "eve_hearts")
            task = DayPlanTask(phase_sequence=apply_wife_gift([
                EXIT_TO_FARM_PHASE,
                PhaseSpec("SOCIAL_SHOP", PhaseKind.SHOP_BUY, {"max_bags": 1}),
                GO_TO_SLEEP_PHASE,
            ], ram))
            drive(env, task, 45_000)
            ram = env.get_ram()
            assert value(ram, "day") == expected_day
            assert value(ram, "nina_hearts") == expected_nina
            assert value(ram, "eve_hearts") >= before_eve
            assert any(row["phase"] == "GIVE_WIFE" and row["status"] == "success"
                       for row in task.phase_results), task.phase_results
            assert value(ram, "held_item") == 0
            print("GRADUAL_SOCIAL", {k: value(ram, k) for k in
                  ("day", "money", "nina_hearts", "maria_hearts", "eve_hearts")})
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
