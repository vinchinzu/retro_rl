"""Livestock economy (`harvest.planner.livestock_econ`): one price table, named gates, the ledger."""

import numpy as np
import pytest

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_MAP, MATURE_GRASS_TILES
from harvest.planner import livestock_econ as econ
from harvest.planner.spring_opt import CROPS
from harvest.tasks.buy_seeds import SEED_BAG_PRICE, TOOL_SHELVES
from harvest.tasks.grass_planter import PASTURE_BOUNDS


def ram(**fields):
    result = np.zeros(0x20000, dtype=np.uint8)
    for name, value in fields.items():
        spec = field_spec(name)
        value = spec.to_storage(value)
        for shift in range(spec.width):
            result[spec.address + shift] = (value >> (8 * shift)) & 255
    return result


def mature_grass(result, count):
    x0, y0, x1, _ = PASTURE_BOUNDS
    grass = next(iter(MATURE_GRASS_TILES))
    for i in range(count):
        result[ADDR_MAP + y0 * 64 + x0 + i % (x1 - x0 + 1) + 64 * (i // (x1 - x0 + 1))] = grass
    return result


# Hen and cow funded, pasture and hay in, a Summer Tuesday.
COW_READY = dict(season=1, day=14, weekday=2, money=5_400, num_chickens=1,
                 planted_grass=32, stored_grass=8)


def test_year_two_on_buys_a_bigger_herd_in_spring_too():
    """A happy cow is 28 ending points and ~350G of milk a day; barn chores stop the clock."""
    later = COW_READY | dict(year=1, num_cows=2, planted_grass=80, stored_grass=40)
    assert econ.livestock_block(ram(**later), "cow") is None
    assert econ.livestock_block(ram(**(later | dict(season=0))), "cow") is None
    assert econ.livestock_block(ram(**(later | dict(season=3))), "cow") == "season"
    assert econ.livestock_block(ram(**(later | dict(num_cows=econ.COW_TARGET_LATER))), "cow") == "cow_target"
    # Year 1 keeps its two cows and Summer/Fall buys.
    assert econ.livestock_block(ram(**(later | dict(year=0))), "cow") == "cow_target"


# Fall D5 of the Fall-D1 pin replay: two cows, two hens, 103 planted tiles, 146 hay.
FALL = COW_READY | dict(season=2, day=5, weekday=2, money=24_000, num_cows=2, num_chickens=2,
                        planted_grass=103, stored_grass=146)


def test_fall_buys_a_third_cow_only_when_the_hay_carries_it_to_spring():
    assert econ.cow_target(ram(**FALL)) == econ.COW_TARGET_FALL
    assert econ.livestock_block(ram(**FALL), "cow") is None
    # Same herd, thinner hay: the cow would starve in Winter, so no purchase.
    thin = FALL | dict(stored_grass=60)
    assert econ.cow_target(ram(**thin)) == econ.COW_TARGET
    assert econ.livestock_block(ram(**thin), "cow") == "cow_target"
    # Late in Fall the stock must already cover Winter and Spring's bridge on its own.
    assert econ.cow_target(ram(**(FALL | dict(day=28, stored_grass=240)))) == econ.COW_TARGET_FALL
    assert econ.cow_target(ram(**(FALL | dict(day=28, stored_grass=200)))) == econ.COW_TARGET


def test_third_cow_is_a_fall_only_year_one_rule():
    assert econ.cow_target(ram(**(FALL | dict(season=1)))) == econ.COW_TARGET
    assert econ.cow_target(ram(**(FALL | dict(num_cows=3)))) == econ.COW_TARGET
    assert econ.cow_target(ram(**(FALL | dict(year=1)))) == econ.COW_TARGET_LATER


def test_every_price_reads_one_table():
    for kind in ("potato", "corn", "tomato", "grass"):
        assert SEED_BAG_PRICE[f"{kind}_seeds"] == econ.SHOP_PRICE_G[f"{kind}_seed"]
    for tool, shelf in TOOL_SHELVES.items():
        assert shelf.price == econ.TOOL_PRICE[tool]
    for name in ("turnip", "potato", "tomato", "corn"):
        assert CROPS[name].seed_cost_g == econ.SHOP_PRICE_G[f"{name}_seed"]
        assert CROPS[name].sell_price_g == econ.SHIP_PRICE_G[name]


def test_livestock_block_names_the_first_closed_gate():
    assert econ.livestock_block(ram(**COW_READY), "cow") is None
    assert econ.livestock_block(ram(**COW_READY), "chicken") == "have_hen"
    cases = {
        "season": dict(season=0),
        "shop_closed": dict(weekday=0),
        "cow_target": dict(num_cows=2),
        "pasture": dict(planted_grass=31),
        "money": dict(money=4_999),
        "hay": dict(stored_grass=7),
        "reserve": dict(money=5_299),
    }
    for reason, change in cases.items():
        assert econ.livestock_block(ram(**(COW_READY | change)), "cow") == reason, reason
    # The shop's own grass gate: 16 per animal after the buy, chicks included.
    assert econ.livestock_block(ram(**(COW_READY | dict(num_chickens=3, planted_grass=63))), "cow") == "grass"


def test_ranch_gates_explain_the_morning():
    gates = econ.ranch_gates(ram(**COW_READY))
    assert gates["cow"] is None and gates["hen"] == "have_hen"
    assert gates["tools"] == ["brush", "milker"]
    assert gates["house_pay"] is False
    assert gates["hay_wanted"] == econ.HAY_DAYS_WANTED * 2
    winter = econ.ranch_gates(ram(**(COW_READY | dict(season=3, num_cows=1))))
    assert winter["cow"] == "season"
    assert winter["house_pay"] is True
    # The trip waits for a cow and the store's hours; the Winter ban is the season policy.
    assert gates["tool_trip"] is False
    owned = COW_READY | dict(num_cows=1)
    assert econ.ranch_gates(ram(**owned))["tool_trip"] is True
    assert econ.ranch_gates(ram(**(owned | dict(weekday=6))))["tool_trip"] is False


def test_pasture_errand_plants_first_and_cuts_when_hay_runs_short():
    summer = dict(season=1, day=10, weekday=2, money=3_000, num_chickens=1, planted_grass=16,
                  stored_grass=40, grass_seeds=1, corn_seeds=99)
    errand = econ.pasture_errand(ram(**summer))
    assert errand.plant and errand.buy_to_stock > 1 and not errand.cut
    # Mature grass does not preempt planting while the barn holds a few days of hay.
    assert not econ.pasture_errand(mature_grass(ram(**summer), 10)).cut
    assert econ.pasture_errand(mature_grass(ram(**(summer | dict(stored_grass=7))), 10)) == econ.PastureErrand(cut=True)
    # Weekend: plant the bag in hand, no shop.
    weekend = econ.pasture_errand(ram(**(summer | dict(weekday=0))))
    assert (weekend.buy_to_stock, weekend.plant) == (0, True)
    # Fall grass never raises the shop's counter: no planting, a cut at 8 mature tiles.
    fall = dict(summer, season=2)
    assert econ.pasture_errand(ram(**fall)) == econ.PastureErrand()
    assert econ.pasture_errand(mature_grass(ram(**fall), econ.CUT_MIN_MATURE)).cut


def test_afternoon_pasture_errand_is_only_a_short_herds_cut():
    """y23_final Y2 Summer: wood trips ate every morning and hay hit 0 on D13."""
    from harvest.planner.day_phase_catalog import pasture_day_phases

    summer = dict(season=1, day=10, weekday=2, money=3_000, num_cows=2, num_chickens=3,
                  planted_grass=econ.PASTURE_MAX_TILES, grass_seeds=0, corn_seeds=99)
    short = mature_grass(ram(**(summer | dict(stored_grass=5))), 10)
    assert [p.phase for p in pasture_day_phases(short, afternoon=True)] == ["CUT_FODDER"]
    # A stocked barn's top-up cut and any planting wait for a morning.
    stocked = mature_grass(ram(**(summer | dict(stored_grass=30))), 10)
    assert econ.pasture_errand(stocked).cut
    assert pasture_day_phases(stocked, afternoon=True) == []
    planting = ram(**(summer | dict(planted_grass=16, stored_grass=5, grass_seeds=1)))
    assert pasture_day_phases(planting, afternoon=True) == []


def test_fall_stocking_uses_cover_grass_after_the_morning_chores(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases

    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    data = ram(year=1, season=2, day=28, num_cows=12, num_chickens=2, stored_grass=157, planted_grass=1168)
    # The little pasture is still regrowing; the cover field can fill the loft.
    data[ADDR_MAP + 20 * 64 + 40:ADDR_MAP + 20 * 64 + 60] = 0x79
    phases = pasture_day_phases(data, afternoon=True)
    assert [p.phase for p in phases] == ["CUT_FODDER"]
    assert phases[0].params["hay_goal"] == econ.hay_cut_target(data)
    assert phases[0].params["bounds"] == (2, 3, 62, 60)
    full = ram(year=1, season=2, day=28, num_cows=12, num_chickens=2, stored_grass=658, planted_grass=1168)
    full[ADDR_MAP + 20 * 64 + 40:ADDR_MAP + 20 * 64 + 60] = 0x79
    assert pasture_day_phases(full, afternoon=True) == []


@pytest.mark.rom
def test_natural_twelve_cow_fall_pin_banks_winter_hay_from_the_cover(monkeypatch):
    from pathlib import Path
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, pasture_day_phases
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.nav import make_action
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    pin = "PR_hw12b_20261111_s1_Y2_Fall_D28"
    if not (Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes" / f"{pin}.state").exists():
        pytest.skip("local natural twelve-cow Fall pin is absent")
    monkeypatch.setenv("HARVEST_STRATEGY", "score_herd")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.winter_hay"))
    try:
        env.reset()
        def drive(phases, limit):
            task = DayPlanTask(phase_sequence=phases, policy=DayPlannerPolicy(include_end_day=False))
            task.reset(WorldState(0, env.get_ram(), {}, None))
            for frame in range(limit):
                result = task.step(WorldState(frame, env.get_ram(), {}, None))
                if result.status != TaskStatus.RUNNING:
                    return result, frame
                env.step(result.action.action if result.action else make_action())
            pytest.fail(f"hay task did not finish: {task.progress_snapshot()}")
        result, _ = drive([EXIT_TO_FARM_PHASE], 6000)
        assert result.status == TaskStatus.SUCCESS, result.reason
        target = econ.hay_cut_target(env.get_ram())
        phases = pasture_day_phases(env.get_ram(), afternoon=True)
        assert any(p.phase == "CUT_FODDER" for p in phases), phases
        result, frames = drive(phases, 240_000)
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "stored_grass") >= target
        assert read_ram_value(env.get_ram(), "num_cows") == 12
        assert read_ram_value(env.get_ram(), "day") == 28
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("NATIVE_WINTER_HAY", frames, target, read_ram_value(env.get_ram(), "stored_grass"), env.audit())
    finally:
        env.close()


def test_spring_sows_the_wiped_pasture_once_there_is_a_herd():
    # Year 2 Spring D1: Winter wiped the grass and the planted counter, hay is gone.
    spring = dict(year=1, season=0, day=1, weekday=2, money=15_000, num_cows=1, num_chickens=2,
                  planted_grass=0, stored_grass=0, grass_seeds=0)
    errand = econ.pasture_errand(ram(**spring))
    assert errand.plant and errand.buy_to_stock > 0
    # Year 1 Spring has no animals: crops only, as before.
    assert econ.pasture_errand(ram(**(spring | dict(year=0, num_cows=0, num_chickens=0)))) == econ.PastureErrand()
    # The Fall cut also stocks the Spring days before the new grass is cut.
    fall = ram(year=0, season=2, day=30, num_cows=1, num_chickens=2)
    assert econ.hay_cut_target(fall) == 3 * (31 + econ.SPRING_HAY_DAYS)


def test_pasture_errand_waits_for_the_rings_corn_bags_on_early_summer_days():
    # Blank farm RAM: every ring is bare and wants a bag. The grass trip would
    # be a second town trip on a day when corn planted by D3 still gets six harvests.
    early = dict(season=1, day=econ.RING_FIRST_PLANT_DAYS, weekday=2, money=3_000, num_chickens=1,
                 planted_grass=16, stored_grass=40, grass_seeds=1)
    assert econ.pasture_errand(ram(**early)) == econ.PastureErrand()
    # Bags in hand for every ring, or past the planting days: the errand runs.
    stocked = early | dict(corn_seeds=99)
    assert econ.pasture_errand(ram(**stocked)).plant
    assert econ.pasture_errand(ram(**(early | dict(day=econ.RING_FIRST_PLANT_DAYS + 1)))).plant


def test_rancher_ledger_result_is_pinned():
    ledger = econ.build_rancher_ledger()
    assert (ledger.end_g, ledger.min_g, ledger.house1_index, ledger.grass_bags_paid) == (620_000, 40, 30, 328)
    assert ledger.cow_buys[:3] == (42, 44, 45)


# ── grass cover: the pasture errand is the herd's alone; the whole field is planner/field_cover ──

def _rich_summer(**over):
    base = COW_READY | dict(year=2, season=1, day=10, weekday=2, money=200_000, num_cows=4, num_chickens=2,
                            planted_grass=120, stored_grass=100, grass_seeds=0)
    return ram(**(base | over))


def test_the_pasture_errand_never_buys_field_cover(monkeypatch):
    """Rich or not, the herd errand stops at the herd's pasture; the cover bags ride the planner's own errand."""
    for name in ("cash", "score", "score_herd"):
        monkeypatch.setenv("HARVEST_STRATEGY", name)
        r = mature_grass(_rich_summer(planted_grass=econ.pasture_target_tiles(_rich_summer())), 0)
        assert not econ.pasture_errand(r).plant, name
        assert econ.pasture_errand(r).buy_to_stock == 0, name


def test_pasture_phases_carry_no_bag_limit_beyond_the_default(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases

    monkeypatch.setenv("HARVEST_STRATEGY", "cash")
    phases = pasture_day_phases(_rich_summer(planted_grass=20))
    plant = [p for p in phases if p.phase == "PLANT_PASTURE"]
    assert plant and "bag_limit" not in plant[0].params


def test_late_summer_pasture_funds_the_next_cow_and_winter_bridge():
    # Four animals currently eat. Budget for the third cow too, through the
    # remaining 40 growing days and 45 days without a productive new pasture.
    r = ram(year=0, season=1, day=20, num_cows=2, num_chickens=2,
            stored_grass=50, planted_grass=96)
    target = econ.pasture_target_tiles(r)
    assert target > 96
    assert target >= 5 * econ.GRASS_PER_ANIMAL
    assert 50 + int(target * 4 * econ.FALL_CUT_YIELD) >= 5 * (40 + 31 + econ.SPRING_HAY_DAYS)
    # Already-funded winters need no new hay reserve beyond the shop gate.
    full = r.copy()
    spec = field_spec("stored_grass")
    full[spec.address] = 250
    full[spec.address + 1] = 1
    assert econ.pasture_target_tiles(full) == 96
