"""Livestock economy (`harvest.planner.livestock_econ`): one price table, named gates, the ledger."""

import numpy as np

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
