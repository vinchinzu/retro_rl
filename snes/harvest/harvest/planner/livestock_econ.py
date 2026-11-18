"""Livestock economy: every gold, grass and hay number behind an animal.

One module owns the ranch economy. The day planner, the shop tasks, the
journal and the bench read it; nothing else defines a price or a gate.

1. **ROM facts.** Shelf and ship prices, the animal shop's gates, the milk
   tier. Prices are the shelf table at ``DATA8_81A1A5`` (displayed G = raw x 10)
   and the ship block at ``Items_Price_Table`` (``bank_81.asm:3060``).
2. **Year-1 policy.** The planner's choices: reserves, the cow target, the
   pasture target, the hay floor. Change a knob here, then run the Year-1
   bench (``docs/Y1_BENCH.md``).
3. **Gates.** Pure functions of RAM. :func:`livestock_block` names the first
   reason an animal is not bought today; :func:`ranch_gates` puts every gate in
   the day journal so a bench can say why the cow waited.
4. **Herd rules.** Cow happiness, growth and care from the F3 research. The
   barn task does not call them; they are the 999 controller's arithmetic.
5. **Rancher ledger.** A separate 300-morning paper plan for the 999 score
   (``docs/perfect_rancher/research/F5_economy.md``). It does not drive play.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from harvest.core.animal_status import (
    COW_STATUS_ADULT_FLAG,
    COW_STATUS_YOUNG_FLAG,
    count_chicken_slots,
    existing_cow_slots,
    read_cow_status,
    read_cow_timer,
)
from harvest.core.ram_catalog import read_ram_u16, read_ram_value
from harvest.core.tile_catalog import MATURE_GRASS_TILES, Tool
from harvest.planner import run_strategy


# ── 1. ROM facts ─────────────────────────────────────────────────────

# Shelf index is shelf_id - $44 (CODE_8189BB). Tool ids below $44 alias other rows.
SHOP_PRICE_G = {
    "grass_seed": 500,       # $44
    "corn_seed": 300,        # $45
    "tomato_seed": 300,      # $46
    "potato_seed": 200,      # $47
    "turnip_seed": 200,      # $48
    "brush": 800,            # $4C, shed row 2 bit $40
    "milker": 1800,          # $4B, shed row 2 bit $20
    "sprinkler": 2000,       # $51, shed row 3 bit $10
    "medicine": 1000,        # $52, shed row 2 bit $01
    "miracle_potion": 4000,  # $53, shed row 2 bit $02. Not the tool-id $0A alias.
    "chicken_feed": 300,     # $54, STA feed_chicks_N = 12
    "fodder": 600,           # $55, STA feed_cow_N = 12
}
# Egg 50 and milk S/M/L follow corn/tomato/potato/turnip in the same block
# (index inference off a matched run, not a decoded item-id table).
SHIP_PRICE_G = {
    "turnip": 60, "potato": 80, "tomato": 100, "corn": 120,
    "egg": 50, "milk_s": 150, "milk_m": 250, "milk_l": 350,
}
COW_PRICE = 5_000                  # AddMoney $FE0C/$FF on the cow buy
CHICKEN_PRICE = 1_000              # AddMoney $FF9C
# A sale pays a flat 500G, adults only (status bit $08): $7F1F15 = $0032 in
# Event_UNKChickenRelated ($849857, bank_84.asm ~3672). Eggs and chicks never
# eat; only adults take the night's feed (bank_83.asm $83C084).
CHICKEN_SALE_PRICE = 500
ANIMAL_PRICE = {"cow": COW_PRICE, "chicken": CHICKEN_PRICE}
HOUSE_UPGRADE_G = (5_000, 10_000)  # AddMoney $FFFE0C / $FFFC18 at $81F14C
TOOL_PRICE = {int(Tool.SPRINKLER): SHOP_PRICE_G["sprinkler"], 9: SHOP_PRICE_G["medicine"], int(Tool.BRUSH): SHOP_PRICE_G["brush"], int(Tool.MILKER): SHOP_PRICE_G["milker"]}

SHED_ROW_2_ADDR = 0x11F01
SHED_ROW_2_MILKER_BIT = 0x20
SHED_ROW_2_BRUSH_BIT = 0x40

# Animal shop (bank_81 script). One animal a day, 12 of each, and the shop
# compares planted grass with 16 tiles per animal after the buy. It refuses
# delivery in rain or snow. Grass planted in Fall never raises the counter.
ANIMAL_CAP = 12
GRASS_PER_ANIMAL = 16
WEATHER_BLOCK_MASK = 0x000A        # $0196
BOUGHT_TODAY_MASK = 0x0020         # SRAM 0x11F5A
_COUNT_KEY = {"cow": "num_cows", "chicken": "num_chickens"}


def milk_ship_g(happiness: int) -> int:
    """Item chosen at ``$849E17``. Happiness 0 still ships small milk."""
    if happiness < 0x60:
        return SHIP_PRICE_G["milk_s"]
    if happiness < 0xC0:
        return SHIP_PRICE_G["milk_m"]
    return SHIP_PRICE_G["milk_l"]


# ── 2. Year-1 policy ─────────────────────────────────────────────────

WALLET_RESERVE_G = 300             # never spend below this, except the Fall tool trip
COW_TARGET = 2                     # cows bought in Year 1
# A Fall cow is adult 21 nights later and milks the last Fall days and all of
# Winter (~35 days), more than its price comes back; the hay gate decides.
COW_TARGET_FALL = 3
FALL_CUT_YIELD = 0.8               # share of a tile's regrowth that gets cut: cuts are lazy
# Year 2+: a cow kept happy is 28 ending points (3 + min(25, happiness >> 3))
# and ships 350G of milk a day, and barn chores stop the clock. The 2-cow cap
# left 20-30k idle in Year 3 of y23_final. Spring opens too: the pasture is
# sown from D1 once there is a herd, and the shop's grass gate still applies.
COW_TARGET_LATER = 4
COW_PASTURE_TILES = 32             # planted tiles before the first cow trip
ANIMAL_BUY_SEASONS = (1, 2)        # Summer and Fall. Spring and Winter stay closed.
ANIMAL_BUY_SEASONS_LATER = (0, 1, 2)
# A planted tile gives one hay per nine days and each animal eats one a day, so
# nine tiles feed one animal. The shop's 16 per animal gates buying; this one
# gates hatching, which the shop never checks.
HAY_TILES_PER_ANIMAL = 9
HAY_DAYS_WANTED = 4                # hay on hand per animal before a buy or a skipped cut
COW_HAY_RESERVE_DAYS = 5           # hay held back for the barn adults before a surplus hen eats
PASTURE_MAX_TILES = 180
GRASS_BAG_TILES = 8                # a bag seeds 8 to 9 tiles once debris is lifted
# Corn ripens 12 nights after planting (measured on the ROM) and then regrows
# every 3, so a bag planted after Summer D18 cannot pay before Fall.
SUMMER_LAST_CORN_BAG_DAY = 18
# A ring planted by D3 still gets six harvests, one planted on D4 only five. The
# grass errand waits until then so the rings' bag is the day's only shop trip:
# a second town trip fails (exit_walk timeout) or lands past the 13:00 window.
RING_FIRST_PLANT_DAYS = 3
PASTURE_LAST_BUY_DAY = 29          # Summer D29 grass still matures before Fall
# Year 2+: Winter D1 wipes the pasture and the night check decrements
# ``planted_grass`` with it, so Spring starts bare. A bag counts in Spring and
# Summer only (ROM: season < 2). Spring sows only with a herd to feed, which
# Year 1 never has, and the Fall cut also stocks the Spring days before the new
# grass matures (sown D1-D3, nine nights, then a cut).
PASTURE_SOW_SEASONS = (0, 1)
SPRING_HAY_DAYS = 14
CUT_MIN_MATURE = 8                 # a cut costs a sickle fetch: wait for 8 mature tiles
HAY_CUT_FLOOR = 48                 # a stocked barn below this still cuts
# Keep two laying hens, counting chicks and the incubating egg toward the
# cap. Surplus breeding adds daily care and town sale trips; the finishing
# route no longer expands the flock for the ending score.
HENS_KEPT = 2
FLOCK_CAP = HENS_KEPT
CHICKEN_CAP = 12  # ROM capacity, independent of the route's workload limit


# ── 3. Gates ─────────────────────────────────────────────────────────

def _value(ram: np.ndarray, key: str) -> int:
    return int(read_ram_value(ram, key))


def _check_species(species: str) -> None:
    if species not in ANIMAL_PRICE:
        raise ValueError(f"unknown animal species {species!r}")


def _animals(ram: np.ndarray) -> int:
    return _value(ram, "num_cows") + _value(ram, "num_chickens")


def weather_blocks_animal_delivery(ram: np.ndarray) -> bool:
    """Shopkeeper refuses delivery when $0196 has rain or snow."""
    return (read_ram_u16(ram, 0x0196) & WEATHER_BLOCK_MASK) != 0


def animal_bought_today(ram: np.ndarray) -> bool:
    """SRAM 0x11F5A bit 0x20 is set after either animal is bought that day."""
    return (read_ram_u16(ram, 0x11F5A) & BOUGHT_TODAY_MASK) != 0


def purchase_grass_required(cows: int, chickens: int, species: str) -> int:
    """Planted grass tiles the shop script compares before AddMoney."""
    _check_species(species)
    return (cows + chickens + 1) * GRASS_PER_ANIMAL


def purchase_block_reason(ram: np.ndarray, species: str) -> Optional[str]:
    """The shop's own refusal for ``species``, or None when it would sell.

    Weather, one animal a day, the cap, the wallet, then grass.
    """
    _check_species(species)
    if weather_blocks_animal_delivery(ram):
        return "weather"
    if animal_bought_today(ram):
        return "bought_today"
    if _value(ram, _COUNT_KEY[species]) >= ANIMAL_CAP:
        return "full"
    if _value(ram, "money") < ANIMAL_PRICE[species]:
        return "money"
    required = purchase_grass_required(_value(ram, "num_cows"), _value(ram, "num_chickens"), species)
    if _value(ram, "planted_grass") < required:
        return "grass"
    return None


def pasture_target_tiles(ram: np.ndarray) -> int:
    """Planted tiles wanted: the shop's 16 per animal for the herd and two more.

    That also yields about 1.8 hay a day per animal, which leaves a surplus to
    cut into the Winter stock (the pasture is wiped on Winter D1).

    A strategy that grows the herd sizes the pasture for the animals it plans to
    own, from Year 2 on: the shop wants 16 planted tiles per animal after the buy, and
    the cash-optimal herd table (``block_calendar.grass_needed``) puts 8 cows and 2 hens
    at 234 tiles, past the 180-tile default cap (``run_strategy.pasture_max_tiles``).
    """
    animals_now = _animals(ram)
    if run_strategy.grows_herd() and _value(ram, "year") != 0:
        animals_now = max(animals_now, cow_target(ram) + kept_adults(ram))
    target = max(COW_PASTURE_TILES, GRASS_PER_ANIMAL * (animals_now + 2))
    if _value(ram, "season") == 1 and _value(ram, "day") >= 19 and _value(ram, "num_cows"):
        # Last profitable corn bags are funded. Sow Winter feed before Fall,
        # when the vendor counter stops growing. New grass first cuts after
        # nine nights; allow only whole cuts at the observed 80% yield.
        import math
        cows = (run_strategy.cow_target_fall(COW_TARGET_FALL) if _value(ram, "year") == 0
                else run_strategy.cow_target_later(COW_TARGET_LATER))
        animals = max(_animals(ram), cows + _value(ram, "num_chickens"))
        growing_days = 60 - _value(ram, "day")
        cuts = max(1, growing_days // HAY_TILES_PER_ANIMAL)
        demand = animals * (growing_days + WINTER_DAYS + 1 + SPRING_HAY_DAYS)
        needed = max(0, demand - _value(ram, "stored_grass"))
        target = max(target, GRASS_PER_ANIMAL * animals, math.ceil(needed / (cuts * FALL_CUT_YIELD)))
    return min(run_strategy.pasture_max_tiles(PASTURE_MAX_TILES), target)


def pasture_feeds_more(ram: np.ndarray, *, reserve_first_cow: bool = True) -> bool:
    """Allow a hatch only when feed and the first cow's shop gate stay covered."""
    animals = _animals(ram)
    required = HAY_TILES_PER_ANIMAL * (animals + 1)
    if reserve_first_cow and _value(ram, "num_cows") == 0:
        required = max(required, GRASS_PER_ANIMAL * (animals + 2))
    return _value(ram, "planted_grass") >= required


def flock_birds(ram: np.ndarray) -> int:
    """Adults, chicks and incubating eggs. ``num_chickens`` misses the egg."""
    return max(sum(count_chicken_slots(ram)), _value(ram, "num_chickens"))


def kept_adults(ram: np.ndarray) -> int:
    """Keep the small income flock, expanding for score after the cow-care window."""
    strategy = run_strategy.active()
    index = _value(ram, "year") * 120 + _value(ram, "season") * 30 + _value(ram, "day") - 1
    # ``HARVEST_FINAL_FLOCK_START=<day index>`` moves the date for an A/B (240 = Y3 Spring D1).
    start = int(os.environ.get("HARVEST_FINAL_FLOCK_START") or strategy.final_flock_start_index)
    if strategy.final_flock_target is not None and index >= start:
        return int(strategy.final_flock_target)
    return run_strategy.hens_kept(HENS_KEPT)


def hatch_allowed(ram: np.ndarray, *, reserve_first_cow: bool = True) -> bool:
    """Replace a missing hen only when pasture supports her and the first cow."""
    return flock_birds(ram) < kept_adults(ram) and pasture_feeds_more(
        ram, reserve_first_cow=reserve_first_cow,
    )


def hay_eaters(ram: np.ndarray) -> int:
    """Animals the loft has to carry. A strategy that grows the herd counts only what the ROM feeds.

    A bought cow is a child for 21 nights and is never tested for a fed bit, and an unfed hen only stops laying,
    so children stay out until a week before they grow up and hens past the kept pair do not count.
    """
    if not run_strategy.grows_herd():
        return _animals(ram)
    cows = 0
    for slot in existing_cow_slots(ram):
        status = read_cow_status(ram, slot)
        grown = status & COW_STATUS_ADULT_FLAG or (status & COW_STATUS_YOUNG_FLAG and read_cow_timer(ram, slot) >= CHILD_NIGHTS_TO_ADULT - 7)
        cows += bool(grown)
    return cows + min(_value(ram, "num_chickens"), HENS_KEPT)


def hens_to_feed(ram: np.ndarray) -> int:
    """Hens that get hay today: the kept pair, then whatever the loft holds past the cows' reserve.

    An unfed hen stops laying for three nights and scores the same (bank_83 ``$83C0CF``); an unfed cow rolls
    sickness. So a short loft goes to the cows.
    """
    cows = sum(bool(read_cow_status(ram, slot) & COW_STATUS_ADULT_FLAG) for slot in existing_cow_slots(ram))
    spare = _value(ram, "stored_grass") - COW_HAY_RESERVE_DAYS * cows
    return max(HENS_KEPT, spare)


def hay_wanted(ram: np.ndarray) -> int:
    """Hay on hand that covers a few days for the herd and the next animal."""
    return HAY_DAYS_WANTED * (hay_eaters(ram) + 1)


def hay_cut_target(ram: np.ndarray) -> int:
    """Hay a batch cut should reach: Fall stocks the rest of Fall, Winter, and Spring's regrowth."""
    if _value(ram, "season") == 2:
        left = 30 - _value(ram, "day")
        return max(hay_wanted(ram), _animals(ram) * (left + 31 + SPRING_HAY_DAYS))
    return hay_wanted(ram)


def pasture_season(ram: np.ndarray) -> bool:
    """Grass work happens: Summer and Fall, and Spring once there is a herd."""
    season = _value(ram, "season")
    if season == 0:
        return _animals(ram) > 0
    return season in ANIMAL_BUY_SEASONS


def ring_bags_wanted(ram: np.ndarray) -> int:
    """Corn bags the pocket rings still lack: empty or about-to-be-empty rings less stock.

    Reads the loaded map (ring tiles), so call it on the farm.
    """
    from harvest.maps.farm_pond import pocket_plant_targets

    return max(0, len(pocket_plant_targets(ram, replant_ripe=True)) - _value(ram, "corn_seeds"))


def pasture_reserve_g(ram: np.ndarray) -> int:
    """Gold the grass buyer leaves alone: Summer seed bags, then the next animal.

    A corn bag returns far more than a pasture tile, so the rings' seed bags
    come first, until ``SUMMER_LAST_CORN_BAG_DAY``. After that the late Summer
    gold is the only gold that can still open the cow gate. The hen fund is
    held until she is bought, and a wallet that covers a cow keeps it covered.
    Reads the loaded map (ring tiles), so call it on the farm.
    """
    reserve = WALLET_RESERVE_G
    if _value(ram, "season") == 1 and _value(ram, "day") <= SUMMER_LAST_CORN_BAG_DAY:
        reserve += ring_bags_wanted(ram) * SHOP_PRICE_G["corn_seed"]
    if _value(ram, "num_chickens") == 0:
        reserve += CHICKEN_PRICE
    if _value(ram, "num_cows") < cow_target(ram) and _value(ram, "money") >= COW_PRICE + reserve:
        reserve += COW_PRICE
    return reserve


def grass_bags_to_buy(
    money: int, stock: int, planted: int, *, reserve: int = WALLET_RESERVE_G,
    target: int = COW_PASTURE_TILES,
) -> int:
    """Bags still needed for ``target`` pasture tiles, without spending below ``reserve``."""
    bags = (target - planted + (GRASS_BAG_TILES - 1)) // GRASS_BAG_TILES
    need = max(0, bags - stock)
    return min(need, max(0, (money - reserve) // SHOP_PRICE_G["grass_seed"]))


def fall_cow_hay_ok(ram: np.ndarray) -> bool:
    """True when one more animal still gets hay to Spring's new grass.

    Winter D1 wipes the pasture, so Fall's cuts are the whole supply. Stock plus
    ``FALL_CUT_YIELD`` of what the planted tiles regrow in the Fall days left
    must cover every animal (the new cow included) through Fall, Winter and
    ``SPRING_HAY_DAYS``. A cow that outruns the hay is unfed, sick and dead
    seven nights later (``pasture_ctl`` vs ``pasture_fix`` in Y1_BENCH).
    """
    left = max(0, 30 - _value(ram, "day"))
    animals = _animals(ram) + 1
    supply = _value(ram, "stored_grass") + int(
        FALL_CUT_YIELD * (_value(ram, "planted_grass") / HAY_TILES_PER_ANIMAL) * left
    )
    return supply >= animals * (left + WINTER_DAYS + 1 + SPRING_HAY_DAYS)


def cow_target(ram: np.ndarray) -> int:
    """Cows to own: two in Year 1, a third in Fall when the hay carries it."""
    if _value(ram, "year") != 0:
        target = run_strategy.cow_target_later(COW_TARGET_LATER)
        if (
            run_strategy.grows_herd()
            and _value(ram, "season") == 2
            and _value(ram, "num_cows") >= COW_TARGET_LATER
            and not fall_cow_hay_ok(ram)
        ):
            # Year 1's Fall hay gate, for the bigger herd: Winter D1 wipes the pasture, so a Fall cow the
            # stock cannot carry to Spring is unfed, sick and dead (e2e h5: slots 0-2 fell 255 -> 8).
            return min(target, _value(ram, "num_cows"))
        return target
    if (
        _value(ram, "season") == 2
        and _value(ram, "num_cows") >= COW_TARGET
        and fall_cow_hay_ok(ram)
    ):
        return run_strategy.cow_target_fall(COW_TARGET_FALL)
    return COW_TARGET


def animal_buy_season(ram: np.ndarray) -> bool:
    seasons = ANIMAL_BUY_SEASONS if _value(ram, "year") == 0 else ANIMAL_BUY_SEASONS_LATER
    return _value(ram, "season") in seasons


def livestock_block(ram: np.ndarray, species: str) -> Optional[str]:
    """First reason the planner will not buy ``species`` today, or None.

    The hen comes first: 1,000G that pays back in about three weeks, and each
    egg can hatch a free hen. A 5,000G cow loses money until the brush and
    milker are owned and its 21 child nights have passed. The shop's grass
    gate counts every hatched chick, so the cow also waits for
    ``COW_PASTURE_TILES``. A few days of hay is a gate: an unfed hen lays
    nothing for three nights.
    """
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS

    _check_species(species)
    if not animal_buy_season(ram):
        return "season"
    if _value(ram, "weekday") in SHOP_CLOSED_WEEKDAYS:
        return "shop_closed"
    if species == "chicken" and _value(ram, "num_chickens") > 0:
        return "have_hen"
    if species == "cow":
        if _value(ram, "num_cows") >= cow_target(ram):
            return "cow_target"
        if _value(ram, "planted_grass") < COW_PASTURE_TILES:
            return "pasture"
    shop = purchase_block_reason(ram, species)
    if shop is not None:
        return shop
    if _value(ram, "stored_grass") < hay_wanted(ram):
        return "hay"
    if _value(ram, "money") < ANIMAL_PRICE[species] + WALLET_RESERVE_G:
        return "reserve"
    floor = run_strategy.late_cash_floor_g()
    if floor is not None and _value(ram, "year") >= 2 and _value(ram, "money") - ANIMAL_PRICE[species] < floor:
        return "cash_floor"
    return None


def next_livestock(ram: np.ndarray, *, cows: bool = True, chickens: bool = True) -> Optional[str]:
    """The hen first, then cows up to :func:`cow_target`. None when neither is bought today."""
    if chickens and livestock_block(ram, "chicken") is None:
        return "chicken"
    if cows and livestock_block(ram, "cow") is None:
        return "cow"
    return None


def animal_shop_errand(ram: np.ndarray, *, cows: bool = True, chickens: bool = True) -> Optional[str]:
    """``"hen_pickup"`` for a bought hen still outside the coop, else :func:`next_livestock`."""
    from harvest.core.animal_probe import stray_yard_hen_px

    if chickens and stray_yard_hen_px(ram) is not None:
        return "hen_pickup"
    return next_livestock(ram, cows=cows, chickens=chickens)


def house_payment_allowed(ram: np.ndarray) -> bool:
    """Carpenter payment waits for the first cow. Wood gathering does not."""
    return _value(ram, "num_cows") >= 1


def missing_cow_tools(ram: np.ndarray) -> tuple[int, ...]:
    """Brush then milker (Tool ids) that are neither on shed row 2 nor carried."""
    from harvest.tasks.buy_seeds import TOOL_SHELVES, tool_owned

    return tuple(t for t in (int(Tool.BRUSH), int(Tool.MILKER))
                 if t in TOOL_SHELVES and not tool_owned(ram, t))


def cow_tool_trip_affordable(ram: np.ndarray) -> bool:
    """The wallet covers the next missing tool (brush first) plus the reserve.

    A milker-only trip must not leave on the brush's threshold. Fall keeps no
    reserve so the milker lands before Winter's snow days.
    """
    missing = missing_cow_tools(ram)
    if not missing:
        return False
    reserve = 0 if _value(ram, "season") == 2 else WALLET_RESERVE_G
    return _value(ram, "money") >= TOOL_PRICE[missing[0]] + reserve


def tool_shop_open(ram: np.ndarray) -> bool:
    """General-store door rule (``Unk_Table13`` flags 0x32): shut Sat, Sun, from 17:00."""
    from harvest.calendar_schedule import shop_open

    return shop_open(
        "tool",
        weekday=_value(ram, "weekday"),
        hour=_value(ram, "hour"),
        season=_value(ram, "season"),
        day=_value(ram, "day"),
        weather_flags=_value(ram, "weather"),
    )


@dataclass(frozen=True)
class PastureErrand:
    """This morning's grass work. A cut replaces buying and planting."""

    buy_to_stock: int = 0          # grass bags to hold after the shop; 0 = no buy
    plant: bool = False
    cut: bool = False


def pasture_errand(ram: np.ndarray) -> PastureErrand:
    """Plant while Spring or Summer can still count it, else cut hay. One errand a morning.

    A Fall bag never raises ``planted_grass``, so a cut waits behind planting
    unless the herd is short of hay. Reads the pasture tiles: call it on the farm.
    """
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS
    from harvest.tasks.grass_planter import PASTURE_BOUNDS
    from harvest.tasks.nav import get_tile_at

    season = _value(ram, "season")
    day = _value(ram, "day")
    if not pasture_season(ram):
        return PastureErrand()
    if season == 1 and day <= RING_FIRST_PLANT_DAYS and ring_bags_wanted(ram) > 0:
        return PastureErrand()
    x0, y0, x1, y1 = (2, 3, 62, 60) if run_strategy.field_cover() else PASTURE_BOUNDS
    mature = sum(
        get_tile_at(ram, x, y) in MATURE_GRASS_TILES
        for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
    )
    hay = _value(ram, "stored_grass")
    planted = _value(ram, "planted_grass")
    stock = _value(ram, "grass_seeds")
    target = pasture_target_tiles(ram)
    buy = 0
    plant = False
    # Planted tiles mature nine nights later, so Summer D29 still feeds Fall.
    buy_open = season == 0 or day <= PASTURE_LAST_BUY_DAY
    if season in PASTURE_SOW_SEASONS and planted < target and (buy_open or stock > 0):
        closed = _value(ram, "weekday") in SHOP_CLOSED_WEEKDAYS
        if not closed and buy_open:
            money = _value(ram, "money")
            reserve = pasture_reserve_g(ram)
            buy = grass_bags_to_buy(money, stock, planted, reserve=reserve, target=target)
        plant = bool(stock or buy > 0)
    # A cut costs a sickle fetch (2.5-3.8k frames): short herds cut at once,
    # a stocked barn waits for a worthwhile harvest, and for a planting errand.
    if mature and (
        hay < hay_wanted(ram)
        or (not plant and mature >= CUT_MIN_MATURE
            and (hay < HAY_CUT_FLOOR or hay < hay_cut_target(ram)))
    ):
        return PastureErrand(cut=True)
    return PastureErrand(buy_to_stock=stock + buy if buy > 0 else 0, plant=plant)


def ranch_gates(ram: np.ndarray) -> dict[str, Any]:
    """Every economy gate on this RAM, for the day journal. Read-only.

    ``hen`` and ``cow`` are :func:`livestock_block` (None = bought today).
    ``tools`` lists the cow tools still missing; ``tool_trip`` is whether a cow
    is owned and the wallet and the store's hours allow the trip. Map-reading gates are
    left out: the morning read is taken indoors.
    """
    missing = missing_cow_tools(ram)
    return {
        "hen": livestock_block(ram, "chicken"),
        "cow": livestock_block(ram, "cow"),
        "pasture_target": pasture_target_tiles(ram),
        "hay_wanted": hay_wanted(ram),
        "hatch_ok": hatch_allowed(ram),
        "tools": [Tool(t).name.lower() for t in missing],
        "tool_trip": (bool(missing) and _value(ram, "num_cows") >= 1
                      and cow_tool_trip_affordable(ram) and tool_shop_open(ram)),
        "house_pay": house_payment_allowed(ram),
    }


# ── 4. Herd rules (F3; the 999 controller's arithmetic) ──────────────
# Every slot has its own trough bit, including slot 0's $0001 / tile $E2.
# A filled trough is tile $2F; clearing fed flags in a seed without
# regenerating the barn map leaves that stale tile. Native re-entry restores it.

CARE_DAY_DELTA = 4                 # talk +1, brush +3. Milk and the daytime feed add 0.
FED_ADULT_NIGHT_DELTA = 0
UNFED_ADULT_NIGHT_DELTA = -8
COW_SICK_HAPPINESS_PENALTY = -12   # bank_83.asm:6875 LDA #$FFF4
UNFED_ADULT_SICK_DELTA = -20       # -8 unfed + -12 sick on losing the 25% RNG roll (bank_83.asm)
SICK_ROLL_DENOMINATOR = 4          # RNGReturn0toA(4): 1 in 4 (25%) sickness chance when unfed
SICK_TIMER_START = 7               # Sickness countdown timer initialized to 7 nights
COW_MEDICINE_PRICE = 1000          # 1,000G at Animal Shop ($81A1A5 / item 0x09)
ITEM_COW_MEDICINE = 0x09
COW_CRANKY_HAPPINESS_PENALTY = -8  # bank_83.asm:6902 LDA #$FFF8
COW_CRANKY_ROLL_DENOMINATOR = 16   # RNGReturn0toA(16): 1 in 16 (6.25%) cranky chance outdoors when $7F1F5A:1000
COW_CRANKY_TIMER_START = 3         # bank_83.asm:6938 LDA #$03
WILD_DOG_BREACH_FLAG = 0x0400      # $0196 bit 0x0400: broken fence wild dog breach guarantees cranky
HAPPINESS_SCORE_BYTE = 200
HAPPINESS_BYTE_CAP = 255
# Long enough for a feedable adult to climb from 0, and for a birth-100
# calf to hit 200 inside the safe nights.
REACH_HORIZON_DAYS = 60
CHILD_NIGHTS_TO_ADULT = 21
BABY_NIGHTS_TO_CHILD = 14
BORN_SAFE_NIGHTS = BABY_NIGHTS_TO_CHILD + CHILD_NIGHTS_TO_ADULT
WINTER_DAYS = 30
COW_CRANKY_FLAG = 0x10
COW_SICK_FLAG = 0x20
COW_ADULT_FLAG = 0x08
COW_CHILD_FLAG = 0x04
COW_BABY_FLAG = 0x02
COW_EXISTS_FLAG = 0x01


def clamp_happiness(value: int) -> int:
    """AddCowHappiness: negative stores 0, a sum >= 256 stores 255."""
    if value < 0:
        return 0
    if value >= 256:
        return HAPPINESS_BYTE_CAP
    return int(value)


def slot_has_stall(slot: int) -> bool:
    """All twelve native cow slots have a matching trough."""
    return 0 <= int(slot) < ANIMAL_CAP


def _care_clock(start: int, safe_nights: int, care_day: int, *, feedable: bool, after_brush: bool) -> int:
    if care_day < 1:
        raise ValueError(f"care_day must be >= 1, got {care_day}")
    done = care_day - 1
    safe = min(done, safe_nights)
    adult = max(0, done - safe_nights)
    value = int(start) + safe * CARE_DAY_DELTA
    night = FED_ADULT_NIGHT_DELTA if feedable else UNFED_ADULT_NIGHT_DELTA
    value = clamp_happiness(value + adult * (CARE_DAY_DELTA + night))
    if after_brush:
        value = clamp_happiness(value + CARE_DAY_DELTA)
    return value


def bought_happiness(care_day: int, *, feedable: bool, after_brush: bool) -> int:
    """Happiness of a cow bought at 0 on care day ``care_day`` (1-based).

    ``after_brush`` is false for the 06:00 byte and true after that day's
    talk and brush, before the night pass. Child nights 1-21 skip the feed
    test. A fed adult night adds 0; an unfed night is -8.
    """
    return _care_clock(0, CHILD_NIGHTS_TO_ADULT, care_day, feedable=feedable, after_brush=after_brush)


def born_happiness(care_day: int, birth: int, *, feedable: bool, after_brush: bool) -> int:
    """Same clock for a named calf. Baby (14) plus child (21) nights skip the feed test."""
    return _care_clock(birth, BORN_SAFE_NIGHTS, care_day, feedable=feedable, after_brush=after_brush)


@dataclass(frozen=True)
class HerdCow:
    slot: int
    happiness: int
    origin: str = "bought"
    status: int = COW_EXISTS_FLAG | COW_ADULT_FLAG


def _adult(status: int) -> bool:
    return bool(status & COW_ADULT_FLAG)


def _skips_feed_penalty(status: int) -> bool:
    if status & COW_BABY_FLAG:
        return True
    return bool(status & COW_CHILD_FLAG) and not _adult(status)


def adult_night_delta(cow: HerdCow, *, fed: bool) -> int:
    if _skips_feed_penalty(cow.status) or not _adult(cow.status):
        return 0
    if slot_has_stall(cow.slot) and fed:
        return FED_ADULT_NIGHT_DELTA
    return UNFED_ADULT_NIGHT_DELTA


def _stage_nights_left(status: int) -> tuple[int, int]:
    """Baby nights still owed, then child nights, before the feed test."""
    if status & COW_BABY_FLAG:
        return BABY_NIGHTS_TO_CHILD, CHILD_NIGHTS_TO_ADULT
    if (status & COW_CHILD_FLAG) and not (status & COW_ADULT_FLAG):
        return 0, CHILD_NIGHTS_TO_ADULT
    return 0, 0


def cannot_reach_200(cow: HerdCow, days_left: int) -> bool:
    """True when max care cannot put the byte at >= 200 inside ``days_left``.

    Invalid slots cannot use the barn feed projection; baby and child stages
    still age inside the projection.
    """
    if cow.origin == "bought" and not slot_has_stall(cow.slot):
        return True
    happy = int(cow.happiness)
    if happy >= HAPPINESS_SCORE_BYTE and (slot_has_stall(cow.slot) or cow.origin == "born"):
        return False
    baby_left, child_left = _stage_nights_left(cow.status)
    for _ in range(max(0, int(days_left))):
        happy = clamp_happiness(happy + CARE_DAY_DELTA)
        skipping = baby_left > 0 or child_left > 0
        if happy >= HAPPINESS_SCORE_BYTE and (slot_has_stall(cow.slot) or skipping):
            return False
        if baby_left > 0:
            baby_left -= 1
        elif child_left > 0:
            child_left -= 1
        else:
            night = FED_ADULT_NIGHT_DELTA if slot_has_stall(cow.slot) else UNFED_ADULT_NIGHT_DELTA
            happy = clamp_happiness(happy + night)
        if happy >= HAPPINESS_SCORE_BYTE:
            return False
    return happy < HAPPINESS_SCORE_BYTE


def controller_day(cows: Sequence[HerdCow]) -> tuple[list[HerdCow], list[str], list[dict[str, int]]]:
    """One care day. Cows already at 200 are not brushed, so a laggard gains.

    The night pass then applies the stall rule. Lowest happiness is cared first.
    """
    alarms: list[str] = []
    deltas: list[dict[str, int]] = []
    by_slot = {cow.slot: cow for cow in cows}
    for cow in sorted(cows, key=lambda row: (row.happiness, row.slot)):
        if cannot_reach_200(cow, days_left=REACH_HORIZON_DAYS):
            alarms.append(f"slot {cow.slot} cannot reach 200")
        hold = cow.happiness >= HAPPINESS_SCORE_BYTE and (
            slot_has_stall(cow.slot) or cow.origin == "born"
        )
        before = cow.happiness
        after_day = before if hold else clamp_happiness(before + CARE_DAY_DELTA)
        after = clamp_happiness(
            after_day + adult_night_delta(
                HerdCow(cow.slot, after_day, cow.origin, cow.status),
                fed=slot_has_stall(cow.slot),
            )
        )
        deltas.append({"slot": cow.slot, "before": before, "after": after, "delta": after - before})
        by_slot[cow.slot] = HerdCow(cow.slot, after, cow.origin, cow.status)
    return [by_slot[cow.slot] for cow in cows], alarms, deltas


def apply_cow_medicine(status: int, timer: int) -> tuple[int, int]:
    """Adult medicine clears the sick bit and byte +3. Happiness is not restored."""
    if not (status & COW_ADULT_FLAG):
        return int(status), int(timer)
    return int(status) & ~COW_SICK_FLAG, 0


def herd_count_after_night(
    *,
    cows: int,
    in_barn: bool,
    sick_timer: Optional[int],
    weather_flags: int,
    sick: bool = False,
) -> int:
    """Cows still present after one night.

    Death is the sick-adult arm: byte +3 hits 0 and ``$0196 & $001A`` is
    clear. Medicine clears the sick bit, so a cured timer of 0 is not a
    death. Hurricane bit ``$0010`` is inside that mask, so a storm night
    keeps the herd. ``in_barn`` is the stay-in record; outdoors happiness
    is a separate delta and does not delete a healthy cow.
    """
    del in_barn
    lost = 0
    if sick and sick_timer is not None and sick_timer - 1 <= 0 and (int(weather_flags) & 0x001A) == 0:
        lost += 1
    return max(0, int(cows) - lost)


def outside_cow_cranky_probability(*, wild_dog_breach: bool = False, flag_1f5a_1000: bool = True) -> float:
    """Probability that an adult cow left in the pasture overnight becomes cranky/mad.

    - 100% (1.0) if a wild dog breach occurred ($0196 & 0x0400 from broken fences).
    - 1/16 (0.0625) if outdoors and flag $7F1F5A:1000 is active.
    - 0% if safely housed in barn.
    """
    if wild_dog_breach:
        return 1.0
    if flag_1f5a_1000:
        return 1.0 / COW_CRANKY_ROLL_DENOMINATOR
    return 0.0


def winter_unfed_days(hay: int, cows: int = ANIMAL_CAP, days: int = WINTER_DAYS) -> int:
    """Days on which the morning stock cannot feed every cow. One hay each."""
    stock = int(hay)
    unfed = 0
    for _ in range(int(days)):
        if stock < cows:
            unfed += 1
        else:
            stock -= cows
    return unfed


def cow_tools_ready(shed_row_2: int, selected: int, backpack: int) -> bool:
    """Brush and milker, either still on shed row 2 or already carried."""
    carried = {int(selected), int(backpack)}
    brush = bool(shed_row_2 & SHED_ROW_2_BRUSH_BIT) or int(Tool.BRUSH) in carried
    milker = bool(shed_row_2 & SHED_ROW_2_MILKER_BIT) or int(Tool.MILKER) in carried
    return brush and milker


# ── 5. Rancher ledger (999 paper plan) ───────────────────────────────

VISITOR_DEBIT_G = 2_000            # raw units: the peddler asks 20,000G; the bot declines (scene.peddler_decline_action)
VISITOR_GATE_G = 30_000            # money_raw >= 3000 at $83D12E (ADC #$F448)
STARTING_G = 300                   # NewGameSetup LDA #30 into moneyL
WINDOW_LO_G = 99_840               # raw 9984 >> 7 == 78
WINDOW_HI_G = 655_350              # raw 65535, low 16 bits still full
HOLD_G = 620_000
WORK_DAYS = 299                    # mornings 0..298; index 299 is Y3 Summer D30
D30_INDEX = 299
PAID_GRASS_BAGS = 328              # one free bag is NewGameSetup seeds_grass_N = 1
BAG_TILES = 9
FEED_COVER = 12
VISITOR_INDEX = 124                # Y2 Spring D5
MARRIAGE_INDEX = 188               # Y2 Fall D9, F4 latest wife bit


def calendar_ymd(index: int) -> tuple[int, int, int]:
    """Year (1-based), season (0 spring .. 3 winter), day (1..30).

    ``Calendar.season_of`` stops at the first winter. This index runs all
    three years: Y1 Spring D1 is 0, Y3 Summer D30 is 299.
    """
    year = index // 120 + 1
    rem = index % 120
    return year, rem // 30, rem % 30 + 1


def shop_open_on(index: int) -> bool:
    """Sunday closure from :class:`spring_opt.Calendar`. Festivals are not in its blocked tuple."""
    from harvest.planner.spring_opt import Calendar

    return Calendar().shop_open(index + 1)


def cow_morning_happiness(care_day: int, *, feedable: bool) -> Optional[int]:
    """Morning byte before talk/brush. ``None`` until the first adult morning.

    Care day 1 is the buy day. Twenty-one child nights skip the feed test, so
    care day 22 is the first adult morning at 84. A feedable stall then gains
    4 a day and the brush on care day 50 stores 200. Slot 0's mask is ``$0001``
    and no stall feeds it, so after that 84 the morning falls by 4 and sticks at 0.
    """
    if care_day < CHILD_NIGHTS_TO_ADULT + 1:
        return None
    if feedable:
        return min(HAPPINESS_SCORE_BYTE, CARE_DAY_DELTA * (care_day - 1))
    return max(0, 84 - CARE_DAY_DELTA * (care_day - (CHILD_NIGHTS_TO_ADULT + 1)))


@dataclass(frozen=True)
class LedgerMorning:
    index: int
    year: int
    season: int
    day: int
    morning_g: int
    spend_g: int
    ship_g: int
    balance_g: int
    note: str = ""


@dataclass(frozen=True)
class RanchLedger:
    """Morning-of-D30 cash under the F2 cover, F3 herd, and F4 house dates."""

    hold_g: int
    end_g: int
    min_g: int
    min_index: int
    stop_index: Optional[int]
    visited: bool
    house1_index: int
    house2_index: int
    cow_buys: Tuple[int, ...]
    hen_index: int
    adult_count: int
    grass_bags_paid: int
    planted_grass: int
    bags: Tuple[Tuple[str, int], ...]
    shipped: Tuple[Tuple[str, int], ...]
    totals: Tuple[Tuple[str, int], ...]
    withheld_g: int
    mornings: Tuple[LedgerMorning, ...]

    @property
    def in_window(self) -> bool:
        return WINDOW_LO_G <= self.end_g <= WINDOW_HI_G


def _crop_prices_match_rom(crops) -> None:
    for name in ("turnip", "potato", "tomato", "corn"):
        spec = crops[name]
        if (spec.seed_cost_g, spec.sell_price_g) != (SHOP_PRICE_G[f"{name}_seed"], SHIP_PRICE_G[name]):
            raise ValueError(f"{name} price drifted from the shop/ship tables")


def build_rancher_ledger(hold_g: int = HOLD_G) -> RanchLedger:
    """299 work days plus the Y3 Summer D30 morning.

    Crop seed and ship prices are ``spring_opt.CROPS`` after a check that
    they still match the ROM tables. Shop days are ``Calendar.shop_open``.
    Animal shipments stop being fully banked once the next morning would pass
    ``hold_g``; later feed buys are filled back up to that hold. House,
    grass, cows, and the summer seed pair are bought before that hold binds.
    """
    from harvest.planner.spring_opt import CROPS

    _crop_prices_match_rom(CROPS)
    bal = STARTING_G
    pending = 0
    planted = BAG_TILES
    paid = 0
    cows: list[int] = []
    hen: Optional[int] = None
    lays: list[int] = []
    chicks = feed = block = eggs_ready = 0
    incub: Optional[int] = None
    brush = milker = sprinkler = False
    h1: Optional[int] = None
    h2: Optional[int] = None
    visited = False
    bags = {name: 0 for name in ("turnip", "potato", "tomato", "corn")}
    plants: list[tuple] = []
    shipped = {name: 0 for name in bags}
    spend_keys = ("seeds", "grass", "cows", "chicken", "tools", "house", "feed", "visitor")
    totals = {key: 0 for key in (*spend_keys, "milk", "eggs", "crops")}
    withheld = 0
    min_g, min_index, stop = bal, 0, None
    mornings: list[LedgerMorning] = []

    def spend(amount: int, key: str) -> bool:
        nonlocal bal
        if bal < amount:
            return False
        bal -= amount
        totals[key] += amount
        return True

    def keeps(amount: int, floor: int) -> bool:
        return bal - amount >= floor

    def harvest_gold(index: int) -> tuple[int, dict[str, int]]:
        year, season, day = calendar_ymd(index)
        gold = 0
        counts = {name: 0 for name in bags}
        for crop, py, ps, pd, tiles in plants:
            spec = CROPS[crop]
            sell = spec.sell_price_g
            if spec.regrow_days is None:
                harvest_day = pd + spec.days_to_first_harvest
                if harvest_day <= 30:
                    hit = (year, season, day) == (py, ps, harvest_day)
                else:
                    hit = (year, season, day) == (py, ps + 1, harvest_day - 30)
            else:
                hit = False
                if (py, ps) == (year, season):
                    harvest_day = pd + spec.days_to_first_harvest
                    while harvest_day <= 30:
                        if day == harvest_day:
                            hit = True
                            break
                        harvest_day += spec.regrow_days
            if hit:
                gold += tiles * sell
                counts[crop] += tiles
        return gold, counts

    for index in range(WORK_DAYS):
        year, season, day = calendar_ymd(index)
        spent_at = sum(totals[key] for key in spend_keys)
        bal += pending
        pending = 0
        morning = bal
        notes: list[str] = []
        open_shop = shop_open_on(index)
        visitor_floor = VISITOR_GATE_G if (not visited and year >= 2 and index <= VISITOR_INDEX and bal >= VISITOR_GATE_G) else 0

        if index == VISITOR_INDEX and not visited and bal >= VISITOR_GATE_G:
            spend(VISITOR_DEBIT_G, "visitor")
            visited = True
            visitor_floor = 0
            notes.append("visitor")
        if h1 is None and year == 1 and season == 1 and bal >= HOUSE_UPGRADE_G[0]:
            if spend(HOUSE_UPGRADE_G[0], "house"):
                h1 = index
                notes.append("house1")
        if h1 is not None and h2 is None and index >= h1 + 4 and index <= 184 and bal >= HOUSE_UPGRADE_G[1]:
            seeds_in = bags["tomato"] == 8 and bags["corn"] == 8
            if (seeds_in or season >= 2 or year > 1) and keeps(HOUSE_UPGRADE_G[1], visitor_floor):
                if spend(HOUSE_UPGRADE_G[1], "house"):
                    h2 = index
                    notes.append("house2")

        if open_shop and season == 1 and day == 1:
            for crop in ("tomato", "corn"):
                price = CROPS[crop].seed_cost_g
                if bags[crop] == 0 and keeps(price * 8, visitor_floor):
                    for _ in range(8):
                        spend(price, "seeds")
                    bags[crop] = 8
                    plants.append((crop, year, season, 1, 8 * BAG_TILES))
                    notes.append(crop)

        if open_shop and season == 0:
            cap = 4 if year == 1 else 8
            bought = 0
            order = ("turnip", "potato") if year > 1 or bags["turnip"] >= 20 else ("turnip",)
            for crop in order:
                price = CROPS[crop].seed_cost_g
                need = 56
                while bags[crop] < need and bought < cap and keeps(price, visitor_floor) and spend(price, "seeds"):
                    bags[crop] += 1
                    bought += 1
                    plants.append((crop, year, season, day, BAG_TILES))
            if bought:
                notes.append(f"seed*{bought}")

        if open_shop and h1 is not None:
            if not brush and keeps(SHOP_PRICE_G["brush"], visitor_floor) and spend(SHOP_PRICE_G["brush"], "tools"):
                brush = True
                notes.append("brush")
            if not milker and keeps(SHOP_PRICE_G["milker"], visitor_floor) and spend(SHOP_PRICE_G["milker"], "tools"):
                milker = True
                notes.append("milker")
        sprinkler_on = year >= 2 or season >= 2 or (season == 1 and day >= 20)
        if (
            open_shop and sprinkler_on and h1 is not None and not sprinkler
            and (cows or bal >= 20_000)
            and keeps(SHOP_PRICE_G["sprinkler"], visitor_floor)
            and spend(SHOP_PRICE_G["sprinkler"], "tools")
        ):
            sprinkler = True
            notes.append("sprinkler")

        gate = GRASS_PER_ANIMAL * (chicks + len(cows) + 1)
        if open_shop and season < 2:
            binge = (
                h2 is not None and len(cows) == ANIMAL_CAP and bags["tomato"] == 8 and bags["corn"] == 8
                and bal >= 50_000 and (visited or index > VISITOR_INDEX)
            )
            need_paid = PAID_GRASS_BAGS if binge else max(0, (gate + BAG_TILES - 1) // BAG_TILES - 1)
            room = 8 if binge else 2
            bought_grass = 0
            while (
                paid < min(PAID_GRASS_BAGS, need_paid) and room
                and keeps(SHOP_PRICE_G["grass_seed"], visitor_floor)
                and spend(SHOP_PRICE_G["grass_seed"], "grass")
            ):
                paid += 1
                planted += BAG_TILES
                room -= 1
                bought_grass += 1
            if bought_grass:
                notes.append(f"grass*{bought_grass}")

        took_animal = False
        if open_shop and h1 is not None and hen is None and planted >= GRASS_PER_ANIMAL and keeps(CHICKEN_PRICE, visitor_floor):
            if spend(CHICKEN_PRICE, "chicken"):
                hen = index
                chicks = 1
                took_animal = True
                notes.append("chicken")
        gate = GRASS_PER_ANIMAL * (chicks + len(cows) + 1)
        if (
            open_shop and not took_animal and h1 is not None and brush and milker and len(cows) < ANIMAL_CAP
            and planted >= gate and keeps(COW_PRICE, visitor_floor) and spend(COW_PRICE, "cows")
        ):
            cows.append(index)
            notes.append(f"cow{len(cows)}")

        eating = (1 if hen is not None and index >= hen else 0) + sum(1 for lay in lays if index > lay)
        if open_shop and eating > feed and keeps(SHOP_PRICE_G["chicken_feed"], visitor_floor):
            if spend(SHOP_PRICE_G["chicken_feed"], "feed"):
                feed = FEED_COVER
                notes.append("feed")

        crop_g, counts = harvest_gold(index)
        for name, count in counts.items():
            shipped[name] += count
        milk = 0
        if brush and milker:
            for slot, bought in enumerate(cows):
                happiness = cow_morning_happiness(index - bought + 1, feedable=slot != 0)
                if happiness is not None:
                    milk += milk_ship_g(happiness)
        animal_g = milk + eggs_ready * SHIP_PRICE_G["egg"]
        room = hold_g - bal - crop_g
        animal_ship = min(animal_g, max(0, room))
        if animal_g > animal_ship and stop is None:
            stop = index
            notes.append("hold")
        withheld += animal_g - animal_ship
        pending = crop_g + animal_ship
        totals["crops"] += crop_g
        if animal_g:
            milk_part = int(animal_ship * milk / animal_g)
            totals["milk"] += milk_part
            totals["eggs"] += animal_ship - milk_part

        layers = (1 if hen is not None and index >= hen else 0) + sum(1 for lay in lays if index >= lay)
        new_adult = incub is not None and index == incub
        if new_adult:
            layers += 1
        if eating and feed < eating:
            block = 3
        use = min(feed, eating)
        feed -= use
        if block:
            laid = 0
            block -= 1
        else:
            laid = min(layers - (1 if new_adult else 0), use) + (1 if new_adult else 0)
        eggs_ready = laid
        if hen is not None and chicks < ANIMAL_CAP and incub is None and laid > 0:
            incub = index + 10
            eggs_ready -= 1
        if incub is not None and index == incub - 7:
            chicks += 1
        if incub is not None and index == incub:
            lays.append(index)
            incub = None

        if bal < min_g:
            min_g, min_index = bal, index
        mornings.append(LedgerMorning(
            index, year, season, day, morning, sum(totals[key] for key in spend_keys) - spent_at,
            pending, bal, ",".join(notes),
        ))

    bal += pending
    year, season, day = calendar_ymd(D30_INDEX)
    mornings.append(LedgerMorning(D30_INDEX, year, season, day, bal, 0, 0, bal, "d30"))
    if h1 is None or h2 is None or hen is None:
        raise RuntimeError("rancher ledger missed a house payment or the hen")
    ledger = RanchLedger(
        hold_g=hold_g,
        end_g=bal,
        min_g=min_g,
        min_index=min_index,
        stop_index=stop,
        visited=visited,
        house1_index=h1,
        house2_index=h2,
        cow_buys=tuple(cows),
        hen_index=hen,
        adult_count=(1 if hen is not None else 0) + len(lays),
        grass_bags_paid=paid,
        planted_grass=planted,
        bags=tuple(bags.items()),
        shipped=tuple(shipped.items()),
        totals=tuple(totals.items()),
        withheld_g=withheld,
        mornings=tuple(mornings),
    )
    _require_plan(ledger, chicks)
    return ledger


def _require_plan(ledger: RanchLedger, chicks: int) -> None:
    if not ledger.in_window:
        raise RuntimeError(f"D30 balance {ledger.end_g} is outside {WINDOW_LO_G}..{WINDOW_HI_G}")
    if ledger.min_g < 0:
        raise RuntimeError(f"wallet went negative on index {ledger.min_index}")
    if len(ledger.cow_buys) != ANIMAL_CAP or ledger.adult_count != ANIMAL_CAP or chicks != ANIMAL_CAP:
        raise RuntimeError("herd or flock did not reach 12")
    if ledger.grass_bags_paid != PAID_GRASS_BAGS:
        raise RuntimeError(f"paid grass bags {ledger.grass_bags_paid}")
    if ledger.house2_index + 4 > MARRIAGE_INDEX:
        raise RuntimeError("house level 2 bit lands after Y2 Fall D9")
    hy, hs, _hd = calendar_ymd(ledger.house1_index)
    if (hy, hs) != (1, 1):
        raise RuntimeError("house upgrade 1 was not ordered in Y1 summer")
    if not ledger.visited:
        raise RuntimeError("Y2 visitor gate was not met")
    for name, count in ledger.shipped:
        if count != 504:
            raise RuntimeError(f"{name} shipped {count}, wanted 504")
    last = ledger.cow_buys[-1]
    if last + 50 > D30_INDEX:
        raise RuntimeError("last feedable cow is not at 200 on D30 morning")
